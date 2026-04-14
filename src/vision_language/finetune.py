import os
import torch
import sys
from copy import deepcopy
from torch.utils.data import DataLoader
from engine.config import parser
from engine.tools.utils import makedirs, set_random_seed, Tee
from engine.datasets.utils import TextTensorDataset
from engine.datasets.utils import get_few_shot_setup_name
from engine.models.head import UML, UMLClip
from engine.optimizer.default import HYPER_DICT
from engine.optimizer.optim import build_optimizer
from engine.optimizer.scheduler import build_lr_scheduler
from engine.datasets.utils import DatasetWrapper, get_few_shot_setup_name, get_few_shot_benchmark
from features import text_outdir
from engine.transforms.default import build_transform
from engine.clip import clip
from timm.models import create_model
from transformers import AutoModel
from itertools import product
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")


EVAL_FREQ = 100 # Evaluate on val set per 100 iterations (for early stopping)
FLAG = 0  # runs despite an existing experiments directory if FLAG is set to 1. 

def fetch_next(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

def clip_outdim(model_name):
    model, _ = clip.load(model_name, jit=False)
    return model.embed_dim

def vision_model_outdim(model_name):
    model = create_model(model_name, pretrained=True)
    assert model.num_classes == 0, "Vision model should not have a classification head"
    return model.num_features

def language_model_outdim(model_name):
    model = AutoModel.from_pretrained(model_name)
    return model.config.hidden_size


def hparam_str(optim, lr, wd, batch_size, iters, dropout, learnable_temp):
    base = f"optim_{optim}-lr_{lr}-wd_{wd}-bs_{batch_size}-iters_{iters}"
    if dropout is not None:
        base += f"-dropout_{dropout}"
    if learnable_temp is True:
        base += f"-learnable_temp"
    return base


def savedir(outdir, dataset, encoder, train_shot, seed, text_type, text_shots, image_augmentation, mode, init_mode='zeroshot', alpha=0.0, text_bs=0, custom_name=''):
    benchname = '-'.join([dataset, get_few_shot_setup_name(train_shot, seed)])
    text_name = f"text_{text_type}"
    if text_shots is not None:
        text_name += f"_n_{text_shots}" 
    image_name = f"image_{image_augmentation}{custom_name}"
    mod_name = f"finetune-{text_name}-{image_name}" if mode == 'crossmodal' else f"finetune-{image_name}" if mode == 'image' else text_name
    mod_name = f'{mod_name}-alpha_{alpha}' if (alpha > 0 and mode == 'crossmodal') else mod_name
    mod_name = f"{mod_name}-text_bs_{text_bs}" if text_bs > 0 else mod_name
    return os.path.join(outdir, benchname, encoder.replace("/", "-"), mod_name, init_mode)


def train(model, image_loader, text_loader, val_loader, test_loader, optimizer, scheduler, device="cuda", max_iters=1000, alpha=1.0, eval_freq=100, patience = 5):
    out = {'iter': None, 'val_acc': None, 'model': None, 'val_classwise': None, 'val_loss': None}
    model.train()
    assert image_loader is not None or text_loader is not None, "At least one of the loaders should be provided"
    
    image_iter = iter(image_loader) if image_loader is not None else None
    text_iter = iter(text_loader) if text_loader is not None else None
    no_improve = 0

    img_alpha = 1.0
    for i in range(max_iters):
        labels, modality_flags = [], []
        if image_iter is not None:
            batch, image_iter = fetch_next(image_loader, image_iter)
            images, image_labels = batch['img'], batch['label']
            raw_images, image_labels = images.to(device), image_labels.to(device)
            labels.append(image_labels)
            modality_flags.append(torch.ones_like(image_labels))  
        if text_iter is not None:
            (text_features, text_labels, _), text_iter = fetch_next(text_loader, text_iter)
            text_features, text_labels = text_features.to(device), text_labels.to(device)
            labels.append(text_labels)
            modality_flags.append(torch.zeros_like(text_labels))  
        else:
            text_features = None
        labels = torch.cat(labels, dim=0).to(device) 
        modality_flags = torch.cat(modality_flags, dim=0).to(device)  
        
        optimizer.zero_grad()
        image_logits, text_logits = model(raw_images, text_features)
        image_indices = modality_flags == 1
        text_indices = modality_flags == 0
        image_loss = torch.nn.functional.cross_entropy(image_logits, labels[image_indices]) if image_indices.any() else 0.0
        text_loss = torch.nn.functional.cross_entropy(text_logits, labels[text_indices]) if text_indices.any() else 0.0
        loss = img_alpha * image_loss + alpha * text_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        img_acc = (torch.argmax(image_logits, dim=1) == labels[image_indices]).float().mean().item() if image_indices.any() else 0.0
        text_acc = (torch.argmax(text_logits, dim=1) == labels[text_indices]).float().mean().item() if text_indices.any() else 0.0

        if i % eval_freq == 0:
            val_loss, val_acc = validate(model, val_loader, device=device)
            testlog = ''
            if test_loader is not None:
                _, test_acc = validate(model, test_loader, device=device)
                testlog = f" | Test Acc: {test_acc:.4f}"
            if out['val_acc'] is None or val_acc > out['val_acc']:
                out['iter'] = i
                out['val_acc'] = val_acc
                out['val_loss'] = val_loss
                out['model'] = deepcopy(model.state_dict()) 
                no_improve = 0
            else:
                no_improve += 1
            
            print(f"Iter {i} | Img Loss: {image_loss:.4f} | Text Loss: {text_loss:.4f} | Img Acc: {img_acc:.4f} | Text Acc: {text_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc {val_acc:.4f}{testlog} | Count {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"=> Early stopping at Iter {i}")
                break
            
    model.load_state_dict(out['model'])
    val_loss, val_acc = validate(model, val_loader, device=device)
    print(f"=> Best Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f} at Iter {out['iter']}")
    return out


def validate(model, val_loader, device="cuda"):
    all_preds, all_labels, loss = [], [], []
    with torch.no_grad():
        model.eval()
        for batch in val_loader:
            image, image_label = batch['img'], batch['label']
            image, image_label = image.to(device), image_label.to(device)
            logits, _ = model(image)
            pred = torch.argmax(logits, dim=1)
            batch_loss = torch.nn.functional.cross_entropy(logits, image_label)
            
            all_preds.append(pred)
            all_labels.append(image_label)
            loss.append(batch_loss)

        all_preds, all_labels = torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
        val_acc = (all_preds == all_labels).float().mean().item()
        val_loss = torch.stack(loss).mean().item()

    return val_loss, val_acc


def setup(datasets, hparams, args):
    device = args.device
    ckpt_dir = os.path.join(args.savepath, hparam_str(hparams['optim'], hparams['lr'], hparams['weight_decay'], hparams['batch_size'], hparams['max_iter'],
                                                       hparams['dropout'], hparams['learnable_temp']))
    makedirs(ckpt_dir)
    test_path = os.path.join(ckpt_dir, 'test_result.pth')
    if os.path.exists(test_path) and (not FLAG):
        print(f"=> Skipping {ckpt_dir} as it already exists!")
        return torch.load(test_path)
    print(f"=> Setting up {ckpt_dir}")
    
    if args.use_clip:
        model = UMLClip(args.clip_encoder, args.nclasses, logit_scale_init=args.logit, bias=False, learnable_temp = hparams['learnable_temp']).to(device)                
    else:
        model = UML(args.vision_model, args.text_indim if args.modality == 'crossmodal' else 0, args.nclasses, bias=False, learnable_temp = hparams['learnable_temp'])

    # Initialize shared head with text embedding weights only when using both modalities
    if args.classifier_init == 'zeroshot' and args.modality == 'crossmodal':
        model.zero_shot_init(datasets['text_ds'])
    model.to(device)

    optimizer = build_optimizer(model.parameters(), hparams['optim'], hparams['lr'], hparams['weight_decay'])
    scheduler = build_lr_scheduler(optimizer, hparams['lr_scheduler'], hparams['warmup_iter'], hparams['max_iter'], warmup_type=hparams['warmup_type'], warmup_lr=hparams['warmup_min_lr'])

    image_loader = DataLoader(DatasetWrapper(datasets['img_tr_ds'], transform=datasets["tr_transform"]), batch_size=hparams['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    text_loader = DataLoader(datasets['text_ds'], batch_size=hparams['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    if args.modality == 'image':
        text_loader = None
        print("=> Running Unimodal: Image Only Model")
    elif args.modality == 'text':
        image_loader = None
        print("=> Running Unimodal: Text Only Model")

    val_loader = DataLoader(DatasetWrapper(datasets['img_val_ds'], transform=datasets["te_transform"]), batch_size=hparams['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(DatasetWrapper(datasets['img_te_ds'], transform=datasets["te_transform"]), batch_size=hparams['batch_size'], shuffle=False, num_workers=args.num_workers, pin_memory=True)


    result_dict = train(model, image_loader, text_loader, val_loader, 
                        test_loader if args.eval_test else None, optimizer, scheduler, device=device, 
                        max_iters=hparams['max_iter'], alpha=args.alpha, eval_freq=EVAL_FREQ, patience=hparams['patience'])
    
    test_loss, test_acc = validate(model, test_loader, device=device)
    test_dict = {'test_acc': test_acc, 'val_acc': result_dict['val_acc'], 'model': result_dict['model'], 'iter': result_dict['iter']}
    print(f"=> Test Acc: {test_acc:.4f}")
    if not FLAG or args.overwrite:
        print(f"=> Saving Test Results for hparams to {test_path}")
        torch.save(test_dict, test_path)
    return test_dict

def sweep(datasets, hyperparams, args):
    hyperparams ={k: (v if isinstance(v, list) else [v]) for k, v in hyperparams.items()}

    keys = list(hyperparams.keys())
    values = [hyperparams[k] for k in keys]
    total = 1
    for v in values:
        total *= len(v)

    results = {'test_acc': [], 'val_acc': [], 'hparams': []}
    
    # Iterate over all combinations of hyperparameter values.
    best_val_acc = 0
    best_test_acc = 0
    best_hparams = None
    for idx, combination in enumerate(product(*values)):
        combo_dict = dict(zip(keys, combination))
        print(f"=> Running {idx + 1}/{total}: {combo_dict}")
        out = setup(datasets, combo_dict, args)
        results['test_acc'].append(out['test_acc']); results['val_acc'].append(out['val_acc']); results['hparams'].append(combo_dict)
        if out['val_acc'] > best_val_acc:
            best_val_acc = out['val_acc']
            best_hparams = combo_dict
            best_test_acc = out['test_acc']
            print(f"=> New Best Val Acc: {best_val_acc:.4f} | Test Acc: {best_test_acc:.4f}")
        print(f"=> Best Val Acc (so far): {best_val_acc:.4f} | Test Acc (corresponding): {best_test_acc:.4f}")
        print(f"=> Best Hyperparameters (so far): {best_hparams}")
        print('--------------------------------------------------------\n')

    if not FLAG or args.overwrite:
        print(f"=> Saving results across all hparams to {args.savepath}")
        torch.save(results, os.path.join(args.savepath, 'results.pth'))
    
    best_idx = torch.argmax(torch.tensor(results['val_acc']))
    best_hparams = results['hparams'][best_idx]
    best_test_acc = results['test_acc'][best_idx]
    best_val_acc = results['val_acc'][best_idx]
    print(f"=> [FINAL] Best Val Acc: {best_val_acc:.4f} | Best Test Acc: {best_test_acc:.4f}")
    print(f"=> [FINAL] Mean Val Acc: {torch.tensor(results['val_acc']).mean():.4f} | Mean Test Acc: {torch.tensor(results['test_acc']).mean():.4f}")
    print(f"=> [FINAL] Best Hyperparameters: {best_hparams}")
    return results, best_val_acc, best_test_acc


def main(args):
    if args.seed >= 0:
        print("=> Setting fixed seed: {}".format(args.seed))
        set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.use_clip = args.vision_model=='' and args.language_model==''

    encoder_name = args.clip_encoder if args.use_clip else f'{args.vision_model}-{args.language_model}'
    args.savepath = savedir(args.result_dir, args.dataset, encoder_name, args.train_shot, 
                            args.seed, args.text_type, args.text_shot, args.image_augmentation, 
                            args.modality, args.classifier_init, args.alpha, 
                            args.text_batch_size if hasattr(args, 'text_batch_size') else 0, args.custom_name)
    makedirs(args.savepath)

    logfile = open(os.path.join(args.savepath, 'log.txt'), 'w')
    sys.stdout = Tee(sys.stdout, logfile) # Redirect stdout to the Tee object, which writes to both sys.__stdout__ and the log file
    print('=> Arguments:', args)

    # Load Text Features
    text_encoder_name = args.clip_encoder if args.use_clip else f'{args.language_model}'
    text_path = text_outdir(args.feature_dir, text_encoder_name, args.dataset, args.text_type)
    text_features = torch.load(text_path)
    text_ds = TextTensorDataset(text_features['features'], text_features['labels'], text_features['eot_indices'], n_shots=int(args.text_shot) if (args.text_shot!='average' and args.text_shot is not None) else args.text_shot)

    # Load Raw Images
    datasets = get_few_shot_benchmark(args.data_dir, args.indices_dir, args.dataset, args.train_shot, args.seed)
    img_tr_ds, img_val_ds, img_te_ds = datasets['train'], datasets['val'], datasets['test']
    lab2cname = datasets['lab2cname']
    tr_transform = build_transform(args.image_augmentation)
    te_transform = build_transform('crop')

    if args.use_clip:
        args.img_indim = args.text_indim = clip_outdim(args.clip_encoder)
    else:
        vision_indim = vision_model_outdim(args.vision_model)
        language_indim = language_model_outdim(args.language_model)
        args.img_indim = vision_indim
        args.text_indim = language_indim
    
    args.nclasses = len(lab2cname)
    datasets = {'img_tr_ds': img_tr_ds, 'text_ds': text_ds, 'img_val_ds': img_val_ds, 'img_te_ds': img_te_ds, "tr_transform": tr_transform, "te_transform": te_transform}

    hyperparams = HYPER_DICT[args.hyperparams]
    results, best_val_acc, best_test_acc = sweep(datasets, hyperparams, args)
    print("Done!")
    return results, best_val_acc, best_test_acc


if __name__ == "__main__":
    outer_parser = argparse.ArgumentParser(description="Synthetic Search Experiment")
    outer_parser.add_argument("-c", "--config", type=str, default="config.json", help="Configuration file")
    outer_parser.add_argument("-s", "--slurm", action="store_true", help="Launched with slurm")
    outer_parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    outer_parser.add_argument("-f", "--flag", action="store_true", help="Run despite existing experiments directory")
    outer_parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing experiments directory")
    outer_args, remaining_args = outer_parser.parse_known_args()

    FLAG = int(outer_args.flag)

    if outer_args.debug:
        print("Running command-line arguments...")
        args = parser.parse_args(remaining_args)
        args.overwrite = outer_args.overwrite
        main(args)
        sys.exit(0)
    
    with open(outer_args.config, "r") as f:
        sweep_args = yaml.load(f, Loader=yaml.FullLoader)
    
    keys, values = zip(*sweep_args.items())
    combinations = [dict(zip(keys, v)) for v in product(*[v if isinstance(v, list) else [v] for v in values])]

    print("Total combinations:", len(combinations))
    for i, combo in enumerate(combinations):
        print(f"Combination {i}: {combo}")

    if outer_args.slurm:
        job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "-1"))
        if job_id < 0 or job_id >= len(combinations):
            print("Invalid SLURM_ARRAY_TASK_ID")
            sys.exit(1)
        combination = combinations[job_id]
        print(f"=> Running combination {job_id}: {combination}")
        args = parser.parse_args([], argparse.Namespace(**combination))
        args.overwrite = outer_args.overwrite
        main(args)
    else:
        for i, combo in enumerate(combinations):
            print(f"=> Running job {i}")
            args = parser.parse_args([], argparse.Namespace(**combo))
            args.overwrite = outer_args.overwrite
            main(args)