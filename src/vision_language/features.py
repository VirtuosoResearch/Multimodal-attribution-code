import os
from engine.config import parser
import torch
from engine.tools.utils import makedirs, set_random_seed, cname2lab
import torch.nn.functional as F
from engine.transforms.default import build_transform
from engine.datasets.utils import DatasetWrapper, get_few_shot_setup_name, get_few_shot_benchmark, get_testset
from engine.clip import clip
from engine.templates import get_templates
from engine.descriptors.default import DESCRIPTOR_DICT
from engine.descriptors.gpt3_descriptors import load_gpt_descriptions
from timm.models import create_model
from tqdm import tqdm
from engine.models.languagemodel import TextModel
import argparse
import yaml
import sys
from itertools import product
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # If not set to false, can give "Too many open files" error or "Bus error" error

IMAGENET_TESTSETS = [
    'imagenetv2',
    'imagenet_sketch',
    'imagenet_a',
    'imagenet_r',
]


def img_outdir(outdir, encoder, ds, augmentation, tr_shot, seed, mode='train', return_tokens=False):
    subpath = 'patch-token' if return_tokens else ''
    if mode == 'train':
        return os.path.join(outdir, subpath, 'image', encoder.replace("/", "-"), ds, augmentation, f"{get_few_shot_setup_name(tr_shot, seed)}.pth")
    return os.path.join(outdir, subpath, 'image', encoder.replace("/", "-"), ds, "test.pth")
        
def text_outdir(outdir, encoder, ds, text_augmentation, return_tokens=False):
    subpath = 'patch-token' if return_tokens else ''
    return os.path.join(outdir, subpath, 'text', encoder.replace("/", "-"), ds, f"{text_augmentation}.pth")

def descriptor_outdir(outdir, encoder, ds, descriptor_type, return_tokens=False):
    subpath = 'patch-token' if return_tokens else ''
    return os.path.join(outdir, subpath, 'text', encoder.replace("/", "-"), ds, f"{descriptor_type}.pth")


def load(model_name='RN50', device='cuda'):
    model, _ = clip.load(model_name, jit=False)
    model.float()
    model.requires_grad_(False) # Freeze the model
    model.to(device)
    return model

def descriptor_features(model, tokenizer, descriptors, lab2cname, device='cuda', is_clip=True, return_tokens=False):
    features, labels, eot_indices, prompts_dict, all_prompts = [], [], [], {}, []
    model.eval()
    cname2lab_dict = cname2lab(lab2cname)
    
    with torch.no_grad():
        for idx, (cls, descriptions) in enumerate(descriptors.items()):
            try:
                label = cname2lab_dict[cls.replace(' ', '_').lower()] 
            except:
                print(f"[!!!] Class not found in lab2cname dict corresponding to {cls}")
                continue
            
            if is_clip:
                prompts = clip.tokenize(descriptions).to(device)
                out, indices = model.encode_text(prompts, return_eot=True, return_tokens=return_tokens)
            else:
                prompts = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt").to(device)
                out = model(prompts, return_tokens=return_tokens)
                if return_tokens:
                    indices = torch.tensor([prompts['attention_mask'][i].sum().item() for i in range(prompts['attention_mask'].size(0))], dtype=torch.long).to(device)
                else:
                    indices = torch.tensor([len(t) for t in descriptions], dtype=torch.long).to(device)
            
            if return_tokens and idx==0:
                print('Shape of token embeddings:', out.shape)
            features.append(out.cpu())
            all_prompts.extend(descriptions)
            labels.append(torch.tensor([label]*len(descriptions), dtype=torch.long))
            eot_indices.append(indices.cpu())
            prompts_dict[label] = descriptions

    if return_tokens and (not is_clip):
        max_len = max(feat.shape[1] for feat in features)
        padded = []
        for feat in features:
            if feat.shape[1] < max_len:
                pad_amt = max_len - feat.shape[1]
                padded.append(F.pad(feat, (0, 0, 0, pad_amt)))  # pad tokens on length dim
            else:
                padded.append(feat)
        features = padded
    return {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "eot_indices": torch.cat(eot_indices, dim=0),
        "prompts": prompts_dict,
        "lab2cname": lab2cname,
        "cname2lab": cname2lab_dict
    }



def text_features(model, tokenizer, dsname, lab2cname, augmentation, device='cuda', is_clip=True, return_tokens=False):
    templates = get_templates(dsname, augmentation)
    tot_features, tot_labels, tot_eot_indices, prompts_dict = [], [], [], {}
    model.eval()

    with torch.no_grad():
        for i, (label, cname) in enumerate(lab2cname.items()):
            text_prompts = [t.format(cname.replace("_", " ")) for t in templates]
            if is_clip:
                prompts = clip.tokenize(text_prompts).to(device)
                out, indices = model.encode_text(prompts, return_eot=True, return_tokens=return_tokens)
            else:
                prompts = tokenizer(text_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
                out = model(prompts, return_tokens=return_tokens)
                if return_tokens:
                    indices = torch.tensor([prompts['attention_mask'][i].sum().item() for i in range(prompts['attention_mask'].size(0))], dtype=torch.long).to(device)
                else:
                    indices = torch.tensor([len(t) for t in text_prompts], dtype=torch.long).to(device)
            
            if return_tokens and i==0:
                print('Shape of token embeddings:', out.shape)
            tot_features.append(out.cpu())
            tot_labels.append(torch.tensor([label]*len(templates), dtype=torch.long))
            tot_eot_indices.append(indices.cpu())
            prompts_dict[label] = text_prompts
    
    if return_tokens and (not is_clip):
        max_len = max(feat.shape[1] for feat in tot_features)
        padded = []
        for feat in tot_features:
            if feat.shape[1] < max_len:
                pad_amt = max_len - feat.shape[1]
                padded.append(F.pad(feat, (0, 0, 0, pad_amt)))
            else:
                padded.append(feat)
        tot_features = padded
    return {
        "features": torch.cat(tot_features, dim=0),
        "labels": torch.cat(tot_labels, dim=0),
        "eot_indices": torch.cat(tot_eot_indices, dim=0),
        "prompts": prompts_dict,
        'lab2cname': lab2cname
    }


def image_features(model, ds, transform, bs, num_workers, device='cuda', is_clip=True, return_tokens=False):
    loader = torch.utils.data.DataLoader(
        DatasetWrapper(ds, transform=transform),
        batch_size=bs,
        sampler=None,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    features, labels, paths= [], [], []
    for i, batch in enumerate(tqdm(loader)):
        images = batch["img"].to(device)
        with torch.no_grad():
            if is_clip:
                out = model.encode_image(images, return_tokens=return_tokens).cpu()
            else:
                out = model.forward(images).cpu()
                if return_tokens:
                    out = model.forward_features(images).cpu()
            if return_tokens and i==0:
                print('Shape of image patch embeddings:', out.shape)
            features.append(out)
            labels.append(batch['label'])
            paths.extend(batch['impath'])
    
    return {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "paths": paths
    }

def prepare_text_features(model, tokenizer, args, ds):
    if args.descriptor_type is not None:
        hparams = DESCRIPTOR_DICT[args.descriptor_type]
        dirname = hparams['dirname']
        if not os.path.join(args.description_dir, dirname, f"descriptors_{args.dataset}.json"):
            print(f'Descriptor file not found at {os.path.join(args.description_dir, dirname, f"descriptors_{args.dataset}.json")}!')
        else:
            text_encoder_name = args.clip_encoder if args.use_clip else args.language_model
            descriptor_path = descriptor_outdir(args.feature_dir, text_encoder_name, args.dataset, args.descriptor_type, args.return_tokens)

            if args.overwrite or not os.path.exists(descriptor_path):
                if args.overwrite:
                    print(f"=> Saving descriptor features to {descriptor_path} because overwrite is set to True")
                else:
                    print(f"=> Saving descriptor features to {descriptor_path} because it does not exist")
                hparams['fname'] = os.path.join(args.description_dir, dirname, f"descriptors_{args.dataset}.json")
                hparams['dsname'] = args.dataset
                descriptions, _ = load_gpt_descriptions(hparams)
                features = descriptor_features(model, tokenizer, descriptions, ds['lab2cname'], args.device, is_clip=args.use_clip, return_tokens=args.return_tokens)
                
                makedirs(os.path.dirname(descriptor_path))
                torch.save(features, descriptor_path)
            else:
                print(f"=> Descriptor features already saved at {descriptor_path} and overwrite is set to False")

    text_encoder_name = args.clip_encoder if args.use_clip else args.language_model
    feature_path = text_outdir(args.feature_dir, text_encoder_name, args.dataset, args.text_augmentation, args.return_tokens)
    makedirs(os.path.dirname(feature_path))
    
    if args.overwrite or not os.path.exists(feature_path):
        if args.overwrite:
            print(f"=> Saving text features to {feature_path} because overwrite is set to True")
        else:
            print(f"=> Saving text features to {feature_path} because it does not exist")
        features = text_features(model, tokenizer, args.dataset, ds['lab2cname'], args.text_augmentation, args.device, is_clip=args.use_clip, return_tokens=args.return_tokens)
        torch.save(features, feature_path)
    else:
        print(f"=> Text features already saved at {feature_path} and overwrite is set to False")

def prepare_image_features(model, args, ds, mode='train'):
    encoder_name = args.clip_encoder if args.use_clip else args.vision_model
    feature_path = img_outdir(args.feature_dir, encoder_name, args.dataset, args.image_augmentation, args.train_shot,  args.seed, mode, args.return_tokens)
    makedirs(os.path.dirname(feature_path))

    if args.overwrite or not os.path.exists(feature_path):
        if args.overwrite:
            print(f"=> Saving image features to {feature_path} because overwrite is set to True")
        else:
            print(f"=> Saving image features to {feature_path} because it does not exist")
        
        tr_transform = build_transform(args.image_augmentation)
        te_transform = build_transform('crop')
        if mode == 'train':
            features = {'train': {}, 'val': {}}
            features['train'] = image_features(model, ds['train'], tr_transform, args.batch_size, args.num_workers, args.device, is_clip=args.use_clip, return_tokens=args.return_tokens)
            features['val'] = image_features(model, ds['val'], tr_transform, args.batch_size, args.num_workers, args.device, is_clip=args.use_clip, return_tokens=args.return_tokens)            
        else:
            features = image_features(model, ds['test'], te_transform, args.batch_size, args.num_workers, args.device, is_clip=args.use_clip, return_tokens=args.return_tokens)
        
        features['lab2cname'] = ds['lab2cname'] if 'lab2cname' in ds.keys() else None
        torch.save(features, feature_path)
    else:
        print(f"=> Image features already saved at {feature_path} and overwrite is set to False")


def main(args):
    if args.seed >= 0:
        print("Setting fixed seed: {}".format(args.seed))
        set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset not in IMAGENET_TESTSETS:
        datasets = get_few_shot_benchmark(args.data_dir, args.indices_dir, args.dataset, args.train_shot, args.seed)
        print(f'=> Dataset sizes: train: {len(datasets["train"])}, val: {len(datasets["val"])}, test: {len(datasets["test"])}')
    else:
        datasets = get_testset(args.dataset, args.data_dir)
    
    args.use_clip = args.vision_model=='' and args.language_model==''
    if args.use_clip:
        print("=> Using CLIP model")
        vision_model = load(args.clip_encoder, args.device)
        text_model = load(args.clip_encoder, args.device)
        tokenizer = None
    else:
        print(f"=> Using {args.vision_model} for vision and {args.language_model} for language")
        vision_model = create_model(args.vision_model, pretrained=True, img_size=224).to(args.device)
        assert vision_model.num_classes == 0, "Vision model should not have a classification head !!!!"
        text_model = TextModel(args.language_model).to(args.device)
        tokenizer = text_model.tokenizer

    if args.dataset not in IMAGENET_TESTSETS:
        prepare_image_features(vision_model, args, datasets, mode='train')
        prepare_image_features(vision_model, args, datasets, mode='test')
        prepare_text_features(text_model, tokenizer, args, datasets)
    else:
        print(f"=> Saving ImageNet testset: {args.dataset}, only preparing image features")
        datasets_dict = {'test': datasets}
        prepare_image_features(vision_model, args, datasets_dict, mode='test')
    print("Done!")


if __name__ == "__main__":
    outer_parser = argparse.ArgumentParser(description="Feature Extraction")
    outer_parser.add_argument("-c", "--config", type=str, default="config.json", help="Configuration file")
    outer_parser.add_argument("-s", "--slurm", action="store_true", help="Launched with slurm")
    outer_parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    outer_parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing features")
    outer_args, remaining_args = outer_parser.parse_known_args()

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
        print("=> Parsed arguments:", args)
        main(args)
    else:
        for i, combo in enumerate(combinations):
            print(f"==> Running job {i}")
            args = parser.parse_args([], argparse.Namespace(**combo))
            args.overwrite = outer_args.overwrite
            main(args)
