import os
import sys
from models import Transformer, Linear
from train import UML, train, set_seed
from torch import optim
from datasets.affect.get_data import get_dataloader
import torch
import yaml
from itertools import product
import random
import numpy as np
import argparse
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='x')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--zdim', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--n_seeds', type=int, default=1)
parser.add_argument('--ds_name', type=str, default='mosi')
parser.add_argument('--step_k', type=int, default=-1)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--pos_embd', action='store_true')
parser.add_argument('--pos_learnable', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--log_dir', type=str, default='./logs')



def main(args):
    log_dir = args.log_dir
    fname = f"log_{args.run_name}{args.ds_name}_mod{args.modality}_zdim{args.zdim}_epochs{args.num_epochs}_pos_embd_{args.pos_embd}_learnable_{args.pos_learnable}_step_k{args.step_k}_n_seeds{args.n_seeds}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join(args.results_dir, fname)
    
    if not args.debug:  
        os.makedirs(log_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = log_dir + "/wandb"
        wandb.init(entity="<USENAME>", project="<PROJECTNAME>", name = fname)
        wandb.config.update(args)

    print("Command-line arguments:", sys.argv)
    print("Parsed arguments:", args)

    seeds = [i for i in range(args.n_seeds)]
    outs = {'score_x': [], 'score_y': [], 'score_xy': [], 'val_score_x': [], 'val_score_y': [], 'val_score_xy': []}
    for seed in seeds:
        set_seed(seed)
        if args.ds_name == 'mosi':
            batch_size=32
            indims = [20, 300]
            # Sample two train loaders to shuffle x-only and y-only data within each dataloader
            train_loader, *_ = get_dataloader('./data_files/mosi_data.pkl', robust_test=False, batch_size=batch_size, train_shuffle=True)
            train_loader_2, *_ = get_dataloader('./data_files/mosi_data.pkl', robust_test=False, batch_size=batch_size, train_shuffle=True)
            eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader('./data_files/mosi_data.pkl', robust_test=False, batch_size=batch_size, train_shuffle=False)

        elif args.ds_name == 'sarcasm':
            batch_size=128
            indims = [371, 300]
            train_loader, *_ = get_dataloader('./data_files/sarcasm.pkl', batch_size=batch_size, data_type='sarcasm', vision_norm=True)
            train_loader_2, *_ = get_dataloader('./data_files/sarcasm.pkl', batch_size=batch_size, data_type='sarcasm', vision_norm=True)
            eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader('./data_files/sarcasm.pkl', batch_size=batch_size, data_type='sarcasm', train_shuffle=False, vision_norm=True)
            

        elif args.ds_name == 'humor':
            batch_size=128
            indims = [371, 300]
            train_loader, *_ = get_dataloader('./data_files/humor.pkl', batch_size=batch_size, data_type='humor')
            train_loader_2, *_ = get_dataloader('./data_files/humor.pkl', batch_size=batch_size, data_type='humor')
            eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader('./data_files/humor.pkl', batch_size=batch_size, data_type='humor', train_shuffle=False)
            
        elif args.ds_name == 'mimic':
            from datasets.mimic.get_data import get_dataloader as get_mimic_dataloader
            batch_size = 128
            indims = [5, 12]
            train_loader, *_ = get_mimic_dataloader(7, batch_size=batch_size, imputed_path='./data_files/im.pk')
            train_loader_2, *_ = get_mimic_dataloader(7, batch_size=batch_size, imputed_path='./data_files/im.pk')
            eval_train_loader, eval_valid_loader, eval_test_loader = get_mimic_dataloader(7, imputed_path='./data_files/im.pk', train_shuffle=False)        
            eval_test_loader = eval_valid_loader # as per FACTOR-CL codebase
        
        elif args.ds_name == 'mosei':
            batch_size=32
            indims = [35, 300]
            train_loader, *_ = get_dataloader('./data_files/mosei_senti_data.pkl', robust_test=False, batch_size=batch_size, data_type='mosei', train_shuffle=True)
            train_loader_2, *_ = get_dataloader('./data_files/mosei_senti_data.pkl', robust_test=False, batch_size=batch_size, data_type='mosei', train_shuffle=True)
            eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader('./data_files/mosei_senti_data.pkl', robust_test=False, batch_size=batch_size, data_type='mosei', train_shuffle=False)
        else:
            raise NotImplementedError('Dataset not implemented yet')

        # Dataset stats
        print("Dataset: ", args.ds_name)
        print("Batch size: ", batch_size)
        print("Train dataset: ", len(train_loader)*batch_size)
        print("Eval train dataset: ", len(eval_train_loader)*batch_size)
        print("Eval test dataset: ", len(eval_test_loader)*batch_size)
        print(f"Modality Info: xdim: {indims[0]}, ydim: {indims[1]}, zdim: {args.zdim}")


        # Initialize model and optimizer
        xproj_in = Linear(indims[0], args.zdim)
        yproj_in = Linear(indims[1], args.zdim)
        shared_encoder = Transformer(args.zdim, args.zdim, nhead=5, num_layers=5, conv1d=True, out_last=False, pos_embd=args.pos_embd, pos_learnable=args.pos_learnable, max_len=128)
        decoders = [Linear(args.zdim, indims[0]), Linear(args.zdim, indims[1])]
        model = UML(xproj_in, yproj_in, shared_encoder, decoders, modality=args.modality).cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        score = train(model, args.modality,train_loader, train_loader_2,optimizer, 
                        num_epoch=args.num_epochs, step_k=args.step_k,ds_name=args.ds_name,
                        eval_config={'train': eval_train_loader, 'val': eval_valid_loader, 'test': eval_test_loader, 'freq': 100},
                        augment=args.augment, debug=args.debug)

        print('seed: ', seed, ' score: ', score)
        print('=====================================')
        outs['score_x'].append(100*score[0])
        outs['score_y'].append(100*score[1])
        outs['score_xy'].append(100*score[2])
        outs['val_score_x'].append(100*score[3])
        outs['val_score_y'].append(100*score[4])
        outs['val_score_xy'].append(100*score[5])

    print(outs)
    
    # Mean across seeds
    outs_mean = {k: np.mean(v) for k, v in outs.items()}
    outs_std = {k: np.std(v) for k, v in outs.items()}
    print("Final scores (mean): ", outs_mean)
    print("Final scores (std): ", outs_std)
    
    if not args.debug:
        wandb.log({f'final_{k}': v for k, v in outs_mean.items()})
        wandb.log({f'final_std_{k}': v for k, v in outs_std.items()})

    # save model and log outputs
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving model to {os.path.join(results_dir, 'model.pth')}")
    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
    with open(os.path.join(results_dir, "outputs.txt"), "w") as f:
        f.write(f"Final scores: {outs_mean}\n")
        f.write(f"Final scores std: {outs_std}\n")


if __name__ == "__main__":
    outer_parser = argparse.ArgumentParser(description="MultiBench Experiment")
    outer_parser.add_argument("-c", "--config", type=str, default="config.json", help="Configuration file")
    outer_parser.add_argument("-s", "--slurm", action="store_true", help="Launched with slurm")
    outer_parser.add_argument("-d", "--outer_debug", action="store_true", help="Debug mode")
    outer_parser.add_argument("-f", "--flag", action="store_true", help="Run despite existing experiments directory")
    outer_parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing experiments directory")
    outer_args, remaining_args = outer_parser.parse_known_args()

    FLAG = int(outer_args.flag)

    if outer_args.outer_debug:
        print("Running command-line arguments...")
        args = parser.parse_args(remaining_args)
        args.overwrite = outer_args.overwrite
        args.debug = True
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
        # args.debug = True
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

