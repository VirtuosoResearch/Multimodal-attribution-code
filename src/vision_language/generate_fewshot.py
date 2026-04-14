import os
from engine.config import parser
from engine.datasets import dataset_classes

from engine.tools.utils import makedirs, save_as_json, set_random_seed
from engine.datasets.utils import get_few_shot_setup_name
from engine.datasets.benchmark import generate_fewshot_dataset
import argparse
import yaml
import sys
from itertools import product

def main(args):
    if args.seed >= 0:
        print("Setting fixed seed: {}".format(args.seed))
        set_random_seed(args.seed)

    # Check if the dataset is supported
    assert args.dataset in dataset_classes
    few_shot_index_file = os.path.join(
        args.indices_dir,
        args.dataset,
        get_few_shot_setup_name(args.train_shot, args.seed) + ".json"
    )
    if os.path.exists(few_shot_index_file):
        print(f"Few-shot data exists at {few_shot_index_file}.")
    else:
        print(f"Few-shot data does not exist at {few_shot_index_file}. Sample a new split.")
        makedirs(os.path.dirname(few_shot_index_file))
        benchmark = dataset_classes[args.dataset](args.data_dir)
        few_shot_dataset = generate_fewshot_dataset(
            benchmark.train,
            benchmark.val,
            num_shots=args.train_shot,
            max_val_shots=args.max_val_shot,
        )
        save_as_json(few_shot_dataset, few_shot_index_file)
    print("Done!")

if __name__ == "__main__":
    outer_parser = argparse.ArgumentParser(description="Synthetic Search Experiment")
    outer_parser.add_argument("-c", "--config", type=str, default="config.json", help="Configuration file")
    outer_parser.add_argument("-s", "--slurm", action="store_true", help="Launched with slurm")
    outer_parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    outer_args, remaining_args = outer_parser.parse_known_args()

    if outer_args.debug:
        print("Running command-line arguments...")
        args = parser.parse_args(remaining_args)
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
        print("=> Parsed arguments:", args)
        main(args)
    else:
        for i, combo in enumerate(combinations):
            print(f"=> Running job {i}")
            args = parser.parse_args([], argparse.Namespace(**combo))
            main(args)

