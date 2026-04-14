import os, torch, statistics

def hparam_str(optim, lr, wd, batch_size, iters, dropout, learnable_temp, model_str):
    base = f"optim_{optim}-lr_{lr}-wd_{wd}-bs_{batch_size}-iters_{iters}"
    if dropout is not None:
        base += f"-dropout_{dropout}"
    if learnable_temp is True:
        base += f"-learnable_temp"
    if model_str is not None:
        base += f"-{model_str}"
    return base


def collect_results(datasets,
                    seeds,
                    encoders,
                    train_shots,
                    init_types,
                    modality_types,
                    experiments_dir="experiments",
                    text_datasets=[]):
    """
    Scans experiments/{dataset}-shot_{shot}-seed_{seed}/{encoder}/{modality}/{init_type}/results.pth
    and aggregates per-key across seeds.
    Key = (dataset, encoder, train_shot, init_type, modality_type)
    """

    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(seeds, list):
        seeds = [seeds]
    if not isinstance(encoders, list):
        encoders = [encoders]   
    if not isinstance(train_shots, list):
        train_shots = [train_shots]
    if not isinstance(init_types, list):
        init_types = [init_types]
    if not isinstance(modality_types, list):
        modality_types = [modality_types]
    if not text_datasets:
        text_datasets = ['']

    # 1) Gather all raw results by key
    raw = {} 
    for dataset in datasets:
        for text_ds in text_datasets:
            for shot in train_shots:
                for seed in seeds:

                    if text_ds != '':
                        seed_dir = os.path.join(
                            experiments_dir,
                            f"{dataset}-{text_ds}-shot_{shot}-seed_{seed}"
                        )
                    else:   
                        seed_dir = os.path.join(
                            experiments_dir,
                            f"{dataset}-shot_{shot}-seed_{seed}"
                        )
                    if not os.path.isdir(seed_dir):
                        continue

                    for encoder in encoders:
                        enc_dir = os.path.join(seed_dir, encoder)
                        if not os.path.isdir(enc_dir):
                            continue
                        for modality in modality_types:
                            mod_dir = os.path.join(enc_dir, modality)
                            if not os.path.isdir(mod_dir):
                                continue

                            for init in init_types:
                                init_dir = os.path.join(mod_dir, init)
                                results_file = os.path.join(init_dir, "results.pth")
                                if not os.path.isfile(results_file):
                                    continue

                                res = torch.load(results_file, map_location="cpu")
                                if text_ds != '':
                                    key = (dataset, text_ds, encoder, str(shot), init, modality)
                                else:
                                    key = (dataset, encoder, str(shot), init, modality)
                                raw.setdefault(key, []).append((int(seed), res, results_file))

    # 2) Aggregate per-key
    summary = {}
    for key, entries in raw.items():
        # collect each seed's best‐val index
        n_seeds = len(entries)
        vals, tests = [], []
        for seed, res, path in entries:
            best_i = max(range(len(res["val_acc"])),
                         key=lambda i: res["val_acc"][i])
            vals.append(res["val_acc"][best_i])
            tests.append(res["test_acc"][best_i])

        # compute means & stds
        mean_val = statistics.mean(vals)
        std_val  = statistics.stdev(vals) if len(vals)>1 else 0.0
        mean_test = statistics.mean(tests)
        std_test  = statistics.stdev(tests) if len(tests)>1 else 0.0

        # pick the single best seed (highest val) to get hparams + path
        best_entry = max(entries,
                         key=lambda tup: max(tup[1]["val_acc"]))
        best_seed, best_res, best_path = best_entry
        best_i = max(range(len(best_res["val_acc"])),
                     key=lambda i: best_res["val_acc"][i])
        best_hparams = best_res["hparams"][best_i]
        
        # parent directory of best_path
        model_str = f"pos_embd_{best_hparams.get('pos_embd', None)}-pos_learnable_{best_hparams.get('pos_learnable', None)}" if best_hparams.get('pos_embd', None) is not None or best_hparams.get('pos_learnable', None) is not None else None
        best_path = os.path.dirname(best_path)
        best_path = os.path.join(best_path, 
                                 hparam_str(best_hparams['optim'], best_hparams['lr'], best_hparams['weight_decay'], best_hparams['batch_size'], best_hparams['max_iter'], best_hparams.get('dropout', None), best_hparams.get('learnable_temp', None), model_str),
                                 'test_result.pth')

        summary[key] = {
            "mean_val_acc":  mean_val,
            "std_val_acc":   std_val,
            "mean_test_acc": mean_test,
            "std_test_acc":  std_test,
            "n_seeds":       n_seeds,
            "best_seed":     best_seed,
            "best_hparams":  best_hparams,
            "best_path":     best_path
        }

    return summary


if __name__ == "__main__":

    # Specify datasets, encoders, shots etc. across which to collect results
    experiments_dir = "experiments"
    datasets = ['fgvc_aircraft']
    text_shots = ['average', 1, 5, 15]
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0]
    seeds = [1, 2, 3]
    train_shots = [-1]
    init_types = ['random', 'zeroshot']
    modality_types = [f'finetune-text_gpt3_cupl_n_{k}-image_crop-alpha_{a}/' for k in text_shots for a in alphas]
    modality_types.append('finetune-image_crop')
    encoders_list = ['vit_small_patch14_dinov2.lvd142m-openlm-research-open_llama_3b_v2']
    
    summary = collect_results(
        datasets       = datasets,
        seeds          = seeds,
        encoders       = encoders_list,
        train_shots    = train_shots,
        init_types     = init_types,
        modality_types = modality_types,
        experiments_dir= experiments_dir
    )

    # pretty‐print
    header = f"{'Dataset':<12} {'Encoder':<40} {'Shot':<5} {'Init':<10} {'Modality':<35} {'Test':>7} {'Val':>7}"
    print(header)
    print("-"*len(header))
    for key in sorted(summary):
        ds, enc, shot, init, mod = key
        info = summary[key]
        print(key, info['mean_test_acc'],  info['mean_val_acc'])
