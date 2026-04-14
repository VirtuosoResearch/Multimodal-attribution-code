HYPER_DICT = {
    # for full finetuning experiments
    'full_ds_full_model_finetune': {
        'optim': "adamw",
        'lr': [5.0e-5, 0.001, 0.0001],
        'weight_decay': [0.0],
        'lr_scheduler': "cosine",
        'batch_size': [64],
        'max_iter': [12800],
        'warmup_iter': 50,
        'warmup_type': "linear",
        'warmup_min_lr': 1e-5,
        'dropout': [0.0],
        'learnable_temp': [False],
        'patience': [10]
    },
    # for linear probe experiments using CLIP encoders
    'clip_linear': {
        'optim': "adamw",
        'lr': [0.001, 0.0001],
        'weight_decay': [0.0, 0.01, 0.001],
        'lr_scheduler': "cosine",
        'batch_size': [32],
        'max_iter': [12800],
        'warmup_iter': 50,
        'warmup_type': "linear",
        'warmup_min_lr': 1e-5,
        'dropout': [0.0],
        'learnable_temp': [False],
        'patience': [5]
    },
    # for linear probe experiments using unimodalvision and language encoders
    'linear': {
        'optim': "adamw",
        'lr': [0.001, 0.0001],
        'weight_decay': [0.0, 0.01, 0.001],
        'lr_scheduler': "cosine",
        'batch_size': [32],
        'max_iter': [12800],
        'warmup_iter': 50,
        'warmup_type': "linear",
        'warmup_min_lr': 1e-5,
        'dropout': [0.0],
        'learnable_temp': [True],
        'patience': [10]
    },
    'audio': {
        'optim': "adamw",
        'lr': [0.1, 0.01, 0.001, 0.0001],
        'weight_decay': [0.0, 0.01, 0.0001],
        'lr_scheduler': "cosine",
        'batch_size': [8],
        'max_iter': [12800],
        'warmup_iter': 50,
        'warmup_type': "linear",
        'warmup_min_lr': 1e-5,
        'dropout': [0.0],
        'learnable_temp': [False],
        'patience': [5]
    }
}