from engine.config import defaults
from engine.datasets import dataset_classes
import argparse
from typing import Union

parser = argparse.ArgumentParser()

###########################
# Directory Config (modify if using your own paths)
###########################
parser.add_argument(
    "--data_dir",
    type=str,
    default=defaults.DATA_DIR,
    help="where the dataset is saved",
)
parser.add_argument(
    "--indices_dir",
    type=str,
    default=defaults.FEW_SHOT_DIR,
    help="where the (few-shot) indices is saved",
)
parser.add_argument(
    "--description_dir",
    type=str,  
    default=defaults.DESCRIPTION_DIR,
    help="where the text descriptions are saved",
)
parser.add_argument(
    "--feature_dir",
    type=str,
    default=defaults.FEATURE_DIR,
    help="where to save pre-extracted features",
)
parser.add_argument(
    "--result_dir",
    type=str,
    default=defaults.RESULT_DIR,
    help="where to save experiment results",
)

###########################
# Dataset Config (few_shot_split.py)
###########################
parser.add_argument(
    "--dataset",
    type=str,
    default="fgvc_aircraft",
    choices=dataset_classes.keys(),
    help="number of train shot",
)
parser.add_argument(
    "--train-shot",
    type=int,
    default=1,
    help="number of train shot",
)
parser.add_argument(
    "--max-val-shot",
    type=int,
    default=4,
    help="number of val shot is min(max_val_shot, train_shot)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed number",
)


# Feature Extraction Config (features.py)
parser.add_argument(
    "--clip-encoder",
    type=str,
    default="RN50",
    choices=["ViT-B/16", "ViT-B/32", "RN50", "RN101"],
    help="specify the clip encoder to use",
)

parser.add_argument(
    "--vision-model",
    type=str,
    default="",
    choices=["vit_base_patch16_224_dino",
             "vit_base_patch8_224_dino",
             "vit_small_patch14_dinov2.lvd142m",
             "vit_base_patch14_dinov2.lvd142m",
             'vit_large_patch14_dinov2.lvd142m',
             ],
    help="specify the vision encoder to use",
)
parser.add_argument(
    "--language-model",
    type=str,
    default="",
    choices=["bert-base-uncased", 
             "bert-large-uncased",
             "roberta-base",
             "roberta-large",
             "openlm-research/open_llama_3b_v2", 
             "meta-llama/Llama-2-7b-chat-hf",
             "gpt2",
             "gpt2-medium",
             "gpt2-large",
             "mistralai/Mistral-7B-v0.1",
             "bigscience/bloom-1b1"
             ],
    help="specify the language encoder to use",
)

parser.add_argument(
    '--descriptor_type',
    type=str,
    default=None,
    choices=['gpt3_cupl'],
    help='specify the descriptor type to use',
)
parser.add_argument(
    "--text-augmentation",
    type=str,
    default='vanilla',
    choices=['hand_crafted', # tip_adapter selected
             'classname', # plain class name
             'vanilla', # a photo of a {cls}.
             'template_mining' # examples of best zero-shot templates for few-shot val set
             ],
    help="specify the text augmentation to use.",
)

parser.add_argument(
    "--image-augmentation",
    type=str,
    default='crop',
    choices=['crop', # only a single center crop
             'flip', # add random flip view
             'randomcrop', # add random crop view
             ],
    help="specify the image augmentation to use.",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="batch size for test (feature extraction and evaluation)",
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
    help="number of workers for dataloader",
)


###########################
# Training Config (finetune.py)
###########################

parser.add_argument(
    '--text_shot',
    default=None,
    help='number of text shot',
)

parser.add_argument(
    '--custom-name',
    default='',
    help='custom name for the experiment save_dir',
)


parser.add_argument(
    "--modality",
    type=str,
    default="image",
    choices=["crossmodal", # using unpaired image and text samples
             "image", # using only image samples
             "text", # using only text samples
    ],
    help="whether or not to perform cross-modal training (ie. half batch is image, half batch is text)",
)

parser.add_argument(
    "--classifier_init",
    type=str,
    default="zeroshot",
    choices=["zeroshot", # text-based initialization
             "random", # random initialization
    ],
    help="classifier head initialization",
)

# This argument to jointly use one argument for descriptions and standard text prompts like "a photo of a {cls}"
parser.add_argument( 
    "--text_type",
    type=str,
    default="hand_crafted",
    choices=['gpt3_dclip', # gpt3_dclip
             'hand_crafted', # tip_adapter selected
             'classname', # plain class name
             'vanilla', # a photo of a {cls}.
             'template_mining' # examples of best zero-shot templates for few-shot val set
             ],
    help="text type to use for training",
)

parser.add_argument(
    "--logit",
    type=float,
    default=4.60517, # CLIP's default logit scaling
    choices=[4.60517, # CLIP's default logit scaling
             4.0, # for partial finetuning
    ],
    help="logit scale (exp(logit) is the inverse softmax temperature)",
)
parser.add_argument(
    "--hyperparams",
    type=str,
    default="linear",
    help="hyperparams sweep",
)


parser.add_argument(
    "--eval_test",
    action="store_true",
    default=False,
    help="Evaluate on test set during training to compute test accuracy",
)

parser.add_argument(
    '--alpha',
    type=float,
    default=0.0,
    help='weight for text modality in the loss function during crossmodal training',
)

parser.add_argument(
    '--flip_projection',
    type=bool,
    default=False,
    help='flip projection for linear head',
)
