## Goal

- Evaluate if the top-k Fisher specturm mass can represent the modal interaction
- If the top-k Fisher specturm mass can not represent the modal interaction, try other fisher / hessian related metrics

## Impelment environment

- conda vlm

## Datasets - '/data/zhenshuo/datasets/'

- imdb
- humor
- sarcasm
- MOSI
- MOSEI
- avmnist
- oxford_pet

## Multimodal Model arch

- concat MLP
- cross attention
- shared MLP

## Training

- supervised training

## Two kinds of unrelated modality

- randomly select another unpaired sample in the dataset
- add Gaussian noise (in $\{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1\}$)

## Encoder

- OpenLlama-v3
- CLIP
- Dino-v2
