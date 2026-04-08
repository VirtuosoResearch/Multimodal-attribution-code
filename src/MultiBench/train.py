from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math
import copy
from sklearn.linear_model import LogisticRegression
import numpy as np
import wandb
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Set seed
def set_seed(seed):
    import os
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Critical: disable cuDNN benchmarking
    os.environ['PYTHONHASHSEED'] = str(seed)
    
# Simple Augmentations 
def permute(x):
  # shuffle the sequence order
  idx = torch.randperm(x.shape[0])
  return x[idx]

def noise(x):
  noise = torch.randn(x.shape) * 0.1
  return x + noise.to(x.device)

def drop(x):
  # drop 20% of the sequences
  drop_num = x.shape[0] // 5
  
  x_aug = torch.clone(x)
  drop_idxs = np.random.choice(x.shape[0], drop_num, replace=False)
  x_aug[drop_idxs] = 0.0
  return x_aug  

def mixup(x, alpha=1.0):
    indices = torch.randperm(x.shape[0])
    lam = np.random.beta(alpha, alpha)
    aug_x = x * lam + x[indices] * (1 - lam)

    return aug_x

def identity(x):
  return x

def augment(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [permute, noise, drop, identity]

  for i in range(x_batch.shape[0]):
    t_idxs = np.random.choice(4, 2, replace=False)
    t1 = transforms[t_idxs[0]]
    t2 = transforms[t_idxs[1]]
    v1[i] = t1(v1[i])
    v2[i] = t2(v2[i])
  
  return v1, v2

def augment_single(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [permute, noise, drop, identity]

  for i in range(x_batch.shape[0]):
    t_idxs = np.random.choice(4, 1, replace=False)
    t = transforms[t_idxs[0]]
    v2[i] = t(v2[i])
  
  return v2


def augment_embed_single(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [noise, mixup, identity]

  t_idxs = np.random.choice(3, 1, replace=False)
  t = transforms[t_idxs[0]]
  v2 = t(v2)

  return v2


def augment_mimic(x_batch):
  if x_batch.dim() == 2:
    return augment_embed_single(x_batch)
  else:
    return augment_single(x_batch)

# MOSI/MOSEI Training
def mosi_label(y_batch):
  res = copy.deepcopy(y_batch)
  res[y_batch >= 0] = 1
  res[y_batch < 0] = 0
  return res

# Sarcasm/Humor Training
def sarcasm_label(y_batch):
  res = copy.deepcopy(y_batch)
  res[y_batch == -1] = 0
  return res


# MSE loss
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, x, y):
        return (x - y).pow(2).mean()


# UML model
class UML(nn.Module):
    def __init__(self, xproj_in, yproj_in, shared_encoder, decoders, modality='x'):
        super().__init__()
        self.xproj_in = xproj_in
        self.yproj_in = yproj_in
        self.encoder = shared_encoder
        self.decoders = nn.ModuleList(decoders)
        self.modality = modality
        self.critic = MSE()
        print("Training using MUSE with modality: ", modality)
    
    def forward(self, x, y):
        # pool sequence dim of z
        loss_x = loss_y = torch.tensor(0.0)
        if x is not None:
            x = x.unsqueeze(1).float() if x.ndim == 2 else x
            x_proj = self.xproj_in(x)
            zx = self.encoder(x_proj)
            x_recon = self.decoders[0](zx)
            if x_recon.shape[1] == 1:
                loss_x = self.critic(x_recon[:, 0, :], x[:, 0, :])
            else:
                # next embedding prediction loss
                loss_x = self.critic(x_recon[:, :-1,:], x[:,1:,:])
        if y is not None:
            y = y.unsqueeze(1).float() if y.ndim == 2 else y
            y_proj = self.yproj_in(y)
            zy = self.encoder(y_proj)
            y_recon = self.decoders[1](zy)
            if y_recon.shape[1] == 1:
                loss_y = self.critic(y_recon[:, 0, :], y[:, 0, :])
            else:
                # next embedding prediction loss
                loss_y = self.critic(y_recon[:, :-1,:], y[:,1:,:])
        
        return {'loss_x': loss_x, 'loss_y': loss_y}

    def get_embedding(self, x, y):
        x = x.unsqueeze(1).float() if x.ndim == 2 else x
        y = y.unsqueeze(1).float() if y.ndim == 2 else y
        x = self.xproj_in(x)
        y = self.yproj_in(y)
        return self.encoder(x).mean(dim=1), self.encoder(y).mean(dim=1)


def evaluate(model, config, ds_name='mosi'):
    embds = {'train': {}, 'val': {}, 'test': {}}

    model.eval()
    if ds_name == 'mimic':
        for type in ['train', 'val', 'test']:
            embds[type]['x1'] = np.concatenate([model.get_embedding(data[0].float().cuda(), data[1].float().cuda())[0].detach().cpu().numpy() for data in config[type]])
            embds[type]['x2'] = np.concatenate([model.get_embedding(data[0].float().cuda(), data[1].float().cuda())[1].detach().cpu().numpy() for data in config[type]])
            embds[type]['labels'] = np.concatenate([data[2].detach().cpu().numpy() for data in config[type]])
    else:
        for type in ['train', 'val', 'test']:
            embds[type]['x1'] = np.concatenate([model.get_embedding(data[0][0].cuda(), data[0][2].cuda())[0].detach().cpu().numpy() for data in config[type]])
            embds[type]['x2'] = np.concatenate([model.get_embedding(data[0][0].cuda(), data[0][2].cuda())[1].detach().cpu().numpy() for data in config[type]])
            embds[type]['labels'] = np.concatenate([data[3].detach().cpu().numpy() for data in config[type]])

    for type in ['train', 'val', 'test']:
        if ds_name == 'mosi' or ds_name == 'mosei':
            embds[type]['labels'] = mosi_label(embds[type]['labels'])
        elif ds_name == 'sarcasm' or ds_name == 'humor':
            embds[type]['labels'] = sarcasm_label(embds[type]['labels'])
        else:
            raise NotImplementedError('Dataset not implemented yet')
        
        embds[type]['labels'] = np.asarray(embds[type]['labels']).reshape(-1).astype(int)

    # Train Logistic Classifier on X alone
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
    clf.fit(embds['train']['x1'], embds['train']['labels'])
    val_score_x = clf.score(embds['val']['x1'], embds['val']['labels'])
    score_x = clf.score(embds['test']['x1'], embds['test']['labels'])
    
    # Train Logistic Classifier on Y alone
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
    clf.fit(embds['train']['x2'], embds['train']['labels'])
    val_score_y = clf.score(embds['val']['x2'], embds['val']['labels'])
    score_y = clf.score(embds['test']['x2'], embds['test']['labels'])
    
    # Train Logistic Classifier on XY together
    train_embeds = np.concatenate([embds['train']['x1'], embds['train']['x2']], axis=1)
    val_embeds = np.concatenate([embds['val']['x1'], embds['val']['x2']], axis=1)
    test_embeds = np.concatenate([embds['test']['x1'], embds['test']['x2']], axis=1)
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
    clf.fit(train_embeds, embds['train']['labels'])
    score_xy = clf.score(test_embeds, embds['test']['labels'])
    val_score_xy = clf.score(val_embeds, embds['val']['labels'])
    return score_x, score_y, score_xy, val_score_x, val_score_y, val_score_xy


def train(model, train_mode, train_loader_1, train_loader_2, optimizer, modalities=[0,2], num_epoch=100, step_k=30, ds_name='mosi', eval_config = {}, augment=False, debug=False):
    model.train()
    for _iter in range(num_epoch):
        alpha_x = alpha_y = 1.0
        if _iter <= step_k and train_mode == 'xy':
            print(f"Training only on y, step: [{_iter}/{step_k}]; total steps: {num_epoch}")
            alpha_x = 0.0 # train only on y for warmup when training on unpaired x+y data

        for i_batch, (data_batch_1, data_batch_2) in enumerate(zip(train_loader_1, train_loader_2)):
            if ds_name != 'mimic':
                x1_batch = data_batch_1[0][modalities[0]].float().cuda()
                x2_batch = data_batch_2[0][modalities[1]].float().cuda()
            else:
                x1_batch = data_batch_1[0].float().cuda()
                x2_batch = data_batch_2[1].float().cuda()

            out_loss = model(x1_batch, x2_batch)
            loss_x, loss_y = out_loss['loss_x'], out_loss['loss_y']
            loss_x = torch.tensor(0.0) if train_mode == 'y' else loss_x
            loss_y = torch.tensor(0.0) if train_mode == 'x' else loss_y
            loss = alpha_x * loss_x + alpha_y * loss_y
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if eval_config and i_batch % eval_config['freq'] == 0:
                model.eval()
                score = evaluate(model, eval_config, ds_name)
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss_x: ', loss_x.item(), ' loss_y: ', loss_y.item(), ' loss: ', loss.item(), ' score_x: ', score[0], ' score_y: ', score[1], ' score_xy: ', score[2])
                if not debug: 
                    wandb.log({'loss_x': loss_x.item(), 'loss_y': loss_y.item(), 'loss': loss.item(),
                    'score_x': score[0], 'score_y': score[1], 'score_xy': score[2], 'val_score_x': score[3], 'val_score_y': score[4], 'val_score_xy': score[5]})

                model.train()

        if eval_config and _iter == num_epoch-1:
            print('Final evaluation...')
            model.eval()
            score = evaluate(model, eval_config, ds_name)
            model.train()

    return score
            

