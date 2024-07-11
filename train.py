import os
import numpy as np
import pandas as pd

import yaml
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from libs.efficientnet import EffNetModel, ModelEma
from libs.datasets import train_transform, val_transform, CancerDataset, CustomBatchSampler
from libs.utils import set_seed, train_one_batch, val_one_batch 

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # set hyper-parameters and seed
    if os.path.isfile(args.config):
        with open(args.config, encoding='utf-8') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    epochs, batch_size, num_workers, lr = config["epochs"], config['batch-size'], config['num-workers'], config['lr']
    set_seed(args.seed)

    # determine starting fold
    if args.resume:
        start_fold = int(args.resume.split("/")[-1][1])
    else:
        start_fold = 0
    
    for fold in range(start_fold, 4):
        # create dataset and dataloader
        val_df = pd.read_pickle(f"./df/val_f{fold}.pkl")
        train_df = pd.read_pickle(f"./df/train_f{fold}.pkl")
        train_ex = pd.read_pickle(f"./df/train_ex.pkl")
        train_df = pd.concat([train_df, train_ex],axis=0).reset_index(drop=True)
        
        if args.label_smooth:
            pos_label = 0.8
            train_df['cancer'] = np.where(train_df['cancer']==1, pos_label, 0.0)
            val_df['cancer'] = np.where(val_df['cancer']==1, pos_label, 0.0)
        else:
            pos_label = 1
 
        train_dataset = CancerDataset(train_df, train_transform)
        val_dataset = CancerDataset(val_df, val_transform)
        
        pos_indices = train_df.index[train_df['cancer']==pos_label].tolist()
        batchsampler = CustomBatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size, False, pos_indices)
        
        train_dataloader = DataLoader(train_dataset, batch_sampler=batchsampler, num_workers=num_workers, prefetch_factor=2)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        # create model, criterion, optimizer, scheduler
        model = EffNetModel()    
        model.to(DEVICE)
        model_ema = ModelEma(model)
        print("Using model EMA ...")

        criterion = nn.BCELoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

        # resume from checkpoint
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=torch.device(DEVICE))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            del checkpoint
            print(f"Resuming from pre-trained weights: {args.resume.split('/')[-1]}")
        else:
            start_epoch = 1
            if os.path.isfile(f"./ckpt/f{fold}/log.txt"):
                os.remove(f"./ckpt/f{fold}/log.txt")
            
        # training and vlidation loops
        train_losses, val_losses, val_roc_aucs = [], [], []
        for epoch in range(start_epoch, epochs + 1):
            print(f"Fold {fold} Epoch {epoch}\n", "-" * 50)

            train_loss = 0
            model.train()
            for batch in tqdm(train_dataloader):
                loss = train_one_batch(batch, model, model_ema, scheduler, criterion, optimizer, DEVICE)
                with torch.no_grad():
                    train_loss += loss
                # update the learning rate at every batch
                scheduler.step()
            print(f"Train Loss: {train_loss / len(train_dataloader):.4f}")
            
            val_loss, val_targets, val_outputs = 0, [], []
            model.eval()
            for batch in tqdm(val_dataloader):
                with torch.no_grad():
                    loss, y_true, y_pred = val_one_batch(batch, model, criterion, DEVICE)
                    val_loss += loss
                    val_targets.extend(y_true)
                    val_outputs.extend(y_pred)

            val_roc_auc = roc_auc_score(val_targets, val_outputs)
            print(f"Val Loss: {val_loss / len(val_dataloader):.4f}, ROC_AUC: {val_roc_auc:.4f}")
            print(" ")
    
            # save checkpoint, model EMA, scheduler, optimizer
            save_states = {
                'epoch': epoch + 1, 
                'state_dict': model.state_dict(),
                'state_dict_ema': model_ema.module.state_dict(),
                'scheduler': scheduler.state_dict(), 
                'optimizer': optimizer.state_dict()
            }
            torch.save(save_states, f"./ckpt/f{fold}/f{fold}_ep{epoch}_roc_{val_roc_auc:.3f}_loss_{val_loss/len(val_dataloader):.3f}.pth")
    
            train_losses.append(train_loss / len(train_dataloader))
            val_losses.append(val_loss / len(val_dataloader))
            val_roc_aucs.append(val_roc_auc)
    
            # write results to log file
            with open(f"./ckpt/f{fold}/log.txt", "a") as log:   
                log.write(f"Fold {fold} Epoch {epoch}\n")
                log.write(f"Train Loss: {train_loss / len(train_dataloader):.4f}\n")
                log.write(f"Val Loss: {val_loss / len(val_dataloader):.4f}, ROC_AUC: {val_roc_auc:.4f}\n")
                log.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smooth", type=bool, default=False)
    parser.add_argument("--resume", type=str, metavar="PATH", default="")
    args = parser.parse_args()
    main(args)