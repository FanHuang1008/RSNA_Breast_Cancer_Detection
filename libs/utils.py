import numpy as np
import random
import itertools

import torch

import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def train_one_batch(batch, model, model_ema, scheduler, criterion, optimizer, device):
    optimizer.zero_grad()
    
    pngs, labels = batch
    pngs, labels = pngs.to(device), labels.to(device)
    
    preds = model(pngs)
    loss = criterion(preds, labels)
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    model_ema.update(model)
    
    return loss.cpu().detach().tolist()

def val_one_batch(batch, model, criterion, device):
    pngs, labels = batch
    pngs, labels = pngs.to(device), labels.to(device)
    
    preds = model(pngs)

    y_true = labels.cpu().detach().squeeze(1).tolist()
    y_pred = preds.cpu().detach().squeeze(1).tolist()
    
    loss = criterion(preds, labels)

    #  y_true must be binary when calculating roc_auc_score
    y_true = [1 if num > 0.5 else 0 for num in y_true]
    
    return loss.cpu().detach().tolist(), y_true, y_pred

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True, save=True):

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label'.format(accuracy, misclass))
    if save:
        plt.savefig("./output/confusion_matrix.png")
    else:
        plt.show()