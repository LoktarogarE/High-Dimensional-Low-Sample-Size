import torch
from sklearn.metrics import roc_auc_score
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=FutureWarning)


def masked_accuracy(preds, labels, mask):    
    acc = torch.eq(preds.argmax(1), labels).float()
    _mask = torch.FloatTensor(mask)
    _mask /= _mask.sum()
    acc *= _mask
    return acc.sum().item()

def auc_score(y_pred_softmax, y, mask):
    y_pred_softmax = y_pred_softmax.numpy()[mask]
    y = y.numpy()[mask]
    auc = roc_auc_score(y, y_pred_softmax, multi_class='ovr')
    return auc

y_pred = torch.load('y_pred.pt')
y = torch.load('y.pt')
test_mask = torch.load('test_mask.pt')


print('ACC: ', masked_accuracy(y_pred, y, test_mask))
print('AUC: ', auc_score(y_pred, y, test_mask))