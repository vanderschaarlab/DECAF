import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return precision_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return recall_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


def aucroc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return roc_auc_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
