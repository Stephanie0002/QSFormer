import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def eval_hits_and_mrr(y_pred_pos, y_pred_neg, k_value=10):
    y_pred_neg = y_pred_neg.reshape(-1,y_pred_pos.shape[0]).T
    y_pred_pos = y_pred_pos.reshape(-1, 1)
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    #print(optimistic_rank,pessimistic_rank)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    hitsK_list = (ranking_list <= k_value).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)
    return hitsK_list.mean(),mrr_list.mean()



def get_link_prediction_metrics(y_pred_pos: torch.Tensor, y_pred_neg: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = torch.cat([y_pred_pos, y_pred_neg], dim=0)
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    hitk,mrr = eval_hits_and_mrr(y_pred_pos, y_pred_neg)

    return {'average_precision': average_precision, 'roc_auc': roc_auc, 'mrr': mrr.cpu().detach().numpy()}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
