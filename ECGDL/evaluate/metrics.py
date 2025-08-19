from typing import List, Tuple

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from imblearn.metrics import sensitivity_specificity_support


EvalReport = Tuple[List[int], List[int], List[int], List[int], List[int], float]


def multiclass_roc_auc_score(y_truth, y_pred, labels) -> List[float]:
    oh = OneHotEncoder(categories=[labels], sparse=False)
    oh.fit([[l] for l in labels])
    y_truth = oh.transform(y_truth.reshape(-1, 1))
    return [roc_auc_score(y_truth[:, i], y_pred[:, i]) for i, _cn in enumerate(oh.categories_[0])]


def sensitivity_specificity_support_with_avg(y_truth, y_pred, labels):
    return sensitivity_specificity_support(y_truth, y_pred, labels=labels), \
        sensitivity_specificity_support(y_truth, y_pred, labels=labels, average="macro"), \
        sensitivity_specificity_support(y_truth, y_pred, labels=labels, average="weighted")
