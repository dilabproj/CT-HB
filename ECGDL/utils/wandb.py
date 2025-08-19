import logging

from typing import Optional, Dict, List, Any

import plotly.figure_factory as ff
import pandas as pd
import numpy as np

from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

import wandb
from ECGDL.evaluate.metrics import multiclass_roc_auc_score, sensitivity_specificity_support_with_avg

# Initiate Logger
logger = logging.getLogger(__name__)


def log_plotly_ff(wandb_name: str, wandb_step: int, fig: List[Any], title: Optional[str] = None):
    wandb_key = f"{title}: {wandb_name}" if title is not None else wandb_name
    # h = plotly.offline.plot(fig, output_type="div")
    # wandb.log({wandb_key: wandb.Html(h, inject=False)}, step=wandb_step)
    wandb.log({wandb_key: fig}, step=wandb_step)


def log_df(wandb_name: str,
           wandb_step: int,
           writer: SummaryWriter,
           df: pd.DataFrame,
           title: Optional[str] = None):

    # Log table to WandB
    fig = ff.create_table(df, index=True)
    log_plotly_ff(wandb_name, wandb_step, fig, title)

    # Log to Logger
    tb = PrettyTable()
    tb.field_names = ["Category"] + list(df.columns)
    tb.align = "r"
    for row in df.itertuples():
        tb.add_row(row)
        # Log metrics to WandB
        index_name = row[0]
        for metric_name, metric_value in zip(df.columns, row[1:]):
            writer.add_scalar(f'{wandb_name}/{index_name}_{metric_name}', metric_value, wandb_step)

    if title is not None:
        logger.info("%s\n%s", title, tb.get_string())
    else:
        logger.info("\n%s", tb.get_string())


def log_confusion_matrix(wandb_name: str,
                         wandb_step: int,
                         writer: SummaryWriter,  # pylint: disable=unused-argument
                         labels: List[str],
                         cf_matrix: np.array,
                         title: Optional[str] = None):

    cf_matrix_norm = np.around(
        cf_matrix.astype('float32') * 100 / cf_matrix.sum(axis=1)[:, np.newaxis], decimals=3)

    # Log confusion map to WandB
    hover = []
    for t, n_t in zip(cf_matrix, cf_matrix_norm):
        h = []
        for p, n_p in zip(t, n_t):
            h.append(f"{p} ({n_p} %)")
        hover.append(h)

    fig = ff.create_annotated_heatmap(np.flip(cf_matrix_norm, 0), x=labels, y=list(reversed(labels)),
                                      annotation_text=np.flip(cf_matrix, 0), text=list(reversed(hover)),
                                      zmin=0, zmax=100, zauto=False,
                                      colorscale="Greens", reversescale=False, showscale=True,
                                      hoverinfo='x+y+text', font_colors=['black', 'white'])
    # fig.layout.title = title
    fig.layout.xaxis.title = "Prediction"
    fig.layout.xaxis.side = "bottom"
    fig.layout.xaxis.automargin = True
    fig.layout.yaxis.title = "Truth"
    fig.layout.yaxis.automargin = True

    log_plotly_ff(wandb_name, wandb_step, fig, title)

    # Log to Logger
    tb = PrettyTable()
    tb.field_names = ["truth \\ pred"] + labels
    tb.align = "r"
    for index_name, row in zip(labels, cf_matrix):
        row = list(row)
        tb.add_row([index_name] + row)
        # Log metrics to WandB
        # for l, cnt in zip(labels, row):
        #     writer.add_scalar(f'{wandb_name}/truth_{index_name}_pred_{l}', cnt, wandb_step)

    if title is not None:
        logger.info("%s\n%s", title, tb.get_string())
    else:
        logger.info("\n%s", tb.get_string())


def round_dict(d: Dict[Any, float], p: int = 5) -> Dict[Any, float]:
    return {k: round(v, p) for k, v in d.items()}


def log_evaluation_report(wandb_name: str,
                          wandb_step: int,
                          writer: SummaryWriter,
                          cid2name: Dict[int, str],
                          loss: float,
                          y_truth: np.array,
                          y_pred: np.array,
                          y_pred_prob: np.ndarray,
                          unique_y: List[int]):

    report_dict = classification_report(y_true=y_truth, y_pred=y_pred, labels=unique_y, output_dict=True)
    sss, macro_sss, weighted_sss = sensitivity_specificity_support_with_avg(y_truth, y_pred, unique_y)
    report_auroc = multiclass_roc_auc_score(y_truth, y_pred_prob, unique_y)
    mcc = matthews_corrcoef(y_truth, y_pred)

    logger.info("%s Loss: %s", wandb_name, loss)
    writer.add_scalar(f'{wandb_name}/Loss', loss, wandb_step)
    logger.info("%s Accuracy: %s", wandb_name, report_dict["accuracy"])
    writer.add_scalar(f'{wandb_name}/Accuracy', report_dict["accuracy"], wandb_step)

    support_list: List[float] = []
    df_dict: Dict = {}
    for cid in cid2name:
        df_dict[cid2name[cid]] = report_dict[str(cid)]
        df_dict[cid2name[cid]]["Sensitivity"] = sss[0][cid]
        df_dict[cid2name[cid]]["Specificity"] = sss[1][cid]
        df_dict[cid2name[cid]]["AUROC"] = report_auroc[cid]
        df_dict[cid2name[cid]] = round_dict(df_dict[cid2name[cid]])
        support_list.append(report_dict[str(cid)]["support"])

    df_dict["Macro AVG"] = report_dict["macro avg"]
    df_dict["Macro AVG"]["Sensitivity"], df_dict["Macro AVG"]["Specificity"], _ = macro_sss
    df_dict["Macro AVG"]["AUROC"] = np.mean(report_auroc)
    df_dict["Macro AVG"] = round_dict(df_dict["Macro AVG"])

    df_dict["Weighted AVG"] = report_dict["weighted avg"]
    df_dict["Weighted AVG"]["Sensitivity"], df_dict["Weighted AVG"]["Specificity"], _ = weighted_sss
    df_dict["Weighted AVG"]["AUROC"] = np.dot(report_auroc, support_list) / np.sum(support_list)
    df_dict["Weighted AVG"] = round_dict(df_dict["Weighted AVG"])
    df_dict["MCC"] = round(mcc, 5)

    report_df = pd.DataFrame(data=df_dict).T
    log_df(wandb_name, wandb_step, writer, report_df, "Classification Report")

    log_confusion_matrix(wandb_name, wandb_step, writer, list(cid2name.values()),
                         confusion_matrix(y_truth, y_pred, labels=unique_y), "Confusion Matrix")

    # TODO: Plot AUC Curve, PR Curve
    # TODO: Log 90% Sensitivity Curve, Sensitivity==Specificity Point

    ret = df_dict
    ret["Accuracy"] = report_dict["accuracy"]
    ret["Loss"] = loss
    ret["MCC"] = mcc

    return ret
