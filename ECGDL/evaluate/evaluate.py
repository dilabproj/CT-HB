import logging

from typing import Dict, Tuple, Any
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from ECGDL.utils.wandb import log_evaluation_report
from ECGDL.utils.utils import pretty_stream

# Initiate Logger
logger = logging.getLogger(__name__)


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device,
                   wandb_name: str,
                   wandb_step: int,
                   writer: SummaryWriter,
                   loss_function: Optional[nn.Module] = None,
                   ensemble_logic: str = 'softmax_logit_then_avg') -> Tuple[Dict, Tuple[Any, Any, Any]]:

    # Set model to Eval Mode (For Correct Dropout and BatchNorm Behavior)
    model.eval()

    test_loss = 0.0
    correct_cnt = 0

    # Save Predictions, Predicted probability and Truth Data for Evaluation Report
    y_pred, y_pred_prob, y_truth = [], [], []

    with torch.no_grad():
        testing_data = tqdm(test_loader, dynamic_ncols=True, leave=False)
        for data, target in testing_data:
            if data.dim() == 3:
                bs, c, w = data.size()
                data = data.view(bs, 1, c, w)
            bs, n_crops, c, w = data.size()
            # Move data to device, model shall already be at device
            data, target = data.view(-1, c, w).to(device), target.to(device)
            # Run batch data through model
            output_prob, output_logits = model(data)
            output_logits = output_logits.view(bs, n_crops, -1).mean(1)
            if ensemble_logic == "avg_logit_then_softmax":
                output_prob = F.softmax(output_logits, dim=1)
            elif ensemble_logic == "softmax_logit_then_avg":
                output_prob = output_prob.view(bs, n_crops, -1).mean(1)
            else:
                raise ValueError("Ensemble Logic not Found!")
            prediction = output_prob.max(1, keepdim=True)[1]
            # Get and Sum up Batch Loss
            if loss_function is None:
                batch_loss = F.cross_entropy(output_logits, target, reduction='sum')
            else:
                batch_loss = loss_function(output_logits, target, reduction='sum')
            test_loss += batch_loss.item()
            # Increment Correct Count and Total Count
            correct_cnt += prediction.eq(target.view_as(prediction)).sum().item()
            # Append Prediction Results
            y_truth.append(target.cpu())
            y_pred_prob.append(output_prob.cpu())
            y_pred.append(prediction.reshape(-1).cpu())
        pretty_stream(testing_data)

    # Calculate average evaluation loss
    test_loss = test_loss / len(test_loader.dataset)

    # Merge results from each batch
    y_truth = np.concatenate(y_truth)
    y_pred = np.concatenate(y_pred)
    y_pred_prob = np.concatenate(y_pred_prob)

    # Get unique y values
    unique_y = np.unique(np.concatenate([y_truth, y_pred])).tolist()

    # Print Evaluation Metrics and log to wandb
    report = log_evaluation_report(wandb_name, wandb_step, writer, test_loader.dataset.cid2name,  # type: ignore
                                   test_loss, y_truth, y_pred, y_pred_prob, unique_y)

    # Return detail predition results for further analysis
    prediction_details = (y_truth, y_pred, y_pred_prob)

    # TODO: Add method to save best metric

    return report, prediction_details
