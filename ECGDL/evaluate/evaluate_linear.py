import logging

from typing import Dict

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from ECGDL.utils.wandb import log_evaluation_report
from ECGDL.utils.utils import pretty_stream

# Initiate Logger
logger = logging.getLogger(__name__)


def evaluate_model_linear(model: nn.Module,
                          test_loader: DataLoader,
                          device: torch.device,
                          wandb_name: str,
                          wandb_step: int,
                          writer: SummaryWriter,
                          clf: LogisticRegression,
                          scaler: preprocessing) -> Dict:

    # Set model to Eval Mode (For Correct Dropout and BatchNorm Behavior)
    model.eval()

    test_loss = 0.0

    # Save Predictions, Predicted probability and Truth Data for Evaluation Report
    y_pred, y_pred_prob, y_truth = [], [], []

    X_test_feature, y_test = [], []

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
            _, output_logits = model(data)
            output_logits = output_logits.view(bs, n_crops, -1).mean(1)

            X_test_feature.extend(output_logits.cpu().detach().numpy())
            y_test.extend(target.cpu().detach().numpy())

        pretty_stream(testing_data)

    # Get unique y values
    unique_y = np.unique(y_test).tolist()

    X_test_feature = np.array(X_test_feature)
    y_truth = np.array(y_test)

    logger.info("Test LR...")
    X_test = scaler.transform(X_test_feature)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
    logger.info("Logistic Regression feature eval")

    # Print Evaluation Metrics and log to wandb
    report = log_evaluation_report(wandb_name, wandb_step, writer, test_loader.dataset.cid2name,  # type: ignore
                                   test_loss, y_truth, y_pred, y_pred_prob, unique_y)
    MCC_score = metrics.matthews_corrcoef(y_truth, y_pred)
    logger.info('MCC score: %4f', MCC_score)

    # TODO: Add method to save best metric

    return report
