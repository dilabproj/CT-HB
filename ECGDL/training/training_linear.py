import logging

from typing import Tuple, Any

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tqdm import tqdm

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier

from ECGDL.utils.utils import pretty_stream

# Initiate Logger
logger = logging.getLogger(__name__)


def train_model_linear(model: nn.Module,
                       train_loader: DataLoader,
                       device: torch.device,
                       dataset_type: str) -> Tuple[None, int, Any, Any]:

    # Remember total instances trained for plotting
    total_steps = 0

    X_train_feature, y_train = [], []

    # Set model to Training Mode
    model.eval()

    training_data = tqdm(train_loader, dynamic_ncols=True, leave=False)
    for data, target in training_data:
        if data.dim() == 3:
            bs, c, w = data.size()
            data = data.view(bs, 1, c, w)
        bs, n_crops, c, w = data.size()
        # Move data to device, model shall already be at device
        if dataset_type == "image_downstream":
            data, target = data.view(bs, 1, -1).to(device), target.to(device)
        else:
            data, target = data.view(-1, c, w).to(device), target.to(device)
        # Run batch data through model
        _, output_logits = model(data)
        output_logits = output_logits.view(bs, n_crops, -1).mean(1)

        X_train_feature.extend(output_logits.cpu().detach().numpy())
        y_train.extend(target.cpu().detach().numpy())

        total_steps += train_loader.batch_size
    pretty_stream(training_data)

    X_train_feature = np.array(X_train_feature)
    y_train = np.array(y_train)
    logger.info("Train LR...")
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train_feature)
    X_train = scaler.transform(X_train_feature)
    clf = LogisticRegression(random_state=0, max_iter=100000, solver='lbfgs', C=1.0, n_jobs=24)
    clf.fit(X_train, y_train)
    # clf = KNeighborsClassifier(n_neighbors=10)
    # clf.fit(X_train, y_train)
    logger.info("Logistic Regression feature eval")
    logger.info("Train Accuracy: %4f", clf.score(X_train, y_train))
    logger.info("Finish training LR...")
    return None, total_steps, clf, scaler
