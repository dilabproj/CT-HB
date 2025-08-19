import logging

from typing import List, Dict, Optional, Any, Type

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from ECGDL.utils.utils import pretty_stream, save_model

# Initiate Logger
logger = logging.getLogger(__name__)


def train_model_unsup_mts(model: nn.Module,
                          train_loader: DataLoader,
                          optimizer: torch.optim.Optimizer,  # type: ignore
                          device: torch.device,
                          # writer: SummaryWriter,
                          loss_function: Optional[nn.Module] = None,
                          num_epochs: int = 100,
                          lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,  # pylint: disable=protected-access
                          lr_scheduler_args: Optional[Dict[str, Any]] = None,
                          class_weight: Optional[List[int]] = None,
                          checkpoint_root: Optional[str] = None,
                          **kwargs):  # pylint: disable=unused-argument

    # Remember total instances trained for plotting
    total_steps = 0

    # Save Per Epoch Progress
    result = []

    # Setup Adjust learning rate
    scheduler = lr_scheduler(optimizer, **lr_scheduler_args) if lr_scheduler is not None else None  # type: ignore

    # Setup class weight if given
    if class_weight is not None:
        class_weight = torch.tensor(class_weight, dtype=torch.float32).to(device)  # type: ignore  # pylint: disable=not-callable
        logger.info("Setting Class Weight to %s", class_weight)

    epochs = trange(num_epochs, dynamic_ncols=True)
    for epoch in epochs:
        epochs.set_description(f'Training Epoch: {epoch}')

        # Set model to Training Mode
        model.train()

        train_loss = 0.0
        total_batches = 0

        training_data = tqdm(train_loader, dynamic_ncols=True, leave=False)
        for anchor, negative in training_data:
            if loss_function is not None:
                batch_loss = loss_function(anchor, model, negative, device)
            else:
                logger.warning("Should specify loss function!")
            # Get and Sum up Batch Loss
            train_loss += batch_loss.item()
            total_batches += 1

            # Back Propagation the Loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            training_data.set_description(f'Train loss: {train_loss / total_batches:.4f}')
            # Write Progress to Tensorboard
            total_steps += train_loader.batch_size
            writer.add_scalar(f'BATCH/Training Loss', train_loss / total_batches, total_steps)
        pretty_stream(training_data)

        # Log per epoch metric
        per_epoch_metric: Dict[str, Dict] = {"train": {}, "valid": {}}
        writer.add_scalar(f'Epoch', epoch, total_steps)

        per_epoch_metric['train']['Loss'] = train_loss / total_batches
        logger.info("Training Loss: %s", per_epoch_metric['train']['Loss'])
        writer.add_scalar(f'Training/Loss', per_epoch_metric['train']['Loss'], total_steps)

        result.append(per_epoch_metric)

        if epoch % 100 == 0 and checkpoint_root is not None:
            save_model(epoch, checkpoint_root, model, optimizer, scheduler)

    # Save Final model
    if checkpoint_root is not None:
        save_model(num_epochs, checkpoint_root, model, optimizer, scheduler)

    return result, total_steps
