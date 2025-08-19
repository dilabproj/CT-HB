import logging

from typing import Callable, List, Dict, Tuple, Optional, Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from ECGDL.utils.utils import pretty_stream, save_model

# Initiate Logger
logger = logging.getLogger(__name__)


def train_model_unsup(model: nn.Module,  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
                      train_loader: DataLoader,
                      optimizer: torch.optim.Optimizer,  # type: ignore
                      device: torch.device,
                      writer: SummaryWriter,
                      pretexts: Optional[Tuple[Callable, ...]] = None,
                      num_epochs: int = 100,
                      loss_function: Optional[nn.Module] = None,
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
    for epoch in epochs:  # pylint: disable=too-many-nested-blocks
        epochs.set_description(f'Training Epoch: {epoch}')

        # Set model to Training Mode
        model.train()

        train_loss = 0.0
        total_batches, total_cnt = 0, 0
        correct_cnt_list = [0 for i in range(len(pretexts) + 1)] if pretexts is not None else []

        training_data = tqdm(train_loader, dynamic_ncols=True, leave=False)
        for data, target in training_data:
            if data.dim() == 3:
                bs, c, w = data.size()
                data = data.view(bs, 1, c, w)
            bs, _, c, w = data.size()
            # Move data to device, model shall already be at device
            data, target = data.view(-1, c, w).to(device), target.to(device)
            if pretexts is None:
                # Get and Sum up Batch Loss
                if loss_function is not None:
                    output_logits = model(data)
                    batch_loss = loss_function(output_logits, target.squeeze())
                else:
                    output_logits, kld = model(data)
                    batch_loss = F.binary_cross_entropy(
                        output_logits, target.squeeze()) + min(1, total_steps/2500) * kld
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
            else:
                output_prob_list, output_logits_list = model(data)
                for idx, (output_prob, output_logits) in enumerate(zip(output_prob_list, output_logits_list)):
                    prediction = output_prob.max(1, keepdim=True)[1]
                    # Get and Sum up Batch Loss
                    if idx == 0:
                        if loss_function is None:
                            batch_loss = F.cross_entropy(
                                output_logits, target[:, idx], weight=class_weight)  # type: ignore
                        else:
                            batch_loss = loss_function(output_logits, target[:, idx])
                    else:
                        if loss_function is None:
                            batch_loss += F.cross_entropy(
                                output_logits, target[:, idx], weight=class_weight)  # type: ignore
                        else:
                            batch_loss += loss_function(output_logits, target[:, idx])
                    train_loss += batch_loss.item()
                    # Increment Correct Count and Total Count
                    correct_cnt_list[idx] += prediction.eq(target[:, idx].view_as(prediction)).sum().item()
                    total_batches += 1
                total_cnt += train_loader.batch_size

                # Back Propagation the Loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                description_name = f'Train loss: {train_loss / total_batches:.4f}'
                if pretexts is not None:
                    for i in range(len(pretexts) + 1):
                        description_name += f', Accuracy: {correct_cnt_list[i] / total_cnt:.4f}'
                    training_data.set_description(description_name)

                # Write Progress to Tensorboard
                total_steps += train_loader.batch_size
                writer.add_scalar(f'BATCH/Training Loss', train_loss / total_batches, total_steps)
                if pretexts is not None:
                    for i, pretext in enumerate(pretexts):
                        name = pretext.__class__.__name__
                        writer.add_scalar(
                            f'BATCH/Training Accuracy/{name}', correct_cnt_list[i] / total_cnt, total_steps)
                    writer.add_scalar(
                        f'BATCH/Training Accuracy/original', correct_cnt_list[i + 1] / total_cnt, total_steps)
        pretty_stream(training_data)

        # Log per epoch metric
        per_epoch_metric: Dict[str, Dict] = {"train": {}, "valid": {}}
        writer.add_scalar(f'Epoch', epoch, total_steps)

        per_epoch_metric['train']['Loss'] = train_loss / total_batches
        logger.info("Training Loss: %s", per_epoch_metric['train']['Loss'])
        writer.add_scalar(f'Training/Loss', per_epoch_metric['train']['Loss'], total_steps)

        if pretexts is not None:
            for i, pretext in enumerate(pretexts):
                name = pretext.__class__.__name__
                per_epoch_metric['train'][f'Accuracy_{name}'] = correct_cnt_list[i] / total_cnt
                logger.info("Training Accuracy_%s: %s", name, per_epoch_metric['train'][f'Accuracy_{name}'])
                writer.add_scalar(
                    f'Training/Accuracy_{name}', per_epoch_metric['train'][f'Accuracy_{name}'], total_steps)
            per_epoch_metric['train'][f'Accuracy_original'] = correct_cnt_list[i + 1] / total_cnt
            logger.info("Training Accuracy_original: %s", per_epoch_metric['train'][f'Accuracy_original'])
            writer.add_scalar(
                f'Training/Accuracy_original', per_epoch_metric['train'][f'Accuracy_original'], total_steps)

        result.append(per_epoch_metric)

        if epoch % 100 == 0 and checkpoint_root is not None:
            save_model(epoch, checkpoint_root, model, optimizer, scheduler)

    # Save Final model
    if checkpoint_root is not None:
        save_model(num_epochs, checkpoint_root, model, optimizer, scheduler)

    return result, total_steps
