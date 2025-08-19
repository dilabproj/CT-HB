import logging
import copy

from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from ECGDL.evaluate.evaluate import evaluate_model
from ECGDL.utils.utils import pretty_stream, save_model
from ECGDL.training.model_picker import ModelPicker

# Initiate Logger
logger = logging.getLogger(__name__)


def train_model(model: nn.Module,  # pylint: disable=too-many-locals, too-many-statements
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,  # type: ignore
                device: torch.device,
                writer: SummaryWriter,
                model_picker: ModelPicker,
                num_epochs: int = 100,
                loss_function: Optional[nn.Module] = None,
                lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,  # pylint: disable=protected-access
                class_weight: Optional[List[int]] = None,
                validation_loader: Optional[DataLoader] = None,
                checkpoint_root: Optional[str] = None,
                ensemble_logic: str = 'softmax_logit_then_avg') -> Tuple[List[Dict], int]:

    # Remember total instances trained for plotting
    total_steps = 0

    # Save Per Epoch Progress
    result = []

    # Setup class weight if given
    if class_weight is not None:
        class_weight = torch.tensor(class_weight, dtype=torch.float32).to(device)  # type: ignore  # pylint: disable=not-callable
        logger.info("Setting Class Weight to %s", class_weight)

    epochs = trange(1, num_epochs + 1, dynamic_ncols=True)
    for epoch in epochs:
        epochs.set_description(f'Training Epoch: {epoch}')

        # Set model to Training Mode
        model.train()

        train_loss = 0.0
        correct_cnt, total_batches, total_cnt = 0, 0, 0

        training_data = tqdm(train_loader, dynamic_ncols=True, leave=False)
        for data, target in training_data:
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
                batch_loss = F.cross_entropy(output_logits, target)
            else:
                batch_loss = loss_function(output_logits, target)
            train_loss += batch_loss.item()
            # Increment Correct Count and Total Count
            correct_cnt += prediction.eq(target.view_as(prediction)).sum().item()
            total_batches += 1
            total_cnt += train_loader.batch_size

            # Back Propagation the Loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            training_data.set_description(
                f'Train loss: {train_loss / total_batches:.4f}, Accuracy: {correct_cnt / total_cnt:.4f}')

            # Write Progress to Tensorboard
            total_steps += train_loader.batch_size
            writer.add_scalar(f'BATCH/Training Loss', train_loss / total_batches, total_steps)
            writer.add_scalar(f'BATCH/Training Accuracy', correct_cnt / total_cnt, total_steps)
        pretty_stream(training_data)

        # Log per epoch metric
        per_epoch_metric: Dict[str, Dict] = {"epoch": epoch, "train": {}, "valid": {}}
        writer.add_scalar(f'Epoch', epoch, total_steps)

        per_epoch_metric['train']['Loss'] = train_loss / total_batches
        logger.info("Training Loss: %s", per_epoch_metric['train']['Loss'])
        writer.add_scalar(f'Training/Loss', per_epoch_metric['train']['Loss'], total_steps)

        per_epoch_metric['train']['Accuracy'] = correct_cnt / total_cnt
        logger.info("Training Accuracy: %s", per_epoch_metric['train']['Accuracy'])
        writer.add_scalar(f'Training/Accuracy', per_epoch_metric['train']['Accuracy'], total_steps)

        if validation_loader is not None:
            epochs.set_description(f'Validating Epoch: {epoch}')
            per_epoch_metric['valid'], _ = evaluate_model(
                model, validation_loader, device, "Validation", total_steps, writer, loss_function)
            if lr_scheduler is not None:
                lr_scheduler.step(per_epoch_metric['valid']['Loss'])
        if per_epoch_metric['train']['Accuracy'] > 0.90:
            break

        result.append(per_epoch_metric)

        # Add model into model picker
        model_picker.add_model(copy.deepcopy(model.state_dict()), per_epoch_metric)

    # Save Final model
    if checkpoint_root is not None:
        save_model(num_epochs, checkpoint_root, model, optimizer, lr_scheduler)

    return result, total_steps
