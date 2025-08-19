import logging
import uuid

from typing import List, Tuple, Dict, Any, Generator
from heapq import heappush, heappop

import torch

from ECGDL.utils.utils import safe_dir

# Initiate Logger
logger = logging.getLogger(__name__)


class ModelPicker:
    def __init__(self, k_best: int = 3):
        # How many models to save per metric
        self.k_best = k_best
        # Metric we look at to pick the best models (min/max, keys to follow down metric dict)
        # Eg. ('min', 'valid', 'loss'), ('max', 'valid', 'Macro AVG', 'AUROC'), ('max', 'epoch')
        # TODO: Change metrics to dataclass and use individual k_best
        self.metrics: List[Tuple[str, ...]] = []
        # Save best models parameters (uuid -> model state_dict)
        self.save_models: Dict[str, Any] = {}
        # Save best models reference count (uuid -> model ref_cnt)
        self.save_models_refcnt: Dict[str, int] = {}
        # Map metric best model to save models (metric hash -> prioirty queue of uuid)
        self.best_models: Dict[str, List[str]] = {}

    @staticmethod
    def metric_hash(metric: Tuple[str, ...]) -> str:
        return '-'.join(metric)

    def add_metric(self, metric: Tuple[str, ...]):
        # Save the metrics
        self.metrics.append(metric)
        # Init the prioirty queue for best models
        self.best_models[self.metric_hash(metric)] = []

    def add_metrics(self, metrics: List[Tuple[str, ...]]):
        for metric in metrics:
            self.add_metric(metric)

    def add_model(self, model, model_metrics: Dict[str, Any]):
        # Get an unique id for the model
        model_uuid = f"{uuid.uuid4().hex}_Epoch_{model_metrics['epoch']}"
        assert model_uuid not in self.save_models
        # Record the total usage count of this model
        new_model_ref_cnt = 0
        for tar_metric in self.metrics:
            # Get the performance of the target metric
            performance = model_metrics
            for k in tar_metric[1:]:
                performance = performance.get(k, {})
            if performance == {}:
                # Performance not found, skip this metric and produce a warning
                logger.warning("Model metrics does not contain targeted metric: %s", self.metric_hash(tar_metric))
                continue
            # Heap pop will pop smallest item, so if our target is to save min, we reverse the sign of performance
            record_performance = -performance if tar_metric[0] == 'min' else performance  # type: ignore
            # Push the model into the heap
            heappush(self.best_models[self.metric_hash(tar_metric)], (record_performance, model_uuid))  # type: ignore
            # See if we need to pop any model
            if len(self.best_models[self.metric_hash(tar_metric)]) > self.k_best:
                _, worst_model_uuid = heappop(self.best_models[self.metric_hash(tar_metric)])
                # If we sucessfully performed better, increment ref_count and remove worst model
                if worst_model_uuid != model_uuid:
                    new_model_ref_cnt += 1
                    ref_cnt = self.save_models_refcnt[worst_model_uuid]
                    if ref_cnt == 1:
                        # Last reference is going to be removed, remove model from save_models
                        del self.save_models[worst_model_uuid]
                        del self.save_models_refcnt[worst_model_uuid]
                    else:
                        self.save_models_refcnt[worst_model_uuid] = ref_cnt - 1
            else:
                new_model_ref_cnt += 1

        # If any metric uses this model, save it into save models
        if new_model_ref_cnt > 0:
            self.save_models[model_uuid] = (model, model_metrics)
            self.save_models_refcnt[model_uuid] = new_model_ref_cnt

    def store_models(self, checkpoint_root: str):
        checkpoint_root = safe_dir(checkpoint_root)
        for model, model_uuid, reasons, reason_str in self.get_best_models():
            checkpoint_path = f'{checkpoint_root}/model_{model_uuid}_{reason_str}'
            logger.info('Save model at %s!', checkpoint_path)
            save_dict = {
                'model_uuid': model_uuid,
                'reasons': reasons,
                'reason_str': reason_str,
                'state_dict': model,
            }
            torch.save(save_dict, checkpoint_path)

    def get_best_model_by_metric(self, metric: Tuple[str, ...]) -> Generator[Tuple[Any, str, int], None, None]:
        # Rank is returned from 1 ~ k
        for rank, model_uuid in enumerate(self.best_models[self.metric_hash(metric)]):
            yield self.save_models[model_uuid][0], model_uuid, self.k_best - rank

    def get_best_models(self) -> Generator[Tuple[Any, str, List[Tuple[Tuple[str, ...], int]], str], None, None]:
        # Rank is returned from 1 ~ k
        for model_uuid, save_model in self.save_models.items():
            reasons = []
            for metric in self.metrics:
                for rank, (_perf, perf_model_uuid) in enumerate(self.best_models[self.metric_hash(metric)]):
                    if model_uuid == perf_model_uuid:
                        reasons.append((metric, self.k_best - rank))
            reason_str = ",".join([f"{self.metric_hash(reason[0])}_{reason[1]}" for reason in reasons]).replace(" ", "")
            yield save_model[0], model_uuid, reasons, reason_str
