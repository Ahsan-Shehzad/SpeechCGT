from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Batch

from ..constants import ACOUSTIC_FEATURE_INDICES, TRANSCRIPT_FEATURE_INDICES
from ..data.preprocessing import preprocess_sample
from ..data.schemas import PreprocessedSample, SpeechSample, SplitManifest
from ..features.extractor import SpeechFeatureExtractor
from ..features.normalizer import FeatureNormalizer, FeatureStats
from ..graph.builder import NeurolinguisticGraphBuilder
from ..models.framework import SpeechCGTFramework
from ..utils.io import ensure_dir, save_json, save_torch
from ..utils.logging import get_logger
from ..utils.reproducibility import count_parameters, seed_everything
from .augment import add_gaussian_feature_noise, apply_modality_dropout
from .calibrator import TemperatureScaler
from .losses import (
    compute_classification_loss,
    compute_consistency_loss,
    compute_gate_sparsity_loss,
    compute_total_loss,
)
from .metrics import compute_classification_metrics


def _build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def schedule(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

    return LambdaLR(optimizer, lr_lambda=schedule)


def _preprocess_batch(
    samples: list[SpeechSample],
    config: dict[str, Any],
) -> list[PreprocessedSample]:
    return [
        preprocess_sample(
            sample,
            target_sample_rate=config["audio"]["target_sample_rate"],
            target_lufs=config["audio"]["target_lufs"],
            transcript_config=config["transcript"],
            audio_config=config["audio"],
        )
        for sample in samples
    ]


def _extract_raw_features(
    preprocessed_samples: list[PreprocessedSample],
    feature_extractor: SpeechFeatureExtractor,
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    features = []
    masks = []
    for sample in preprocessed_samples:
        x_raw, mask = feature_extractor.extract(sample)
        features.append(x_raw)
        masks.append(mask)
    return torch.stack(features, dim=0), torch.stack(masks, dim=0)


def _prepare_model_batch(
    samples: list[SpeechSample],
    feature_extractor: SpeechFeatureExtractor,
    normalizer: FeatureNormalizer,
    graph_builder: NeurolinguisticGraphBuilder,
    feature_stats: FeatureStats,
    config: dict[str, Any],
    device: torch.device,
    training: bool,
) -> tuple[Batch, torch.FloatTensor, torch.BoolTensor, torch.FloatTensor]:
    preprocessed = _preprocess_batch(samples, config)
    x_raw, feature_mask = _extract_raw_features(preprocessed, feature_extractor)
    x_hat, feature_mask = normalizer.transform_features(x_raw, feature_mask, feature_stats)
    if training:
        x_hat = apply_modality_dropout(
            x_hat,
            config["augmentation"]["modality_dropout_p"],
            ACOUSTIC_FEATURE_INDICES,
            TRANSCRIPT_FEATURE_INDICES,
        )
        x_hat = add_gaussian_feature_noise(x_hat, config["augmentation"]["gaussian_noise_std"])
    data_list = []
    labels = []
    for index, sample in enumerate(samples):
        data_list.append(graph_builder.build_subject_graph(x_hat[index], feature_mask[index], sample.label))
        labels.append(float(sample.label) if sample.label is not None else 0.0)
    graph_batch = Batch.from_data_list(data_list).to(device)
    return graph_batch, x_hat.to(device), feature_mask.to(device), torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(-1)


def _fit_stats_from_loader(
    loader,
    feature_extractor: SpeechFeatureExtractor,
    normalizer: FeatureNormalizer,
    config: dict[str, Any],
    source_corpus: str,
    fold_id: int,
) -> FeatureStats:
    feature_batches = []
    mask_batches = []
    for samples in loader:
        preprocessed = _preprocess_batch(samples, config)
        x_raw, feature_mask = _extract_raw_features(preprocessed, feature_extractor)
        feature_batches.append(x_raw)
        mask_batches.append(feature_mask)
    all_features = torch.cat(feature_batches, dim=0)
    all_masks = torch.cat(mask_batches, dim=0)
    return normalizer.fit_feature_stats(all_features, all_masks, source_corpus, fold_id)


def _compute_pos_weight(loader) -> torch.FloatTensor:
    positives = 0
    negatives = 0
    for samples in loader:
        for sample in samples:
            if sample.label == 1:
                positives += 1
            elif sample.label == 0:
                negatives += 1
    if positives == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([negatives / max(positives, 1)], dtype=torch.float32)


def train_one_epoch(
    model: SpeechCGTFramework,
    loader,
    feature_extractor: SpeechFeatureExtractor,
    normalizer: FeatureNormalizer,
    graph_builder: NeurolinguisticGraphBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    config: dict[str, Any],
    device: torch.device,
    feature_stats: FeatureStats,
    pos_weight: torch.FloatTensor,
) -> dict[str, float]:
    model.train()
    all_logits = []
    all_labels = []
    running_loss = 0.0
    batch_count = 0
    for samples in loader:
        graph_batch, x_hat, feature_mask, labels = _prepare_model_batch(
            samples,
            feature_extractor,
            normalizer,
            graph_builder,
            feature_stats,
            config,
            device,
            training=True,
        )
        output, aux = model(graph_batch, x_hat, feature_mask, return_aux=True)
        cls_loss = compute_classification_loss(output.ci_logit, labels, pos_weight.to(device))
        consistency_loss = compute_consistency_loss(
            aux["u_function"],
            aux["propagated_function"],
            aux["u_domain"],
            aux["propagated_domain"],
        )
        gate_loss = compute_gate_sparsity_loss(aux["gate_fc"], aux["gate_cd"])
        loss_dict = compute_total_loss(
            cls_loss,
            consistency_loss,
            gate_loss,
            config["losses"]["consistency_weight"],
            config["losses"]["gate_sparsity_weight"],
        )
        optimizer.zero_grad()
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += float(loss_dict["total"].detach().cpu().item())
        batch_count += 1
        all_logits.append(output.ci_prob.detach())
        all_labels.append(labels.detach())
    probabilities = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_classification_metrics(probabilities, labels, config["calibration"]["threshold"])
    metrics["loss"] = running_loss / max(batch_count, 1)
    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: SpeechCGTFramework,
    loader,
    feature_extractor: SpeechFeatureExtractor,
    normalizer: FeatureNormalizer,
    graph_builder: NeurolinguisticGraphBuilder,
    config: dict[str, Any],
    device: torch.device,
    feature_stats: FeatureStats,
) -> tuple[dict[str, float], torch.FloatTensor, torch.FloatTensor]:
    model.eval()
    logits = []
    probabilities = []
    labels_list = []
    for samples in loader:
        graph_batch, x_hat, feature_mask, labels = _prepare_model_batch(
            samples,
            feature_extractor,
            normalizer,
            graph_builder,
            feature_stats,
            config,
            device,
            training=False,
        )
        output = model(graph_batch, x_hat, feature_mask)
        logits.append(output.ci_logit.detach().cpu())
        probabilities.append(output.ci_prob.detach().cpu())
        labels_list.append(labels.detach().cpu())
    all_logits = torch.cat(logits, dim=0)
    all_probabilities = torch.cat(probabilities, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    metrics = compute_classification_metrics(all_probabilities, all_labels, config["calibration"]["threshold"])
    return metrics, all_logits, all_labels


@torch.no_grad()
def evaluate_split(
    model: SpeechCGTFramework,
    loader,
    feature_extractor: SpeechFeatureExtractor,
    normalizer: FeatureNormalizer,
    graph_builder: NeurolinguisticGraphBuilder,
    threshold: float,
    temperature_scaler: Optional[TemperatureScaler],
    device: torch.device,
    config: dict[str, Any],
    feature_stats: FeatureStats,
) -> dict[str, float]:
    model.eval()
    probabilities = []
    labels_list = []
    for samples in loader:
        graph_batch, x_hat, feature_mask, labels = _prepare_model_batch(
            samples,
            feature_extractor,
            normalizer,
            graph_builder,
            feature_stats,
            config,
            device,
            training=False,
        )
        output = model(graph_batch, x_hat, feature_mask)
        if temperature_scaler is not None:
            probs = temperature_scaler.transform(output.ci_logit.detach().cpu())
        else:
            probs = output.ci_prob.detach().cpu()
        probabilities.append(probs)
        labels_list.append(labels.detach().cpu())
    all_probabilities = torch.cat(probabilities, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return compute_classification_metrics(all_probabilities, all_labels, threshold)


class SpeechCGTTrainer:
    def __init__(
        self,
        model: SpeechCGTFramework,
        feature_extractor: SpeechFeatureExtractor,
        normalizer: FeatureNormalizer,
        graph_builder: NeurolinguisticGraphBuilder,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.normalizer = normalizer
        self.graph_builder = graph_builder
        self.config = config
        self.logger = get_logger("speechcgt.trainer")
        device_name = config["experiment"].get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.model.to(self.device)

    def fit(
        self,
        train_loader,
        val_loader,
        test_loader,
        external_loader,
        split_manifest: SplitManifest,
    ) -> dict[str, Any]:
        seed_everything(int(self.config["experiment"]["seed"]))
        output_dir = ensure_dir(Path(self.config["experiment"]["output_dir"]) / f"fold_{split_manifest.fold_id}")
        feature_stats = _fit_stats_from_loader(
            train_loader,
            self.feature_extractor,
            self.normalizer,
            self.config,
            split_manifest.source_corpus,
            split_manifest.fold_id,
        )
        pos_weight = _compute_pos_weight(train_loader)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["optimization"]["lr"],
            weight_decay=self.config["optimization"]["weight_decay"],
        )
        total_steps = int(self.config["training"]["epochs"]) * max(len(train_loader), 1)
        warmup_steps = int(self.config["optimization"]["warmup_epochs"]) * max(len(train_loader), 1)
        scheduler = _build_scheduler(optimizer, total_steps, warmup_steps)
        best_state = None
        best_f1 = float("-inf")
        best_accuracy = float("-inf")
        history = []
        for epoch in range(int(self.config["training"]["epochs"])):
            train_metrics = train_one_epoch(
                self.model,
                train_loader,
                self.feature_extractor,
                self.normalizer,
                self.graph_builder,
                optimizer,
                scheduler,
                self.config,
                self.device,
                feature_stats,
                pos_weight,
            )
            val_metrics, _, _ = validate_one_epoch(
                self.model,
                val_loader,
                self.feature_extractor,
                self.normalizer,
                self.graph_builder,
                self.config,
                self.device,
                feature_stats,
            )
            history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            if (val_metrics["f1"] > best_f1) or (
                val_metrics["f1"] == best_f1 and val_metrics["accuracy"] > best_accuracy
            ):
                best_f1 = val_metrics["f1"]
                best_accuracy = val_metrics["accuracy"]
                best_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint.")
        self.model.load_state_dict(best_state)
        val_metrics, val_logits, val_labels = validate_one_epoch(
            self.model,
            val_loader,
            self.feature_extractor,
            self.normalizer,
            self.graph_builder,
            self.config,
            self.device,
            feature_stats,
        )
        scaler = TemperatureScaler(self.config) if self.config["calibration"]["enabled"] else None
        temperature = None
        if scaler is not None:
            temperature = scaler.fit(val_logits.to(self.device), val_labels.to(self.device))
        test_metrics = evaluate_split(
            self.model,
            test_loader,
            self.feature_extractor,
            self.normalizer,
            self.graph_builder,
            self.config["calibration"]["threshold"],
            scaler,
            self.device,
            self.config,
            feature_stats,
        )
        external_metrics = evaluate_split(
            self.model,
            external_loader,
            self.feature_extractor,
            self.normalizer,
            self.graph_builder,
            self.config["calibration"]["threshold"],
            scaler,
            self.device,
            self.config,
            feature_stats,
        )
        checkpoint_payload = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "graph_schema": self.graph_builder.graph_schema.to_dict(),
            "feature_stats": feature_stats.to_dict(),
            "temperature": temperature,
            "parameter_count": count_parameters(self.model),
        }
        save_torch(checkpoint_payload, output_dir / "checkpoint.pt")
        save_json(feature_stats.to_dict(), output_dir / "feature_stats.json")
        save_json(self.graph_builder.graph_schema.to_dict(), output_dir / "graph_schema.json")
        save_json(split_manifest.to_dict(), output_dir / "split_manifest.json")
        save_json(history, output_dir / "history.json")
        save_json(
            {
                "validation": val_metrics,
                "test": test_metrics,
                "external": external_metrics,
                "temperature": temperature,
                "parameter_count": count_parameters(self.model),
            },
            output_dir / "summary.json",
        )
        return {
            "checkpoint_path": str(output_dir / "checkpoint.pt"),
            "feature_stats": feature_stats,
            "temperature": temperature,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "external_metrics": external_metrics,
            "history": history,
        }

