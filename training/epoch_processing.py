import torch
import numpy as np
from evaluation import metrics
from training import note_conversion_utils
import config


def train_one_epoch(model, dataloader, optimizer, criterion_combined, device_to_use):
    model.train()
    epoch_total_loss = 0.0
    for (
        features_data,
        labels_data_tuple,
        sequence_lengths,
        _raw_labels_batch,
        _,
    ) in dataloader:
        features_data = features_data.to(device_to_use)
        onset_targets_data, fret_targets_data = labels_data_tuple
        onset_targets_data = onset_targets_data.to(device_to_use)
        fret_targets_data = fret_targets_data.to(device_to_use)
        sequence_lengths = sequence_lengths.to(device_to_use)

        optimizer.zero_grad()
        onset_predictions_logits, fret_predictions_logits = model(features_data)

        loss_val, loss_onset_val, loss_fret_val = criterion_combined(
            onset_predictions_logits,
            fret_predictions_logits,
            onset_targets_data,
            fret_targets_data,
            sequence_lengths,
        )
        loss_val.backward()
        optimizer.step()
        epoch_total_loss += loss_val.item()

    num_batches = len(dataloader) if len(dataloader) > 0 else 1
    return {"train_total_loss": epoch_total_loss / num_batches}


def evaluate_one_epoch(
    model_to_eval,
    eval_dataloader,
    eval_criterion_combined,
    device_to_use,
    config_obj,
):
    model_to_eval.eval()
    epoch_total_loss = 0.0
    all_onset_probs, all_fret_preds, all_fret_targets, all_raw_labels = [], [], [], []

    with torch.no_grad():
        for features, labels, lengths, raw_labels_batch, _ in eval_dataloader:
            features = features.to(device_to_use)
            onset_targets_b, fret_targets_b = labels
            onset_targets_b = onset_targets_b.to(device_to_use)
            fret_targets_b = fret_targets_b.to(device_to_use)
            sequence_lengths = lengths.to(device_to_use)
            onset_logits, fret_logits = model_to_eval(features)

            if eval_criterion_combined:
                loss, _, _ = eval_criterion_combined(
                    onset_logits,
                    fret_logits,
                    onset_targets_b,
                    fret_targets_b,
                    sequence_lengths,
                )
                epoch_total_loss += loss.item()

            onset_probs_b = torch.sigmoid(onset_logits)
            fret_pred_indices_b = torch.argmax(fret_logits, dim=-1)

            for i in range(features.size(0)):
                true_len = lengths[i].item()
                all_onset_probs.append(onset_probs_b[i, :true_len, :].cpu())
                all_fret_preds.append(fret_pred_indices_b[i, :true_len, :].cpu())
                all_fret_targets.append(fret_targets_b[i, :true_len, :].cpu())
                all_raw_labels.append(raw_labels_batch[i])

    pred_notes_at_fixed_thresh = [
        note_conversion_utils.frames_to_notes_for_eval(
            (p > 0.5).float(),
            f,
            config_obj.HOP_LENGTH,
            config_obj.SAMPLE_RATE,
            config_obj.MAX_FRETS,
        )
        for p, f in zip(all_onset_probs, all_fret_preds)
    ]
    tdr_scores = [
        metrics.calculate_note_level_metrics(pred, raw)
        for pred, raw in zip(pred_notes_at_fixed_thresh, all_raw_labels)
    ]
    onset_event_scores = [
        metrics.calculate_onset_event_metrics(pred, raw)
        for pred, raw in zip(pred_notes_at_fixed_thresh, all_raw_labels)
    ]
    avg_tdr = (
        {
            f"val_{key}_at_0.5": np.mean([s[key] for s in tdr_scores])
            for key in tdr_scores[0]
        }
        if tdr_scores
        else {}
    )
    avg_onset_event = (
        {
            f"val_{key}_at_0.5": np.mean([s[key] for s in onset_event_scores])
            for key in onset_event_scores[0]
        }
        if onset_event_scores
        else {}
    )

    silence_fret_idx = config_obj.MAX_FRETS + config_obj.FRET_SILENCE_CLASS_OFFSET
    mpe_metrics_results = metrics.calculate_mpe_metrics(
        torch.cat(all_fret_preds), torch.cat(all_fret_targets), silence_fret_idx
    )
    val_mpe_metrics = {
        f"val_{key}": value for key, value in mpe_metrics_results.items()
    }

    metrics_results = {
        "val_total_loss": epoch_total_loss
        / (len(eval_dataloader) if len(eval_dataloader) > 0 else 1),
        **avg_tdr,
        **avg_onset_event,
        **val_mpe_metrics,
    }
    return metrics_results
