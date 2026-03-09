import torch
import numpy as np
import mir_eval
from training import note_conversion_utils
import config


def calculate_mpe_metrics(pred_frets, gt_frets, silence_fret_idx):
    gt_active_mask = (gt_frets != silence_fret_idx) & (
        gt_frets != config.FRET_PADDING_VALUE
    )
    pred_active_mask = pred_frets != silence_fret_idx

    tp = ((gt_active_mask & pred_active_mask) & (gt_frets == pred_frets)).sum().item()
    fp = (pred_active_mask & ~gt_active_mask).sum().item() + (
        (gt_active_mask & pred_active_mask) & (gt_frets != pred_frets)
    ).sum().item()
    fn = (~pred_active_mask & gt_active_mask).sum().item() + (
        (gt_active_mask & pred_active_mask) & (gt_frets != pred_frets)
    ).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "mpe_precision": precision,
        "mpe_recall": recall,
        "mpe_f1": f1,
    }


def calculate_note_level_metrics(predicted_notes, gt_notes_raw, onset_window=0.05):
    gt_notes_eval = [
        {"start_time": n[0], "end_time": n[1], "string": int(n[2]), "fret": int(n[3])}
        for n in gt_notes_raw.numpy()
        if 0 <= int(n[2]) < config.DEFAULT_NUM_STRINGS
        and 0 <= int(n[3]) <= config.MAX_FRETS
    ]

    if not gt_notes_eval and not predicted_notes:
        return {"tdr_precision": 1.0, "tdr_recall": 1.0, "tdr_f1": 1.0}

    tp_tdr = 0
    matched_pred_indices = set()
    for gt_note in gt_notes_eval:
        for pred_idx, pred_note in enumerate(predicted_notes):
            if pred_idx in matched_pred_indices:
                continue

            onset_match = (
                abs(gt_note["start_time"] - pred_note["start_time"]) <= onset_window
            )
            string_match = gt_note["string"] == pred_note["string"]
            fret_match = gt_note["fret"] == pred_note["fret"]

            if onset_match and string_match and fret_match:
                tp_tdr += 1
                matched_pred_indices.add(pred_idx)
                break

    p_tdr = (
        tp_tdr / len(predicted_notes)
        if predicted_notes
        else (1.0 if not gt_notes_eval else 0.0)
    )
    r_tdr = (
        tp_tdr / len(gt_notes_eval)
        if gt_notes_eval
        else (1.0 if not predicted_notes else 0.0)
    )
    f1_tdr = 2 * p_tdr * r_tdr / (p_tdr + r_tdr) if (p_tdr + r_tdr) > 0 else 0.0

    return {"tdr_precision": p_tdr, "tdr_recall": r_tdr, "tdr_f1": f1_tdr}


def calculate_onset_event_metrics(predicted_notes, gt_notes_raw, onset_window=0.05):
    gt_onsets_times = np.unique(np.array([n[0] for n in gt_notes_raw.numpy()]))
    pred_onsets_times = np.unique(np.array([n['start_time'] for n in predicted_notes]))

    if len(pred_onsets_times) == 0:
        if len(gt_onsets_times) == 0:
            return {"onset_precision_event": 1.0, "onset_recall_event": 1.0, "onset_f1_event": 1.0}
        else:
            return {"onset_precision_event": 0.0, "onset_recall_event": 0.0, "onset_f1_event": 0.0}

    ons_f1, ons_p, ons_r = mir_eval.onset.f_measure(gt_onsets_times, pred_onsets_times, window=onset_window)
    return {"onset_precision_event": ons_p, "onset_recall_event": ons_r, "onset_f1_event": ons_f1}


def full_evaluation(model, dataloader, device, config_obj, onset_threshold):
    model.eval()

    all_pred_notes_at_thresh = []
    all_gt_raw_labels = []

    all_fret_pred_indices_list = []
    all_fret_targets_list = []

    with torch.no_grad():
        for features, labels, lengths, raw_labels_batch, _ in dataloader:
            features = features.to(device)
            _, fret_targets_b = labels

            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits)

            onset_preds_binary = (onset_probs > onset_threshold).float()
            fret_pred_indices = torch.argmax(fret_logits, dim=-1)

            for i in range(features.size(0)):
                true_len = lengths[i].item()

                pred_notes_sample = note_conversion_utils.frames_to_notes_for_eval(
                    onset_preds_binary[i, :true_len, :].cpu(),
                    fret_pred_indices[i, :true_len, :].cpu(),
                    frame_hop_length=config_obj.HOP_LENGTH,
                    audio_sample_rate=config_obj.SAMPLE_RATE,
                    max_fret_value=config_obj.MAX_FRETS,
                )
                all_pred_notes_at_thresh.append(pred_notes_sample)
                all_gt_raw_labels.append(raw_labels_batch[i])

                all_fret_pred_indices_list.append(
                    fret_pred_indices[i, :true_len, :].cpu()
                )
                all_fret_targets_list.append(fret_targets_b[i, :true_len, :].cpu())

    tdr_scores, onset_event_scores = [], []
    for pred_n, gt_raw in zip(all_pred_notes_at_thresh, all_gt_raw_labels):
        tdr_scores.append(calculate_note_level_metrics(pred_n, gt_raw))
        onset_event_scores.append(calculate_onset_event_metrics(pred_n, gt_raw))

    avg_tdr = {key: np.mean([s[key] for s in tdr_scores]) for key in tdr_scores[0]}
    avg_onset_event = {
        key: np.mean([s[key] for s in onset_event_scores])
        for key in onset_event_scores[0]
    }

    silence_fret_idx = config_obj.MAX_FRETS + config_obj.FRET_SILENCE_CLASS_OFFSET
    mpe_metrics = calculate_mpe_metrics(
        torch.cat(all_fret_pred_indices_list),
        torch.cat(all_fret_targets_list),
        silence_fret_idx=silence_fret_idx,
    )

    final_metrics = {**avg_tdr, **avg_onset_event, **mpe_metrics}

    return final_metrics
