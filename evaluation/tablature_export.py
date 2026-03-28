import os
import config
import torch
import numpy as np


def _generate_tablature_matrix_slots(
    onset_data_frames,
    fret_data_frames,
    num_total_frames,
    num_total_strings,
    max_fret_val,
    onset_threshold,
):
    silence_fret_val = max_fret_val + config.FRET_SILENCE_CLASS_OFFSET
    num_tab_slots = (
        num_total_frames + config.TAB_FRAMES_PER_SLOT - 1
    ) // config.TAB_FRAMES_PER_SLOT
    tab_matrix = [[""] * num_tab_slots for _ in range(num_total_strings)]

    for slot_idx in range(num_tab_slots):
        start_frame_for_slot = slot_idx * config.TAB_FRAMES_PER_SLOT
        end_frame_for_slot = min(
            start_frame_for_slot + config.TAB_FRAMES_PER_SLOT, num_total_frames
        )

        for string_model_idx in range(num_total_strings):
            onset_detected_in_slot = False
            fret_at_onset = -1

            for frame_idx in range(start_frame_for_slot, end_frame_for_slot):
                if (
                    onset_data_frames[frame_idx, string_model_idx].item()
                    >= onset_threshold
                ):
                    fret_val = fret_data_frames[frame_idx, string_model_idx].item()
                    if 0 <= fret_val <= max_fret_val:
                        fret_at_onset = fret_val
                    else:
                        fret_at_onset = silence_fret_val
                    onset_detected_in_slot = True
                    break

            if onset_detected_in_slot:
                if fret_at_onset != silence_fret_val:
                    slot_content = str(fret_at_onset).ljust(
                        config.TAB_SLOT_CHAR_WIDTH, config.TAB_SUSTAIN_CHAR
                    )[: config.TAB_SLOT_CHAR_WIDTH]
                else:
                    slot_content = config.TAB_SUSTAIN_CHAR * config.TAB_SLOT_CHAR_WIDTH
            else:
                slot_content = config.TAB_SUSTAIN_CHAR * config.TAB_SLOT_CHAR_WIDTH

            tab_matrix[string_model_idx][slot_idx] = slot_content
    return tab_matrix


def _format_tablature_matrix_to_text(tab_matrix_data_slots, num_total_strings):
    output_text_lines = []
    string_display_names = config.TAB_STRING_NAMES_DISPLAY_6_STRING
    if not tab_matrix_data_slots or not tab_matrix_data_slots[0]:
        return "No data to format."

    total_num_slots = len(tab_matrix_data_slots[0])
    for slot_start_batch_idx in range(
        0, total_num_slots, config.TAB_LINE_BREAK_AFTER_SLOTS
    ):
        slot_end_batch_idx = min(
            slot_start_batch_idx + config.TAB_LINE_BREAK_AFTER_SLOTS, total_num_slots
        )
        for display_string_idx in range(num_total_strings):
            model_string_idx_for_display = (num_total_strings - 1) - display_string_idx
            line_str = (
                string_display_names[display_string_idx] + config.TAB_SEPARATOR_CHAR
            )
            current_slot_count_in_line = 0
            for current_slot_idx in range(slot_start_batch_idx, slot_end_batch_idx):
                line_str += tab_matrix_data_slots[model_string_idx_for_display][
                    current_slot_idx
                ]
                current_slot_count_in_line += 1
                if (
                    current_slot_count_in_line
                    % config.TAB_GROUP_SEPARATOR_EVERY_N_SLOTS
                    == 0
                    and current_slot_idx < slot_end_batch_idx - 1
                ):
                    line_str += config.TAB_SEPARATOR_CHAR
            output_text_lines.append(line_str)
        output_text_lines.append("")
    return "\n".join(output_text_lines)


def generate_text_tablature_comparison(
    onset_probs,
    fret_indices,
    onset_gt,
    fret_gt,
    track_id,
    onset_threshold_optimal,
    max_fret_val,
    output_directory_path,
):
    try:
        gt_tab_matrix = _generate_tablature_matrix_slots(
            onset_gt,
            fret_gt,
            onset_gt.shape[0],
            config.DEFAULT_NUM_STRINGS,
            max_fret_val,
            onset_threshold=0.5,
        )
        gt_tab_text = _format_tablature_matrix_to_text(
            gt_tab_matrix, config.DEFAULT_NUM_STRINGS
        )

        base_filename = str(track_id).replace(os.sep, "_")
        gt_filename = f"{base_filename}_ground_truth.txt"
        gt_filepath = os.path.join(output_directory_path, gt_filename)

        with open(gt_filepath, "w", encoding="utf-8") as f:
            f.write(
                f"Track ID: {track_id}\n--- Ground Truth Tablature ---\n\n{gt_tab_text}"
            )

        pred_tab_matrix = _generate_tablature_matrix_slots(
            onset_probs,
            fret_indices,
            onset_probs.shape[0],
            config.DEFAULT_NUM_STRINGS,
            max_fret_val,
            onset_threshold=onset_threshold_optimal,
        )
        pred_tab_text = _format_tablature_matrix_to_text(
            pred_tab_matrix, config.DEFAULT_NUM_STRINGS
        )

        pred_filename = (
            f"{base_filename}_predicted_thresh{onset_threshold_optimal:.2f}.txt"
        )
        pred_filepath = os.path.join(output_directory_path, pred_filename)

        with open(pred_filepath, "w", encoding="utf-8") as f:
            f.write(
                f"Track ID: {track_id}\n--- Predicted Tablature (Onset Thresh: {onset_threshold_optimal:.2f}) ---\n\n{pred_tab_text}"
            )
    except Exception as e:
        print(f"\nError generating tablature for track {track_id}: {e}")

def save_notes_to_ascii_tab(notes_list, output_filepath, track_id, config_obj):
    if not notes_list:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(f"Track ID: {track_id}\n\n--- Predicted Tablature ---\n\n(No notes detected)")
        return

    try:
        max_time = max(note['end_time'] for note in notes_list) if notes_list else 0
    except (TypeError, KeyError):
        max_time = 0

    time_per_frame = config_obj.HOP_LENGTH / config_obj.SAMPLE_RATE
    num_frames = int(np.ceil(max_time / time_per_frame)) + 1
    num_strings = config_obj.DEFAULT_NUM_STRINGS

    onset_frames = torch.zeros((num_frames, num_strings), dtype=torch.float32)
    fret_frames = torch.full((num_frames, num_strings),
                             config_obj.MAX_FRETS + config_obj.FRET_SILENCE_CLASS_OFFSET,
                             dtype=torch.long)

    for note in notes_list:
        try:
            start_frame = int(round(note['start_time'] / time_per_frame))
            end_frame = int(round(note['end_time'] / time_per_frame))
            string_idx = note['string']
            fret_val = note['fret']

            if 0 <= string_idx < num_strings:
                if start_frame < num_frames:
                    onset_frames[start_frame, string_idx] = 1.0
                for frame_idx in range(start_frame, min(end_frame, num_frames)):
                    fret_frames[frame_idx, string_idx] = fret_val
        except (TypeError, KeyError):
            continue

    tab_matrix = _generate_tablature_matrix_slots(
        onset_frames,
        fret_frames,
        num_frames,
        num_strings,
        config_obj.MAX_FRETS,
        onset_threshold=0.5
    )
    tab_text = _format_tablature_matrix_to_text(tab_matrix, num_strings)

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(f"Track ID: {track_id}\n\n--- Predicted Tablature ---\n\n{tab_text}")
