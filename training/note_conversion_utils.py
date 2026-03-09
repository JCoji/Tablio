import config

def frames_to_notes_for_eval(
        onset_preds_binary_frames,
        fret_pred_indices_frames,
        frame_hop_length,
        audio_sample_rate,
        max_fret_value=config.MAX_FRETS,
        min_note_duration_frames=config.MIN_NOTE_DURATION_FRAMES,
        open_string_pitches=None
):
    if open_string_pitches is None:
        open_string_pitches = config.OPEN_STRING_PITCHES_MIDI

    num_frames, num_strings = onset_preds_binary_frames.shape
    time_per_frame = frame_hop_length / audio_sample_rate
    predicted_notes_list = []

    silence_fret_class_idx = max_fret_value + config.FRET_SILENCE_CLASS_OFFSET

    for string_idx in range(num_strings):
        active_note_start_frame = None
        active_note_fret_val = None

        for frame_idx in range(num_frames):
            is_onset_active = onset_preds_binary_frames[
                                  frame_idx, string_idx].item() > 0.5
            current_fret_val = fret_pred_indices_frames[frame_idx, string_idx].item()

            note_should_terminate = False
            if active_note_start_frame is not None:
                if is_onset_active and frame_idx > active_note_start_frame:
                    note_should_terminate = True
                elif current_fret_val == silence_fret_class_idx:
                    note_should_terminate = True
                elif current_fret_val != active_note_fret_val:
                    note_should_terminate = True
                elif frame_idx == num_frames - 1:
                    note_should_terminate = True

                if note_should_terminate:
                    start_time_sec = active_note_start_frame * time_per_frame
                    end_time_sec = frame_idx * time_per_frame
                    duration_in_frames = frame_idx - active_note_start_frame

                    if duration_in_frames >= min_note_duration_frames and active_note_fret_val != silence_fret_class_idx:
                        if 0 <= active_note_fret_val <= max_fret_value:
                            pitch_midi_val = open_string_pitches[string_idx] + active_note_fret_val
                            predicted_notes_list.append({
                                'start_time': start_time_sec,
                                'end_time': end_time_sec,
                                'pitch_midi': int(round(pitch_midi_val)),
                                'string': string_idx,
                                'fret': int(active_note_fret_val)
                            })
                    active_note_start_frame = None
                    active_note_fret_val = None

            if is_onset_active and current_fret_val != silence_fret_class_idx:
                if active_note_start_frame is not None and frame_idx > active_note_start_frame:
                    pass

                active_note_start_frame = frame_idx
                active_note_fret_val = current_fret_val

        if active_note_start_frame is not None:
            start_time_sec = active_note_start_frame * time_per_frame
            end_time_sec = num_frames * time_per_frame
            duration_in_frames = num_frames - active_note_start_frame
            if duration_in_frames >= min_note_duration_frames and active_note_fret_val != silence_fret_class_idx:
                if 0 <= active_note_fret_val <= max_fret_value:
                    pitch_midi_val = open_string_pitches[string_idx] + active_note_fret_val
                    predicted_notes_list.append({
                        'start_time': start_time_sec,
                        'end_time': end_time_sec,
                        'pitch_midi': int(round(pitch_midi_val)),
                        'string': string_idx,
                        'fret': int(active_note_fret_val)
                    })
    return predicted_notes_list