import torch
import os
import time
import json
from torch import optim
from tqdm import tqdm
from model import architecture, utils
from . import epoch_processing, loss_functions
from evaluation import performance_metrics
from vizualization import plotting


def run_training_loop(
    model_instance,
    device_to_use,
    train_loader,
    validation_loader,
    optimizer_instance,
    scheduler_instance,
    criterion_instance_combined,
    run_training_config,
    config_obj,
):
    num_epochs_total = run_training_config.get(
        "NUM_EPOCHS", config_obj.NUM_EPOCHS_DEFAULT
    )
    artifacts_output_dir = run_training_config.get("ARTIFACTS_DIR")
    training_log_filepath = run_training_config.get("LOG_FILE_PATH")
    early_stop_patience_val = run_training_config.get(
        "EARLY_STOPPING_PATIENCE", config_obj.EARLY_STOPPING_PATIENCE_DEFAULT
    )
    checkpoint_metric_to_track = run_training_config.get(
        "CHECKPOINT_METRIC", config_obj.CHECKPOINT_METRIC_DEFAULT
    )

    best_tracked_metric_val = (
        -float("inf")
        if "loss" not in checkpoint_metric_to_track.lower()
        else float("inf")
    )
    epochs_without_improvement = 0

    training_history = {
        "train_total_loss": [],
        "val_total_loss": [],
        "lr": [],
        "val_tdr_f1_at_0.5": [],
        "val_tdr_precision_at_0.5": [],
        "val_tdr_recall_at_0.5": [],
        "val_onset_f1_event_at_0.5": [],
        "val_mpe_f1": [],
    }

    with open(training_log_filepath, "a", encoding="utf-8") as log_file_handle:
        for current_epoch_num in range(num_epochs_total):
            epoch_description_str = f"Epoka {current_epoch_num + 1}/{num_epochs_total}"

            train_pbar = tqdm(
                train_loader,
                desc=f"{epoch_description_str} [Trening]",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            train_epoch_metrics = epoch_processing.train_one_epoch(
                model_instance,
                train_pbar,
                optimizer_instance,
                criterion_instance_combined,
                device_to_use,
            )

            if "train_total_loss" in training_history:
                training_history["train_total_loss"].append(
                    train_epoch_metrics["train_total_loss"]
                )

            val_pbar = tqdm(
                validation_loader,
                desc=f"{epoch_description_str} [Walidacja]",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            val_epoch_all_metrics = epoch_processing.evaluate_one_epoch(
                model_instance,
                val_pbar,
                criterion_instance_combined,
                device_to_use,
                config_obj,
            )

            for key, value in val_epoch_all_metrics.items():
                if key in training_history:
                    training_history[key].append(value)

            current_learning_rate = optimizer_instance.param_groups[0]["lr"]
            training_history["lr"].append(current_learning_rate)

            log_lines_for_file = [
                f"\n--- {epoch_description_str} ---",
                f"  LR: {current_learning_rate:.2e} | Train Loss: {train_epoch_metrics['train_total_loss']:.4f} | Val Loss: {val_epoch_all_metrics.get('val_total_loss', 0.0):.4f}",
                f"  MPE F1 (ramki): {val_epoch_all_metrics.get('val_mpe_f1', 0.0):.4f}",
                f"  TDR F1 (nuty, thr=0.5): {val_epoch_all_metrics.get('val_tdr_f1_at_0.5', 0.0):.4f} (P: {val_epoch_all_metrics.get('val_tdr_precision_at_0.5', 0.0):.4f}, R: {val_epoch_all_metrics.get('val_tdr_recall_at_0.5', 0.0):.4f})",
                f"  Onset F1 (event, thr=0.5): {val_epoch_all_metrics.get('val_onset_f1_event_at_0.5', 0.0):.4f}",
            ]
            log_text_chunk = "\n".join(log_lines_for_file)
            log_file_handle.write(log_text_chunk + "\n")
            log_file_handle.flush()
            print(log_text_chunk)

            effective_checkpoint_metric = checkpoint_metric_to_track
            if effective_checkpoint_metric not in val_epoch_all_metrics:
                corrected_metric_key = f"{effective_checkpoint_metric}_at_0.5"
                if corrected_metric_key in val_epoch_all_metrics:
                    effective_checkpoint_metric = corrected_metric_key

            metric_value_for_checkpoint = val_epoch_all_metrics.get(
                effective_checkpoint_metric, float("-inf")
            )

            scheduler_mode = (
                "max" if "loss" not in effective_checkpoint_metric.lower() else "min"
            )
            if scheduler_instance:
                scheduler_instance.step(metric_value_for_checkpoint)

            is_improved = (
                (metric_value_for_checkpoint < best_tracked_metric_val)
                if scheduler_mode == "min"
                else (metric_value_for_checkpoint > best_tracked_metric_val)
            )
            if is_improved and metric_value_for_checkpoint != float("-inf"):
                epochs_without_improvement = 0
                best_tracked_metric_val = metric_value_for_checkpoint
                torch.save(
                    model_instance.state_dict(),
                    os.path.join(artifacts_output_dir, "best_model.pth"),
                )
                print(
                    f"    -> Zapisano nowy najlepszy model ({effective_checkpoint_metric}: {best_tracked_metric_val:.4f})"
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience_val:
                    print(
                        f"\n    Wczesne zatrzymanie treningu po {epochs_without_improvement} epokach bez poprawy dla '{effective_checkpoint_metric}'."
                    )
                    break

    if training_history and hasattr(plotting, "plot_training_history"):
        history_plot_path = os.path.join(
            artifacts_output_dir, "training_history_summary.png"
        )
        plotting.plot_training_history(
            training_history, output_save_path=history_plot_path
        )

    return training_history


def process_single_hyperparameter_run(
    run_id,
    hyperparams_combo,
    current_augmentation_params,
    config_obj,
    main_artifacts_dir,
    train_loader,
    validation_loader,
    test_loader,
):
    run_start_time = time.time()

    run_description_str = hyperparams_combo.get("run_description", f"run_{run_id}")
    aug_suffix = (
        "_augEnabled"
        if current_augmentation_params.get("enable_audio_augmentations", False)
        or current_augmentation_params.get("enable_specaugment", False)
        else "_augDisabled"
    )
    current_run_folder_name_sanitized = (
        f"run_{run_id}_{run_description_str.replace(' ', '_')[:50]}{aug_suffix}"
    )
    current_run_artifacts_dir = os.path.join(
        main_artifacts_dir, current_run_folder_name_sanitized
    )
    current_run_log_file = os.path.join(current_run_artifacts_dir, "training_log.txt")
    run_config_path = os.path.join(
        current_run_artifacts_dir, "run_configuration.json"
    )

    os.makedirs(current_run_artifacts_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        temp_cnn_model = architecture.TabCNN()
        with torch.no_grad():
            calculated_cnn_out_dim = (
                temp_cnn_model(torch.randn(1, 1, config_obj.N_BINS_CQT, 32)).shape[1]
                * temp_cnn_model(torch.randn(1, 1, config_obj.N_BINS_CQT, 32)).shape[2]
            )
        del temp_cnn_model
    except Exception as e:
        return {
            "run_index": run_id,
            "params_combo": hyperparams_combo,
            "status": "CNN_DIM_ERROR",
            "error": str(e),
        }

    model_init_params = {
        "num_frames_rnn_input_dim": calculated_cnn_out_dim,
        "rnn_type": hyperparams_combo.get("RNN_TYPE", config_obj.RNN_TYPE_DEFAULT),
        "rnn_hidden_size": hyperparams_combo["RNN_HIDDEN_SIZE"],
        "rnn_layers": hyperparams_combo["RNN_LAYERS"],
        "rnn_dropout": hyperparams_combo["RNN_DROPOUT"],
        "rnn_bidirectional": hyperparams_combo.get("RNN_BIDIRECTIONAL", config_obj.RNN_BIDIRECTIONAL_DEFAULT),
    }
    current_model = architecture.GuitarTabCRNN(**model_init_params)
    current_model.to(device)

    onset_pos_weight = (
        torch.tensor(
            [hyperparams_combo["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device
        )
        if hyperparams_combo.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0
        else None
    )
    combined_loss_criterion = loss_functions.CombinedLoss(
        onset_pos_weight=onset_pos_weight,
        onset_loss_weight=hyperparams_combo["ONSET_LOSS_WEIGHT"],
    ).to(device)
    optimizer = optim.AdamW(
        current_model.parameters(),
        lr=hyperparams_combo["LEARNING_RATE_INIT"],
        weight_decay=hyperparams_combo["WEIGHT_DECAY"],
    )
    scheduler_mode = (
        "max" if "loss" not in config_obj.CHECKPOINT_METRIC_DEFAULT.lower() else "min"
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=hyperparams_combo["SCHEDULER_FACTOR"],
        patience=hyperparams_combo["SCHEDULER_PATIENCE"],
    )

    static_params_to_log = {
        "SAMPLE_RATE": config_obj.SAMPLE_RATE,
        "HOP_LENGTH": config_obj.HOP_LENGTH,
        "MAX_FRETS": config_obj.MAX_FRETS,
        "N_BINS_CQT": config_obj.N_BINS_CQT,
        "BINS_PER_OCTAVE_CQT": config_obj.BINS_PER_OCTAVE_CQT,
        "VALIDATION_SPLIT_SIZE": config_obj.VALIDATION_SPLIT_SIZE,
        "TEST_SPLIT_SIZE": config_obj.TEST_SPLIT_SIZE,
        "RANDOM_SEED": config_obj.RANDOM_SEED,
    }

    default_train_params_to_log = {
        "NUM_EPOCHS_DEFAULT": config_obj.NUM_EPOCHS_DEFAULT,
        "BATCH_SIZE_DEFAULT": config_obj.BATCH_SIZE_DEFAULT,
        "FRET_LOSS_WEIGHT_DEFAULT": config_obj.FRET_LOSS_WEIGHT_DEFAULT,
        "EARLY_STOPPING_PATIENCE_DEFAULT": config_obj.EARLY_STOPPING_PATIENCE_DEFAULT,
        "CHECKPOINT_METRIC_DEFAULT": config_obj.CHECKPOINT_METRIC_DEFAULT,
    }

    full_augmentation_params = current_augmentation_params.copy()
    if hasattr(config_obj, "DATASET_TRAIN_AUGMENTATION_REVERB_PARAMS"):
        full_augmentation_params["reverb_params"] = (
            config_obj.DATASET_TRAIN_AUGMENTATION_REVERB_PARAMS
        )
    if hasattr(config_obj, "DATASET_TRAIN_AUGMENTATION_EQ_PARAMS"):
        full_augmentation_params["eq_params"] = (
            config_obj.DATASET_TRAIN_AUGMENTATION_EQ_PARAMS
        )
    if hasattr(config_obj, "DATASET_TRAIN_AUGMENTATION_CLIPPING_PARAMS"):
        full_augmentation_params["clipping_params"] = (
            config_obj.DATASET_TRAIN_AUGMENTATION_CLIPPING_PARAMS
        )

    full_run_config_for_json = {
        "static_parameters": static_params_to_log,
        "default_training_parameters": default_train_params_to_log,
        "hyperparameters_tuned": hyperparams_combo,
        "augmentations": full_augmentation_params,
    }

    config_for_training_loop = {
        **hyperparams_combo,
        "ARTIFACTS_DIR": current_run_artifacts_dir,
        "LOG_FILE_PATH": current_run_log_file,
        "CHECKPOINT_METRIC": config_obj.CHECKPOINT_METRIC_DEFAULT,
        "EARLY_STOPPING_PATIENCE": config_obj.EARLY_STOPPING_PATIENCE_DEFAULT,
        "NUM_EPOCHS": config_obj.NUM_EPOCHS_DEFAULT,
    }

    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(full_run_config_for_json, f, indent=4)

    with open(current_run_log_file, "w", encoding="utf-8") as f:
        f.write(
            f"--- Konfiguracja Przebiegu: {current_run_folder_name_sanitized} ---\n\n"
        )
        f.write(json.dumps(full_run_config_for_json, indent=4))
        f.write("\n\n" + "=" * 50 + "\n")

    training_run_history = run_training_loop(
        model_instance=current_model,
        device_to_use=device,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer_instance=optimizer,
        scheduler_instance=scheduler,
        criterion_instance_combined=combined_loss_criterion,
        run_training_config=config_for_training_loop,
        config_obj=config_obj,
    )

    final_test_metrics = {}
    best_model_path = os.path.join(current_run_artifacts_dir, "best_model.pth")
    if test_loader and os.path.exists(best_model_path):
        final_loaded_model = utils.load_best_model(
            model_class=architecture.GuitarTabCRNN,
            model_path=best_model_path,
            run_config_path=run_config_path,
            device=device,
        )
        if final_loaded_model:
            print(f"\nEwaluacja na zbiorze testowym (próg 0.5)...")
            final_test_metrics = performance_metrics.evaluate_model_on_test_set(
                model_to_eval=final_loaded_model,
                test_dataloader=test_loader,
                device_to_use=device,
                config_obj=config_obj,
                optimal_onset_threshold=0.5,
            )
            print("Zakończono ewaluację na zbiorze testowym.")

    stopped_at_epoch = len(training_run_history.get("train_total_loss", []))

    final_checkpoint_metric_key = config_obj.CHECKPOINT_METRIC_DEFAULT
    if f"{final_checkpoint_metric_key}_at_0.5" in training_run_history:
        final_checkpoint_metric_key = f"{final_checkpoint_metric_key}_at_0.5"

    tracked_metric_history = training_run_history.get(final_checkpoint_metric_key, [])

    best_val_metric_final = 0.0
    if tracked_metric_history:
        best_val_metric_final = (
            max(tracked_metric_history)
            if "loss" not in final_checkpoint_metric_key.lower()
            else min(tracked_metric_history)
        )

    current_run_summary = {
        "run_index": run_id,
        "run_folder_name": current_run_folder_name_sanitized,
        "params_combo": hyperparams_combo,
        "augmentation_params": full_augmentation_params,
        f"best_{config_obj.CHECKPOINT_METRIC_DEFAULT}": best_val_metric_final,
        "test_metrics_at_0.5": final_test_metrics,
        "run_duration_minutes": (time.time() - run_start_time) / 60,
        "stopped_epoch": stopped_at_epoch,
        "status": "COMPLETED",
    }
    return current_run_summary
