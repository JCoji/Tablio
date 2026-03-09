from evaluation import metrics


def evaluate_model_on_test_set(
    model_to_eval,
    test_dataloader,
    device_to_use,
    config_obj,
    optimal_onset_threshold,
):
    test_set_metrics_results = metrics.full_evaluation(
        model=model_to_eval,
        dataloader=test_dataloader,
        device=device_to_use,
        config_obj=config_obj,
        onset_threshold=optimal_onset_threshold,
    )

    return test_set_metrics_results
