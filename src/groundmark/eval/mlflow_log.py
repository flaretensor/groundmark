from pathlib import Path


def log_metrics_artifact(uri: str, experiment_name: str, path: Path, params: dict, metrics: dict):
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment(experiment_name)
    )
    run_id = client.create_run(experiment_id, run_name=path.parent.name).info.run_id
    try:
        for name, value in params.items():
            client.log_param(run_id, name, value)
        for name, value in metrics.items():
            client.log_metric(run_id, name, value)
        client.log_artifact(run_id, str(path))
        client.set_terminated(run_id)
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    return run_id
