import json
import os
import time
import logging
from typing import Any, Dict, Optional

from pycaret.classification import setup, compare_models, pull, save_model, predict_model
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.utils import autologging_utils


def load_data(data_path: Optional[str] = None):
    """Загружает данные для обучения.
    
    Args:
        data_path: Путь к CSV файлу с данными. Если None, используется Iris датасет.
                   При переобучении после дрифта передается путь к augmented датасету.
    """
    if data_path is not None:
        import pandas as pd
        df = pd.read_csv(data_path)
        logging.getLogger("automl").info("Loaded augmented data from %s (%d samples)", data_path, len(df))
        return df
    
    # Fallback: используем Iris — классический учебный датасет
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # Rename target column for PyCaret
    df = df.rename(columns={iris.target.name: 'target'})
    return df

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def patch_mlflow_stack_copy():
    """Add missing copy() helper to Mlflow's autologging stack (needed for PyCaret)."""

    stack = getattr(autologging_utils, "_active_run_stack", None)
    if stack is None or hasattr(stack, "copy"):
        return

    def _copy():
        try:
            return list(stack.value)
        except Exception:
            return []

    stack.copy = _copy


def wait_for_model_version_ready(name: str, version: str, client: MlflowClient, timeout: int = 120):
    """Poll MLflow until the newly registered model version becomes READY."""

    start = time.time()
    while time.time() - start <= timeout:
        model_version = client.get_model_version(name=name, version=version)
        status = getattr(model_version, "status", None)
        if status == "READY":
            return model_version
        if status == "FAILED":
            raise MlflowException(f"Model version {version} for {name} failed to register")
        time.sleep(2)
    raise TimeoutError(f"Timed out while waiting for model {name}:{version} to become READY")


def run_automl(data_path: Optional[str] = None):
    """Run AutoML and log metrics/artifacts to MLflow.
    
    Args:
        data_path: Путь к CSV файлу с данными для обучения.
                   Если None, используется базовый Iris датасет.
                   При переобучении после дрифта передается augmented датасет.
    """

    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "pycaret-automl")

    # Setup logging to file
    logger = logging.getLogger("automl")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/automl.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    logger.info("Starting AutoML run")

    ensure_dirs()
    patch_mlflow_stack_copy()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    df = load_data(data_path)
    logger.info("Training on dataset with %d samples", len(df))

    # Split for holdout evaluation
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

    run_result: Dict[str, Any] = {}
    with mlflow.start_run(run_name="pycaret-automl-run"):
        active_run = mlflow.active_run()
        # Train
        s = setup(
            data=train_df,
            target='target',
            html=False,
            session_id=42,
            verbose=False,
            # log_experiment=True,
            experiment_name=experiment_name,
        )
        best = compare_models()
        results = pull()
        logger.info("Compare models results:\n%s", results.head().to_string())

        # Predict on holdout
        pred_df = predict_model(best, data=test_df)
        # Determine predicted labels
        if 'Label' in pred_df.columns:
            y_pred = pred_df['Label'].values
        elif 'prediction_label' in pred_df.columns:
            y_pred = pred_df['prediction_label'].values
        else:
            # fallback: last column
            y_pred = pred_df.iloc[:, -1].values

        y_true = test_df['target'].values

        # Compute metrics
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics_payload = {
            "test_accuracy": acc,
            "test_precision_macro": prec,
            "test_recall_macro": rec,
            "test_f1_macro": f1,
        }

        logger.info("Test metrics: accuracy=%f, precision=%f, recall=%f, f1=%f", acc, prec, rec, f1)

        # Log metrics
        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric('test_precision_macro', prec)
        mlflow.log_metric('test_recall_macro', rec)
        mlflow.log_metric('test_f1_macro', f1)

        # Save local fallback
        model_path = os.path.join("models", "best_model_pycaret")
        save_model(best, model_path)
        artifact_file = model_path + ".pkl"
        artifact_present = os.path.exists(artifact_file)

        # Log model to MLflow as sklearn model (artifact path: model)
        registered_name = os.getenv("MLFLOW_REGISTERED_NAME", "iris_pycaret_model")
        run_based_uri = None
        registration_info: Optional[Dict[str, Any]] = None
        try:
            mlflow.sklearn.log_model(best, artifact_path="model")
            if active_run:
                run_based_uri = f"runs:/{active_run.info.run_id}/model"
            model_uri = run_based_uri
            # Register and stage
            registered_model = mlflow.register_model(model_uri, registered_name)
            logger.info(
                "Registered model in MLflow Registry as %s (version %s)",
                registered_name,
                registered_model.version,
            )
            ready_model = wait_for_model_version_ready(registered_name, registered_model.version, client)
            client.transition_model_version_stage(
                name=registered_name,
                version=ready_model.version,
                stage="Staging",
            )
            logger.info(
                "Moved model %s version %s to Staging stage",
                registered_name,
                ready_model.version,
            )
            registration_info = {
                "model_name": registered_name,
                "staging_version": ready_model.version,
                "staging_model_uri": f"models:/{registered_name}/Staging",
                "model_uri": model_uri,
                "run_based_uri": run_based_uri,
                "local_model_uri": f"file://{os.path.abspath(artifact_file)}" if artifact_present else None,
            }
        except (MlflowException, TimeoutError) as e:
            logger.warning("Failed to stage model in MLflow Registry: %s", e)
            registration_info = {
                "model_name": registered_name,
                "staging_version": None,
                "staging_model_uri": None,
                "model_uri": run_based_uri,
                "local_model_uri": f"file://{os.path.abspath(artifact_file)}" if artifact_present else None,
                "error": str(e),
            }
        except Exception as e:
            logger.warning("Unexpected error during model registration: %s", e)
            registration_info = {
                "model_name": registered_name,
                "staging_version": None,
                "staging_model_uri": None,
                "model_uri": run_based_uri,
                "local_model_uri": f"file://{os.path.abspath(artifact_file)}" if artifact_present else None,
                "error": str(e),
            }
        run_result = {
            "run_id": active_run.info.run_id if active_run else None,
            "experiment_name": experiment_name,
            "metrics": metrics_payload,
            "model_artifact": artifact_file if artifact_present else None,
            "registration": registration_info,
        }
        logger.info("AutoML result payload: %s", run_result)
    return run_result


if __name__ == '__main__':
    result = run_automl()
    print(json.dumps(result, indent=2, ensure_ascii=False))
