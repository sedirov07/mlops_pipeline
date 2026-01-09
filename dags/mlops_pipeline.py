import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from evidently.metrics import DatasetDriftMetric
from evidently.report import Report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Ensure project sources on sys.path so imports like `from src...` work in container
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
EXTRA_PATHS = [PROJECT_ROOT, PROJECT_ROOT / "project", PROJECT_ROOT / "src"]
for candidate in EXTRA_PATHS:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

from src.automl import load_data, run_automl  # noqa: E402


LOGGER = logging.getLogger("mlops_pipeline")


def ensure_dirs() -> None:
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(exist_ok=True)


def load_baseline() -> pd.DataFrame:
    return load_data()


def generate_synthetic_drift(baseline: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create a drifted dataset by shifting numeric features and altering class balance."""

    rng = np.random.default_rng(seed)
    drift = baseline.copy()
    numeric_cols = [col for col in drift.select_dtypes(include="number").columns if col != "target"]
    if numeric_cols:
        noise = rng.normal(loc=0.75, scale=0.25, size=(len(drift), len(numeric_cols)))
        drift.loc[:, numeric_cols] = drift[numeric_cols].values + noise

    # Alter class distribution: oversample class 2 and undersample others.
    class_two = drift[drift["target"] == drift["target"].max()]
    others = drift[drift["target"] != drift["target"].max()]
    sampled_others = others.sample(frac=0.4, random_state=seed)
    drifted = pd.concat([class_two, sampled_others], ignore_index=True)
    drifted = drifted.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return drifted


def load_current_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    current_path = os.getenv("DRIFT_DATA_PATH", PROJECT_ROOT / "data" / "current.csv")
    baseline = load_baseline()
    current: pd.DataFrame
    path_obj = Path(current_path)
    if path_obj.exists():
        current = pd.read_csv(current_path)
    else:
        current = generate_synthetic_drift(baseline)
    missing_cols = set(baseline.columns) - set(current.columns)
    if missing_cols:
        raise ValueError(f"Current data missing columns: {missing_cols}")
    current = current[baseline.columns]
    return baseline, current


def prepare_augmented_dataset(baseline: pd.DataFrame, drifted: pd.DataFrame) -> pd.DataFrame:
    """Объединяет baseline и drifted данные для переобучения модели.
    
    При обнаружении дрифта модель должна быть переобучена на расширенном датасете,
    включающем как исходные данные, так и новые данные с дрифтом.
    Это позволяет модели адаптироваться к изменившемуся распределению.
    """
    augmented = pd.concat([baseline, drifted], ignore_index=True)
    # Перемешиваем данные для лучшего обучения
    augmented = augmented.sample(frac=1.0, random_state=42).reset_index(drop=True)
    LOGGER.info(
        "Prepared augmented dataset: baseline=%d + drifted=%d = total %d samples",
        len(baseline), len(drifted), len(augmented)
    )
    return augmented


def save_augmented_data_for_training() -> str:
    """Сохраняет объединенный датасет для использования при переобучении."""
    ensure_dirs()
    baseline, drifted = load_current_data()
    augmented = prepare_augmented_dataset(baseline, drifted)
    augmented_path = PROJECT_ROOT / "data" / "augmented_train.csv"
    augmented.to_csv(augmented_path, index=False)
    LOGGER.info("Saved augmented training data to %s", augmented_path)
    return str(augmented_path)


def generate_drift_report(**context) -> bool:
    ensure_dirs()
    baseline, current = load_current_data()
    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=baseline, current_data=current)
    report_path = PROJECT_ROOT / "logs" / "evidently_drift_report.html"
    report.save_html(str(report_path))
    raw_metrics = report.as_dict().get("metrics", [])
    metric_payload = raw_metrics[0].get("result", {}) if raw_metrics else {}
    metric_payload["report_path"] = str(report_path)
    dataset_drift = bool(metric_payload.get("dataset_drift"))
    ti = context.get("ti")
    if ti is not None:
        ti.xcom_push(key="dataset_drift", value=dataset_drift)
        ti.xcom_push(key="dataset_drift_payload", value=metric_payload)
    LOGGER.info("Data drift status: %s", dataset_drift)
    return dataset_drift


def branch_on_drift(**context) -> str:
    ti = context.get("ti")
    drift_detected = ti.xcom_pull(task_ids="check_data_drift") if ti else False
    return "train_model" if drift_detected else "skip_retraining"


def train_model(**context) -> Dict[str, Any]:
    """Переобучает модель на расширенном датасете (baseline + drifted данные).
    
    При обнаружении дрифта важно включить новые данные в обучающую выборку,
    чтобы модель могла адаптироваться к изменившемуся распределению.
    """
    # Подготавливаем augmented датасет с данными дрифта
    augmented_path = save_augmented_data_for_training()
    LOGGER.info("Training model on augmented dataset: %s", augmented_path)
    
    # Передаем путь к augmented данным в run_automl
    result = run_automl(data_path=augmented_path)
    return result or {}


def prepare_eval_split_on_drifted_data() -> tuple[pd.DataFrame, pd.Series]:
    """Подготавливает тестовую выборку из данных с дрифтом.
    
    Для корректного сравнения старой (Production) и новой (Staging) моделей
    необходимо оценивать их именно на данных с дрифтом — это показывает,
    насколько хорошо каждая модель справляется с изменившимся распределением.
    """
    _, drifted = load_current_data()
    
    # Разделяем drifted данные на train/test для честной оценки
    _, test_df = train_test_split(
        drifted,
        test_size=0.3,
        random_state=42,
        stratify=drifted["target"],
    )
    features = test_df.drop(columns=["target"])
    target = test_df["target"]
    LOGGER.info("Prepared evaluation split from drifted data: %d samples", len(test_df))
    return features, target


def _predict_labels(model: Any, features: pd.DataFrame) -> pd.Series:
    try:
        preds = model.predict(features)
    except Exception:
        preds = model.predict(features.values)
    if hasattr(preds, "values"):
        preds = preds.values
    return pd.Series(preds).astype(int)


def evaluate_candidate(**context) -> Dict[str, Any]:
    """Оценивает Staging и Production модели на данных с дрифтом.
    
    Сравнение проводится именно на drifted данных, чтобы определить,
    какая модель лучше справляется с изменившимся распределением.
    """
    ti = context.get("ti")
    train_info: Dict[str, Any] = ti.xcom_pull(task_ids="train_model") if ti else {}
    registration = train_info.get("registration") or {}
    model_name = registration.get("model_name") or os.getenv("MLFLOW_REGISTERED_NAME", "iris_pycaret_model")

    client = MlflowClient()
    # Оцениваем модели на данных с дрифтом!
    features, target = prepare_eval_split_on_drifted_data()
    evaluation: Dict[str, Any] = {"model_name": model_name, "eval_data": "drifted"}

    def eval_stage(stage: str) -> Optional[Dict[str, Any]]:
        try:
            versions = client.get_latest_versions(model_name, stages=[stage])
        except MlflowException as exc:
            LOGGER.warning("Model registry lookup failed for %s stage %s: %s", model_name, stage, exc)
            return None
        if not versions:
            return None
        version = versions[0].version
        model_uri = f"models:/{model_name}/{stage}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            preds = _predict_labels(model, features)
            metrics_payload = {
                "accuracy": float(accuracy_score(target, preds)),
                "precision_macro": float(precision_score(target, preds, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(target, preds, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(target, preds, average="macro", zero_division=0)),
            }
        except Exception as exc:
            LOGGER.warning("Failed to load/eval model %s stage %s: %s", model_name, stage, exc)
            return {
                "stage": stage,
                "version": version,
                "model_uri": model_uri,
                "error": str(exc),
            }
        return {
            "stage": stage,
            "version": version,
            "metrics": metrics_payload,
            "model_uri": model_uri,
        }

    production_info = eval_stage("Production")
    staging_info = eval_stage("Staging")
    evaluation["production"] = production_info
    evaluation["staging"] = staging_info

    promote = False
    if staging_info is None:
        LOGGER.warning("No staging model available for evaluation")
    elif staging_info.get("error"):
        LOGGER.warning("Staging model load/eval failed: %s", staging_info.get("error"))
        promote = False
    elif production_info is None:
        promote = True
    else:
        if staging_info.get("metrics") and production_info.get("metrics"):
            promote = staging_info["metrics"].get("f1_macro", 0.0) >= production_info["metrics"].get("f1_macro", 0.0)
        else:
            LOGGER.warning("Missing metrics for staging or production; skip promotion")
            promote = False

    evaluation["promote"] = promote
    LOGGER.info("Evaluation result: %s", evaluation)
    if ti is not None:
        ti.xcom_push(key="evaluation", value=evaluation)
    return evaluation


def promote_if_better(**context) -> Dict[str, Any]:
    ti = context.get("ti")
    evaluation = ti.xcom_pull(task_ids="evaluate_candidate") if ti else {}
    train_info = ti.xcom_pull(task_ids="train_model") if ti else {}
    registration = train_info.get("registration") or {}
    model_name = evaluation.get("model_name") or registration.get("model_name")
    if not evaluation or not evaluation.get("promote") or not model_name:
        LOGGER.info("No promotion performed. Evaluation: %s", evaluation)
        return {"promoted": False, "reason": "no-promotion", "evaluation": evaluation}

    staging_info = evaluation.get("staging") or {}
    staging_version = staging_info.get("version") or registration.get("staging_version")
    if staging_version is None:
        LOGGER.warning("Staging version not found for promotion")
        return {"promoted": False, "reason": "missing-staging", "evaluation": evaluation}

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version,
        stage="Production",
        archive_existing_versions=True,
    )
    LOGGER.info("Promoted model %s version %s to Production", model_name, staging_version)
    result = {"promoted": True, "model_name": model_name, "version": staging_version}
    ti.xcom_push(key="promotion_result", value=result)
    return result


with DAG(
    dag_id="mlops_monitoring_pipeline",
    default_args={"owner": "airflow"},
    schedule_interval="0 6 * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "pycaret", "mlflow", "drift"],
) as dag:
    check_drift = PythonOperator(
        task_id="check_data_drift",
        python_callable=generate_drift_report,
        provide_context=True,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=branch_on_drift,
        provide_context=True,
    )

    skip_retraining = EmptyOperator(task_id="skip_retraining")

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_candidate",
        python_callable=evaluate_candidate,
        provide_context=True,
    )

    promote_task = PythonOperator(
        task_id="promote_if_better",
        python_callable=promote_if_better,
        provide_context=True,
    )

    pipeline_done = EmptyOperator(task_id="pipeline_finished")

    check_drift >> branch
    branch >> skip_retraining >> pipeline_done
    branch >> train_model_task >> evaluate_task >> promote_task >> pipeline_done
