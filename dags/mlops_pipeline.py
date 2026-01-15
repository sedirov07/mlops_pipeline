import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

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

import requests
from scipy import stats

LOGGER = logging.getLogger("mlops_pipeline")
REPORTS_DIR = PROJECT_ROOT / "logs" / "pipeline_reports"

# Configuration
FLASK_API_URL = os.getenv("FLASK_API_URL", "http://localhost:8000")
STATISTICAL_SIGNIFICANCE_ALPHA = 0.05  # p-value threshold
BOOTSTRAP_ITERATIONS = 1000


def ensure_dirs() -> None:
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline() -> pd.DataFrame:
    return load_data()


def generate_synthetic_drift(baseline: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
    """Create a drifted dataset by shifting numeric features and altering class balance.
    
    При каждом запуске генерируется уникальный датасет с дрифтом, используя
    текущее время как seed (если не указан явно).
    """
    # Генерируем уникальный seed на основе времени, если не указан
    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % (2**31)
    
    LOGGER.info("Generating synthetic drift with seed=%d", seed)
    
    rng = np.random.default_rng(seed)
    drift = baseline.copy()
    numeric_cols = [col for col in drift.select_dtypes(include="number").columns if col != "target"]
    
    # Варьируем параметры дрифта для разнообразия
    drift_magnitude = rng.uniform(0.5, 1.5)  # Сила смещения
    noise_scale = rng.uniform(0.15, 0.4)     # Разброс шума
    
    if numeric_cols:
        noise = rng.normal(loc=drift_magnitude, scale=noise_scale, size=(len(drift), len(numeric_cols)))
        drift.loc[:, numeric_cols] = drift[numeric_cols].values + noise

    # Alter class distribution: варьируем степень дисбаланса
    undersample_frac = rng.uniform(0.3, 0.6)  # От 30% до 60% оставляем
    class_two = drift[drift["target"] == drift["target"].max()]
    others = drift[drift["target"] != drift["target"].max()]
    sampled_others = others.sample(frac=undersample_frac, random_state=seed)
    drifted = pd.concat([class_two, sampled_others], ignore_index=True)
    drifted = drifted.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    LOGGER.info(
        "Generated drifted dataset: %d samples, drift_magnitude=%.2f, noise_scale=%.2f, undersample=%.2f",
        len(drifted), drift_magnitude, noise_scale, undersample_frac
    )
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


def bootstrap_f1_confidence_interval(
    y_true: pd.Series, 
    y_pred: pd.Series, 
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95
) -> Dict[str, float]:
    """Вычисляет доверительный интервал для F1-score методом bootstrap.
    
    Returns:
        dict с mean, std, lower, upper границами доверительного интервала
    """
    rng = np.random.default_rng()
    n_samples = len(y_true)
    f1_scores = []
    
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        f1 = f1_score(y_true_arr[indices], y_pred_arr[indices], average="macro", zero_division=0)
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    alpha = 1 - confidence
    lower = np.percentile(f1_scores, alpha / 2 * 100)
    upper = np.percentile(f1_scores, (1 - alpha / 2) * 100)
    
    return {
        "mean": float(np.mean(f1_scores)),
        "std": float(np.std(f1_scores)),
        "lower": float(lower),
        "upper": float(upper),
    }


def test_statistical_significance(
    y_true: pd.Series,
    staging_preds: pd.Series,
    production_preds: pd.Series,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
) -> Dict[str, Any]:
    """Проверяет статистическую значимость различия между моделями.
    
    Использует paired bootstrap test для сравнения F1-scores.
    
    Returns:
        dict с результатами теста: p_value, is_significant, delta_mean, delta_ci
    """
    rng = np.random.default_rng()
    n_samples = len(y_true)
    
    y_true_arr = np.array(y_true)
    staging_arr = np.array(staging_preds)
    production_arr = np.array(production_preds)
    
    # Bootstrap разницы F1-scores
    delta_scores = []
    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        f1_staging = f1_score(y_true_arr[indices], staging_arr[indices], average="macro", zero_division=0)
        f1_production = f1_score(y_true_arr[indices], production_arr[indices], average="macro", zero_division=0)
        delta_scores.append(f1_staging - f1_production)
    
    delta_scores = np.array(delta_scores)
    delta_mean = np.mean(delta_scores)
    delta_std = np.std(delta_scores)
    
    # P-value: доля случаев, когда разница <= 0 (H0: staging не лучше)
    p_value = np.mean(delta_scores <= 0)
    
    # Доверительный интервал для разницы
    lower = np.percentile(delta_scores, 2.5)
    upper = np.percentile(delta_scores, 97.5)
    
    is_significant = p_value < STATISTICAL_SIGNIFICANCE_ALPHA and lower > 0
    
    LOGGER.info(
        "Statistical test: delta_mean=%.4f, p_value=%.4f, CI=[%.4f, %.4f], significant=%s",
        delta_mean, p_value, lower, upper, is_significant
    )
    
    return {
        "p_value": float(p_value),
        "is_significant": is_significant,
        "alpha": STATISTICAL_SIGNIFICANCE_ALPHA,
        "delta_mean": float(delta_mean),
        "delta_std": float(delta_std),
        "delta_ci_lower": float(lower),
        "delta_ci_upper": float(upper),
        "n_bootstrap_iterations": n_iterations,
    }


def run_ab_test_on_drifted_data(**context) -> Dict[str, Any]:
    """Прогоняет данные с дрифтом через Flask API для A/B тестирования.
    
    Отправляет запросы к /predict эндпоинту и собирает статистику
    по распределению между Production и Staging моделями.
    """
    _, drifted = load_current_data()
    features_df = drifted.drop(columns=["target"])
    targets = drifted["target"].tolist()
    
    # Формируем список признаков для API
    feature_names = features_df.columns.tolist()
    
    results = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "production_requests": 0,
        "staging_requests": 0,
        "production_predictions": [],
        "staging_predictions": [],
        "production_actuals": [],
        "staging_actuals": [],
        "errors": [],
    }
    
    LOGGER.info("Starting A/B test with %d samples from drifted data", len(features_df))
    
    for idx, row in features_df.iterrows():
        features = {name: float(row[name]) for name in feature_names}
        actual = targets[idx] if idx < len(targets) else None
        
        try:
            response = requests.post(
                f"{FLASK_API_URL}/predict",
                json={"features": features, "user_id": idx},
                timeout=5
            )
            results["total_requests"] += 1
            
            if response.status_code == 200:
                data = response.json()
                results["successful_requests"] += 1
                stage = data.get("stage", "").lower()
                prediction = data.get("prediction")
                
                if stage == "production":
                    results["production_requests"] += 1
                    results["production_predictions"].append(prediction)
                    results["production_actuals"].append(actual)
                elif stage == "staging":
                    results["staging_requests"] += 1
                    results["staging_predictions"].append(prediction)
                    results["staging_actuals"].append(actual)
            else:
                results["failed_requests"] += 1
                results["errors"].append(f"HTTP {response.status_code}: {response.text[:100]}")
                
        except requests.RequestException as exc:
            results["failed_requests"] += 1
            results["total_requests"] += 1
            if len(results["errors"]) < 10:  # Limit error storage
                results["errors"].append(str(exc)[:100])
    
    # Вычисляем метрики для каждого варианта
    ab_metrics = {"production": None, "staging": None}
    
    for variant in ["production", "staging"]:
        preds = results[f"{variant}_predictions"]
        actuals = results[f"{variant}_actuals"]
        
        if preds and actuals and len(preds) >= 10:
            # Фильтруем None значения
            valid = [(p, a) for p, a in zip(preds, actuals) if p is not None and a is not None]
            if valid:
                preds_clean, actuals_clean = zip(*valid)
                preds_arr = np.array(preds_clean).astype(int)
                actuals_arr = np.array(actuals_clean).astype(int)
                
                ab_metrics[variant] = {
                    "accuracy": float(accuracy_score(actuals_arr, preds_arr)),
                    "f1_macro": float(f1_score(actuals_arr, preds_arr, average="macro", zero_division=0)),
                    "precision_macro": float(precision_score(actuals_arr, preds_arr, average="macro", zero_division=0)),
                    "recall_macro": float(recall_score(actuals_arr, preds_arr, average="macro", zero_division=0)),
                    "n_samples": len(preds_clean),
                }
    
    results["ab_metrics"] = ab_metrics
    
    # Вычисляем traffic share
    total = results["production_requests"] + results["staging_requests"]
    if total > 0:
        results["production_share"] = round(results["production_requests"] / total, 4)
        results["staging_share"] = round(results["staging_requests"] / total, 4)
    
    LOGGER.info(
        "A/B test completed: total=%d, production=%d (%.1f%%), staging=%d (%.1f%%)",
        results["total_requests"],
        results["production_requests"],
        results.get("production_share", 0) * 100,
        results["staging_requests"],
        results.get("staging_share", 0) * 100,
    )
    
    # Сохраняем в XCom
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="ab_test_results", value=results)
    
    return results


def evaluate_candidate(**context) -> Dict[str, Any]:
    """Оценивает Staging и Production модели на данных с дрифтом.
    
    Сравнение проводится именно на drifted данных, чтобы определить,
    какая модель лучше справляется с изменившимся распределением.
    
    Для принятия решения о promotion используется статистическая проверка:
    Staging модель продвигается только если её преимущество статистически значимо.
    """
    ti = context.get("ti")
    train_info: Dict[str, Any] = ti.xcom_pull(task_ids="train_model") if ti else {}
    registration = train_info.get("registration") or {}
    model_name = registration.get("model_name") or os.getenv("MLFLOW_REGISTERED_NAME", "iris_pycaret_model")

    client = MlflowClient()
    # Оцениваем модели на данных с дрифтом!
    features, target = prepare_eval_split_on_drifted_data()
    evaluation: Dict[str, Any] = {"model_name": model_name, "eval_data": "drifted"}
    
    # Сохраняем предсказания для статистического теста
    predictions_cache: Dict[str, pd.Series] = {}

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
            predictions_cache[stage] = preds  # Сохраняем для статистического теста
            
            # Вычисляем bootstrap confidence interval для F1
            f1_ci = bootstrap_f1_confidence_interval(target, preds)
            
            metrics_payload = {
                "accuracy": float(accuracy_score(target, preds)),
                "precision_macro": float(precision_score(target, preds, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(target, preds, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(target, preds, average="macro", zero_division=0)),
                "f1_confidence_interval": f1_ci,
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
    statistical_test = None
    
    if staging_info is None:
        LOGGER.warning("No staging model available for evaluation")
        evaluation["promotion_reason"] = "no_staging_model"
    elif staging_info.get("error"):
        LOGGER.warning("Staging model load/eval failed: %s", staging_info.get("error"))
        evaluation["promotion_reason"] = "staging_eval_error"
    elif production_info is None:
        # Нет Production модели — продвигаем Staging без теста
        promote = True
        evaluation["promotion_reason"] = "no_production_model"
        LOGGER.info("No production model exists, promoting staging by default")
    else:
        # Обе модели существуют — проводим статистический тест
        if "Staging" in predictions_cache and "Production" in predictions_cache:
            statistical_test = test_statistical_significance(
                target,
                predictions_cache["Staging"],
                predictions_cache["Production"],
            )
            evaluation["statistical_test"] = statistical_test
            
            if statistical_test["is_significant"]:
                promote = True
                evaluation["promotion_reason"] = "statistically_significant_improvement"
                LOGGER.info(
                    "Staging model is statistically better: delta=%.4f, p=%.4f",
                    statistical_test["delta_mean"],
                    statistical_test["p_value"]
                )
            else:
                promote = False
                staging_f1 = staging_info["metrics"].get("f1_macro", 0)
                prod_f1 = production_info["metrics"].get("f1_macro", 0)
                
                if staging_f1 > prod_f1:
                    evaluation["promotion_reason"] = "improvement_not_statistically_significant"
                    LOGGER.info(
                        "Staging F1 (%.4f) > Production F1 (%.4f) but NOT statistically significant (p=%.4f)",
                        staging_f1, prod_f1, statistical_test["p_value"]
                    )
                else:
                    evaluation["promotion_reason"] = "staging_not_better"
                    LOGGER.info("Staging model is not better than production")
        else:
            LOGGER.warning("Missing predictions for statistical test")
            evaluation["promotion_reason"] = "missing_predictions"

    evaluation["promote"] = promote
    LOGGER.info("Evaluation result: promote=%s, reason=%s", promote, evaluation.get("promotion_reason"))
    
    if ti is not None:
        ti.xcom_push(key="evaluation", value=evaluation)
    return evaluation


def promote_if_better(**context) -> Dict[str, Any]:
    """Продвигает модель в Production если статистический тест пройден.
    """
    ti = context.get("ti")
    evaluation = ti.xcom_pull(task_ids="evaluate_candidate", key="evaluation") if ti else {}
    train_info = ti.xcom_pull(task_ids="train_model") if ti else {}
    registration = train_info.get("registration") or {}
    model_name = evaluation.get("model_name") or registration.get("model_name")
    
    result = {
        "promoted": False,
        "model_name": model_name,
        "evaluation_passed": False,
        "reason": None,
    }
    
    # Проверка: Статистический тест (офлайн оценка)
    if not evaluation or not evaluation.get("promote"):
        result["reason"] = "statistical_test_failed"
        LOGGER.info("Promotion blocked: statistical test not passed. Evaluation: %s", evaluation)
        ti.xcom_push(key="promotion_result", value=result)
        return result
    
    result["evaluation_passed"] = True
    LOGGER.info("Statistical test passed for staging model")
    
    # Статистический тест пройден — продвигаем модель
    staging_info = evaluation.get("staging") or {}
    staging_version = staging_info.get("version") or registration.get("staging_version")
    
    if staging_version is None:
        result["reason"] = "missing_staging_version"
        LOGGER.warning("Staging version not found for promotion")
        ti.xcom_push(key="promotion_result", value=result)
        return result

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version,
        stage="Production",
        archive_existing_versions=True,
    )
    
    result["promoted"] = True
    result["version"] = staging_version
    result["reason"] = "statistical_test_passed"
    
    LOGGER.info(
        "✅ Model %s version %s promoted to Production (stat_test=PASS)",
        model_name, staging_version
    )
    
    ti.xcom_push(key="promotion_result", value=result)
    return result


def generate_pipeline_report(**context) -> Dict[str, Any]:
    """Генерирует итоговый отчёт о запуске пайплайна переобучения.
    
    Отчёт включает:
    - Информацию о дрифте данных
    - Метрики моделей (Production vs Staging)
    - Результаты A/B тестирования
    - Статус модели (продвинута ли в Production)
    """
    ensure_dirs()
    ti = context.get("ti")
    run_id = context.get("run_id", str(uuid.uuid4())[:8])
    timestamp = datetime.now().isoformat()

    report: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dag_id": context.get("dag").dag_id if context.get("dag") else "mlops_monitoring_pipeline",
        "execution_date": str(context.get("execution_date", "")),
    }

    # 1. Drift information
    drift_detected = ti.xcom_pull(task_ids="check_data_drift", key="dataset_drift") if ti else None
    drift_payload = ti.xcom_pull(task_ids="check_data_drift", key="dataset_drift_payload") if ti else {}
    
    report["drift"] = {
        "detected": drift_detected,
        "share_of_drifted_columns": drift_payload.get("share_of_drifted_columns"),
        "number_of_drifted_columns": drift_payload.get("number_of_drifted_columns"),
        "number_of_columns": drift_payload.get("number_of_columns"),
        "drift_by_columns": drift_payload.get("drift_by_columns", {}),
        "report_path": drift_payload.get("report_path"),
    }

    # 2. Training information
    train_info = ti.xcom_pull(task_ids="train_model") if ti else None
    if train_info:
        report["training"] = {
            "executed": True,
            "model_name": train_info.get("registration", {}).get("model_name"),
            "model_version": train_info.get("registration", {}).get("staging_version"),
            "run_id": train_info.get("registration", {}).get("run_id"),
            "best_model": train_info.get("best_model"),
            "metrics": train_info.get("metrics"),
        }
    else:
        report["training"] = {"executed": False, "reason": "No drift detected or training skipped"}

    # 3. Evaluation metrics (Production vs Staging)
    evaluation = ti.xcom_pull(task_ids="evaluate_candidate", key="evaluation") if ti else None
    if evaluation:
        report["evaluation"] = {
            "production": evaluation.get("production"),
            "staging": evaluation.get("staging"),
            "evaluated_on": evaluation.get("eval_data", "drifted"),
            "recommendation": "promote" if evaluation.get("promote") else "keep_current",
            "promotion_reason": evaluation.get("promotion_reason"),
        }
        # Добавляем результаты статистического теста
        if evaluation.get("statistical_test"):
            report["statistical_test"] = evaluation["statistical_test"]
    else:
        report["evaluation"] = None
        report["statistical_test"] = None

    # 4. Promotion result
    promotion_result = ti.xcom_pull(task_ids="promote_if_better", key="promotion_result") if ti else None
    if promotion_result:
        report["promotion"] = {
            "promoted": promotion_result.get("promoted", False),
            "model_name": promotion_result.get("model_name"),
            "version": promotion_result.get("version"),
            "new_stage": "Production" if promotion_result.get("promoted") else None,
        }
    else:
        report["promotion"] = {"promoted": False, "reason": "No promotion performed"}

    # 5. Current model status in registry
    client = MlflowClient()
    model_name = os.getenv("MLFLOW_REGISTERED_NAME", "iris_pycaret_model")
    report["model_registry"] = {"model_name": model_name}
    
    for stage in ["Production", "Staging"]:
        try:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                v = versions[0]
                report["model_registry"][stage.lower()] = {
                    "version": v.version,
                    "run_id": v.run_id,
                    "status": v.status,
                    "creation_timestamp": v.creation_timestamp,
                }
        except MlflowException:
            report["model_registry"][stage.lower()] = None

    # 6. Summary and recommendations
    summary_lines = []
    if drift_detected:
        summary_lines.append("✓ Data drift detected - retraining triggered")
    else:
        summary_lines.append("✗ No data drift detected")
    
    if train_info:
        summary_lines.append(f"✓ Model retrained: {train_info.get('best_model', 'N/A')}")
    
    if evaluation and evaluation.get("staging"):
        staging_metrics = evaluation.get("staging", {}).get("metrics", {})
        prod_metrics = evaluation.get("production", {}).get("metrics", {})
        if staging_metrics and prod_metrics:
            staging_f1 = staging_metrics.get("f1_macro", 0)
            prod_f1 = prod_metrics.get("f1_macro", 0)
            delta = staging_f1 - prod_f1
            summary_lines.append(
                f"✓ Offline Evaluation: Staging F1={staging_f1:.4f}, Production F1={prod_f1:.4f}, Δ={delta:+.4f}"
            )
    
    # Статистический тест
    stat_test = report.get("statistical_test")
    if stat_test:
        if stat_test.get("is_significant"):
            summary_lines.append(
                f"✓ Statistical test: PASSED (p={stat_test['p_value']:.4f} < α={stat_test['alpha']})"
            )
        else:
            summary_lines.append(
                f"✗ Statistical test: FAILED (p={stat_test['p_value']:.4f} >= α={stat_test['alpha']})"
            )
    
    # Итоговое решение о promotion
    if promotion_result:
        if promotion_result.get("promoted"):
            summary_lines.append(f"🚀 Model v{promotion_result.get('version')} PROMOTED to Production!")
        else:
            reason = promotion_result.get("reason", "unknown")
            if reason == "statistical_test_failed":
                summary_lines.append("❌ Promotion blocked: Statistical test not passed")
            else:
                summary_lines.append(f"❌ Promotion blocked: {reason}")
    
    report["summary"] = summary_lines

    # Save report to JSON file
    report_filename = f"report_{timestamp.replace(':', '-').replace('.', '-')}.json"
    report_path = REPORTS_DIR / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    LOGGER.info("Pipeline report saved to %s", report_path)
    report["report_file"] = str(report_path)
    
    # Also save as HTML for easy viewing
    html_report = _generate_html_report(report)
    html_path = REPORTS_DIR / f"report_{timestamp.replace(':', '-').replace('.', '-')}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    
    LOGGER.info("HTML report saved to %s", html_path)
    return report


def _generate_html_report(report: Dict[str, Any]) -> str:
    """Генерирует HTML версию отчёта для удобного просмотра."""
    
    drift = report.get("drift", {})
    training = report.get("training", {})
    evaluation = report.get("evaluation", {})
    promotion = report.get("promotion", {})
    registry = report.get("model_registry", {})
    summary = report.get("summary", [])
    stat_test = report.get("statistical_test", {})
    
    drift_status = "🔴 Detected" if drift.get("detected") else "🟢 Not Detected"
    promoted_status = "🟢 Yes" if promotion.get("promoted") else "🔴 No"
    stat_status = "🟢 Significant" if stat_test and stat_test.get("is_significant") else "🔴 Not Significant"
    
    # Metrics tables
    eval_html = ""
    if evaluation:
        staging = evaluation.get("staging", {})
        prod = evaluation.get("production", {})
        if staging and prod:
            staging_m = staging.get("metrics", {})
            prod_m = prod.get("metrics", {})
            
            # Safe extraction with defaults
            s_acc = staging_m.get('accuracy') or 0
            p_acc = prod_m.get('accuracy') or 0
            s_prec = staging_m.get('precision_macro') or 0
            p_prec = prod_m.get('precision_macro') or 0
            s_rec = staging_m.get('recall_macro') or 0
            p_rec = prod_m.get('recall_macro') or 0
            s_f1 = staging_m.get('f1_macro') or 0
            p_f1 = prod_m.get('f1_macro') or 0
            
            eval_html = f"""
            <h2>📊 Model Evaluation (on drifted data)</h2>
            <table>
                <tr><th>Metric</th><th>Production (v{prod.get('version', 'N/A')})</th><th>Staging (v{staging.get('version', 'N/A')})</th><th>Δ</th></tr>
                <tr><td>Accuracy</td><td>{p_acc:.4f}</td><td>{s_acc:.4f}</td><td>{s_acc - p_acc:+.4f}</td></tr>
                <tr><td>Precision (macro)</td><td>{p_prec:.4f}</td><td>{s_prec:.4f}</td><td>{s_prec - p_prec:+.4f}</td></tr>
                <tr><td>Recall (macro)</td><td>{p_rec:.4f}</td><td>{s_rec:.4f}</td><td>{s_rec - p_rec:+.4f}</td></tr>
                <tr><td>F1 (macro)</td><td>{p_f1:.4f}</td><td>{s_f1:.4f}</td><td>{s_f1 - p_f1:+.4f}</td></tr>
            </table>
            """
    
    # Statistical test section
    stat_test_html = ""
    if stat_test:
        p_value = stat_test.get('p_value') or 0
        alpha = stat_test.get('alpha') or 0.05
        delta_mean = stat_test.get('delta_mean') or 0
        delta_ci_lower = stat_test.get('delta_ci_lower') or 0
        delta_ci_upper = stat_test.get('delta_ci_upper') or 0
        n_bootstrap = stat_test.get('n_bootstrap_iterations', 'N/A')
        
        stat_test_html = f"""
        <h2>📐 Statistical Significance Test</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Result</td><td><strong>{stat_status}</strong></td></tr>
            <tr><td>p-value</td><td>{p_value:.4f}</td></tr>
            <tr><td>α (threshold)</td><td>{alpha}</td></tr>
            <tr><td>Mean Δ F1 (Staging - Production)</td><td>{delta_mean:+.4f}</td></tr>
            <tr><td>95% CI for Δ</td><td>[{delta_ci_lower:.4f}, {delta_ci_upper:.4f}]</td></tr>
            <tr><td>Bootstrap iterations</td><td>{n_bootstrap}</td></tr>
        </table>
        <p style="color: #666; font-size: #666; font-size: 14px;">
            <em>Модель продвигается в Production только если p-value &lt; α И нижняя граница 95% CI &gt; 0</em>
        </p>
        """
    
    training_html = ""
    if training.get("executed"):
        training_html = f"""
        <h2>🏋️ Training Results</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Best Model</td><td>{training.get('best_model', 'N/A')}</td></tr>
            <tr><td>Model Name</td><td>{training.get('model_name', 'N/A')}</td></tr>
            <tr><td>Version</td><td>{training.get('model_version', 'N/A')}</td></tr>
            <tr><td>MLflow Run ID</td><td>{training.get('run_id', 'N/A')}</td></tr>
        </table>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>MLOps Pipeline Report - {report.get('timestamp', '')}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f8f9fa; font-weight: 600; }}
            tr:hover {{ background: #f8f9fa; }}
            .summary {{ background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .summary li {{ margin: 8px 0; font-size: 16px; }}
            .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: 600; }}
            .badge-success {{ background: #d4edda; color: #155724; }}
            .badge-danger {{ background: #f8d7da; color: #721c24; }}
            .badge-info {{ background: #d1ecf1; color: #0c5460; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 22px; font-weight: bold; color: #007bff; }}
            .metric-label {{ color: #666; margin-top: 5px; font-size: 13px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📋 MLOps Pipeline Report</h1>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{report.get('run_id', 'N/A')}</div>
                    <div class="metric-label">Run ID</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{drift_status}</div>
                    <div class="metric-label">Data Drift</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{stat_status if stat_test else 'N/A'}</div>
                    <div class="metric-label">Statistical Test</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{promoted_status}</div>
                    <div class="metric-label">Model Promoted</div>
                </div>
            </div>
            
            <p><strong>Timestamp:</strong> {report.get('timestamp', 'N/A')}</p>
            <p><strong>DAG:</strong> {report.get('dag_id', 'N/A')}</p>
            
            <h2>📝 Summary</h2>
            <div class="summary">
                <ul>
                    {''.join(f'<li>{line}</li>' for line in summary)}
                </ul>
            </div>
            
            <h2>📈 Data Drift Analysis</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Drift Detected</td><td>{drift_status}</td></tr>
                <tr><td>Drifted Columns</td><td>{drift.get('number_of_drifted_columns', 'N/A')} / {drift.get('number_of_columns', 'N/A')}</td></tr>
                <tr><td>Drift Share</td><td>{drift.get('share_of_drifted_columns', 'N/A')}</td></tr>
                <tr><td>Evidently Report</td><td><a href="{drift.get('report_path', '#')}">View Full Report</a></td></tr>
            </table>
            
            {training_html}
            
            {eval_html}
            
            {stat_test_html}
            
            <h2>🏷️ Model Registry Status</h2>
            <table>
                <tr><th>Stage</th><th>Version</th><th>Run ID</th></tr>
                <tr><td>Production</td><td>{registry.get('production', {}).get('version', 'N/A') if registry.get('production') else 'N/A'}</td><td>{registry.get('production', {}).get('run_id', 'N/A')[:8] if registry.get('production') else 'N/A'}...</td></tr>
                <tr><td>Staging</td><td>{registry.get('staging', {}).get('version', 'N/A') if registry.get('staging') else 'N/A'}</td><td>{registry.get('staging', {}).get('run_id', 'N/A')[:8] if registry.get('staging') else 'N/A'}...</td></tr>
            </table>
            
            <h2>🚀 Promotion Decision</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Promoted</td><td>{promoted_status}</td></tr>
                <tr><td>Reason</td><td>{evaluation.get('promotion_reason', 'N/A') if evaluation else 'N/A'}</td></tr>
                <tr><td>Model Name</td><td>{promotion.get('model_name', 'N/A')}</td></tr>
                <tr><td>Version</td><td>{promotion.get('version', 'N/A')}</td></tr>
                <tr><td>New Stage</td><td>{promotion.get('new_stage', 'N/A')}</td></tr>
            </table>
            
            <hr style="margin-top: 40px;">
            <p style="color: #666; font-size: 12px;">Generated by MLOps Pipeline • {report.get('timestamp', '')}</p>
        </div>
    </body>
    </html>
    """
    return html


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

    generate_report_task = PythonOperator(
        task_id="generate_report",
        python_callable=generate_pipeline_report,
        provide_context=True,
        trigger_rule="all_done",  # Run even if upstream tasks failed/skipped
    )

    pipeline_done = EmptyOperator(task_id="pipeline_finished")

    # Pipeline flow:
    # 1. Check drift
    # 2. If no drift: skip → report → done
    # 3. If drift: train → evaluate → promote (if stat test passes) → report → done
    check_drift >> branch
    branch >> skip_retraining >> generate_report_task >> pipeline_done
    branch >> train_model_task >> evaluate_task >> promote_task >> generate_report_task >> pipeline_done
