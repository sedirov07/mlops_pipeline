import logging
import os
import random
import threading
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

load_dotenv()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
REGISTRY_NAME = os.getenv("MLFLOW_REGISTERED_NAME", "iris_pycaret_model")

# Определяем корень проекта для корректных путей
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_FILE = os.path.join(PROJECT_ROOT, 'models', 'best_model_pycaret.pkl')

# Настраиваем путь к артефактам MLflow (для локального запуска)
# Артефакты хранятся в Docker volume, но смонтированы в ./mlflow_db/artifacts
MLFLOW_ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "mlflow_db", "artifacts")
if os.path.exists(MLFLOW_ARTIFACTS_PATH):
    os.environ["MLFLOW_ARTIFACT_URI"] = f"file://{MLFLOW_ARTIFACTS_PATH}"

mlflow.set_tracking_uri(TRACKING_URI)

app = Flask(__name__)

os.makedirs("logs", exist_ok=True)
router_logger = logging.getLogger("ab_router")
if not router_logger.handlers:
    router_logger.setLevel(logging.INFO)
    handler = logging.FileHandler("logs/predictions.log")
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    router_logger.addHandler(handler)

client = MlflowClient()
traffic_lock = threading.Lock()
traffic_split = {"production": 0.5, "staging": 0.5}


def _clamp_share(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def set_traffic_split(production_share: Any) -> Dict[str, float]:
    share = round(_clamp_share(production_share), 4)
    with traffic_lock:
        traffic_split["production"] = share
        traffic_split["staging"] = round(1 - share, 4)
        return dict(traffic_split)


set_traffic_split(os.getenv("AB_TRAFFIC_PRODUCTION", 0.5))


def load_local_model() -> Optional[Any]:
    if os.path.exists(LOCAL_MODEL_FILE):
        return joblib.load(LOCAL_MODEL_FILE)
    return None


def load_model_from_registry(stage: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        versions = client.get_latest_versions(REGISTRY_NAME, stages=[stage])
    except MlflowException as exc:
        router_logger.warning("Registry lookup failed for %s stage %s: %s", REGISTRY_NAME, stage, exc)
        return None, None

    if not versions:
        router_logger.info("No model versions found for %s in %s stage", REGISTRY_NAME, stage)
        return None, None

    version = versions[0].version
    run_id = versions[0].run_id
    source = versions[0].source  # путь к артефакту в MLflow
    
    # Для локального запуска: преобразуем Docker путь в локальный
    # Docker: /mlflow/artifacts/1/<run_id>/artifacts/model
    # Local:  ./mlflow_db/artifacts/1/<run_id>/artifacts/model
    if source.startswith("/mlflow/artifacts"):
        local_source = source.replace("/mlflow/artifacts", MLFLOW_ARTIFACTS_PATH)
        if os.path.exists(local_source):
            try:
                model = mlflow.pyfunc.load_model(local_source)
                router_logger.info("Loaded model %s version %s from %s stage (local artifacts path)", 
                                  REGISTRY_NAME, version, stage)
                return model, version
            except Exception as exc:
                router_logger.warning("Failed to load from local path %s: %s", local_source, exc)
    
    # Пробуем загрузить модель через runs:/ URI
    model_uri = f"runs:/{run_id}/model"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        router_logger.info("Loaded model %s version %s from %s stage (run_id=%s)", 
                          REGISTRY_NAME, version, stage, run_id)
        return model, version
    except Exception as exc:
        router_logger.warning("Failed to load model %s stage %s via runs URI: %s", REGISTRY_NAME, stage, exc)
        
    # Fallback: пробуем через models:/ URI
    model_uri = f"models:/{REGISTRY_NAME}/{stage}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        router_logger.info("Loaded model %s version %s from %s stage (models URI)", 
                          REGISTRY_NAME, version, stage)
        return model, version
    except Exception as exc:
        router_logger.warning("Failed to load model %s stage %s via models URI: %s", REGISTRY_NAME, stage, exc)
        return None, version


def choose_stage(user_id: Optional[int] = None) -> str:
    with traffic_lock:
        production_share = traffic_split["production"]
    if user_id is not None:
        try:
            bucket = abs(int(user_id)) % 100
        except (TypeError, ValueError):
            bucket = random.randint(0, 99)
        return 'Production' if bucket < int(production_share * 100) else 'Staging'
    return 'Production' if random.random() < production_share else 'Staging'


def log_inference(variant: str, stage: str, version: Any, features: Any, prediction: Any, user_id: Any):
    router_logger.info(
        "variant=%s stage=%s version=%s user_id=%s features=%s prediction=%s",
        variant,
        stage,
        version,
        user_id,
        features,
        prediction,
    )


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/config/traffic', methods=['GET', 'POST'])
def traffic_config():
    if request.method == 'GET':
        with traffic_lock:
            return jsonify(dict(traffic_split))

    payload = request.get_json(silent=True) or {}
    production_share = payload.get('production')
    if production_share is None:
        return jsonify({'error': 'production share is required'}), 400

    updated = set_traffic_split(production_share)
    router_logger.info("Updated traffic split: %s", updated)
    return jsonify(updated)


@app.route('/models/promote', methods=['POST'])
def promote_model():
    payload = request.get_json(silent=True) or {}
    version = payload.get('version')
    source_stage = payload.get('from_stage', 'Staging')
    target_stage = payload.get('to_stage', 'Production')

    try:
        if version is None:
            candidates = client.get_latest_versions(REGISTRY_NAME, stages=[source_stage])
            if not candidates:
                return jsonify({'error': f'No versions in {source_stage} stage'}), 404
            version = candidates[0].version

        client.transition_model_version_stage(
            name=REGISTRY_NAME,
            version=version,
            stage=target_stage,
            archive_existing_versions=(target_stage == 'Production'),
        )
        message = {
            'message': f'version {version} moved to {target_stage}',
            'version': version,
            'stage': target_stage,
        }
        router_logger.info("Model promotion: %s", message)
        return jsonify(message)
    except MlflowException as exc:
        router_logger.exception("Failed to promote model: %s", exc)
        return jsonify({'error': str(exc)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}
    features = payload.get('features')
    if features is None:
        return jsonify({'error': 'no features provided'}), 400

    user_id = payload.get('user_id')
    stage = choose_stage(user_id)
    variant = 'A' if stage == 'Production' else 'B'

    model, version = load_model_from_registry(stage)
    active_stage = stage

    # If staging model does not exist, route to production
    if model is None and stage == 'Staging':
        model, version = load_model_from_registry('Production')
        active_stage = 'Production'
        variant = 'A'

    if model is None:
        model = load_local_model()
        active_stage = 'Local'
        version = 'local'

    if model is None:
        return jsonify({'error': 'model not found'}), 500

    features_df = pd.DataFrame([features])
    prediction_raw = model.predict(features_df)
    # Безопасное извлечение первого элемента (избегаем DeprecationWarning)
    if hasattr(prediction_raw, 'item'):
        prediction_value = prediction_raw[0].item() if hasattr(prediction_raw[0], 'item') else prediction_raw[0]
    else:
        prediction_value = prediction_raw[0]
    
    # Приводим к числу если возможно
    try:
        prediction_value = float(prediction_value)
    except (TypeError, ValueError):
        pass

    log_inference(variant, active_stage, version, features, prediction_value, user_id)
    response = {
        'variant': variant,
        'stage': active_stage,
        'version': str(version),
        'prediction': prediction_value,
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
