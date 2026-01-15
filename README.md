# MLOps Project: Автоматизированный ML-пайплайн с мониторингом дрифта

## Описание проекта

Реализован полноценный MLOps-пайплайн для автоматизации жизненного цикла ML-модели: от обучения до мониторинга и автоматического переобучения при обнаружении дрифта данных. Включает Flask-приложение для inference с A/B тестированием и систему отчётности.

## Архитектура

```
┌──────────────────┐     ┌──────────────────────┐      ┌──────────────────┐
│   Apache Airflow │────▶│   MLflow Server      │◀────│   PyCaret AutoML │
│   (Orchestration)│     │   (Tracking/Registry)│      │   (Training)     │
└────────┬─────────┘     └──────────────────────┘      └──────────────────┘
         │                         ▲
         ▼                         │
┌────────────────────┐    ┌────────────────────┐
│   Evidently AI     │    │   Flask App        │
│   (Drift Detection)│    │   (A/B Testing)    │
└────────────────────┘    └────────────────────┘
```

## Реализованный функционал

### 1. AutoML с PyCaret
- Автоматический подбор лучшей модели из множества алгоритмов
- Обучение на датасете Iris (классификация)
- Расчет метрик: accuracy, precision, recall, F1-score (macro)

### 2. Трекинг экспериментов (MLflow)
- Логирование всех метрик и параметров
- Версионирование моделей в Model Registry
- Стейджинг моделей: Staging → Production

### 3. Оркестрация (Apache Airflow)
- DAG `mlops_monitoring_pipeline` запускается ежедневно (6:00 UTC)
- Автоматическая проверка дрифта данных
- Условное переобучение при обнаружении дрифта

### 4. Мониторинг дрифта (Evidently)
- Сравнение текущих данных с baseline (reference) распределением
- Генерация HTML-отчетов о дрифте
- Детекция смещения признаков и целевой переменной

### 5. Автоматическое переобучение
При обнаружении дрифта:
1. **Формируется augmented датасет** — объединение baseline данных и новых данных с дрифтом
2. Модель переобучается на расширенных данных для адаптации к изменившемуся распределению
3. Новая модель регистрируется в MLflow как Staging
4. **Сравнение моделей проводится на данных с дрифтом** — Production и Staging модели оцениваются именно на новых данных, чтобы определить, какая лучше справляется с изменившимся распределением
5. При **статистически значимом** улучшении F1-score — автоматическое продвижение в Production

### 6. Статистическая проверка улучшения
- **Paired Bootstrap Test** для сравнения F1-scores моделей
- Promotion только при p-value < 0.05 **И** нижняя граница 95% CI > 0
- Предотвращает случайное продвижение моделей с незначительным улучшением

### 7. Flask-приложение с A/B тестированием
- Inference API с поддержкой A/B тестирования между Production и Staging моделями
- Настраиваемое распределение трафика между вариантами
- Сбор статистики по вариантам в реальном времени
- API для управления моделями и получения отчётов
- Легковесный REST API на базе Flask

### 8. Система отчётности
После каждого запуска пайплайна автоматически генерируется отчёт:
- **JSON формат** — для программного доступа через API
- **HTML формат** — для визуального просмотра
- Включает: статус дрифта, метрики моделей, статистический тест, результаты A/B тестирования

### 9. Контейнеризация (Docker)
- `docker-compose.yml` для запуска всей инфраструктуры
- Сервисы: Airflow (webserver, scheduler), MLflow, PostgreSQL

## Пайплайн обработки дрифта

```
check_data_drift ─▶ branch_on_drift ─┬─▶ skip_retraining ─────────────────────────────────────────────────┬─▶ generate_report ─▶ done
                                     │                                                                     │
                                     └─▶ train_model ─▶ evaluate_candidate ─▶ run_ab_test ─▶ promote_if_better ─┘
```

### Двухэтапная валидация перед Production:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PROMOTION DECISION FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Stage 1: Offline Statistical Test                                              │
│  ─────────────────────────────────                                              │
│  • Bootstrap test на drifted данных                                             │
│  • p-value < 0.05 И нижняя граница 95% CI > 0                                  │
│  • ❌ FAIL → Promotion blocked                                                   │
│  • ✓ PASS → Continue to Stage 2                                                 │
│                                                                                 │
│  Stage 2: Online A/B Test via API                                               │
│  ─────────────────────────────────                                              │
│  • Drifted данные прогоняются через Flask /predict                             │
│  • Сравнение F1-score Production vs Staging                                     │
│  • ❌ Staging F1 < Production F1 → Promotion blocked                            │
│  • ✓ Staging F1 >= Production F1 → PROMOTE TO PRODUCTION 🚀                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Ключевые особенности:

1. **Динамическая генерация дрифта** — каждый запуск создаёт уникальный датасет с дрифтом (seed на основе времени)
2. **Вариативность дрифта** — разная степень смещения признаков и дисбаланса классов
3. **Статистическая значимость** — модель продвигается только если улучшение достоверно (bootstrap test)
4. **A/B тест через API** — после promotion данные с дрифтом прогоняются через Flask

**Важно:** При переобучении используется **augmented датасет**:
- Данные с дрифтом искусственно смещены (шум в признаках, изменение баланса классов)
- Объединяем baseline + drifted данные для адаптации модели
- Это предотвращает "катастрофическое забывание" старых паттернов

## Flask API

### Inference и A/B тестирование

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/predict` | POST | Получить предсказание (автоматически выбирает модель по A/B) |
| `/health` | GET | Проверка здоровья сервиса |
| `/config/traffic` | GET/POST | Получить/установить распределение трафика |

### Управление моделями

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/models/promote` | POST | Продвинуть модель из Staging в Production |
| `/models/status` | GET | Текущий статус моделей в реестре |

### Статистика A/B тестирования

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/stats/ab` | GET | Статистика запросов по вариантам |
| `/stats/ab/reset` | POST | Сбросить статистику A/B тестирования |

### Отчёты пайплайна

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/reports` | GET | Список всех отчётов |
| `/reports/latest` | GET | Последний отчёт |
| `/reports/<report_id>` | GET | Конкретный отчёт по ID |

### Примеры запросов

```bash
# Предсказание
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}'

# Установить 70% трафика на Production
curl -X POST http://localhost:8000/config/traffic \
  -H "Content-Type: application/json" \
  -d '{"production": 0.7}'

# Получить статистику A/B
curl http://localhost:8000/stats/ab

# Последний отчёт пайплайна
curl http://localhost:8000/reports/latest

# Статус моделей
curl http://localhost:8000/models/status
```

## Структура отчёта пайплайна

Каждый запуск пайплайна создаёт отчёт в `logs/pipeline_reports/`:

```json
{
  "run_id": "abc123",
  "timestamp": "2026-01-15T12:00:00",
  "drift": {
    "detected": true,
    "share_of_drifted_columns": 0.8,
    "number_of_drifted_columns": 4
  },
  "training": {
    "executed": true,
    "best_model": "LightGBM",
    "model_version": "5"
  },
  "evaluation": {
    "production": {"version": "4", "metrics": {"f1_macro": 0.92}},
    "staging": {"version": "5", "metrics": {"f1_macro": 0.95}},
    "promotion_reason": "statistically_significant_improvement"
  },
  "statistical_test": {
    "p_value": 0.023,
    "is_significant": true,
    "alpha": 0.05,
    "delta_mean": 0.031,
    "delta_ci_lower": 0.008,
    "delta_ci_upper": 0.054,
    "n_bootstrap_iterations": 1000
  },
  "promotion": {
    "promoted": true,
    "version": "5"
  },
  "ab_testing_api": {
    "total_requests": 100,
    "production_requests": 48,
    "staging_requests": 52,
    "production_metrics": {"f1_macro": 0.91},
    "staging_metrics": {"f1_macro": 0.94}
  },
  "summary": [
    "✓ Data drift detected - retraining triggered",
    "✓ Model retrained: LightGBM",
    "✓ Evaluation: Staging F1=0.9500, Production F1=0.9200, Δ=+0.0300",
    "✓ Model v5 promoted to Production"
  ]
}
```

## Структура проекта

```
├── app.py                   # Flask-приложение с A/B тестированием
├── dags/
│   └── mlops_pipeline.py    # Airflow DAG с полным пайплайном
├── src/
│   └── automl.py            # PyCaret AutoML + MLflow интеграция
├── logs/
│   ├── evidently_drift_report.html  # Отчеты о дрифте
│   ├── predictions.log              # Лог предсказаний
│   └── pipeline_reports/            # Отчёты пайплайна (JSON + HTML)
├── models/                  # Локальные модели
├── mlflow_db/               # MLflow данные и артефакты моделей
│   └── artifacts/           # Сохраненные модели
├── docker-compose.yml       # Инфраструктура
├── Dockerfile               # Образ контейнера
└── requirements.txt         # Зависимости
```

## Запуск

```bash
# Сборка и запуск контейнеров
docker-compose up -d

# Airflow UI: http://localhost:8080
# MLflow UI: http://localhost:5000
# Flask API: http://localhost:8000
```

## Технологии

- **Python 3.11**
- **PyCaret** — AutoML фреймворк
- **MLflow** — трекинг экспериментов и Model Registry
- **Apache Airflow** — оркестрация пайплайнов
- **Evidently AI** — мониторинг дрифта данных
- **Flask** — REST API для inference и A/B тестирования
- **Docker / Docker Compose** — контейнеризация
- **PostgreSQL** — хранение метаданных Airflow

## Автор

Седиров Арсен
Проект выполнен в рамках курса MLOps (3 семестр)
