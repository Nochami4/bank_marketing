# Bank Marketing (UCI id=222)

pet‑проект, где учим модель предсказывать, подпишется ли клиент на депозит, и поднимаем простой inference API.

Датасет: UCI Bank Marketing (https://archive.ics.uci.edu/dataset/222/bank+marketing)

## Setup
```bash
python -m venv .venv
# Linux/WSL
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
python src/train.py
```
Альтернативная модель:
Скрипт печатает метрики и сохраняет модель в `models/model.joblib`.

### API
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Пример запроса:
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 42,
    "job": "services",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1200,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day_of_week": 1,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }'
```

Ответ:
```json
{"proba": 0.1234, "label": 0}
```

## Metrics
- **ROC‑AUC**: качество ранжирования
- **PR‑AUC (Average Precision)**: качество на положительном классе
- Считаются на Stratified K‑Fold и на holdout 20%
