# MLOps classification pipeline

End-to-end **MLOps** demo for **binary classification**: data ingestion → cleaning → training → evaluation → **MLflow** logging and model registry → **FastAPI** serving → optional **Docker** deployment. Uses **DVC** for reproducible pipeline stages and **scikit-learn** for modeling.

**Stack:** DVC · MLflow · FastAPI · Docker · scikit-learn · pandas

---

## What this project does

| Stage | What happens |
|--------|----------------|
| Ingestion | Load train/test CSVs (from `config.yml`). |
| Cleaning | Handle missing values; ensure numeric target. |
| Training | Sklearn **pipeline** (e.g. scaling, one-hot encoding) + classifier from config. |
| Evaluation | Accuracy, ROC AUC, classification report. |
| MLflow | Logs params, metrics, artifacts; registers model **`insurance_model`**. |
| API | Loads `models/model.pkl` and serves predictions at `POST /predict`. |

**Outputs (local, after `python main.py`):**

- `models/model.pkl` — saved pipeline used by the API.
- `mlruns/` — MLflow file store (see note below; not committed to Git).

---

## Repository

```bash
git clone https://github.com/nithineleti/mlops_project.git
cd mlops_project
```

---

## Prerequisites

- Python 3.10+ recommended (Dockerfile uses Python 3.10).
- For DVC pipeline runs: [DVC](https://dvc.org/) installed (`pip install dvc` or use `requirements.txt`).

---

## Installation

Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the training pipeline

From the project root:

```bash
python3 main.py
```

This runs ingestion → cleaning → training → evaluation, logs to MLflow under `./mlruns`, and registers **`insurance_model`** in the local model registry.

### MLflow UI

After at least one run:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to browse experiments, metrics, and registered models.

**Note:** The `mlruns/` directory is listed in `.gitignore`. Clone the repo, then run training locally to generate MLflow data. Do not commit large run artifacts to Git.

---

## Run with DVC

```bash
dvc repro
dvc dag
```

Stages typically follow: data ingestion → data cleaning → model training (see `dvc.yaml`).

---

## Model serving (FastAPI)

Train first so `models/model.pkl` exists, then:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

- Health: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### API request body (`POST /predict`)

Fields must match `app.py` (order does not matter in JSON):

```json
{
  "Gender": "Male",
  "Age": 35,
  "HasDrivingLicense": 1,
  "RegionID": 28.0,
  "Switch": 0,
  "PastAccident": "No",
  "AnnualPremium": 25000.0
}
```

Example response:

```json
{
  "predicted_class": 1
}
```

---

## Docker

This repo’s image definition is named **`dockerfile`** (lowercase). Build and run:

```bash
docker build -t mlops-classification -f dockerfile .
docker run -p 8000:8000 mlops-classification
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Ensure `models/model.pkl` is present in the build context (the Dockerfile copies `models/` into the image).

---

## Configuration (`config.yml`)

Central place for data paths, target column, model class name, and hyperparameters.

Supported model names (see `steps/train.py` for the full list) include:

- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`

Example shape:

```yaml
data:
  train_path: data/train.csv
  test_path: data/test.csv
  target_column: target

model:
  name: DecisionTreeClassifier
  params:
    criterion: entropy
    max_depth: null
  store_path: models/
```

---

## Tests

```bash
python3 -m pytest tests/ -v
```

---

## Project layout

```
.
├── config.yml          # Paths, model name, hyperparameters
├── main.py             # MLflow training entrypoint
├── app.py              # FastAPI app
├── dataset.py          # Synthetic dataset generation
├── dvc.yaml / dvc.lock # DVC pipeline
├── dockerfile          # API container (lowercase filename)
├── requirements.txt
├── steps/              # ingest, clean, train, predict
├── data/               # Train/test CSVs (not all may be in Git; see .gitignore)
├── models/             # Trained model.pkl (artifact; often local)
├── mlruns/             # MLflow file store (gitignored)
├── tests/
├── monitor.ipynb
├── production_drift.html
├── test_drift.html
└── README.md
```

---

## Data drift

HTML reports `production_drift.html` and `test_drift.html` illustrate comparing distributions between datasets (open in a browser).

---

## Example metrics

Runs are stochastic if data is regenerated; typical ballpark:

- Accuracy ≈ 0.92  
- ROC AUC ≈ 0.92  

See MLflow or the printed classification report after `python3 main.py`.

---

## DVC remote storage (optional)

```bash
dvc remote add -d storage s3://my-bucket/dvc-storage
dvc push
```

Supports S3, GCS, Azure, SSH, and other [DVC remotes](https://dvc.org/doc/command-reference/remote).

---

## Author

**Nithin Pious** — AI/ML engineer · MLOps
