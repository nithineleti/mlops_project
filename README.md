# MLOps Classification Pipeline

This project is an end-to-end Machine Learning Operations (MLOps) pipeline for a classification problem. It leverages **DVC** for data and pipeline tracking and **MLflow** for experiment tracking and model registry.

## 🚀 Features
* **Data Generation & Ingestion**: Generates synthetic classification data (`dataset.py`) to simulate real-world data feeds.
* **Data Cleaning**: Handles missing values and enforces proper data types (`steps/clean.py`).
* **Model Training & Preprocessing**: Utilizes `scikit-learn` pipelines for scaling (`StandardScaler`), encoding (`OneHotEncoder`), and classification (`DecisionTreeClassifier`, `RandomForestClassifier`, etc.).
* **Experiment Tracking**: Automatically logs parameters, metrics (Accuracy, ROC AUC, Precision, Recall), and model artifacts using **MLflow** (`main.py`).
* **Pipeline Management**: Managed via **DVC** (`dvc.yaml`) to ensure deterministic and reproducible execution stages.

## 📂 Project Structure

```text
.
├── config.yml            # Centralized configuration (model params, data paths)
├── dataset.py            # Script to generate synthetic train/test datasets
├── dvc.yaml              # DVC pipeline stages definition
├── main.py               # Main entry point running the full MLflow pipeline
├── steps/
│   ├── ingest.py         # Data ingestion logic
│   ├── clean.py          # Data cleaning logic
│   ├── train.py          # Model training pipeline logic
│   └── predict.py        # Standalone model evaluation script
└── data/                 # Directory containing generated CSV files
```

## ⚙️ Configuration (`config.yml`)
You can easily change the model type and its hyperparameters without modifying the code by updating the `config.yml` file. Supported models include:
- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`

## 💻 Installation & Setup

1. Clone the repository and navigate to the project directory.
2. Create a virtual environment and activate it.
3. Install the required dependencies:

```bash
pip install pandas scikit-learn joblib pyyaml mlflow dvc
```

## ▶️ Running the Pipeline

### Option 1: Using Python
To run the end-to-end pipeline and log the experiment to MLflow:
```bash
python main.py
```

### Option 2: Using DVC
To reproduce the pipeline steps defined in `dvc.yaml`:
```bash
dvc repro
```

## 📊 Viewing MLflow Results
After running the pipeline, you can view the logged experiments, parameters, metrics, and models in the MLflow UI:

```bash
mlflow ui --backend-store-uri file:./mlruns
```
Then, open your browser and go to `http://127.0.0.1:5000`.