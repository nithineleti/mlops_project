# MLOps Classification Pipeline

This project implements a **complete end-to-end Machine Learning Operations (MLOps) pipeline** for a classification problem. The pipeline automates data generation, preprocessing, model training, experiment tracking, and deployment using modern MLOps tools.

The project integrates the following MLOps tools:

- **DVC** → Data and pipeline versioning  
- **MLflow** → Experiment tracking and model registry  
- **FastAPI** → Model serving API  
- **Docker** → Containerized deployment  
- **Scikit-learn** → Machine learning model development  

This project demonstrates how to build **production-ready machine learning pipelines** used in real-world ML systems.

---

# 🚀 Features

### Data Generation & Ingestion
- Generates synthetic classification datasets using `sklearn.datasets.make_classification`.
- Simulates real-world ML data ingestion pipelines.

### Data Cleaning
- Handles missing values.
- Converts the target column into numeric format.
- Ensures clean and consistent training data.

### Model Training & Preprocessing
Uses **Scikit-learn pipelines** for preprocessing and modeling.

Preprocessing techniques:
- `StandardScaler`
- `OneHotEncoder`

Supported models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### Experiment Tracking
All experiments are logged using **MLflow**, including:

- hyperparameters
- metrics
- trained models
- experiment runs

Tracked metrics include:
- Accuracy
- ROC AUC
- Precision
- Recall

### Model Registry
Trained models are registered in the **MLflow Model Registry**, enabling:

- model versioning
- model stage transitions
- controlled production deployment

### Pipeline Automation
The full ML pipeline is automated using **DVC**, ensuring reproducible and versioned ML workflows.

### Model Serving API
The trained model is served through a **FastAPI REST API**.

### Containerized Deployment
The entire project can be deployed using **Docker containers**.

### Data Drift Monitoring
Includes drift detection reports to monitor dataset changes over time.

---

# 🧪 Synthetic Dataset Generation

The project generates synthetic data using **Scikit-learn's `make_classification`** function.

Example:

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    random_state=42
)
```

This allows us to simulate real-world classification datasets for building the ML pipeline.

---

# 📂 Project Structure

```
.
├── config.yml                # Centralized configuration (model params, data paths)
├── dataset.py                # Script to generate synthetic train/test datasets
├── dvc.yaml                  # DVC pipeline stages definition
├── dvc.lock                  # DVC pipeline lock file
├── main.py                   # Main entry point running the MLflow training pipeline
├── app.py                    # FastAPI model serving API
│
├── steps/
│   ├── ingest.py             # Data ingestion logic
│   ├── clean.py              # Data cleaning logic
│   ├── train.py              # Model training pipeline logic
│   └── predict.py            # Standalone model evaluation script
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── clean_train.csv
│   └── clean_test.csv
│
├── models/
│   └── model.pkl             # Trained model artifact
│
├── mlruns/                   # MLflow experiment tracking
│
├── Dockerfile                # Docker container configuration
├── requirements.txt          # Project dependencies
│
├── monitor.ipynb             # Monitoring notebook
├── production_drift.html     # Production drift report
├── test_drift.html           # Test drift report
│
└── README.md
```

---

# ⚙️ Configuration (`config.yml`)

The project uses a centralized configuration file to control the pipeline.

Example:

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

Supported models:

- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier

Changing the model or parameters **requires only editing the config file**.

---

# 💻 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/your-username/mlops-classification-pipeline.git
cd mlops-classification-pipeline
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install pandas scikit-learn joblib pyyaml mlflow dvc fastapi uvicorn
```

---

# ▶️ Running the Pipeline

### Option 1: Run using Python

```bash
python main.py
```

Pipeline steps:

1. Data ingestion  
2. Data cleaning  
3. Model training  
4. Model evaluation  
5. MLflow experiment logging  
6. Model registry  

---

### Option 2: Run using DVC

Run the automated pipeline:

```bash
dvc repro
```

Pipeline stages:

```
data_ingestion
      │
      ▼
data_cleaning
      │
      ▼
model_training
```

---

# 🔁 Pipeline Visualization

Visualize the pipeline graph:

```bash
dvc dag
```

Example output:

```
data_ingestion
      │
      ▼
data_cleaning
      │
      ▼
model_training
```

This helps understand the **dependencies between pipeline stages**.

---

# 📊 Viewing MLflow Results

Start MLflow UI:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open browser:

```
http://127.0.0.1:5000
```

Inside MLflow you can view:

- experiment runs
- hyperparameters
- metrics
- model artifacts
- registered models

---

# 📦 MLflow Model Registry

The trained model is automatically registered in MLflow.

Model stages include:

- None
- Staging
- Production
- Archived

This allows safe model promotion and deployment.

---

# 🌐 Model Serving API

Start FastAPI server:

```bash
uvicorn app:app --reload
```

Open API documentation:

```
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "feature_0": 0.5,
  "feature_1": -1.2,
  "feature_2": 0.3
}
```

Example response:

```json
{
  "prediction": 1
}
```

---

# 🐳 Docker Deployment

Build Docker image:

```bash
docker build -t mlops-classification .
```

Run container:

```bash
docker run -p 8000:8000 mlops-classification
```

Access API:

```
http://localhost:8000/docs
```

---

# ☁️ DVC Remote Storage (Optional)

DVC can store datasets and models in remote storage.

Example configuration:

```bash
dvc remote add -d storage s3://mybucket/dvc-storage
dvc push
```

Supported storage backends:

- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- SSH storage

---

# 📈 Model Performance

Example metrics:

```
Accuracy Score: 0.9185
ROC AUC Score: 0.9186
```

Classification Report:

| Class | Precision | Recall | F1 Score |
|------|----------|-------|--------|
| 0 | 0.92 | 0.91 | 0.92 |
| 1 | 0.91 | 0.92 | 0.92 |

---

# 🔁 Data Drift Monitoring

The project includes drift detection reports:

```
production_drift.html
test_drift.html
```

These reports detect **data distribution changes** between training and production datasets.

---

# ⭐ Key MLOps Components

| Component | Tool Used |
|------|------|
| Data Pipeline | DVC |
| Experiment Tracking | MLflow |
| Model Registry | MLflow |
| Model Training | Scikit-learn |
| API Serving | FastAPI |
| Containerization | Docker |
| Monitoring | Drift Reports |

---

# 📚 Future Improvements

Possible improvements include:

- CI/CD pipeline with GitHub Actions
- Kubernetes deployment
- Feature Store integration
- Automated model retraining
- Real-time monitoring dashboards

---

# 👨‍💻 Author

**Nithin Pious**

AI/ML Engineer | MLOps Enthusiast
