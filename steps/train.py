import pandas as pd
import os
import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class Trainer:
    """
    A class for training the machine learning model.
    """
    def __init__(self):
        """
        Initializes the Trainer, loading configuration.
        """
        with open('config.yml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model'].get('params') or {}
        self.model = self._create_model()
        self.pipeline = None

    def _create_model(self):
        """
        Creates a model instance based on the configuration.
        """
        if self.model_name == 'LogisticRegression':
            return LogisticRegression(**self.model_params)
        elif self.model_name == 'DecisionTreeClassifier':
            return DecisionTreeClassifier(**self.model_params)
        elif self.model_name == 'GradientBoostingClassifier':
            return GradientBoostingClassifier(**self.model_params)
        elif self.model_name == 'RandomForestClassifier':
            return RandomForestClassifier(**self.model_params)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

    def feature_target_separator(self, data: pd.DataFrame):
        """
        Separates features and target from a DataFrame.
        """
        target_column = self.config['data']['target_column']
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    def _get_preprocessor(self, X: pd.DataFrame):
        """
        Creates a preprocessor for numerical and categorical features.
        """
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        return preprocessor

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the model using a pipeline that includes preprocessing.
        """
        preprocessor = self._get_preprocessor(X_train)
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', self.model)])
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        """
        Saves the trained pipeline to a file.
        """
        model_path = self.config['model']['store_path']
        os.makedirs(model_path, exist_ok=True)
        file_path = os.path.join(model_path, 'model.pkl')
        joblib.dump(self.pipeline, file_path)
        print(f"Model saved to {file_path}")

if __name__ == "__main__":
    trainer = Trainer()
    train_data = pd.read_csv("data/clean_train.csv")
    X_train, y_train = trainer.feature_target_separator(train_data)
    
    trainer.train_model(X_train, y_train)
    trainer.save_model()
