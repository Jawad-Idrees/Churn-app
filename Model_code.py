import pandas as pd

import numpy as np
import mlflow
import mlflow.sklearn
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
import cloudpickle as pickle






np.random.seed(42)
df=pd.read_csv('telecom_churn_mock_data.csv')





df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)
X=df.drop('Churn', axis=1)
y= df['Churn']
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Custom functions ---
# def drop_columns(X):
#     try:
#         return X.drop('CustomerID', axis=1)
#     except:
#         return X

# def add_totalcharges_bin(X):
#     num_bins = 10
#     bins = pd.cut(X['TotalCharges'], bins=num_bins)
#     bin_means = X.groupby(bins)['TotalCharges'].mean()
#     X['TotalCharges_Binned'] = bins.map(bin_means)
#     return X

# --- Assume df is your full dataset, and 'Churn' is the target ---
# Replace this line with your actual data
# df = pd.read_csv("your_data.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define columns
binary_cols = [col for col in X_train.columns if X_train[col].nunique() == 2 and X_train[col].dtype == 'object']
categorical_cols = [col for col in X_train.columns if X_train[col].nunique() > 2 and X_train[col].dtype == 'object' and col != 'CustomerID']

# --- Transformers ---
# dropper = FunctionTransformer(drop_columns)
# bin_creator = FunctionTransformer(add_totalcharges_bin)

preprocessor = ColumnTransformer(transformers=[
    ("binary", OneHotEncoder(drop='if_binary'), binary_cols),
    ("onehot", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ("imputer", KNNImputer(), ['TotalCharges']),
], remainder='passthrough')

feature_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100),
    threshold=0.005
)

pipeline = Pipeline(steps=[
    # ("dropper", dropper),
    # ("bin_creator", bin_creator),
    ("preprocessor", preprocessor),
    ("feature_selector", feature_selector),
    ("classifier", RandomForestClassifier())
])

# --- MLflow Tracking ---
mlflow.set_experiment("Churn_Prediction_Experiment")

with mlflow.start_run():
    # Log Parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("feature_selector_threshold", 0.005)
    mlflow.log_param("imputer", "KNNImputer")
    mlflow.log_param("binary_encoding", "OneHot(drop='if_binary')")
    mlflow.log_param("categorical_encoding", "OneHot(handle_unknown='ignore')")
    mlflow.log_param("num_bins_totalcharges", 10)

    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log model
    mlflow.sklearn.log_model(pipeline, "churn_pipeline_model")

    print(f"Pipeline trained and logged to MLflow with accuracy: {acc:.4f}")


joblib.dump(pipeline, "pipeline.pkl")
