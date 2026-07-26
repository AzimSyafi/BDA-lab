"""Iris species prediction pipeline: load, explore, preprocess, train, evaluate."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
FEATURES = COLUMNS[:-1]
DATA_PATH = "iris/iris.data"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["species"] = df["species"].str.replace("Iris-", "", regex=False)
    return df


def explore_data(df: pd.DataFrame) -> dict:
    return {
        "class_counts": df["species"].value_counts(),
        "summary_stats": df[FEATURES].describe(),
        "summary_by_species": df.groupby("species")[FEATURES].mean(),
    }


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["species"])
    X = df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, encoder


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, encoder: LabelEncoder) -> dict:
    y_pred = model.predict(X_test)
    class_names = encoder.classes_

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "class_names": class_names,
        "report": classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        ),
        "y_pred": y_pred,
    }
