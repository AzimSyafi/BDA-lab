"""
BDA Lab 8: Full Data Analytics Revision
Option 1: Predicting Iris Flower Species

Pipeline: load data, explore, preprocess, train multiple classifiers,
evaluate, and compare results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

sns.set_style("whitegrid")
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT, exist_ok=True)

# 1. LOAD DATA
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 60)
print("STEP 1: DATA OVERVIEW")
print("=" * 60)
print(df.head())
print("\nShape:", df.shape)
print("\nClass balance:\n", df["species"].value_counts())
print("\nSummary statistics:\n", df.describe())

# 2. EXPLORATORY DATA ANALYSIS
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

pairplot = sns.pairplot(df, hue="species", diag_kind="hist", palette="Set2")
pairplot.savefig(f"{OUT}/01_pairplot.png", dpi=150)
plt.close("all")

plt.figure(figsize=(6, 5))
sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/02_correlation_heatmap.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
for col in iris.feature_names:
    sns.kdeplot(data=df, x=col, hue="species", common_norm=False)
    plt.title(f"Distribution of {col} by Species")
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_dist_{col.replace(' ', '_').replace('(cm)','')}.png", dpi=150)
    plt.close()

print("EDA plots saved: pairplot, correlation heatmap, distribution plots.")
print("Petal length and petal width show the clearest separation between species.")

# 3. PREPROCESSING
print("\n" + "=" * 60)
print("STEP 3: PREPROCESSING")
print("=" * 60)

X = df[iris.feature_names]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print("Features standardised with StandardScaler (mean 0, std 1).")

# 4. MODEL TRAINING AND EVALUATION
print("\n" + "=" * 60)
print("STEP 4: MODEL TRAINING AND EVALUATION")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=iris.target_names))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{OUT}/04_confusion_{safe_name}.png", dpi=150)
    plt.close()

# 5. MODEL COMPARISON
print("\n" + "=" * 60)
print("STEP 5: MODEL COMPARISON")
print("=" * 60)
results_df = pd.Series(results).sort_values(ascending=False)
print(results_df)

plt.figure(figsize=(6, 4))
results_df.plot(kind="bar", color=sns.color_palette("Set2"))
plt.ylabel("Accuracy")
plt.title("Classifier Accuracy Comparison on Iris Test Set")
plt.ylim(0.8, 1.02)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/05_model_comparison.png", dpi=150)
plt.close()

# 6. FEATURE IMPORTANCE (Random Forest)
print("\n" + "=" * 60)
print("STEP 6: FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
print(importances)

plt.figure(figsize=(6, 4))
importances.plot(kind="barh", color=sns.color_palette("Set2")[2])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/06_feature_importance.png", dpi=150)
plt.close()

print("\nAll plots saved to:", OUT)
print("Best performing model:", results_df.index[0], f"({results_df.iloc[0]:.4f} accuracy)")