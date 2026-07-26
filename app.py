import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from pipeline import (
    evaluate_model,
    explore_data,
    get_models,
    load_data,
    preprocess_data,
    train_model,
)

st.set_page_config(page_title="Iris Species Prediction", layout="wide")

sns.set_theme(style="whitegrid")


@st.cache_data
def cached_load_data():
    return load_data()


@st.cache_data
def cached_preprocess(df: pd.DataFrame, test_size: float, random_state: int):
    return preprocess_data(df, test_size=test_size, random_state=random_state)


@st.cache_resource
def cached_train_model(model_name: str, X_train, y_train):
    model = get_models()[model_name]
    return train_model(model, X_train, y_train)


# ---------------------------------------------------------------------------
# 1. Intro
# ---------------------------------------------------------------------------
st.title("Iris Flower Species Prediction")
st.markdown(
    """
**Objective:** predict the species of an iris flower (*setosa*, *versicolor*, or
*virginica*) from four numeric measurements — sepal length, sepal width,
petal length, and petal width.

**Dataset:** the classic Fisher/Anderson Iris dataset (`iris.data`), 150
samples, 50 per species, no missing values. One species (*setosa*) is
linearly separable from the other two; *versicolor* and *virginica* overlap,
which is the main source of misclassification later in this app.
"""
)

df = cached_load_data()

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
st.header("1. Exploratory Data Analysis")

eda = explore_data(df)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Class balance")
    st.dataframe(eda["class_counts"])
    st.caption("Perfectly balanced: 50 samples per species.")

with col2:
    st.subheader("Summary statistics")
    st.dataframe(eda["summary_stats"])

st.subheader("Feature separability")
fig = sns.pairplot(df, hue="species", diag_kind="hist", palette="deep")
st.pyplot(fig)
st.caption(
    "Setosa (one color) sits in its own cluster across every feature pair — "
    "it is linearly separable. Versicolor and virginica overlap noticeably, "
    "especially on sepal length/width, which is why most misclassifications "
    "later happen between those two species."
)

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
st.header("2. Preprocessing")

c1, c2 = st.columns(2)
with c1:
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
with c2:
    random_state = st.number_input("Random state", value=42, step=1)

st.markdown(
    f"""
- **Label encoding:** species names encoded to integers (0/1/2) with
  `LabelEncoder`.
- **Stratified split:** {int((1 - test_size) * 150)}/{int(test_size * 150)}
  train/test split, stratified on species so all three classes stay
  balanced in both sets — important given the small (150-row) dataset.
"""
)

X_train, X_test, y_train, y_test, encoder = cached_preprocess(
    df, test_size, int(random_state)
)
st.write(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# ---------------------------------------------------------------------------
# 4. Model selection
# ---------------------------------------------------------------------------
st.header("3. Model Training")

model_names = list(get_models().keys())
selected_models = st.multiselect(
    "Choose classifiers to train and compare",
    model_names,
    default=model_names,
)

if not selected_models:
    st.warning("Select at least one model to continue.")
    st.stop()

trained_models = {
    name: cached_train_model(name, X_train, y_train) for name in selected_models
}

# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------
st.header("4. Results")

focus_model = st.selectbox("Model to inspect in detail", selected_models)

results = {
    name: evaluate_model(model, X_test, y_test, encoder)
    for name, model in trained_models.items()
}

if len(selected_models) > 1:
    st.subheader("Accuracy comparison")
    acc_df = pd.DataFrame(
        {"model": selected_models, "accuracy": [results[n]["accuracy"] for n in selected_models]}
    ).set_index("model")
    st.bar_chart(acc_df)

res = results[focus_model]
st.subheader(f"Detail: {focus_model}")

m1, m2 = st.columns(2)
with m1:
    st.metric("Accuracy", f"{res['accuracy']:.3f}")
    st.subheader("Classification report")
    report_df = pd.DataFrame(res["report"]).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))

with m2:
    st.subheader("Confusion matrix")
    fig_cm, ax = plt.subplots()
    sns.heatmap(
        res["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=res["class_names"],
        yticklabels=res["class_names"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{focus_model} — Confusion Matrix")
    st.pyplot(fig_cm)

st.caption(
    "Most, if not all, errors occur between versicolor and virginica — "
    "consistent with the overlap seen in the EDA pairplot above."
)
