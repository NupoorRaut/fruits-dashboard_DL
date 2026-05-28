import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fruits Classification Dashboard",
    page_icon="🍎",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = "."

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "plots"
)

# ============================================================
# TITLE
# ============================================================

st.title("🍎 Fruits Classification Dashboard")

st.markdown(
    "Deep Learning Model Evaluation & Prediction System"
)

# ============================================================
# LOAD RESULTS
# ============================================================

results_path = os.path.join(
    RESULT_DIR,
    "recovered_results.xlsx"
)

if not os.path.exists(results_path):

    st.error(
        "Results file not found"
    )

    st.stop()

df = pd.read_excel(results_path)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Overview",
        "Model Comparison",
        "Advanced Analytics",
        "Plots",
        "Predictions",
        "Raw Results"
    ]
)

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("📊 Project Overview")

    total_models = len(df)

    best_accuracy = df["Accuracy"].max()

    best_model = df.loc[
        df["Accuracy"].idxmax(),
        "Model"
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Models",
        total_models
    )

    col2.metric(
        "Best Accuracy",
        f"{best_accuracy:.4f}"
    )

    col3.metric(
        "Best Model",
        str(best_model)
    )

    st.subheader("Top Models")

    top_df = df.sort_values(
        by="Accuracy",
        ascending=False
    ).head(10)

    st.table(top_df)

# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    import matplotlib.pyplot as plt

    st.header("📈 Advanced Model Comparison")

    metric_options = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity"
    ]

    selected_metrics = st.multiselect(
        "Select Metrics",
        metric_options,
        default=metric_options
    )

    top_n = st.slider(
        "Select Top Models",
        min_value=3,
        max_value=20,
        value=10
    )

    # ========================================================
    # SORT MODELS
    # ========================================================

    sorted_df = df.sort_values(
        by="Accuracy",
        ascending=False
    ).head(top_n)

    # ========================================================
    # CREATE LABELS
    # ========================================================

    sorted_df["Label"] = (
        sorted_df["Model"].astype(str)
        + " | "
        + sorted_df["Optimizer"].astype(str)
    )

    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(18, 8)
    )

    x = np.arange(len(sorted_df))

    width = 0.15

    for i, metric in enumerate(selected_metrics):

        ax.bar(
            x + i * width,
            sorted_df[metric],
            width,
            label=metric
        )

    ax.set_xlabel(
        "Models",
        fontsize=14
    )

    ax.set_ylabel(
        "Score",
        fontsize=14
    )

    ax.set_title(
        "Model Performance Comparison",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xticks(
        x + width * (len(selected_metrics)-1)/2
    )

    ax.set_xticklabels(
        sorted_df["Label"],
        rotation=45,
        ha="right"
    )

    ax.legend()

    ax.grid(
        linestyle="--",
        alpha=0.5
    )

    st.pyplot(fig)

    # ========================================================
    # BEST MODEL SUMMARY
    # ========================================================

    st.subheader("🏆 Best Performing Model")

    best_row = sorted_df.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model",
        best_row["Model"]
    )

    col2.metric(
        "Accuracy",
        f"{best_row['Accuracy']:.4f}"
    )

    col3.metric(
        "F1 Score",
        f"{best_row['F1']:.4f}"
    )

    # ========================================================
    # TABLE
    # ========================================================

    st.subheader("📋 Detailed Comparison")

    st.dataframe(
        sorted_df,
        use_container_width=True
    )
    
# ============================================================
# ADVANCED ANALYTICS
# ============================================================

elif page == "Advanced Analytics":

    st.header("📊 Advanced Analytics Dashboard")

    analytics_option = st.selectbox(
        "Select Visualization",
        [
            "Correlation Heatmap",
            "Metric Distribution",
            "Optimizer Comparison",
            "Training vs Validation",
            "Top Models Radar Chart",
            "Metric Pairplot"
        ]
    )

    # ========================================================
    # CORRELATION HEATMAP
    # ========================================================

    if analytics_option == "Correlation Heatmap":

        st.subheader("🔥 Correlation Heatmap")

        numeric_cols = [

            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Specificity",
            "ROC-AUC"
        ]

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax
        )

        st.pyplot(fig)

    # ========================================================
    # METRIC DISTRIBUTION
    # ========================================================

    elif analytics_option == "Metric Distribution":

        st.subheader("📈 Metric Distribution")

        metric = st.selectbox(
            "Select Metric",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Specificity"
            ]
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.histplot(
            df[metric],
            kde=True,
            bins=10,
            ax=ax
        )

        ax.set_title(
            f"{metric} Distribution"
        )

        st.pyplot(fig)

    # ========================================================
    # OPTIMIZER COMPARISON
    # ========================================================

    elif analytics_option == "Optimizer Comparison":

        st.subheader("⚙️ Optimizer Comparison")

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        sns.boxplot(
            data=df,
            x="Optimizer",
            y="Accuracy",
            ax=ax
        )

        ax.set_title(
            "Optimizer vs Accuracy"
        )

        st.pyplot(fig)

    # ========================================================
    # TRAINING VS VALIDATION
    # ========================================================

    elif analytics_option == "Training vs Validation":

        st.subheader("📉 Training vs Validation Accuracy")

        compare_df = df[[
            "Model",
            "Train Accuracy",
            "Validation Accuracy"
        ]].dropna()

        compare_df = compare_df.sort_values(
            by="Validation Accuracy",
            ascending=False
        )

        fig, ax = plt.subplots(
            figsize=(16, 7)
        )

        x = np.arange(len(compare_df))

        width = 0.35

        ax.bar(
            x - width/2,
            compare_df["Train Accuracy"],
            width,
            label="Train"
        )

        ax.bar(
            x + width/2,
            compare_df["Validation Accuracy"],
            width,
            label="Validation"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            compare_df["Model"],
            rotation=45,
            ha="right"
        )

        ax.legend()

        ax.set_title(
            "Training vs Validation Accuracy"
        )

        st.pyplot(fig)

    # ========================================================
    # RADAR CHART
    # ========================================================

    elif analytics_option == "Top Models Radar Chart":

        st.subheader("🕸️ Radar Chart")

        top_models = df.sort_values(
            by="Accuracy",
            ascending=False
        ).head(3)

        metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Specificity"
        ]

        import math

        angles = np.linspace(
            0,
            2 * np.pi,
            len(metrics),
            endpoint=False
        ).tolist()

        angles += angles[:1]

        fig = plt.figure(
            figsize=(8, 8)
        )

        ax = plt.subplot(
            111,
            polar=True
        )

        for _, row in top_models.iterrows():

            values = [
                row[m]
                for m in metrics
            ]

            values += values[:1]

            ax.plot(
                angles,
                values,
                label=row["Model"]
            )

            ax.fill(
                angles,
                values,
                alpha=0.1
            )

        ax.set_xticks(
            angles[:-1]
        )

        ax.set_xticklabels(
            metrics
        )

        ax.set_title(
            "Top Models Radar Chart"
        )

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.1)
        )

        st.pyplot(fig)

    # ========================================================
    # PAIRPLOT
    # ========================================================

    elif analytics_option == "Metric Pairplot":

        st.subheader("🔗 Metric Pairplot")

        metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Specificity"
        ]

        fig = sns.pairplot(
            df[metrics]
        )

        st.pyplot(fig)

# ============================================================
# PLOTS PAGE
# ============================================================

elif page == "Plots":

    st.header("🖼️ Saved Plots")

    plot_category = st.selectbox(
        "Select Plot Category",
        [
            "confusion",
            "roc",
            "curves",
            "comparisons"
        ]
    )

    category_path = os.path.join(
        PLOT_DIR,
        plot_category
    )

    if os.path.exists(category_path):

        image_files = [

            f for f in os.listdir(category_path)

            if f.endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg"
                )
            )
        ]

        if len(image_files) == 0:

            st.warning(
                "No images found"
            )

        else:

            selected_image = st.selectbox(
                "Select Plot",
                image_files
            )

            image_path = os.path.join(
                category_path,
                selected_image
            )

            st.image(
                image_path,
                use_container_width=True
            )

    else:

        st.warning(
            "Plot folder not found"
        )

# ============================================================
# PREDICTIONS PAGE
# ============================================================

elif page == "Predictions":

    st.header("🔍 Fruit Prediction")

    model_files = [

        f for f in os.listdir(MODEL_DIR)

        if f.endswith(".keras")
    ]

    if len(model_files) == 0:

        st.error(
            "No models found"
        )

        st.stop()

    selected_model = st.selectbox(
        "Select Model",
        model_files
    )

    model_path = os.path.join(
        MODEL_DIR,
        selected_model
    )

    @st.cache_resource
    def load_model_cached(path):

        return tf.keras.models.load_model(path)

    model = load_model_cached(
        model_path
    )

    uploaded_file = st.file_uploader(
        "Upload Fruit Image",
        type=["jpg", "jpeg", "png"]
    )

    class_names = [
        "apple",
        "banana",
        "cherry",
        "chickoo",
        "grapes",
        "kiwi",
        "mango",
        "orange",
        "strawberry"
    ]

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )

        img = image.resize(
            (224, 224)
        )

        img_array = np.array(
            img
        ).astype("float32")

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = model.predict(
            img_array,
            verbose=0
        )

        pred_class = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        )

        st.success(
            f"Prediction: {class_names[pred_class]}"
        )

        st.info(
            f"Confidence: {confidence:.4f}"
        )

        prob_df = pd.DataFrame({

            "Class": class_names,

            "Probability": prediction[0]

        })

        st.bar_chart(
            prob_df.set_index("Class")
        )

# ============================================================
# RAW RESULTS
# ============================================================

elif page == "Raw Results":

    st.header("📄 Raw Results")

    st.table(df)
