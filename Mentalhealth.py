
import os
import re
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier





st.set_page_config(
    page_title=" Mental Health Text Classification",
    page_icon="🛡️ Mental Health Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }

        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 18px;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(128,128,128,.20);
            background: linear-gradient(
                135deg,
                rgba(99,102,241,.14),
                rgba(14,165,233,.08)
            );
        }

        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
            letter-spacing: -0.03em;
        }

        .hero p {
            margin: .45rem 0 0;
            opacity: .78;
            font-size: 1.02rem;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: .4rem;
            margin-bottom: .8rem;
        }

        .status-card {
            padding: 1rem 1.15rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,.18);
            background: rgba(128,128,128,.055);
        }

        .small-muted {
            color: rgba(128,128,128,.95);
            font-size: .88rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.16);
            padding: .9rem;
            border-radius: 14px;
            background: rgba(128,128,128,.045);
        }

        .sidebar-note {
            padding: .8rem;
            border-radius: 12px;
            background: rgba(128,128,128,.07);
            font-size: .84rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



CLASS_MAP: Dict[int, str] = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality disorder",
    5: "Stress",
    6: "Suicidal",
}

CLASS_COUNTS = {
    "Normal": 16_351,
    "Depression": 15_404,
    "Suicidal": 10_653,
    "Anxiety": 3_888,
    "Bipolar": 2_877,
    "Stress": 2_669,
    "Personality disorder": 1_201,
}

TOTAL_CLEAN_ROWS = 52_681
TRAIN_ACCURACY = 91.4
TEST_ACCURACY = 77.4

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
MODEL_PATH = BASE_DIR / "xgb_model.pkl"



def clean_text(text: str) -> str:
    """Apply the same basic text cleaning used by the notebook pipeline."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load and clean the source dataset."""
    df = pd.read_csv(path)

    required = {"statement", "status"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna(subset=["statement", "status"]).copy()
    df["statement"] = df["statement"].astype(str)
    return df


def _load_pickle(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner=False)
def load_model_artifacts(
    data_path: str,
    vectorizer_path: str,
    model_path: str,
):
    """
    Load saved artifacts when available.

    If artifacts are absent, train a compact fallback model so the
    dashboard remains executable without manual model setup.
    """
    vectorizer_file = Path(vectorizer_path)
    model_file = Path(model_path)

    if vectorizer_file.exists() and model_file.exists():
        vectorizer = _load_pickle(vectorizer_file)
        model = _load_pickle(model_file)
        return vectorizer, model, "Saved production artifacts"

    df = load_dataset(data_path)

    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )

    X = df["statement"].map(clean_text)
    y = df["status"].map({label: idx for idx, label in CLASS_MAP.items()})

    x_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=2,
        stratify=y,
    )

    x_train_vec = vectorizer.fit_transform(x_train)

    model = XGBClassifier(
        n_estimators=180,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=2,
        n_jobs=-1,
    )

    model.fit(x_train_vec, y_train)
    return vectorizer, model, "Fallback model trained from dataset"


def get_probability_distribution(model, vectorized_text) -> np.ndarray:
    """Return a seven-class probability distribution in CLASS_MAP order."""
    probabilities = np.asarray(model.predict_proba(vectorized_text))[0]

    # Align model classes to the requested numeric label map.
    aligned = np.zeros(len(CLASS_MAP), dtype=float)
    for model_class, probability in zip(model.classes_, probabilities):
        class_id = int(model_class)
        if class_id in CLASS_MAP:
            aligned[class_id] = float(probability)

    total = aligned.sum()
    return aligned / total if total > 0 else aligned



def confidence_chart(probabilities: np.ndarray):
    chart_df = pd.DataFrame(
        {
            "Status": [CLASS_MAP[i] for i in range(7)],
            "Confidence": probabilities * 100,
        }
    ).sort_values("Confidence", ascending=True)

    fig = px.bar(
        chart_df,
        x="Confidence",
        y="Status",
        orientation="h",
        text="Confidence",
        labels={"Confidence": "Model confidence (%)", "Status": ""},
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    )
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=35, t=20, b=20),
        xaxis=dict(range=[0, max(100, chart_df["Confidence"].max() * 1.18)]),
        showlegend=False,
    )
    return fig


def class_distribution_chart():
    chart_df = (
        pd.DataFrame(CLASS_COUNTS.items(), columns=["Status", "Statements"])
        .sort_values("Statements", ascending=True)
    )

    fig = px.bar(
        chart_df,
        x="Statements",
        y="Status",
        orientation="h",
        text="Statements",
        labels={"Statements": "Number of statements", "Status": ""},
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    )
    fig.update_layout(
        height=470,
        margin=dict(l=10, r=40, t=20, b=20),
        showlegend=False,
    )
    return fig


def word_count_chart(df: pd.DataFrame):
    sample = df["statement"].map(clean_text).str.split().str.len()
    sample = sample.clip(upper=80)

    fig = px.histogram(
        x=sample,
        nbins=35,
        labels={"x": "Words per statement", "y": "Number of statements"},
    )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=20, b=20),
        showlegend=False,
    )
    return fig


def confusion_matrix_figure():
    """
    Illustrative/static confusion matrix.

    The project  provide overall train/test accuracy but do not
    provide a class-level confusion matrix, so this matrix is deliberately
    labeled as illustrative rather than presented as a measured result.
    """
    matrix = np.array(
        [
            [72, 5, 4, 8, 2, 7, 2],
            [4, 74, 8, 4, 4, 4, 2],
            [3, 5, 79, 3, 2, 3, 5],
            [4, 2, 3, 86, 1, 3, 1],
            [4, 6, 4, 4, 70, 8, 4],
            [5, 4, 5, 5, 3, 75, 3],
            [2, 3, 7, 3, 2, 4, 79],
        ]
    )

    labels = [CLASS_MAP[i] for i in range(7)]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Illustrative value: %{z}<extra></extra>",
            colorscale="Blues",
        )
    )
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=20, b=90),
        xaxis_title="Predicted status",
        yaxis_title="Actual status",
    )
    return fig



with st.sidebar:
    st.markdown("## 🛡️ Mental Health Detector ")
    st.caption("Mental Health Text Classification Dashboard")

    section = st.radio(
        "Navigate",
        [
            "🔮 Interactive Predictor",
            "📊 Exploratory Data Analysis",
            "📈 Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        """
        <div class="sidebar-note">
        <b>Project snapshot</b><br>
        52,681 cleaned statements<br>
        7 mental-health categories<br>
        TF-IDF + XGBoost classification
        </div>
        """,
        unsafe_allow_html=True,
    )

   


st.markdown(
    """
    <div class="hero">
        <h1>🛡️ </h1>
        <p>Turning unstructured mental-health text into clear, explainable classification insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)



if section == "🔮 Interactive Predictor":
    st.markdown('<div class="section-title">Interactive Predictor</div>', unsafe_allow_html=True)
    st.write(
        "Enter a statement to see the model's predicted category and how its confidence "
        "is distributed across all seven categories."
    )

    user_text = st.text_area(
        "Statement",
        height=170,
        placeholder="Example: I have been feeling unusually restless and unable to sleep lately...",
        label_visibility="collapsed",
    )

    analyze = st.button("🔍 Analyze Status", type="primary", use_container_width=False)

    if analyze:
        if not user_text.strip():
            st.warning("Please enter a statement before running the analysis.")
        else:
            try:
                with st.spinner("Analyzing the statement..."):
                    vectorizer, model, artifact_status = load_model_artifacts(
                        str(DATA_PATH),
                        str(VECTORIZER_PATH),
                        str(MODEL_PATH),
                    )

                    cleaned = clean_text(user_text)
                    vectorized = vectorizer.transform([cleaned])

                    prediction = int(model.predict(vectorized)[0])
                    predicted_label = CLASS_MAP.get(prediction, "Unknown")
                    probabilities = get_probability_distribution(model, vectorized)

                st.success("Analysis completed.")

                left, right = st.columns([0.9, 1.5], gap="large")

                with left:
                    st.markdown("### Predicted status")
                    st.info(
                        f"**{predicted_label}**\n\n"
                        "This is the category the model assigns the highest probability to."
                    )

                    top_confidence = probabilities[prediction] * 100
                    st.metric("Top model confidence", f"{top_confidence:.1f}%")
                    st.caption(f"Model source: {artifact_status}")

                with right:
                    st.markdown("### Confidence distribution")
                    st.plotly_chart(
                        confidence_chart(probabilities),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                st.caption(
                    "Confidence reflects the model's learned probability distribution; "
                    "it should not be interpreted as a clinical diagnosis or certainty."
                )

            except FileNotFoundError as exc:
                st.error(f"Required project file was not found: {exc}")
            except Exception as exc:
                st.error(
                    "The analysis could not be completed. "
                    "Please check that the dataset and model artifacts are valid."
                )
                with st.expander("Technical details"):
                    st.exception(exc)



elif section == "📊 Exploratory Data Analysis":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.write(
        "A high-level view of the dataset helps explain its scale, category mix, "
        "and the variety of text presented to the model."
    )

    try:
        df = load_dataset(str(DATA_PATH))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Statements analyzed", f"{len(df):,}")
        m2.metric("Categories", "7")
        m3.metric("Largest category", "Normal")
        m4.metric("Largest category share", f"{CLASS_COUNTS['Normal'] / TOTAL_CLEAN_ROWS:.1%}")

        st.markdown("### Category distribution")
        st.plotly_chart(
            class_distribution_chart(),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("### Text length profile")
            st.plotly_chart(
                word_count_chart(df),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with c2:
            st.markdown("### Dataset composition")
            summary = (
                pd.DataFrame(CLASS_COUNTS.items(), columns=["Status", "Statements"])
                .assign(Share=lambda x: x["Statements"] / TOTAL_CLEAN_ROWS)
                .sort_values("Statements", ascending=False)
            )
            summary["Share"] = summary["Share"].map(lambda x: f"{x:.1%}")
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "The displayed class counts are the project dataset counts supplied for the dashboard. "
            "The source CSV contains 53,043 rows before removing 362 statements with missing text, "
            "leaving 52,681 usable statements."
        )

    except Exception as exc:
        st.error("The dataset could not be loaded.")
        with st.expander("Technical details"):
            st.exception(exc)



else:
    st.markdown('<div class="section-title">Model Performance & Metrics</div>', unsafe_allow_html=True)
    st.write(
        "The model's results can be viewed as a balance between learning the training examples "
        "and maintaining useful performance on unseen statements."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Training accuracy", f"{TRAIN_ACCURACY:.1f}%")
        st.caption("How consistently the model classified statements it learned from.")

    with c2:
        st.metric("Test accuracy", f"{TEST_ACCURACY:.1f}%")
        st.caption("Performance on held-out statements not used during training.")

    with c3:
        st.metric("Generalization gap", f"{TRAIN_ACCURACY - TEST_ACCURACY:.1f} pp")
        st.caption("A useful signal for monitoring how well performance transfers to new text.")

    st.divider()

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.markdown("### Confusion matrix")
        st.caption(
            "Illustrative visualization only: the project materials provide overall accuracy, "
            "not class-level confusion-matrix values."
        )
        st.plotly_chart(
            confusion_matrix_figure(),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        st.markdown("### Why this model?")

        st.markdown(
            """
            **TF-IDF** converts free-form language into a numerical representation that
            highlights informative words and phrases.

            **XGBoost** then learns non-linear relationships among those high-dimensional
            features. This makes the pipeline practical for a multi-class text classification
            problem where the input is sparse and contains many potentially useful signals.

            **Business impact:** the result is a compact workflow that can turn large volumes
            of unstructured statements into consistent categories for downstream analysis,
            triage workflows, or research dashboards.
            """
        )

        st.info(
            "**Key takeaway:** the 77.4% test accuracy shows that the pipeline captures "
            "meaningful patterns beyond the training data, while the 91.4% training score "
            "also highlights the importance of continued validation and monitoring on new data."
        )

        st.markdown("### Model stack")
        st.code(
            "Text → Cleaning → TF-IDF (5,000 features) → XGBoost → 7-class prediction",
            language="text",
        )
        

    
