import os
import re
import string
import sqlite3

import streamlit as st
import pandas as pd
import nltk
import torch

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util


# ============================================================
# 1. BASIC SETUP
# ============================================================

st.set_page_config(
    page_title="MoodMentor - ML Integration",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MoodMentor - Complete ML Integration")
st.write(
    "AI-Based Employee Wellness Management Platform"
)


# ============================================================
# 2. NLTK RESOURCES
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()


# ============================================================
# 3. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "milestone 2",
    "distilbert_multilabel_emotion_model"
)

EMOTION_HISTORY_FILE = os.path.join(
    BASE_DIR,
    "emotion_history_dataset.csv"
)

HYBRID_FILE = os.path.join(
    BASE_DIR,
    "hybrid_recommendation_dataset.csv"
)

RANKING_FILE = os.path.join(
    BASE_DIR,
    "ranking_recommendation_dataset.csv"
)

WELLNESS_FILE = os.path.join(
    BASE_DIR,
    "wellness_content_dataset.csv"
)

DATABASE_FILE = os.path.join(
    BASE_DIR,
    "moodmentor.db"
)


# ============================================================
# 4. LOAD ML MODELS
# ============================================================

@st.cache_resource
def load_emotion_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


@st.cache_resource
def load_semantic_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


tokenizer, emotion_model = load_emotion_model()
semantic_model = load_semantic_model()


# ============================================================
# 5. EMOTION LABELS
# ============================================================

EMOTION_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]


# ============================================================
# 6. DATABASE
# ============================================================

def initialize_database():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotional_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            text TEXT,
            emotion TEXT,
            confidence REAL,
            intensity TEXT,
            sentiment TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            recommendation TEXT,
            accepted TEXT,
            rating INTEGER
        )
    """)

    connection.commit()
    connection.close()


initialize_database()


# ============================================================
# 7. TEXT VALIDATION
# ============================================================

def validate_text(text):

    return bool(
        text and text.strip()
    )


# ============================================================
# 8. PREPROCESSING
# ============================================================

def preprocess(text):

    if not validate_text(text):

        return {
            "error": "Empty text provided."
        }

    cleaned = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    no_special = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        cleaned
    )

    no_punct = no_special.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    tokens = word_tokenize(
        no_punct.lower()
    )

    filtered_tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    lemmatized = [
        lemmatizer.lemmatize(token)
        for token in filtered_tokens
    ]

    final_text = " ".join(
        lemmatized
    )

    return {
        "final_processed_text": final_text,
        "tokens": tokens,
        "lemmatized": lemmatized
    }


# ============================================================
# 9. SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(text):

    if not validate_text(text):

        return {
            "error": "Empty text provided."
        }

    scores = sia.polarity_scores(
        text
    )

    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"

    elif compound <= -0.05:
        label = "Negative"

    else:
        label = "Neutral"

    return {
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "compound": compound,
        "label": label
    }


# ============================================================
# 10. EMOTION DETECTION
# ============================================================

def detect_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = emotion_model(
            **inputs
        )

    probabilities = torch.sigmoid(
        outputs.logits
    )[0]

    probabilities = probabilities.tolist()

    emotion_scores = {}

    for index, emotion in enumerate(
        EMOTION_LABELS
    ):

        emotion_scores[emotion] = (
            probabilities[index]
        )

    dominant_emotion = max(
        emotion_scores,
        key=emotion_scores.get
    )

    confidence = emotion_scores[
        dominant_emotion
    ]

    return {
        "scores": emotion_scores,
        "dominant_emotion":
            dominant_emotion,
        "confidence":
            confidence
    }


# ============================================================
# 11. EMOTION INTENSITY
# ============================================================

def calculate_intensity(confidence):

    if confidence >= 0.70:
        return "High"

    elif confidence >= 0.40:
        return "Medium"

    return "Low"


# ============================================================
# 12. SAVE EMOTIONAL HISTORY
# ============================================================

def save_emotional_history(
    user_id,
    text,
    emotion,
    confidence,
    intensity,
    sentiment
):

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO emotional_history
        (
            user_id,
            text,
            emotion,
            confidence,
            intensity,
            sentiment
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        text,
        emotion,
        confidence,
        intensity,
        sentiment
    ))

    connection.commit()
    connection.close()


# ============================================================
# 13. LOAD HYBRID RECOMMENDATION DATA
# ============================================================

@st.cache_data
def load_hybrid_data():

    return pd.read_csv(
        HYBRID_FILE
    )


hybrid_df = load_hybrid_data()


# ============================================================
# 14. HYBRID RECOMMENDATION
# ============================================================

def generate_hybrid_recommendations(
    emotion,
    intensity,
    preference,
    previous_interaction,
    recommendation_history
):

    results = []

    for _, row in hybrid_df.iterrows():

        score = 0

        if row["emotion"] == emotion:
            score += 3

        if row["intensity"] == intensity:
            score += 2

        if row["preference"] == preference:
            score += 2

        if (
            row["previous_interaction"]
            == previous_interaction
        ):
            score += 1

        if (
            row["recommendation_history"]
            == recommendation_history
        ):
            score += 1

        results.append({
            "recommendation":
                row["recommendation"],
            "hybrid_score":
                score
        })

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "hybrid_score",
        ascending=False
    )

    result_df = result_df.drop_duplicates(
        subset=["recommendation"]
    )

    return result_df.reset_index(
        drop=True
    )


# ============================================================
# 15. SEMANTIC CONTENT MATCHING
# ============================================================

@st.cache_data
def load_wellness_data():

    return pd.read_csv(
        WELLNESS_FILE
    )


wellness_df = load_wellness_data()


def semantic_matching(text):

    query_embedding = semantic_model.encode(
        text,
        convert_to_tensor=True
    )

    content_embeddings = semantic_model.encode(
        wellness_df["content"].tolist(),
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        query_embedding,
        content_embeddings
    )[0]

    result = wellness_df.copy()

    result["semantic_score"] = [
        float(score)
        for score in similarities
    ]

    result = result.sort_values(
        "semantic_score",
        ascending=False
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# 16. FINAL RECOMMENDATION RANKING
# ============================================================

def rank_final_recommendations(
    hybrid_results,
    semantic_results
):

    final_results = hybrid_results.copy()

    # ------------------------------------------------
    # Get semantic score from the user's text
    # ------------------------------------------------
    if not semantic_results.empty:

        semantic_score = (
            semantic_results["semantic_score"].max()
        )

    else:

        semantic_score = 0.0

    # ------------------------------------------------
    # Add semantic score to every recommendation
    # ------------------------------------------------
    final_results["semantic_score"] = semantic_score

    # ------------------------------------------------
    # Calculate final ranking score
    # ------------------------------------------------
    final_results["final_score"] = (
        final_results["hybrid_score"] * 0.7
        + final_results["semantic_score"] * 10 * 0.3
    )

    # ------------------------------------------------
    # Sort recommendations dynamically
    # ------------------------------------------------
    final_results = final_results.sort_values(
        "final_score",
        ascending=False
    )

    return final_results.reset_index(
        drop=True
    )

    # ------------------------------------------------
    # Dynamic ranking
    # ------------------------------------------------
    final_results = final_results.sort_values(
        "final_score",
        ascending=False
    )

    return final_results.reset_index(
        drop=True
    )

# ============================================================
# 17. SAVE FEEDBACK
# ============================================================

def save_feedback(
    user_id,
    recommendation,
    accepted,
    rating
):

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO feedback
        (
            user_id,
            recommendation,
            accepted,
            rating
        )
        VALUES (?, ?, ?, ?)
    """, (
        user_id,
        recommendation,
        accepted,
        rating
    ))

    connection.commit()
    connection.close()


# ============================================================
# 18. STREAMLIT INPUT
# ============================================================

st.header("1. User Information")

user_id = st.text_input(
    "User ID",
    value="U1"
)

text_input = st.text_area(
    "Enter your current thoughts or feelings:",
    height=150
)

preference = st.selectbox(
    "Wellness preference",
    [
        "meditation",
        "music",
        "physical activity",
        "writing",
        "activity"
    ]
)

previous_interaction = st.selectbox(
    "Previous interaction",
    [
        "breathing",
        "meditation",
        "music",
        "walking",
        "exercise",
        "journaling"
    ]
)

recommendation_history = st.selectbox(
    "Recommendation history",
    [
        "breathing",
        "meditation",
        "music",
        "walking",
        "exercise",
        "journaling"
    ]
)


# ============================================================
# 19. COMPLETE WORKFLOW
# ============================================================

if st.button(
    "Run Complete MoodMentor Workflow"
):

    if not validate_text(text_input):

        st.error(
            "Please enter some text."
        )

    else:

        try:

            # ------------------------------------------------
            # PREPROCESSING
            # ------------------------------------------------

            preprocessing_result = preprocess(
                text_input
            )

            st.subheader(
                "2. Preprocessing"
            )

            st.write(
                preprocessing_result[
                    "final_processed_text"
                ]
            )


            # ------------------------------------------------
            # SENTIMENT
            # ------------------------------------------------

            sentiment = analyze_sentiment(
                text_input
            )

            st.subheader(
                "3. Sentiment Analysis"
            )

            st.write(
                "Sentiment:",
                sentiment["label"]
            )

            st.write(
                "Compound:",
                round(
                    sentiment["compound"],
                    3
                )
            )


            # ------------------------------------------------
            # EMOTION
            # ------------------------------------------------

            emotion_result = detect_emotion(
                text_input
            )

            emotion = emotion_result[
                "dominant_emotion"
            ]

            confidence = emotion_result[
                "confidence"
            ]

            st.subheader(
                "4. Emotion Detection"
            )

            st.write(
                "Dominant Emotion:",
                emotion
            )

            st.write(
                "Confidence:",
                round(
                    confidence,
                    3
                )
            )

            st.write(
                "Emotion Probabilities:"
            )

            st.dataframe(
                pd.DataFrame(
                    [
                        emotion_result["scores"]
                    ]
                )
            )


            # ------------------------------------------------
            # INTENSITY
            # ------------------------------------------------

            intensity = calculate_intensity(
                confidence
            )

            st.subheader(
                "5. Emotion Intensity"
            )

            st.write(
                "Intensity:",
                intensity
            )


            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

            save_emotional_history(
                user_id,
                text_input,
                emotion,
                confidence,
                intensity,
                sentiment["label"]
            )

            st.subheader(
                "6. Emotional History"
            )

            st.success(
                "Current emotional state stored successfully."
            )


            # ------------------------------------------------
            # HYBRID RECOMMENDATION
            # ------------------------------------------------

            hybrid_results = (
                generate_hybrid_recommendations(
                    emotion,
                    intensity,
                    preference,
                    previous_interaction,
                    recommendation_history
                )
            )

            st.subheader(
                "7. Hybrid Recommendation"
            )

            st.dataframe(
                hybrid_results
            )


            # ------------------------------------------------
            # SEMANTIC MATCHING
            # ------------------------------------------------

            semantic_results = semantic_matching(
                text_input
            )

            st.subheader(
                "8. Semantic Content Matching"
            )

            st.dataframe(
                semantic_results[
                    [
                        "title",
                        "emotion",
                        "semantic_score"
                    ]
                ].head(5)
            )


            # ------------------------------------------------
            # FINAL RANKING
            # ------------------------------------------------

            final_results = (
                rank_final_recommendations(
                    hybrid_results,
                    semantic_results
                )
            )

            st.subheader(
                "9. Dynamic Recommendation Ranking"
            )

            st.dataframe(
                final_results
            )


            # ------------------------------------------------
            # FINAL RECOMMENDATION
            # ------------------------------------------------

            final_recommendation = (
                final_results.iloc[0][
                    "recommendation"
                ]
            )

            st.subheader(
                "10. Personalized Wellness Recommendation"
            )

            st.success(
                final_recommendation
            )


            # ------------------------------------------------
            # FEEDBACK
            # ------------------------------------------------

            st.session_state[
                "last_recommendation"
            ] = final_recommendation

            st.session_state[
                "workflow_completed"
            ] = True

        except Exception as error:

            st.error(
                "An error occurred while processing "
                "the workflow."
            )

            st.exception(error)


# ============================================================
# 20. FEEDBACK SECTION
# ============================================================

if st.session_state.get(
    "workflow_completed",
    False
):

    st.header(
        "11. Recommendation Feedback"
    )

    feedback_choice = st.radio(
        "Did you accept this recommendation?",
        ["Yes", "No"]
    )

    rating = st.slider(
        "Rate the recommendation",
        min_value=1,
        max_value=5,
        value=3
    )

    if st.button(
        "Save Feedback"
    ):

        save_feedback(
            user_id,
            st.session_state[
                "last_recommendation"
            ],
            feedback_choice,
            rating
        )

        st.success(
            "Feedback stored successfully."
        )


# ============================================================
# 21. DATABASE VIEW
# ============================================================

st.header(
    "12. Stored Feedback"
)

try:

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    feedback_df = pd.read_sql_query(
        "SELECT * FROM feedback",
        connection
    )

    connection.close()

    if feedback_df.empty:

        st.info(
            "No feedback has been stored yet."
        )

    else:

        st.dataframe(
            feedback_df,
            use_container_width=True
        )

except Exception as error:

    st.error(
        f"Could not read feedback database: {error}"
    )