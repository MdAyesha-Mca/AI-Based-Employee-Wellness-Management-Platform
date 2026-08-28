import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

st.title("Milestone 1 - Text Ingestion & Baseline Sentiment")


def validate_text(text: str) -> bool:
    """Basic format check: non-empty after stripping whitespace."""
    return bool(text and text.strip())


def preprocess(text: str) -> dict:
    """Full preprocessing pipeline with step-by-step output."""
    result = {}

    if not text or not text.strip():
        return {"error": "Empty text provided."}

    original = text

    cleaned = re.sub(r'\s+', ' ', text).strip()
    result["after_space_cleanup"] = cleaned

    no_special = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)
    result["after_special_char_removal"] = no_special

    no_punct = no_special.translate(str.maketrans('', '', string.punctuation))
    result["after_punctuation_removal"] = no_punct

    tokens = word_tokenize(no_punct.lower())
    result["tokens"] = tokens

    filtered_tokens = [t for t in tokens if t not in stop_words]
    result["after_stopword_removal"] = filtered_tokens

    lemmatized = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    result["after_lemmatization"] = lemmatized

    final_text = " ".join(lemmatized)
    result["final_processed_text"] = final_text
    result["original_length"] = len(original)
    result["final_length"] = len(final_text)

    return result


def analyze_sentiment(text: str) -> dict:
    """Run VADER sentiment analysis and return scores dynamically."""
    if not text or not text.strip():
        return {"error": "Empty text provided."}

    scores = sia.polarity_scores(text)

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


# ---------------- TASK 1: Manual Text Input ----------------
st.header("1. Manual Text Input")
text_input = st.text_area("Enter your text:")

if st.button("Process Text"):
    if validate_text(text_input):
        st.success("Valid text received!")
        output = preprocess(text_input)
        st.write("Original:", text_input)
        st.write("Preprocessed:", output.get("final_processed_text", ""))
    else:
        st.error("Please enter valid text.")

# ---------------- TASK 1: File Upload ----------------
st.header("2. File Upload (.txt or .csv)")
uploaded_file = st.file_uploader(
    "Upload TXT or CSV file",
    type=["txt", "csv"]
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_type == "txt":
            content = uploaded_file.read().decode("utf-8")
            if validate_text(content):
                st.success(f"Valid .txt file: {uploaded_file.name}")
                st.write("Preview:", content[:500])
                output = preprocess(content)
                st.write("Preprocessed preview:", output.get("final_processed_text", "")[:500])
            else:
                st.error("The uploaded .txt file is empty or invalid.")

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded .csv file is empty.")
            else:
                st.success(f"Valid .csv file: {uploaded_file.name}")
                st.write("Preview:", df.head())

                text_col = None
                for col in df.columns:
                    if df[col].dtype == object:
                        text_col = col
                        break

                if text_col:
                    st.info(f"Using column '{text_col}' as text data.")
                    valid_rows = df[df[text_col].apply(lambda x: validate_text(str(x)))]
                    invalid_count = len(df) - len(valid_rows)
                    st.write(f"Valid rows: {len(valid_rows)} | Invalid/empty rows: {invalid_count}")
                    valid_rows["preprocessed"] = valid_rows[text_col].apply(
                        lambda x: preprocess(str(x)).get("final_processed_text", "")
                    )
                    st.write(valid_rows.head())
                else:
                    st.warning("No text column detected in CSV.")

        else:
            st.error("Unsupported file type.")

    except Exception as e:
        st.error(f"Error reading file: {e}")

# ---------------- TASK 2: Preprocessing Validation ----------------
st.header("3. Preprocessing Validation (Task 2)")

st.write("Test the preprocessing pipeline against different cases.")

test_cases = {
    "Normal sentence": "The quick brown foxes are jumping over the lazy dogs!",
    "Special characters": "Hello!!! @World #2026 $$$ %^&*()",
    "Punctuation heavy": "Wow... this is, amazing!!! Isn't it?",
    "Empty text": "",
    "Only spaces": "     ",
    "Repeated spaces": "This    has     many      spaces",
    "Very short": "Hi",
    "Very long": (
        "This is a much longer piece of text that contains many words and "
        "should test how the pipeline handles longer sequences of tokens "
        "including stop words like the and is and a which should all be "
        "removed during preprocessing."
    ),
}

selected_case = st.selectbox("Choose a test case:", list(test_cases.keys()))
test_input = st.text_area("Or enter your own text to test:", value=test_cases[selected_case])

if st.button("Run Preprocessing Test"):
    output = preprocess(test_input)

    if "error" in output:
        st.error(output["error"])
    else:
        st.success("Preprocessing completed successfully!")

        st.write("**Original text:**", test_input)
        st.write("**Original length:**", output["original_length"], "characters")

        with st.expander("Step 1: After removing repeated spaces"):
            st.write(output["after_space_cleanup"])

        with st.expander("Step 2: After removing special characters"):
            st.write(output["after_special_char_removal"])

        with st.expander("Step 3: After removing punctuation"):
            st.write(output["after_punctuation_removal"])

        with st.expander("Step 4: Tokens (Tokenization)"):
            st.write(output["tokens"])

        with st.expander("Step 5: After stop-word removal"):
            st.write(output["after_stopword_removal"])

        with st.expander("Step 6: After lemmatization"):
            st.write(output["after_lemmatization"])

        st.write("**Final processed text:**", output["final_processed_text"])
        st.write("**Final length:**", output["final_length"], "characters")

# ---------------- TASK 3: VADER Sentiment Validation ----------------
st.header("4. VADER Sentiment Validation (Task 3)")

st.write("Test the baseline sentiment analysis module using VADER.")

sentiment_test_cases = {
    "Clearly positive": "I absolutely love this product! It's amazing and works perfectly.",
    "Clearly negative": "This is terrible. I hate how badly this was made.",
    "Neutral statement": "The report was submitted on Monday.",
    "Mixed sentiment": "The service was good but the delivery was very late.",
    "Sarcastic (edge case)": "Oh great, another Monday. Just what I needed.",
    "Empty text": "",
}

selected_sentiment_case = st.selectbox(
    "Choose a sentiment test case:",
    list(sentiment_test_cases.keys())
)
sentiment_input = st.text_area(
    "Or enter your own text to test sentiment:",
    value=sentiment_test_cases[selected_sentiment_case],
    key="sentiment_input"
)

if st.button("Run Sentiment Test"):
    result = analyze_sentiment(sentiment_input)

    if "error" in result:
        st.error(result["error"])
    else:
        if result["label"] == "Positive":
            st.success(f"Sentiment: {result['label']} :)")
        elif result["label"] == "Negative":
            st.error(f"Sentiment: {result['label']} :(")
        else:
            st.info(f"Sentiment: {result['label']} :|")

        st.write("**Compound score:**", result["compound"])
        st.write("**Positive score:**", result["positive"])
        st.write("**Negative score:**", result["negative"])
        st.write("**Neutral score:**", result["neutral"])