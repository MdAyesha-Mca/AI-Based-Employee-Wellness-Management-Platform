import streamlit as st
import pandas as pd

st.title("Milestone 1 – Text Ingestion & Baseline Sentiment")

def validate_text(text: str) -> bool:
    """Basic format check: non-empty after stripping whitespace."""
    return bool(text and text.strip())

def preprocess(text: str) -> str:
    """Placeholder preprocessing step (lowercase + strip)."""
    return text.strip().lower()

st.header("1. Manual Text Input")
text_input = st.text_area("Enter your text:")

if st.button("Process Text"):
    if validate_text(text_input):
        st.success("Valid text received!")
        processed = preprocess(text_input)
        st.write("Original:", text_input)
        st.write("Preprocessed:", processed)
    else:
        st.error("Please enter valid text.")

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
                processed = preprocess(content)
                st.write("Preprocessed preview:", processed[:500])
            else:
                st.error("The uploaded .txt file is empty or invalid.")

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded .csv file is empty.")
            else:
                st.success(f"Valid .csv file: {uploaded_file.name}")
                st.write("Preview:", df.head())

                # Try to find a text column to validate/preprocess
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
                    valid_rows["preprocessed"] = valid_rows[text_col].apply(lambda x: preprocess(str(x)))
                    st.write(valid_rows.head())
                else:
                    st.warning("No text column detected in CSV.")

        else:
            st.error("Unsupported file type.")

    except Exception as e:
        st.error(f"Error reading file: {e}")