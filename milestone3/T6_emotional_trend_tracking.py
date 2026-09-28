import os
import pandas as pd
from collections import Counter


# ============================================================
# 1. LOAD HISTORICAL EMOTION DATA
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "emotion_history_dataset.csv"
)

print("Loading emotional history dataset...")

df = pd.read_csv(DATASET_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    by="date"
).reset_index(drop=True)

print("Emotional history dataset loaded successfully!")

print("\nHistorical Emotion Data:")
print(df)


# ============================================================
# 2. EMOTION FREQUENCY
# ============================================================

emotion_frequency = Counter(
    df["emotion"]
)

print("\n======================================")
print("EMOTION FREQUENCY")
print("======================================")

for emotion, count in emotion_frequency.items():

    print(
        f"{emotion}: {count}"
    )


# ============================================================
# 3. DOMINANT EMOTION
# ============================================================

dominant_emotion = (
    emotion_frequency.most_common(1)[0][0]
)

dominant_count = (
    emotion_frequency.most_common(1)[0][1]
)

print("\n======================================")
print("DOMINANT EMOTION")
print("======================================")

print(
    "Dominant Emotion:",
    dominant_emotion
)

print(
    "Frequency:",
    dominant_count
)


# ============================================================
# 4. INTENSITY CONVERSION
# ============================================================

intensity_values = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

df["intensity_value"] = (
    df["intensity"].map(intensity_values)
)


# ============================================================
# 5. EMOTION INTENSITY OVER TIME
# ============================================================

print("\n======================================")
print("EMOTION INTENSITY OVER TIME")
print("======================================")

for _, row in df.iterrows():

    print(
        f"{row['date'].date()} - "
        f"{row['emotion']} - "
        f"{row['intensity']} "
        f"({row['intensity_value']})"
    )


# ============================================================
# 6. RECENT EMOTIONAL STATE
# ============================================================

recent_state = df.iloc[-1]

print("\n======================================")
print("RECENT EMOTIONAL STATE")
print("======================================")

print(
    "Recent Emotion:",
    recent_state["emotion"]
)

print(
    "Recent Intensity:",
    recent_state["intensity"]
)

print(
    "Recent Polarity:",
    recent_state["polarity"]
)

print(
    "Recent Date:",
    recent_state["date"].date()
)


# ============================================================
# 7. POSITIVE / NEGATIVE TREND
# ============================================================

positive_count = (
    (df["polarity"] == "Positive").sum()
)

negative_count = (
    (df["polarity"] == "Negative").sum()
)

total_records = len(df)

positive_percentage = (
    positive_count / total_records
) * 100

negative_percentage = (
    negative_count / total_records
) * 100


print("\n======================================")
print("POSITIVE / NEGATIVE TREND")
print("======================================")

print(
    "Positive Records:",
    positive_count
)

print(
    "Negative Records:",
    negative_count
)

print(
    "Positive Percentage:",
    f"{positive_percentage:.2f}%"
)

print(
    "Negative Percentage:",
    f"{negative_percentage:.2f}%"
)


if positive_count > negative_count:

    overall_trend = "Positive"

elif negative_count > positive_count:

    overall_trend = "Negative"

else:

    overall_trend = "Balanced"


print(
    "Overall Emotional Trend:",
    overall_trend
)


# ============================================================
# 8. REPEATED EMOTIONAL PATTERNS
# ============================================================

repeated_emotions = {
    emotion: count
    for emotion, count in emotion_frequency.items()
    if count > 1
}


print("\n======================================")
print("REPEATED EMOTIONAL PATTERNS")
print("======================================")

if repeated_emotions:

    for emotion, count in repeated_emotions.items():

        print(
            f"{emotion} repeated "
            f"{count} times"
        )

else:

    print(
        "No repeated emotional patterns found."
    )


# ============================================================
# 9. RECENT EMOTION WINDOW
# ============================================================

recent_window = df.tail(3)

recent_emotions = (
    recent_window["emotion"].tolist()
)

recent_emotion_counts = Counter(
    recent_emotions
)

recent_dominant_emotion = (
    recent_emotion_counts
    .most_common(1)[0][0]
)


print("\n======================================")
print("RECENT EMOTIONAL PATTERN")
print("======================================")

print(
    "Recent emotions:",
    recent_emotions
)

print(
    "Recent dominant emotion:",
    recent_dominant_emotion
)


# ============================================================
# 10. EMOTION TREND DETECTION
# ============================================================

if recent_dominant_emotion == dominant_emotion:

    emotion_trend = (
        "Historical and recent emotion are consistent"
    )

else:

    emotion_trend = (
        "Recent emotional state differs "
        "from historical dominant emotion"
    )


print("\n======================================")
print("EMOTION TREND DETECTION")
print("======================================")

print(
    "Trend:",
    emotion_trend
)


# ============================================================
# 11. PERSONALIZED RECOMMENDATION
# ============================================================

def generate_recommendation(
    recent_emotion,
    recent_intensity,
    repeated_emotions,
    overall_trend
):

    # Repeated fear pattern
    if (
        recent_emotion == "fear"
        and "fear" in repeated_emotions
    ):

        return (
            "Continue with breathing exercises "
            "and guided meditation."
        )

    # Repeated sadness pattern
    elif (
        recent_emotion == "sadness"
        and "sadness" in repeated_emotions
    ):

        return (
            "Try journaling and calming activities."
        )

    # Anger pattern
    elif recent_emotion == "anger":

        return (
            "Take a short walk and practice "
            "relaxation exercises."
        )

    # Positive recent state
    elif (
        recent_emotion == "joy"
        and overall_trend == "Positive"
    ):

        return (
            "Continue with healthy activities "
            "that maintain your positive mood."
        )

    # Default recommendation
    else:

        return (
            "Practice a short mindfulness "
            "or relaxation activity."
        )


personalized_recommendation = generate_recommendation(
    recent_state["emotion"],
    recent_state["intensity"],
    repeated_emotions,
    overall_trend
)


# ============================================================
# 12. FINAL USER STATE
# ============================================================

print("\n======================================")
print("CURRENT USER EMOTIONAL STATE")
print("======================================")

print(
    "Recent Emotion:",
    recent_state["emotion"]
)

print(
    "Recent Intensity:",
    recent_state["intensity"]
)

print(
    "Historical Dominant Emotion:",
    dominant_emotion
)

print(
    "Overall Trend:",
    overall_trend
)

print(
    "Repeated Patterns:",
    repeated_emotions
)

print(
    "Personalized Recommendation:",
    personalized_recommendation
)


# ============================================================
# 13. FINAL STATUS
# ============================================================

print("\n======================================")
print("TASK 6 COMPLETED")
print("======================================")