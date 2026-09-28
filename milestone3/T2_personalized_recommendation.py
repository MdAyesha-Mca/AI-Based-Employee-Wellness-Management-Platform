import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ==========================================
# MILESTONE 3 - TASK 2
# PERSONALIZED RECOMMENDATION MODEL
# ==========================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "recommendation_dataset.csv"
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading recommendation dataset...")

df = pd.read_csv(DATASET_PATH)

print("Recommendation dataset loaded successfully!")

print("\nDataset:")
print(df)


# ==========================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "emotion",
    "intensity",
    "preference",
    "previous_interaction",
    "recommendation_history",
    "trend",
    "recommendation"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ==========================================
# 3. PREPARE FEATURES
# ==========================================

feature_columns = [
    "emotion",
    "intensity",
    "preference",
    "previous_interaction",
    "recommendation_history",
    "trend"
]

X = df[feature_columns]

y = df["recommendation"]


# Convert categorical values into numerical values
X_encoded = pd.get_dummies(X)


print("\nEncoded Features:")
print(X_encoded)


# ==========================================
# 4. TRAIN ML MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_encoded,
    y
)

print("\nPersonalized recommendation model trained successfully!")


# ==========================================
# 5. PERSONALIZED RECOMMENDATION FUNCTION
# ==========================================

def get_recommendation(
    emotion,
    intensity,
    preference,
    previous_interaction,
    recommendation_history,
    trend
):

    user_input = pd.DataFrame([{

        "emotion": emotion,

        "intensity": intensity,

        "preference": preference,

        "previous_interaction":
            previous_interaction,

        "recommendation_history":
            recommendation_history,

        "trend": trend

    }])


    # Convert categorical input into numbers
    user_input_encoded = pd.get_dummies(
        user_input
    )


    # Make sure input has exactly
    # the same columns as training data
    user_input_encoded = user_input_encoded.reindex(
        columns=X_encoded.columns,
        fill_value=0
    )


    recommendation = model.predict(
        user_input_encoded
    )[0]


    return recommendation


# ==========================================
# 6. TEST PERSONALIZED RECOMMENDATION
# ==========================================

print("\n======================================")
print("PERSONALIZED RECOMMENDATION TEST")
print("======================================")


# These values come from Task 1
detected_emotion = "fear"
emotion_intensity = "Medium"

# User information
user_preference = "meditation"

previous_interaction = "breathing"

recommendation_history = "breathing"

emotional_trend = "increasing"


print("\nUser Emotional Information:")
print(
    "Detected Emotion:",
    detected_emotion
)

print(
    "Emotion Intensity:",
    emotion_intensity
)

print(
    "User Preference:",
    user_preference
)

print(
    "Previous Interaction:",
    previous_interaction
)

print(
    "Recommendation History:",
    recommendation_history
)

print(
    "Emotional Trend:",
    emotional_trend
)


# Generate recommendation
recommendation = get_recommendation(

    detected_emotion,

    emotion_intensity,

    user_preference,

    previous_interaction,

    recommendation_history,

    emotional_trend

)


print("\n--------------------------------------")

print("PERSONALIZED RECOMMENDATION:")

print(recommendation)

print("--------------------------------------")


print("\n======================================")
print("TASK 2 COMPLETED")
print("======================================")