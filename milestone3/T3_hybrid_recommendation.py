import os
import pandas as pd
from collections import defaultdict


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hybrid_recommendation_dataset.csv"
)

print("Loading hybrid recommendation dataset...")

df = pd.read_csv(DATASET_PATH)

print("Hybrid recommendation dataset loaded successfully!")

print("\nDataset:")
print(df)


# ============================================================
# 2. RULE-BASED RECOMMENDATIONS
# ============================================================

def rule_based_recommendation(emotion, intensity):

    rules = {
        ("fear", "High"):
            "Try a guided breathing exercise",

        ("fear", "Medium"):
            "Practice a short meditation session",

        ("fear", "Low"):
            "Practice a short meditation session",

        ("sadness", "High"):
            "Listen to calming music and relax",

        ("sadness", "Medium"):
            "Write down your thoughts and feelings",

        ("sadness", "Low"):
            "Listen to calming music and relax",

        ("anger", "High"):
            "Take a short walk and get some fresh air",

        ("anger", "Medium"):
            "Take a short walk and get some fresh air",

        ("joy", "High"):
            "Continue with a healthy physical activity",

        ("joy", "Medium"):
            "Continue with a healthy physical activity"
    }

    return rules.get(
        (emotion, intensity),
        "Take a short break and practice relaxation"
    )


# ============================================================
# 3. CONTENT-BASED FILTERING
# ============================================================

def content_based_score(preference, recommendation):

    preference_mapping = {
        "meditation": [
            "Try a guided breathing exercise",
            "Practice a short meditation session"
        ],

        "music": [
            "Listen to calming music and relax"
        ],

        "physical activity": [
            "Take a short walk and get some fresh air"
        ],

        "activity": [
            "Continue with a healthy physical activity"
        ],

        "writing": [
            "Write down your thoughts and feelings"
        ]
    }

    if recommendation in preference_mapping.get(
        preference, []
    ):
        return 1

    return 0


# ============================================================
# 4. USER PREFERENCE MATCHING
# ============================================================

def preference_score(preference, recommendation):

    if content_based_score(
        preference,
        recommendation
    ) == 1:
        return 1

    return 0


# ============================================================
# 5. EMOTION SIMILARITY
# ============================================================

def emotion_similarity(
    current_emotion,
    recommendation,
    dataset
):

    emotion_recommendations = dataset[
        dataset["emotion"] == current_emotion
    ]["recommendation"].tolist()

    if recommendation in emotion_recommendations:
        return 1

    return 0


# ============================================================
# 6. COLLABORATIVE FILTERING
# ============================================================

def collaborative_score(
    current_emotion,
    recommendation,
    dataset
):

    matching_users = dataset[
        dataset["emotion"] == current_emotion
    ]

    if recommendation in matching_users[
        "recommendation"
    ].values:

        return 1

    return 0


# ============================================================
# 7. HISTORICAL USER BEHAVIOR
# ============================================================

def historical_behavior_score(
    previous_interaction,
    recommendation_history,
    recommendation,
    dataset
):

    score = 0

    matching_previous = dataset[
        dataset["previous_interaction"]
        == previous_interaction
    ]

    if recommendation in matching_previous[
        "recommendation"
    ].values:
        score += 1

    matching_history = dataset[
        dataset["recommendation_history"]
        == recommendation_history
    ]

    if recommendation in matching_history[
        "recommendation"
    ].values:
        score += 1

    return score


# ============================================================
# 8. HYBRID RECOMMENDATION ENGINE
# ============================================================

def hybrid_recommendation(
    emotion,
    intensity,
    preference,
    previous_interaction,
    recommendation_history,
    dataset
):

    recommendations = dataset[
        "recommendation"
    ].unique()

    scores = defaultdict(float)

    rule_recommendation = rule_based_recommendation(
        emotion,
        intensity
    )

    for recommendation in recommendations:

        # Rule-based score
        if recommendation == rule_recommendation:
            rule_score = 1
        else:
            rule_score = 0

        # Content-based score
        content_score = content_based_score(
            preference,
            recommendation
        )

        # Preference matching score
        pref_score = preference_score(
            preference,
            recommendation
        )

        # Collaborative filtering score
        collab_score = collaborative_score(
            emotion,
            recommendation,
            dataset
        )

        # Emotion similarity score
        emotion_score = emotion_similarity(
            emotion,
            recommendation,
            dataset
        )

        # Historical behavior score
        history_score = historical_behavior_score(
            previous_interaction,
            recommendation_history,
            recommendation,
            dataset
        )

        # ----------------------------------------------------
        # HYBRID SCORE
        # ----------------------------------------------------

        total_score = (
            rule_score * 3
            + content_score * 2
            + pref_score * 2
            + collab_score * 1
            + emotion_score * 2
            + history_score * 1
        )

        scores[recommendation] = total_score

    best_recommendation = max(
        scores,
        key=scores.get
    )

    return best_recommendation, scores


# ============================================================
# 9. TEST THE HYBRID ENGINE
# ============================================================

print("\n======================================")
print("MILESTONE 3 - TASK 3")
print("HYBRID RECOMMENDATION ENGINE")
print("======================================")


# User information

detected_emotion = "fear"
emotion_intensity = "Medium"
user_preference = "meditation"
previous_interaction = "breathing"
recommendation_history = "breathing"


print("\nUser Information:")
print("Detected Emotion:", detected_emotion)
print("Emotion Intensity:", emotion_intensity)
print("User Preference:", user_preference)
print("Previous Interaction:", previous_interaction)
print("Recommendation History:", recommendation_history)


# Generate recommendation

final_recommendation, recommendation_scores = (
    hybrid_recommendation(
        detected_emotion,
        emotion_intensity,
        user_preference,
        previous_interaction,
        recommendation_history,
        df
    )
)


# ============================================================
# 10. DISPLAY SCORES
# ============================================================

print("\n--------------------------------------")
print("HYBRID RECOMMENDATION SCORES")
print("--------------------------------------")

for recommendation, score in recommendation_scores.items():

    print(
        f"{recommendation}: {score}"
    )


# ============================================================
# 11. FINAL RECOMMENDATION
# ============================================================

print("\n--------------------------------------")
print("FINAL HYBRID RECOMMENDATION")
print("--------------------------------------")

print(final_recommendation)

print("\n======================================")
print("TASK 3 COMPLETED")
print("======================================")