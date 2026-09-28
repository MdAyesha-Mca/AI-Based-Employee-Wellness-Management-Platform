import os
import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ranking_recommendation_dataset.csv"
)

print("Loading recommendation ranking dataset...")

df = pd.read_csv(DATASET_PATH)

print("Recommendation ranking dataset loaded successfully!")

print("\nDataset:")
print(df)


# ============================================================
# 2. REMOVE DUPLICATE RECOMMENDATIONS
# ============================================================

df = df.drop_duplicates(
    subset=["recommendation"]
).reset_index(drop=True)

print("\nDuplicate recommendations removed.")

print(
    "Number of unique recommendations:",
    len(df)
)


# ============================================================
# 3. EMOTION RELEVANCE
# ============================================================

def calculate_emotion_relevance(
    user_emotion,
    recommendation_emotion
):

    if user_emotion == recommendation_emotion:
        return 1.0

    return 0.0


# ============================================================
# 4. EMOTIONAL INTENSITY
# ============================================================

def calculate_intensity_score(
    user_intensity,
    recommendation_intensity
):

    if user_intensity == recommendation_intensity:
        return 1.0

    if user_intensity == "High":
        if recommendation_intensity == "Medium":
            return 0.7
        elif recommendation_intensity == "Low":
            return 0.4

    if user_intensity == "Medium":
        if recommendation_intensity == "High":
            return 0.7
        elif recommendation_intensity == "Low":
            return 0.6

    if user_intensity == "Low":
        if recommendation_intensity == "Medium":
            return 0.7
        elif recommendation_intensity == "High":
            return 0.4

    return 0.0


# ============================================================
# 5. USER PREFERENCE MATCHING
# ============================================================

def calculate_preference_score(
    user_preference,
    recommendation_preference
):

    if user_preference == recommendation_preference:
        return 1.0

    return 0.0


# ============================================================
# 6. CONTENT SIMILARITY
# ============================================================

def calculate_content_similarity(
    user_content,
    recommendation_content
):

    if user_content == recommendation_content:
        return 1.0

    return 0.0


# ============================================================
# 7. PREVIOUS INTERACTION SCORE
# ============================================================

def calculate_previous_interaction_score(
    previous_interaction,
    recommendation_interaction
):

    if previous_interaction == recommendation_interaction:
        return 1.0

    return 0.0


# ============================================================
# 8. RECOMMENDATION HISTORY SCORE
# ============================================================

def calculate_history_score(
    recommendation_history,
    recommendation_history_value
):

    if recommendation_history == recommendation_history_value:
        return 1.0

    return 0.0


# ============================================================
# 9. DYNAMIC RECOMMENDATION RANKING
# ============================================================

def rank_recommendations(
    user_emotion,
    user_intensity,
    user_preference,
    user_content,
    previous_interaction,
    recommendation_history,
    dataset
):

    ranking_results = []

    for _, row in dataset.iterrows():

        # --------------------------------------------
        # Calculate individual scores
        # --------------------------------------------

        emotion_score = calculate_emotion_relevance(
            user_emotion,
            row["emotion"]
        )

        intensity_score = calculate_intensity_score(
            user_intensity,
            row["intensity"]
        )

        preference_score = calculate_preference_score(
            user_preference,
            row["preference"]
        )

        content_score = calculate_content_similarity(
            user_content,
            row["content"]
        )

        interaction_score = calculate_previous_interaction_score(
            previous_interaction,
            row["previous_interaction"]
        )

        history_score = calculate_history_score(
            recommendation_history,
            row["recommendation_history"]
        )

        # --------------------------------------------
        # Calculate final recommendation score
        # --------------------------------------------

        final_score = (
            emotion_score * 3
            + intensity_score * 2
            + preference_score * 2
            + content_score * 2
            + interaction_score * 1
            + history_score * 1
        )

        ranking_results.append({
            "recommendation":
                row["recommendation"],

            "emotion_relevance":
                emotion_score,

            "intensity_score":
                intensity_score,

            "preference_score":
                preference_score,

            "content_similarity":
                content_score,

            "previous_interaction":
                interaction_score,

            "recommendation_history":
                history_score,

            "final_score":
                final_score
        })

    ranking_df = pd.DataFrame(
        ranking_results
    )

    # --------------------------------------------
    # Sort dynamically by final score
    # --------------------------------------------

    ranking_df = ranking_df.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True)

    # Add ranking position

    ranking_df.insert(
        0,
        "rank",
        range(1, len(ranking_df) + 1)
    )

    return ranking_df


# ============================================================
# 10. TEST USER
# ============================================================

print("\n======================================")
print("MILESTONE 3 - TASK 4")
print("RECOMMENDATION RANKING MODEL")
print("======================================")


user_emotion = "fear"
user_intensity = "Medium"
user_preference = "meditation"
user_content = "meditation"
previous_interaction = "meditation"
recommendation_history = "meditation"


print("\nUser Information:")

print("Emotion:", user_emotion)
print("Intensity:", user_intensity)
print("Preference:", user_preference)
print("Content:", user_content)
print("Previous Interaction:", previous_interaction)
print(
    "Recommendation History:",
    recommendation_history
)


# ============================================================
# 11. GENERATE RANKING
# ============================================================

ranking = rank_recommendations(
    user_emotion,
    user_intensity,
    user_preference,
    user_content,
    previous_interaction,
    recommendation_history,
    df
)


# ============================================================
# 12. DISPLAY RANKING
# ============================================================

print("\n--------------------------------------")
print("DYNAMIC RECOMMENDATION RANKING")
print("--------------------------------------")

for _, row in ranking.iterrows():

    print(
        f"Rank {int(row['rank'])}: "
        f"{row['recommendation']}"
    )

    print(
        f"Score: {row['final_score']:.2f}"
    )

    print(
        f"Emotion relevance: "
        f"{row['emotion_relevance']:.2f}"
    )

    print(
        f"Intensity: "
        f"{row['intensity_score']:.2f}"
    )

    print(
        f"Preference: "
        f"{row['preference_score']:.2f}"
    )

    print(
        f"Content similarity: "
        f"{row['content_similarity']:.2f}"
    )

    print(
        f"Previous interaction: "
        f"{row['previous_interaction']:.2f}"
    )

    print(
        f"Recommendation history: "
        f"{row['recommendation_history']:.2f}"
    )

    print("--------------------------------------")


# ============================================================
# 13. TOP RECOMMENDATION
# ============================================================

top_recommendation = ranking.iloc[0]

print("\n======================================")
print("TOP RECOMMENDATION")
print("======================================")

print(
    "Recommendation:",
    top_recommendation["recommendation"]
)

print(
    "Recommendation Score:",
    f"{top_recommendation['final_score']:.2f}"
)


# ============================================================
# 14. LOW-RELEVANCE CHECK
# ============================================================

low_relevance = ranking[
    ranking["final_score"] <= 1
]

print("\n======================================")
print("LOW-RELEVANCE CHECK")
print("======================================")

if len(low_relevance) > 0:
    print(
        "Low-relevance recommendations detected:",
        len(low_relevance)
    )
else:
    print(
        "No extremely low-relevance recommendations."
    )


# ============================================================
# 15. FINAL STATUS
# ============================================================

print("\n======================================")
print("TASK 4 COMPLETED")
print("======================================")