import os
import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "recommendation_explainability_dataset.csv"
)

print("Loading recommendation explainability dataset...")

df = pd.read_csv(DATASET_PATH)

print("Recommendation explainability dataset loaded successfully!")


# ============================================================
# 2. USER INPUT
# ============================================================

user_emotion = "fear"
user_intensity = "High"
user_preference = "meditation"
previous_interaction = "breathing"
recommendation_history = "breathing"


print("\n======================================")
print("MILESTONE 3 - TASK 8")
print("RECOMMENDATION EXPLAINABILITY")
print("======================================")

print("\nUser Information:")
print("Detected Emotion:", user_emotion)
print("Emotion Intensity:", user_intensity)
print("User Preference:", user_preference)
print("Previous Interaction:", previous_interaction)
print("Recommendation History:", recommendation_history)


# ============================================================
# 3. CALCULATE RELEVANCE
# ============================================================

def calculate_relevance(row):

    score = 0

    # Emotion relevance
    if row["emotion"] == user_emotion:
        score += 3

    # Intensity relevance
    if row["intensity"] == user_intensity:
        score += 2

    # Preference relevance
    if row["preference"] == user_preference:
        score += 2

    # Previous interaction relevance
    if row["previous_interaction"] == previous_interaction:
        score += 1

    # Recommendation history relevance
    if row["recommendation_history"] == recommendation_history:
        score += 1

    return score


df["relevance_score"] = df.apply(
    calculate_relevance,
    axis=1
)


# ============================================================
# 4. RANK RECOMMENDATIONS
# ============================================================

ranking = df.sort_values(
    by="relevance_score",
    ascending=False
).reset_index(drop=True)

ranking.insert(
    0,
    "rank",
    range(1, len(ranking) + 1)
)


print("\n======================================")
print("RECOMMENDATION RANKING")
print("======================================")

for _, row in ranking.iterrows():

    print(
        f"Rank {int(row['rank'])}: "
        f"{row['recommendation']}"
    )

    print(
        f"Relevance Score: "
        f"{row['relevance_score']}"
    )


# ============================================================
# 5. GENERATE DYNAMIC EXPLANATION
# ============================================================

def generate_explanation(row):

    reasons = []

    # --------------------------------------------------------
    # Emotion
    # --------------------------------------------------------

    if row["emotion"] == user_emotion:

        reasons.append(
            f"{user_emotion.capitalize()} emotion was detected"
        )

    # --------------------------------------------------------
    # Intensity
    # --------------------------------------------------------

    if row["intensity"] == user_intensity:

        reasons.append(
            f"{user_intensity.lower()} emotional intensity matches"
        )

    # --------------------------------------------------------
    # User Preference
    # --------------------------------------------------------

    if row["preference"] == user_preference:

        reasons.append(
            f"user preference matches {user_preference} content"
        )

    # --------------------------------------------------------
    # Previous Interaction
    # --------------------------------------------------------

    if row["previous_interaction"] == previous_interaction:

        reasons.append(
            "a similar activity was used previously"
        )

    # --------------------------------------------------------
    # Recommendation History
    # --------------------------------------------------------

    if row["recommendation_history"] == recommendation_history:

        reasons.append(
            "similar recommendation appears in recommendation history"
        )

    # --------------------------------------------------------
    # No matching factors
    # --------------------------------------------------------

    if len(reasons) == 0:

        reasons.append(
            "the recommendation has general wellness relevance"
        )

    return reasons


# ============================================================
# 6. GENERATE EXPLANATION FOR TOP RECOMMENDATION
# ============================================================

top_recommendation = ranking.iloc[0]

reasons = generate_explanation(
    top_recommendation
)


print("\n======================================")
print("TOP RECOMMENDATION")
print("======================================")

print(
    "\nRecommended:",
    top_recommendation["recommendation"]
)

print(
    "Relevance Score:",
    top_recommendation["relevance_score"]
)

print("\nReason:")

for reason in reasons:

    print("•", reason)


# ============================================================
# 7. SHOW EXPLANATION FOR EVERY RECOMMENDATION
# ============================================================

print("\n======================================")
print("EXPLANATIONS FOR ALL RECOMMENDATIONS")
print("======================================")

for _, row in ranking.iterrows():

    reasons = generate_explanation(row)

    print(
        f"\n{row['recommendation']}"
    )

    print(
        f"Relevance Score: "
        f"{row['relevance_score']}"
    )

    print("Why selected:")

    for reason in reasons:

        print("•", reason)


# ============================================================
# 8. VERIFICATION
# ============================================================

print("\n======================================")
print("EXPLAINABILITY VERIFICATION")
print("======================================")

print(
    "Detected emotion checked: Yes"
)

print(
    "Emotion intensity checked: Yes"
)

print(
    "User preference checked: Yes"
)

print(
    "Historical behavior checked: Yes"
)

print(
    "Recommendation relevance checked: Yes"
)

print(
    "Dynamic explanation generated: Yes"
)


# ============================================================
# 9. FINAL STATUS
# ============================================================

print("\n======================================")
print("TASK 8 COMPLETED")
print("======================================")