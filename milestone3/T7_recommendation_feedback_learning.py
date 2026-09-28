import os
import pandas as pd


# ============================================================
# 1. LOAD FEEDBACK DATA
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "recommendation_feedback_dataset.csv"
)

print("Loading recommendation feedback dataset...")

df = pd.read_csv(DATASET_PATH)

print("Recommendation feedback dataset loaded successfully!")

print("\nFeedback Data:")
print(df)


# ============================================================
# 2. FEEDBACK STORAGE
# ============================================================

def store_feedback(
    user_id,
    recommendation,
    viewed,
    accepted,
    rejected,
    rating,
    preference
):

    new_feedback = pd.DataFrame([{
        "user_id": user_id,
        "recommendation": recommendation,
        "viewed": viewed,
        "accepted": accepted,
        "rejected": rejected,
        "rating": rating,
        "preference": preference
    }])

    return new_feedback


# ============================================================
# 3. CALCULATE FEEDBACK SCORE
# ============================================================

def calculate_feedback_score(row):

    score = 0

    # Recommendation was viewed
    if row["viewed"] == "Yes":
        score += 1

    # Recommendation was accepted
    if row["accepted"] == "Yes":
        score += 3

    # Recommendation was rejected
    if row["rejected"] == "Yes":
        score -= 3

    # User rating
    score += row["rating"]

    return score


# ============================================================
# 4. CALCULATE HISTORICAL FEEDBACK
# ============================================================

df["feedback_score"] = df.apply(
    calculate_feedback_score,
    axis=1
)

print("\n======================================")
print("HISTORICAL FEEDBACK SCORES")
print("======================================")

print(
    df[
        [
            "recommendation",
            "rating",
            "accepted",
            "rejected",
            "feedback_score"
        ]
    ]
)


# ============================================================
# 5. LEARN FROM USER PREFERENCE
# ============================================================

def calculate_preference_learning(
    recommendation,
    user_preference,
    dataset
):

    matching_rows = dataset[
        dataset["preference"] == user_preference
    ]

    if len(matching_rows) == 0:
        return 0

    matching_recommendations = matching_rows[
        matching_rows["recommendation"] == recommendation
    ]

    if len(matching_recommendations) == 0:
        return 0

    return matching_recommendations[
        "feedback_score"
    ].mean()


# ============================================================
# 6. RANK RECOMMENDATIONS USING FEEDBACK
# ============================================================

def rank_recommendations(
    recommendations,
    user_preference,
    dataset
):

    ranking_results = []

    for recommendation in recommendations:

        recommendation_rows = dataset[
            dataset["recommendation"]
            == recommendation
        ]

        # Average historical feedback
        if len(recommendation_rows) > 0:

            average_feedback = (
                recommendation_rows[
                    "feedback_score"
                ].mean()
            )

            average_rating = (
                recommendation_rows[
                    "rating"
                ].mean()
            )

        else:

            average_feedback = 0
            average_rating = 0

        # Preference learning
        preference_score = (
            calculate_preference_learning(
                recommendation,
                user_preference,
                dataset
            )
        )

        # Final learned score
        final_score = (
            average_feedback
            + preference_score
            + average_rating
        )

        ranking_results.append({
            "recommendation": recommendation,
            "average_feedback": average_feedback,
            "average_rating": average_rating,
            "preference_score": preference_score,
            "final_score": final_score
        })

    ranking_df = pd.DataFrame(
        ranking_results
    )

    ranking_df = ranking_df.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True)

    ranking_df.insert(
        0,
        "rank",
        range(1, len(ranking_df) + 1)
    )

    return ranking_df


# ============================================================
# 7. USER INFORMATION
# ============================================================

user_id = "U1"

user_preference = "meditation"


recommendations = [
    "Practice a short meditation session",
    "Try a guided breathing exercise",
    "Listen to calming music and relax",
    "Take a short walk and get some fresh air",
    "Write down your thoughts and feelings"
]


print("\n======================================")
print("MILESTONE 3 - TASK 7")
print("RECOMMENDATION FEEDBACK LEARNING")
print("======================================")

print("\nUser ID:", user_id)

print(
    "Current Preference:",
    user_preference
)


# ============================================================
# 8. INITIAL RANKING
# ============================================================

print("\n--------------------------------------")
print("INITIAL RECOMMENDATION RANKING")
print("--------------------------------------")

initial_ranking = rank_recommendations(
    recommendations,
    user_preference,
    df
)

for _, row in initial_ranking.iterrows():

    print(
        f"Rank {int(row['rank'])}: "
        f"{row['recommendation']}"
    )

    print(
        f"Score: {row['final_score']:.2f}"
    )


# ============================================================
# 9. CAPTURE NEW USER FEEDBACK
# ============================================================

print("\n======================================")
print("NEW USER FEEDBACK")
print("======================================")

new_feedback = store_feedback(
    user_id="U1",
    recommendation="Practice a short meditation session",
    viewed="Yes",
    accepted="Yes",
    rejected="No",
    rating=5,
    preference="meditation"
)

print("\nNew Feedback:")
print(new_feedback)


# ============================================================
# 10. ADD FEEDBACK TO STORAGE
# ============================================================

df_updated = pd.concat(
    [df, new_feedback],
    ignore_index=True
)

df_updated["feedback_score"] = (
    df_updated.apply(
        calculate_feedback_score,
        axis=1
    )
)


print("\nFeedback stored successfully!")


# ============================================================
# 11. UPDATED RANKING
# ============================================================

print("\n======================================")
print("UPDATED RECOMMENDATION RANKING")
print("======================================")

updated_ranking = rank_recommendations(
    recommendations,
    user_preference,
    df_updated
)

for _, row in updated_ranking.iterrows():

    print(
        f"Rank {int(row['rank'])}: "
        f"{row['recommendation']}"
    )

    print(
        f"Updated Score: "
        f"{row['final_score']:.2f}"
    )


# ============================================================
# 12. COMPARE BEFORE AND AFTER
# ============================================================

print("\n======================================")
print("FEEDBACK LEARNING COMPARISON")
print("======================================")

before = initial_ranking.iloc[0]

after = updated_ranking.iloc[0]

print(
    "Top recommendation before feedback:"
)

print(
    before["recommendation"]
)

print(
    "Score before feedback:",
    f"{before['final_score']:.2f}"
)

print("\nTop recommendation after feedback:")

print(
    after["recommendation"]
)

print(
    "Score after feedback:",
    f"{after['final_score']:.2f}"
)


# ============================================================
# 13. FINAL STATUS
# ============================================================

print("\n======================================")
print("TASK 7 COMPLETED")
print("======================================")