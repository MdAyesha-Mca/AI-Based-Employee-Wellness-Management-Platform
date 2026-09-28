import os
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "wellness_content_dataset.csv"
)

print("Loading wellness content dataset...")

df = pd.read_csv(DATASET_PATH)

print("Wellness content dataset loaded successfully!")

print("\nDataset:")
print(df)


# ============================================================
# 2. LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

print("\nLoading semantic embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Semantic embedding model loaded successfully!")


# ============================================================
# 3. CREATE WELLNESS CONTENT EMBEDDINGS
# ============================================================

print("\nCreating wellness content embeddings...")

content_embeddings = model.encode(
    df["content"].tolist()
)

print("Wellness content embeddings created successfully!")


# ============================================================
# 4. USER EMOTIONAL STATE
# ============================================================

user_emotion = "fear"

user_emotional_text = (
    "I am feeling scared and worried. "
    "I am afraid that something may go wrong "
    "and I need help to calm myself."
)


print("\n======================================")
print("MILESTONE 3 - TASK 5")
print("SEMANTIC WELLNESS CONTENT MATCHING")
print("======================================")

print("\nUser Emotional State:")
print("Emotion:", user_emotion)

print("\nUser Emotional Text:")
print(user_emotional_text)


# ============================================================
# 5. CREATE USER EMOTION EMBEDDING
# ============================================================

print("\nCreating user emotion embedding...")

user_embedding = model.encode(
    [user_emotional_text]
)

print("User emotion embedding created successfully!")


# ============================================================
# 6. CALCULATE SEMANTIC SIMILARITY
# ============================================================

similarity_scores = cosine_similarity(
    user_embedding,
    content_embeddings
)[0]


# ============================================================
# 7. ADD SIMILARITY SCORES TO DATASET
# ============================================================

df["semantic_similarity"] = similarity_scores


# ============================================================
# 8. SORT CONTENT BY SIMILARITY
# ============================================================

ranked_content = df.sort_values(
    by="semantic_similarity",
    ascending=False
).reset_index(drop=True)


ranked_content.insert(
    0,
    "rank",
    range(1, len(ranked_content) + 1)
)


# ============================================================
# 9. DISPLAY RANKING
# ============================================================

print("\n--------------------------------------")
print("SEMANTIC WELLNESS CONTENT RANKING")
print("--------------------------------------")

for _, row in ranked_content.iterrows():

    print(
        f"\nRank {int(row['rank'])}"
    )

    print(
        "Title:",
        row["title"]
    )

    print(
        "Emotion:",
        row["emotion"]
    )

    print(
        "Similarity Score:",
        f"{row['semantic_similarity']:.4f}"
    )


# ============================================================
# 10. TOP RELEVANT CONTENT
# ============================================================

top_content = ranked_content.iloc[0]

print("\n======================================")
print("TOP SEMANTICALLY RELEVANT CONTENT")
print("======================================")

print(
    "Title:",
    top_content["title"]
)

print(
    "Similarity Score:",
    f"{top_content['semantic_similarity']:.4f}"
)

print(
    "Content:",
    top_content["content"]
)


# ============================================================
# 11. FINAL STATUS
# ============================================================

print("\n======================================")
print("TASK 5 COMPLETED")
print("======================================")