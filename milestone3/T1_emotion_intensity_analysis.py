import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# MILESTONE 3 - TASK 1
# Emotion Intensity & Emotional State Analysis
# ==========================================

# Path to the trained model from Milestone 2
MODEL_PATH = os.path.join(
    "..",
    "milestone 2",
    "distilbert_multilabel_emotion_model"
)

# Six emotion labels used in Milestone 2
EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]


# ==========================================
# 1. LOAD MODEL
# ==========================================

print("Loading emotion model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Emotion model loaded successfully!")


# ==========================================
# 2. EMOTION DETECTION
# ==========================================

def detect_emotions(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    # Convert model outputs into probabilities
    probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    emotion_probabilities = {}

    for i, emotion in enumerate(EMOTIONS):

        emotion_probabilities[emotion] = float(
            probabilities[i]
        )

    # Find the emotion with highest probability
    dominant_emotion = max(
        emotion_probabilities,
        key=emotion_probabilities.get
    )

    confidence = emotion_probabilities[
        dominant_emotion
    ]

    return emotion_probabilities, dominant_emotion, confidence


# ==========================================
# 3. EMOTIONAL INTENSITY
# ==========================================

def calculate_intensity(confidence):

    if confidence >= 0.70:

        return "High"

    elif confidence >= 0.40:

        return "Medium"

    else:

        return "Low"


# ==========================================
# 4. MULTIPLE EMOTIONS
# ==========================================

def detect_multiple_emotions(emotion_probabilities):

    threshold = 0.40

    multiple_emotions = []

    for emotion, probability in emotion_probabilities.items():

        if probability >= threshold:

            multiple_emotions.append(emotion)

    return multiple_emotions


# ==========================================
# 5. POSITIVE / NEGATIVE POLARITY
# ==========================================

def calculate_polarity(emotion_probabilities):

    positive_score = (
        emotion_probabilities["joy"]
        + emotion_probabilities["surprise"]
    )

    negative_score = (
        emotion_probabilities["sadness"]
        + emotion_probabilities["anger"]
        + emotion_probabilities["fear"]
        + emotion_probabilities["disgust"]
    )

    if positive_score > negative_score + 0.10:

        polarity = "Positive"

    elif negative_score > positive_score + 0.10:

        polarity = "Negative"

    else:

        polarity = "Mixed"

    return (
        round(positive_score, 3),
        round(negative_score, 3),
        polarity
    )


# ==========================================
# 6. MIXED EMOTIONAL STATE
# ==========================================

def detect_mixed_state(multiple_emotions):

    if len(multiple_emotions) > 1:

        return "Yes"

    else:

        return "No"


# ==========================================
# 7. EMOTION SEVERITY
# ==========================================

def calculate_severity(confidence):

    if confidence >= 0.70:

        return "High"

    elif confidence >= 0.40:

        return "Medium"

    else:

        return "Low"


# ==========================================
# 8. COMPLETE EMOTIONAL STATE ANALYSIS
# ==========================================

def analyze_emotional_state(text):

    emotion_probabilities, dominant_emotion, confidence = (
        detect_emotions(text)
    )

    intensity = calculate_intensity(
        confidence
    )

    multiple_emotions = detect_multiple_emotions(
        emotion_probabilities
    )

    positive_score, negative_score, polarity = (
        calculate_polarity(
            emotion_probabilities
        )
    )

    mixed_state = detect_mixed_state(
        multiple_emotions
    )

    severity = calculate_severity(
        confidence
    )

    final_state = {

        "dominant_emotion": dominant_emotion,

        "confidence": round(
            confidence,
            3
        ),

        "intensity_score": round(
            confidence,
            3
        ),

        "intensity": intensity,

        "multiple_emotions":
            multiple_emotions,

        "positive_score":
            positive_score,

        "negative_score":
            negative_score,

        "polarity":
            polarity,

        "mixed_state":
            mixed_state,

        "severity":
            severity
    }

    return (
        emotion_probabilities,
        final_state
    )


# ==========================================
# 9. TEST
# ==========================================

if __name__ == "__main__":

    test_text = (
        "I am excited about getting the job, "
        "but I am also scared that I might fail."
    )

    print("\n======================================")
    print("MILESTONE 3 - TASK 1")
    print("EMOTION INTENSITY & EMOTIONAL STATE")
    print("======================================")

    print("\nInput Text:")
    print(test_text)

    emotion_probabilities, final_state = (
        analyze_emotional_state(test_text)
    )

    print("\nEmotion Probabilities:")

    for emotion, probability in (
        emotion_probabilities.items()
    ):

        print(
            f"{emotion}: "
            f"{probability:.3f}"
        )

    print("\n--------------------------------------")
    print("FINAL EMOTIONAL STATE")
    print("--------------------------------------")

    print(
        "Dominant Emotion:",
        final_state["dominant_emotion"]
    )

    print(
        "Confidence:",
        final_state["confidence"]
    )

    print(
        "Intensity Score:",
        final_state["intensity_score"]
    )

    print(
        "Intensity:",
        final_state["intensity"]
    )

    print(
        "Multiple Emotions:",
        final_state["multiple_emotions"]
    )

    print(
        "Positive Score:",
        final_state["positive_score"]
    )

    print(
        "Negative Score:",
        final_state["negative_score"]
    )

    print(
        "Polarity:",
        final_state["polarity"]
    )

    print(
        "Mixed Emotional State:",
        final_state["mixed_state"]
    )

    print(
        "Severity:",
        final_state["severity"]
    )

    print("\n======================================")
    print("TASK 1 COMPLETED")
    print("======================================")