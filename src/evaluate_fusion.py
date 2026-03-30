import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from fusion_model import predict_image, predict_voice


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOICE_DATA = os.path.join(BASE_DIR, "dataset", "voice", "pd_speech_features.csv")

HEALTHY_DIR = os.path.join(BASE_DIR, "dataset", "handwriting", "Healthy")
PARKINSON_DIR = os.path.join(BASE_DIR, "dataset", "handwriting", "Parkinson")


def evaluate_fusion():

    print("\nEvaluating Fusion Model...\n")

    voice_df = pd.read_csv(VOICE_DATA)

    y_voice = voice_df["class"].values
    X_voice = voice_df.drop(["id", "class"], axis=1).values

    image_paths = []
    image_labels = []

    for img in os.listdir(HEALTHY_DIR):
        image_paths.append(os.path.join(HEALTHY_DIR, img))
        image_labels.append(0)

    for img in os.listdir(PARKINSON_DIR):
        image_paths.append(os.path.join(PARKINSON_DIR, img))
        image_labels.append(1)

    n = min(len(image_paths), len(X_voice))

    predictions = []
    actual = []
    probs = []

    for i in range(n):

        print(f"Processing {i+1}/{n}")

        image_pred = predict_image(image_paths[i])
        voice_pred = predict_voice(X_voice[i])

        fusion_prob = (image_pred + voice_pred) / 2
        fusion_pred = round(fusion_prob)

        predictions.append(fusion_pred)
        probs.append(fusion_prob)
        actual.append(y_voice[i])

    predictions = np.array(predictions)
    actual = np.array(actual)
    probs = np.array(probs)

    accuracy = accuracy_score(actual, predictions)

    print("\nFusion Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report\n")
    print(classification_report(actual, predictions))

    plot_confusion_matrix(actual, predictions)
    plot_roc(actual, probs)
    plot_precision_recall(actual, probs)


# ----------------------------
# Confusion Matrix
# ----------------------------

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0,1], ["Healthy","Parkinson"])
    plt.yticks([0,1], ["Healthy","Parkinson"])

    plt.show()


# ----------------------------
# ROC Curve
# ----------------------------

def plot_roc(y_true, probs):

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1])

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.show()


# ----------------------------
# Precision Recall Curve
# ----------------------------

def plot_precision_recall(y_true, probs):

    precision, recall, _ = precision_recall_curve(y_true, probs)

    plt.figure()
    plt.plot(recall, precision)

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.show()


if __name__ == "__main__":
    evaluate_fusion()