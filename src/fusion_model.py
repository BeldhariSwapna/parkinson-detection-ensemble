import numpy as np
import pandas as pd
import joblib
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==============================
# Load Image Model
# ==============================

image_model = load_model("models/efficientnet_parkinson_model.keras")

# ==============================
# Load Voice Models
# ==============================

voice_dnn = load_model("models/voice_dnn_model.keras")

xgb = joblib.load("models/voice_xgb_model.pkl")
rf = joblib.load("models/voice_rf.pkl")
svm = joblib.load("models/voice_svm.pkl")

scaler = joblib.load("models/voice_scaler.pkl")
pca = joblib.load("models/voice_pca.pkl")

feature_names = joblib.load("models/voice_feature_names.pkl")

# ==============================
# Image Prediction
# ==============================

def predict_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(img, (192, 192))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = image_model.predict(img)

    return int(prediction[0][0] > 0.5)


# ==============================
# Voice Prediction
# ==============================

def predict_voice(features):

    # convert to dataframe (fix sklearn warning)
    features = pd.DataFrame([features], columns=feature_names)

    features = scaler.transform(features)

    features = pca.transform(features)

    # DNN
    dnn_pred = (voice_dnn.predict(features) > 0.5).astype(int)[0][0]

    # ML models
    xgb_pred = xgb.predict(features)[0]

    rf_pred = rf.predict(features)[0]

    svm_pred = svm.predict(features)[0]

    # ensemble
    score = (
        0.4 * dnn_pred +
        0.2 * xgb_pred +
        0.2 * rf_pred +
        0.2 * svm_pred
    )

    final_pred = int(score >= 0.5)

    return final_pred


# ==============================
# Fusion Prediction
# ==============================

def fusion_prediction(image_path, voice_features):

    image_result = predict_image(image_path)

    voice_result = predict_voice(voice_features)

    print("\nImage Prediction :", image_result)

    print("Voice Prediction :", voice_result)

    # fusion decision
    final = int((image_result + voice_result) >= 1)

    print()

    if final == 1:
        diagnosis = "Parkinson Disease Detected"
    else:
        diagnosis = "Healthy"

    print("Final Diagnosis :", diagnosis)

    return diagnosis


# ==============================
# Test Example
# ==============================

if __name__ == "__main__":

    image_path = "test_image.png"

    # random voice features for testing
    voice_features = np.random.rand(len(feature_names))

    fusion_prediction(image_path, voice_features)