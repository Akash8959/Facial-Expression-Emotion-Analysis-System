import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---- Config ----
MODEL_PATH = r"C:/Facial_Emotion_Recognition/outputs/models/CNNModel_fer_3emo.h5"
DATASET_PATH = r"C:/Facial_Emotion_Recognition/inputs/fer"
IMAGE_SIZE = (48, 48)
CLASS_MAPPING = {0: 'Angry', 1: 'Happy', 2: 'Surprise'}
CONF_MATRIX_FOLDER = r"C:/Facial_Emotion_Recognition/outputs/confusion_matrix"
os.makedirs(CONF_MATRIX_FOLDER, exist_ok=True)

# ---- Load trained model ----
model = load_model(MODEL_PATH)

# ---- Prepare test data ----
X_test, y_test = [], []
for label_idx, emotion in CLASS_MAPPING.items():
    emotion_folder = os.path.join(DATASET_PATH, emotion)
    if not os.path.exists(emotion_folder):
        continue
    for img_name in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        X_test.append(img)
        y_test.append(label_idx)

X_test = np.array(X_test)
y_test = np.array(y_test)

# ---- Predict on test data ----
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# ---- Metrics ----
report = classification_report(y_test, y_pred_classes, target_names=list(CLASS_MAPPING.values()))
cm = confusion_matrix(y_test, y_pred_classes)

# ---- Save confusion matrix image ----
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(len(CLASS_MAPPING)), CLASS_MAPPING.values())
plt.yticks(np.arange(len(CLASS_MAPPING)), CLASS_MAPPING.values())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
conf_matrix_file = os.path.join(CONF_MATRIX_FOLDER, "conf_matrix_test.png")
plt.savefig(conf_matrix_file)
plt.close()

# ---- Return outputs as strings ----
with open(os.path.join(CONF_MATRIX_FOLDER, "report.txt"), "w") as f:
    f.write(report)
print(report)
