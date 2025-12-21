import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# -----------------------------
# Paths and settings
# -----------------------------
MODEL_PATH = r"C:\Facial_Emotion_Recognition\outputs\models\CNNModel_fer_3emo.h5"
TEST_DIR = r"C:\Facial_Emotion_Recognition\inputs\fer"  # <-- change if different
OUTPUT_DIR = r"C:\Facial_Emotion_Recognition\outputs\confusion_matrix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTIONS = ['Angry', 'Happy', 'Surprise']

# -----------------------------
# Load the trained model
# -----------------------------
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# Load test images
# -----------------------------
print("Loading test data...")
X_test = []
y_test = []

for idx, emotion in enumerate(EMOTIONS):
    folder = os.path.join(TEST_DIR, emotion)
    if not os.path.exists(folder):
        print(f"Warning: Folder not found - {folder}")
        continue
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (48, 48))
        img = img.astype("float32") / 255.0
        X_test.append(img)
        y_test.append(idx)

X_test = np.expand_dims(np.array(X_test), -1)
y_test = np.array(y_test)

print(f"Loaded {len(X_test)} test samples")

# -----------------------------
# Evaluate model
# -----------------------------
print("Evaluating model...")
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=EMOTIONS)
print(report)

cm = confusion_matrix(y_test, y_pred)

# -----------------------------
# Save confusion matrix plot
# -----------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
cm_path = os.path.join(OUTPUT_DIR, "conf_matrix.png")
plt.savefig(cm_path)
print(f"\nConfusion matrix saved to {cm_path}")
try:
    print("\n✅ Testing completed successfully!")
except UnicodeEncodeError:
    print("\nTesting completed successfully!")

# Path to the confusion matrix image
conf_matrix_path = r"C:\Facial_Emotion_Recognition\outputs\confusion_matrix\conf_matrix.png"

# Copy to static folder so Flask can display it
dashboard_image_path = r"C:\Facial_Emotion_Recognition\static\conf_matrix.png"
shutil.copy(conf_matrix_path, dashboard_image_path)


