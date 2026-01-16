from preprocess import train_data, val_data, test_data
from model import build_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

print("🚀 Starting Tomato Disease Model Training...")

model = build_model()

# ---------------- TRAINING ----------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ---------------- EVALUATION ----------------
print("\n📌 Evaluating on TEST set")
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ---------------- SAVE MODEL ----------------
model.save("tomato_model.keras")
print("✅ Model saved successfully!")

# ---------------- PLOT ACCURACY GRAPH ----------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Graph")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Graph")
plt.legend()

plt.savefig("training_graphs.png")
plt.close()

# ---------------- CONFUSION MATRIX ----------------
print("📊 Generating Confusion Matrix...")

y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# ---------------- CLASSIFICATION REPORT ----------------
report = classification_report(y_true, y_pred)
print("\n📄 Classification Report\n")
print(report)

with open("classification_report.txt","w") as f:
    f.write(report)

print("\n🎯 Training Completed Successfully!")
