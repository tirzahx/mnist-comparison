import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(np.int32)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Logistic Regression Train")
log_reg = LogisticRegression(max_iter=1000, solver="lbfgs")
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {acc_log:.4f}")

print("SVM Train")
svm_clf = SVC(kernel="rbf", gamma=0.05)
svm_clf.fit(X_train[:5000], y_train[:5000])
y_pred_svm = svm_clf.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {acc_svm:.4f}")

print("ANN Train")
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model = Sequential([
    Flatten(input_shape=(784,)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train_cat, epochs=10, batch_size=128,
                    validation_split=0.1, verbose=2)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"ANN Accuracy: {test_acc:.4f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred_ann = model.predict(X_test).argmax(axis=1)

cm_log = confusion_matrix(y_test, y_pred_log)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_ann = confusion_matrix(y_test, y_pred_ann)

acc_log = accuracy_score(y_test, y_pred_log)
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_ann = accuracy_score(y_test, y_pred_ann)

fig = plt.figure(figsize=(18, 10))

ax1 = plt.subplot2grid((2,3), (0,0), colspan=1)
ax1.plot(history.history["loss"], label="Train Loss")
ax1.plot(history.history["val_loss"], label="Validation Loss")
ax1.set_title("ANN Loss Curve")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2 = plt.subplot2grid((2,3), (0,1), colspan=2)
ax2.plot(history.history["accuracy"], label="Train Accuracy")
ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
ax2.set_title("ANN Accuracy Curve")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

axes_cm = [plt.subplot2grid((2,3), (1,i)) for i in range(3)]

ConfusionMatrixDisplay(cm_log).plot(ax=axes_cm[0], cmap="Blues", colorbar=False)
axes_cm[0].set_title(f"Logistic Regression\nAccuracy: {acc_log:.2%}")

ConfusionMatrixDisplay(cm_svm).plot(ax=axes_cm[1], cmap="Blues", colorbar=False)
axes_cm[1].set_title(f"SVM\nAccuracy: {acc_svm:.2%}")

ConfusionMatrixDisplay(cm_ann).plot(ax=axes_cm[2], cmap="Blues", colorbar=False)
axes_cm[2].set_title(f"ANN\nAccuracy: {acc_ann:.2%}")

plt.suptitle("MNIST Model Performance Summary", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()