# ==============================
# 1. Import Libraries
# ==============================
import cv2, os, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# ==============================
# 2. Load Images from Folders
# ==============================
data, labels = [], []
for folder in os.listdir("."):
    if not os.path.isdir(folder): continue
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None: continue
        
        img = cv2.resize(img, (64,64)) / 255.0
        data.append(img)
        labels.append(folder)

X = np.array(data)
y = LabelEncoder().fit_transform(labels)
# ======================
# 2. Split Data
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)


# ==============================
# 5. Build CNN Model (MULTI-CLASS)
# ==============================
num_classes = len(np.unique(y))

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')   # ✅ FIXED
])

# ==============================
# 6. Compile Model
# ==============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # ✅ FIXED
    metrics=['accuracy']
)

# ==============================
# 7. Train Model
# ==============================
model.fit(X_train, y_train, epochs=15, validation_split=0.2)

# ==============================
# 8. Evaluate Model
# ==============================
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

# ==============================
# 9. Single Image Prediction
# ==============================
img = cv2.imread("cats/cat_1.jpg")
img = cv2.resize(img, (64,64)) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

print("Dog 🐶" if pred[0][0] > 0.5 else "Cat 🐱")