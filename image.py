import os, cv2, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# ======================
# 1. Load Images
# ======================
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# ======================
# 3. Build Model
# ======================
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ======================
# 4. Train
# ======================
model.fit(X_train, y_train, epochs=5)

# ======================
# 5. Test Accuracy
# ======================
print("Accuracy:", model.evaluate(X_test, y_test)[1])

# ======================
# 6. Predict New Image
# ======================
img = cv2.imread("cats/cat_1.jpg")
img = cv2.resize(img, (64,64)) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

print("Dog 🐶" if pred[0][0] > 0.5 else "Cat 🐱")