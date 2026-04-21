# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("email_spam.csv", encoding='latin1')
df.rename(columns={'v1': 'type', 'v2': 'content'}, inplace=True)
print(df.head())
print(df.info())

# ==============================
# 3. Text Cleaning
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['content'] = df['content'].apply(clean_text)

# ==============================
# 4. Encode Labels
# ==============================
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# ==============================
# 5. TF-IDF Feature Extraction
# ==============================
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['content'])
y = df['type']

# ==============================
# 6. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ==============================
# 7. Naive Bayes Model
# ==============================
model = MultinomialNB()
model.fit(X_train, y_train)
 
# ==============================
# 8. Prediction & Evaluation
# ==============================
y_pred = model.predict(X_test)

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 9. Test with New Text
# ==============================
sample = ["send this imediately"]

sample_clean = [clean_text(s) for s in sample]
sample_vec = vectorizer.transform(sample_clean)

prediction = model.predict(sample_vec)

print("\nPrediction:", le.inverse_transform(prediction))