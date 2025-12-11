import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

print("🔄 Retraining model with custom data...")

print("Loading original dataset...")
df1 = pd.read_csv('spam.csv', encoding='latin-1')
df1 = df1.iloc[:, :2]
df1.columns = ['label', 'message']
df1 = df1.dropna()

try:
    df2 = pd.read_csv('custom_spam_data.csv')
    print(f"✅ Loaded {len(df2)} custom examples")
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"📊 Total dataset: {len(df)} messages")
except FileNotFoundError:
    print("⚠️ No custom data found, using original only")
    df = df1

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

print("Preprocessing data...")
df['clean_message'] = df['message'].apply(preprocess_text)
X = df['clean_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=2,
    max_df=0.8,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)

from sklearn.metrics import accuracy_score
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained! Accuracy: {accuracy*100:.2f}%")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved successfully!")