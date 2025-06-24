import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Download necessary nltk data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sample dataset with more rows for better training
data = {
    "text": [
        "I love this product!",
        "Worst experience ever.",
        "Very satisfied with the service.",
        "Not worth the price.",
        "Absolutely fantastic!",
        "Horrible, will never buy again.",
        "This is the best thing I bought.",
        "I am extremely disappointed.",
        "Great value for the money.",
        "Terrible customer service.",
        "I am happy with my purchase.",
        "It broke after one use.",
        "Highly recommend this to everyone.",
        "I hate this so much.",
        "Superb quality and fast delivery.",
        "Not what I expected, very bad.",
        "Exceeded my expectations!",
        "Poor packaging and slow shipping.",
        "Very pleased with the support.",
        "Do not waste your money on this."
    ],
    "sentiment": [
        1,0,1,0,1,0,1,0,1,0,
        1,0,1,0,1,0,1,0,1,0
    ]
}

df = pd.DataFrame(data)

print("Sample data:")
print(df.head())

# Function to clean text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

print("\nCleaning text data...")
df['clean_text'] = df['text'].apply(clean_text)

print("\nCleaned sample text:")
print(df[['text','clean_text']].head())

# Generate and show word cloud for all cleaned text
all_text = " ".join(df['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Cleaned Text")
plt.show()

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Train Logistic Regression with balanced class weights
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Show some test samples with predictions
print("\nTest samples and predictions:")
test_indices = y_test.index
for i in range(len(y_test)):
    idx = test_indices[i]
    print(f"Text: {df.loc[idx, 'text']}")
    print(f"Cleaned: {df.loc[idx, 'clean_text']}")
    print(f"Actual Sentiment: {y_test.iloc[i]} | Predicted Sentiment: {y_pred[i]}\n")

print("✅ Sentiment Analysis Completed.")
