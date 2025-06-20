import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
import ssl

# Bypass SSL verification
try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Explicitly download punkt_tab
nltk.download('stopwords', quiet=True) 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




# Sample dataset (in a real project, use a larger dataset like Enron or Kaggle's phishing email datasets)
data = {
    'email': [
        "Your account has been compromised, click here to reset your password: http://fake-link.com",
        "Dear user, win a $1000 gift card by completing this survey: http://scam-survey.com",
        "Meeting scheduled for tomorrow at 10 AM, please confirm your attendance.",
        "Urgent: Verify your bank details to avoid account suspension: http://phish-bank.com",
        "Here's the project report you requested, attached for your review.",
    ],
    'label': [1, 1, 0, 1, 0]  # 1 = phishing, 0 = legitimate
}
df = pd.DataFrame(data)

# Preprocess text: clean and tokenize email content
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned_email'] = df['email'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['cleaned_email']).toarray()
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict if a new email is phishing
def predict_phishing(email):
    cleaned_email = preprocess_text(email)
    email_vector = vectorizer.transform([cleaned_email]).toarray()
    prediction = model.predict(email_vector)
    return "Phishing" if prediction[0] == 1 else "Legitimate"

# Example usage
new_email = "Urgent: Your account needs verification, click here: http://fake-site.com"
print(f"Email: {new_email}\nPrediction: {predict_phishing(new_email)}")