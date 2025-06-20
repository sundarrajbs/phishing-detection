# Phishing Detection with Machine Learning

This project implements a simple phishing email detection tool using Python, leveraging Natural Language Processing (NLP) and machine learning to classify emails as phishing or legitimate. It uses a Naive Bayes classifier and TF-IDF vectorization to analyze email text, making it a great learning exercise for cybersecurity and data science enthusiasts.

## Project Overview
The script processes a small dataset of emails, cleans and tokenizes the text, converts it into numerical features using TF-IDF, and trains a Naive Bayes classifier to predict whether an email is phishing (1) or legitimate (0). It includes error handling for NLTK resource downloads and an SSL workaround for compatibility.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/phishing-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   ```
   pandas
   numpy
   scikit-learn
   nltk
   ```
3. Run the script:
   ```bash
   python phishing_detection.py
   ```

## Dataset
The project uses a small, hardcoded dataset for demonstration:
- 5 sample emails with labels (1 for phishing, 0 for legitimate).
- For production, use larger datasets like the Enron email corpus or Kaggle’s phishing email datasets.

## Functions and Methods Used

### Custom Functions
1. **preprocess_text(text)**
   - **Purpose**: Cleans and preprocesses email text for NLP by removing noise and preparing it for feature extraction.
   - **Input**: A string containing the email text.
   - **Output**: A cleaned string of tokenized words.
   - **Steps**:
     - Converts text to lowercase for consistency.
     - Removes URLs using `re.sub(r'http\S+', '', text)` to eliminate links common in phishing emails.
     - Removes special characters and numbers using `re.sub(r'[^a-zA-Z\s]', '', text)` to focus on textual content.
     - Tokenizes the text into words using `nltk.tokenize.word_tokenize`.
     - Filters out stop words (e.g., “the”, “is”) using `nltk.corpus.stopwords` to reduce noise.
     - Joins tokens back into a single string for vectorization.
   - **Example**:
     ```python
     text = "Urgent: Verify your account at http://fake.com!"
     preprocess_text(text)  # Returns: "urgent verify account"
     ```

2. **predict_phishing(email)**
   - **Purpose**: Classifies a new email as phishing or legitimate using the trained model.
   - **Input**: A string containing the email text.
   - **Output**: A string (“Phishing” or “Legitimate”) based on the model’s prediction.
   - **Steps**:
     - Calls `preprocess_text` to clean the input email.
     - Converts the cleaned text to a TF-IDF feature vector using `vectorizer.transform`.
     - Uses the trained Naive Bayes model (`model.predict`) to predict the label (1 or 0).
     - Returns “Phishing” for label 1 or “Legitimate” for label 0.
   - **Example**:
     ```python
     email = "Urgent: Your account needs verification, click here: http://fake-site.com"
     predict_phishing(email)  # Returns: "Phishing"
     ```

### Library Methods
1. **pandas.DataFrame(data)**
   - **Module**: `pandas`
   - **Purpose**: Creates a structured DataFrame from the `data` dictionary containing emails and labels.
   - **Usage**: Organizes the dataset into a table with columns `email` and `label`, enabling easy manipulation (e.g., applying `preprocess_text` to create `cleaned_email`).
   - **Example**:
     ```python
     df = pd.DataFrame(data)
     ```

2. **re.sub(pattern, repl, string)**
   - **Module**: `re`
   - **Purpose**: Performs regular expression-based text replacement to clean email content.
   - **Usage**: Removes URLs (`http\S+`) and special characters (`[^a-zA-Z\s]`) in `preprocess_text`.
   - **Example**:
     ```python
     text = re.sub(r'http\S+', '', "Visit http://fake.com")  # Returns: "Visit "
     ```

3. **nltk.tokenize.word_tokenize(text)**
   - **Module**: `nltk.tokenize`
   - **Purpose**: Splits text into individual words (tokens) using NLTK’s pre-trained tokenizer.
   - **Usage**: Tokenizes cleaned email text in `preprocess_text` for stop word removal.
   - **Example**:
     ```python
     word_tokenize("urgent verify account")  # Returns: ['urgent', 'verify', 'account']
     ```

4. **nltk.corpus.stopwords.words('english')**
   - **Module**: `nltk.corpus`
   - **Purpose**: Provides a list of common English stop words to filter out irrelevant words.
   - **Usage**: Removes stop words in `preprocess_text` to focus on meaningful terms.
   - **Example**:
     ```python
     [word for word in ['urgent', 'the', 'account'] if word not in stopwords.words('english')]
     # Returns: ['urgent', 'account']
     ```

5. **TfidfVectorizer(max_features=500)**
   - **Module**: `sklearn.feature_extraction.text`
   - **Purpose**: Converts text into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
   - **Usage**: Transforms cleaned emails into a feature matrix (`X`) with up to 500 features, used for training the model.
   - **Methods**:
     - `fit_transform`: Fits the vectorizer to the data and transforms text to a TF-IDF matrix.
     - `transform`: Converts new text (e.g., in `predict_phishing`) to a TF-IDF vector using the fitted vocabulary.
   - **Example**:
     ```python
     X = vectorizer.fit_transform(df['cleaned_email']).toarray()
     ```

6. **train_test_split(X, y, test_size=0.2, chaussure converse femme blanche, random_state=42)**
   - **Module**: `sklearn.model_selection`
   - **Purpose**: Splits data into training and testing sets for model evaluation.
   - **Usage**: Divides the feature matrix `X` and labels `y` into 80% training and 20% testing sets.
   - **Example**:
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

7. **MultinomialNB().fit(X_train, y_train)**
   - **Module**: `sklearn.naive_bayes`
   - **Purpose**: Trains a Naive Bayes classifier for text classification.
   - **Usage**: Fits the model to the training data to learn patterns for classifying emails as phishing or legitimate.
   - **Example**:
     ```python
     model.fit(X_train, y_train)
     ```

8. **accuracy_score(y_test, y_pred)**
   - **Module**: `sklearn.metrics`
   - **Purpose**: Calculates the accuracy of the model’s predictions.
   - **Usage**: Measures the proportion of correct predictions on the test set.
   - **Example**:
     ```python
     print("Accuracy:", accuracy_score(y_test, y_pred))
     ```

9. **classification_report(y_test, y_pred)**
   - **Module**: `sklearn.metrics`
   - **Purpose**: Generates a detailed report of precision, recall, and F1-score for model evaluation.
   - **Usage**: Provides metrics to assess the model’s performance on phishing detection.
   - **Example**:
     ```python
     print("Classification Report:\n", classification_report(y_test, y_pred))
     ```

10. **ssl._create_unverified_context()**
    - **Module**: `ssl`
    - **Purpose**: Bypasses SSL certificate verification for NLTK downloads (workaround for SSL issues).
    - **Usage**: Temporarily disables SSL verification to ensure NLTK resources download successfully.
    - **Example**:
      ```python
      ssl._create_default_https_context = ssl._create_unverified_context
      ```

## Usage
- Run the script to train the model and test it on a sample email:
  ```python
  python phishing_detection.py
  ```
- The script outputs the model’s accuracy, classification report, and a prediction for a sample email.

## Notes
- **Dataset Limitation**: The small dataset is for demonstration. Use larger datasets for better accuracy.
- **SSL Workaround**: The script includes an SSL bypass for NLTK downloads. Ensure system certificates are updated for secure downloads (see troubleshooting section).
- **Troubleshooting**: If NLTK resources fail to download, manually download `punkt`, `punkt_tab`, and `stopwords` from the NLTK data repository and set `nltk.data.path`.

## Future Improvements
- Use a larger, real-world dataset (e.g., Enron or Kaggle phishing datasets).
- Experiment with advanced models like LogisticRegression or BERT.
- Add a web interface using Flask or Django for user-friendly email input.
- Enhance preprocessing with lemmatization or advanced NLP techniques.

## Learning Outcomes
This project teaches:
- NLP basics: Text cleaning, tokenization, stop words, and TF-IDF.
- Machine learning: Training and evaluating a Naive Bayes classifier.
- Cybersecurity: Understanding phishing detection challenges.
- Dependency management: Handling NLTK and SSL issues.