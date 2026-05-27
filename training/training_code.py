import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def load_and_clean_data(file_path):
    """Loads dataset and handles missing values."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    initial_nulls = df['email'].isnull().sum()
    if initial_nulls > 0:
        print(f" Found {initial_nulls} missing entry in 'email' column. Handling it gracefully...")
        df['email'] = df['email'].fillna('')
        
    return df

def prepare_splits(df):
    """Splits target and features into stratified train and test groups."""
    X = df['email']
    y = df['label']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate_model(model, name, X_test_vec, y_test):
    """Predicts, prints performance metrics, and extracts confusion matrix values."""
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n==================== {name} Performance ====================")
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nDetailed Classification Report:")
    
    cm = confusion_matrix(y_test, preds)
    return cm, model

def save_artifacts(vectorizer, model, model_dir="trained_models"):
    """Saves the fitted vectorizer and model weights for later production/inference."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(model_dir, "spam_detector_model.pkl"))

if __name__ == "__main__":
    # 1. Pipeline Input Configuration
    DATA_PATH = 'spam_or_not_spam.csv'  # Update if paths differ
    
    # 2. Data Preparation
    data = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_splits(data)
    
    
    # ====== NEW CODE TO SAVE THE TEST SET ======
    print("Saving isolated test set for GUI testing...")
    test_df = pd.DataFrame({
        'email': X_test,
        'label': y_test
    })
    # Saves it right next to your raw data
    test_df.to_csv('data_test_split.csv', index=False)
    # ===========================================
    
    # 3. Feature Extraction (TF-IDF Vectorization)
    print("\nVectorizing raw text datasets via TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    
    # Production Champion Model: Logistic Regression
    print("\nTraining Logistic Regression Classifier...")
    lr_raw = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_raw.fit(X_train_vec, y_train)
    lr_cm, production_model = evaluate_model(lr_raw, "Production Logistic Regression Classifier", X_test_vec, y_test)
    
    # 5. Serialization (Saving the best pipeline model)
    save_artifacts(vectorizer, production_model)
