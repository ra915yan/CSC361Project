import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_and_clean_data(file_path):
    """Loads dataset and handles missing values."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Check for missing values and fill them
    initial_nulls = df['email'].isnull().sum()
    if initial_nulls > 0:
        print(f" Found {initial_nulls} missing entry in 'email' column. Handling it gracefully...")
        df['email'] = df['email'].fillna('')
        
    return df

def prepare_splits(df):
    """Splits target and features into stratified train and test groups."""
    X = df['email']
    y = df['label']
    
    # Stratify ensures the distribution of spam/ham remains equal in both splits
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
    print(f"\n[INFO] Pipeline artifacts successfully saved to directory: '{model_dir}/'")

if __name__ == "__main__":
    # 1. Pipeline Input Configuration
    DATA_PATH = 'spam_or_not_spam.csv'  # Update if paths differ
    
    # 2. Data Preparation
    data = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_splits(data)
    
    # 3. Feature Extraction (TF-IDF Vectorization)
    print("\nVectorizing raw text datasets via TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    
    # Production Champion Model: Random Forest
    print("\nTraining Random Forest Ensemble Classifier (this might take a few seconds)...")
    rf_raw = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_raw.fit(X_train_vec, y_train)
    rf_cm, production_model = evaluate_model(rf_raw, "Production Random Forest Classifier", X_test_vec, y_test)
    
    # 5. Serialization (Saving the best pipeline model)
    save_artifacts(vectorizer, production_model)