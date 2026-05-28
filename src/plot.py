# src/explore_data.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import config

def generate_class_distribution(df):
    """Generates the basic Spam vs Ham bar count chart."""
    plt.figure(figsize=(7, 5))
    sns.countplot(x='label', data=df, hue='label', palette='viridis', legend=False)
    plt.title('Spam vs. Not Spam Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Label (0: Ham, 1: Spam)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(ticks=[0, 1], labels=['Not Spam (0)', 'Spam (1)'])
    
    path = os.path.join(config.DATA_DIR, "spam_distribution.png")
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"[SUCCESS] Saved: {path}")

def generate_text_length_distribution(df):
    """Generates a histogram showing message word length distributions."""
    df['word_count'] = df['email'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x='word_count', hue='label', element='step', 
                 stat='density', common_norm=False, kde=True, palette='viridis')
    plt.title('Email Word Count Distribution by Class', fontsize=12, fontweight='bold')
    plt.xlabel('Word Count', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.xlim(0, df['word_count'].quantile(0.95)) # Cap long tail outliers safely
    
    path = os.path.join(config.DATA_DIR, "text_length_distribution.png")
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"[SUCCESS] Saved: {path}")

def generate_top_words_chart(df, target_label, label_name, filename):
    """Extracts and plots the top 15 most frequent unique words."""
    sub_df = df[df['label'].astype(str) == str(target_label)]
    
    # Exclude standard english stop words ('the', 'and', etc.)
    vec = CountVectorizer(stop_words='english', max_features=15)
    bag_of_words = vec.fit_transform(sub_df['email'].astype(str))
    sum_words = bag_of_words.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words_df = pd.DataFrame(words_freq, columns=['Word', 'Count'])

    plt.figure(figsize=(7, 5))
    sns.barplot(x='Count', y='Word', data=words_df, hue='Word', palette='magma', legend=False)
    plt.title(f'Top 15 Common Words in {label_name}', fontsize=12, fontweight='bold')
    
    path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"[SUCCESS] Saved: {path}")

if __name__ == "__main__":
    print("--- Starting Data Analytics Image Pipeline ---")
    
    # Ensure data save folder directory path exists
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
        
    # Read flat CSV data array
    if not os.path.exists(config.DATA1_PATH):
        print(f"Error: Could not locate your dataset at {config.DATA1_PATH}")
    else:
        dataset_df = pd.read_csv(config.DATA1_PATH)
        
        # Run and export all three data visualizations
        generate_class_distribution(dataset_df)
        generate_text_length_distribution(dataset_df)
        generate_top_words_chart(dataset_df, target_label=1, label_name="Spam", filename="top_words_spam.png")
        generate_top_words_chart(dataset_df, target_label=0, label_name="Ham", filename="top_words_ham.png")
        
        print("--- All analytic charts generated and exported successfully! ---")