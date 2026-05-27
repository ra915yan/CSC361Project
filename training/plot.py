import pandas as pd

# Load the dataset
df = pd.read_csv('spam_or_not_spam.csv')

# Inspect the dataset
print(df.head())
print(df.info())
print(df['label'].value_counts())



import matplotlib.pyplot as plt
import seaborn as sns


def generate_plot(df, output_path = "data/label"):
    # Plotting the distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Spam vs. Not Spam Distribution')
    plt.xlabel('Label (0: Not Spam, 1: Spam)')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Not Spam (0)', 'Spam (1)'])
    plt.savefig('spam_distribution.png')

    plt.savefig()

