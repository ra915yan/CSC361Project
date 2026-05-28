# CSC 361 Spam Detector Project

This project is an AI spam detector for the CSC 361 Artificial Intelligence course. It uses machine learning to classify email text as either spam or not spam.

## Project Idea

The project implements an intelligent spam filter. The system learns from a labeled email dataset and then predicts whether a new email message is spam.

## AI Technique Used

The project uses supervised machine learning:

- TF-IDF vectorization to convert email text into numerical features
- Logistic Regression for binary classification
- A trained model saved with `joblib`

## Dataset

The dataset is stored in:

```text
data/spam_or_not_spam.csv
```

Labels:

- `0` = Not spam
- `1` = Spam

## Project Files

```text
data/
  spam_or_not_spam.csv
  spam_distribution.png
  top_words_ham.png
  top_words_spam.png

models/
  spam_detector_model.pkl
  tfidf_vectorizer.pkl

src/
  config.py
  gui.py
  plot.py
  SpamNotSpamDataset.py
  training_code.py

run_gui.bat
Final_Report_CSC361_Spam_Detector.docx
```

## Requirements

Install the required Python packages:

```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```

## How to Run the GUI

On Windows, open the project folder and run:

```bash
run_gui.bat
```

Or run the GUI directly:

```bash
python src/gui.py
```

## How to Use

1. Click **Random Sample** to load an email from the dataset.
2. Check the ground-truth label shown beside the button.
3. Click **Predict / Run Model** to classify the text.
4. You can also type your own email message into the text box and run the model.

## Training the Model

The trained model is already included in the `models` folder. To retrain the model, run:

```bash
python src/training_code.py
```

## Generating Plots

To regenerate the data analysis plots, run:

```bash
python src/plot.py
```

## Result Summary

The model was tested on a held-out test set and achieved approximately **98.5% accuracy**.

## Deliverables

This repository includes:

- Source code
- Dataset
- Trained model files
- Data visualization images
- GUI launcher
- Final report
- README file
