import os


#return absolute path for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Path
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA1_PATH = os.path.join(DATA_DIR,"spam_or_not_spam.csv")


# Model Path

MODEL_DIR = os.path.join(BASE_DIR,"models")

MODEL_PATH = os.path.join(MODEL_DIR, "spam_detector_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")



# for tuning

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURE = 5000
