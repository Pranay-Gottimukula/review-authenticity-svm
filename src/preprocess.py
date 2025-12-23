import pandas as pd
import string
import re
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv("../data/raw/archive/deceptive-opinion.csv")

# Keep only relevant columns
df = df.drop(["hotel", "polarity", "source"], axis=1)

# Helper functions

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

# NLP tools
# stop_words = set(stopwords.words("english"))
# lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""

    # lowercase
    text = text.lower()

    # remove punctuation
    text = remove_punctuation(text)

    # remove digits
    text = remove_numbers(text)

    # remove extra whitespace
    text = remove_extra_spaces(text)

    # Stopword removal
    # text = " ".join([word for word in text.split() if word not in stop_words])

    # Lemmatization
    # text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# Shuffle (optional)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save processed version
df.to_csv("../data/clean/deceptive-opinion-clean.csv", index=False)

print(df.head())