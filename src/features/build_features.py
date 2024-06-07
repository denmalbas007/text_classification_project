import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def preprocess_data(input_filepath: str, output_filepath: str, vectorizer_filepath: str):
    df = pd.read_csv(input_filepath)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['target']
    feature_df = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())
    feature_df['target'] = y
    feature_df.to_csv(output_filepath, index=False)
    joblib.dump(vectorizer, vectorizer_filepath)
    print(f"Data preprocessed and saved to {output_filepath}")
    print(f"Vectorizer saved to {vectorizer_filepath}")

if __name__ == "__main__":
    preprocess_data('data/raw/newsgroups.csv', 'data/processed/newsgroups_features.csv', 'models/vectorizer.joblib')
