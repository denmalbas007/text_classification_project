import joblib
import pandas as pd

def predict(input_filepath: str, model_filepath: str, vectorizer_filepath: str, output_filepath: str):
    model = joblib.load(model_filepath)
    vectorizer = joblib.load(vectorizer_filepath)
    df = pd.read_csv(input_filepath)
    X = vectorizer.transform(df['text']).toarray()
    predictions = model.predict(X)
    df['predictions'] = predictions
    df.to_csv(output_filepath, index=False)
    print(f"Predictions saved to {output_filepath}")

if __name__ == "__main__":
    predict('data/raw/newsgroups.csv', 'models/text_classification_model.joblib', 'models/vectorizer.joblib', 'data/processed/newsgroups_predictions.csv')
