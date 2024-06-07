import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.datasets import fetch_20newsgroups
import logging

# Configure logging
logging.basicConfig(filename='reports/evaluation.log', level=logging.INFO, format='%(asctime)s %(message)s')

def train_model(input_filepath: str, model_filepath: str, report_filepath: str):
    df = pd.read_csv(input_filepath)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_filepath)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=fetch_20newsgroups(subset='all').target_names)
    accuracy = accuracy_score(y_test, y_pred)
    
    with open(report_filepath, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(report)
    
    # Log metrics
    logging.info(f"Model trained and saved to {model_filepath}")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(report)
    
    print(f"Model trained and saved to {model_filepath}")
    print(f"Evaluation report saved to {report_filepath}")

if __name__ == "__main__":
    train_model('data/processed/newsgroups_features.csv', 'models/text_classification_model.joblib', 'reports/evaluation.txt')

