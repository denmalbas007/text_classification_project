from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def load_data(output_filepath: str):
    newsgroups = fetch_20newsgroups(subset='all')
    df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})
    df.to_csv(output_filepath, index=False)
    print(f"Data loaded and saved to {output_filepath}")

if __name__ == "__main__":
    load_data('data/raw/newsgroups.csv')
