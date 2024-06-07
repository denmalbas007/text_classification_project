from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('models/text_classification_model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    X = vectorizer.transform(df['text']).toarray()
    predictions = model.predict(X)
    return jsonify(predictions.tolist())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
