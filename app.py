from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import requests

app = Flask(__name__)
iris = load_iris()

MODELS = {
    "random_forest": "https://fae0-185-20-16-26.ngrok-free.app/predict",
    "svm": "https://6387-2a01-cb00-18d-a500-7050-6c92-4352-4208.ngrok-free.app/predict",
    "logistic_regression": "https://0bc7-2001-861-5865-6b10-b8b0-7cd0-d0e0-ed26.ngrok-free.app/predict"
}

@app.route('/predict', methods=['GET'])
def predict():
    try:
        sepal_length = float(request.args.get("sepal_length"))
        sepal_width = float(request.args.get("sepal_width"))
        petal_length = float(request.args.get("petal_length"))
        petal_width = float(request.args.get("petal_width"))
    except (TypeError, ValueError) as e:
        print(f"Invalid input error: {e}")
        return jsonify({"error": "Invalid input. Please provide valid numerical values."}), 400

    features = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    predictions = []
    responses = {}

    for model_name, url in MODELS.items():
        try:
            print(f"Requesting prediction from {model_name} at {url}")  # Debugging
            response = requests.get(url, params=features)
            response_json = response.json()
            print(f"Response from {model_name}: {response_json}")  # Debugging

            pred = response_json.get("prediction")
            if pred is not None:
                predictions.append(pred)
                responses[model_name] = pred
        except Exception as e:
            print(f"Error contacting {model_name}: {e}")  # Debugging
            responses[model_name] = f"Error: {str(e)}"

    print(f"Final Predictions: {responses}")  # Debugging

    if not predictions:
        return jsonify({"error": "No valid responses from models.", "debug": responses}), 500

    # Consensus Prediction using Majority Voting
    consensus_prediction = max(set(predictions), key=predictions.count)
    consensus_accuracy = accuracy_score([0]*len(predictions), [consensus_prediction]*len(predictions))

    return jsonify({
        "Model Predictions": responses,
        "Consensus Prediction": {
            "species": iris.target_names[consensus_prediction],
            "prediction": consensus_prediction,
            "accuracy": consensus_accuracy
        }
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005, debug=True)
