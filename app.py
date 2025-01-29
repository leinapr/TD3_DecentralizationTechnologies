from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train different models
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Define the /predict route for API access
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extract feature inputs from query parameters
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input. Please provide all four features as valid numbers."}), 400

    rf_prediction = rf_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    rf_accuracy = rf_model.score(X_test, y_test)
    rf_species = iris.target_names[rf_prediction[0]]

    # Return predictions from all models in JSON format
    return jsonify({
        "Random Forest Prediction": {
            "rf_accuracy": rf_accuracy,
            "rf_prediction": int(rf_prediction[0]),
            "rf_species": rf_species
        }
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
