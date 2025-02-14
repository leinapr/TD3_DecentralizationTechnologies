import requests
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🛠️ Load the model weights from `database.json`
with open("database.json", "r") as f:
    database = json.load(f)

# 🏆 Build a dictionary of weights
model_weights = {model["id"]: model["weight"] for model in database["models"]}

# 🔗 URLs of participants' APIs (without parameters)
api_urls = {
    "lisa": "https://44e6-89-30-29-68.ngrok-free.app/predict",
    "leina": "https://513c-89-30-29-68.ngrok-free.app/predict"
}

# Function to query an API and get the prediction
def get_prediction(api_url, features):
    try:
        response = requests.get(api_url, params=features)
        data = response.json()
        return data["predicted_class"]
    except Exception as e:
        print(f"⚠️ Error with {api_url}: {e}")
        return None

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and test sets
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

correct = 0
total = len(X_test)
model_correct_counts = {model_id: 0 for model_id in api_urls}  # Count correct predictions per model

for i, sample in enumerate(X_test):
    features = {
        "sepal_length": sample[0],
        "sepal_width": sample[1],
        "petal_length": sample[2],
        "petal_width": sample[3]
    }

    weighted_predictions = {}  # Store weighted predictions
    sum_weights = 0

    for model_id, url in api_urls.items():
        result = get_prediction(url, features)

        # Get the model object from the database based on model_id
        model = next(model for model in database["models"] if model["id"] == model_id)

        if result is not None:
            # Convert the class name to an integer (e.g., "setosa" -> 0)
            class_index = list(iris.target_names).index(result)

            # Apply weighting
            weight = model_weights.get(model_id, 1.0)  # Default value is 1.0 if the model doesn't exist in database.json
            weighted_predictions[class_index] = weighted_predictions.get(class_index, 0) + weight
            sum_weights += weight

            # Track correct predictions locally
            if class_index == y_test[i]:
                model_correct_counts[model_id] += 1

        # After processing predictions for all models, update the correct_predictions field in the model
        model["correct_predictions"] = model_correct_counts[model_id]

    if weighted_predictions:
        # Determine the class with the highest weighted score
        final_prediction = max(weighted_predictions, key=weighted_predictions.get)
        if final_prediction == y_test[i]:
            correct += 1

# Calculate the new accuracy
accuracy = correct / total
print(f"✅ Weighted consensus meta-model accuracy: {accuracy:.2f}")

# Update the model weights based on correct predictions
for model_id in model_correct_counts:
    if total > 0:
        model_weights[model_id] = model_correct_counts[model_id] / total

# Save the new weights to `database.json`
for model in database["models"]:
    if model["id"] in model_weights:
        model["weight"] = model_weights[model["id"]]

with open("database.json", "w") as f:
    json.dump(database, f, indent=4)

print("📌 Weights update saved in database.json")
