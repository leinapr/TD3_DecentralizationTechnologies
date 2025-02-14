import requests
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data from `database.json`
with open("database.json", "r") as f:
    database = json.load(f)

# Build dictionaries for weights and balances
model_weights = {model["id"]: model["weight"] for model in database["models"]}
model_balances = {model["id"]: model["balance"] for model in database["models"]}

# URLs of participant APIs
api_urls = {
    "lisa": "https://44e6-89-30-29-68.ngrok-free.app/predict",
    "leina": "https://513c-89-30-29-68.ngrok-free.app/predict"
}

# Function to query an API and retrieve the prediction
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
model_correct_counts = {model_id: 0 for model_id in api_urls}  # Track correct predictions per model
model_error_counts = {model_id: 0 for model_id in api_urls}  # Track incorrect predictions

for i, sample in enumerate(X_test):
    features = {
        "sepal_length": sample[0],
        "sepal_width": sample[1],
        "petal_length": sample[2],
        "petal_width": sample[3]
    }

    weighted_predictions = {}
    sum_weights = 0
    all_predictions = []

    for model_id, url in api_urls.items():
        result = get_prediction(url, features)
        if result is not None:
            class_index = list(iris.target_names).index(result)
            all_predictions.append(class_index)

            # 🔢 Apply weighting
            weight = model_weights.get(model_id, 1.0)
            weighted_predictions[class_index] = weighted_predictions.get(class_index, 0) + weight
            sum_weights += weight

            # 📊 Update score tracking
            if class_index == y_test[i]:
                model_correct_counts[model_id] += 1
            else:
                model_error_counts[model_id] += 1  # Track incorrect predictions

    if weighted_predictions:
        final_prediction = max(weighted_predictions, key=weighted_predictions.get)
        if final_prediction == y_test[i]:
            correct += 1

# Calculate model accuracy
accuracy = correct / total
print(f"✅ Weighted consensus meta-model accuracy: {accuracy:.2f}")

# Update model weights based on correct predictions
for model_id in model_correct_counts:
    if total > 0:
        model_weights[model_id] = model_correct_counts[model_id] / total

# 🏆 Apply Slashing (Penalty for errors)
PENALTY = 10  # Fine per incorrect prediction
REWARD = 5  # Reward per correct prediction

for model_id in model_error_counts:
    errors = model_error_counts[model_id]

    # ⚠️ If a model makes too many errors, it gets penalized financially
    if errors > 0:
        model_balances[model_id] -= errors * PENALTY
        print(f"❌ {model_id} lost {errors * PENALTY}€ (Slashing)")

    # 🎉 Reward for correct predictions
    if model_correct_counts[model_id] > 0:
        model_balances[model_id] += model_correct_counts[model_id] * REWARD
        print(f"💰 {model_id} earned {model_correct_counts[model_id] * REWARD}€")

# Save updated weights and balances to `database.json`
for model in database["models"]:
    model["weight"] = model_weights[model["id"]]
    model["balance"] = model_balances[model["id"]]

with open("database.json", "w") as f:
    json.dump(database, f, indent=4)

print("📌 Weights and balances update saved in `database.json` ✅")
print(f"📊 Updated balances: {model_balances}")