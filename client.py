import requests
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# List of model APIs exposed via ngrok
api_urls = [
    "https://ec82-2a01-cb00-18d-a500-7050-6c92-4352-4208.ngrok-free.app/predict",
    "https://02ff-185-20-16-26.ngrok-free.app/predict"
]

def get_prediction(api_url, features):
    try:
        response = requests.get(api_url, params=features)
        data = response.json()
        return data["predicted_class"]
    except Exception as e:
        print(f"⚠️ Error with {api_url}: {e}")
        return None

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets (80% train, 20% test)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

correct = 0
total = len(X_test)

# Iterate over each test sample
for i, sample in enumerate(X_test):
    features = {
        "sepal_length": sample[0],
        "sepal_width": sample[1],
        "petal_length": sample[2],
        "petal_width": sample[3]
    }

    all_predictions = []
    # Get predictions from each model
    for url in api_urls:
        result = get_prediction(url, features)
        if result is not None:  # Check if prediction was received
            all_predictions.append(result)

    if all_predictions:
        # Majority voting on the classes (convert to string to avoid errors)
        final_prediction = max(set(all_predictions), key=all_predictions.count)

        # Map text labels to numeric indices
        label_to_index = {"setosa": 0, "versicolor": 1, "virginica": 2}
        if final_prediction in label_to_index:
            final_prediction = label_to_index[final_prediction]

        # Check if the prediction matches the true label
        if final_prediction == y_test[i]:
            correct += 1

# Calculate accuracy of the consensus model
accuracy = correct / total
print(f"✅ Consensus meta-model accuracy: {accuracy:.2f}")
