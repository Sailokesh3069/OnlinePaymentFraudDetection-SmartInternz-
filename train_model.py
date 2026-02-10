import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Function to generate dynamic synthetic dataset
def generate_dataset(n_samples=10000):
    transaction_types = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "OTHER"]

    data = {
        "type": np.random.choice(transaction_types, n_samples),
        "amount": np.round(np.random.uniform(10, 5000, n_samples), 2),
        "oldbalanceOrg": np.round(np.random.uniform(0, 10000, n_samples), 2),
        "newbalanceOrig": np.round(np.random.uniform(0, 10000, n_samples), 2),
        # Fraud label: 0 = legitimate, 1 = fraudulent (10% fraud)
        "target": np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    return pd.DataFrame(data)

# Generate dataset
df = generate_dataset()
print("Dynamic dataset created!")

# Map transaction types to numeric
type_mapping = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "OTHER": 5}
df["type"] = df["type"].map(type_mapping)

# Features and target
X = df[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
y = df["target"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
with open("static/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Dynamic model trained and saved successfully!")
