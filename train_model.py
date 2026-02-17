import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("creditcard.csv")  # make sure dataset is in folder

X = data.drop("Class", axis=1)
y = data["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model saved as model.pkl")
