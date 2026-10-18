
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data_path = Path("data/house_price.csv")
df = pd.read_csv(data_path)

# Separate input features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Save the trained model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "model.pkl")

print("Model trained successfully!")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")
print("Model saved to models/model.pkl")
