
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Load the trained machine learning model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

model = joblib.load(MODEL_PATH)

# Input features required by the model
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Example exchange rate: 1 USD = 85 INR
# Accurate results kosam current exchange rate tho update cheyyandi.
USD_TO_INR = 85.0


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/prediction", methods=["POST"])
def prediction():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please provide input as a JSON object"
            }), 400

        # Check whether all required features are present
        missing = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing,
            }), 400

        # Convert inputs to numeric values
        input_values = []

        for feature in FEATURES:
            value = float(data[feature])

            if not pd.notna(value):
                return jsonify({
                    "error": f"{feature} must be a valid number"
                }), 400

            input_values.append(value)

        # Prepare input for prediction
        input_data = pd.DataFrame(
            [input_values],
            columns=FEATURES,
        )

        # Predict house value
        predicted_value = float(model.predict(input_data)[0])

        # California Housing target is measured in units of $100,000
        predicted_value_usd = predicted_value * 100000

        # Convert USD to Indian Rupees
        predicted_value_inr = predicted_value_usd * USD_TO_INR

        return jsonify({
            "predicted_house_value": round(predicted_value_inr, 2),
            "currency": "INR",
        }), 200

    except (TypeError, ValueError, OverflowError):
        return jsonify({
            "error": "All input features must be valid numbers"
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Unable to generate prediction"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)