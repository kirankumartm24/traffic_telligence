from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)
model = joblib.load("traffic_model_rf.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required = ['hour', 'weather', 'temp', 'day', 'month']
        if not all(k in data for k in required):
            return jsonify({"error": "Missing required fields"}), 400
        df = pd.DataFrame([{
            "hour": int(data["hour"]),
            "weather": encoders['weather'].transform([data['weather']])[0],
            "temp": float(data["temp"]),
            "day": int(data["day"]),
            "month": int(data["month"])
        }])
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]
        return jsonify({
            "predicted_volume": round(prediction),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)