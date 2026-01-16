from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("deployment/best_heart_disease_model.pkl")

CLASS_NAMES = ["No Disease", "Immediate Danger", "Severe", "Mild", "Very Mild"]

CSS_CLASS_MAP = {
    "No Disease": "no-disease",
    "Very Mild": "very-mild",
    "Mild": "mild",
    "Severe": "severe",
    "Immediate Danger": "immediate-danger"
}

@app.route("/", methods=["GET"])
def home():
    return render_template("index25rp20156.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    entered_data = data.copy()

    numeric_fields = ["age","trestbps","chol","thalach","oldpeak","ca"]
    for field in numeric_fields:
        data[field] = float(data[field])

    data['sex'] = 1 if data['sex'] == "Male" else 0
    data['fbs'] = 1 if data['fbs'] == "True" else 0
    data['exang'] = 1 if data['exang'] == "Yes" else 0

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3
    }
    data['cp'] = cp_map[data['cp']]

    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    data['restecg'] = restecg_map[data['restecg']]

    slope_map = {
        "Upsloping": 0, 
        "Flat": 1,
        "Downsloping": 2
    }
    data['slope'] = slope_map[data['slope']]

    thal_map = {
        "Normal": 1,
        "Fixed defect": 2,
        "Reversible defect": 3
    }
    data['thal'] = thal_map[data['thal']]

    input_df = pd.DataFrame([data])

    pred_index = model.predict(input_df)[0]
  
    prediction = CLASS_NAMES[pred_index]
    prediction_class = CSS_CLASS_MAP.get(prediction, "no-disease")

 
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        for idx, prob in enumerate(probs):
            probabilities[CLASS_NAMES[idx]] = round(prob * 100, 2)

    return render_template(
        "index25rp20156.html",
        prediction=prediction,
        prediction_class=prediction_class,
        probabilities=probabilities,
        entered_data=entered_data
    )

if __name__ == "__main__":
    app.run(debug=True)
