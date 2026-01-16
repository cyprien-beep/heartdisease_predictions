import os
import joblib
import pandas as pd
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_heart_disease_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)


CLASS_NAMES = {
    0: "No Disease",
    1: "Immediate Danger",
    2: "Severe",
    3: "Mild",
    4: "Very Mild"
}

test_data = [
    {
        "age": 55, "sex": "Male", "cp": "Typical Angina", "trestbps": 140, "chol": 240,
        "fbs": "False", "restecg": "Normal", "thalach": 150, "exang": "No",
        "oldpeak": 1.2, "slope": "Upsloping", "ca": 0, "thal": "Normal"
    },
    {
        "age": 60, "sex": "Female", "cp": "Asymptomatic", "trestbps": 130, "chol": 200,
        "fbs": "True", "restecg": "ST-T Wave Abnormality", "thalach": 140,
        "exang": "Yes", "oldpeak": 2.5, "slope": "Flat", "ca": 2,
        "thal": "Reversible defect"
    },
    {
        "age": 45, "sex": "Male", "cp": "Non-Anginal Pain", "trestbps": 120, "chol": 210,
        "fbs": "False", "restecg": "Left Ventricular Hypertrophy", "thalach": 170,
        "exang": "No", "oldpeak": 0.0, "slope": "Downsloping", "ca": 1,
        "thal": "Fixed defect"
    }
]

df_test = pd.DataFrame(test_data)

df_test["sex"] = df_test["sex"].map({"Male": 1, "Female": 0})
df_test["fbs"] = df_test["fbs"].map({"True": 1, "False": 0})
df_test["exang"] = df_test["exang"].map({"Yes": 1, "No": 0})

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}
df_test["cp"] = df_test["cp"].map(cp_map)

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
df_test["restecg"] = df_test["restecg"].map(restecg_map)

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
df_test["slope"] = df_test["slope"].map(slope_map)

thal_map = {
    "Normal": 1,
    "Fixed defect": 2,
    "Reversible defect": 3
}
df_test["thal"] = df_test["thal"].map(thal_map)

predictions = model.predict(df_test)

probs_all = None
if hasattr(model, "predict_proba"):
    probs_all = model.predict_proba(df_test)

results = []

for i, pred in enumerate(predictions):
    row = {
        "Patient": f"Patient {i+1}",
        "Predicted Class": CLASS_NAMES.get(pred, "Unknown")
    }

    if probs_all is not None:
        for cls_idx, prob in enumerate(probs_all[i]):
            cls_name = CLASS_NAMES.get(cls_idx, "Unknown")
            row[f"{cls_name} (%)"] = round(prob * 100, 2)

    results.append(row)

results_df = pd.DataFrame(results)


print("\n================ PREDICTION RESULTS ================\n")
print(results_df.to_string(index=False))


results_df.to_csv("test_predictions_results.csv", index=False)
