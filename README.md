# Heart Disease Risk Prediction System


This project is an intelligent **Heart Disease Risk Prediction System** that leverages machine learning to predict a patient's risk level based on clinical, demographic, and diagnostic data. The system includes a **trained ML model**, a **Flask web interface**, and a **responsive front-end** to display predictions along with probability distributions.

---

## Features

- Predicts heart disease risk levels for patients using structured clinical data.
- Supports multi-class risk levels: `No Disease`, `Very Mild`, `Mild`, `Severe`, `Immediate Danger`.
- Displays **probability distribution** for all risk classes.
- Responsive web interface compatible with **mobile, tablet, and desktop** devices.
- Includes a **preprocessing pipeline** to handle numeric and categorical data automatically.
- Backend and frontend integration with Flask.

---

## Technologies Used

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn, pandas, joblib
- **Frontend:** HTML, CSS, JavaScript (responsive design)
- **Data Serialization:** `joblib` for saving/loading model pipeline
- **Testing:** CSV-based verification for multiple patient cases

---

## Project Structure


ITLML_801_S_A_25RP20156/
│
├─ app_25rp20156.py 
├─ deployment/
│ ├─ best_heart_disease_model.pkl 
│ └─ test.py 
├─ templates/
│ └─ index25rp20156.html 
├─ static/
│ └─ style.css 
├─ README.md 

## Installation

1. **Clone the repository:**
```bash
git clone <repository_url>
cd ITLML_801_S_A_25RP20156
## setup python environment
python -m venv ILTML_801_S_A_25RP20156
## Author

Cyprien NZAYISENGA
Machine learning summative assessment

RP-HUYE College, Rwanda

