🏥 Hospital Readmission Risk Predictor (Diabetes Care)
🚀 http://localhost:8501/#patient-readmission-risk-predictor
📌 Project Overview
This project addresses a critical challenge in Healthcare Administration: Unplanned Hospital Readmissions. Using the UCI Diabetes dataset (100k+ clinical records), I developed an end-to-end Machine Learning pipeline that predicts whether a diabetic patient will be readmitted within 30 days.

This tool serves as a Clinical Decision Support System, helping hospitals prioritize high-risk patients for intensive transitional care, ultimately reducing costs and improving patient outcomes.

🛠️ Tech Stack
Language: Python 3.14

ML Framework: XGBoost

Data Science: Pandas, NumPy, Scikit-learn

Explainability: SHAP (SHapley Additive exPlanations)

Deployment: Streamlit Cloud

🧪 Key Features & Engineering
Clinical ICD-9 Mapping: Transformed raw diagnostic codes into clinically meaningful categories (Circulatory, Respiratory, Neoplasms, etc.) to improve model interpretability.

Class Imbalance Handling: Utilized scale_pos_weight to address the minority class (readmitted patients), prioritizing Recall to ensure high-risk patients are not missed.

Feature Importance: Integrated SHAP to provide "Black Box" transparency, revealing that prior inpatient visits and number of medications are the strongest predictors of risk.

📊 Model Performance
Recall (High Risk): 68% (Prioritizing patient safety)

AUC-ROC: 0.68

Precision-Recall Trade-off: Tuned for clinical intervention rather than simple accuracy.

🚀 Installation & Usage
Clone the repo: git clone https://github.com/classicwebdev/HealthTech-Readmission-AI.git

Install dependencies: pip install -r requirements.txt

Run the trainer: python main.py

Launch the UI: streamlit run app.py