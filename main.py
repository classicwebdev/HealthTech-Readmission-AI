import pandas as pd
import numpy as np
import requests
import zipfile
import io


def load_hospital_data():
    # URL to the official UCI dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"

    print("--- Step 1: Ingesting Global Health Data ---")
    response = requests.get(url)
    # We use BytesIO to treat the downloaded content as a file in memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # CRITICAL FIX: The file is inside a subfolder in the ZIP
        target_file = 'dataset_diabetes/diabetic_data.csv'

        if target_file in z.namelist():
            with z.open(target_file) as f:
                df = pd.read_csv(f)
            print(f"Success! Dataset Loaded.")
            print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        else:
            # This helps us debug if the ZIP content changes
            print("Files found in ZIP:", z.namelist())
            raise KeyError(f"Could not find {target_file} in the archive.")


# Run it again
df = load_hospital_data()
# --- RE-FIX Step 2: Ensure target is integer ---
# (Make sure this happens before Step 5)
df['target'] = (df['readmitted'] == '<30').astype(int)
# Drop the original 'readmitted' column so it doesn't get encoded
df.drop('readmitted', axis=1, inplace=True)
print("\n--- Step 3: Handling Missing Clinical Data ---")

# 1. Weight: Indicator variable
df['has_weight'] = df['weight'].notnull().astype(int)

# 2. Medical Specialty: Fill missing with 'Unknown'
# and keep only the top 10 specialties (to reduce noise)
top_specialties = df['medical_specialty'].value_counts().nlargest(10).index
df['medical_specialty'] = df['medical_specialty'].apply(lambda x: x if x in top_specialties else 'Other/Unknown')

# 3. Payer Code: Standardize missing values
df['payer_code'] = df['payer_code'].fillna('Not_Available')

# 4. Drop columns that are completely useless or have too much leakage
# 'encounter_id' and 'patient_nbr' are just IDs, not clinical features.
df.drop(['weight', 'encounter_id', 'patient_nbr', 'payer_code'], axis=1, inplace=True)

print(f"Missing data handled. New shape: {df.shape}")


def map_diagnosis(code):
    if pd.isnull(code):
        return 'Other'

    # Some codes start with 'V' or 'E' (Supplementary classifications)
    if str(code).startswith(('V', 'E')):
        return 'Other'

    try:
        val = float(code)
        if 390 <= val <= 459 or val == 785:
            return 'Circulatory'
        elif 460 <= val <= 519 or val == 786:
            return 'Respiratory'
        elif 520 <= val <= 579 or val == 787:
            return 'Digestive'
        elif val == 250:
            return 'Diabetes'
        elif 800 <= val <= 999:
            return 'Injury'
        elif 710 <= val <= 739:
            return 'Musculoskeletal'
        elif 580 <= val <= 629 or val == 788:
            return 'Genitourinary'
        elif 140 <= val <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'


print("\n--- Step 4: Mapping Clinical ICD-9 Codes ---")
df['diag_1_group'] = df['diag_1'].apply(map_diagnosis)

# Now we can drop the original messy diagnosis columns
df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

print(f"ICD-9 codes grouped. New columns created: {df['diag_1_group'].unique()}")

print("\n--- Step 5: Encoding Categorical Data ---")
# --- RE-FIX Step 5: Encoding ---
# We want to encode everything EXCEPT our target
cols_to_encode = df.select_dtypes(include=['object', 'string']).columns
df_final = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

# --- Checkpoint ---
print(f"Is 'target' in df_final? {'target' in df_final.columns}")

print(f"Encoding complete! Final feature count: {df_final.shape[1]}")

print("\n--- Step 5.5: Cleaning Feature Names for XGBoost ---")

# This one-liner replaces [, ], and < with underscores
df_final.columns = df_final.columns.str.replace('[', '(', regex=False)\
                                   .str.replace(']', ')', regex=False)\
                                   .str.replace('<', 'less_than', regex=False)

# Let's verify the 'Age' columns are now safe
safe_age_cols = [col for col in df_final.columns if 'age' in col]
print(f"Cleaned Age columns sample: {safe_age_cols[:3]}")

from sklearn.model_selection import train_test_split

print("\n--- Step 6: Stratified Data Splitting ---")

# Define X (features) and y (target)
print("Columns in df_final:", df_final.columns.tolist())
X = df_final.drop('target', axis=1)
y = df_final['target']

# We use 'stratify=y' to ensure both training and testing sets
# have the same percentage of high-risk patients.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

print("\n--- Step 7: Training XGBoost Model ---")

# Initialize the model
# scale_pos_weight helps the model focus more on the minority 'High Risk' class
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=9, # Since roughly 10-11% are positive, we weight them higher
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Model Training Complete!")

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("\n--- Step 8: Global Industry Evaluation Metrics ---")

# 1. Classification Report
print("Clinical Performance Report:")
print(classification_report(y_test, y_pred))

# 2. AUC-ROC (Discriminative Power)
auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC Score: {auc:.4f}")

# 3. Visualizing the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stable', 'High Risk'],
            yticklabels=['Stable', 'High Risk'])
plt.xlabel('Predicted by AI')
plt.ylabel('Actual Patient Outcome')
plt.title('Patient Readmission Confusion Matrix')
plt.show()

import shap

print("\n--- Step 9: Explaining the Model (SHAP) ---")
# Initialize the explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary Plot - This shows which features are most important globally
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Save the model to a file
model.save_model('hospital_readmission_model.json')
print("Model saved as hospital_readmission_model.json")

import json
# Save the feature names so app.py can build the correct input matrix
feature_names = X_train.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("Clinical feature names saved to feature_names.json")