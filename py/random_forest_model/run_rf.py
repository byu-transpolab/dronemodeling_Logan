import pandas as pd
import os
import matplotlib
from pathlib import Path # Added for easier parent directory navigation

# This prevents plots from displaying/rendering in notebooks or pop-up windows
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 1. SETUP PATHS ---
input_file = '/Users/willicon/Desktop/dronemodeling_Logan/output_data/Logan_utah_seed50_2026_01_16_124241/yolo_pred/yolo_output_data.csv'
random_state = 42
test_size = 0.3 # 0.3 is the standard

# Determine script directory (for fallback)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# --- NEW DIRECTORY LOGIC ---
target_folder_name = "random forest prediction"
input_path_obj = Path(input_file)

# Logic: Try to find the grandparent folder of the input file
# input: .../Experiment_Folder/yolo_pred/data.csv
# parent: .../Experiment_Folder/yolo_pred
# grandparent: .../Experiment_Folder
if input_path_obj.exists():
    grandparent_dir = input_path_obj.parent.parent
    if grandparent_dir.exists():
        # Set output to: .../Experiment_Folder/random forest prediction
        output_dir = os.path.join(grandparent_dir, target_folder_name)
    else:
        # Fallback: Script Directory/random forest prediction
        output_dir = os.path.join(script_dir, target_folder_name)
else:
    # Fallback if input file not found yet
    output_dir = os.path.join(script_dir, target_folder_name)

# Create the directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Saving to existing directory: {output_dir}")

# Generate filename (Keep original filename logic for the CSV itself)
filename = os.path.basename(input_file)
raw_name = os.path.splitext(filename)[0]
short_name = raw_name.replace('_final', '') 

output_csv = os.path.join(output_dir, f'{short_name}_predictions.csv')

# --- 2. LOAD & SPLIT ---
df = pd.read_csv(input_file)

X = df[['Area_sqm', 'Perimeter_m', 'Confidence']]
y = df['yolo_pred']

# Splits into testing size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)
print(f"Training on {(test_size)*100:.0f}% of the data")

# --- 3. TRAIN ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# --- 4. PREDICT & SAVE ---
df['RF_Prediction'] = rf_model.predict(X)
df.to_csv(output_csv, index=False)
print(f"Predictions saved to: {output_csv}")

# --- 5. VALIDATION ---
y_pred_test = rf_model.predict(X_test)
print(f"Model Validation Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(classification_report(y_test, y_pred_test, zero_division=0))

# --- 6. PLOTS (SAVE ONLY) ---

# A. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close() 

# B. Feature Importance
importances = rf_model.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=X.columns)
plt.title('Feature Importances')
plt.ylabel('Importance Score')
plt.xlabel('Feature Name')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# C. Correlation Matrix
plt.figure(figsize=(8, 6))
# numeric_only=True ensures we don't crash on non-numeric columns
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.xlabel('Features')     
plt.ylabel('Features')     
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

print("All plots saved successfully.")