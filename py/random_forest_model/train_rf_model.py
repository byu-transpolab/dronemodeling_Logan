import pandas as pd
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. PARAMETERS ---
input_file = '/Users/willicon/Desktop/dronemodeling_Logan/output_data/Logan_utah_seed50_2026_01_16_124241/neighbor calculations/spatial_features_k5.csv'
random_state = 42 # Used to create the random string of variables when creating the random forrest.
test_size = 0.001 # The lower the test size, the more of the buildings used in training. 
                  #Keep at 0.001 if the data used to train will not be the same data used to perdict.
name_adden = 'nn_conf_class_'

# Determine script directory to save the model
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# --- 2. MODEL NAMING --- Use the grandparent folder
grandparent_name = Path(input_file).parent.parent.name

# Create filename: "..._rf_model.joblib"
model_filename = f"{name_adden}{grandparent_name}_rf_model.joblib"
model_output_path = os.path.join(script_dir, model_filename)

# --- 3. LOAD & SPLIT ---
df = pd.read_csv(input_file)


#THIS DETERMINES WHAT VARIABLES ARE CONSIDERED WHEN RUNNING THE RANDOM FORREST MODEL. 
#feature_cols = ['Area_sqm', 'Perimeter_m', 'Confidence'], 'neighbor_mean_perimeter','neighbor_mean_area'
feature_cols = ['Area_sqm', 'Perimeter_m', 'Confidence','neighbor_mean_conf', 'neighbor_majority_class']
X = df[feature_cols]

#THIS IS WHAT IS BEING PERDICTED BY THE feature_cols VARIABLES
y = df['yolo_pred']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

print(f"Training on {len(X_train)} samples...")

# --- 4. TRAIN MODEL ---
rf_model = RandomForestClassifier(random_state=random_state)
rf_model.fit(X_train, y_train)

# --- 5. SAVE MODEL ---
joblib.dump(rf_model, model_output_path)

print(f"✅ Model trained and saved to: {model_output_path}")