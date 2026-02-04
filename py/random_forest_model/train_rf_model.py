import pandas as pd
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. PARAMETERS ---
input_file = 'output_data/orem_caschade_orchard_seed15_2026_02_04_083646/neighbor calculations/spatial_features_k5.csv'
random_state = 42 # Used to create the random string of variables when creating the random forrest.
test_size = 0.4 # The lower the test size, the more of the buildings used in training. 
                  #Keep at 0.001 if the data used to train will not be the same data used to perdict.

name_adden = 'anno_type_' #Addend to beganning of name. Use '' if none is wanted

# Variables are what the model will train itself on. Give as a list of columns.
# Product is what column is being solved for.
variables = ['Area_sqm', 'Perimeter_m', 'Confidence','neighbor_mean_conf', 'neighbor_majority_class','yolo_pred']
product = 'annotation_type'

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
# 1. Read file
df = pd.read_csv(input_file)

# 2. Remove rows where the target is NaN (Empty/Null)
initial_count = len(df)
df = df.dropna(subset=[product])
print(f"Removed {initial_count - len(df)} rows with missing '{product}'. Remaining: {len(df)}")

# 3. Define Features
# feature_cols = ['Area_sqm', 'Perimeter_m', 'Confidence'], 'neighbor_mean_perimeter','neighbor_mean_area'
feature_cols = variables
X = df[feature_cols]

# 4. Define Target (Now guaranteed to have no NaNs)
y = df[product]

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