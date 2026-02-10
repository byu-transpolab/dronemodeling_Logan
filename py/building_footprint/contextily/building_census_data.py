import requests
import pandas as pd
import datetime

# 1. SETUP
# ---------------------------------------------------------
# Leave key empty "" to test without a key (works for small/infrequent requests)
API_KEY = "" 
YEAR = "2022"
DATASET = "acs/acs5"
BASE_URL = f"https://api.census.gov/data/{YEAR}/{DATASET}"

variables = {
    "NAME": "City",
    "B19013_001E": "Median Household Income",
    "B25035_001E": "Median Year Structure Built",
    "B25018_001E": "Median Rooms (House Size)",
    "B25010_001E": "Avg Household Size"
}

# FIPS code for Utah is "49"
state_fips = "49"

# 2. FETCH DATA
# ---------------------------------------------------------
print("Fetching data for Utah...")

# We fetch all places in Utah, then filter in Python
params = {
    "get": ",".join(variables.keys()),
    "for": "place:*",
    "in": f"state:{state_fips}"
}

if API_KEY:
    params["key"] = API_KEY

try:
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()
    
    # 3. PROCESS DATA
    # ---------------------------------------------------------
    df = pd.DataFrame(data[1:], columns=data[0])
    df.rename(columns=variables, inplace=True)
    
    # Filter for specific cities
    # Note: Census names often include "city" or "town"
    target_cities = [
        "Logan city, Utah", 
        "St. George city, Utah", 
        "Price city, Utah", 
        "American Fork city, Utah", 
        "Saratoga Springs city, Utah", 
        "Lehi city, Utah", 
        "Nephi city, Utah", 
        "Brigham City city, Utah",
        "Provo city, Utah",
        "Bountiful city, Utah",
        "West Valley City city, Utah",
        "Vernal city, Utah",
        "Ogden city, Utah",
        "Heber city, Utah",
        "Payson city, Utah"
        ]
    df_filtered = df[df["City"].isin(target_cities)].copy()
    
    # Convert types
    cols_to_convert = [
        "Median Household Income", 
        "Median Year Structure Built", 
        "Median Rooms (House Size)", 
        "Avg Household Size"
    ]
    for col in cols_to_convert:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
    # Calculate Age
    current_year = datetime.datetime.now().year
    df_filtered['Median House Age'] = current_year - df_filtered['Median Year Structure Built']
    
    # Select Final Columns
    final_cols = [
        "City", 
        "Median Household Income", 
        "Median House Age", 
        "Median Rooms (House Size)", 
        "Avg Household Size"
    ]
    
    df_final = df_filtered[final_cols]
    
    # Save
    df_final.to_csv("utah_comparison.csv", index=False)
    print("Success! Saved to utah_comparison.csv")
    print(df_final)

except Exception as e:
    print(f"Error: {e}")