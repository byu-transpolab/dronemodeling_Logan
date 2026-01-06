import os

"""
This code will modify the class of a annotation label. Currently it changes class 1 (Church) to class 3 (residential)
"""

# This is the path to the folder containing the label files
folder_path = r'/Users/willicon/Desktop/Annotated Photos/Ogden_Utah_Validation/labels' 

# Counter to track progress
files_processed = 0

# check if folder exists
if os.path.exists(folder_path):
    # Loop through every file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Step 1: Read the existing data
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Step 2: Process the lines in memory
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                
                # Check if line is valid and starts with "1"
                if parts and parts[0] == '1':
                    parts[0] = '3'
                
                # Reassemble the line
                new_lines.append(" ".join(parts) + "\n")
            
            # Step 3: Overwrite the file with the new data
            with open(file_path, 'w') as file:
                file.writelines(new_lines)
                
            files_processed += 1

    print(f"Done! Processed and overwrote {files_processed} files in '{folder_path}'.")
else:
    print(f"Error: The folder '{folder_path}' does not exist.")