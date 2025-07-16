import pandas as pd
import os
import glob

# Directory containing the CSV files
data_dir = r"C:\Users\admin\Desktop\training\data"

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

for file_path in csv_files:
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Filter rows where elapsed_time_perc is less than or equal to 50
        df_filtered = df[df['elapsed_time_perc'] <= 50]
        
        # Save the filtered data back to the same file
        df_filtered.to_csv(file_path, index=False)
        
        print(f"Processed {os.path.basename(file_path)}")
        print(f"Removed {len(df) - len(df_filtered)} rows")
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")

print("\nProcessing completed!")
