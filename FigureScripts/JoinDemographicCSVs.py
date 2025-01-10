import os
import pandas as pd
import sys
def combine_csvs(directory, key_columns, keep_columns):
    # List to store dataframes
    dfs = []

    # Loop through all CSV files in the directory
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Select only the required columns
            df_selected = df[key_columns + keep_columns]

            # Add the dataframe to the list
            dfs.append(df_selected)

    # Merge all dataframes on the key columns (age, sex, diagnosis)
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=key_columns, suffixes=('_file1', '_file2'), how='outer')

    return merged_df

# Directory containing the CSV files
directory = sys.argv[1]  # Replace with the actual directory path

# Columns to merge on (age, sex, diagnosis)
key_columns = ['age', 'sex', 'diagnosis']

# Columns to keep from each CSV (Subject, Session, Run, Acq)
keep_columns = ['Subject', 'Session', 'Run', 'acq']

# Combine CSV files
combined_df = combine_csvs(directory, key_columns, keep_columns)

# Save the result to a new CSV
combined_df.to_csv('combined_output.csv', index=False)

print("CSV files successfully combined and saved to 'combined_output.csv'")
