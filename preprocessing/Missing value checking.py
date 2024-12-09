import pandas as pd

# Read the CSV file
file_path = 'data\\yiqingqian.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Check for missing values
missing_values = df.isnull().sum()

# Check for NaN values
nan_values = df.isna().sum()

# Output results
print("Check for missing values:")
print(missing_values)
print("\nCheck for NaN values:")
print(nan_values)

# Locate rows with missing or NaN values
missing_values_location = df[df.isnull().any(axis=1)]

# Output results
print("Locations of missing and NaN values:")
print(missing_values_location)

