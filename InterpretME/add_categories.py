import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('/Users/a.arnous/workspace/InterpretME/example/output/survshap/combined_survshaps/combined_survhshap.csv')

# Function to categorize age
def categorize_age(row):
    if row['Features'] == 'age':
        if row['Values'] >= 65:
            return 'old'
        elif row['Values'] < 18:
            return 'child'
        else:
            return 'adult'
    return row['Values']

# Apply the categorization function to each row
df['Values'] = df.apply(categorize_age, axis=1)

# Save the modified DataFrame back to a CSV file
df.to_csv('/Users/a.arnous/workspace/InterpretME/example/output/survshap/combined_survshaps/categorized_combined_survhshap.csv', index=False)
