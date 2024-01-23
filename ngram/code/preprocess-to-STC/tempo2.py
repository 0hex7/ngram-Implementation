import pandas as pd

# Load the Parquet file into a DataFrame
df = pd.read_parquet('/Users/arun/Desktop/ngram-Implementation/ngram/code/preprocess-to-STC/trajectories_as_geocodes_output/batch_4_trajectory_with_geocodes.parquet')

# Extract the first five rows
first_five_rows = df.head(5)

# Save the extracted rows into another Parquet file
first_five_rows.to_parquet('/Users/arun/Desktop/ngram-Implementation/ngram/code/preprocess-to-STC/trajectories_as_geocodes_output/sample_data/sampledatawith5rows.parquet', index=False)
