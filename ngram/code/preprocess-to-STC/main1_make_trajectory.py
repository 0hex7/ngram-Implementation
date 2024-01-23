import pandas as pd

csv_file_path = "/Users/arun/Desktop/Internship/datasets/combineddataset/yellow_trip_04_2012_to_02_2013_dataset.csv"
output_file_path = "/Users/arun/Desktop/Internship/preprocess/trajectory.csv"

# Read the CSV file in chunks
chunk_size = 10**6
chunks = pd.read_csv(csv_file_path, chunksize=chunk_size)

# Initialize an empty list to store DataFrames
concatenated_trajectories = []

# Iterate through chunks
for chunk in chunks:
    # Convert datetime columns to datetime format
    chunk['tpep_pickup_datetime'] = pd.to_datetime(chunk['tpep_pickup_datetime'])
    chunk['date'] = chunk['tpep_pickup_datetime'].dt.date

    # Group by Vendor ID and Date, concatenate pick-up and drop-off locations
    grouped = chunk.groupby(['VendorID', 'date'])[['PULocationID', 'DOLocationID']].agg(lambda x: list(x))
    grouped.reset_index(inplace=True)

    # Concatenate pick-up and drop-off locations to form trajectories
    grouped['Trajectory'] = grouped.apply(lambda row: row['PULocationID'] + row['DOLocationID'], axis=1)

    # Keep only necessary columns
    grouped = grouped[['VendorID', 'date', 'Trajectory']]

    # Append the grouped DataFrame to the list
    concatenated_trajectories.append(grouped)

# Concatenate all DataFrames in the list
concatenated_trajectories = pd.concat(concatenated_trajectories, ignore_index=True)

# Save the result to a new CSV file
concatenated_trajectories.to_csv(output_file_path, index=False)

# Display the first few rows of the concatenated trajectories DataFrame
print(concatenated_trajectories.head())
