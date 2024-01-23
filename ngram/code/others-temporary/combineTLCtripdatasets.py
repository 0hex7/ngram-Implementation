import os
import pandas as pd

def combine_parquet_files(input_folder, output_file):
    parquet_files = [f for f in os.listdir(input_folder) if f.endswith('.parquet')]

    # Ensure the output file does not exist
    if os.path.exists(output_file):
        os.remove(output_file)

    # Initialize an empty list to store DataFrames
    data_frames = []

    for parquet_file in parquet_files:
        file_path = os.path.join(input_folder, parquet_file)

        # Read the Parquet file into a DataFrame
        data = pd.read_parquet(file_path)

        # Append the data to the list
        data_frames.append(data)

    # Concatenate the list of DataFrames
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Write the combined DataFrame to a new Parquet file
    combined_data.to_parquet(output_file)

    # Display the contents of the combined DataFrame
    print("Combined DataFrame:")
    print(combined_data)

if __name__ == "__main__":
    # Replace 'input_folder' with the path to the folder containing your monthly Parquet files
    input_folder = '/Users/arun/Desktop/Internship/datasets/'

    # Replace 'output_file' with the desired path for the combined Parquet file
    output_file = '/Users/arun/Desktop/Internship/datasets/combineddataset/yellow_trip_04_2012_to_02_2013_dataset.parquet'

    combine_parquet_files(input_folder, output_file)
    print("The operation is completed successfully!")
