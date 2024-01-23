'''import pandas as pd
from shapely.geometry import Point
import os


def process_trajectory(input_csv, coordinates_df, output_dir):
    print(f"\nProcessing file: {input_csv}")

    # Read the input CSV file with trajectories
    print("\nReading the input CSV file...")
    df = pd.read_csv(input_csv)
    print("Input CSV file has been read.")

    # Display the number of trajectories
    print(f"Number of trajectories in {input_csv}: {len(df)}")

    # Check column names
    print("Column names in the dataframe:", df.columns)

    # Create an empty list to store modified trajectories
    modified_trajectories = []

    # Iterate through each row and replace zone names with geocodes
    for index, row in df.iterrows():
        updated_trajectory = eval(row['updated_trajectory'])
        modified_trajectory = []

        for zone in updated_trajectory:
            # Fetch coordinates for the zone
            latitude, longitude = fetch_coordinates_from_csv(zone, coordinates_df)

            # If coordinates are found, replace zone name with geocode
            if latitude is not None and longitude is not None:
                modified_trajectory.append(f"({latitude}, {longitude})")
            else:
                modified_trajectory.append(zone)

        modified_trajectories.append(modified_trajectory)

    # Create a new DataFrame with modified trajectories
    df_modified = df.copy()
    df_modified['updated_trajectory'] = modified_trajectories

    # Generate the output Parquet file path
    output_parquet_geocodes = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_csv))[0]}_trajectory_with_geocodes.parquet")

    # Save the results to the output Parquet file
    df_modified.to_parquet(output_parquet_geocodes, index=False)

    print(f"\nResults saved to {output_parquet_geocodes}")

# Fetch coordinates function remains the same
def fetch_coordinates_from_csv(zone, coordinates_df):
    try:
        row = coordinates_df[coordinates_df['zone'] == zone].iloc[0]
        return row['latitude'], row['longitude']
    except IndexError:
        print(f"No coordinates found for {zone}.")
        return None, None
    except Exception as e:
        print(f"Error fetching coordinates for {zone} from CSV: {e}")
        return None, None

if __name__ == '__main__':
    # Example usage
    input_directory = "/Users/arun/Desktop/Internship/preprocess/trajectories_as_zones_output_batches/xox/"
    output_directory = "/Users/arun/Desktop/Internship/preprocess/trajectories_as_geocodes_output/"
    coordinates_csv = "/Users/arun/Desktop/Internship/preprocess/zones_and_geocodes.csv"  # Replace with the actual path to your coordinates CSV file

    print("\nThe main function is starting now..")
    geocode_zones(input_directory, output_directory, coordinates_csv)
    '''
    
    
    
    
    
import os
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

def fetch_coordinates_from_csv(zone, coordinates_df):
    try:
        row = coordinates_df[coordinates_df['zone'] == zone].iloc[0]
        return row['latitude'], row['longitude']
    except IndexError:
        #print(f"No coordinates found for {zone}.")
        return None, None
    except Exception as e:
        print(f"Error fetching coordinates for {zone} from CSV: {e}")
        return None, None

def process_trajectory(input_csv, coordinates_df, output_dir):
    print(f"\nProcessing file: {input_csv}")

    # Read the input CSV file with trajectories
    print("\nReading the input CSV file...")
    df = pd.read_csv(input_csv)
    print("Input CSV file has been read.")

    # Display the number of trajectories
    print(f"Number of trajectories in {input_csv}: {len(df)}")

    # Check column names
    print("Column names in the dataframe:", df.columns)

    # Create an empty list to store modified trajectories
    modified_trajectories = []

    # Iterate through each row and replace zone names with geocodes
    for index, row in tqdm(df.iterrows(), desc="Processing Rows", total=len(df)):
        updated_trajectory = eval(row['updated_trajectory'])
        modified_trajectory = []

        for zone in updated_trajectory:
            # Fetch coordinates for the zone
            latitude, longitude = fetch_coordinates_from_csv(zone, coordinates_df)

            # If coordinates are found, replace zone name with geocode
            if latitude is not None and longitude is not None:
                modified_trajectory.append(f"({latitude}, {longitude})")
            else:
                modified_trajectory.append(zone)

        modified_trajectories.append(modified_trajectory)

    # Create a new DataFrame with modified trajectories
    df_modified = df.copy()
    df_modified['updated_trajectory'] = modified_trajectories

    # Remove rows where 'updated_trajectory' is None
    original_rows = len(df_modified)
    df_modified = df_modified.dropna(subset=['updated_trajectory'])

    # Calculate the number of removed zones
    removed_zones = original_rows - len(df_modified)
    print(f"\nRemoved {removed_zones} rows where 'updated_trajectory' is None.")

    # Generate the output Parquet file path
    output_parquet_geocodes = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_csv))[0]}_trajectory_with_geocodes.parquet")

    # Save the results to the output Parquet file
    df_modified.to_parquet(output_parquet_geocodes, index=False)

    print(f"\nResults saved to {output_parquet_geocodes}")

def geocode_zones(input_dir, output_dir, coordinates_csv):
    print("\nInside the geocode_zones function..")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all CSV files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    # Read the coordinates CSV file
    coordinates_df = pd.read_csv(coordinates_csv)

    for input_csv in input_files:
        process_trajectory(os.path.join(input_dir, input_csv), coordinates_df, output_dir)

if __name__ == '__main__':
    # Example usage
    input_directory = "/Users/arun/Desktop/Internship/preprocess/trajectories_as_zones_output_batches/"
    output_directory = "/Users/arun/Desktop/Internship/preprocess/trajectories_as_geocodes_output/"
    coordinates_csv = "/Users/arun/Desktop/Internship/preprocess/zones_and_geocodes.csv"  # Replace with the actual path to your coordinates CSV file

    print("\nThe main function is starting now..")
    geocode_zones(input_directory, output_directory, coordinates_csv)
