import os
import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm
import ast
import json
from datetime import datetime, timedelta
import random

def top_2000(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Count the occurrences of each 'venueId'
    venue_counts = df['venueId'].value_counts()

    # Select the top 2000 most frequently visited POIs
    top_2000_venues = venue_counts.head(2000).index.tolist()

    # Filter the DataFrame to include only the top 2000 venue IDs
    top_2000_df = df[df['venueId'].isin(top_2000_venues)][['venueId', 'latitude', 'longitude', 'venueCategoryId', 'venueCategory', 'timezoneOffset', 'utcTimestamp']].drop_duplicates(subset='venueId')

    # Convert the 'utcTimestamp' column to datetime format
    top_2000_df['utcTimestamp'] = pd.to_datetime(top_2000_df['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y').dt.strftime('%H-%M-%S')

    # Save the top 2000 venues' latitude, longitude, and formatted time to CSV
    top_2000_df.to_csv(output_csv, index=False, columns=['venueId', 'latitude', 'longitude', 'venueCategoryId', 'venueCategory', 'timezoneOffset', 'utcTimestamp'])

    print(f"\nTop 2000 venues' latitude, longitude, and formatted time saved to: {output_csv}\n")
    print("\n========================================================================================\n")


def step0(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all .parquet files in the input directory
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]

    total_values_removed = 0  # Variable to store the total number of values removed

    for parquet_file in tqdm(parquet_files, desc="\nStep 0: Processing all the .parquet files in the directory"):
        input_path = os.path.join(input_dir, parquet_file)
        output_path = os.path.join(output_dir, f"step0_{parquet_file}")

        # Read the .parquet file
        df = pd.read_parquet(input_path)

        # Count the number of rows before removal
        num_rows_before = len(df)

        # Remove individual None values in trajectories
        df['updated_trajectory'] = df['updated_trajectory'].apply(lambda traj: [point for point in traj if point is not None])

        # Count the number of rows after removal
        num_rows_after = len(df)

        # Update the total number of values removed
        total_values_removed += num_rows_before - num_rows_after

        # Save the updated DataFrame to a new .parquet file
        df.to_parquet(output_path, index=False)

        # Print information about the updated Parquet file
        #print(f"Updated Parquet file: {output_path}")
        #print(f"Number of rows before removal: {num_rows_before}")
        #print(f"Number of rows after removal: {num_rows_after}")

    print(f"\nStep 0 completed, and updated .parquet files saved to: {output_dir}")
    print(f"\nTotal values removed across all files: {total_values_removed}\n")
    print("\n========================================================================================\n")

def calculate_distance(point1, point2):
    try:
        # Convert points to tuples if they are strings
        if isinstance(point1, str):
            point1 = ast.literal_eval(point1)
        if isinstance(point2, str):
            point2 = ast.literal_eval(point2)

        distance = geodesic(point1, point2).meters
        return distance
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return float('inf')

def is_within_100m(point, top_2000_df):
    #print(top_2000_df.columns)

    try:
        # Check if the point is within 100 meters of any top_2000 geocodes
        for _, top_point in top_2000_df.iterrows():
            distance = calculate_distance(point, (top_point['latitude'], top_point['longitude']))
            if distance <= 100:
                #print(f"Point {point} is within 100 meters of top_2000 geocode ({top_point['latitude']}, {top_point['longitude']})")

                # Extract venue ID and coordinates after stripping extraneous characters
                venue_id = str(top_point['venueId'])
                #print("venueid:",venue_id)
                latitude = str(top_point['latitude']).strip("{'")
                #print("lat:",latitude)
                longitude = str(top_point['longitude']).strip("'}")
                #print("long:",longitude)
                venueCategoryId = str(top_point['venueCategoryId'])
                venueCategory = str(top_point['venueCategory'])
                #timezoneOffset = str(top_point['timezoneOffset'])
                utcTimestamp = str(top_point['utcTimestamp'])
                
                
                #print(f"Venue ID: {venue_id}, Latitude: {latitude}, Longitude: {longitude}")

                return True, (venue_id, latitude, longitude, venueCategoryId, venueCategory, utcTimestamp)

        return False, None
    except Exception as e:
        print(f"Error in is_within_100m: {e}")
        return False, None


def convert_to_tuple(value):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None


# ...

# def step1(input_dir, top_2000_file, output_dir):
#     """With the latitudes and longtitudes, we can match the co-ordinate data with the nearest POI, 
#     These POI are taken from the Foursquare check-in top-2000 frequently visited POIs
# if no POIs within 100m are found, we discard the point
# """
#     os.makedirs(output_dir, exist_ok=True)

#     # Read the top 2000 venues CSV file
#     top_2000_df = pd.read_csv(top_2000_file)

#     # Get a list of all .parquet files in the input directory
#     parquet_files = [f for f in os.listdir(input_dir) if f.startswith('step0') and f.endswith('.parquet')]

#     total_geocodes_removed = 0
#     total_geocodes = 0

#     # ...

#     for parquet_file in tqdm(parquet_files, desc="Step 1: Processing .parquet files"):
#         input_path = os.path.join(input_dir, parquet_file)
#         output_path = os.path.join(output_dir, f"step1_{parquet_file}")

#         # Read the .parquet file without specifying columns
#         df = pd.read_parquet(input_path)

#         # Initialize an empty list to store trajectory strings for each row
#         trajectory_strings = []

#         with tqdm(total=len(df), desc=f"Processing {parquet_file}", position=0, leave=False) as row_progress:
#             for index, row in df.iterrows():
#                 # Remove geocodes which are not within 100 meters of top 2000 geocodes
#                 updated_trajectory = []
#                 for point in row['updated_trajectory']:
#                     within_100m, venueID = is_within_100m(point, top_2000_df)
#                     if within_100m:
#                         updated_trajectory.append(venueID)

#                 # Extract the first point in the updated trajectory
#                 first_point = updated_trajectory[0][1:] if updated_trajectory else (None, None)
#                 #print(f"Updated Trajectory for Row {index}: {updated_trajectory}\n")

#                 # Construct the trajectory string for the current row
#                 trajectory_string = ",".join(["'" + f"{element[0]}:{element[1]},{element[2]},{element[3]},{element[4]},{element[5]},{element[6]} " + "'" for element in updated_trajectory])

#                 # Append the trajectory string to the list
#                 trajectory_strings.append(trajectory_string)

#                 # Update the progress bar for each row
#                 row_progress.update(1)

#         # Assign the list of trajectory strings to the 'trajectory_as_stc' column
#         df['trajectory_as_stc'] = trajectory_strings
        
#         # drop the updated_trajectory column, to have only the stc column
#         df = df.drop(columns=['updated_trajectory'])

#         total_geocodes_removed += sum(len(points) for points in updated_trajectory)

#         # Save the updated DataFrame to a new .parquet file
#         df.to_parquet(output_path, index=False)

#         # Assuming 'df' is your DataFrame
#         # Set pandas display options to show more content

#         pd.set_option('display.max_columns', None)
#         pd.set_option('display.max_rows', None)
#         pd.set_option('display.width', None)
#         pd.set_option("display.max_colwidth", None)
#         pd.options.display.max_columns = None
#         # columnname = 'trajectory_as_stc'
#         # first_row = df.iloc[[0]]

#         # #first_row = df.at[0, columnname]
#         # second_row = df.at[1, columnname]

#         # # Print the first row
#         # print(f"\nFirst Row:\n{first_row}")
#         # print(f"\nsecond Row:\n{second_row}")
        

#         # # ... (rest of your code)


#         # #print(f"Type: {type(df)}")
#         # #print(f"Head:\n{df.head()}")
#         #print(f"Entire Values:\n{df}")
#         # #print(f"Shape: {df.shape}\n")

#     print(f"\nStep 1 completed. Processed .parquet files saved to: {output_dir}")
#     print(f"Removed {total_geocodes_removed} out of {total_geocodes} geocodes across all files\n")



def step1(input_dir, top_2000_file, output_dir):
    """With the latitudes and longitudes, we can match the coordinate data with the nearest POI, 
    These POI are taken from the Foursquare check-in top-2000 frequently visited POIs.
    If no POIs within 100m are found, we discard the point.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read the top 2000 venues CSV file
    top_2000_df = pd.read_csv(top_2000_file)

    # Get a list of all .parquet files in the input directory
    parquet_files = [f for f in os.listdir(input_dir) if f.startswith('step0') and f.endswith('.parquet')]

    total_geocodes_removed = 0
    total_geocodes = 0

    for parquet_file in tqdm(parquet_files, desc="\nStep 1: Processing all the .parquet files in the directory"):
        input_path = os.path.join(input_dir, parquet_file)
        output_path = os.path.join(output_dir, f"step1_{parquet_file}")

        # Read the .parquet file without specifying columns
        df = pd.read_parquet(input_path)

        # Initialize an empty list to store trajectory strings for each row
        trajectory_strings = []

        with tqdm(total=len(df), desc=f"Processing {parquet_file}", position=0, leave=False) as row_progress:
            for index, row in df.iterrows():
                # Remove geocodes which are not within 100 meters of top 2000 geocodes
                updated_trajectory = []
                for point in row['updated_trajectory']:
                    within_100m, venueID = is_within_100m(point, top_2000_df)
                    if within_100m:
                        updated_trajectory.append(venueID)

                # Extract the first point in the updated trajectory
                #first_point = updated_trajectory[0][1:] if updated_trajectory else (None, None)
                #print(f"Updated Trajectory for Row {index}: {updated_trajectory}\n")

                # Construct the trajectory string for the current row
                # trajectory_string = ",".join(["'" + f"{element[0]}:{element[1]},{element[2]},{element[3]},{element[4]},{element[5]} " + "'" for element in updated_trajectory])
                trajectory_string = "|".join(["'" + f"{element[0]},{element[1]},{element[2]},{element[3]},{element[4]},{element[5]} " + "'" for element in updated_trajectory])

                # Append the trajectory string to the list
                trajectory_strings.append(trajectory_string)

                # Update the progress bar for each row
                row_progress.update(1)

                # Update the total geocodes count
                total_geocodes += len(row['updated_trajectory'])
                total_geocodes_removed += len(updated_trajectory)

        # Assign the list of trajectory strings to the 'trajectory_as_stc' column
        df['trajectory_as_stc'] = trajectory_strings

        # drop the updated_trajectory column, to have only the stc column
        df = df.drop(columns=['updated_trajectory'])

        # Save the updated DataFrame to a new .parquet file
        df.to_parquet(output_path, index=False)

        # Assuming 'df' is your DataFrame
        # Set pandas display options to show more content
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option("display.max_colwidth", None)
        pd.options.display.max_columns = None
        #print(f"Type: {type(df)}")
        #print(f"Head:\n{df.head()}")
        #print(f"\n\nEntire Values:\n{df}")
        #print(f"Shape: {df.shape}\n")

    print(f"\nStep 1 completed, and processed .parquet files saved to: {output_dir}")
    print(f"\nRemoved {total_geocodes_removed} out of {total_geocodes} geocodes across all files\n")
    print("\n========================================================================================\n")



def step2(input_dir, output_dir):
    """Clean the data by removing repeat points with the same venue ID
    or exact latitude-longitude location."""
    
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all .parquet files in the input directory
    parquet_files = [f for f in os.listdir(input_dir) if f.startswith('step1') and f.endswith('.parquet')]

    total_removed_combos = 0
    total_geocodes = 0

    for parquet_file in tqdm(parquet_files, desc="\nStep 2: Processing all the .parquet files in the directory"):
        input_path = os.path.join(input_dir, parquet_file)
        output_path = os.path.join(output_dir, f"step2_{parquet_file}")

        # Read the .parquet file without specifying columns
        df = pd.read_parquet(input_path)

        # Initialize a list to store cleaned trajectory strings
        cleaned_trajectories = []

        with tqdm(total=len(df), desc=f"\nStep 2 being processed for {parquet_file}", position=0, leave=False) as row_progress:
            for index, row in df.iterrows():
                # Remove repeated venue IDs or exact latitude-longitude locations
                cleaned_trajectory = []
                unique_combos = set()  # To keep track of unique combos in the trajectory
                #print("&&&&&&&&&&&&")
                #print(df)
                for point in row['trajectory_as_stc'].split("|"):
                    #print("point is:\n", point)
                    try:
                        # Remove single quotes from the trajectory point
                        point = point.strip("'")
                        #print("point is:\n", point)
                        
                        # Assuming point structure: 'venueId,latitude,longitude,venueCategoryId,venueCategory,utcTimestamp'
                        # Split the point data and extract relevant information
                        venue_id, latitude, longitude, venueCategoryId, venueCategory, utcTimestamp = point.split(',')
                        #print("point inside is\n",venue_id, latitude, longitude)

                        trajectory_part = f"{venue_id},{latitude},{longitude}"
                        #print(trajectory_part,"is trajectory part\n")
                        # Check for uniqueness based on venueId or latitude-longitude combination
                        if venue_id not in unique_combos and trajectory_part not in unique_combos:
                            cleaned_trajectory.append(point)
                            unique_combos.add(venue_id)
                            unique_combos.add(trajectory_part)
                        else:
                            # Print information about the removed trajectory
                            print(f"\n[-] Removed duplicate trajectory: {point}")
                    except ValueError:
                        # Print the problematic point for debugging
                        print(f"Error in splitting point: {point}")

                # Calculate the number of removed location combos
                removed_combos = len(row['trajectory_as_stc'].split('|')) - len(cleaned_trajectory)
                total_removed_combos += removed_combos
                total_geocodes += len(row['trajectory_as_stc'].split('|'))

                # Join the cleaned trajectory points to form a string
                cleaned_trajectory_string = '|'.join(cleaned_trajectory)
                cleaned_trajectories.append(cleaned_trajectory_string)

                # Update the progress bar for each row
                row_progress.update(1)

        # Assign the list of cleaned trajectory strings to the 'cleaned_trajectory_as_stc' column
        df['cleaned_trajectory_as_stc'] = cleaned_trajectories

        df = df.drop(columns=['trajectory_as_stc'])

        # Save the updated DataFrame to a new .parquet file
        df.to_parquet(output_path, index=False)
        #print(df)
    print(f"\nStep 2 completed, and processed .parquet files saved to: {output_dir}")
    print(f"\nTotal geocodes removed across all files: {total_removed_combos} from {total_geocodes} geocodes.\n")
    print("\n========================================================================================\n")
# Example usage:
# step2('/path/to/input', '/path/to/output')


def step2duplicate(input_dir, output_dir):
    """“We clean the data by removing repeat points with the same venue ID 
    or exact latitude-longitude location.”"""
    """This checks for adjacent repeated occurances!!"""
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all .parquet files in the input directory
    parquet_files = [f for f in os.listdir(input_dir) if f.startswith('step1') and f.endswith('.parquet')]

    total_removed_combos = 0
    total_geocodes = 0

    for parquet_file in tqdm(parquet_files, desc="Step 2: Duplicate Processing .parquet files"):
        input_path = os.path.join(input_dir, parquet_file)
        output_path = os.path.join(output_dir, f"step2_{parquet_file}")

        # Read the .parquet file without specifying columns
        df = pd.read_parquet(input_path)

        # Initialize a list to store cleaned trajectory strings
        cleaned_trajectories = []

        with tqdm(total=len(df), desc=f"Processing {parquet_file}", position=0, leave=False) as row_progress:
            for index, row in df.iterrows():
                # Remove only adjacent repeated venue IDs or exact latitude-longitude locations
                cleaned_trajectory = []
                last_point = None

                for point in row['trajectory_as_stc'].split('|'):
                    if point != last_point:
                        cleaned_trajectory.append(point)
                        last_point = point

                # Calculate the number of removed location combos
                removed_combos = len(row['trajectory_as_stc'].split(',')) - len(cleaned_trajectory)
                total_removed_combos += removed_combos
                total_geocodes += len(row['trajectory_as_stc'].split(','))

                # Join the cleaned trajectory points to form a string
                cleaned_trajectory_string = '|'.join(cleaned_trajectory)
                cleaned_trajectories.append(cleaned_trajectory_string)

                # Update the progress bar for each row
                row_progress.update(1)

        # Assign the list of cleaned trajectory strings to the 'cleaned_trajectory_as_stc' column
        df['cleaned_trajectory_as_stc'] = cleaned_trajectories

        df = df.drop(columns=['trajectory_as_stc'])
        #print(f"Entire Values:\n{df}")
        #print(f"Removed {total_removed_combos} from {total_geocodes} geocodes in {parquet_file}")

        # Save the updated DataFrame to a new .parquet file
        df.to_parquet(output_path, index=False)

    print(f"\nStep 2: Duplicate Processing completed. Processed .parquet files saved to: {output_dir}")
    print(f"Total location combos removed across all files: {total_removed_combos} from {total_geocodes} geocodes.")




# from datetime import datetime, timedelta

# import os
# import pandas as pd
# from tqdm import tqdm
# import random

# def step3(input_dir, output_dir, gt=250):
#     """
#     Processes .parquet files in the input directory, filtering geocodes based on time differences.

#     Args:
#         input_dir (str): Path to the directory containing input .parquet files.
#         output_dir (str): Path to the directory where processed files will be saved.
#         gt (int, optional): Minimum time difference (in minutes) to keep a geocode. Defaults to 10.
#     """

#     os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

#     # Get a list of all .parquet files in the input directory
#     parquet_files = [f for f in os.listdir(input_dir) if f.startswith('step2') and f.endswith('.parquet')]

#     total_removed_geocodes = 0  # Counter for removed geocodes
#     total_geocodes = 0  # Counter for total geocodes

#     for parquet_file in tqdm(parquet_files, desc="Step 3: Processing .parquet files"):
#         input_path = os.path.join(input_dir, parquet_file)
#         output_path = os.path.join(output_dir, f"step3_{parquet_file}")

#         # Read the .parquet file
#         df = pd.read_parquet(input_path)

#         with tqdm(total=len(df), desc=f"Processing {parquet_file}", position=0, leave=False) as row_progress:
#             for index, row in df.iterrows():
#                 cleaned_trajectory = []  # List to store filtered geocodes

#                 # Split the trajectory into individual geocodes
#                 geocodes = [point for point in row['cleaned_trajectory_as_stc'].split("|")]
#                 total_geocodes += len(geocodes)  # Count total geocodes

#                 utcTimestamps = []
#                 for geocode in geocodes:
#                     try:
#                         # Split the geocode data
#                         parts = geocode.rstrip(",").split(",")
#                         venue_id, latitude, longitude, venueCategoryId, venueCategory, utcTimestamp = parts[:6]

#                         utcTimestamps.append(utcTimestamp)  # Collect timestamps
#                         utcTimestamp = utcTimestamp.replace(" ", "")
#                         print("****************", utcTimestamp)
#                     except ValueError as e:
#                         # Print problematic geocode for debugging
#                         print(f"Error in splitting geocode: {geocode}, Error: {e}")

#                 # Filter geocodes based on time differences using pandas functionality
#                 temp_df = pd.DataFrame({'timestamp': [pd.to_datetime(utcTimestamp.strip(), format='%H-%M-%S') for utcTimestamp in utcTimestamps],
#                                         'geocode': geocodes})
#                 print("___________ temp_df",temp_df)
#                 temp_df['time_diff'] = (abs(temp_df['timestamp'].shift() - temp_df['timestamp']).dt.total_seconds() / 60)
#                 print("+++++++++++ temp_df after diff check", temp_df)
#                 filtered_df = temp_df[temp_df['time_diff'].values >= gt]
#                 print("########### filtered_df", filtered_df)

#                 # Access the Series using .values
#                 filtered_geocodes = filtered_df['geocode'].tolist()  # Extract filtered geocodes

#                 # Randomly select one geocode from filtered_geocodes
#                 #cleaned_trajectory.append(random.choice(filtered_geocodes) if filtered_geocodes else "")
#                 cleaned_trajectory.append("|".join(filtered_geocodes) if filtered_geocodes else "")
                
#                 print("cleaned trajectory is _________",cleaned_trajectory)
#                 row_progress.update(1)  # Update progress bar
#                 df['STC_trajectory'] = [cleaned_trajectory] * len(df)

#         # Assign the filtered geocodes to the 'STC_trajectory' column
#         #df['STC_trajectory'] = [cleaned_trajectory] * len(df)
#         df = df.drop(columns=['cleaned_trajectory_as_stc'])

#         print("^^^^^^^^^",df)
#         # Save the processed DataFrame to a new .parquet file
#         df.to_parquet(output_path, index=False)

#         # Print the number of geocodes removed for the current file
#         print(f"Total geocodes removed from {len(df)} geocodes in {parquet_file} is: {total_removed_geocodes}")


def step3(input_dir, output_dir, gt=200):
    """
    Processes .parquet files in the input directory, filtering geocodes based on time differences.

    Args:
        input_dir (str): Path to the directory containing input .parquet files.
        output_dir (str): Path to the directory where processed files will be saved.
        gt (int, optional): Minimum time difference (in minutes) to keep a geocode. Defaults to 10.
    """

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Get a list of all .parquet files in the input directory
    parquet_files = [f for f in os.listdir(input_dir) if f.startswith('step2') and f.endswith('.parquet')]

    for parquet_file in tqdm(parquet_files, desc="\nStep 3: Processing all the .parquet files in the directory"):
        input_path = os.path.join(input_dir, parquet_file)
        output_path = os.path.join(output_dir, f"step3_{parquet_file}")

        # Read the .parquet file
        df = pd.read_parquet(input_path)

        # Count total trajectories in the initial DataFrame
        total_initial_trajectories = sum(len(row['cleaned_trajectory_as_stc'].split("|")) for _, row in df.iterrows())

        total_removed_geocodes = 0  # Counter for removed geocodes
        cleaned_trajectories = []  # List to store cleaned trajectories

        with tqdm(total=len(df), desc=f"\nStep 3 being processed for {parquet_file}", position=0, leave=False) as row_progress:
            for _, row in df.iterrows():
                cleaned_trajectory = []  # List to store filtered geocodes

                # Split the trajectory into individual geocodes
                geocodes = [point for point in row['cleaned_trajectory_as_stc'].split("|")]

                utcTimestamps = []
                for geocode in geocodes:
                    try:
                        # Split the geocode data
                        parts = geocode.rstrip(",").split(",")
                        venue_id, latitude, longitude, venueCategoryId, venueCategory, utcTimestamp = parts[:6]

                        utcTimestamps.append(utcTimestamp)  # Collect timestamps
                        utcTimestamp = utcTimestamp.replace(" ", "")
                    except ValueError as e:
                        # Print problematic geocode for debugging
                        print(f"Error in splitting geocode: {geocode}, Error: {e}")

                # Filter geocodes based on time differences using pandas functionality
                temp_df = pd.DataFrame({'timestamp': [pd.to_datetime(utc.strip(), format='%H-%M-%S') for utc in utcTimestamps],
                                        'geocode': geocodes})
                temp_df['time_diff'] = abs((temp_df['timestamp'].shift() - temp_df['timestamp']).dt.total_seconds() / 60)
                temp_df['time_diff'] = temp_df['time_diff'].fillna(float('nan'))  # Handle NaN for the first trajectory

                # Filter geocodes based on time differences
                mask = (temp_df['time_diff'] >= gt) | temp_df['time_diff'].isna()
                filtered_df = temp_df[mask]

                # Access the Series using .values
                filtered_geocodes = filtered_df['geocode'].tolist()  # Extract filtered geocodes
                cleaned_trajectory.append("|".join(filtered_geocodes) if filtered_geocodes else "")

                # Print time_diff for each trajectory
                for time_diff, geocode in zip(temp_df['time_diff'], temp_df['geocode']):
                    if not pd.isna(time_diff) and time_diff < gt:
                        print(f"\n[-] Deleting point due to time_diff < {gt} minutes: {geocode}, Time Difference: {time_diff}")

                row_progress.update(1)  # Update progress bar
                cleaned_trajectories.append("|".join(filtered_geocodes) if filtered_geocodes else "")

        # Assign the cleaned trajectories to the 'STC_trajectory' column
        df['STC_trajectory'] = cleaned_trajectories
        df = df.drop(columns=['cleaned_trajectory_as_stc'])

        # Save the original and cleaned DataFrames to separate .parquet files
        df.to_parquet(output_path, index=False)

        #cleaned_df = pd.DataFrame({'STC_trajectory': cleaned_trajectories})
        #print(cleaned_df)
        #cleaned_df.to_parquet(os.path.join(output_dir, f"cleaned_{parquet_file}"), index=False)

        # Print statistics
        total_removed_geocodes += total_initial_trajectories - sum(len(cleaned_trajectory.split("|")) for cleaned_trajectory in cleaned_trajectories)
        print(f"\nStatistics for {parquet_file}:")
        print(f"Total initial trajectories: {total_initial_trajectories}")
        print(f"Total trajectories after processing: {sum(len(cleaned_trajectory.split('|')) for cleaned_trajectory in cleaned_trajectories)}")
        print(f"Total trajectories removed: {total_removed_geocodes}")
        print(f"Processing completed for {parquet_file}\n")
    print(f"\nStep 3 completed, and processed .parquet files saved to: {output_dir}")
    print("\n========================================================================================\n")


def step4(input_dir, output_dir):
    """
    Processes .parquet files in the input directory, converting trajectories and saving the output.

    Args:
        input_dir (str): Path to the directory containing input .parquet files (with 'step3_' prefix).
        output_dir (str): Path to the directory where processed files will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Get a list of all .parquet files in the input directory with 'step3_' prefix
    parquet_files = [f for f in os.listdir(input_dir) if f.startswith('step3_') and f.endswith('.parquet')]

    # Progress bar for processing the entire directory
    with tqdm(total=len(parquet_files), desc="Step 4: Processing all the .parquet files in the directory", position=0, leave=True) as dir_progress:
        for parquet_file in parquet_files:
            input_path = os.path.join(input_dir, parquet_file)
            output_path = os.path.join(output_dir, f"step4_{parquet_file}")

            # Read the .parquet file
            df = pd.read_parquet(input_path)

            # Process each row's trajectories
            processed_trajectories = []
            total_rows = 0  # Counter for total rows
            total_trajectories = 0  # Counter for total trajectories

            # Progress bar for processing each file's rows
            with tqdm(total=len(df), desc=f"\nStep 4 being processed for {parquet_file}", position=1, leave=False) as row_progress:
                for idx, row in df.iterrows():
                    total_rows += 1  # Increment total rows counter
                    new_trajectories = []

                    # Split the row's trajectories using '|'
                    for trajectory in row['STC_trajectory'].split('|'):
                        parts = trajectory.split(',')
                        # Extract only the 2nd, 3rd, 5th, and 6th values and create a new trajectory
                        new_trajectory = f"{parts[1]},{parts[2]},{parts[4]},{parts[5]}"
                        print(f"\n[+] Row {idx + 1}, New trajectory: {new_trajectory}")
                        new_trajectories.append(new_trajectory)

                    # Combine the new trajectories using '|'
                    processed_trajectory = '|'.join(new_trajectories)
                    processed_trajectories.append(processed_trajectory)

                    # Print each processed row for better understanding
                    print(f"\nRow {idx + 1}, Entire trajectories as POI for this row is: {processed_trajectory}")
                    row_progress.update(1)  # Update progress bar for rows
                    total_trajectories += len(new_trajectories)  # Increment total trajectories counter

            # Create a new DataFrame with the processed trajectories
            processed_df = pd.DataFrame({'STC_trajectory as POI': processed_trajectories})
            #print(processed_df)
            # Save the processed DataFrame to a new .parquet file
            processed_df.to_parquet(output_path, index=False)
            
            # Print summary information for the processed file
            print(f"\nProcessed data saved to: {output_path}")
            print(f"\nTotal rows in file: {total_rows}")
            print(f"Total trajectories in file: {total_trajectories}\n")

            dir_progress.update(1)  # Update progress bar for files

    print(f"\nStep 4 completed, and processed .parquet files saved to: {output_dir}")
    print("\n========================================================================================\n")

# Example usage
input_csv = "/Users/arun/Desktop/ngram-Implementation/ngram/data/dataset_TSMC2014_NYC.csv"
output_top_2000_csv = "/Users/arun/Desktop/ngram-Implementation/ngram/code/preprocess-to-STC/top_2000_csv.csv"
input_parquet_dir = "/Users/arun/Desktop/ngram-Implementation/ngram/code/preprocess-to-STC/trajectories_as_geocodes_output/xox/"
output_processed_dir = "/Users/arun/Desktop/ngram-Implementation/ngram/code/preprocess-to-STC/trajectories_as_stc_stepwise_outputs"

# Generate the top 2000 venues CSV
top_2000(input_csv, output_top_2000_csv)

# Run Step 0 to remove individual None values
step0(input_parquet_dir, output_processed_dir)

# Run Step 1 to process .parquet files based on top 2000 venues
step1(output_processed_dir, output_top_2000_csv, output_processed_dir)

step2(output_processed_dir, output_processed_dir)
#step2duplicate(output_processed_dir, output_processed_dir)

step3(output_processed_dir, output_processed_dir)

step4(output_processed_dir, output_processed_dir)
