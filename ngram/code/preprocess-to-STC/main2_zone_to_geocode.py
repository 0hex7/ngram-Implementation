import pandas as pd
from gomaps import geocoder
from tqdm import tqdm

def geocode_zone_with_gomaps(zone):
    print(f"{zone} is being processed")
    try:
        results = geocoder(f"{zone}, New York")
        if results:
            print(f"{zone}, New York has geocodes")
            print(results)
            return results
        else:
            print("no results")
            return None, None
    except Exception as e:
        print(f"Error geocoding zone {zone} with gomaps: {e}")
        return None, None

def append_and_geocode(input_csv, output_csv):
    # Read the input CSV file
    print("\nCSV file is being read")
    df = pd.read_csv(input_csv)

    # Geocode using the provided function
    df[['latitude', 'longitude']] = df['Zone'].apply(geocode_zone_with_gomaps).apply(pd.Series)

    # Create a new dataframe with 'zone', 'latitude', and 'longitude' columns
    result_df = df[['Zone', 'latitude', 'longitude']]

    # Save the result to the output CSV file
    result_df.to_csv(output_csv, index=False)
    print(f"Result saved to {output_csv}")

if __name__ == '__main__':
    # Example usage
    input_csv = "/Users/arun/Desktop/Internship/datasets/taxi-zone-lookup.csv"
    output_csv = "/Users/arun/Desktop/Internship/preprocess/zones_and_geocodes.csv"

    append_and_geocode(input_csv, output_csv)
