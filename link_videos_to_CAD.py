import pandas as pd
import numpy as np
from datetime import timedelta
import os
import warnings
from tqdm import tqdm
import sys

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### KEY FUNCTIONS ###

def match_videos(df_videos, df_reports):
    '''
    Function to match CAD reports with DFs if start_time of DF is within 10
    minutes of the CAD report.

    NOTE: The CAD is mapped to the start_time timestamp of the bodycam because
    officers should call dispatch before an interaction.
    '''
    # Initialize new columns in df_videos
    df_videos['CAD'] = np.nan
    df_videos['report_time'] = np.nan
    df_videos['video_report_distance'] = np.nan
    df_videos['Other_Matches'] = np.nan

    # Combine start_date and start_time into start_datetime
    try:
        df_videos['start_datetime'] = pd.to_datetime(df_videos['start_date'] + ' ' + df_videos['start_time'], errors='coerce')
        df_videos['end_datetime'] = pd.to_datetime(df_videos['end_date'] + ' ' + df_videos['end_time'], errors='coerce')
    except Exception as e:
        print(f"Error in converting dates: {e}")
        return df_videos

    # Iterate through each video in df_videos
    for i in tqdm(range(len(df_videos)), desc="Processing videos"):
        best_distance = pd.Timedelta.max
        best_cad = None
        matched_report_time = None
        other_matches = []

        try:
            # Iterate through each written record in df_reports
            for j in range(len(df_reports)):
                a = df_videos.iloc[i]['start_datetime']
                b = pd.to_datetime(df_reports.iloc[j]['date and time'], errors='coerce')

                # Skip if 'b' is NaT
                if pd.isna(b):
                    continue

                distance = abs(a - b)

                # Check if this distance is the best one we've found so far
                if distance < best_distance:
                    best_distance = distance
                    best_cad = df_reports.iloc[j]['cad event number']
                    matched_report_time = b

                # Check for other matches within 10 minutes
                if pd.Timedelta(0) < distance <= timedelta(minutes=10):
                    other_matches.append(df_reports.iloc[j]['cad event number'])

            # Update the original df_videos DataFrame
            df_videos.at[df_videos.index[i], 'CAD'] = str(best_cad) if best_cad is not None else np.nan
            df_videos.at[df_videos.index[i], 'report_time'] = str(matched_report_time) if matched_report_time is not None else np.nan
            df_videos.at[df_videos.index[i], 'video_report_distance'] = str(best_distance) if best_distance is not None else np.nan

            # Convert other matches list to string and update DataFrame
            df_videos.at[df_videos.index[i], 'Other_Matches'] = ', '.join(map(str, other_matches)) if other_matches else np.nan

        except Exception as e:
            # If an error occurs, mark the row with 'DISTANCE CALCULATION ERROR'
            df_videos.at[df_videos.index[i], 'CAD'] = 'DISTANCE CALCULATION ERROR'
            df_videos.at[df_videos.index[i], 'report_time'] = 'DISTANCE CALCULATION ERROR'
            df_videos.at[df_videos.index[i], 'video_report_distance'] = 'DISTANCE CALCULATION ERROR'
            df_videos.at[df_videos.index[i], 'Other_Matches'] = 'DISTANCE CALCULATION ERROR'

    # Drop the specified columns before returning
    df_videos = df_videos.drop(columns=['start_date', 'start_time', 'end_date', 'end_time'])

    return df_videos

def clean_other_matches(df_videos):
    '''
    Function to remove duplicate CAD numbers in columns 'CAD' and 'Other_Matches'
    '''
    # Iterate through each video in df_videos
    for i in range(len(df_videos)):
        # Split the 'Other_Matches' column by commas to get a list of CAD numbers
        other_matches = df_videos.at[df_videos.index[i], 'Other_Matches']
        if pd.isna(other_matches):
            continue  # Skip if there are no other matches
        other_matches_list = other_matches.split(', ')
        
        # Get the CAD number from the 'CAD' column
        cad_number = str(df_videos.at[df_videos.index[i], 'CAD'])
        
        # Remove the CAD number from the 'Other_Matches' list if it's present
        cleaned_matches = [cad for cad in other_matches_list if cad != cad_number]
        
        # Update the 'Other_Matches' column with the cleaned list
        df_videos.at[df_videos.index[i], 'Other_Matches'] = ', '.join(cleaned_matches)
    
    return df_videos

def reorder_and_rename_columns(df):
    '''
    Function to reorder and rename columns as specified
    '''
    columns_order = ['file_name', 'start_datetime', 'end_datetime', 'creation_time', 'duration', 'CAD', 'report_time', 'video_report_distance', 'Other_Matches']
    df = df[columns_order]
    df = df.rename(columns={
        'creation_time': 'video_creation_time',
        'duration': 'video_duration',
        'Other_Matches': 'other_matches'
    })
    return df

def load_existing_video_names(output_dir):
    '''
    Function to load existing video names from linked_videos.csv and unmatched_videos.csv
    '''
    existing_video_names = set()
    linked_path = os.path.join(output_dir, 'linked_videos.csv')
    unmatched_path = os.path.join(output_dir, 'unmatched_videos.csv')

    if os.path.exists(linked_path):
        linked_videos = pd.read_csv(linked_path)
        existing_video_names.update(linked_videos['file_name'].unique())

    if os.path.exists(unmatched_path):
        unmatched_videos = pd.read_csv(unmatched_path)
        existing_video_names.update(unmatched_videos['file_name'].unique())

    return existing_video_names

def main():
    if len(sys.argv) != 4:
        sys.exit(1)

    cad_csv_path = sys.argv[1]
    video_csv_path = sys.argv[2]
    output_dir = sys.argv[3]

    # Read in data
    bodycam_df = pd.read_csv(video_csv_path)
    combined_df = pd.read_csv(cad_csv_path)

    # Ensure that file_name column is present in bodycam_df
    if 'file_name' not in bodycam_df.columns:
        bodycam_df['file_name'] = os.path.basename(video_csv_path)

    # Load existing video names
    existing_video_names = load_existing_video_names(output_dir)

    # Filter out videos that are already processed
    bodycam_df = bodycam_df[~bodycam_df['file_name'].isin(existing_video_names)]

    if bodycam_df.empty:
        print("No new videos to process.")
        return

    # Connect videos to CAD
    linked_videos = match_videos(bodycam_df, combined_df)

    # Ensure 'video_report_distance' is converted to Timedelta properly
    linked_videos['video_report_distance'] = pd.to_timedelta(linked_videos['video_report_distance'], errors='coerce')

    # Separate valid and invalid distance values
    valid_distances_mask = linked_videos['video_report_distance'].notna()
    invalid_distances_mask = ~valid_distances_mask

    # Separate matched and unmatched videos
    matched_videos = linked_videos[valid_distances_mask & (linked_videos['video_report_distance'] <= pd.Timedelta(minutes=120))]
    unmatched_videos = linked_videos[invalid_distances_mask | (linked_videos['video_report_distance'] > pd.Timedelta(minutes=120))]

    # Clean up final spreadsheet for matched videos
    matched_videos_cleaned = clean_other_matches(matched_videos)

    # Reorder and rename columns for both matched and unmatched videos
    matched_videos_cleaned = reorder_and_rename_columns(matched_videos_cleaned)
    unmatched_videos = reorder_and_rename_columns(unmatched_videos)

    # Save linked spreadsheet to a separate file in user provided directory
    linked_output_path = os.path.join(output_dir, 'linked_videos.csv')
    unmatched_output_path = os.path.join(output_dir, 'unmatched_videos.csv')

    # Append or save the results
    def append_to_csv(df, path):
        if os.path.exists(path):
            existing_df = pd.read_csv(path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)

    append_to_csv(matched_videos_cleaned, linked_output_path)
    append_to_csv(unmatched_videos, unmatched_output_path)

    print(f"Linked videos saved to {linked_output_path}")
    print(f"Unmatched videos saved to {unmatched_output_path}")

if __name__ == "__main__":
    main()
