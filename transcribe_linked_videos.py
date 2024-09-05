import pandas as pd
import os
import sys

def add_transcripts_to_dataframe(df, transcript_dir):
    transcripts = {}
    
    # Read all transcripts into a dictionary
    for filename in os.listdir(transcript_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(transcript_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                transcripts[os.path.splitext(filename)[0]] = file.read()

    # Add the transcript content to the DataFrame
    df['transcript'] = df['file_name'].apply(lambda x: transcripts.get(os.path.splitext(x)[0], ''))
    
    return df

def convert_cad_to_int(df):
    '''
    Convert CAD column to integer, replacing non-numeric or missing values with NaN.
    '''
    df['CAD'] = pd.to_numeric(df['CAD'], errors='coerce').astype('Int64')
    return df

def main():
    '''
    Add transcripts to your linked CAD df with metadata
    '''
    if len(sys.argv) != 4:
        sys.exit(1)

    # Get the command-line arguments
    video_metadata_csv_path = sys.argv[1]
    transcript_dir = sys.argv[2]
    output_csv_path = sys.argv[3]
    
    # Read the video metadata into a DataFrame
    df = pd.read_csv(video_metadata_csv_path)
    
    # Add transcripts to the DataFrame
    df_with_transcripts = add_transcripts_to_dataframe(df, transcript_dir)
    
    # Convert CAD column to integer
    df_with_transcripts = convert_cad_to_int(df_with_transcripts)
    
    # Save the updated DataFrame to a new CSV file
    df_with_transcripts.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    main()
