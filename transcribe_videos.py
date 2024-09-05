import os
import assemblyai as aai
from tqdm import tqdm
import sys

### KEY FUNCTIONS ###
def transcribe_with_timestamps(file_paths):
    '''
    Utilize AssemblyAI API to generate a transcription from a video file
    '''
    aai.settings.api_key = "" # SET YOUR API KEY

    results = {}

    transcriber = aai.Transcriber()

    for file_path in file_paths:
        if os.path.isfile(file_path):
            transcript = transcriber.transcribe(file_path)

            if transcript.status == aai.TranscriptStatus.error:
                results[file_path] = transcript.error
            else:
                words = transcript.words
                segments = []
                for i in range(0, len(words), 10):
                    start_time = round(words[i].start / 1000)
                    end_time = round(words[i+9].end / 1000) if i+9 < len(words) else round(words[-1].end / 1000)
                    text_segment = ' '.join([word_info.text for word_info in words[i:i+10]])
                    segments.append(f"{start_time} - {end_time}\n{text_segment}\n")
                results[file_path] = '\n'.join(segments)
        else:
            print(f"Skipping non-file: {file_path}")
    
    return results

def list_audio_files(directory):
    '''
    Function to create a list of all audio filepaths in a given directory
    '''
    audio_extensions = {'.mp3', '.wav', '.mp4', '.m4a'}
    audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in audio_extensions]
    return audio_files

def filter_untranscribed_files(audio_files, transcripts_dir):
    '''
    Function to check whether or not an audio file has already been transcribed.
    To minimize API calls, this function will only transcribe new videos that do 
    not yet have associated .txt files.
    '''
    untranscribed_files = []
    for audio_file in audio_files:
        transcript_file = os.path.join(transcripts_dir, os.path.splitext(os.path.basename(audio_file))[0] + '.txt')
        if not os.path.exists(transcript_file):
            untranscribed_files.append(audio_file)
    return untranscribed_files

### MAIN PROCESS ###
def main():
    if len(sys.argv) != 3:
        print(f'ERROR: {len(sys.argv)} ARGUMENTS')
        sys.exit(1)

    # Get the command-line arguments
    audio_directory = sys.argv[1]
    transcripts_directory = sys.argv[2]

    # Validate the provided paths
    if not os.path.isdir(audio_directory):
        print(f"Error: The path '{audio_directory}' is not a valid directory.")
        return

    # Ensure the 'Transcripts' directory exists
    if not os.path.exists(transcripts_directory):
        os.makedirs(transcripts_directory)

    # Create a list of audio files in the given directory
    audio_files = list_audio_files(audio_directory)

    # Filter the list to only include audio files that do not have corresponding .txt files in the 'Transcripts' subdirectory
    untranscribed_files = filter_untranscribed_files(audio_files, transcripts_directory)

    # Print out the list of untranscribed files for debugging
    print(f"Found {len(untranscribed_files)} untranscribed files: {untranscribed_files}")
    
    if not untranscribed_files:
        print("No new files to transcribe.")
        return

    # Transcribe the filtered list of audio files with progress bar
    print("Starting transcription...")
    transcribed_files = []
    for file_path in tqdm(untranscribed_files, desc="Transcribing files"):
        transcription_results = transcribe_with_timestamps([file_path])

        # Save the transcriptions to .txt files in the 'Transcripts' subdirectory
        for file_path, transcription in transcription_results.items():
            transcript_file_path = os.path.join(transcripts_directory, os.path.splitext(os.path.basename(file_path))[0] + '.txt')
            with open(transcript_file_path, 'w') as f:
                f.write(transcription)
            transcribed_files.append(file_path)

    print("Transcription completed.")
    print("Files transcribed:")
    for file in transcribed_files:
        print(file)

if __name__ == "__main__":
    main()
