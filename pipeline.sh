#!/bin/bash

# Function to prompt the user for input
prompt_for_input() {
    local prompt_message=$1
    read -p "$prompt_message: " input
    echo $input
}

# Prompt user for necessary directories and files
VIDEOS_DIR=$(prompt_for_input "Enter the directory path containing bodycam videos")
TRANSCRIPTS_DIR=$(prompt_for_input "Enter the directory path to save transcripts into")
CAD_FILE=$(prompt_for_input "Enter the file path for the CAD file")
# RIPA_FILE=$(prompt_for_input "Enter the file path for the RIPA file") # Commented out for now

# Define output files
TIMESTAMPS_FILE="./data/timestamps.csv"
LINKED_VIDEOS_DIR="./data"
# LINKED_RIPA_FILE="./data/linked_ripa.csv" # Commented out for now
LINKED_VIDEOS_TRANSCRIBED_FILE="./data/linked_videos_transcribed.csv"
LINKED_VIDEOS_FILE="$LINKED_VIDEOS_DIR/linked_videos.csv"

# Check if directories exist
if [ ! -d "$VIDEOS_DIR" ]; then
    echo "Error: Directory $VIDEOS_DIR does not exist."
    exit 1
fi

if [ ! -d "$TRANSCRIPTS_DIR" ]; then
    echo "Error: Directory $TRANSCRIPTS_DIR does not exist."
    exit 1
fi

# Step 1: Transcribe videos
echo ""
echo "Transcribing videos..."
python3 transcribe_videos.py "$VIDEOS_DIR" "$TRANSCRIPTS_DIR"

# Step 2: Extract timestamps from videos
echo ""
echo "Extracting timestamps from videos..."
python3 extract_timestamps.py "$VIDEOS_DIR" "$TIMESTAMPS_FILE"

# Step 3: Link videos to CAD using timestamps
echo ""
echo "Linking videos to CAD..."
python3 link_videos_to_CAD.py "$CAD_FILE" "$TIMESTAMPS_FILE" "$LINKED_VIDEOS_DIR"

# Step 4: Link RIPA to CAD (Commented out for now)
# echo "Linking RIPA to CAD..."
# python3 link_ripa_to_cad.py $RIPA_FILE $CAD_FILE $LINKED_RIPA_FILE

# Step 5: Combine linked videos and transcripts
echo ""
echo "Combining linked videos with transcripts..."
python3 transcribe_linked_videos.py "$LINKED_VIDEOS_FILE" "$TRANSCRIPTS_DIR" "$LINKED_VIDEOS_TRANSCRIBED_FILE"

echo ""
echo "Pipeline completed."
echo "Final outputs:"
echo "1. Linked videos and transcripts: $LINKED_VIDEOS_TRANSCRIBED_FILE"
# echo "2. Linked RIPA: $LINKED_RIPA_FILE" # Commented out for now
