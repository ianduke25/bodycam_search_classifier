import sys
import cv2
import pytesseract
import pandas as pd
from pytesseract import Output
import re
import os
import ffmpeg
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_fps(video_file_path):
    '''
    Function to extract frames per second of each video.
    '''
    video = cv2.VideoCapture(video_file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return round(fps)

def extract_frame(video_path, frame_number):
    '''
    Function to extract frames of the video as an image to perform OCR on.
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

def ocr_from_image(image, type):
    '''
    Function to edit images to be more amenable to OCR and perform OCR
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1,1), np.uint8)
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(processed_image, -1, sharpen_kernel)
    h, w = sharpened_image.shape
    if type == 'bwc':
        cropped_image = sharpened_image[:int(h * .1), int(w * 0.35):]
    elif type == 'bwc2':
        cropped_image = sharpened_image[int(h*.9):, int(w * 0.65):]
    else:
        cropped_image = sharpened_image
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(cropped_image, config=custom_config)
    return text

def extract_timestamp_from_text(text):
    '''
    Function extracting desired regular expressions from text acquired via OCR
    '''
    pattern = r'[0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{2}/[0-9]{2}/[0-9]{4}'
    match = re.search(pattern, text)
    if match is None:
        pattern = r'[0-9]{2}/[0-9]{2}/[0-9]{4}.{1}[0-9]{2}:[0-9]{2}:[0-9]{2}'
        match = re.search(pattern, text) 
    if match is None:
        pattern = r'[0-9]{4}-[0-9]{2}-[0-9]{2}.{1}[0-9]{2}:[0-9]{2}:[0-9]{2}'
        match = re.search(pattern, text)
    if match:
        return match.group()
    return None

def get_video_metadata(file_path):
    '''
    Function to extract metadata from video file (video creation time and video duration)
    '''
    try:
        probe = ffmpeg.probe(file_path)
        probe_json = json.dumps(probe, indent=4)
        data = json.loads(probe_json)
        video_stream = next((stream for stream in data['streams'] if stream['codec_type'] == 'video'), None)
        creation_time = None
        if video_stream and 'tags' in video_stream:
            creation_time = video_stream['tags'].get('creation_time')
        duration = data['format'].get('duration')
        return creation_time, duration
    except ffmpeg.Error as e:
        print("Error:", e)
    except Exception as e:
        print("An error occurred:", e)

def format_text_for_timestamp(text):
    '''
    Function to clean up text from improperly OCR'd documents
    '''
    text = text.replace('-8', '-0')
    text = text.replace('/8', '/0')
    text = text.replace('282', '202')
    text = text.replace('002', '202')
    pattern = re.compile(r'\d{2}/\d{2}/\d{4}.{1}\d{2}:\d{2}:\d{2}')
    matches = pattern.findall(text)
    for match in matches:
        corrected = match.replace('-', ' ')
        text = text.replace(match, corrected)
    return text

def extract_timestamp(video_path):
    '''
    Main function to extract timestamps.
    This function filters for different video formats (for example, dash vs bwc2)
    and searches for timestamps accordingly.
    If a timestamp in one frame is not detected, the code selects the next frame one
    second away and adds or subtracts a second to the extracted timestamp.
    '''
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise Exception("Error: Could not open video file.")
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_iterations = 0
    end_iterations = 0
    start_timestamp = None
    end_timestamp = None
    iteration = 1
    last_iteration = frame_count - 1
    fps = get_fps(video_path)
    while start_timestamp is None or end_timestamp is None:
        if start_timestamp is None:
            frame_start = iteration
            screenshot_start = extract_frame(video_path, frame_start)
            if screenshot_start is not None:
                start_text = ocr_from_image(screenshot_start, type='dash')
                if 'MPH' in start_text:
                    start_text = format_text_for_timestamp(start_text)
                    start_timestamp = extract_timestamp_from_text(start_text)
                elif '_REDACTED_' in video_path:
                    start_text = ocr_from_image(screenshot_start, type='bwc2')
                    start_text = format_text_for_timestamp(start_text)
                    start_timestamp = extract_timestamp_from_text(start_text)
                else:
                    start_text = ocr_from_image(screenshot_start, type='bwc')
                    start_text = format_text_for_timestamp(start_text)
                    start_timestamp = extract_timestamp_from_text(start_text)
                if start_timestamp is None:
                    # If a timestamp is not extracted, move forward one second and iterate through while loop again
                    iteration += fps
                    start_iterations += 1
        if end_timestamp is None:
            frame_end = last_iteration
            screenshot_end = extract_frame(video_path, frame_end)
            if screenshot_end is not None:
                end_text = ocr_from_image(screenshot_end, type='dash')
                if '_REDACTED_' in video_path:
                    end_text = ocr_from_image(screenshot_end, type='bwc2')
                    end_text = format_text_for_timestamp(end_text)
                elif 'MPH' in end_text:
                    end_text = ocr_from_image(screenshot_end, type='dash')
                    end_text = format_text_for_timestamp(end_text)
                else:
                    end_text = ocr_from_image(screenshot_end, type='bwc')
                    end_text = format_text_for_timestamp(end_text)
                end_timestamp = extract_timestamp_from_text(end_text)
                if end_timestamp is None:
                    # If a timestamp is not extracted, move backward one second and iterate through while loop again
                    end_iterations += 1
                    last_iteration = last_iteration - fps
        last_iteration = int(last_iteration - fps)
        if iteration >= last_iteration:
            print("NO VIABLE FRAMES")
            break
    creation_time, duration = get_video_metadata(video_path)
    return start_timestamp, end_timestamp, creation_time, duration, start_iterations, end_iterations

def standardize_timestamp(timestamp):
    if timestamp is None:
        return None
    formats = [
        '%H:%M:%S %m/%d/%Y',
        '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S'
    ]
    for fmt in formats:
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    return None

def process_videos(directory, csv_path):
    '''
    Function to iterate through video files, extract timestamps, and format into a csv
    '''
    results = []
    video_files = []
    new_files = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(subdir, file))
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        processed_files = results_df['file_name'].tolist()
    else:
        results_df = pd.DataFrame(columns=['file_name', 'start_date', 'start_time', 'end_date', 'end_time', 'creation_time', 'duration'])
        processed_files = []
    new_files = [file for file in video_files if os.path.basename(file) not in processed_files]
    print(f"Found {len(new_files)} new videos: {new_files}")
    if not new_files:
        print('No new timestamps.')
        return
    with tqdm(total=len(new_files), desc="Processing Videos") as pbar:
        for file_path in new_files:
            start_timestamp, end_timestamp, creation_time, duration, start_iterations, end_iterations = extract_timestamp(file_path)
            standardized_start_timestamp = standardize_timestamp(start_timestamp)
            standardized_end_timestamp = standardize_timestamp(end_timestamp)
            if standardized_start_timestamp and start_iterations is not None:
                adjusted_start_timestamp = standardized_start_timestamp - timedelta(seconds=start_iterations)
                start_date, start_time = adjusted_start_timestamp.date(), adjusted_start_timestamp.time()
            else:
                start_date, start_time = None, None
            if standardized_end_timestamp and end_iterations is not None:
                adjusted_end_timestamp = standardized_end_timestamp + timedelta(seconds=end_iterations) + timedelta(seconds=end_iterations)
                end_date, end_time = adjusted_end_timestamp.date(), adjusted_end_timestamp.time()
            else:
                end_date, end_time = None, None
            results.append({'file_name': os.path.basename(file_path),
                            'start_date': start_date, 'start_time': start_time,
                            'end_date': end_date, 'end_time': end_time,
                            'creation_time': creation_time, 'duration': duration})
            results_df = pd.concat([results_df, pd.DataFrame(results)], ignore_index=True)
            results = []
            results_df.to_csv(csv_path, index=False)
            pbar.update(1)
    print(f"CSV file updated: {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    directory = sys.argv[1]
    csv_path = sys.argv[2]
    process_videos(directory, csv_path)
