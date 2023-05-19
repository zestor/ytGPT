import csv
import datetime
import json
import math
import numpy as np
import os
import re
import tensorflow_hub as hub
import time
import sys
import whisper
import shutil
import openai
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from vectoro_client import vectoro_client

AUDIO_ROOT_FOLDER = "../../youtube-audio/"
KNOWLEDGE_FOLDER = "./knowledge/"

def saveFile(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def convert_to_txt(audio_file):
    try:
        txt_file = os.path.splitext(audio_file)[0] + '.txt'
        tsv_file = os.path.splitext(audio_file)[0] + '.tsv'

        result = model.transcribe(audio_file, verbose=True)
        saveFile(txt_file, result["text"])
        update_file_created_modifled(audio_file, txt_file)

        with open(tsv_file, 'w', encoding='utf-8', newline='') as tsv_output_file:
            writer = csv.writer(tsv_output_file, delimiter='\t')
            writer.writerow(['start', 'end', 'text'])  # header row
            for row in result["segments"]:
                start = math.floor(row["start"])
                end = math.ceil(row["end"])
                text = row["text"]
                writer.writerow([start, end, text])
                print(f"[{start} ####> {end} {text}")
        update_file_created_modifled(audio_file, tsv_file)

    except Exception as e:
        # Code to handle the exception
        print("An error occurred:", str(e))

def split_text_into_chunks(text):
    # Split input string into sentences based on punctuation marks
    sentences = re.split('(?<=[.!?]) +', text)
    # Initialize variables
    text_chunks = []
    current_chunk = ""
    # Iterate through sentences to create text chunks
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:
            # If adding the sentence won't exceed 500 characters, append it to the current chunk
            current_chunk += sentence
        else:
            # If adding the sentence would exceed 500 characters, save the current chunk and start a new one
            text_chunks.append(current_chunk.strip())
            current_chunk = sentence
    # Append the last chunk if it's not empty
    if current_chunk.strip():
        text_chunks.append(current_chunk.strip())
    return text_chunks

def save_chunk(txt_file_with_path, chunk_count, payload):
    # create some filename variables
    dir_path, txt_file_wo_path = os.path.split(txt_file_with_path)
    youtube_channel = os.path.split(dir_path)[1]
    payload["source2"]=youtube_channel
    audio_file_with_path = os.path.join(dir_path, os.path.splitext(txt_file_wo_path)[0] + '.m4a')
    txt_file_wo_path_or_ext = os.path.splitext(txt_file_wo_path)[0]
    # put these chunks in a folder with same name as audio_file without m4a extension
    chunk_path = KNOWLEDGE_FOLDER + f"youtube/{youtube_channel}/"
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)
    chunk_path = KNOWLEDGE_FOLDER + f"youtube/{youtube_channel}/{txt_file_wo_path_or_ext}/"
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)
    # create the chunk file
    filename = f"{txt_file_wo_path_or_ext}-{chunk_count}.json"
    print(".",end="")
    text_chunk_filename = chunk_path + filename
    if not os.path.exists(text_chunk_filename):
        with open(text_chunk_filename, 'w', encoding='utf-8') as outfile:
            json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=1)
        # change the file attribs to match the audio file created, modified datetimes
        update_file_created_modifled(audio_file_with_path, text_chunk_filename)

def convert_to_chunks(embed, txt_file_with_path, source):
    with open(txt_file_with_path, 'r', encoding='utf-8') as file:
        content = file.read()
        chunk_count = 0
        chunks = split_text_into_chunks(content)

        # Define a worker function to save the chunks
        def save_chunk_worker(chunk, count):
            vectors = embed([chunk]).numpy().tolist()
            chunk_vector = vectors[0]
            save_chunk(txt_file_with_path, count, {'date': get_file_date_string(txt_file_with_path), 'request': chunk, 'vector': chunk_vector, 'response': chunk, 'chunk': count, 'source': source, 'filename': os.path.basename(txt_file_with_path), 'folder': os.path.basename(os.path.dirname(txt_file_with_path))})

        # Use ThreadPoolExecutor to run the save_chunk_worker function with multiple threads
        with ThreadPoolExecutor(max_workers=20) as executor:
            for index, chunk in enumerate(chunks):
                chunk_count = index + 1
                executor.submit(save_chunk_worker, chunk, chunk_count)

def walk_folder_for_extension(directory, extension):
    found_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
                if file.endswith(extension):
                        found_files.append(os.path.join(root, file))
    return found_files

def get_file_date_string(source) -> str:
    date_string = ""
    if os.path.isfile(source):
        source_stat = os.stat(source)
        date_object = datetime.fromtimestamp(source_stat.st_mtime)
        date_string = date_object.strftime("%Y-%m-%d")
    return date_string

def update_file_created_modifled(source, target):
        if os.path.isfile(source):
            if os.path.isfile(target):
                source_stat = os.stat(source)
                os.utime(target, (source_stat.st_ctime, source_stat.st_mtime))

# MAIN PROGRAM
if __name__ == '__main__':

    """
    # If you want to download all the whisper models locally

    models = [ 'base', 'small', 'medium', 'large', 'tiny.en', 'base.en', 'small.en', 'medium.en', 'tiny']

    for model in models:
        print('Downloading %s' % (model))
        model = whisper.load_model(model,"cpu","models",False)
    """

    # update all existing txt and tsv files to have same modified date as m4a audio
    audio_files = walk_folder_for_extension(AUDIO_ROOT_FOLDER, ".m4a")
    print(f"total audio files found {len(audio_files)}")
    current_file_number = 0
    for m4a_file in audio_files:
        current_file_number += 1

        dir_path, file_name = os.path.split(m4a_file)
        txt_file = os.path.join(dir_path, os.path.splitext(file_name)[0] + '.txt')
        tsv_file = os.path.join(dir_path, os.path.splitext(file_name)[0] + '.tsv')

        update_file_created_modifled(m4a_file, txt_file)      
        update_file_created_modifled(m4a_file, tsv_file)   

    print("loading whisper speech to text...")
    # load whisper model
    model = whisper.load_model("tiny.en", "cpu", "models", True)

    # for each audio m4a file convert to txt and tsv file using whisper
    audio_files = walk_folder_for_extension(AUDIO_ROOT_FOLDER, ".m4a")
    print(f"Total audio files found {len(audio_files)}")
    current_file_number = 0
    for audio_file_with_path in audio_files:
        current_file_number += 1
        if os.path.isfile(audio_file_with_path):
            if os.path.isfile(os.path.splitext(audio_file_with_path)[0] + '.tsv'):
                print(f"skipping file {current_file_number} of {len(audio_files)} {audio_file_with_path}")
            else:
                print(f"working file  {current_file_number} of {len(audio_files)} {audio_file_with_path}")
                convert_to_txt(audio_file_with_path)
        else:
            print(f"skipping file {current_file_number} of {len(audio_files)} {audio_file_with_path}")

    print("loading google universal sentence encoder...")
    # Google Universal Sentence Encode v5
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    """
    start = time.time()
    # for each txt file save to chunks with embeddings for semantic search
    text_files = walk_folder_for_extension(AUDIO_ROOT_FOLDER, ".txt")
    print(f"Total text files found {len(text_files)}")
    current_file_number = 0
    for txt_file_with_path in text_files:
        current_file_number += 1
        print(f"working text file  {current_file_number} of {len(text_files)} {txt_file_with_path}",end="") 
        convert_to_chunks(embed, txt_file_with_path, source="youtube")
        print("")
    end = time.time()
    print(f"Saving Chunks Elapsed time: {end - start:.2f} seconds")
    """

    # Define a worker function to save the chunks
    def txt_process_worker(vectoro_client,current_file_number,text_files_count,embed,txt_file_with_path, source):
        print(f"working text file  {current_file_number} of {text_files_count} {txt_file_with_path}",end="") 
        with open(txt_file_with_path, 'r', encoding='utf-8') as file:
            content = file.read()
        chunks = split_text_into_chunks(content)
        chunk_count = 0
        for index, chunk in enumerate(chunks):
            chunk_count = index + 1
            vectors = embed([chunk]).numpy().tolist()
            chunk_vector = vectors[0]
            dir_path, txt_file_wo_path = os.path.split(txt_file_with_path)
            youtube_channel = os.path.split(dir_path)[1]
            video_name = os.path.splitext(txt_file_wo_path)[0]
            vectoro_client.add_vector(chunk_vector, 
                                        chunk, 
                                        get_file_date_string(txt_file_with_path),
                                        source,
                                        youtube_channel,
                                        video_name
                                      )

    start = time.time()
    # for each txt file save to chunks with embeddings for semantic search
    text_files = walk_folder_for_extension(AUDIO_ROOT_FOLDER, ".txt")
    print(f"Total text files found {len(text_files)}")
    current_file_number = 0
    text_files_count = len(text_files)
    vectoro_client = vectoro_client()
    # Use ThreadPoolExecutor to run the files function with multiple threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index, txt_file_with_path in enumerate(text_files):
            if index % 100 == 0:
                print("Waiting for threads to finish...")
                time.sleep(.5)
            current_file_number = index + 1
            executor.submit(txt_process_worker, vectoro_client, current_file_number,text_files_count,embed,txt_file_with_path, source="youtube")
    vectoro_client.save()
    end = time.time()
    print(f"Saving Chunks Elapsed time: {end - start:.2f} seconds")
