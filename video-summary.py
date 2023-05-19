from typing import Optional, List, Tuple
import random
from enum import Enum
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

VIDEO_ROOT_FOLDER = "../../obsidian/media/"

class OpenAIModel(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_32k = "gpt-4-32k"

def saveFile(filepath, content) -> None:
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def readFile(filepath) -> str:
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def split_text_into_chunks(text: str, chunksize: Optional[int]=500) -> List[str]:
    # Split input string into sentences based on punctuation marks
    sentences = re.split('(?<=[.!?]) +', text)
    # Initialize variables
    text_chunks = []
    current_chunk = ""
    # Iterate through sentences to create text chunks
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunksize:
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

def convert_to_txt(video_file) -> str:
    retval = None
    try:
        txt_file = os.path.splitext(video_file)[0] + '.txt'
        tsv_file = os.path.splitext(video_file)[0] + '.tsv'

        start_time = time.time()  # Get the current time before executing the function
        print(f"Whisper starting...")
        result = model.transcribe(video_file, verbose=True)
        print(f"Whisper stop...")
        end_time = time.time()  # Get the current time after executing the function
        execution_time = end_time - start_time  # Calculate the time difference
        print(f"The example_function took {execution_time:.2f} seconds to execute.")

        retval = result["text"]
        saveFile(txt_file, retval)
        update_file_created_modifled(video_file, txt_file)

        with open(tsv_file, 'w', encoding='utf-8', newline='') as tsv_output_file:
            writer = csv.writer(tsv_output_file, delimiter='\t')
            writer.writerow(['start', 'end', 'text'])  # header row
            for row in result["segments"]:
                start = math.floor(row["start"])
                end = math.ceil(row["end"])
                text = row["text"]
                writer.writerow([start, end, text])
                print(f"[{start} ####> {end} {text}")
        update_file_created_modifled(video_file, tsv_file)
    except Exception as e:
        # Code to handle the exception
        print("An error occurred:", str(e))
    return retval

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

def openai_chat_call(prompt: str, llmmodel: Optional[str]=OpenAIModel.GPT_3_5_TURBO.value) -> str:
    print(f"_openai_chat_call===============================================")
    model_tokens = 0
    prompt_cost_per_thousand = 0
    completion_cost_per_thousand = 0
    if llmmodel == OpenAIModel.GPT_3_5_TURBO.value:
        model_tokens = 4096
        prompt_cost_per_thousand = 0.002
        completion_cost_per_thousand = 0.002
    if llmmodel == OpenAIModel.GPT_4.value:
        model_tokens = 8192
        prompt_cost_per_thousand = 0.03
        completion_cost_per_thousand = 0.06
    if llmmodel == OpenAIModel.GPT_4_32k.value:
        model_tokens = 32768
        prompt_cost_per_thousand = 0.06
        completion_cost_per_thousand = 0.12

    response = None
    messages = [{"role": "system", "content": prompt}]     
    json_string = json.dumps(messages)
    prompt_tokens = int(len(json_string) / 3.8)
    max_tokens = int(model_tokens - prompt_tokens)
    if max_tokens < 0:
        print(f"Max tokens is less than 0. Model: {llmmodel} Max Tokens: {max_tokens}")
        quit()
    else:
        print(f"OpenAI Model: {llmmodel} Prompt Tokens: {prompt_tokens} Response Tokens: {max_tokens} Total Tokens: {prompt_tokens + max_tokens}")
    try:
        start_time = time.time()
        print(f"OpenAI start")
        result = openai.ChatCompletion.create(
            model=llmmodel,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.25,
            presence_penalty=0.25)
        print(f"prompt_length: {len(prompt)}")
        print(f"prompt_tokens: {result['usage']['prompt_tokens']}")
        print(f"completion_tokens: {result['usage']['completion_tokens']}")
        print(f"total_tokens: {result['usage']['total_tokens']}")
        prompt_cost = round(float((int(result['usage']['prompt_tokens'])/1000) * prompt_cost_per_thousand),6)
        completion_cost = round(float((int(result['usage']['completion_tokens'])/1000) * completion_cost_per_thousand),6)
        print(f"cost: ${round(prompt_cost+completion_cost,6)}")
        print(f"OpenAI duration: {(time.time() - start_time) * 60000} s")
        text = result['choices'][0]['message']['content']
        print(f"prompt: {prompt}")
        print(f"result: {text}")
        response = text
    except Exception as oops:
        print('OpenAI Error:', oops)
    return response

# MAIN PROGRAM
if __name__ == '__main__':

    """
    # If you want to download all the whisper models locally

    models = [ 'base', 'small', 'medium', 'large', 'tiny.en', 'base.en', 'small.en', 'medium.en', 'tiny']

    for model in models:
        print('Downloading %s' % (model))
        model = whisper.load_model(model,"cpu","models",False)
    """ 

    print("loading whisper speech to text...")
    # load whisper model
    model = whisper.load_model("tiny.en", "cpu", "models", True)

    # for each video mp4 file convert to txt and tsv file using whisper
    video_files = walk_folder_for_extension(VIDEO_ROOT_FOLDER, ".m4a")
    print(f"Total video files found {len(video_files)}")
    current_file_number = 0
    for video_file_with_path in video_files:
        if "AIdelimma.m4a" in video_file_with_path:
            current_file_number += 1
            if os.path.isfile(video_file_with_path):
                if os.path.isfile(os.path.splitext(video_file_with_path)[0] + '.tsv'):
                    print(f"skipping file {current_file_number} of {len(video_files)} {video_file_with_path}")
                else:
                    print(f"working file  {current_file_number} of {len(video_files)} {video_file_with_path}")
                    transcript = convert_to_txt(video_file_with_path)
                    #transcript = readFile(os.path.splitext(video_file_with_path)[0] + '.txt')
                    if transcript is not None:
                        summary = ""
                        for chunk in split_text_into_chunks(transcript, 7500):
                            summary = summary + openai_chat_call(f"write a comprehensive summary of these meeting notes:\n{chunk}")
                        saveFile(os.path.splitext(video_file_with_path)[0] + '-summary.txt', summary)
                        summary = openai_chat_call(f"write a comprehensive summary of these meeting notes as bullet points:\n{summary}")
                        saveFile(os.path.splitext(video_file_with_path)[0] + '-bullets.txt', summary)
            else:
                print(f"skipping file {current_file_number} of {len(video_files)} {video_file_with_path}")



