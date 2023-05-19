import whisper
import time
import os
import math
import csv
import sys
import subprocess


def saveFile(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def convert_to_txt(file_path):
    try:
        t0 = time.time()
        result = model.transcribe(file_path, verbose=True)
        saveFile(os.path.splitext(file_path)[0] + '.txt',result["text"])
        t1 = time.time()
        total = t1-t0
        print('Time:' + str(total))

        with open(os.path.splitext(file_path)[0] + '.tsv', 'w', encoding='utf-8', newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            writer.writerow(['start', 'end', 'text'])  # header row
            for row in result["segments"]:
                start = math.floor(row["start"])
                end = math.ceil(row["end"])
                text = row["text"]
                writer.writerow([start, end, text])
                print(f"[{start} =-=> {end} {text}")

    except Exception as e:
        # Code to handle the exception
        print("An error occurred:", str(e))
    
# MAIN PROGRAM
if __name__ == '__main__':

    """
    models = [ 'base', 'small', 'medium', 'large', 'tiny.en', 'base.en', 'small.en', 'medium.en', 'tiny']

    for model in models:
        print('Downloading %s' % (model))
        model = whisper.load_model(model,"cpu","models",False)
    """

    model = whisper.load_model("tiny.en","cpu","models",True)

    video_link = sys.argv[1]

    output = subprocess.check_output(["youtube-dl", "-f", "140", "-o","\"%(title)s.%(ext)s\"", "--get-filename", video_link])
    print(f"output->{output.decode('utf-8')}")

    process = os.popen(f"youtube-dl -f 140 -ciw -o \"%(title)s.%(ext)s\" --get-filename {video_link}")
    output_str = process.read()
    process.close()
    print(f"output->{output_str}")

    #print(f"output->{result}")

    #convert_to_txt(audio_filename)



