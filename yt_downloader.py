import os
import subprocess
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
from typing import List, Optional
import json
import argparse
import youtube_dl
import requests
from PIL import Image
from io import BytesIO
import openai
import re
from typing import List
from enum import Enum

class ModelType(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_32k = "gpt-4-32k"

class YouTubeProcessor:
    def __init__(self, whisper_model_path: str = "tiny.en"):
        self.model = whisper.load_model(whisper_model_path, "cpu", "models", True)

    def call_openai(self,prompt: str, model: ModelType) -> str:
        messages = [
            {"role": "system", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model=model.value,
            messages=messages,
            temperature=0.49,
            max_tokens=1280,
            top_p=1,
            frequency_penalty=0.25,
            presence_penalty=0.25)
        openai_response=response['choices'][0]['message']['content'].strip()
        print (openai_response)
        return openai_response

    def summarize_text(self, text: str, video_id: str) -> None:
        tokens_for_model = 4096
        tokens_for_response = 1280
        token_size = 2.7
        chunk_size = int((tokens_for_model - tokens_for_response) * token_size)

        # Function to chunk text into complete sentences
        def chunk_text(text: str, chunk_size: int) -> list[str]:
            words = text.split()
            chunks = []
            current_chunk = []

            for word in words:
                if len(" ".join(current_chunk) + " " + word) <= chunk_size:
                    current_chunk.append(word)
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        chunks = chunk_text(text, chunk_size)
        bullet_points = []

        for chunk in chunks:
            prompt = f"summarize this text into bullet points: \n{chunk}"
            openai_response=self.call_openai(prompt, model=ModelType.GPT_3_5_TURBO)
            bullet_points.append(openai_response)

        aggregated_bullet_points = '\n'.join(bullet_points)

        with open(f"{video_id}_bulletpoints.txt", "w") as file:
            file.write(aggregated_bullet_points)
        
        prompt = f"write an article that satisfies these requirements: \n\
                - has a clever title \n\
                - use colorful language \n\
                - make it interesting and engaging \n\
                - do not mention, quote, or reference any specific individuals by name.\n\
                - will be used to gain notoriety among my peers without stating such or asking for it. \n\
                - any article points must include all relevant details from the bullet points to create a well rounded point with full context. \n\
                \n\
                Here are some bullet points which you can choose to cover in the article: \n\
                {aggregated_bullet_points}"
        openai_response=self.call_openai(prompt, model=ModelType.GPT_4)

        with open(f"{video_id}_summary.txt", "w") as file:
            file.write(openai_response)

    def download_image_as_jpg(self,url: str, output_file: str) -> None:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.save(output_file, "JPEG")
            print(f"Image saved as {output_file}")
        except Exception as e:
            print(f"Could not download image. Error: {e}")
    
    def download_transcript(self, video_id: str) -> tuple[str,bool]:
        text = ''
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            file_name = f"{video_id}_transcript.txt"
            text = ''
            with open(file_name, "w") as f:
                for t in transcript:
                    #f.write(f"{t['start']} {t['duration']} {t['text']} ")
                    f.write(f"{t['text']} ")
                    text += f"{t['text']} "
            return text, True
        except:
            print(f"Could not download transcript for {video_id}.")
        return text, False

    def download_metadata(self, video_id: str) -> tuple[str,bool]:
        try:
            yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
            metadata = {
                "Title": yt.title,
                "Author": yt.author,
                "Channel ID": yt.channel_id,
                "Views": yt.views,
                "Rating": yt.rating,
                "Length": yt.length,
                "Description": yt.description,
                "Published": str(yt.publish_date),
                "Keywords": yt.keywords,
                "Age Restricted": yt.age_restricted,
                "Video ID": yt.video_id,
                "Thumbnail URL": yt.thumbnail_url,
                "Embed URL": yt.embed_url
            }
            file_name = f"{video_id}_metadata.json"
            with open(file_name, "w") as f:
                json.dump(metadata, f, indent=4)

            self.download_image_as_jpg(yt.thumbnail_url, f"{video_id}_thumbnail.jpg")
            return metadata, True
        except:
            print(f"Could not download metadata for {video_id}.")
        return None, False

    def download_audio(self, video_id: str) -> Optional[str]:
        try:
            print(f"Downloading audio for video {video_id}...")
            url = f'https://www.youtube.com/watch?v={video_id}'
            output_file = f"{video_id}.m4a"
            command = f"youtube-dl -f 140 -o {output_file} {url}"
            subprocess.run(command, shell=True, check=True)
            return output_file
        except subprocess.CalledProcessError:
            print(f"Could not download audio for video {video_id}.")
        return None

    def transcribe_m4a_to_text(self, m4a_file_path: str) -> tuple[str,bool]:
        try:
            result = self.model.transcribe(m4a_file_path, verbose=True)
            with open(os.path.splitext(m4a_file_path)[0] + '.txt', 'w', encoding='utf-8') as outfile:
                outfile.write(result["text"])
            return result["text"], True
        except:
            print("Could not transcribe m4a to text.")
        return None, False

    def process_video(self, video_id: str) -> None:

        transcript=''
        """Attempt to download transcript with youtube_transcript_api """
        transcript, transcript_downloaded = self.download_transcript(video_id)
        if not transcript_downloaded:
            """Attempt raw audio to text conversion using whisper """
            try:
                m4a_file = self.download_audio(video_id)
                if m4a_file:
                    print("Transcribing MP3 to text...")
                    transcript, transcript_downloaded = self.transcribe_m4a_to_text(m4a_file)
                    print("Transcription done!")
            except:
                print(f"Could not download and convert video {video_id}.")

        metadata, metadata_downloaded = self.download_metadata(video_id)
        if metadata_downloaded:
            show_notes = f"Title:  {metadata['Title']}\n"
            show_notes += f"Description: {metadata['Description']}\n"
            show_notes += f"Transcript: {transcript}"

        self.summarize_text(show_notes, video_id)

    def read_video_ids_from_file(self, file_path: str) -> List[str]:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
        
    """
    def _download_metadata_youtube_dl(self, video_id: str, language: Optional[str] = "en") -> bool:
        transcript_downloaded = False
        ydl_opts = {
            "skip_download": True,
            "write-info-json": True,
            "outtmpl": f"{video_id}_metadata.json",
        }
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                transcript_downloaded = True
        except Exception as e:
            print(f"Could not download transcript for {video_id}. Error: {e}")
        return transcript_downloaded

    def _download_transcript_youtube_dl(self, video_id: str, language: Optional[str] = "en") -> bool:
        transcript_downloaded = False
        ydl_opts = {
            "skip_download": True,
            "write_auto_sub": True,
            "sub_lang": language,
            "subtitlesformat": "srt",
            "outtmpl": f"{video_id}_transcript.txt",
        }
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                transcript_downloaded = True
        except Exception as e:
            print(f"Could not download transcript for {video_id}. Error: {e}")
        return transcript_downloaded
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YouTube videos.")
    parser.add_argument("--file", help="Path to the text file containing YouTube video IDs.", required=False)
    args = parser.parse_args()

    youtube_processor = YouTubeProcessor()

    if args.file:
        file_path = args.file
    else:
        file_path = input("Enter the path to the text file containing YouTube video IDs:")

    video_ids = youtube_processor.read_video_ids_from_file(file_path)
    for video_id in video_ids:
        youtube_processor.process_video(video_id)
