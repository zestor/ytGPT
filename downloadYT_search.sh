#!/bin/bash

#youtube-dl --dateafter now-1months --reject-title "(earn|how to make|\$)" --download-archive search.tracker -f 140 -ciw -o "%(channel)s/%(title)s.%(ext)s" --write-description --write-info-json --write-auto-sub --sub-format "srt" --sub-lang "en" "ytsearch:$1"

cd ~/Documents/youtube-audio-new/

youtube-dl --dateafter now-7days --download-archive search.tracker -f 140  -o "new/%(channel)s/%(title)s.%(ext)s" --write-description --write-info-json --write-auto-sub --sub-format "vtt" --sub-lang "en" "ytsearch20:chatgpt"

youtube-dl --dateafter now-7days --download-archive search.tracker -f 140  -o "new/%(channel)s/%(title)s.%(ext)s" --write-description --write-info-json --write-auto-sub --sub-format "vtt" --sub-lang "en" "ytsearch20:generative ai"

youtube-dl --dateafter now-7days --download-archive search.tracker -f 140  -o "new/%(channel)s/%(title)s.%(ext)s" --write-description --write-info-json --write-auto-sub --sub-format "vtt" --sub-lang "en" "ytsearch20:gpt4 gpt5"

youtube-dl --dateafter now-7days --download-archive search.tracker -f 140  -o "new/%(channel)s/%(title)s.%(ext)s" --write-description --write-info-json --write-auto-sub --sub-format "vtt" --sub-lang "en" "ytsearch20:llama alpaca dalai gpt4all"