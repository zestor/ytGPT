#!/bin/bash

cd ~/Documents/youtube-audio-new/

#youtube-dl --dateafter now-3days --download-archive search.tracker -f 140  -o "new/%(channel)s/%(title)s.%(ext)s" --write-description --write-info-json --write-auto-sub --sub-format "vtt" --sub-lang "en" --playlist-items 10 -v https://www.youtube.com/@markets/videos

youtube-dl --download-archive search.tracker --write-info-json --skip-download -o "new/%(channel)s/%(title)s.%(ext)s" --playlist-items 100 -v https://www.youtube.com/@markets/videos

