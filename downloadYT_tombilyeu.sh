#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 [channel_name]"
    exit 1
fi

channel_name="$1"

# Calculate yesterday's date
before_date=$(date -v -3d +%Y%m%d)

if [ ! -d "$channel_name" ]; then
    mkdir ~/Documents/youtube-audio/${channel_name}/
fi

cd ~/Documents/youtube-audio/${channel_name}/

youtube-dl --datebefore "$before_date" --download-archive downloaded.tracker -f 140 -ciw -o "%(title)s.%(ext)s" -v https://www.youtube.com/@${channel_name}

cd ..
