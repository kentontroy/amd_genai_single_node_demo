#!/bin/bash

curl -X POST "http://127.0.0.1:7851/api/tts-generate" \
     -d "text_input=${1}" \
     -d "text_filtering=standard" \
     -d "character_voice_gen=${2}" \
     -d "narrator_enabled=false" \
     -d "narrator_voice_gen=male_01.wav" \
     -d "text_not_inside=character" \
     -d "language=en" \
     -d "output_file_name=myoutputfile" \
     -d "output_file_timestamp=true" \
     -d "autoplay=true" \
     -d "autoplay_volume=0.8"
