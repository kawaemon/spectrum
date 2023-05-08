#!/bin/sh

ffmpeg -i "$1" -vn -c:a pcm_f32le out.wav
ffmpeg -i out.wav out.mp3
