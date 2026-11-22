#!/usr/bin/env bash
# Turn a clip that does not loop cleanly into a seamless loop by crossfading
# its last FADE seconds into its first FADE seconds. Output is DURATION-FADE long.
# Usage: ./make_loop.sh input.mp4 output.mp4 [fade_seconds]
set -euo pipefail
in=$1; out=$2; fade=${3:-2}
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$in")
tail=$(python3 -c "print($dur-$fade)")
ffmpeg -y -v error -i "$in" -filter_complex "\
[0:v]split=3[a][b][c];\
[a]trim=0:$fade,setpts=PTS-STARTPTS[A];\
[b]trim=$fade:$tail,setpts=PTS-STARTPTS[B];\
[c]trim=$tail:$dur,setpts=PTS-STARTPTS[C];\
[C][A]xfade=transition=fade:duration=$fade:offset=0[X];\
[X][B]concat=n=2:v=1:a=0[v]" \
  -map "[v]" -an -c:v libx264 -preset medium -crf 10 -pix_fmt yuv420p "$out"
