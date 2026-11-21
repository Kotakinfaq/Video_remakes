# Video remakes

## `lava_loop.py`: procedural molten-lava background

An original, fully procedural lava/fire-wave motion background. It is built
from noise, with no source footage. The animation is exactly periodic, so the
clip repeats with no visible seam.

Requirements: Python 3, `numpy`, `numba`, `ffmpeg`.

Render a 60 s 4K loop:

```bash
python3 lava_loop.py --width 3840 --height 2160 --seconds 60 \
  | ffmpeg -f rawvideo -pix_fmt rgb24 -s 3840x2160 -r 30000/1001 -i - \
      -c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p \
      -x264-params "keyint=60:min-keyint=60:scenecut=0:open-gop=0" loop60.mp4
```

Extend to 6 minutes without re-encoding:

```bash
ffmpeg -stream_loop 5 -i loop60.mp4 -c copy -movflags +faststart lava_6min.mp4
```

Look knobs (colour weights, streak size, relief) are constants at the top of the script.
