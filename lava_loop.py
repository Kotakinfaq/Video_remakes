#!/usr/bin/env python3
"""Procedural, seamlessly looping molten-lava background.

Everything is generated from noise; no source footage is used.
Frames are written as raw RGB to stdout so they can be piped into ffmpeg.

    python3 lava_loop.py --width 3840 --height 2160 --fps 29.97 --seconds 60 \
        | ffmpeg -f rawvideo -pix_fmt rgb24 -s 3840x2160 -r 30000/1001 -i - ...

The animation is exactly periodic: frame N equals frame 0, so the clip can be
repeated end to end with no visible seam.
"""
import argparse
import math
import sys

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------- look knobs ----
WARP_Q = 0.6    # strength of second warp stage
WARP_B = 0.7    # how much the warp wobbles the clumps
FUR_ANG = 0.75  # strand angle (radians, rising to the right)
FUR_BEND = 0.25 # how much the strand direction wanders
FUR_LEN = 0.30  # how far clump edges are smeared into strands
HAIR_U = 5.0    # hair frequency along the strands (lower = longer hairs)
HAIR_V = 70.0   # hair frequency across the strands (higher = thinner)
CLUMP_U = 3.0   # clump frequency along the strands
CLUMP_V = 4.5   # clump frequency across the strands
SHADE_K = 70.0   # relief of strands/clumps
H_W = 0.95      # weight of clump height in colour
D_W = 0.28      # weight of lighting in colour
S_W = 0.12      # weight of hair brightness in colour
BIAS = 0.03
PERSP = 0.35
PERSP_Y = 3.2   # vertical world extent factor (keeps features round)    # 0 = flat top-down view, higher = more oblique
FOAM = 0.10     # fine crinkly detail on the clump tops

# ---------------------------------------------------------------- noise ----

GRAD = np.array(
    [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
     [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
     [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float32)


@njit(inline="always", cache=True)
def _hash(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 144665) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h % 12


@njit(inline="always", cache=True)
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(inline="always", cache=True)
def _grad(h, x, y, z):
    return GRAD[h, 0] * x + GRAD[h, 1] * y + GRAD[h, 2] * z


@njit(inline="always", cache=True)
def noise3p(x, y, z, pz, seed):
    """Perlin gradient noise, periodic in z with integer period pz."""
    fx = math.floor(x)
    fy = math.floor(y)
    fz = math.floor(z)
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    x -= fx
    y -= fy
    z -= fz
    z0 = iz % pz
    z1 = (iz + 1) % pz
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)
    n000 = _grad(_hash(ix, iy, z0, seed), x, y, z)
    n100 = _grad(_hash(ix + 1, iy, z0, seed), x - 1, y, z)
    n010 = _grad(_hash(ix, iy + 1, z0, seed), x, y - 1, z)
    n110 = _grad(_hash(ix + 1, iy + 1, z0, seed), x - 1, y - 1, z)
    n001 = _grad(_hash(ix, iy, z1, seed), x, y, z - 1)
    n101 = _grad(_hash(ix + 1, iy, z1, seed), x - 1, y, z - 1)
    n011 = _grad(_hash(ix, iy + 1, z1, seed), x, y - 1, z - 1)
    n111 = _grad(_hash(ix + 1, iy + 1, z1, seed), x - 1, y - 1, z - 1)
    a = n000 + u * (n100 - n000)
    b = n010 + u * (n110 - n010)
    c = n001 + u * (n101 - n001)
    d = n011 + u * (n111 - n011)
    e = a + v * (b - a)
    f = c + v * (d - c)
    return e + w * (f - e)


@njit(inline="always", cache=True)
def fbm(x, y, phase, octaves, pz, seed, gain):
    """Fractal noise; `phase` in [0,1) is the loop position.

    Every octave evolves at the same lattice rate with period pz, so the
    whole sum is periodic in `phase` and fine detail flows rather than boils.
    """
    s = 0.0
    amp = 1.0
    norm = 0.0
    # Rotate each octave a little to hide lattice alignment.
    ca = 0.8
    sa = 0.6
    for k in range(octaves):
        m = 1
        z = phase * pz * m
        s += amp * noise3p(x, y, z, pz * m, seed + k * 31)
        norm += amp
        nx = (ca * x - sa * y) * 2.03 + 1.7
        ny = (sa * x + ca * y) * 2.03 - 3.1
        x = nx
        y = ny
        amp *= gain
    return s / norm


# --------------------------------------------------------------- fields ----

@njit(parallel=True, cache=True)
def warp_field(out, w, h, aspect, phase, scale):
    """Low-frequency two-stage domain warp, computed at reduced resolution.

    out[...,0:2] = q, out[...,2:4] = r
    """
    for j in prange(h):
        py = j / (h - 1)
        for i in range(w):
            px = i / (w - 1) * aspect
            z = 1.0 / (1.0 - PERSP * (1.0 - py))
            x = (px - 0.5 * aspect) * z * scale
            y = -z * scale * PERSP_Y
            # the warp varies more slowly than the clumps it bends
            x *= 0.45
            y *= 0.45
            q0 = fbm(x, y, phase, 3, 3, 11, 0.5)
            q1 = fbm(x + 5.2, y + 1.3, phase, 3, 3, 23, 0.5)
            r0 = fbm(x + WARP_Q * q0 + 1.7, y + WARP_Q * q1 + 9.2, phase, 4, 3, 37, 0.5)
            r1 = fbm(x + WARP_Q * q0 + 8.3, y + WARP_Q * q1 + 2.8, phase, 4, 3, 41, 0.5)
            out[j, i, 0] = q0
            out[j, i, 1] = q1
            out[j, i, 2] = r0
            out[j, i, 3] = r1


@njit(inline="always", cache=True)
def _bilerp(f, c, fx, fy, w, h):
    x0 = int(fx)
    y0 = int(fy)
    if x0 >= w - 1:
        x0 = w - 2
    if y0 >= h - 1:
        y0 = h - 2
    tx = fx - x0
    ty = fy - y0
    a = f[y0, x0, c] + tx * (f[y0, x0 + 1, c] - f[y0, x0, c])
    b = f[y0 + 1, x0, c] + tx * (f[y0 + 1, x0 + 1, c] - f[y0 + 1, x0, c])
    return a + ty * (b - a)


@njit(parallel=True, cache=True)
def height_field(hgt, stk, smo, warp, W, H, ww, wh, aspect, phase, scale):
    """Full-resolution height of wind-combed fur clumps.

    The clump field is sampled at a position pushed along the strand
    direction by thin, elongated 'hair' noise. That smears every clump edge
    into a fringe of fine strands, the combed-fur look.
    """
    for j in prange(H):
        py = j / (H - 1)
        fy = py * (wh - 1)
        for i in range(W):
            px = i / (W - 1) * aspect
            fx = i / (W - 1) * (ww - 1)
            q0 = _bilerp(warp, 0, fx, fy, ww, wh)
            r0 = _bilerp(warp, 2, fx, fy, ww, wh)
            r1 = _bilerp(warp, 3, fx, fy, ww, wh)
            # oblique view of a surface: depth grows toward the top of frame
            z = 1.0 / (1.0 - PERSP * (1.0 - py))
            x = (px - 0.5 * aspect) * z * scale
            y = -z * scale * PERSP_Y
            # strand direction: rising to the right, gently bent by the warp
            ang = FUR_ANG + FUR_BEND * q0
            cd = math.cos(ang)
            sd = -math.sin(ang)          # screen y points down
            u = x * cd + y * sd          # along the strands
            v = -x * sd + y * cd         # across the strands
            # hair noise: long along u, very thin across v
            h1 = fbm(u * HAIR_U, v * HAIR_V, phase, 3, 2, 211, 0.55)
            h2 = fbm(u * HAIR_U * 2.1 + 5.3, v * HAIR_V * 2.3 + 1.9, phase, 2, 2, 223, 0.5)
            hair = h1 + 0.6 * h2
            # smear the clumps along the strands
            cu = u + FUR_LEN * hair
            b = fbm(cu * CLUMP_U + WARP_B * r0, v * CLUMP_V + WARP_B * r1,
                    phase, 6, 4, 101, 0.56)
            # crinkly foam detail, also smeared along the strands
            b += FOAM * fbm(cu * 16.0 + 3.3, v * 16.0 - 7.7, phase, 3, 4, 307, 0.5)
            hgt[j, i] = b
            stk[j, i] = hair
            # smooth, un-smeared surface used only for lighting the waves
            smo[j, i] = fbm(u * CLUMP_U + WARP_B * r0, v * CLUMP_V + WARP_B * r1,
                            phase, 4, 4, 101, 0.56)


@njit(parallel=True, cache=True)
def shade(rgb, hgt, stk, smo, W, H, lut, lo, hi):
    n = lut.shape[0]
    # light from the upper-left
    lx = -0.5
    ly = -0.7
    lz = 0.5
    strength = SHADE_K * W / 3840.0
    for j in prange(H):
        jm = j - 1 if j > 0 else 0
        jp = j + 1 if j < H - 1 else H - 1
        for i in range(W):
            im = i - 1 if i > 0 else 0
            ip = i + 1 if i < W - 1 else W - 1
            dx = (smo[j, ip] - smo[j, im]) * strength * 0.5
            dy = (smo[jp, i] - smo[jm, i]) * strength * 0.5
            inv = 1.0 / math.sqrt(dx * dx + dy * dy + 1.0)
            diff = (-dx * lx - dy * ly + lz) * inv
            hv = (hgt[j, i] - lo) / (hi - lo)
            # clump tops are bright, gaps dark; light adds strand relief
            t = H_W * hv + D_W * (diff - 0.5) + S_W * stk[j, i] * hv + BIAS
            if 0.0 < t < 1.0:
                t = t * t * (3.0 - 2.0 * t) * 0.7 + t * 0.3
            if t < 0.0:
                t = 0.0
            if t > 1.0:
                t = 1.0
            fi = t * (n - 1)
            k = int(fi)
            if k >= n - 1:
                k = n - 2
            f = fi - k
            for c in range(3):
                val = lut[k, c] + f * (lut[k + 1, c] - lut[k, c])
                rgb[j, i, c] = np.uint8(val)


def make_lut(n=1024):
    # dark ember -> deep red -> orange -> amber -> pale gold
    stops = [
        (0.00, (38, 4, 0)),
        (0.20, (88, 12, 3)),
        (0.42, (184, 58, 13)),
        (0.62, (224, 104, 30)),
        (0.80, (246, 176, 78)),
        (0.93, (250, 212, 120)),
        (1.00, (253, 232, 160)),
    ]
    xs = np.linspace(0, 1, n)
    lut = np.zeros((n, 3), np.float32)
    pos = [s[0] for s in stops]
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, [s[1][c] for s in stops])
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--fps", type=float, default=30000 / 1001)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--frames", type=str, default="", help="comma list of frame indices (preview)")
    ap.add_argument("--scale", type=float, default=1.6, help="feature size: bigger = smaller waves")
    ap.add_argument("--out", type=str, default="", help="write PNGs to this dir instead of stdout")
    a = ap.parse_args()

    W, H = a.width, a.height
    aspect = W / H
    n_frames = int(round(a.seconds * a.fps))
    ww, wh = W // 4, H // 4
    warp = np.zeros((wh, ww, 4), np.float32)
    hgt = np.zeros((H, W), np.float32)
    stk = np.zeros((H, W), np.float32)
    smo = np.zeros((H, W), np.float32)
    rgb = np.zeros((H, W, 3), np.uint8)
    lut = make_lut()

    idx = [int(s) for s in a.frames.split(",")] if a.frames else range(n_frames)
    out = sys.stdout.buffer
    for f in idx:
        phase = (f % n_frames) / n_frames
        warp_field(warp, ww, wh, aspect, phase, a.scale)
        height_field(hgt, stk, smo, warp, W, H, ww, wh, aspect, phase, a.scale)
        shade(rgb, hgt, stk, smo, W, H, lut, -0.30, 0.30)
        if a.out:
            import subprocess
            p = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{W}x{H}", "-i", "-", f"{a.out}/f{f:05d}.png"],
                input=rgb.tobytes())
        else:
            out.write(rgb.tobytes())
        print(f"frame {f}/{n_frames}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
