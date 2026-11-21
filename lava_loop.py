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
WARP_Q = 1.4    # strength of second warp stage
WARP_B = 1.1    # how much the warp bends the main waves
WARP_S = 0.9    # how much the warp bends the streaks
STR_U = 11.0    # streak frequency along the flow
STR_V = 52.0    # streak frequency across the flow
SHADE_K = 420.0 # relief of the waves
H_W = 0.80      # weight of wave height in colour
D_W = 0.40      # weight of lighting in colour
S_W = 0.26      # weight of streaks in colour
BIAS = 0.02
SPEC = 0.30     # glow on the crests facing the light

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
            x = px * scale
            y = py * scale
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
def height_field(hgt, stk, warp, W, H, ww, wh, aspect, phase, scale):
    """Full-resolution height: warped fbm + flowing anisotropic streaks."""
    # Streak flow: two phase-offset layers advected along the flow and
    # cross-faded (flow-map technique). Each layer cycles an integer number
    # of times per loop, so the result stays periodic.
    cycles = 4.0
    ph = phase * cycles
    pa = ph - math.floor(ph)
    pb = ph + 0.5 - math.floor(ph + 0.5)
    wa = 1.0 - abs(2.0 * pa - 1.0)       # triangle weights, sum to 1
    wb = 1.0 - wa
    wnorm = 1.0 / math.sqrt(wa * wa + wb * wb)
    flow = 0.35                           # advection distance per cycle
    # streak direction (slightly downward diagonal like rolling waves)
    ang = -0.32
    cd = math.cos(ang)
    sd = math.sin(ang)
    for j in prange(H):
        py = j / (H - 1)
        fy = py * (wh - 1)
        for i in range(W):
            px = i / (W - 1) * aspect
            fx = i / (W - 1) * (ww - 1)
            q0 = _bilerp(warp, 0, fx, fy, ww, wh)
            q1 = _bilerp(warp, 1, fx, fy, ww, wh)
            r0 = _bilerp(warp, 2, fx, fy, ww, wh)
            r1 = _bilerp(warp, 3, fx, fy, ww, wh)
            x = px * scale
            y = py * scale
            # main body of the waves
            b = fbm(x + WARP_B * r0, y + WARP_B * r1, phase, 5, 4, 101, 0.5)
            # streak coordinates: rotate, bend with the warp, then stretch
            sx = x + WARP_S * q0
            sy = y + WARP_S * q1
            u = cd * sx - sd * sy
            v = sd * sx + cd * sy
            # local flow direction follows the warp gradient
            du = 0.8 + 0.6 * r0
            ua = u - du * flow * pa
            ub = u - du * flow * pb
            sa = fbm(ua * STR_U, v * STR_V, phase, 4, 2, 211, 0.6)
            sb = fbm(ub * STR_U + 13.7, v * STR_V + 7.1, phase, 4, 2, 211, 0.6)
            st = (wa * sa + wb * sb) * wnorm
            hgt[j, i] = b
            stk[j, i] = st


@njit(parallel=True, cache=True)
def shade(rgb, hgt, stk, W, H, lut, lo, hi):
    n = lut.shape[0]
    # light from the upper-left
    lx = -0.5
    ly = -0.7
    lz = 0.5
    strength = SHADE_K * W / 3840.0
    for j in prange(H):
        jm = j - 2 if j > 1 else 0
        jp = j + 2 if j < H - 2 else H - 1
        for i in range(W):
            im = i - 2 if i > 1 else 0
            ip = i + 2 if i < W - 2 else W - 1
            dx = (hgt[j, ip] - hgt[j, im]) * strength * 0.25
            dy = (hgt[jp, i] - hgt[jm, i]) * strength * 0.25
            inv = 1.0 / math.sqrt(dx * dx + dy * dy + 1.0)
            diff = (-dx * lx - dy * ly + lz) * inv
            hv = (hgt[j, i] - lo) / (hi - lo)
            st = stk[j, i]
            # fur-like streaks: strongest on lit wave faces
            # ridged strands: thin bright hairs on a darker base
            rs = 1.0 - abs(st) * 3.2
            if rs < 0.0:
                rs = 0.0
            rs = rs * rs - 0.35
            spec = diff * diff
            spec = spec * spec * spec
            t = (H_W * hv + D_W * (diff - 0.5) + S_W * rs * (0.4 + hv)
                 + SPEC * spec * hv + BIAS)
            # push the extremes: darker troughs, hotter crests
            t = t * t * (3.0 - 2.0 * t) * 0.6 + t * 0.4 if 0.0 < t < 1.0 else t
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
        (0.00, (18, 2, 0)),
        (0.18, (70, 8, 1)),
        (0.38, (150, 34, 4)),
        (0.58, (214, 92, 16)),
        (0.76, (240, 154, 52)),
        (0.90, (252, 206, 112)),
        (1.00, (255, 238, 176)),
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
    ap.add_argument("--scale", type=float, default=1.25, help="feature size: bigger = smaller waves")
    ap.add_argument("--out", type=str, default="", help="write PNGs to this dir instead of stdout")
    a = ap.parse_args()

    W, H = a.width, a.height
    aspect = W / H
    n_frames = int(round(a.seconds * a.fps))
    ww, wh = W // 4, H // 4
    warp = np.zeros((wh, ww, 4), np.float32)
    hgt = np.zeros((H, W), np.float32)
    stk = np.zeros((H, W), np.float32)
    rgb = np.zeros((H, W, 3), np.uint8)
    lut = make_lut()

    idx = [int(s) for s in a.frames.split(",")] if a.frames else range(n_frames)
    out = sys.stdout.buffer
    for f in idx:
        phase = (f % n_frames) / n_frames
        warp_field(warp, ww, wh, aspect, phase, a.scale)
        height_field(hgt, stk, warp, W, H, ww, wh, aspect, phase, a.scale)
        shade(rgb, hgt, stk, W, H, lut, -0.5, 0.5)
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
