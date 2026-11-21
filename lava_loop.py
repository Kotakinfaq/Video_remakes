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
WARP_B = 0.35   # how much the warp wobbles the clumps
PERSP = 0.30    # 0 = flat top-down view, higher = more oblique
PERSP_Y = 2.6 
WAVE_STRETCH = 0.42  # <1 stretches waves horizontally  # vertical world extent factor (keeps clumps round)
CLUMP = 2.6     # clump frequency (higher = smaller clumps)
BILLOW = 0.40
FOLD_F = 2.0    # size of the cauliflower folds relative to clumps   # cauliflower folds on the clumps
FOAM = 0.20     # fine crinkly detail on the clump tops
FUR_ANG = 0.38  # hair angle on screen (radians, rising to the right)
FUR_LEN = 0.010 # how far (screen heights) clump edges smear into hairs
HAIR_U = 18.0   # hair frequency along the hairs (lower = longer hairs)
HAIR_V = 260.0   # hair frequency across the hairs (higher = thinner)
SHADE_K = 700.0 # relief of the clumps (lighting slope scale)
DETAIL_K = 0.55 # how much fine fur/crinkle relief catches the light
LIGHT_X = -0.3  # light from upper-left, behind the waves
LIGHT_Z = 0.9
SH_STEPS = 14   # cast-shadow march length
SH_SLOPE = 0.012
SH_SOFT = 12.0
SH_MIN = 0.25
EDGE = 1.0      # crispness of cloud-top edges
H_W = 0.80      # weight of clump height in colour
D_W = 0.70      # weight of lighting in colour
S_W = 0.14      # weight of thin hair lines in colour
BIAS = -0.27
HAIR_SP = 6.0   # spacing of hair roots in pixels at 4K
HAIR_LAYERS = 2 # hairs per root cell
HAIR_LEN = 120.0 # hair length in pixels at 4K (near the bottom of frame)
HAIR_BEND = 0.6 # how much the hair direction wanders across the frame
HAIR_JIT = 0.12 # per-hair direction randomness
HAIR_CURL = 0.25 # per-hair curvature
HAIR_VAR = 0.05 # per-hair brightness randomness
HAIR_ROOT = 0.55 # root brightness relative to tip
HAIR_TIP = 0.10 # extra glow at the tip
HAIR_ALPHA = 0.95
HAIR_W = 1.4    # hair thickness in pixels at 4K
BG_DIM = 0.65   # surface brightness showing between hairs

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


@njit(inline="always", cache=True)
def _world(px, py, aspect, scale):
    """Screen -> world coords for an obliquely viewed surface."""
    z = 1.0 / (1.0 - PERSP * (1.0 - py))
    return (px - 0.5 * aspect) * z * scale * WAVE_STRETCH, -z * scale * PERSP_Y


@njit(parallel=True, cache=True)
def height_field(hgt, stk, smo, warp, W, H, ww, wh, aspect, phase, scale):
    """Puffy, cloud-like lava clumps covered in fine, straight, combed hairs.

    hgt: clump brightness, smeared along the hair direction so clump edges
         break up into fringes of strands.
    stk: thin hair-line pattern (screen space, constant size on screen).
    smo: smooth clump surface used for lighting.
    """
    ca = math.cos(FUR_ANG)
    sa = math.sin(FUR_ANG)
    dxs = ca                  # hair direction on screen (rising to the right)
    dys = -sa
    for j in prange(H):
        py = j / (H - 1)
        fy = py * (wh - 1)
        for i in range(W):
            px = i / (W - 1) * aspect
            fx = i / (W - 1) * (ww - 1)
            r0 = _bilerp(warp, 2, fx, fy, ww, wh)
            r1 = _bilerp(warp, 3, fx, fy, ww, wh)
            # hair noise in screen space: long along the hair, very thin across
            u = px * ca - py * sa
            v = px * sa + py * ca
            h1 = fbm(u * HAIR_U, v * HAIR_V, phase, 2, 2, 211, 0.5)
            h2 = fbm(u * HAIR_U * 0.6 + 5.3, v * HAIR_V * 0.55 + 1.9, phase, 2, 2, 223, 0.5)
            hair = h1 + 0.7 * h2
            # smear: sample the clumps a little way along the hair
            spx = px + dxs * FUR_LEN * hair
            spy = py + dys * FUR_LEN * hair
            if spy < 0.0:
                spy = 0.0
            if spy > 1.0:
                spy = 1.0
            x, y = _world(spx, spy, aspect, scale)
            x += WARP_B * r0
            y += WARP_B * r1
            # billowy clumps: crisp cauliflower edges from folded octaves
            c = fbm(x * CLUMP, y * CLUMP, phase, 4, 4, 101, 0.55)
            fold = fbm(x * CLUMP * FOLD_F + 7.7, y * CLUMP * FOLD_F - 2.2, phase, 4, 4, 131, 0.55)
            c += BILLOW * (0.25 - abs(fold))
            # crinkly detail, strongest on the tops
            cr = fbm(x * CLUMP * 9.0 + 3.3, y * CLUMP * 9.0 - 7.7, phase, 3, 4, 307, 0.6)
            hgt[j, i] = c + FOAM * cr * (0.5 + 2.0 * max(c, 0.0))
            stk[j, i] = hair
            x0, y0 = _world(px, py, aspect, scale)
            x0 += WARP_B * r0
            y0 += WARP_B * r1
            c0 = fbm(x0 * CLUMP, y0 * CLUMP, phase, 4, 4, 101, 0.55)
            f0 = fbm(x0 * CLUMP * FOLD_F + 7.7, y0 * CLUMP * FOLD_F - 2.2, phase, 3, 4, 131, 0.55)
            smo[j, i] = c0 + BILLOW * (0.25 - abs(f0))


@njit(parallel=True, cache=True)
def shade(tone, hgt, stk, smo, W, H, lo, hi):
    """Colour from lighting on the clump surface: lit crinkly tops, shadowed
    undersides, cast shadows behind crests, fur relief on the slopes."""
    lx = LIGHT_X
    ly = -1.0
    lz = LIGHT_Z
    ln = 1.0 / math.sqrt(lx * lx + ly * ly + lz * lz)
    lx *= ln
    ly *= ln
    lz *= ln
    px_scale = W / 3840.0
    strength = SHADE_K * px_scale
    st = max(1, int(round(3 * px_scale)))      # gradient stencil (px)
    sh_step = max(1, int(round(10 * px_scale))) # shadow march step (px)
    for j in prange(H):
        jm = j - st if j >= st else 0
        jp = j + st if j < H - st else H - 1
        for i in range(W):
            im = i - st if i >= st else 0
            ip = i + st if i < W - st else W - 1
            dx = ((smo[j, ip] - smo[j, im]) + DETAIL_K * (hgt[j, ip] - hgt[j, im])) * strength / (2 * st)
            dy = ((smo[jp, i] - smo[jm, i]) + DETAIL_K * (hgt[jp, i] - hgt[jm, i])) * strength / (2 * st)
            inv = 1.0 / math.sqrt(dx * dx + dy * dy + 1.0)
            diff = (-dx * lx - dy * ly + lz) * inv
            if diff < 0.0:
                diff = 0.0
            # cast shadow: march toward the light (up the screen)
            h0 = smo[j, i]
            shadow = 1.0
            for k in range(1, SH_STEPS + 1):
                jj = j - k * sh_step
                ii = i + int(k * sh_step * lx / -ly * 0.5)
                if jj < 0 or ii < 0 or ii >= W:
                    break
                occ = smo[jj, ii] - h0 - k * SH_SLOPE
                if occ > 0.0:
                    s2 = 1.0 - occ * SH_SOFT
                    if s2 < shadow:
                        shadow = s2
            if shadow < SH_MIN:
                shadow = SH_MIN
            lit = diff * shadow
            hv = (hgt[j, i] - lo) / (hi - lo)
            if hv < 0.0:
                hv = 0.0
            if hv > 1.0:
                hv = 1.0
            # crisp cloud-top edges
            hv = hv * hv * (3.0 - 2.0 * hv)
            hv = hv * hv * (3.0 - 2.0 * hv) * EDGE + hv * (1.0 - EDGE)
            tone[j, i] = H_W * hv + D_W * lit + BIAS


@njit(inline="always", cache=True)
def _rand(a, b, c):
    """Deterministic hash -> [0, 1)."""
    h = (a * 73856093 ^ b * 19349663 ^ c * 83492791) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@njit(inline="always", cache=True)
def _splat(img, zb, x, y, col, a, depth, W, H):
    """Bilinear (anti-aliased) blend of one hair sample, nearer hairs win."""
    xi = int(math.floor(x))
    yi = int(math.floor(y))
    fx = x - xi
    fy = y - yi
    for dy in range(2):
        yy = yi + dy
        if yy < 0 or yy >= H:
            continue
        wy = fy if dy == 1 else 1.0 - fy
        for dx in range(2):
            xx = xi + dx
            if xx < 0 or xx >= W:
                continue
            if depth < zb[yy, xx] - 6.0:
                continue
            w = (fx if dx == 1 else 1.0 - fx) * wy * a
            img[yy, xx] = img[yy, xx] * (1.0 - w) + col * w
            if depth > zb[yy, xx]:
                zb[yy, xx] = depth


@njit(parallel=True, cache=True)
def draw_hairs(img, zb, tone, W, H, aspect, phase, scale):
    """Grow individual hairs from a jittered grid of roots.

    Each hair takes its brightness from the lit surface at its root, is dark
    at the root and bright at the tip, and leans up-right with a direction
    that wanders smoothly across the frame and sways over the loop.
    """
    px_scale = W / 3840.0
    sp = HAIR_SP * px_scale
    nx = int(W / sp) + 2
    ny = int(H / sp) + int(HAIR_LEN * px_scale / sp) + 2
    for gy in prange(ny):
        for gx in range(nx):
            for layer in range(HAIR_LAYERS):
                r1 = _rand(gx, gy, layer * 7 + 1)
                r2 = _rand(gx, gy, layer * 7 + 2)
                r3 = _rand(gx, gy, layer * 7 + 3)
                r4 = _rand(gx, gy, layer * 7 + 4)
                r5 = _rand(gx, gy, layer * 7 + 5)
                rx = (gx + r1 - 0.5) * sp
                ry = (gy + r2 - 0.5) * sp
                if rx < 0 or rx >= W or ry < 0 or ry >= H:
                    continue
                pyn = ry / H
                persp = 1.0 - PERSP * (1.0 - pyn)      # nearer = bigger
                # direction field: smooth wander + per-hair jitter + sway
                wx = rx / H * 3.0
                wy = pyn * 3.0
                ang = (FUR_ANG + HAIR_BEND * fbm(wx, wy, phase, 2, 3, 401, 0.5)
                       + HAIR_JIT * (r3 - 0.5))
                curl = HAIR_CURL * (r4 - 0.5)
                ln = HAIR_LEN * px_scale * persp * (0.45 + 0.9 * r5)
                hw = HAIR_W * px_scale * persp
                base = tone[int(ry), int(rx)]
                bright = base + HAIR_VAR * (r3 - 0.5)
                steps = int(ln * 1.5) + 2
                ca = math.cos(ang)
                sa = math.sin(ang)
                x = rx
                y = ry
                stepl = ln / steps
                for k in range(steps):
                    t = k / (steps - 1)
                    # dark root, glowing tip
                    col = bright * (HAIR_ROOT + (1.0 - HAIR_ROOT) * t) + HAIR_TIP * t * t * t
                    a = HAIR_ALPHA * (1.0 - 0.5 * t)
                    # thickness: a few parallel samples across the hair
                    wid = hw * (1.0 - 0.55 * t)
                    o = -0.5 * wid
                    while o <= 0.5 * wid + 1e-3:
                        _splat(img, zb, x + sa * o, y + ca * o, col, a, ry, W, H)
                        o += 1.0
                    a2 = curl * stepl / ln
                    c2 = ca * math.cos(a2) - sa * math.sin(a2)
                    sa = sa * math.cos(a2) + ca * math.sin(a2)
                    ca = c2
                    x += ca * stepl
                    y -= sa * stepl


@njit(parallel=True, cache=True)
def to_rgb(rgb, img, W, H, lut):
    n = lut.shape[0]
    for j in prange(H):
        for i in range(W):
            t = img[j, i]
            if t < 0.0:
                t = 0.0
            if t > 1.0:
                t = 1.0
            fi = t * (n - 1)
            k = int(fi)
            if k >= n - 1:
                k = n - 2
            fr = fi - k
            for c in range(3):
                val = lut[k, c] + fr * (lut[k + 1, c] - lut[k, c])
                rgb[j, i, c] = np.uint8(val)


def make_lut(n=1024):
    # dark ember -> deep red -> orange -> amber -> pale gold
    stops = [
        (0.00, (40, 4, 2)),
        (0.18, (104, 14, 7)),
        (0.36, (162, 34, 14)),
        (0.52, (204, 74, 30)),
        (0.68, (230, 128, 60)),
        (0.82, (240, 172, 68)),
        (0.93, (246, 204, 104)),
        (1.00, (248, 220, 150)),
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
    tone = np.zeros((H, W), np.float32)
    img = np.zeros((H, W), np.float32)
    zb = np.zeros((H, W), np.float32)
    rgb = np.zeros((H, W, 3), np.uint8)
    lut = make_lut()

    idx = [int(s) for s in a.frames.split(",")] if a.frames else range(n_frames)
    out = sys.stdout.buffer
    for f in idx:
        phase = (f % n_frames) / n_frames
        warp_field(warp, ww, wh, aspect, phase, a.scale)
        height_field(hgt, stk, smo, warp, W, H, ww, wh, aspect, phase, a.scale)
        shade(tone, hgt, stk, smo, W, H, -0.30, 0.30)
        img[:] = tone * BG_DIM
        zb[:] = -1e9
        draw_hairs(img, zb, tone, W, H, aspect, phase, a.scale)
        to_rgb(rgb, img, W, H, lut)
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
