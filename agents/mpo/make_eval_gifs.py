"""Compose per-env GIFs from seed eval videos, with seed labels on each tile.

Example:
  python make_eval_gifs.py
  python make_eval_gifs.py --env humanoid --fps 10 --duration 10
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from collections import defaultdict

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .plot_runs import ENV_ORDER, ENV_TITLES, env_from_dir, seed_from_dir
except ImportError:
    from plot_runs import ENV_ORDER, ENV_TITLES, env_from_dir, seed_from_dir


def find_eval_videos(runs_dir: str, env_name: str | None = None) -> dict[str, list[tuple[int, str]]]:
    """Return {env: [(seed, video_path), ...]} using the latest eval mp4 per run."""
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    pattern = os.path.join(runs_dir, "*_seed_*", "videos", "eval_*.mp4")
    for path in glob.glob(pattern):
        run_dir = os.path.dirname(os.path.dirname(path))
        env = env_from_dir(run_dir)
        if env is None:
            continue
        if env_name is not None and env != env_name:
            continue
        seed = seed_from_dir(run_dir)
        grouped[env].append((seed, path))

    # Keep the newest eval video per seed (by mtime).
    out: dict[str, list[tuple[int, str]]] = {}
    for env, items in grouped.items():
        best: dict[int, tuple[float, str]] = {}
        for seed, path in items:
            mtime = os.path.getmtime(path)
            if seed not in best or mtime > best[seed][0]:
                best[seed] = (mtime, path)
        out[env] = sorted((seed, path) for seed, (_, path) in best.items())
    return out


def load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ):
        if os.path.isfile(name):
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def video_duration_sec(path: str) -> float:
    import subprocess

    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        text=True,
    ).strip()
    return float(out)


def subsample_video(path: str, n_out: int, tile_w: int) -> list[np.ndarray]:
    """Extract n_out evenly spaced frames via ffmpeg (scaled to tile_w)."""
    import subprocess
    import tempfile

    duration = max(1e-3, video_duration_sec(path))
    # Sample uniformly over the full clip.
    fps = n_out / duration
    frames: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="eval_gif_") as tmp:
        pattern = os.path.join(tmp, "%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            path,
            "-vf",
            f"fps={fps:.6f},scale={tile_w}:-1",
            "-frames:v",
            str(n_out),
            pattern,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        paths = sorted(glob.glob(os.path.join(tmp, "*.png")))
        if not paths:
            raise RuntimeError(f"ffmpeg produced no frames for {path}")
        for p in paths[:n_out]:
            frames.append(np.asarray(Image.open(p).convert("RGB")))
        # Pad by repeating last frame if short.
        while len(frames) < n_out:
            frames.append(frames[-1].copy())
    return frames


def labeled_tile(
    frame: np.ndarray,
    label: str,
    tile_w: int,
    label_h: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    aspect = img.height / max(1, img.width)
    tile_h = max(1, int(round(tile_w * aspect)))
    img = img.resize((tile_w, tile_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (tile_w, label_h + tile_h), color=(20, 20, 20))
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (tile_w - tw) // 2
    y = (label_h - th) // 2
    # Soft outline for readability on any background.
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), label, font=font, fill=(0, 0, 0))
    draw.text((x, y), label, font=font, fill=(255, 255, 255))
    return canvas


def grid_shape(n: int) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    cols = min(5, n)
    rows = int(math.ceil(n / cols))
    return rows, cols


def compose_gif(
    env: str,
    seed_videos: list[tuple[int, str]],
    *,
    out_path: str,
    duration: float,
    fps: float,
    tile_w: int,
    label_h: int,
) -> None:
    if not seed_videos:
        print(f"[!] No videos for {env}")
        return

    n_out = max(1, int(round(duration * fps)))
    rows, cols = grid_shape(len(seed_videos))
    font = load_font(max(12, tile_w // 10))
    title = ENV_TITLES.get(env, env.capitalize())

    print(f"  loading {len(seed_videos)} videos → {n_out} frames each")
    per_seed_frames: list[tuple[int, list[np.ndarray]]] = []
    for seed, path in seed_videos:
        print(f"    seed {seed}: {os.path.basename(path)}")
        per_seed_frames.append((seed, subsample_video(path, n_out, tile_w)))

    frames_out: list[np.ndarray] = []
    for t in range(n_out):
        tiles: list[Image.Image] = []
        for seed, frames in per_seed_frames:
            tiles.append(labeled_tile(frames[t], f"seed {seed}", tile_w, label_h, font))

        cell_h = tiles[0].height
        grid_w = cols * tile_w
        grid_h = rows * cell_h
        title_h = max(22, label_h + 4)
        canvas = Image.new("RGB", (grid_w, title_h + grid_h), color=(12, 12, 12))
        draw = ImageDraw.Draw(canvas)
        title_font = load_font(max(14, tile_w // 8))
        title_text = f"{title}  ({len(seed_videos)} seeds)"
        bbox = draw.textbbox((0, 0), title_text, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(((grid_w - tw) // 2, 4), title_text, font=title_font, fill=(240, 240, 240))

        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            canvas.paste(tile, (c * tile_w, title_h + r * cell_h))

        frames_out.append(np.asarray(canvas))
        if (t + 1) % max(1, n_out // 5) == 0:
            print(f"  [{env}] composed {t + 1}/{n_out}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(
        out_path,
        frames_out,
        fps=fps,
        loop=0,
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[√] Wrote {out_path} ({n_out} frames, {size_mb:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--env", type=str, default=None, choices=list(ENV_ORDER))
    p.add_argument("--output_dir", type=str, default=None, help="Defaults to <runs_dir>/eval_gifs")
    p.add_argument("--duration", type=float, default=10.0, help="GIF length in seconds")
    p.add_argument("--fps", type=float, default=10.0, help="GIF playback FPS")
    p.add_argument("--tile_width", type=int, default=192)
    p.add_argument("--label_height", type=int, default=22)
    return p.parse_args()


def main():
    cli = parse_args()
    runs_dir = cli.runs_dir
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.join(os.path.dirname(__file__), runs_dir)
    output_dir = cli.output_dir or os.path.join(runs_dir, "eval_gifs")

    grouped = find_eval_videos(runs_dir, env_name=cli.env)
    if not grouped:
        raise SystemExit(f"No eval_*.mp4 videos found under {runs_dir}")

    envs = [e for e in ENV_ORDER if e in grouped] + sorted(e for e in grouped if e not in ENV_ORDER)
    for env in envs:
        print(f"\n=== {ENV_TITLES.get(env, env)} ({len(grouped[env])} videos) ===")
        out_path = os.path.join(output_dir, f"{env}.gif")
        compose_gif(
            env,
            grouped[env],
            out_path=out_path,
            duration=cli.duration,
            fps=cli.fps,
            tile_w=cli.tile_width,
            label_h=cli.label_height,
        )


if __name__ == "__main__":
    main()
