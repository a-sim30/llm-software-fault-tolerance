"""
Assignment 2 - Interpolation-Based Super-Resolution
====================================================

Pipeline:
  1. Load four grayscale images (512 x 512).
  2. Downsample each to 64 x 64, 128 x 128, 256 x 256 with anti-aliased
     averaging (area interpolation) so we simulate a realistic low-res
     acquisition rather than a naive stride subsample.
  3. Super-resolve each downsampled image back to 512 x 512 with a family
     of interpolation methods spanning increasing polynomial order:
         Order 0  - Nearest neighbour
         Order 1  - Bilinear
         Order 2  - B-spline (quadratic)
         Order 3  - Bicubic
         Order 4  - B-spline (quartic)
         Order 5  - B-spline (quintic)
         Lanczos-4 - Windowed sinc (non-polynomial, listed for comparison)
  4. Compute PSNR and SSIM of every super-resolved image against the
     original 512 x 512 grayscale reference.
  5. Time each interpolation call (median of several runs to smooth noise).
  6. Print consolidated results as pretty tables in the terminal and dump
     CSV / Markdown copies for the report.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORIGINAL_SIZE = 512
DOWNSAMPLE_SIZES = (64, 128, 256)             # low-resolution targets
TIMING_REPEATS = 5                            # runs per (image, method) cell


@dataclass(frozen=True)
class Method:
    """One interpolation method along with a printable label and a callable."""
    order_label: str        # e.g. "0", "1", "3", "Lanczos"
    name: str               # e.g. "Nearest neighbour"
    apply: Callable[[np.ndarray, int], np.ndarray]


# ---------------------------------------------------------------------------
# Interpolation implementations
# ---------------------------------------------------------------------------

def _cv2_resize(flag: int):
    """Return a resize callable that uses OpenCV with the given flag."""
    def _resize(img: np.ndarray, target: int) -> np.ndarray:
        return cv2.resize(img, (target, target), interpolation=flag)
    return _resize


def _spline_resize(order: int):
    """Return a resize callable that uses SciPy's spline interpolation."""
    def _resize(img: np.ndarray, target: int) -> np.ndarray:
        zoom = target / img.shape[0]
        # mode='reflect' avoids dark borders that 'constant' would introduce.
        return ndimage.zoom(img, zoom=zoom, order=order, mode="reflect",
                            prefilter=True)
    return _resize


def build_methods() -> list[Method]:
    """The interpolation catalogue used throughout the study."""
    return [
        Method("0", "Nearest neighbour (zeroth order)", _cv2_resize(cv2.INTER_NEAREST)),
        Method("1", "Bilinear (first order)",           _cv2_resize(cv2.INTER_LINEAR)),
        Method("2", "B-spline quadratic (second order)", _spline_resize(2)),
        Method("3", "Bicubic (third order)",             _cv2_resize(cv2.INTER_CUBIC)),
        Method("4", "B-spline quartic (fourth order)",   _spline_resize(4)),
        Method("5", "B-spline quintic (fifth order)",    _spline_resize(5)),
        Method("L", "Lanczos-4 (windowed sinc)",         _cv2_resize(cv2.INTER_LANCZOS4)),
    ]


# ---------------------------------------------------------------------------
# Image I/O and downsampling
# ---------------------------------------------------------------------------

def load_grayscale_512(path: Path) -> np.ndarray:
    """Read an image from disk, convert to grayscale, force 512 x 512."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        # cv2 loads BGR; use luminance conversion.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape != (ORIGINAL_SIZE, ORIGINAL_SIZE):
        img = cv2.resize(img, (ORIGINAL_SIZE, ORIGINAL_SIZE),
                         interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def downsample(img: np.ndarray, target: int) -> np.ndarray:
    """Downsample with area averaging (anti-aliased) - a fair low-res proxy."""
    return cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Quality metrics + timing
# ---------------------------------------------------------------------------

def timed_apply(method: Method, low_res: np.ndarray,
                target: int, repeats: int) -> tuple[np.ndarray, float]:
    """Run `method.apply` `repeats` times, return (last_output, median_ms)."""
    durations: list[float] = []
    output: np.ndarray | None = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        output = method.apply(low_res, target)
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000.0)   # ms
    assert output is not None
    return output, statistics.median(durations)


def score(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """PSNR (dB) and SSIM against the 512 x 512 grayscale reference."""
    # Normalize dtypes / clamp to 8-bit range for a fair comparison.
    ref = reference.astype(np.float64)
    cand = np.clip(candidate, 0, 255).astype(np.float64)
    psnr_val = compute_psnr(ref, cand, data_range=255.0)
    ssim_val = compute_ssim(ref, cand, data_range=255.0)
    return float(psnr_val), float(ssim_val)


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

@dataclass
class Row:
    image: str
    input_size: int
    factor: int
    order: str
    method: str
    psnr_db: float
    ssim: float
    runtime_ms: float


@dataclass
class Study:
    methods: list[Method] = field(default_factory=build_methods)
    rows: list[Row] = field(default_factory=list)

    def run_image(self, name: str, original: np.ndarray,
                  out_dir: Path, save_images: bool) -> None:
        for lr_size in DOWNSAMPLE_SIZES:
            factor = ORIGINAL_SIZE // lr_size
            low_res = downsample(original, lr_size)

            if save_images:
                cv2.imwrite(str(out_dir / f"{name}_lr{lr_size}.png"), low_res)

            for m in self.methods:
                hr, runtime_ms = timed_apply(m, low_res, ORIGINAL_SIZE,
                                             TIMING_REPEATS)
                # Force output dtype to uint8 so metrics compare like-for-like.
                hr_u8 = np.clip(hr, 0, 255).astype(np.uint8)
                p, s = score(original, hr_u8)

                if save_images:
                    tag = m.order_label.replace(" ", "").lower()
                    cv2.imwrite(str(out_dir /
                                    f"{name}_sr_{lr_size}to512_o{tag}.png"),
                                hr_u8)

                self.rows.append(Row(
                    image=name,
                    input_size=lr_size,
                    factor=factor,
                    order=m.order_label,
                    method=m.name,
                    psnr_db=p,
                    ssim=s,
                    runtime_ms=runtime_ms,
                ))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_number(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def _per_image_table(rows: list[Row], image: str) -> str:
    """Long-form table for one image: order x input size, with metrics."""
    headers = ["Order", "Method", "Input", "Factor",
               "PSNR (dB)", "SSIM", "Runtime (ms)"]
    body = [
        [r.order, r.method, f"{r.input_size}x{r.input_size}",
         f"{r.factor}x",
         _fmt_number(r.psnr_db, 2),
         _fmt_number(r.ssim, 4),
         _fmt_number(r.runtime_ms, 3)]
        for r in rows if r.image == image
    ]
    return tabulate(body, headers=headers, tablefmt="fancy_grid",
                    stralign="left", numalign="right")


def _metric_pivot(rows: list[Row], metric: str) -> str:
    """
    Compact table: rows = interpolation order, cols = input size.
    `metric` is 'psnr_db', 'ssim' or 'runtime_ms'.
    Cells are averaged across the four images to summarize the study.
    """
    orders: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r.order not in seen:
            seen.add(r.order)
            orders.append(r.order)

    sizes = sorted({r.input_size for r in rows})
    method_lookup = {r.order: r.method for r in rows}

    headers = ["Order", "Method"] + [f"{s}x{s} -> 512" for s in sizes]
    body: list[list[str]] = []
    decimals = {"psnr_db": 2, "ssim": 4, "runtime_ms": 3}[metric]

    for order in orders:
        line = [order, method_lookup[order]]
        for s in sizes:
            values = [getattr(r, metric) for r in rows
                      if r.order == order and r.input_size == s]
            line.append(_fmt_number(sum(values) / len(values), decimals)
                        if values else "-")
        body.append(line)

    return tabulate(body, headers=headers, tablefmt="fancy_grid",
                    stralign="left", numalign="right")


def print_reports(rows: list[Row], images: list[str]) -> None:
    banner = "=" * 78
    print(banner)
    print(" INTERPOLATION-BASED SUPER-RESOLUTION - RESULTS")
    print(banner)

    for image in images:
        print(f"\n--- Image: {image} ---")
        print(_per_image_table(rows, image))

    print("\n\n" + banner)
    print(" AVERAGES ACROSS ALL IMAGES")
    print(banner)

    print("\nPSNR (dB) - higher is better")
    print(_metric_pivot(rows, "psnr_db"))

    print("\nSSIM - higher is better (max 1.0)")
    print(_metric_pivot(rows, "ssim"))

    print("\nRuntime (ms) - lower is faster")
    print(_metric_pivot(rows, "runtime_ms"))


def dump_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "input_size", "factor", "order", "method",
                    "psnr_db", "ssim", "runtime_ms"])
        for r in rows:
            w.writerow([r.image, r.input_size, r.factor, r.order, r.method,
                        f"{r.psnr_db:.6f}", f"{r.ssim:.6f}",
                        f"{r.runtime_ms:.6f}"])


def dump_markdown(rows: list[Row], images: list[str], path: Path) -> None:
    lines: list[str] = ["# Assignment 2 - Interpolation-Based Super-Resolution",
                        ""]
    for image in images:
        lines.append(f"## {image}")
        lines.append("")
        lines.append("| Order | Method | Input | Factor | PSNR (dB) | SSIM | Runtime (ms) |")
        lines.append("|-------|--------|-------|--------|-----------|------|--------------|")
        for r in rows:
            if r.image != image:
                continue
            lines.append(
                f"| {r.order} | {r.method} | {r.input_size}x{r.input_size} "
                f"| {r.factor}x | {r.psnr_db:.2f} | {r.ssim:.4f} "
                f"| {r.runtime_ms:.3f} |")
        lines.append("")

    lines += ["## Averages across all images", ""]
    for metric, title in (("psnr_db", "PSNR (dB)"),
                          ("ssim", "SSIM"),
                          ("runtime_ms", "Runtime (ms)")):
        lines.append(f"### {title}")
        lines.append("")
        lines.append(_metric_pivot(rows, metric))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _discover_images(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted(p for p in image_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in exts)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path,
                        default=Path(__file__).with_name("images"),
                        help="Folder containing the four source images.")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).with_name("results"),
                        help="Folder for downsampled/super-resolved outputs.")
    parser.add_argument("--num-images", type=int, default=4,
                        help="How many images from --images-dir to process.")
    parser.add_argument("--no-save-images", action="store_true",
                        help="Skip writing per-method output images.")
    args = parser.parse_args()

    image_paths = _discover_images(args.images_dir)
    if len(image_paths) < args.num_images:
        print(f"[error] found {len(image_paths)} image(s) in "
              f"{args.images_dir}, need {args.num_images}.", file=sys.stderr)
        print("        Drop the four source images into that folder and "
              "re-run.", file=sys.stderr)
        return 2
    image_paths = image_paths[:args.num_images]

    ds_dir = args.results_dir / "downsampled"
    sr_dir = args.results_dir / "superresolved"
    rp_dir = args.results_dir / "reports"
    for d in (ds_dir, sr_dir, rp_dir):
        d.mkdir(parents=True, exist_ok=True)

    save_images = not args.no_save_images
    study = Study()
    image_names: list[str] = []

    for path in image_paths:
        name = path.stem
        image_names.append(name)
        original = load_grayscale_512(path)
        if save_images:
            cv2.imwrite(str(ds_dir / f"{name}_original_gray.png"), original)
        print(f"[info] processing {name} ({path.name})")
        study.run_image(name, original,
                        out_dir=sr_dir if save_images else ds_dir,
                        save_images=save_images)

    print_reports(study.rows, image_names)

    dump_csv(study.rows, rp_dir / "results.csv")
    dump_markdown(study.rows, image_names, rp_dir / "results.md")
    print(f"\n[info] wrote {rp_dir / 'results.csv'}")
    print(f"[info] wrote {rp_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
