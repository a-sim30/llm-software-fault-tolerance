"""
Assignment 2 - Interpolation-Based Super-Resolution
====================================================

Pipeline:
  1. Load four grayscale images (512 x 512).
  2. Downsample each to 64 x 64, 128 x 128, 256 x 256 with anti-aliased
     averaging (area interpolation) so we simulate a realistic low-res
     acquisition rather than a naive stride subsample.
  3. Super-resolve each downsampled image back to 512 x 512 with TWO
     separately-labelled families of interpolation methods:

       Family A - B-spline sweep (orders 0..5).
         A single uniform algorithm where the polynomial order is the only
         variable, so "does higher order help?" is a controlled experiment.

       Family B - convolution kernels (as shipped in production libraries).
         Nearest, bilinear, Keys bicubic (a = -0.75) and Lanczos-4.
         These are what real image software actually uses.

     Keeping the families apart matters: a Keys cubic kernel and a cubic
     B-spline are both "third order" but are different operators, so mixing
     them inside one order column makes the order axis uninterpretable.

  4. Verify the two families agree at orders 0 and 1, where they are the
     same mathematical operator. A mismatch there means the two code paths
     disagree about grid alignment, which silently corrupts every spline
     result - so the study checks its own geometry before collecting data.
  5. Compute PSNR and SSIM of every super-resolved image against the
     original 512 x 512 grayscale reference.
  6. Time each interpolation call (median of several runs to smooth noise).
  7. Print consolidated results as tables in the terminal and dump CSV /
     Markdown copies for the report.
"""

from __future__ import annotations

import argparse
import csv
import math
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

SPLINE = "Spline"
KERNEL = "Kernel"

# Two implementations of the same operator may differ by one grey level from
# integer vs float rounding; anything larger is a real geometry mismatch.
AGREEMENT_TOLERANCE = 1.0


@dataclass(frozen=True)
class Method:
    """One interpolation method along with a printable label and a callable."""
    family: str             # SPLINE or KERNEL
    order_label: str        # "0".."5", or "L" for Lanczos
    name: str               # e.g. "Bicubic (Keys, a=-0.75)"
    apply: Callable[[np.ndarray, int], np.ndarray]

    @property
    def key(self) -> tuple[str, str]:
        return (self.family, self.order_label)


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
        # grid_mode=True selects cell-based (half-pixel-centre) coordinates,
        # matching cv2.resize and the INTER_AREA downsample that built the
        # reference. The default grid_mode=False maps (N-1)/(out-1) instead,
        # which stretches the output by up to half a low-res pixel.
        return ndimage.zoom(img, zoom=zoom, order=order, mode="reflect",
                            prefilter=True, grid_mode=True)
    return _resize


def build_methods() -> list[Method]:
    """The interpolation catalogue used throughout the study."""
    return [
        # Family A - uniform B-spline sweep: order is the only variable.
        Method(SPLINE, "0", "B-spline order 0 (nearest)",   _spline_resize(0)),
        Method(SPLINE, "1", "B-spline order 1 (linear)",    _spline_resize(1)),
        Method(SPLINE, "2", "B-spline order 2 (quadratic)", _spline_resize(2)),
        Method(SPLINE, "3", "B-spline order 3 (cubic)",     _spline_resize(3)),
        Method(SPLINE, "4", "B-spline order 4 (quartic)",   _spline_resize(4)),
        Method(SPLINE, "5", "B-spline order 5 (quintic)",   _spline_resize(5)),
        # Family B - convolution kernels as shipped by OpenCV.
        Method(KERNEL, "0", "Nearest neighbour",
               _cv2_resize(cv2.INTER_NEAREST)),
        Method(KERNEL, "1", "Bilinear",
               _cv2_resize(cv2.INTER_LINEAR)),
        Method(KERNEL, "3", "Bicubic (Keys, a=-0.75)",
               _cv2_resize(cv2.INTER_CUBIC)),
        Method(KERNEL, "L", "Lanczos-4 (windowed sinc)",
               _cv2_resize(cv2.INTER_LANCZOS4)),
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


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Self-validation: the two families must agree where they coincide
# ---------------------------------------------------------------------------

@dataclass
class AgreementCheck:
    order: str
    input_size: int
    max_abs_diff: float

    @property
    def passed(self) -> bool:
        return self.max_abs_diff <= AGREEMENT_TOLERANCE


def verify_family_agreement(original: np.ndarray,
                            methods: list[Method]) -> list[AgreementCheck]:
    """
    Orders 0 and 1 are the same operator in both families (nearest is nearest;
    a first-order B-spline is bilinear), so their outputs must be identical up
    to rounding. This runs the real catalogue rather than a re-declaration, so
    it keeps testing whatever build_methods() actually returns.
    """
    by_key = {m.key: m for m in methods}
    checks: list[AgreementCheck] = []

    for order in ("0", "1"):
        spline = by_key.get((SPLINE, order))
        kernel = by_key.get((KERNEL, order))
        if spline is None or kernel is None:
            continue
        for lr_size in DOWNSAMPLE_SIZES:
            low_res = downsample(original, lr_size)
            a = to_uint8(spline.apply(low_res, ORIGINAL_SIZE)).astype(np.int16)
            b = to_uint8(kernel.apply(low_res, ORIGINAL_SIZE)).astype(np.int16)
            checks.append(AgreementCheck(order, lr_size,
                                         float(np.max(np.abs(a - b)))))
    return checks


def print_agreement(checks: list[AgreementCheck]) -> bool:
    headers = ["Order", "Input", "Spline vs Kernel (max |diff|)", "Result"]
    body = [[c.order, f"{c.input_size}x{c.input_size}",
             f"{c.max_abs_diff:.0f} grey levels",
             "PASS" if c.passed else "FAIL"]
            for c in checks]
    print("\nGeometry self-check - the two families implement the same "
          "operator at orders 0 and 1,")
    print(f"so their outputs must match within {AGREEMENT_TOLERANCE:.0f} grey "
          "level. A FAIL means the code paths")
    print("disagree about grid alignment and every spline number is suspect.")
    print(tabulate(body, headers=headers, tablefmt="fancy_grid",
                   stralign="left", numalign="right"))
    return all(c.passed for c in checks)


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
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    # An exact reconstruction gives MSE 0, so PSNR is legitimately infinite;
    # errstate keeps that from printing a divide-by-zero warning.
    with np.errstate(divide="ignore"):
        psnr_val = compute_psnr(ref, cand, data_range=255.0)
    ssim_val = compute_ssim(ref, cand, data_range=255.0)
    return float(psnr_val), float(ssim_val)


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

@dataclass
class Row:
    image: str
    family: str
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
                hr_u8 = to_uint8(hr)
                p, s = score(original, hr_u8)

                if save_images:
                    fam = m.family.lower()
                    cv2.imwrite(
                        str(out_dir / f"{name}_sr_{lr_size}to512_"
                                      f"{fam}_o{m.order_label}.png"),
                        hr_u8)

                self.rows.append(Row(
                    image=name,
                    family=m.family,
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

def _fmt_psnr(v: float) -> str:
    return "exact" if math.isinf(v) else f"{v:.2f}"


def _mean_cell(values: list[float], decimals: int) -> str:
    """Average a set of values, reporting exact reconstructions separately."""
    finite = [v for v in values if math.isfinite(v)]
    n_exact = len(values) - len(finite)
    if not finite:
        return f"exact ({n_exact}/{len(values)})"
    mean = sum(finite) / len(finite)
    if n_exact:
        return f"{mean:.{decimals}f} (+{n_exact} exact)"
    return f"{mean:.{decimals}f}"


def _ordered_keys(rows: list[Row]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        k = (r.family, r.order)
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _per_image_table(rows: list[Row], image: str) -> str:
    headers = ["Family", "Order", "Method", "Input", "Factor",
               "PSNR (dB)", "SSIM", "Runtime (ms)"]
    body = [
        [r.family, r.order, r.method, f"{r.input_size}x{r.input_size}",
         f"{r.factor}x", _fmt_psnr(r.psnr_db), f"{r.ssim:.4f}",
         f"{r.runtime_ms:.3f}"]
        for r in rows if r.image == image
    ]
    return tabulate(body, headers=headers, tablefmt="fancy_grid",
                    stralign="left", numalign="right")


def _metric_pivot(rows: list[Row], metric: str) -> str:
    """Rows = (family, order), columns = input size, averaged over images."""
    sizes = sorted({r.input_size for r in rows})
    label = {(r.family, r.order): r.method for r in rows}
    decimals = {"psnr_db": 2, "ssim": 4, "runtime_ms": 3}[metric]

    headers = ["Family", "Order", "Method"] + [f"{s}x{s} -> 512" for s in sizes]
    body: list[list[str]] = []
    for family, order in _ordered_keys(rows):
        line = [family, order, label[(family, order)]]
        for s in sizes:
            values = [getattr(r, metric) for r in rows
                      if r.family == family and r.order == order
                      and r.input_size == s]
            line.append(_mean_cell(values, decimals) if values else "-")
        body.append(line)

    return tabulate(body, headers=headers, tablefmt="fancy_grid",
                    stralign="left", numalign="right")


def _cross_family_table(rows: list[Row]) -> str:
    """
    Matched-order comparison between the families. Orders 0 and 1 are the same
    operator and must agree; order 3 is where a cubic B-spline and Keys cubic
    convolution genuinely differ, which is the interesting measurement.
    """
    sizes = sorted({r.input_size for r in rows})
    shared = sorted({o for (f, o) in _ordered_keys(rows) if f == SPLINE}
                    & {o for (f, o) in _ordered_keys(rows) if f == KERNEL})

    def mean_psnr(family: str, order: str, size: int) -> float:
        vals = [r.psnr_db for r in rows
                if r.family == family and r.order == order
                and r.input_size == size]
        finite = [v for v in vals if math.isfinite(v)]
        return sum(finite) / len(finite) if finite else math.inf

    headers = ["Order", "Input", "B-spline (dB)", "Kernel (dB)", "Delta",
               "Interpretation"]
    body: list[list[str]] = []
    for order in shared:
        for s in sizes:
            a, b = mean_psnr(SPLINE, order, s), mean_psnr(KERNEL, order, s)
            same_operator = order in ("0", "1")
            if math.isinf(a) and math.isinf(b):
                delta = "0.00"
            elif math.isinf(a) or math.isinf(b):
                delta = "n/a"
            else:
                delta = f"{b - a:+.2f}"
            note = ("same operator - must agree" if same_operator
                    else "cubic B-spline vs Keys cubic")
            body.append([order, f"{s}x{s}", _fmt_psnr(a), _fmt_psnr(b),
                         delta, note])

    return tabulate(body, headers=headers, tablefmt="fancy_grid",
                    stralign="left", numalign="right")


def print_reports(rows: list[Row], images: list[str]) -> None:
    banner = "=" * 78
    print("\n" + banner)
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
    print("Note: spline timings include SciPy's Python-level call overhead, "
          "so cross-family")
    print("runtime is indicative of real-world cost, not of kernel arithmetic "
          "alone.")

    print("\n\n" + banner)
    print(" CROSS-FAMILY COMPARISON AT MATCHED ORDER")
    print(banner)
    print(_cross_family_table(rows))


def dump_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "family", "input_size", "factor", "order",
                    "method", "psnr_db", "ssim", "runtime_ms"])
        for r in rows:
            w.writerow([r.image, r.family, r.input_size, r.factor, r.order,
                        r.method, f"{r.psnr_db:.6f}", f"{r.ssim:.6f}",
                        f"{r.runtime_ms:.6f}"])


def dump_markdown(rows: list[Row], images: list[str], path: Path) -> None:
    lines = ["# Assignment 2 - Interpolation-Based Super-Resolution", ""]
    for image in images:
        lines += [f"## {image}", "",
                  "| Family | Order | Method | Input | Factor | PSNR (dB) "
                  "| SSIM | Runtime (ms) |",
                  "|--------|-------|--------|-------|--------|-----------"
                  "|------|--------------|"]
        for r in rows:
            if r.image != image:
                continue
            lines.append(
                f"| {r.family} | {r.order} | {r.method} "
                f"| {r.input_size}x{r.input_size} | {r.factor}x "
                f"| {_fmt_psnr(r.psnr_db)} | {r.ssim:.4f} "
                f"| {r.runtime_ms:.3f} |")
        lines.append("")

    lines += ["## Averages across all images", ""]
    for metric, title in (("psnr_db", "PSNR (dB)"),
                          ("ssim", "SSIM"),
                          ("runtime_ms", "Runtime (ms)")):
        lines += [f"### {title}", "", _metric_pivot(rows, metric), ""]

    lines += ["## Cross-family comparison at matched order", "",
              _cross_family_table(rows), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _discover_images(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in image_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in exts)


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
    parser.add_argument("--skip-check", action="store_true",
                        help="Skip the cross-family geometry self-check.")
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

    if not args.skip_check:
        probe = load_grayscale_512(image_paths[0])
        if not print_agreement(verify_family_agreement(probe, study.methods)):
            print("\n[error] geometry self-check failed - the spline and "
                  "kernel paths disagree", file=sys.stderr)
            print("        at an order where they must match. Fix the "
                  "coordinate convention", file=sys.stderr)
            print("        before trusting any result.", file=sys.stderr)
            return 1

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
