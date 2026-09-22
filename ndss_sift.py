import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree
from skimage import data
from skimage.transform import AffineTransform, warp


#scale space
N_OCTAVES      = 4
N_SCALES       = 3
SIGMA_0        = 1.6
K_PERCENTILE   = 70.0
DT_MAX         = 0.28
OCTAVE_DOWNSAMPLE = "decimate"

#detection
RESPONSE_FLOOR = 0.0
CONTRAST_THR   = 1.0e-3
EDGE_RATIO_R   = 10.0
BORDER         = 6

#refinement
MAX_REFINE_ITER = 5
DEDUP_PIXELS    = 1.5
DEDUP_SCALE_RAT = 1.3

#orientation
ORI_BINS        = 36
ORI_RADIUS_FAC  = 4.0
ORI_PEAK_RATIO  = 0.8
ORI_SOFT_BINS   = True

#descriptor
COMPUTE_DESCRIPTORS = True
DESC_NBINS      = 4
DESC_NORI       = 8
DESC_CELL_FAC   = 3.0
DESC_WINDOW_FAC = 2.5
DESC_MAX_RADIUS = 48
USE_ROOT_NORM   = True

#figures and invariance study
SAVE_FIGURES         = True
RUN_BENCHMARK        = True
BENCH_ANGLES         = (15.0, 30.0, 45.0, 90.0)
BENCH_SCALES         = (1.5, 0.75)
BENCH_SHOW_ANGLE     = 30.0
MATCH_RATIO_THR      = 0.8
MATCH_MIN_SEP        = 4.0
MATCH_MUTUAL         = True
MATCH_PIXEL_TOL      = 3.0
REPEAT_PIXEL_TOL     = 2.5
REPEAT_SCALE_TOL     = 1.5
COMMON_REGION_MARGIN = 24

RNG_SEED = 0
np.random.seed(RNG_SEED)


#replicate-pad by one pixel so that derivatives keep the image size
def pad_edge(L):
    return np.pad(L, 1, mode="edge")

#dL/dx  with the central difference [-1, 0, 1] / 2
def d_x(L):
    P = pad_edge(L)
    return 0.5 * (P[1:-1, 2:] - P[1:-1, :-2])

#dL/dy  with the central difference [-1, 0, 1]^T / 2
def d_y(L):
    P = pad_edge(L)
    return 0.5 * (P[2:, 1:-1] - P[:-2, 1:-1])

#d2L/dx2 with [1, -2, 1]
def d_xx(L):
    P = pad_edge(L)
    return P[1:-1, 2:] - 2.0 * L + P[1:-1, :-2]

#d2L/dy2 with [1, -2, 1]^T
def d_yy(L):
    P = pad_edge(L)
    return P[2:, 1:-1] - 2.0 * L + P[:-2, 1:-1]

#d2L/dxdy with the 4-point cross difference
def d_xy(L):
    P = pad_edge(L)
    return 0.25 * (P[2:, 2:] - P[2:, :-2] - P[:-2, 2:] + P[:-2, :-2])

def gradient_magnitude(L):
    gx, gy = d_x(L), d_y(L)
    return np.sqrt(gx * gx + gy * gy)



#P-M contrast parameter k: gradient percentile threshold
def estimate_contrast_factor(L, percentile=K_PERCENTILE):
    m = gradient_magnitude(L)
    pos = m[m > 0]
    if pos.size == 0:
        return 1e-3
    return float(max(np.percentile(pos, percentile), 1e-6))


#isotropic 9-point stencil, weights 2/3 axial and 1/6 diagonal
#with g == 1 it reduces to (1/6)[[1,4,1],[4,-20,4],[1,4,1]], whose leading
#anisotropy error cancels unlike the plain 4-neighbour laplacian
#weights sum to 10/3 so the explicit scheme is stable for dt <= 3/10
STENCIL = (( 0, +1, 2.0 / 3.0), ( 0, -1, 2.0 / 3.0),
           (+1,  0, 2.0 / 3.0), (-1,  0, 2.0 / 3.0),
           (+1, +1, 1.0 / 6.0), (+1, -1, 1.0 / 6.0),
           (-1, +1, 1.0 / 6.0), (-1, -1, 1.0 / 6.0))


#neighbour plane (dy, dx) of a 1-pixel edge-padded array
def shift_plane(P, dy, dx):
    return P[1 + dy:P.shape[0] - 1 + dy, 1 + dx:P.shape[1] - 1 + dx]


#one Perona-Malik step using the isotropic 9-point stencil
#half-pixel conductivities (g_p + g_c)/2 keep the scheme conservative
def diffusion_step(L, k, dt):
    gx, gy = d_x(L), d_y(L)
    g = 1.0 / (1.0 + (gx * gx + gy * gy) / (k * k))

    Lp = pad_edge(L)
    gp = pad_edge(g)

    div = np.zeros_like(L)
    for dy, dx, w in STENCIL:
        div += w * (0.5 * (shift_plane(gp, dy, dx) + g)) * (shift_plane(Lp, dy, dx) - L)
    return L + dt * div


#evolve L for a diffusion time delta_t with stable explicit steps
def diffuse(L, delta_t, k, dt_max=DT_MAX):
    if delta_t <= 1e-12:
        return L.copy()
    n_steps = int(np.ceil(delta_t / dt_max))
    dt = delta_t / n_steps
    out = L
    for _ in range(n_steps):
        out = diffusion_step(out, k, dt)
    return out


#scale-normalised Hessian determinant and curvature ratio
#R = sigma^4 det(H) replaces the DoG: blobs become maxima, edges give det <= 0
def scale_normalised_hessian_response(L, sigma):
    Lxx, Lyy, Lxy = d_xx(L), d_yy(L), d_xy(L)
    det = Lxx * Lyy - Lxy * Lxy
    tr = Lxx + Lyy
    response = (sigma ** 4) * det
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(det > 0, tr * tr / np.where(det > 0, det, 1.0), np.inf)
    return response, ratio


#build nonlinear-diffusion scale space and Hessian responses
def build_scale_space(image,
                      n_octaves=N_OCTAVES,
                      n_scales=N_SCALES,
                      sigma_0=SIGMA_0):
    n_levels = n_scales + 3
    sigmas = np.array([sigma_0 * (2.0 ** (i / float(n_scales)))
                       for i in range(n_levels)])

    octaves = []
    base = image.astype(np.float64)

    for o in range(n_octaves):
        if min(base.shape) < 2 * BORDER + 8:
            break

        k = estimate_contrast_factor(base)

        #octave 0 uses raw image, later octaves use decimated previous level S
        t_prev = 0.0 if o == 0 else 0.5 * sigma_0 ** 2

        levels = []
        L = base
        for i in range(n_levels):
            t_i = 0.5 * sigmas[i] ** 2
            L = diffuse(L, t_i - t_prev, k)
            t_prev = t_i
            levels.append(L)

        resp = np.empty((n_levels,) + base.shape, dtype=np.float64)
        rat = np.empty_like(resp)
        for i in range(n_levels):
            resp[i], rat[i] = scale_normalised_hessian_response(levels[i],
                                                                sigmas[i])

        octaves.append({"levels": levels, "sigmas": sigmas,
                        "response": resp, "ratio": rat,
                        "octave": o, "k": k})

        #next octave: no anti-alias filter
        base = downsample(levels[n_scales])

    return octaves


#halve the resolution without any gaussian pre-blur
#"decimate" is plain sub-sampling, grid-exact and entirely filter-free
#"box3" applies one 3x3 box (uniform) average first, centred on the retained
#samples so the sampling grid does not shift by half a pixel
def downsample(L, method=None):
    method = OCTAVE_DOWNSAMPLE if method is None else method
    if method == "box3":
        P = pad_edge(L)
        acc = np.zeros_like(L)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += shift_plane(P, dy, dx)
        L = acc / 9.0
    return np.ascontiguousarray(L[::2, ::2])



#all strict local maxima of the response volume over (x, y, scale)
def detect_extrema(octaves, floor=RESPONSE_FLOOR, border=BORDER):
    #footprint of the 26 neighbours without the centre, so the test is a
    #strict maximum and plateaus no longer yield duplicate keypoints
    footprint = np.ones((3, 3, 3), dtype=bool)
    footprint[1, 1, 1] = False

    raw = []
    for oct_data in octaves:
        R = oct_data["response"]
        n_lev, H, W = R.shape

        nbr_max = maximum_filter(R, footprint=footprint, mode="nearest")
        is_max = (R > nbr_max) & (R > floor)

        #first and last level have no neighbour in scale
        is_max[0] = False
        is_max[-1] = False
        is_max[:, :border, :] = False
        is_max[:, H - border:, :] = False
        is_max[:, :, :border] = False
        is_max[:, :, W - border:] = False

        s_idx, y_idx, x_idx = np.nonzero(is_max)
        for s, y, x in zip(s_idx, y_idx, x_idx):
            raw.append({"octave": oct_data["octave"],
                        "s": int(s), "y": int(y), "x": int(x),
                        "sigma_oct": float(oct_data["sigmas"][s]),
                        "response": float(R[s, y, x]),
                        "ratio": float(oct_data["ratio"][s, y, x])})
    return raw



#filter low-contrast and edge-like unstable responses
def filter_extrema(raw, contrast_thr=CONTRAST_THR, edge_r=EDGE_RATIO_R):
    limit = (edge_r + 1.0) ** 2 / edge_r
    return [kp for kp in raw
            if kp["response"] >= contrast_thr and kp["ratio"] < limit]


#gradient and hessian of the response volume at an integer sample
def local_grad_hess(R, s, y, x):
    dx = 0.5 * (R[s, y, x + 1] - R[s, y, x - 1])
    dy = 0.5 * (R[s, y + 1, x] - R[s, y - 1, x])
    ds = 0.5 * (R[s + 1, y, x] - R[s - 1, y, x])

    c = R[s, y, x]
    dxx = R[s, y, x + 1] - 2.0 * c + R[s, y, x - 1]
    dyy = R[s, y + 1, x] - 2.0 * c + R[s, y - 1, x]
    dss = R[s + 1, y, x] - 2.0 * c + R[s - 1, y, x]

    dxy = 0.25 * (R[s, y + 1, x + 1] - R[s, y + 1, x - 1] -
                  R[s, y - 1, x + 1] + R[s, y - 1, x - 1])
    dxs = 0.25 * (R[s + 1, y, x + 1] - R[s + 1, y, x - 1] -
                  R[s - 1, y, x + 1] + R[s - 1, y, x - 1])
    dys = 0.25 * (R[s + 1, y + 1, x] - R[s + 1, y - 1, x] -
                  R[s - 1, y + 1, x] + R[s - 1, y - 1, x])

    grad = np.array([dx, dy, ds])
    hess = np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])
    return grad, hess


#fit 3-D quadratic for sub-pixel keypoint refinement
def refine_keypoints(octaves, kps,
                     contrast_thr=CONTRAST_THR,
                     edge_r=EDGE_RATIO_R,
                     border=BORDER,
                     max_iter=MAX_REFINE_ITER):
    limit = (edge_r + 1.0) ** 2 / edge_r
    by_octave = {od["octave"]: od for od in octaves}
    refined = []

    for kp in kps:
        od = by_octave[kp["octave"]]
        R = od["response"]
        n_lev, H, W = R.shape
        s, y, x = kp["s"], kp["y"], kp["x"]

        converged = False
        grad = off = None
        for _ in range(max_iter):
            grad, hess = local_grad_hess(R, s, y, x)
            try:
                off = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                off = None
                break
            if not np.all(np.isfinite(off)):
                off = None
                break
            if np.all(np.abs(off) < 0.5):
                converged = True
                break
            s += int(round(float(off[2])))
            y += int(round(float(off[1])))
            x += int(round(float(off[0])))
            if not (1 <= s <= n_lev - 2 and
                    border <= x < W - border and
                    border <= y < H - border):
                break

        if not converged or off is None:
            continue

        value = float(R[s, y, x] + 0.5 * float(np.dot(grad, off)))
        if value < contrast_thr:
            continue
        if not (od["ratio"][s, y, x] < limit):
            continue

        o = od["octave"]
        s_ref = s + float(off[2])
        x_ref = x + float(off[0])
        y_ref = y + float(off[1])
        sigma_oct = SIGMA_0 * (2.0 ** (s_ref / float(N_SCALES)))

        refined.append({"octave": o,
                        "level": s,              #nearest integer level
                        "x_oct": x_ref, "y_oct": y_ref,
                        "sigma_oct": sigma_oct,
                        "x": x_ref * (2 ** o),   #image coordinates
                        "y": y_ref * (2 ** o),
                        "sigma": sigma_oct * (2 ** o),
                        "response": value})
    return refined


#merge near-identical detections of the same structure
#duplicates carry almost identical descriptors, which defeats the ratio test
#used at matching time, so the strongest of each cluster is kept
def deduplicate(kps, pix_tol=DEDUP_PIXELS, scale_ratio=DEDUP_SCALE_RAT):
    if not kps:
        return kps
    cell = max(pix_tol, 1e-6)
    accepted, buckets = [], {}
    for kp in sorted(kps, key=lambda p: -p["response"]):
        cx, cy = int(kp["x"] / cell), int(kp["y"] / cell)
        duplicate = False
        for jy in (-1, 0, 1):
            for jx in (-1, 0, 1):
                for other in buckets.get((cx + jx, cy + jy), ()):
                    if ((kp["x"] - other["x"]) ** 2 +
                            (kp["y"] - other["y"]) ** 2) > pix_tol ** 2:
                        continue
                    r = kp["sigma"] / other["sigma"]
                    if 1.0 / scale_ratio < r < scale_ratio:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            accepted.append(kp)
            buckets.setdefault((cx, cy), []).append(kp)
    return accepted



#circular triangular (Bartlett) smoothing [1 2 3 2 1] / 9
def smooth_hist_triangular(h):
    return (1.0 * np.roll(h, -2) + 2.0 * np.roll(h, -1) + 3.0 * h +
            2.0 * np.roll(h, 1) + 1.0 * np.roll(h, 2)) / 9.0


def orientation_histogram(mag, ang, x, y, radius, n_bins=ORI_BINS):
    H, W = mag.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - radius), min(W, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(H, yi + radius + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    sub_m = mag[y0:y1, x0:x1]
    sub_a = ang[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    #triangular radial window instead of the gaussian window of SIFT
    w = np.clip(1.0 - dist / float(radius), 0.0, None)
    weight = (sub_m * w).ravel()

    pos = sub_a.ravel() * (n_bins / (2.0 * np.pi))      #continuous bin index
    if ORI_SOFT_BINS:
        #split every sample linearly over its two neighbouring bins, otherwise
        #the 10-degree quantisation alone costs several degrees of orientation
        #error and therefore descriptor mismatches
        b0 = np.floor(pos - 0.5).astype(np.int64)
        frac = pos - 0.5 - b0
        idx = np.concatenate([np.mod(b0, n_bins), np.mod(b0 + 1, n_bins)])
        wts = np.concatenate([weight * (1.0 - frac), weight * frac])
        hist = np.bincount(idx, weights=wts, minlength=n_bins)
    else:
        bins = np.mod(np.floor(pos).astype(np.int64), n_bins)
        hist = np.bincount(bins, weights=weight, minlength=n_bins)
    return smooth_hist_triangular(hist)


#compute dominant orientations and optional 128-D descriptors
def assign_orientations(octaves, kps,
                        n_bins=ORI_BINS,
                        radius_fac=ORI_RADIUS_FAC,
                        peak_ratio=ORI_PEAK_RATIO,
                        compute_descriptors=COMPUTE_DESCRIPTORS):
    out = []
    for od in octaves:
        group = [kp for kp in kps if kp["octave"] == od["octave"]]
        if not group:
            continue

        mags, angs = [], []
        for L in od["levels"]:
            gx, gy = d_x(L), d_y(L)
            mags.append(np.sqrt(gx * gx + gy * gy))
            angs.append(np.mod(np.arctan2(gy, gx), 2.0 * np.pi))

        for kp in group:
            lev = kp["level"]
            mag, ang = mags[lev], angs[lev]
            radius = max(3, int(round(radius_fac * kp["sigma_oct"])))
            hist = orientation_histogram(mag, ang, kp["x_oct"], kp["y_oct"],
                                         radius, n_bins)
            if hist is None:
                continue
            hmax = float(hist.max())
            if hmax <= 0.0:
                continue

            for b in range(n_bins):
                left = hist[(b - 1) % n_bins]
                right = hist[(b + 1) % n_bins]
                if not (hist[b] >= left and hist[b] >= right):
                    continue                      #keep local peaks only
                if hist[b] < peak_ratio * hmax:
                    continue

                denom = left - 2.0 * hist[b] + right
                shift = 0.0 if abs(denom) < 1e-20 else 0.5 * (left - right) / denom
                shift = float(np.clip(shift, -0.5, 0.5))
                theta = 2.0 * np.pi * ((b + 0.5 + shift) % n_bins) / n_bins

                new_kp = dict(kp)
                new_kp["theta"] = float(theta)
                if compute_descriptors:
                    new_kp["descriptor"] = compute_descriptor(
                        mag, ang, kp["x_oct"], kp["y_oct"],
                        kp["sigma_oct"], theta)
                out.append(new_kp)
    return out


#compute rotation-normalised 128-D gradient descriptor
def compute_descriptor(mag, ang, x, y, sigma, theta,
                       n_bins=DESC_NBINS, n_ori=DESC_NORI,
                       cell_fac=DESC_CELL_FAC):
    H, W = mag.shape
    cell = cell_fac * sigma                      #pixels per descriptor cell
    half = 0.5 * cell * n_bins                   #half width of the patch
    radius = int(min(DESC_MAX_RADIUS, np.ceil(half * np.sqrt(2.0))))
    radius = max(radius, 2)

    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - radius), min(W, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(H, yi + radius + 1)
    desc = np.zeros(n_bins * n_bins * n_ori, dtype=np.float64)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return desc

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - x
    dy = yy - y

    ct, st = np.cos(theta), np.sin(theta)
    u = (dx * ct + dy * st) / cell               #rotate by -theta
    v = (-dx * st + dy * ct) / cell

    ub = u + n_bins / 2.0 - 0.5                  #continuous cell coordinates
    vb = v + n_bins / 2.0 - 0.5

    #triangular envelope with support 2.5x the patch half-width, so the outer
    #descriptor cells keep a weight comparable to the one SIFT's gaussian gives
    #them; a cone dying at the patch border throws the corner cells away
    dist = np.sqrt(dx * dx + dy * dy)
    w_rad = np.clip(1.0 - dist / (DESC_WINDOW_FAC * half), 0.0, None)

    a = np.mod(ang[y0:y1, x0:x1] - theta, 2.0 * np.pi)
    ob = a * (n_ori / (2.0 * np.pi))
    weight = mag[y0:y1, x0:x1] * w_rad

    valid = (ub > -1.0) & (ub < n_bins) & (vb > -1.0) & (vb < n_bins) & (weight > 0)
    if not np.any(valid):
        return desc

    ub, vb, ob, weight = ub[valid], vb[valid], ob[valid], weight[valid]

    u0 = np.floor(ub).astype(np.int64)
    v0 = np.floor(vb).astype(np.int64)
    o0 = np.floor(ob).astype(np.int64)
    fu, fv, fo = ub - u0, vb - v0, ob - o0

    #trilinear interpolation into a padded (n_bins+2)^2 x n_ori histogram
    pad = n_bins + 2
    idx_list, w_list = [], []
    for di in (0, 1):
        wu = fu if di else (1.0 - fu)
        ui = u0 + di + 1                         #+1 for the padding
        for dj in (0, 1):
            wv = fv if dj else (1.0 - fv)
            vi = v0 + dj + 1
            for dk in (0, 1):
                wo = fo if dk else (1.0 - fo)
                oi = np.mod(o0 + dk, n_ori)
                idx_list.append((vi * pad + ui) * n_ori + oi)
                w_list.append(weight * wu * wv * wo)

    flat_idx = np.concatenate(idx_list)
    flat_w = np.concatenate(w_list)
    hist = np.bincount(flat_idx, weights=flat_w,
                       minlength=pad * pad * n_ori).reshape(pad, pad, n_ori)

    desc = hist[1:1 + n_bins, 1:1 + n_bins, :].ravel()

    norm = np.linalg.norm(desc)
    if norm > 1e-12:
        desc /= norm
    np.clip(desc, 0.0, 0.2, out=desc)            #suppress illumination spikes
    norm = np.linalg.norm(desc)
    if norm > 1e-12:
        desc /= norm

    if USE_ROOT_NORM:
        #RootSIFT: L1-normalise then take the square root, so the euclidean
        #distance becomes the Hellinger distance between the raw histograms
        #the result is already L2 normalised, so matching is unchanged
        s = desc.sum()
        if s > 1e-12:
            desc = np.sqrt(desc / s)
    return desc


#main pipeline

#run the complete Gaussian-free SIFT pipeline on a [0,1] float image
def ndss_sift(image, verbose=True):
    timings = {}

    t0 = time.time()
    octaves = build_scale_space(image)
    timings["scale space"] = time.time() - t0

    t0 = time.time()
    raw = detect_extrema(octaves)
    timings["extrema"] = time.time() - t0

    t0 = time.time()
    filtered = filter_extrema(raw)
    timings["filtering"] = time.time() - t0

    t0 = time.time()
    refined = deduplicate(refine_keypoints(octaves, filtered))
    timings["refinement"] = time.time() - t0

    t0 = time.time()
    oriented = assign_orientations(octaves, refined)
    timings["orientation + descriptor"] = time.time() - t0

    if verbose:
        print("  raw scale-space extrema      : %d" % len(raw))
        print("  after weak/edge rejection    : %d" % len(filtered))
        print("  after sub-pixel localisation : %d" % len(refined))
        print("  final oriented keypoints     : %d" % len(oriented))
        for key, val in timings.items():
            print("    [%-24s] %6.2f s" % (key, val))

    return {"octaves": octaves, "raw": raw, "filtered": filtered,
            "refined": refined, "oriented": oriented, "timings": timings}



def show(fig, name):
    if SAVE_FIGURES:
        fig.savefig(name, dpi=130, bbox_inches="tight")
    plt.show()


def figure_1_original(image):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Figure 1 - Original Cameraman (%d x %d)" % image.shape)
    ax.axis("off")
    show(fig, "fig1_original.png")


def figure_2_scale_space(octaves):
    wanted = [(0, 1), (0, 4), (1, 3), (2, 1), (2, 4), (3, 3)]
    panels = []
    for o, i in wanted:
        if o < len(octaves) and i < len(octaves[o]["sigmas"]):
            panels.append((o, i))
    if not panels:
        panels = [(0, i) for i in range(len(octaves[0]["sigmas"]))]

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(2.55 * n, 5.6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (o, i) in enumerate(panels):
        od = octaves[o]
        sigma_oct = od["sigmas"][i]
        sigma_img = sigma_oct * (2 ** o)
        L = od["levels"][i]
        R = od["response"][i]

        axes[0, col].imshow(L, cmap="gray")
        axes[0, col].set_title(r"$\sigma$=%.2f px  (oct %d, lvl %d)"
                               "\n%d$\\times$%d" %
                               (sigma_img, o, i, L.shape[0], L.shape[1]),
                               fontsize=9)
        axes[0, col].axis("off")

        vmax = np.percentile(np.abs(R), 99.5)
        vmax = vmax if vmax > 0 else 1.0
        axes[1, col].imshow(R, cmap="inferno", vmin=0.0, vmax=vmax)
        axes[1, col].set_title(r"response $\sigma^4\,\det H$" "\n"
                               r"$\sigma$=%.2f px" % sigma_img, fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle("Figure 2 - Nonlinear-diffusion scale space (top) and "
                 "scale-normalised det(H) responses (bottom)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    show(fig, "fig2_scale_space.png")


def figure_3_raw_extrema(image, raw):
    #raw detections are still in octave coordinates, lift to image coords
    xs = [kp["x"] * (2 ** kp["octave"]) for kp in raw]
    ys = [kp["y"] * (2 ** kp["octave"]) for kp in raw]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.scatter(xs, ys, s=4, c="lime", marker=".", linewidths=0)
    ax.set_title("Figure 3 - Raw scale-space extrema (before rejection / "
                 "refinement)\ntotal detected extrema = %d" % len(raw))
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")
    show(fig, "fig3_raw_extrema.png")


def figure_4_filtered(image, filtered):
    xs = np.array([kp["x"] * (2 ** kp["octave"]) for kp in filtered])
    ys = np.array([kp["y"] * (2 ** kp["octave"]) for kp in filtered])
    sg = np.array([kp["sigma_oct"] * (2 ** kp["octave"]) for kp in filtered])

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if len(xs):
        ax.scatter(xs, ys, s=(1.6 * sg) ** 2, facecolors="none",
                   edgecolors="yellow", linewidths=0.8)
    ax.set_title("Figure 4 - Keypoints after weak-response and edge "
                 "rejection\nkept = %d   (marker size $\\propto$ scale)"
                 % len(filtered))
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")
    show(fig, "fig4_filtered.png")


def figure_5_localized(image, refined):
    xs = np.array([kp["x"] for kp in refined])
    ys = np.array([kp["y"] for kp in refined])
    sg = np.array([kp["sigma"] for kp in refined])

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if len(xs):
        ax.scatter(xs, ys, s=(1.6 * sg) ** 2, facecolors="none",
                   edgecolors="cyan", linewidths=0.8)
        ax.scatter(xs, ys, s=2, c="red", marker=".", linewidths=0)
    ax.set_title("Figure 5 - Sub-pixel / sub-scale localised keypoints\n"
                 "kept = %d   (marker size $\\propto$ refined scale)"
                 % len(refined))
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")
    show(fig, "fig5_localized.png")


def figure_6_oriented(image, oriented):
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    circles, segments = [], []
    for kp in oriented:
        x, y, s, th = kp["x"], kp["y"], kp["sigma"], kp["theta"]
        r = 2.0 * s                              #circle radius = 2 sigma
        circles.append(Circle((x, y), r))
        segments.append([(x, y), (x + r * np.cos(th), y + r * np.sin(th))])

    if circles:
        #collections instead of individual artists same picture, much faster
        ax.add_collection(PatchCollection(circles, facecolors="none",
                                          edgecolors="deepskyblue",
                                          linewidths=0.6, alpha=0.9))
        ax.add_collection(LineCollection(segments, colors="red",
                                         linewidths=0.8, alpha=0.95))

    ax.set_title("Figure 6 - Final oriented keypoints (every keypoint shows "
                 "its orientation)\ntotal = %d oriented keypoints" % len(oriented))
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")
    show(fig, "fig6_oriented.png")



#invariance study
#a warp with a known ground-truth transform is applied, the whole pipeline is
#re-run on the warped image, and two numbers are reported: repeatability (a
#purely geometric detector score) and the matching score of the descriptors
#only the common region is used, so the black corners a rotation creates are
#excluded together with a margin for the keypoint support
#all warping is bilinear with no anti-aliasing, since skimage's is gaussian

#rotation about the image centre, output keeps the original size
def rotation_transform(shape, angle_deg):
    H, W = shape
    c = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    tform = (AffineTransform(translation=-c) +
             AffineTransform(rotation=np.deg2rad(angle_deg)) +
             AffineTransform(translation=c))
    return tform, (H, W), 1.0


#pure zoom, the output canvas follows the zoom factor
def scale_transform(shape, factor):
    H, W = shape
    tform = AffineTransform(scale=(factor, factor))
    return tform, (int(round(H * factor)), int(round(W * factor))), factor


#warp the image and return it together with its validity mask
def warp_image(image, tform, out_shape):
    warped = warp(image, tform.inverse, order=1, mode="constant", cval=0.0,
                  output_shape=out_shape, preserve_range=True)
    valid = warp(np.ones_like(image), tform.inverse, order=0, mode="constant",
                 cval=0.0, output_shape=out_shape, preserve_range=True)
    return warped, valid > 0.5


#erode the validity mask so a keypoint's whole support stays inside
def common_region(valid, margin=COMMON_REGION_MARGIN):
    return minimum_filter(valid.astype(np.uint8), size=int(2 * margin + 1),
                          mode="constant", cval=0) > 0


#boolean test of (x, y) points against a mask
def inside_mask(mask, pts):
    H, W = mask.shape
    xi = np.rint(pts[:, 0]).astype(np.int64)
    yi = np.rint(pts[:, 1]).astype(np.int64)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    out = np.zeros(len(pts), dtype=bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


#Lowe ratio test made duplicate-proof, plus a mutual-consistency check
#the plain ratio test breaks down when the second nearest neighbour is the same
#physical point detected twice, so it must lie at least min_sep pixels away
def match_descriptors(da, db, pb,
                      ratio_thr=MATCH_RATIO_THR,
                      min_sep=MATCH_MIN_SEP,
                      mutual=MATCH_MUTUAL):
    #descriptors are L2 normalised  ->  d^2 = 2 - 2 <a,b>
    d2 = np.maximum(2.0 - 2.0 * (da @ db.T), 0.0)
    order = np.argsort(d2, axis=1)
    n_a = d2.shape[0]

    best = order[:, 0]
    second = np.copy(best)
    for i in range(n_a):
        p0 = pb[best[i]]
        for j in order[i, 1:]:
            if np.hypot(*(pb[j] - p0)) >= min_sep:
                second[i] = j
                break

    rows = np.arange(n_a)
    denom = np.maximum(d2[rows, second], 1e-12)
    ratio = np.sqrt(d2[rows, best] / denom)
    accepted = (ratio < ratio_thr) & (second != best)

    if mutual:
        accepted &= (np.argmin(d2, axis=0)[best] == rows)
    return best, accepted


#run the full pipeline on a warped copy and score detector + descriptor
def evaluate_transform(image, ref, tform, out_shape, scale_factor, label,
                       keep_figure_data=False):
    warped, valid = warp_image(image, tform, out_shape)
    region = common_region(valid)
    res = ndss_sift(warped, verbose=False)

    if not ref["refined"] or not res["refined"]:
        return None

    #detector repeatability
    pa = np.array([[kp["x"], kp["y"]] for kp in ref["refined"]])
    sa = np.array([kp["sigma"] for kp in ref["refined"]])
    pb = np.array([[kp["x"], kp["y"]] for kp in res["refined"]])
    sb = np.array([kp["sigma"] for kp in res["refined"]])

    pa_m = tform(pa)
    keep_a = inside_mask(region, pa_m)
    keep_b = inside_mask(region, pb)
    n_a, n_b = int(keep_a.sum()), int(keep_b.sum())
    if n_a == 0 or n_b == 0:
        return None

    tree = cKDTree(pb)
    n_corr = 0
    for p, s in zip(pa_m[keep_a], sa[keep_a] * scale_factor):
        for j in tree.query_ball_point(p, REPEAT_PIXEL_TOL):
            if 1.0 / REPEAT_SCALE_TOL < sb[j] / s < REPEAT_SCALE_TOL:
                n_corr += 1
                break
    repeatability = n_corr / float(min(n_a, n_b))

    #descriptor matching
    stats = {"label": label, "n_a": n_a, "n_b": n_b, "n_corr": n_corr,
             "repeatability": repeatability, "scale_factor": scale_factor,
             "n_matches": 0, "n_correct": 0, "precision": 0.0,
             "matching_score": 0.0}

    if COMPUTE_DESCRIPTORS and ref["oriented"] and res["oriented"]:
        qa = np.array([[kp["x"], kp["y"]] for kp in ref["oriented"]])
        qb = np.array([[kp["x"], kp["y"]] for kp in res["oriented"]])
        qa_m = tform(qa)
        ka = inside_mask(region, qa_m)
        kb = inside_mask(region, qb)
        if ka.sum() and kb.sum():
            da = np.array([kp["descriptor"] for kp in ref["oriented"]])[ka]
            db = np.array([kp["descriptor"] for kp in res["oriented"]])[kb]
            qa_m, qb = qa_m[ka], qb[kb]

            best, accepted = match_descriptors(da, db, qb)
            err = np.linalg.norm(qb[best] - qa_m, axis=1)
            correct = accepted & (err < MATCH_PIXEL_TOL)

            stats["n_matches"] = int(accepted.sum())
            stats["n_correct"] = int(correct.sum())
            stats["precision"] = float(correct.sum() /
                                       max(accepted.sum(), 1))
            stats["matching_score"] = float(correct.sum() /
                                            min(len(da), len(db)))
            if keep_figure_data:
                stats["figure"] = {"warped": warped,
                                   "src": qa[ka][correct],
                                   "dst": qb[best[correct]]}
    return stats


#side-by-side view of the correctly matched keypoints
def figure_7_match_example(image, stats):
    fd = stats.get("figure")
    if fd is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].scatter(fd["src"][:, 0], fd["src"][:, 1], s=10, c="lime")
    axes[0].set_title("original - %d correctly matched keypoints"
                      % stats["n_correct"])
    axes[0].axis("off")
    axes[1].imshow(fd["warped"], cmap="gray", vmin=0, vmax=1)
    axes[1].scatter(fd["dst"][:, 0], fd["dst"][:, 1], s=10, c="lime")
    axes[1].set_title("%s - the same keypoints" % stats["label"])
    axes[1].axis("off")
    fig.suptitle("Figure 7 (extra) - %s : repeatability %.1f %%, "
                 "%d/%d accepted matches correct (%.1f %%)"
                 % (stats["label"], 100.0 * stats["repeatability"],
                    stats["n_correct"], stats["n_matches"],
                    100.0 * stats["precision"]))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    show(fig, "fig7_match_example.png")


#repeatability and precision curves against rotation and against zoom
def figure_8_invariance(rot_stats, scale_stats):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    if rot_stats:
        ang = [s["angle"] for s in rot_stats]
        axes[0].plot(ang, [100 * s["repeatability"] for s in rot_stats],
                     "o-", label="repeatability")
        axes[0].plot(ang, [100 * s["precision"] for s in rot_stats],
                     "s--", label="match precision")
        axes[0].plot(ang, [100 * s["matching_score"] for s in rot_stats],
                     "^:", label="matching score")
        axes[0].set_xlabel("rotation angle [deg]")
    axes[0].set_ylabel("[%]")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("rotation invariance")

    if scale_stats:
        fac = [s["scale_factor"] for s in scale_stats]
        axes[1].plot(fac, [100 * s["repeatability"] for s in scale_stats],
                     "o-", label="repeatability")
        axes[1].plot(fac, [100 * s["precision"] for s in scale_stats],
                     "s--", label="match precision")
        axes[1].plot(fac, [100 * s["matching_score"] for s in scale_stats],
                     "^:", label="matching score")
        axes[1].set_xlabel("zoom factor")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_title("scale invariance")

    fig.suptitle("Figure 8 (extra) - invariance of NDSS-SIFT")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    show(fig, "fig8_invariance.png")


#rotation and scale study, prints a table and draws Figures 7 and 8
def run_benchmark(image, ref):
    rot_stats, scale_stats, show_me = [], [], None

    for angle in BENCH_ANGLES:
        tform, shape, factor = rotation_transform(image.shape, angle)
        want = abs(angle - BENCH_SHOW_ANGLE) < 1e-9
        st = evaluate_transform(image, ref, tform, shape, factor,
                                "rotated %.0f deg" % angle,
                                keep_figure_data=want)
        if st is None:
            continue
        st["angle"] = angle
        rot_stats.append(st)
        if want:
            show_me = st

    #zooming out is the harder direction: no anti-alias filter is applied and
    #structures smaller than sigma_0 disappear, since the input image is never
    #up-sampled the way Lowe's SIFT does it
    for factor in BENCH_SCALES:
        tform, shape, sf = scale_transform(image.shape, factor)
        st = evaluate_transform(image, ref, tform, shape, sf,
                                "zoom x%.2f" % factor)
        if st is not None:
            scale_stats.append(st)

    rows = rot_stats + scale_stats
    if rows:
        print("    %-16s %7s %7s %8s %8s %9s %9s" %
              ("test", "kp(A)", "kp(B)", "repeat", "matches", "precision",
               "m-score"))
        for s in rows:
            print("    %-16s %7d %7d %7.1f%% %8d %8.1f%% %8.1f%%" %
                  (s["label"], s["n_a"], s["n_b"],
                   100 * s["repeatability"], s["n_matches"],
                   100 * s["precision"], 100 * s["matching_score"]))

    if show_me is not None:
        figure_7_match_example(image, show_me)
    figure_8_invariance(rot_stats, scale_stats)
    return rows



def main():
    t_start = time.time()

    print("=" * 74)
    print(" NDSS-SIFT : Gaussian-free SIFT via nonlinear diffusion scale space")
    print("=" * 74)

    #mandatory test image
    raw_img = data.camera()
    assert raw_img.shape == (512, 512), "unexpected cameraman size"
    image = raw_img.astype(np.float64) / 255.0
    print("\n[0] image : skimage.data.camera(), %dx%d, range [%.2f, %.2f]"
          % (image.shape[0], image.shape[1], image.min(), image.max()))

    figure_1_original(image)

    #pipeline
    print("\n[1-5] running the pipeline ...")
    res = ndss_sift(image, verbose=True)

    octaves = res["octaves"]
    print("\n      octaves built            : %d" % len(octaves))
    print("      levels per octave        : %d  (S = %d)"
          % (len(octaves[0]["sigmas"]), N_SCALES))
    print("      octave-local sigmas      : %s"
          % np.array2string(octaves[0]["sigmas"], precision=3))
    print("      P-M contrast factors k   : %s"
          % ", ".join("%.4f" % od["k"] for od in octaves))

    figure_2_scale_space(octaves)
    figure_3_raw_extrema(image, res["raw"])
    figure_4_filtered(image, res["filtered"])
    figure_5_localized(image, res["refined"])
    figure_6_oriented(image, res["oriented"])

    if COMPUTE_DESCRIPTORS and res["oriented"]:
        desc = np.array([kp["descriptor"] for kp in res["oriented"]])
        print("\n[6] descriptor matrix        : %s (128-D, RootSIFT%s)"
              % (desc.shape, "" if USE_ROOT_NORM else "-off, plain L2"))

    #validation
    if RUN_BENCHMARK:
        print("\n[7] invariance study (rotations %s, zooms %s) ..."
              % (", ".join("%.0f deg" % a for a in BENCH_ANGLES),
                 ", ".join("x%.2f" % s for s in BENCH_SCALES)))
        run_benchmark(image, res)

    #gaussian-kernel audit
    print("\n" + "-" * 74)
    print(" Gaussian-kernel audit")
    print("-" * 74)
    for line in [
        "scale space          : Perona-Malik nonlinear diffusion (no kernel)",
        "diffusion stencil    : isotropic 9-point, weights 2/3 and 1/6",
        "blob response        : sigma^4 det(Hessian) from central differences",
        "octave down-sampling : %s" % ("plain decimation L[::2,::2], no filter"
                                       if OCTAVE_DOWNSAMPLE == "decimate"
                                       else "3x3 BOX (uniform) average"),
        "orientation window   : triangular  w = max(0, 1 - r/R)",
        "histogram smoothing  : triangular  [1 2 3 2 1] / 9",
        "descriptor window    : triangular  w = max(0, 1 - r/R)",
        "derivatives          : [-1,0,1]/2 and [1,-2,1]  (no binomial [1,2,1])",
        "evaluation warps     : bilinear, anti_aliasing never enabled",
    ]:
        print("  * " + line)
    print("  => zero Gaussian kernels anywhere in the pipeline.")

    print("\nTOTAL RUNTIME : %.2f s  (Colab limit: 180 s)"
          % (time.time() - t_start))
    print("=" * 74)


if __name__ == "__main__":
    main()
