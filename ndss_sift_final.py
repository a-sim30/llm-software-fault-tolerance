import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from scipy.ndimage import maximum_filter
from skimage import data


#scale space
N_OCTAVES      = 4
N_SCALES       = 3
SIGMA_0        = 1.6
K_PERCENTILE   = 70.0
DT_MAX         = 0.28
OCTAVE_DOWNSAMPLE = "decimate"

#detection
RESPONSE_FLOOR = 1.0e-6
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


#pad image edges by one pixel
def pad_edge(L):
    return np.pad(L, 1, mode="edge")


#compute x-gradient using central differences
def d_x(L):
    P = pad_edge(L)
    return 0.5 * (P[1:-1, 2:] - P[1:-1, :-2])


#compute y-gradient using central differences
def d_y(L):
    P = pad_edge(L)
    return 0.5 * (P[2:, 1:-1] - P[:-2, 1:-1])


#compute second x-derivative
def d_xx(L):
    P = pad_edge(L)
    return P[1:-1, 2:] - 2.0 * L + P[1:-1, :-2]


#compute second y-derivative
def d_yy(L):
    P = pad_edge(L)
    return P[2:, 1:-1] - 2.0 * L + P[:-2, 1:-1]


#compute mixed second derivative
def d_xy(L):
    P = pad_edge(L)
    return 0.25 * (P[2:, 2:] - P[2:, :-2] - P[:-2, 2:] + P[:-2, :-2])


#compute gradient magnitude
def gradient_magnitude(L):
    gx, gy = d_x(L), d_y(L)
    return np.sqrt(gx * gx + gy * gy)


#estimate P-M contrast threshold from gradients
def estimate_contrast_factor(L, percentile=K_PERCENTILE):
    m = gradient_magnitude(L)
    pos = m[m > 0]
    if pos.size == 0:
        return 1e-3
    return float(max(np.percentile(pos, percentile), 1e-6))


#isotropic 9-point stencil with conservative weights
STENCIL = (( 0, +1, 2.0 / 3.0), ( 0, -1, 2.0 / 3.0),
           (+1,  0, 2.0 / 3.0), (-1,  0, 2.0 / 3.0),
           (+1, +1, 1.0 / 6.0), (+1, -1, 1.0 / 6.0),
           (-1, +1, 1.0 / 6.0), (-1, -1, 1.0 / 6.0))


#shift edge-padded array by one pixel
def shift_plane(P, dy, dx):
    return P[1 + dy:P.shape[0] - 1 + dy, 1 + dx:P.shape[1] - 1 + dx]


#perform one Perona-Malik diffusion step
def diffusion_step(L, k, dt):
    gx, gy = d_x(L), d_y(L)
    g = 1.0 / (1.0 + (gx * gx + gy * gy) / (k * k))

    Lp = pad_edge(L)
    gp = pad_edge(g)

    div = np.zeros_like(L)
    for dy, dx, w in STENCIL:
        div += w * (0.5 * (shift_plane(gp, dy, dx) + g)) * (shift_plane(Lp, dy, dx) - L)
    return L + dt * div


#diffuse image using stable explicit time steps
def diffuse(L, delta_t, k, dt_max=DT_MAX):
    if delta_t <= 1e-12:
        return L.copy()
    n_steps = int(np.ceil(delta_t / dt_max))
    dt = delta_t / n_steps
    out = L
    for _ in range(n_steps):
        out = diffusion_step(out, k, dt)
    return out


#compute normalised Hessian determinant and curvature ratio
def scale_normalised_hessian_response(L, sigma):
    Lxx, Lyy, Lxy = d_xx(L), d_yy(L), d_xy(L)
    det = Lxx * Lyy - Lxy * Lxy
    tr = Lxx + Lyy
    response = (sigma ** 4) * det
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(det > 0, tr * tr / np.where(det > 0, det, 1.0), np.inf)
    return response, ratio


#build nonlinear-diffusion scale space and responses
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

        #later octaves start from decimated level S
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

        #decimate level S to initialize next octave
        base = downsample(levels[n_scales])

    return octaves


#downsample image by factor two
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


#detect strict local maxima across scale space
def detect_extrema(octaves, floor=RESPONSE_FLOOR, border=BORDER):
    #exclude center from the 26-neighbour footprint
    footprint = np.ones((3, 3, 3), dtype=bool)
    footprint[1, 1, 1] = False

    raw = []
    for oct_data in octaves:
        R = oct_data["response"]
        _, H, W = R.shape

        nbr_max = maximum_filter(R, footprint=footprint, mode="nearest")
        is_max = (R > nbr_max) & (R > floor)

        #exclude boundary levels and image borders
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


#filter weak and edge-like unstable responses
def filter_extrema(raw, contrast_thr=CONTRAST_THR, edge_r=EDGE_RATIO_R):
    limit = (edge_r + 1.0) ** 2 / edge_r
    return [kp for kp in raw
            if kp["response"] >= contrast_thr and kp["ratio"] < limit]


#compute response gradient and Hessian
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


#refine keypoints using 3-D quadratic fitting
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

            #move toward the fitted extremum
            s += int(round(float(off[2])))
            y += int(round(float(off[1])))
            x += int(round(float(off[0])))

            #discard refinements leaving valid sampling bounds
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


#remove nearby duplicate detections
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


#smooth orientation histogram with triangular window
def smooth_hist_triangular(h):
    return (1.0 * np.roll(h, -2) + 2.0 * np.roll(h, -1) + 3.0 * h +
            2.0 * np.roll(h, 1) + 1.0 * np.roll(h, 2)) / 9.0


#build weighted gradient orientation histogram
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

    #use triangular radial weighting for orientations
    w = np.clip(1.0 - dist / float(radius), 0.0, None)
    weight = (sub_m * w).ravel()

    pos = sub_a.ravel() * (n_bins / (2.0 * np.pi))      #continuous bin index
    if ORI_SOFT_BINS:
        #distribute samples across neighbouring bins
        b0 = np.floor(pos - 0.5).astype(np.int64)
        frac = pos - 0.5 - b0
        idx = np.concatenate([np.mod(b0, n_bins), np.mod(b0 + 1, n_bins)])
        wts = np.concatenate([weight * (1.0 - frac), weight * frac])
        hist = np.bincount(idx, weights=wts, minlength=n_bins)
    else:
        bins = np.mod(np.floor(pos).astype(np.int64), n_bins)
        hist = np.bincount(bins, weights=weight, minlength=n_bins)
    return smooth_hist_triangular(hist)


#compute dominant orientations and descriptors
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


#compute rotation-normalised 128-D descriptor
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

    #use triangular radial weighting across descriptor patch
    dist = np.sqrt(dx * dx + dy * dy)
    w_rad = np.clip(1.0 - dist / (DESC_WINDOW_FAC * half), 0.0, None)

    a = np.mod(ang[y0:y1, x0:x1] - theta, 2.0 * np.pi)
    ob = a * (n_ori / (2.0 * np.pi))
    weight = mag[y0:y1, x0:x1] * w_rad

    #retain samples inside descriptor spatial bounds
    valid = (ub > -1.0) & (ub < n_bins) & (vb > -1.0) & (vb < n_bins) & (weight > 0)
    if not np.any(valid):
        return desc

    ub, vb, ob, weight = ub[valid], vb[valid], ob[valid], weight[valid]

    u0 = np.floor(ub).astype(np.int64)
    v0 = np.floor(vb).astype(np.int64)
    o0 = np.floor(ob).astype(np.int64)
    fu, fv, fo = ub - u0, vb - v0, ob - o0

    #interpolate samples into padded orientation histogram
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
        #apply RootSIFT normalization to descriptor
        s = desc.sum()
        if s > 1e-12:
            desc = np.sqrt(desc / s)
    return desc


#main pipeline

#run complete Gaussian-free SIFT pipeline
def ndss_sift(image, verbose=True):
    octaves = build_scale_space(image)
    # timings["scale space"] = time.time() - t0

    raw = detect_extrema(octaves)
    # timings["extrema"] = time.time() - t0

    filtered = filter_extrema(raw)
    # timings["filtering"] = time.time() - t0

    refined = deduplicate(refine_keypoints(octaves, filtered))
    # timings["refinement"] = time.time() - t0

    oriented = assign_orientations(octaves, refined)
    # timings["orientation + descriptor"] = time.time() - t0

    if verbose:
        print("  raw scale-space extrema      : %d" % len(raw))
        print("  after weak/edge rejection    : %d" % len(filtered))
        print("  after sub-pixel localisation : %d" % len(refined))
        print("  final oriented keypoints     : %d" % len(oriented))
        # for key, val in timings.items():
        #     print("    [%-24s] %6.2f s" % (key, val))

    return {"octaves": octaves, "raw": raw, "filtered": filtered,
            "refined": refined, "oriented": oriented}


#save figure when enabled and display it
def show(fig, name):
    if SAVE_FIGURES:
        fig.savefig(name, dpi=130, bbox_inches="tight")
    plt.show()


#display original input image
def figure_1_original(image):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Figure 1 - Original Cameraman (%d x %d)" % image.shape)
    ax.axis("off")
    show(fig, "fig1_original.png")


#display selected scale-space levels and responses
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


#display raw extrema in original image coordinates
def figure_3_raw_extrema(image, raw):
    #convert octave coordinates into image coordinates
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


#display keypoints after response filtering
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


#display sub-pixel and sub-scale keypoints
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


#display final keypoints with orientations
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
        #batch artists improve plotting performance
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


def main():
    t_start = time.time()

    print("=" * 74)
    print("Gaussian-free SIFT via nonlinear diffusion scale space")
    print("=" * 74)

    #load mandatory test image
    raw_img = data.camera()
    assert raw_img.shape == (512, 512), "unexpected cameraman size"
    image = raw_img.astype(np.float64) / 255.0
    print("\n[0] image : skimage.data.camera(), %dx%d, range [%.2f, %.2f]"
          % (image.shape[0], image.shape[1], image.min(), image.max()))

    figure_1_original(image)

    #run complete feature extraction pipeline
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


    print("\nTOTAL RUNTIME : %.2f s  (Colab limit: 180 s)"
          % (time.time() - t_start))
    print("=" * 74)


if __name__ == "__main__":
    main()
