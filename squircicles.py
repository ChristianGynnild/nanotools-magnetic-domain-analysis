import numpy as np
import matplotlib.pyplot as plt

# finn sentrum
def find_core_subpixel(X, Y, window=7):
    """
    Estimate subpixel center assuming the core is near the geometric center.
    Works only from X, Y (no mz/intensity data).
    
    X, Y : 2D arrays (same shape)
    window : odd integer, number of pixels around the center to consider
    
    Returns (x0, y0) in same units as X, Y.
    """
    ny, nx = X.shape
    # assume core near array center
    iy = ny // 2
    ix = nx // 2

    # define window bounds
    hw = window // 2
    y0 = max(iy - hw, 0)
    y1 = min(iy + hw + 1, ny)
    x0 = max(ix - hw, 0)
    x1 = min(ix + hw + 1, nx)

    # extract local coordinate patch
    Xs = X[y0:y1, x0:x1]
    Ys = Y[y0:y1, x0:x1]

    # compute subpixel center geometrically (centroid of window)
    sx = np.mean(Xs)
    sy = np.mean(Ys)

    return sx, sy

def to_polar(X, Y, x0, y0):
    Xc = X - x0
    Yc = Y - y0
    R = np.sqrt(Xc**2 + Yc**2)
    Phi = np.arctan2(Yc, Xc)
    return R, Phi

def angle_diff(a, b):
    d = a - b
    d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return d

def compute_angle_rms(Psi, Phi, chi=np.pi/2, correct_global_rotation=True):
    Psi_ideal = Phi + chi
    d = angle_diff(Psi, Psi_ideal)

    if correct_global_rotation:
        # mean_angle = atan2(mean(sin(d)), mean(cos(d)))
        sin_mean = np.nanmean(np.sin(d))
        cos_mean = np.nanmean(np.cos(d))
        mean_offset = np.arctan2(sin_mean, cos_mean)

        # Subtract the global mean offset
        d = angle_diff(d, mean_offset)

    rms = np.sqrt(np.nanmean(d**2))
    return rms, d

def compute_angle_difference(field, n_sectors=100, real_data = False):


    if real_data:
        mx, my = field[:,:,0], field[:,:,1]
    else:
        mx, my = field[:,0,:,:], field[:,1,:,:]

    ny, nx = mx.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    x0, y0 = find_core_subpixel(X, Y, window=9)
    mask = np.hypot(mx, my) > 1e-6
    Psi = np.arctan2(my, mx)
    R, Phi = to_polar(X, Y, x0, y0)

    # Compute only for valid (non-zero) magnetization region
    Psi_masked = np.where(mask, Psi, np.nan)
    Phi_masked = np.where(mask, Phi, np.nan)

    rms_angle, dangle_map = compute_angle_rms(Psi_masked, Phi_masked, chi=np.pi/2)

    rms_angle, dpsi_map = compute_angle_rms(Psi_masked, Phi_masked, chi=np.pi/2, correct_global_rotation=True)

    r_min = 0.0 # minimum randius
    r_max = np.nanmax(R) # maks radius som ikke er nan

    # Sektorindekser (0..n_sectors-1)
    Phi_mod = (Phi + 2*np.pi) % (2*np.pi)
    sector_width = 2*np.pi / n_sectors
    # Hvilken sektor (indeks) hver piksel tilhører
    sector_idx = np.floor(Phi_mod / sector_width).astype(int) % n_sectors

    # Flate ut arrayer for enkel indeksering
    Rr = R.ravel()
    dpsi_r = dpsi_map.ravel()
    sec_r = sector_idx.ravel()
    mask_flat = mask.ravel() # Den fysiske masken fra mx, my

    # Maske for radius range
    mask_r = (Rr >= r_min) & (Rr <= r_max)

    # # Output-arrays
    sector_mean_dpsi = np.full(n_sectors, np.nan)
    sector_rms_dpsi  = np.full(n_sectors, np.nan)

    # Sektorvise statistikker
    for s in range(n_sectors):
        # Kombinert selektor: i riktig sektor, innenfor radius, og gyldig magnetisering
        sel = (sec_r == s) & mask_r & mask_flat #både del av sektor innfenfor radius range og den fysiske masken.

        if np.any(sel):
            vals = dpsi_r[sel]
            # Vektor-gjennomsnitt for vinkler (robust mot 2*pi wrap)
            vec = np.nanmean(np.exp(1j * vals)) #konverter vinkler til komplekstall
            mean_ang = np.angle(vec) # finner gjennomsnitt av vinkler
            sector_mean_dpsi[s] = mean_ang
            # RMS rundt mean_ang
            sector_rms_dpsi[s] = np.sqrt(np.nanmean(angle_diff(vals, mean_ang)**2))

    centers = (np.arange(n_sectors) + 0.5) * sector_width # Sentervinkler for stolpene

    # 1. Klargjør data for plotting
    plot_angles = np.concatenate([centers, [centers[0] + 2*np.pi]]) # gjør om til lukket sirkel
    plot_values = np.concatenate([sector_mean_dpsi, [sector_mean_dpsi[0]]])

    return plot_angles, plot_values, rms_angle, dangle_map, centers

