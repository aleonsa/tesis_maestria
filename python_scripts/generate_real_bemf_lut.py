"""generate_real_bemf_lut.py
Extract real BEMF shape from dSPACE experimental recordings and generate LUT +
ADALINE weight files compatible with fcs_m2pc_v2.

Input:
    datos_axel/rec1_001.csv  (≈52.5 rad/s)
    datos_axel/rec1_002.csv  (≈35.4 rad/s)
    datos_axel/rec1_003.csv  (≈96.1 rad/s)

Output (written to fcs_m2pc_v2/data/):
    bemf_real_lut.mat          – lut_theta, lut_alpha_real, lut_beta_real,
                                  lut_alpha (ADALINE fit), lut_beta (ADALINE fit),
                                  Ke_measured
    bemf_real_adaline_lut.mat  – adaline_W, adaline_H

CSV format (dSPACE ControlDesk export):
    29 header lines; line 30 starts with "trace_values" prefix on first data row.
    Columns after prefix/empty first field (0-indexed):
        0 = time [s]
        1 = Ds (d-axis current) [A]
        2 = Qs (q-axis current) [A]
        3 = Va [V]
        4 = Vb [V]
        5 = Vc [V]
        6 = speed [rad/s]
        7 = Theta_e [rad]
        8 = Id_ref (= 0)
        9 = Iq_ref
"""

from __future__ import annotations

import pathlib
import sys
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; figures saved to disk
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR    = pathlib.Path(__file__).parent.parent / "datos_axel"
OUT_DIR     = pathlib.Path(__file__).parent.parent / "fcs_m2pc_v2" / "data"
FIG_DIR     = OUT_DIR / "figures_real_bemf"

CSV_FILES = [
    DATA_DIR / "rec1_001.csv",   # ~52.5 rad/s
    DATA_DIR / "rec1_002.csv",   # ~35.4 rad/s
    DATA_DIR / "rec1_003.csv",   # ~96.1 rad/s
]

HEADER_LINES = 29   # lines to skip before first data row
N_BINS       = 1000 # angular bins over [0, 2π]
H_ADALINE    = 10   # number of Fourier harmonics
Ke_NOMINAL   = 0.40355  # V·s/rad (motor nominal, for comparison)

# Minimum speed threshold to exclude near-zero samples (div-by-zero risk)
OMEGA_MIN = 5.0  # rad/s

# Clarke transform (amplitude-invariant, 3→2)
# e_α = (2/3)(ea  − eb/2 − ec/2)
# e_β = (2/3)(√3/2)(eb − ec)
_C = (2.0 / 3.0) * np.array([
    [1.0, -0.5,        -0.5       ],
    [0.0,  np.sqrt(3)/2, -np.sqrt(3)/2],
])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class FileData(NamedTuple):
    speed:   np.ndarray   # [N]  rad/s
    theta:   np.ndarray   # [N]  rad, wrapped to [0, 2π)
    e_alpha: np.ndarray   # [N]  V
    e_beta:  np.ndarray   # [N]  V


def load_csv(path: pathlib.Path) -> FileData:
    """Parse a single dSPACE CSV recording and return raw BEMF signals."""
    rows: list[list[str]] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            if lineno <= HEADER_LINES:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            # First data row starts with "trace_values," — remove that token
            if stripped.startswith("trace_values,"):
                stripped = stripped[len("trace_values,"):]
            # Subsequent rows start with a leading comma; strip it
            elif stripped.startswith(","):
                stripped = stripped[1:]
            # Now column order: time, Ds, Qs, Va, Vb, Vc, speed, theta_e, Id_ref, Iq_ref
            parts = stripped.split(",")
            if len(parts) < 8:
                continue
            rows.append(parts)

    if not rows:
        raise ValueError(f"No data found in {path}")

    arr = np.array([[float(x) for x in row[:10]] for row in rows], dtype=np.float64)
    # arr columns: 0=time, 1=Ds, 2=Qs, 3=Va, 4=Vb, 5=Vc, 6=speed, 7=theta_e
    Va     = arr[:, 3]
    Vb     = arr[:, 4]
    Vc     = arr[:, 5]
    speed  = arr[:, 6]          # rad/s (electrical)
    # dSPACE θ_e advances in the negative direction (motor spins CW in lab frame).
    # Flip sign so θ increases with positive ω_m convention used in the simulation.
    theta  = np.mod(-arr[:, 7], 2 * np.pi)

    # Phase voltages ≈ BEMF (negligible resistive + inductive drops at ~21 mA)
    Vabc      = np.column_stack([Va, Vb, Vc])          # [N x 3]
    e_ab      = (Vabc @ _C.T)                          # [N x 2]
    e_alpha   = e_ab[:, 0]
    e_beta    = e_ab[:, 1]

    return FileData(speed=speed, theta=theta, e_alpha=e_alpha, e_beta=e_beta)


# ---------------------------------------------------------------------------
# Ke estimation
# ---------------------------------------------------------------------------

def estimate_Ke(speed: np.ndarray, e_alpha: np.ndarray, e_beta: np.ndarray) -> float:
    """Estimate Ke = ||e_αβ|| / |ω| via least-squares scalar regression.

    ||e_αβ|| = Ke * |ω|  →  Ke = (Σ ||e_αβ||·|ω|) / (Σ ω²)
    """
    e_mag = np.sqrt(e_alpha**2 + e_beta**2)
    omega = np.abs(speed)
    mask  = omega > OMEGA_MIN
    return float(np.sum(e_mag[mask] * omega[mask]) / np.sum(omega[mask]**2))


# ---------------------------------------------------------------------------
# Bin-averaging
# ---------------------------------------------------------------------------

def bin_average(
    all_theta:   np.ndarray,
    all_s_alpha: np.ndarray,
    all_s_beta:  np.ndarray,
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average BEMF shape over equal-width angular bins in [0, 2π)."""
    edges  = np.linspace(0.0, 2 * np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sa = np.zeros(n_bins)
    sb = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    bin_idx = np.searchsorted(edges[1:], all_theta, side="left")
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    np.add.at(sa, bin_idx, all_s_alpha)
    np.add.at(sb, bin_idx, all_s_beta)
    np.add.at(counts, bin_idx, 1)

    empty = counts == 0
    if empty.any():
        print(f"  ⚠  {empty.sum()} empty bins — will interpolate.")
        # linear interpolation over empty bins
        valid = ~empty
        sa[empty] = np.interp(centers[empty], centers[valid], sa[valid] / counts[valid].clip(1))
        sb[empty] = np.interp(centers[empty], centers[valid], sb[valid] / counts[valid].clip(1))
        counts[empty] = 1

    return centers, sa / counts, sb / counts


# ---------------------------------------------------------------------------
# ADALINE (offline Fourier fit via pseudoinverse)
# ---------------------------------------------------------------------------

def build_fourier_basis_matrix(theta: np.ndarray, H: int) -> np.ndarray:
    """Build [N x 2H] Fourier regressor matrix."""
    n_samples = len(theta)
    X = np.zeros((n_samples, 2 * H))
    for n in range(1, H + 1):
        X[:, 2*(n-1)]   = np.cos(n * theta)
        X[:, 2*(n-1)+1] = np.sin(n * theta)
    return X


def fit_adaline(
    theta: np.ndarray,
    s_alpha: np.ndarray,
    s_beta: np.ndarray,
    H: int,
) -> np.ndarray:
    """Fit ADALINE weights W [2H x 2] via least-squares pseudoinverse.

    S = [s_alpha | s_beta] [N x 2]
    W = pinv(X) · S
    """
    X = build_fourier_basis_matrix(theta, H)
    S = np.column_stack([s_alpha, s_beta])
    W, _, _, _ = np.linalg.lstsq(X, S, rcond=None)
    return W   # [2H x 2]


def eval_adaline(W: np.ndarray, theta: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    X = build_fourier_basis_matrix(theta, H)
    pred = X @ W
    return pred[:, 0], pred[:, 1]


# ---------------------------------------------------------------------------
# Synthetic BEMF for comparison
# ---------------------------------------------------------------------------

def _trap_abc(theta: float) -> np.ndarray:
    """Trapezoidal BEMF shape in ABC (matches bemf_trapezoidal.m)."""
    def _trap_phase(t: float) -> float:
        t = t % (2 * np.pi)
        if t < np.pi / 6:
            return t / (np.pi / 6)
        elif t < 5 * np.pi / 6:
            return 1.0
        elif t < 7 * np.pi / 6:
            return 1.0 - (t - 5*np.pi/6) / (np.pi/6)
        elif t < 11 * np.pi / 6:
            return -1.0
        else:
            return -1.0 + (t - 11*np.pi/6) / (np.pi/6)
    a = _trap_phase(theta)
    b = _trap_phase(theta - 2*np.pi/3)
    c = _trap_phase(theta - 4*np.pi/3)
    return np.array([a, b, c])


def synthetic_trap_ab(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sa = np.zeros(len(theta))
    sb = np.zeros(len(theta))
    for i, t in enumerate(theta):
        abc = _trap_abc(t)
        ab  = _C @ abc
        sa[i] = ab[0]
        sb[i] = ab[1]
    return sa, sb


def synthetic_sin_ab(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.sin(theta), -np.cos(theta)


# ---------------------------------------------------------------------------
# θ_e offset optimisation
# ---------------------------------------------------------------------------

def find_theta_offset(
    all_theta:   np.ndarray,
    all_s_alpha: np.ndarray,
    all_s_beta:  np.ndarray,
    candidates:  np.ndarray,
    n_bins: int = N_BINS,
) -> tuple[float, float]:
    """Return (best_offset, best_rmse) that minimises RMSE vs ideal sinusoidal shape.

    The reference is:  s_α(θ) = sin(θ),  s_β(θ) = -cos(θ).
    For each candidate offset δ the theta is shifted: θ' = (θ + δ) mod 2π,
    bin-averaged, and compared to the reference.
    """
    best_offset: float = 0.0
    best_rmse:   float = np.inf
    for offset in candidates:
        theta_off = np.mod(all_theta + offset, 2 * np.pi)
        lut_t, sa, sb = bin_average(theta_off, all_s_alpha, all_s_beta, n_bins)
        rmse = float(np.sqrt(np.mean(
            (sa - np.sin(lut_t))**2 + (sb + np.cos(lut_t))**2
        )))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_offset = float(offset)
    return best_offset, best_rmse


# ---------------------------------------------------------------------------
# Harmonic analysis
# ---------------------------------------------------------------------------

def harmonic_spectrum(s: np.ndarray, n_harmonics: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Return harmonic orders and normalised amplitudes (relative to fundamental)."""
    N = len(s)
    fft = np.fft.fft(s) / N
    amps = 2.0 * np.abs(fft[1:n_harmonics + 1])
    fund = amps[0] if amps[0] > 1e-9 else 1.0
    return np.arange(1, n_harmonics + 1), amps / fund


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(
    files: list[FileData],
    lut_theta: np.ndarray,
    lut_sa: np.ndarray,
    lut_sb: np.ndarray,
    ada_sa: np.ndarray,
    ada_sb: np.ndarray,
    Ke_measured: float,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels   = [f"rec1 — {np.mean(np.abs(f.speed)):.1f} rad/s" for f in files]
    colors   = ["tab:blue", "tab:orange", "tab:green"]

    # ── Figure 1: raw BEMF shape per speed ──────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for fd, label, c in zip(files, labels, colors):
        mask = np.abs(fd.speed) > OMEGA_MIN
        s_a  = fd.e_alpha[mask] / (Ke_measured * np.abs(fd.speed[mask]))
        s_b  = fd.e_beta[mask]  / (Ke_measured * np.abs(fd.speed[mask]))
        axes[0].scatter(fd.theta[mask], s_a, s=0.3, c=c, label=label, alpha=0.4)
        axes[1].scatter(fd.theta[mask], s_b, s=0.3, c=c, label=label, alpha=0.4)
    axes[0].set_ylabel(r"$s_\alpha(\theta)$")
    axes[0].set_title("BEMF shape — α component (all recordings overlaid)")
    axes[0].legend(markerscale=10, loc="upper right")
    axes[0].grid(True)
    axes[1].set_ylabel(r"$s_\beta(\theta)$")
    axes[1].set_title("BEMF shape — β component (all recordings overlaid)")
    axes[1].set_xlabel("θ_e [rad]")
    axes[1].legend(markerscale=10, loc="upper right")
    axes[1].grid(True)
    fig.suptitle("Figura 1 — Forma de la BEMF real (tres velocidades)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_bemf_raw_speeds.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: Real (binned) vs synthetic vs ADALINE fit ─────────────────
    sa_trap, sb_trap = synthetic_trap_ab(lut_theta)
    sa_sin,  sb_sin  = synthetic_sin_ab(lut_theta)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax0, ax1 = axes

    ax0.plot(lut_theta, lut_sa,  lw=2.0,  c="black",       label="Real (binned)")
    ax0.plot(lut_theta, ada_sa,  lw=1.5,  c="tab:purple",  ls="--", label=f"ADALINE fit (H={H_ADALINE})")
    ax0.plot(lut_theta, sa_trap, lw=1.2,  c="tab:red",     ls=":",  label="Sintético (trapezoidal)")
    ax0.plot(lut_theta, sa_sin,  lw=1.0,  c="tab:blue",    ls="-.", label="Sintético (sinusoidal)")
    ax0.set_ylabel(r"$s_\alpha(\theta)$")
    ax0.legend(fontsize=8)
    ax0.grid(True)
    ax0.set_title("Componente α")

    ax1.plot(lut_theta, lut_sb,  lw=2.0,  c="black",       label="Real (binned)")
    ax1.plot(lut_theta, ada_sb,  lw=1.5,  c="tab:purple",  ls="--", label=f"ADALINE fit (H={H_ADALINE})")
    ax1.plot(lut_theta, sb_trap, lw=1.2,  c="tab:red",     ls=":")
    ax1.plot(lut_theta, sb_sin,  lw=1.0,  c="tab:blue",    ls="-.")
    ax1.set_ylabel(r"$s_\beta(\theta)$")
    ax1.set_xlabel("θ_e [rad]")
    ax1.grid(True)
    ax1.set_title("Componente β")

    fig.suptitle("Figura 2 — BEMF real vs sintéticas y ajuste ADALINE", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_bemf_real_vs_synthetic.png", dpi=150)
    plt.close(fig)

    # ── Figure 3: Harmonic spectrum ──────────────────────────────────────────
    n_harmonics = 15
    orders_r, amps_ra = harmonic_spectrum(lut_sa, n_harmonics)
    _,         amps_rb = harmonic_spectrum(lut_sb, n_harmonics)
    _,         amps_ta = harmonic_spectrum(sa_trap, n_harmonics)
    _,         amps_sa = harmonic_spectrum(sa_sin, n_harmonics)

    width = 0.3
    x = np.arange(1, n_harmonics + 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, amps_ra * 100, width, label="Real α", color="black", alpha=0.8)
    ax.bar(x,          amps_rb * 100, width, label="Real β", color="dimgray", alpha=0.8)
    ax.bar(x + width,  amps_ta * 100, width, label="Trapezoidal α", color="tab:red", alpha=0.7)
    ax.set_xlabel("Orden armónico")
    ax.set_ylabel("Amplitud relativa al fundamental [%]")
    ax.set_title("Figura 3 — Espectro armónico: BEMF real vs trapezoidal", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_harmonic_spectrum.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved to: {FIG_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  generate_real_bemf_lut.py — BEMF LUT from dSPACE recordings")
    print("=" * 65)

    # ── 1. Load all CSV files ──────────────────────────────────────────────
    files: list[FileData] = []
    print("\n[1/5] Loading CSV files …")
    for csv_path in CSV_FILES:
        if not csv_path.exists():
            print(f"  ERROR: {csv_path} not found", file=sys.stderr)
            sys.exit(1)
        fd = load_csv(csv_path)
        files.append(fd)
        print(f"  {csv_path.name}: {len(fd.speed)} samples, "
              f"ω̄ = {np.mean(np.abs(fd.speed)):.2f} rad/s, "
              f"θ ∈ [{fd.theta.min():.3f}, {fd.theta.max():.3f}] rad")

    # ── 2. Estimate Ke ────────────────────────────────────────────────────
    print("\n[2/5] Estimating Ke …")
    all_speed   = np.concatenate([f.speed   for f in files])
    all_e_alpha = np.concatenate([f.e_alpha for f in files])
    all_e_beta  = np.concatenate([f.e_beta  for f in files])

    Ke_measured = estimate_Ke(all_speed, all_e_alpha, all_e_beta)
    print(f"  Ke measured  = {Ke_measured:.5f}  V·s/rad")
    print(f"  Ke nominal   = {Ke_NOMINAL:.5f}  V·s/rad")
    print(f"  Δ Ke         = {100*(Ke_measured - Ke_NOMINAL)/Ke_NOMINAL:+.2f}%")

    # ── 3. Build normalised shape and bin-average ─────────────────────────
    print("\n[3/5] Building BEMF shape and bin-averaging …")
    mask   = np.abs(all_speed) > OMEGA_MIN
    n_keep = mask.sum()
    print(f"  Samples after |ω| > {OMEGA_MIN} rad/s filter: {n_keep} / {len(all_speed)}")

    all_s_alpha = all_e_alpha[mask] / (Ke_measured * np.abs(all_speed[mask]))
    all_s_beta  = all_e_beta[mask]  / (Ke_measured * np.abs(all_speed[mask]))
    all_theta   = np.concatenate([f.theta for f in files])[mask]

    # ── 3b. Optimise Hall-sensor alignment offset ─────────────────────────
    print("\n[3b/5] Optimising θ_e alignment offset …")
    # Coronado's Simulink uses PositionOffset = 0.1364 PU  →  0.1364 × 2π ≈ 0.857 rad
    coronado_offset = 0.1364 * 2 * np.pi          # ≈ 0.857 rad

    # Coarse sweep: 73 candidates evenly spaced over [0, 2π) (≈ 5° steps)
    coarse = np.linspace(0.0, 2 * np.pi, 73, endpoint=False)
    best_offset, best_rmse = find_theta_offset(all_theta, all_s_alpha, all_s_beta, coarse)

    # Fine sweep: 181 candidates ±30° around the coarse best (≈ 0.33° steps)
    fine = np.linspace(best_offset - np.pi / 6, best_offset + np.pi / 6, 181)
    best_offset, best_rmse = find_theta_offset(all_theta, all_s_alpha, all_s_beta, fine)

    print(f"  Coronado PositionOffset : {coronado_offset:.4f} rad  ({np.degrees(coronado_offset):.1f}°)")
    print(f"  Best offset found       : {best_offset:.4f} rad  ({np.degrees(best_offset):.1f}°)  RMSE = {best_rmse:.5f}")

    # Apply offset to all theta values
    all_theta = np.mod(all_theta + best_offset, 2 * np.pi)

    lut_theta, lut_sa, lut_sb = bin_average(all_theta, all_s_alpha, all_s_beta, N_BINS)
    print(f"  LUT: {N_BINS} bins, "
          f"α ∈ [{lut_sa.min():.4f}, {lut_sa.max():.4f}], "
          f"β ∈ [{lut_sb.min():.4f}, {lut_sb.max():.4f}]")

    # ── 4. Fit ADALINE offline ────────────────────────────────────────────
    print("\n[4/5] Fitting ADALINE (offline, H={}) …".format(H_ADALINE))
    W_adaline = fit_adaline(lut_theta, lut_sa, lut_sb, H_ADALINE)
    ada_sa, ada_sb = eval_adaline(W_adaline, lut_theta, H_ADALINE)

    rmse_a = float(np.sqrt(np.mean((ada_sa - lut_sa)**2)))
    rmse_b = float(np.sqrt(np.mean((ada_sb - lut_sb)**2)))
    print(f"  RMSE α: {rmse_a:.5f}   RMSE β: {rmse_b:.5f}")

    # Harmonic content of real shape
    _, amps = harmonic_spectrum(lut_sa, 10)
    print(f"  Harmonic content of s_α (rel. to fundamental):")
    for h, a in enumerate(amps, start=1):
        if h in (1, 5, 7, 11, 13) or a > 0.01:
            print(f"    H{h:2d}: {a*100:.2f}%")

    # ── 5. Save .mat files ────────────────────────────────────────────────
    print("\n[5/5] Saving .mat files …")

    lut_path = OUT_DIR / "bemf_real_lut.mat"
    sio.savemat(str(lut_path), {
        "lut_theta":      lut_theta.reshape(-1, 1),
        "lut_alpha_real": lut_sa.reshape(-1, 1),
        "lut_beta_real":  lut_sb.reshape(-1, 1),
        "lut_alpha":      ada_sa.reshape(-1, 1),   # ADALINE eval on grid
        "lut_beta":       ada_sb.reshape(-1, 1),
        "Ke_measured":    float(Ke_measured),
        "theta_offset":   float(best_offset),      # Hall-sensor alignment offset [rad]
    })
    print(f"  ✓  {lut_path}")

    adaline_path = OUT_DIR / "bemf_real_adaline_lut.mat"
    sio.savemat(str(adaline_path), {
        "adaline_W": W_adaline,          # [2H x 2]
        "adaline_H": int(H_ADALINE),
    })
    print(f"  ✓  {adaline_path}")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n[+] Generating diagnostic figures …")
    make_figures(files, lut_theta, lut_sa, lut_sb, ada_sa, ada_sb, Ke_measured)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  RESUMEN")
    print("=" * 65)
    print(f"  Ke medido:       {Ke_measured:.5f} V·s/rad")
    print(f"  Ke nominal:      {Ke_NOMINAL:.5f} V·s/rad  (Δ = {100*(Ke_measured-Ke_NOMINAL)/Ke_NOMINAL:+.2f}%)")
    print(f"  Offset θ_e:      {best_offset:.4f} rad  ({np.degrees(best_offset):.1f}°)")
    print(f"  RMSE vs sinus:   {best_rmse:.5f}")
    print(f"  Muestras totales:{n_keep}")
    print(f"  Bins LUT:        {N_BINS}")
    print(f"  Armónicos ADALINE: H = {H_ADALINE}  ({2*H_ADALINE} coef. por componente)")
    print(f"  RMSE ajuste:     α={rmse_a:.5f}  β={rmse_b:.5f}")
    print()
    print("  Archivos generados:")
    print(f"    {lut_path.name}")
    print(f"    {adaline_path.name}")
    print(f"    {FIG_DIR.name}/fig1_bemf_raw_speeds.png")
    print(f"    {FIG_DIR.name}/fig2_bemf_real_vs_synthetic.png")
    print(f"    {FIG_DIR.name}/fig3_harmonic_spectrum.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
