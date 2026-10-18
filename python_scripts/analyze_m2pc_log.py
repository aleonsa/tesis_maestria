#!/usr/bin/env python3
"""Análisis offline del log de 07_m2pc: reconstrucción de la BEMF.

Lee las líneas `L t th ia ib ua ub w ref` (1 kHz) y calcula

    e_αβ = u_αβ − R·i_αβ − L·di_αβ/dt

con R, L del banco (motor_params.h). Luego proyecta e sobre los ejes del
rotor que da el encoder:

    e_d =  e_α·cos θ_e + e_β·sin θ_e      (debe ser ≈ 0)
    e_q = −e_α·sin θ_e + e_β·cos θ_e      (debe ser ≈ λ·ω_e)

Resultados:
  - λ identificado = <e_q> / <ω_e>            (comparar con 3.5 mWb del planning)
  - forma de la BEMF normalizada e/(λ·ω_e) en función de θ_e, promediada por
    bins de ángulo → es el DATO para entrenar el ADALINE offline (misma
    metodología que fcs_m2pc_v2/adaptation).
  - armónicos de esa forma (FFT sobre los bins).

Uso:
    python3 analyze_m2pc_log.py datos_banco/m2pc_300rpm_2026-10-14.log [--t0 3 --t1 12]

Caveat: u es el comando medio; el dead-time (500 ns a 50 kHz ≈ 2.5 % de Vbus
≈ 0.3 V) distorsiona la tensión real según el signo de la corriente. A baja
velocidad (BEMF ~0.8 V a 300 rpm) ese error no es despreciable: para
identificar fino conviene loguear a 600-900 rpm. Aquí se reporta tal cual.
"""

import argparse
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_RAW_PER_A = 32.93
R_OHM       = 2.7
L_H         = 0.86e-3
POLE_PAIRS  = 7
LAMBDA_NOM  = 0.0035
FS_LOG      = 1000.0

LINE = re.compile(r"^L (\d+) (\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)")


def load(path):
    rows = []
    for ln in open(path, errors="ignore"):
        m = LINE.match(ln.strip())
        if m:
            rows.append([int(x) for x in m.groups()])
    a = np.array(rows, dtype=float)
    t     = a[:, 0] / 1000.0
    th    = a[:, 1] / 65536.0 * 2 * np.pi            # rad eléctricos
    i_ab  = a[:, 2:4] / 256.0 / K_RAW_PER_A          # A
    u_ab  = a[:, 4:6] / 1000.0                       # V
    rpm   = a[:, 6] / 10.0
    ref   = a[:, 7] / 1000.0
    return t, th, i_ab, u_ab, rpm, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--t0", type=float, default=None, help="inicio de régimen [s]")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--bins", type=int, default=72)
    ap.add_argument("--win", type=float, default=30.0,
                    help="ventana [s] para el DC de e_αβ en el tiempo (0 = no imprimir)")
    ap.add_argument("--png", default="/tmp/opencode/bemf_analisis.png")
    args = ap.parse_args()

    t, th, i_ab, u_ab, rpm, ref = load(args.log)
    # El log puede traer varios arranques (reset): quedarse con el último tramo monótono.
    cut = np.where(np.diff(t) < 0)[0]
    if len(cut):
        s = cut[-1] + 1
        t, th, i_ab, u_ab, rpm, ref = t[s:], th[s:], i_ab[s:], u_ab[s:], rpm[s:], ref[s:]
    if args.t0 is not None:
        sel = (t >= args.t0) & (t <= (args.t1 if args.t1 else t[-1]))
        t, th, i_ab, u_ab, rpm, ref = t[sel], th[sel], i_ab[sel], u_ab[sel], rpm[sel], ref[sel]

    w_e = rpm / 60.0 * 2 * np.pi * POLE_PAIRS         # rad/s eléctricos
    di  = np.gradient(i_ab, 1.0 / FS_LOG, axis=0)
    e   = u_ab - R_OHM * i_ab - L_H * di              # V

    # Una BEMF no tiene componente DC. La que aparece es R·i_DC real: el offset
    # de corriente calibrado al arranque deriva con la temperatura, el lazo
    # regula a cero la corriente MEDIDA y queda un DC real que el modelo ve
    # como tensión. Se quita por eje y se reporta como estimación de la deriva.
    e_dc = e.mean(axis=0)
    if args.win > 0 and (t[-1] - t[0]) >= args.win:
        print(f"DC de e_αβ por ventanas de {args.win:.0f} s (mV → LSB de offset = mV/R·K):")
        for t_a in np.arange(t[0], t[-1] - args.win + 1e-9, args.win):
            w = (t >= t_a) & (t < t_a + args.win)
            d = e[w].mean(axis=0)
            print(f"  t={t_a:7.1f} s  α={d[0]*1e3:+7.1f}  β={d[1]*1e3:+7.1f} mV   "
                  f"({d[0]/R_OHM*K_RAW_PER_A:+.2f} / {d[1]/R_OHM*K_RAW_PER_A:+.2f} LSB)  "
                  f"rpm={rpm[w].mean():.0f}")
    e    = e - e_dc
    print(f"DC quitado de e_αβ: {e_dc[0]*1e3:+.0f} / {e_dc[1]*1e3:+.0f} mV  "
          f"→ deriva de offset ≈ {e_dc[0]/R_OHM*1e3:+.0f} / {e_dc[1]/R_OHM*1e3:+.0f} mA "
          f"({e_dc[0]/R_OHM*K_RAW_PER_A:+.1f} / {e_dc[1]/R_OHM*K_RAW_PER_A:+.1f} LSB)")

    c, s = np.cos(th), np.sin(th)
    e_d =  e[:, 0] * c + e[:, 1] * s
    e_q = -e[:, 0] * s + e[:, 1] * c
    i_d =  i_ab[:, 0] * c + i_ab[:, 1] * s
    i_q = -i_ab[:, 0] * s + i_ab[:, 1] * c

    lam = np.sum(e_q * w_e) / np.sum(w_e * w_e)       # ajuste por mínimos cuadrados
    print(f"muestras: {len(t)}  t = {t[0]:.2f}..{t[-1]:.2f} s  rpm = {rpm.mean():.1f} ± {rpm.std():.1f}")
    print(f"i_d = {i_d.mean()*1e3:+.1f} mA   i_q = {i_q.mean()*1e3:+.1f} mA")
    print(f"e_d = {e_d.mean()*1e3:+.1f} mV (debe ~0)   e_q = {e_q.mean()*1e3:+.1f} mV")
    print(f"λ identificado = {lam*1e3:.3f} mWb   (nominal {LAMBDA_NOM*1e3:.1f} mWb, {100*(lam/LAMBDA_NOM-1):+.1f} %)")
    print(f"BEMF de pico media = {np.mean(np.hypot(e[:,0], e[:,1]))*1e3:.0f} mV")

    # Forma normalizada por bins de θ_e (el ADALINE aprende esto).
    norm = lam * w_e
    ok = np.abs(norm) > 1e-3
    sh_a = e[ok, 0] / norm[ok]
    sh_b = e[ok, 1] / norm[ok]
    thb  = th[ok]
    edges = np.linspace(0, 2 * np.pi, args.bins + 1)
    idx = np.digitize(thb, edges) - 1
    fa = np.array([sh_a[idx == k].mean() if np.any(idx == k) else np.nan for k in range(args.bins)])
    fb = np.array([sh_b[idx == k].mean() if np.any(idx == k) else np.nan for k in range(args.bins)])
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Armónicos de la forma (sobre e_α normalizada; ideal: −sin θ → solo 1º).
    if not np.any(np.isnan(fa)):
        F = np.fft.rfft(fa) / (args.bins / 2)
        mag = np.abs(F)
        print("armónicos de e_α/(λω) [% del fundamental]:",
              "  ".join(f"h{k}={100*mag[k]/mag[1]:.1f}" for k in range(2, 8)))

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    ax[0].plot(t, rpm, lw=0.8); ax[0].set_ylabel("rpm"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"07_m2pc — ω̂, i_dq, y forma de la BEMF (λ = {lam*1e3:.2f} mWb)")
    ax[1].plot(t, i_d * 1e3, lw=0.6, label="i_d"); ax[1].plot(t, i_q * 1e3, lw=0.6, label="i_q")
    ax[1].plot(t, ref * 1e3, lw=0.8, label="i_q*", alpha=.7)
    ax[1].set_ylabel("mA"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(np.degrees(centers), fa, "o-", ms=3, label="e_α/(λω)")
    ax[2].plot(np.degrees(centers), fb, "o-", ms=3, label="e_β/(λω)")
    ax[2].plot(np.degrees(centers), -np.sin(centers), "k--", lw=0.8, label="−sin θ (ideal)")
    ax[2].plot(np.degrees(centers), np.cos(centers), "k:", lw=0.8, label="cos θ (ideal)")
    ax[2].set_xlabel("θ_e [deg]"); ax[2].set_ylabel("forma"); ax[2].legend(ncol=4); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(args.png, dpi=120)
    print("plot:", args.png)


if __name__ == "__main__":
    main()
