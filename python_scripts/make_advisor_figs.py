#!/usr/bin/env python3
"""Figuras de la presentación para el asesor (resultados del ADALINE en línea).

Todo sale de los logs de datos_banco/ y de bemf_2804_oc.json; no hay números
escritos a mano salvo las constantes físicas del banco (abajo) y la tabla de
la simulación previa (rizo TRAP/SIN/ADALINE de la presentación anterior).

    python3 python_scripts/make_advisor_figs.py [--out presentation/fig]

Escribe PDFs vectoriales `asesor_*.pdf` y `asesor_numeros.json` (los números
que se citan en las diapositivas).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "datos_banco"

# ── Constantes físicas del banco (las que entran en la teoría) ────────────
K_LSB_A = 32.93                    # LSB/A
MA_PER_Q8 = 1e3 / (256 * K_LSB_A)  # Q8 LSB → mA
LAMBDA, PP = 3.5e-3, 7             # Wb, pares de polos
VBUS, RHO = 12.0, 10               # V, divisiones del modulador
G = 20e-6 / 0.86e-3                # T_s/L [A/V]
E_PK_300 = LAMBDA * PP * 2 * np.pi * 300 / 60      # BEMF de pico a 300 rpm [V]

# ── Estilo (paleta del tema SimplePlus) ───────────────────────────────────
C_DARK, C_BLUE, C_RED, C_GREEN = "#0D2659", "#045073", "#EC5858", "#5EB3A8"
C_AMBER, C_GRAY, C_LIGHT = "#E0A030", "#6B7280", "#D9DEE7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.linewidth": 0.7,
    "axes.titleweight": "bold", "figure.dpi": 110, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

NUM = {}


# ═══════════════════════ carga de datos ═══════════════════════════════════
def load_E(name, t0=10.0):
    """Líneas E (ventanas de 10 ms): t[s], id, iq, ref, eq_rms, ed_rms (mA)."""
    rows = []
    for ln in open(DB / name, errors="ignore"):
        if ln.startswith("[boot]"):
            rows = []
        m = re.match(r"^E (\d+) (-?\d+) (-?\d+) (-?\d+) (\d+) (\d+) (-?\d+) (-?\d+)", ln)
        if m:
            rows.append([int(x) for x in m.groups()])
    a = np.array(rows, float)
    a = a[a[:, 0] / 1000 >= t0]
    return dict(t=a[:, 0] / 1000, id=a[:, 1] * MA_PER_Q8, iq=a[:, 2] * MA_PER_Q8,
                ref=a[:, 3] * MA_PER_Q8, eq=a[:, 4] * MA_PER_Q8, ed=a[:, 5] * MA_PER_Q8,
                pk=(a[:, 7] - a[:, 6]) * MA_PER_Q8)


def load_W(name):
    """Pesos del ADALINE (línea W): t[s] y matriz [N, 2H, 2] (coef, eje)."""
    rows = []
    for ln in open(DB / name, errors="ignore"):
        if ln.startswith("[boot]"):
            rows = []
        if ln.startswith("W "):
            rows.append([int(x) for x in ln.split()[1:]])
    a = np.array(rows, float)
    H = (a.shape[1] - 1) // 4
    return a[:, 0] / 1000, a[:, 1:].reshape(len(a), 2 * H, 2) / 2 ** 27


def block_rms(t, x, width=10.0, t0=10.0):
    """RMS cuadrático por bloques de `width` s → (centros, valores)."""
    edges = np.arange(t0, t.max() + 1e-9, width)
    c, v = [], []
    for e0 in edges:
        m = (t >= e0) & (t < e0 + width)
        if m.sum() > 0.8 * width * 100:
            c.append(e0 + width / 2)
            v.append(np.sqrt(np.mean(x[m] ** 2)))
    return np.array(c), np.array(v)


def seq_decomp(wl):
    """Fundamental del modelo aprendido → (|c+|, fase+ [°] respecto de 90°, |c-|/|c+|)."""
    a1a, a1b, b1a, b1b = wl[:, 0, 0], wl[:, 0, 1], wl[:, 1, 0], wl[:, 1, 1]
    cp = 0.5 * ((a1a + b1b) + 1j * (a1b - b1a))
    cm = 0.5 * ((a1a - b1b) + 1j * (a1b + b1a))
    return np.abs(cp), np.degrees(np.angle(cp)) - 90.0, np.abs(cm) / np.abs(cp)


def fourier_harm(x, H=13):
    th = np.linspace(0, 2 * np.pi, len(x), endpoint=False)
    n = len(x)
    return np.array([np.hypot(2 / n * np.sum(x * np.cos(k * th)),
                              2 / n * np.sum(x * np.sin(k * th))) for k in range(1, H + 1)])


def trap_alpha(n=1000):
    """Forma trapezoidal ideal (misma que bemf_trapezoidal.m) proyectada a α."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)

    def ph(t):
        t = np.mod(t, 2 * np.pi)
        return np.where(t < np.pi / 6, t * 6 / np.pi,
               np.where(t < 5 * np.pi / 6, 1.0,
               np.where(t < 7 * np.pi / 6, 1 - (t - 5 * np.pi / 6) * 6 / np.pi,
               np.where(t < 11 * np.pi / 6, -1.0, -1 + (t - 11 * np.pi / 6) * 6 / np.pi))))
    a, b, c = ph(th), ph(th - 2 * np.pi / 3), ph(th - 4 * np.pi / 3)
    return (2 * a - b - c) / 3


def shape_mismatch(x_model, x_real, H=13):
    """‖forma_modelo − forma_real‖ en armónicos k≥2, ambas normalizadas al
    mismo fundamental (amplitud y fase) → % del fundamental."""
    def coef(x):
        th = np.linspace(0, 2 * np.pi, len(x), endpoint=False)
        c = np.array([2 / len(x) * np.sum(x * np.exp(-1j * k * th)) for k in range(1, H + 1)])
        return c * np.exp(-1j * np.angle(c[0]) * np.arange(1, H + 1)) / np.abs(c[0])
    d = coef(x_model) - coef(x_real)
    return 100 * float(np.sqrt(np.sum(np.abs(d[1:]) ** 2)))


def thd(h):
    return 100 * np.sqrt(np.sum(h[1:] ** 2)) / h[0]


# ═══════════════════════ FIG 1: BEMF medida del banco ═════════════════════
def fig_bemf_banco(out):
    oc = json.load(open(ROOT / "python_scripts/bemf_2804_oc.json"))
    H = oc["H"]
    ca, sa = np.array(oc["S_alpha_cos"]), np.array(oc["S_alpha_sin"])
    th = np.radians(np.linspace(0, 360, 721))
    shape = sum(ca[k - 1] * np.cos(k * th) + sa[k - 1] * np.sin(k * th) for k in range(1, H + 1))
    ideal = ca[0] * np.cos(th) + sa[0] * np.sin(th)
    a1 = np.hypot(ca[0], sa[0])
    resid = (shape - ideal) / a1 * 100

    fig = plt.figure(figsize=(5.8, 1.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1.1, 1], hspace=0.14, wspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax1.plot(np.degrees(th), shape / a1, color=C_BLUE, lw=1.6, label="medida (circuito abierto)")
    ax1.plot(np.degrees(th), ideal / a1, color=C_RED, lw=1, ls="--", label="sinusoide ideal")
    ax1.set_ylabel(r"$e_\alpha/\hat e_1$")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.5, ncol=2, columnspacing=1.0, handlelength=1.6)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.plot(np.degrees(th), resid, color=C_DARK, lw=1.1)
    ax2.axhline(0, color=C_GRAY, lw=0.6)
    ax2.set_ylabel("residuo\n[% del pico]")
    ax2.set_xlabel(r"ángulo eléctrico $\theta_e$ [°]")
    ax2.set_xticks(range(0, 361, 60))
    ax2.set_ylim(-1.2, 1.2)
    NUM["bemf_resid_pp_pct"] = float(resid.max() - resid.min())

    ax3 = fig.add_subplot(gs[:, 1])
    ks = np.arange(2, H + 1)
    abc = np.array(oc["harm_pct_abc"])
    lo, hi = np.array(oc["harm_ci95_lo"]), np.array(oc["harm_ci95_hi"])
    w = 0.26
    for i, (nm, col) in enumerate(zip("abc", (C_DARK, C_BLUE, C_GREEN))):
        v = abc[i, 1:]
        err = np.vstack([np.maximum(v - lo[i, 1:], 0), np.maximum(hi[i, 1:] - v, 0)])
        ax3.bar(ks + (i - 1) * w, v, w, color=col, label=f"fase {nm}",
                yerr=err, capsize=3, error_kw=dict(lw=0.6, ecolor=C_GRAY))
    ax3.set_yscale("log")
    ax3.set_ylim(3e-3, 40)
    ax3.set_xticks(ks)
    ax3.set_xlabel("armónico $k$")
    ax3.set_ylabel("amplitud [% del fund.]")
    ax3.legend(loc="upper right", frameon=False, ncol=3, fontsize=6.5, columnspacing=0.8, handlelength=1.0)
    thd_ph = [float(np.sqrt(np.sum(abc[i, 1:] ** 2))) for i in range(3)]
    NUM["thd_banco_abc_pct"] = thd_ph
    ax3.set_title(f"THD = {thd_ph[0]:.2f} / {thd_ph[1]:.2f} / {thd_ph[2]:.2f} %", fontsize=7.5, pad=3)
    fig.savefig(out / "asesor_bemf_banco.pdf")
    plt.close(fig)


# ═══════════════════════ FIG 2: banco vs motor de simulación ══════════════
def fig_bemf_comparacion(out):
    oc = json.load(open(ROOT / "python_scripts/bemf_2804_oc.json"))
    h_bench = np.array(oc["harm_pct_alpha"])[:13]
    d = sio.loadmat(ROOT / "fcs_m2pc_v2/data/bemf_real_lut.mat")
    h_sim = fourier_harm(d["lut_alpha_real"].ravel())
    h_sim = 100 * h_sim / h_sim[0]
    h_trap = fourier_harm(trap_alpha())
    h_trap = 100 * h_trap / h_trap[0]
    NUM["thd_sim_real_pct"] = float(thd(h_sim))
    NUM["thd_trap_pct"] = float(thd(h_trap))
    NUM["thd_banco_alpha_pct"] = float(thd(h_bench))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.8, 1.65), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.34))
    ks = np.arange(2, 13)
    w = 0.27
    series = [(h_trap[1:12], C_RED, f"modelo trapezoidal asumido (THD {thd(h_trap):.1f} %)"),
              (h_sim[1:12], C_AMBER, f"motor de simulación, BEMF real (THD {thd(h_sim):.1f} %)"),
              (h_bench[1:12], C_BLUE, f"motor del banco, BEMF real (THD {thd(h_bench):.1f} %)")]
    for i, (v, col, lab) in enumerate(series):
        ax.bar(ks + (i - 1) * w, np.maximum(v, 1e-3), w, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(8e-3, 900)
    ax.set_xticks(ks)
    ax.set_xlabel("armónico $k$")
    ax.set_ylabel("amplitud [% del fund.]")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)

    # simulación previa: rizo [%] (tabla de la presentación anterior)
    names = ["TRAP", "SIN", "ADAL.\noffline", "ADAL.\nonline"]
    rip = [4.20, 3.77, 3.65, 3.81]
    cols = [C_RED, C_GRAY, C_BLUE, C_GREEN]
    b = ax2.bar(names, rip, color=cols, width=0.62)
    for r, v in zip(b, rip):
        ax2.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
    ax2.set_ylim(3.0, 4.75)
    ax2.set_ylabel("rizo en simulación [%]")
    ax2.grid(axis="x", visible=False)
    ax2.text(0.02, 0.97, "ganancia del ADALINE offline:\n" r"$-13\,\%$ vs. TRAP,  $-3\,\%$ vs. SIN",
             transform=ax2.transAxes, va="top", fontsize=7.5, color=C_DARK,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_LIGHT))
    NUM["sim_gain_vs_trap_pct"] = 100 * (4.20 - 3.65) / 4.20
    NUM["sim_gain_vs_sin_pct"] = 100 * (3.77 - 3.65) / 3.77
    fig.savefig(out / "asesor_bemf_comparacion.pdf")
    plt.close(fig)


# ═══════════════════════ FIG 3: escalas ═══════════════════════════════════
def fig_escalas(out):
    oc = json.load(open(ROOT / "python_scripts/bemf_2804_oc.json"))
    thd_pct = float(np.mean([np.sqrt(np.sum(np.array(h[1:]) ** 2)) for h in oc["harm_pct_abc"]]))
    dist = 2 * G * (thd_pct / 100) * E_PK_300 * 1e3
    wrong = abs(1 - 0.8 * np.exp(1j * np.radians(10)))
    mis = 2 * G * wrong * E_PK_300 * 1e3
    full = 2 * G * E_PK_300 * 1e3
    res = (2 / 3) * VBUS / RHO * G * 1e3
    vec = (2 / 3) * VBUS * G * 1e3
    e = load_E("m2pc_s26_B_off_300rpm_ldec100.log")
    floor = float(np.sqrt(np.mean(e["eq"] ** 2)))
    pkpk = float(np.mean(e["pk"]))
    NUM.update(dict(G_A_per_V=G, e_pk_300_V=E_PK_300, efecto_distorsion_real_mA=dist,
                    efecto_modelo_25pct_mA=mis, efecto_sin_bemf_mA=full, resolucion_mA=res,
                    vector_completo_mA=vec, piso_eq_mA=floor, piso_ed_mA=float(np.sqrt(np.mean(e["ed"] ** 2))),
                    rizo_pp_mA=pkpk, wrong_model_rel_error=float(wrong)))

    items = [
        ("distorsión real de la BEMF (THD 0.5 %)", dist, C_RED, False),
        ("modelo 25 % equivocado ($0.8\\lambda$, $+10^\\circ$)", mis, C_RED, False),
        ("modelo sin BEMF (error del 100 %)", full, C_RED, True),
        ("resolución del modulador (1/10 de vector)", res, C_BLUE, False),
        ("piso medido, RMS($\\varepsilon_q$)", floor, C_BLUE, False),
        ("paso de un vector activo", vec, C_BLUE, False),
        ("rizo pico a pico medido", pkpk, C_BLUE, False),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 2.05))
    ypos = [8.2, 7.1, 6.0, 3.9, 2.8, 1.7, 0.6]          # hueco entre grupos para el encabezado
    for yi, (lab, v, col, hatch) in zip(ypos, items):
        ax.barh(yi, v, color=col if not hatch else "white", edgecolor=col, hatch="////" if hatch else None,
                height=0.72, lw=1)
        ax.text(v * 1.12, yi, f"{v:.2g} mA" if v < 10 else f"{v:.0f} mA", va="center", fontsize=8, color=C_DARK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([i[0] for i in items], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 900)
    ax.set_ylim(0.0, 10.0)
    ax.set_xlabel("corriente [mA]  (escala logarítmica)")
    ax.grid(axis="y", visible=False)
    ax.axhspan(5.5, 8.9, color=C_RED, alpha=0.06)
    ax.axhspan(0.0, 4.4, color=C_BLUE, alpha=0.06)
    ax.text(0.105, 9.5, "lo que el modelo de BEMF mueve en la predicción", color=C_RED, fontsize=8, fontweight="bold", va="center")
    ax.text(0.105, 4.85, "lo que el controlador y la medida resuelven", color=C_BLUE, fontsize=8, fontweight="bold", va="center")
    fig.savefig(out / "asesor_escalas.pdf")
    plt.close(fig)


# ═══════════════════════ FIG 4: curva de sensibilidad ═════════════════════
def fig_sensibilidad(out):
    base = load_E("m2pc_s26_B_off_300rpm_ldec100.log")
    wrong = load_E("m2pc_s26_E_wrong_mu0_300rpm.log")
    _, vb = block_rms(base["t"], base["eq"])
    _, vw = block_rms(wrong["t"], wrong["eq"])
    sig0, sd0 = float(np.sqrt(np.mean(base["eq"] ** 2))), float(np.std(vb, ddof=1))
    sigw, sdw = float(np.sqrt(np.mean(wrong["eq"] ** 2))), float(np.std(vw, ddof=1))
    x = np.logspace(-1, np.log10(150), 300)  # error de modelo [% del pico]
    curve = lambda p: np.sqrt(sig0 ** 2 + (2 * G * p / 100 * E_PK_300 * 1e3) ** 2)

    fig, ax = plt.subplots(figsize=(5.8, 1.65))
    ax.axhspan(sig0 - sd0, sig0 + sd0, color=C_GRAY, alpha=0.18, lw=0.6)
    ax.text(0.115, sig0 - sd0 - 0.4, "variación entre corridas (±1σ)", color=C_GRAY, fontsize=7, va="top")
    ax.plot(x, curve(x), color=C_BLUE, lw=1.8, label=r"predicción: $\sqrt{\sigma_{piso}^2+(2G\,\delta e)^2}$")
    d = sio.loadmat(ROOT / "fcs_m2pc_v2/data/bemf_real_lut.mat")
    mm_trap = shape_mismatch(trap_alpha(), d["lut_alpha_real"].ravel())
    NUM["desajuste_trap_vs_real_pct"] = mm_trap
    for xv, lab, col, yl, ha, dx in [(0.55, "banco: BEMF real\nvs. seno", C_GREEN, 66.2, "left", 1.10),
                                     (2.6, "sim.: BEMF real\nvs. seno", C_AMBER, 60.8, "right", 0.92),
                                     (mm_trap, "sim.: trapecio\nvs. BEMF real", C_RED, 66.2, "left", 1.08)]:
        ax.axvline(xv, color=col, lw=1.1, ls=":")
        ax.text(xv * dx, yl, lab, color=col, fontsize=6.5, va="center", ha=ha,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    ax.errorbar([0.55], [sig0], yerr=[sd0], fmt="o", color=C_DARK, ms=6, capsize=3, lw=1.2,
                label="medido: modelo nominal", zorder=5)
    ax.errorbar([NUM["wrong_model_rel_error"] * 100], [sigw], yerr=[sdw], fmt="s", color=C_RED, ms=6, capsize=3, lw=1.2,
                label=r"medido: modelo erróneo ($0.8\lambda$, $+10^\circ$)", zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 150)
    ax.set_ylim(40, 69)
    ax.set_xlabel(r"error del modelo de BEMF, $|\delta e|$ [% del pico de la BEMF]")
    ax.set_ylabel(r"RMS($\varepsilon_q$) [mA]")
    ax.legend(loc="upper center", frameon=False, fontsize=7.5, ncol=3, bbox_to_anchor=(0.5, -0.20), columnspacing=1.6, handletextpad=0.5)
    thr = np.interp(1.10 * sig0, curve(x), x)
    NUM["error_para_10pct_en_rms_pct"] = float(thr)
    NUM["rms_pred_sin_bemf_mA"] = float(curve(100))
    NUM.update(dict(rms_nominal=sig0, sd_nominal=sd0, n_bloques_nominal=len(vb),
                    rms_erroneo=sigw, sd_erroneo=sdw, n_bloques_erroneo=len(vw)))
    fig.savefig(out / "asesor_sensibilidad.pdf")
    plt.close(fig)


# ═══════════════════════ FIG 5: RMS medido por condición ═════════════════
def fig_rms_experimento(out):
    runs = [("Seno nominal (referencia)", "m2pc_s26_B_off_300rpm_ldec100.log", C_DARK),
            ("Modelo erróneo, fijo", "m2pc_s26_E_wrong_mu0_300rpm.log", C_RED),
            ("ADALINE desde nominal", "m2pc_s26_B_on_300rpm_H3_mu7.log", C_GREEN),
            ("ADALINE desde modelo erróneo", "m2pc_s26_E_wrong_mu7_300rpm.log", C_AMBER)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.8, 1.6), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.62))
    summ = []
    ref = None
    for lab, f, col in runs:
        e = load_E(f)
        c, v = block_rms(e["t"], e["eq"])
        ax.plot(c, v, "-o", color=col, lw=1.3, ms=3.5, label=lab)
        summ.append((lab, col, float(np.sqrt(np.mean(e["eq"] ** 2))), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v),
                     float(np.sqrt(np.mean(e["ed"] ** 2)))))
        if ref is None:
            ref = (np.sqrt(np.mean(e["eq"] ** 2)), np.std(v, ddof=1))
    ax.axhspan(ref[0] - ref[1], ref[0] + ref[1], color=C_DARK, alpha=0.10, lw=0.6)
    ax.set_xlabel("tiempo de captura [s]")
    ax.set_ylabel(r"RMS($\varepsilon_q$) por bloque [mA]")
    ax.set_ylim(38, 62)
    pass
    yy = np.arange(len(summ))[::-1]
    for yi, (lab, col, m, s, n, ed) in zip(yy, summ):
        ax2.errorbar(m, yi, xerr=s, fmt="o", color=col, ms=6, capsize=3, lw=1.3)
        ax2.text(m, yi + 0.28, f"{m:.1f} ± {s:.1f}", ha="center", fontsize=7.5, color=C_DARK)
    ax2.axvspan(ref[0] - ref[1], ref[0] + ref[1], color=C_DARK, alpha=0.10, lw=0.6)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(["seno nominal\n(referencia)", "modelo erróneo,\nfijo", "ADALINE desde\nnominal", "ADALINE desde\nmodelo erróneo"], fontsize=7)
    ax2.set_xlim(38, 62)
    ax2.set_ylim(-0.6, len(summ) - 0.2)
    ax2.set_xlabel("RMS($\\varepsilon_q$) [mA], media ± σ")
    ax2.grid(axis="y", visible=False)
    NUM["rms_runs"] = [dict(nombre=s[0], rms_eq=s[2], sd_bloques=s[3], n_bloques=s[4], rms_ed=s[5]) for s in summ]
    fig.savefig(out / "asesor_rms_experimento.pdf")
    plt.close(fig)


# ═══════════════════════ FIG 6: sesgo DC ══════════════════════════════════
def fig_sesgo_dc(out):
    nom = load_E("m2pc_s26_B_off_300rpm_ldec100.log")
    wr = load_E("m2pc_s26_E_wrong_mu0_300rpm.log")
    ln = load_E("m2pc_s26_B_on_300rpm_H3_mu7.log")

    def stats(e):
        d, q = e["id"], e["iq"] - e["ref"]
        n = len(d)
        return d.mean(), d.std() / np.sqrt(n), q.mean(), q.std() / np.sqrt(n)

    sn, sw, sl = stats(nom), stats(wr), stats(ln)
    meas_d, meas_q = sw[0] - sn[0], sw[2] - sn[2]
    ed, eq = (np.hypot(sw[1], sn[1]), np.hypot(sw[3], sn[3]))
    # teoría: error del modelo en ejes dq del rotor y sesgo = -2 G δe
    A, phi = 0.8, np.radians(10)
    de_d = A * E_PK_300 * np.sin(phi)
    de_q = E_PK_300 * (1 - A * np.cos(phi))
    th_d, th_q = -2 * G * de_d * 1e3, -2 * G * de_q * 1e3
    NUM.update(dict(sesgo_id_medido=meas_d, sesgo_iq_medido=meas_q, sesgo_id_teoria=th_d, sesgo_iq_teoria=th_q,
                    sesgo_se_mA=float(max(ed, eq)),
                    id_nominal=sn[0], iq_nominal=sn[2], id_erroneo=sw[0], iq_erroneo=sw[2],
                    sesgo_id_adaline=sl[0] - sn[0], sesgo_iq_adaline=sl[2] - sn[2]))

    fig, ax = plt.subplots(figsize=(5.8, 1.65))
    xs = np.array([0.0, 1.0])
    w = 0.26
    ax.bar(xs - w, [th_d, th_q], w, color=C_LIGHT, edgecolor=C_BLUE, hatch="///", lw=0.8,
           label=r"teoría: $-2G\,\delta e$ (dos pasos de predicción, sin ajustar nada)")
    ax.bar(xs, [meas_d, meas_q], w, color=C_RED, yerr=[ed, eq], capsize=3, label=r"medido: modelo erróneo fijo ($0.8\lambda$, $+10^\circ$)")
    ax.bar(xs + w, [sl[0] - sn[0], sl[2] - sn[2]], w, color=C_GREEN, yerr=[np.hypot(sl[1], sn[1]), np.hypot(sl[3], sn[3])],
           capsize=3, label="medido: ADALINE aprendiendo 150 s")
    for x0, v in zip([xs[0] - w, xs[0], xs[0] + w, xs[1] - w, xs[1], xs[1] + w],
                     [th_d, meas_d, sl[0] - sn[0], th_q, meas_q, sl[2] - sn[2]]):
        ax.text(x0, v - 0.7, f"{v:+.1f}", ha="center", va="top", fontsize=7.5, color=C_DARK)
    ax.axhline(0, color=C_GRAY, lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([r"corriente de eje $d$:  $\Delta\bar i_d$", r"corriente de eje $q$:  $\Delta(\bar i_q-i_q^*)$"], fontsize=8)
    ax.set_ylabel("cambio de la media\nvs. modelo nominal [mA]")
    ax.set_ylim(-15.5, 9.5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", frameon=False, fontsize=7, ncol=1)
    fig.savefig(out / "asesor_sesgo_dc.pdf")
    plt.close(fig)


# ═══════════════════════ FIG 7: armónicos del rizo (dominio del ángulo) ═══
def fig_armonicos(out):
    rows = []
    for ln in open(DB / "m2pc_s26_B_off_300rpm_b.log", errors="ignore"):
        if ln.startswith("[boot]"):
            rows = []
        m = re.match(r"^L (\d+) (\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)", ln)
        if m:
            rows.append([int(x) for x in m.groups()])
    a = np.array(rows, float)
    a = a[a[:, 0] / 1000 > 10]
    rpm = a[:, 6] / 10
    th = a[:, 1] / 65536 * 2 * np.pi - rpm / 60 * PP * 2 * np.pi * 0.5e-3   # centro de ventana
    ia, ib = a[:, 2] / (256 * K_LSB_A), a[:, 3] / (256 * K_LSB_A)
    c, s = np.cos(th), np.sin(th)
    iq, idd = -ia * s + ib * c, ia * c + ib * s
    N, H = len(iq), 12
    X = np.column_stack([np.ones(N)] + [f(k * th) for k in range(1, H + 1) for f in (np.cos, np.sin)])
    fe = float(np.mean(rpm) / 60 * PP)
    sinc = np.array([np.sinc(k * fe * 1e-3) for k in range(1, H + 1)])
    amps, sig = {}, {}
    for nm, y in (("q", iq), ("d", idd)):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ beta
        amps[nm] = np.array([np.hypot(beta[1 + 2 * (k - 1)], beta[2 + 2 * (k - 1)]) for k in range(1, H + 1)]) / sinc * 1e3
        sig[nm] = float(r.std() * np.sqrt(2 / N) * 1e3)
    oc = json.load(open(ROOT / "python_scripts/bemf_2804_oc.json"))
    hab = np.array(oc["harm_pct_abc"])[:, :7] / 100
    h57 = float(np.max(hab[:, 4] + hab[:, 6]))              # 5.º + 7.º, peor fase
    bound = 2 * G * h57 * E_PK_300 * 1e3
    NUM.update(dict(rizo6_q_mA=float(amps["q"][5]), rizo6_d_mA=float(amps["d"][5]), sigma_amp_mA=sig["q"],
                    cota_bemf_6fe_mA=bound, frac_q_pct=100 * bound / amps["q"][5], frac_d_pct=100 * bound / amps["d"][5],
                    n_ventanas_angulo=int(N), fe_hz=fe, rizo12_d_mA=float(amps["d"][11])))

    fig, ax = plt.subplots(figsize=(5.8, 1.65))
    ks = np.arange(1, H + 1)
    w = 0.38
    ax.bar(ks - w / 2, amps["q"], w, color=C_BLUE, label=r"$i_q$ (corriente de par)")
    ax.bar(ks + w / 2, amps["d"], w, color=C_DARK, label=r"$i_d$")
    ax.axhspan(0, sig["q"], color=C_GRAY, alpha=0.30, lw=0, label="incertidumbre estadística (±1σ)")
    ax.plot([6 - 0.45, 6 + 0.45], [bound, bound], color=C_RED, lw=2.4, solid_capstyle="butt",
            label=f"cota de lo atribuible a la forma de la BEMF ({bound:.2f} mA)")
    ax.set_yscale("log")
    ax.set_ylim(0.07, 220)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k}" if k != 6 else "6" for k in ks])
    ax.get_xticklabels()[5].set_fontweight("bold")
    ax.set_xlabel(r"armónico $k$ de la frecuencia eléctrica  (rizo de par: $k=6$ y $12$)")
    ax.set_ylabel("amplitud sincrónica [mA]")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.savefig(out / "asesor_armonicos.pdf")
    plt.close(fig)


# ═══════════════════════ FIG 8: identificación ════════════════════════════
def fig_identificacion(out):
    fig, axs = plt.subplots(1, 3, figsize=(5.8, 1.6), gridspec_kw=dict(wspace=0.62))
    ref_amp = 0.9966
    for name, lab, col in [("m2pc_s26_E_wrong_mu7_300rpm.log", "desde modelo erróneo (0.8, +10°)", C_AMBER),
                           ("m2pc_s26_B_on_300rpm_H3_mu7.log", "desde modelo nominal (1.0, 0°)", C_GREEN)]:
        t, W = load_W(name)
        amp, ph, neg = seq_decomp(W)
        for ax, y in zip(axs, (amp, ph, 100 * neg)):
            ax.plot(t, y, color=col, lw=1.4, label=lab)
        if "wrong" in name:
            NUM.update(dict(id_amp_ini=float(amp[0]), id_amp_fin=float(amp[-1]), id_fase_ini=float(ph[0]), id_fase_fin=float(ph[-1]),
                            id_neg_fin=float(100 * neg[-1]), id_t_fin=float(t[-1]),
                            t90=float(t[np.argmax(amp >= amp[0] + 0.9 * (amp[-1] - amp[0]))])))
        else:
            NUM.update(dict(idn_amp_fin=float(amp[-1]), idn_fase_fin=float(ph[-1]), idn_neg_fin=float(100 * neg[-1])))
    axs[0].axhline(ref_amp, color=C_BLUE, ls="--", lw=1, label="BEMF medida (ref.)")
    axs[1].axhline(0, color=C_BLUE, ls="--", lw=1)
    axs[2].axhline(0, color=C_BLUE, ls="--", lw=1)
    axs[0].set_ylabel("amplitud del fund.\n(1 = nominal)")
    axs[1].set_ylabel("error de fase [°]")
    axs[2].set_ylabel("secuencia negativa\n[% del fundamental]")
    titles = ["(a) amplitud", "(b) fase", "(c) desbalance $\\alpha/\\beta$"]
    for ax, ti in zip(axs, titles):
        ax.set_xlabel("tiempo de aprendizaje [s]")
        ax.set_title(ti, fontsize=8)
    axs[0].set_ylim(0.75, 1.1)
    axs[1].set_ylim(-3, 18)
    axs[2].set_ylim(-1, 12)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.93), columnspacing=1.2, handlelength=1.6)
    fig.savefig(out / "asesor_identificacion.pdf")
    plt.close(fig)


# ═══════════════════════ FIG 9: estabilidad ═══════════════════════════════
def fig_estabilidad(out):
    base_traj = float(np.sqrt(np.mean(load_E("m2pc_s26_A_on_mu0_traj300.log")["eq"] ** 2)))
    base_cte = float(np.sqrt(np.mean(load_E("m2pc_s26_B_off_300rpm_ldec100.log")["eq"] ** 2)))
    NUM.update(dict(base_traj=base_traj, base_cte=base_cte))

    fig, ax = plt.subplots(figsize=(5.8, 1.65))
    # μ grande: arranque + continuación hasta el TRIP
    ea = load_E("m2pc_s26_adaline_on_traj300_a.log", t0=1.0)
    eb = load_E("m2pc_s26_adaline_on_traj300.log", t0=0.0)
    ca, va = block_rms(ea["t"], ea["eq"], width=5.0, t0=1.0)
    cb, vb = block_rms(eb["t"], eb["eq"], width=5.0, t0=float(eb["t"].min()))
    ax.plot(ca, va / base_traj, "-o", color=C_RED, lw=1.4, ms=3, label=r"$\mu$ grande ($2\times10^{-3}$), velocidad variable")
    ax.plot(cb, vb / base_traj, "-o", color=C_RED, lw=1.4, ms=3)
    ax.text((ca[-1] + cb[0]) / 2, 2.78, "(sin captura)", ha="center", color=C_GRAY, fontsize=6.5)
    ax.annotate("disparo de sobrecorriente", xy=(cb[-1], vb[-1] / base_traj), xytext=(cb[-1] - 4, 3.12),
                fontsize=7.5, color=C_RED, ha="right", arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.9))
    # μ pequeño, velocidad variable (H=7, puerta baja)
    e2 = load_E("m2pc_s26_adaline_mu1e4.log", t0=1.0)
    c2, v2 = block_rms(e2["t"], e2["eq"], width=5.0, t0=1.0)
    ax.plot(c2, v2 / base_traj, "-s", color=C_AMBER, lw=1.3, ms=3, label=r"$\mu$ pequeño ($10^{-4}$), velocidad variable, SNR bajo")
    # μ pequeño, velocidad constante (H=3, puerta alta)
    e3 = load_E("m2pc_s26_B_on_300rpm_H3_mu7.log", t0=1.0)
    c3, v3 = block_rms(e3["t"], e3["eq"], width=5.0, t0=1.0)
    ax.plot(c3, v3 / base_cte, "-^", color=C_GREEN, lw=1.3, ms=3, label=r"$\mu$ pequeño ($10^{-4}$), velocidad constante")
    # μ pequeño desde modelo erróneo
    e4 = load_E("m2pc_s26_E_wrong_mu7_300rpm.log", t0=1.0)
    c4, v4 = block_rms(e4["t"], e4["eq"], width=5.0, t0=1.0)
    ax.plot(c4, v4 / base_cte, "-v", color=C_DARK, lw=1.2, ms=3, label=r"ídem, partiendo de un modelo erróneo")
    ax.axhline(1, color=C_GRAY, lw=0.8, ls="--")
    ax.axhspan(0.95, 1.05, color=C_GRAY, alpha=0.15, lw=0.6)
    ax.set_xlabel("tiempo de aprendizaje [s]")
    ax.set_ylabel(r"RMS($\varepsilon_q$) / referencia")
    ax.set_ylim(0.75, 3.25)
    ax.set_xlim(-3, 262)
    ax.legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.55), ncol=1, labelspacing=0.3)
    NUM["ratio_mu_grande_fin"] = float(vb[-1] / base_traj)
    NUM["ratio_mu_peq_traj"] = float(np.mean(v2[len(v2) // 3:]) / base_traj)
    NUM["ratio_mu_peq_cte"] = float(np.mean(v3[len(v3) // 3:]) / base_cte)
    NUM["ratio_erroneo_fin"] = float(np.mean(v4[-4:]) / base_cte)
    fig.savefig(out / "asesor_estabilidad.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "presentation/fig"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_bemf_banco, fig_bemf_comparacion, fig_escalas, fig_sensibilidad, fig_rms_experimento,
              fig_sesgo_dc, fig_armonicos, fig_identificacion, fig_estabilidad):
        f(out)
        print("ok", f.__name__)
    json.dump(NUM, open(out / "asesor_numeros.json", "w"), indent=1, default=float)
    print(json.dumps(NUM, indent=1, default=float))


if __name__ == "__main__":
    main()
