#!/usr/bin/env python3
"""Figuras del Cap. 6 (validación experimental en banco).

    uv run --no-project --with scipy --with numpy --with matplotlib make_cap6_figs.py

Lee los logs de datos_banco/ (restaurarlos con el README de esa carpeta), la BEMF en
circuito abierto (bemf_2804_oc.json) y la tabla de la planta del Cap. 5
(fcs_m2pc_v2/data/bemf_lut.mat). Reutiliza los cargadores de make_advisor_figs.py y el
cálculo de kappa de cociente_prediccion.py. Escribe PDF vectoriales c6_*.pdf y
c6_numeros.json en tesis_documento/Documentos/Chapters/Capitulo6/FigureC6/.

Una figura, un mensaje. Colores con significado fijo en todo el capítulo:
  azul    = banco (2804)                 naranja = simulación (planta del Cap. 5)
  aqua    = ADALINE desde el nominal     violeta = ADALINE desde el modelo equivocado
  negro/gris = modelos fijos y referencias
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import cociente_prediccion as kap
from make_advisor_figs import (DB, E_PK_300, G, K_LSB_A, MA_PER_Q8, PP, ROOT, block_rms, load_E,
                               load_W, seq_decomp)

OUT = ROOT / "tesis_documento/Documentos/Chapters/Capitulo6/FigureC6"

# Paleta validada (dataviz/validate_palette.js, modo claro: todas las pruebas pasan;
# aqua con contraste < 3:1 -> siempre con etiqueta directa).
AZUL, NARANJA, AQUA, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
TINTA, GRIS, GRIS_CLARO = "#1f1f1f", "#7a7a76", "#d6d6d2"

W_TEXTO = 6.3  # ancho del cuerpo de texto [in] (165 mm)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": GRIS, "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.labelcolor": TINTA, "text.color": TINTA,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRIS_CLARO, "grid.linewidth": 0.5,
    "lines.linewidth": 1.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "axes.formatter.use_locale": False,
})
NUM = {}


def guardar(fig, nombre):
    fig.savefig(OUT / f"{nombre}.pdf")
    plt.close(fig)
    print("ok", nombre)


def serie_fourier(c, th):
    """Señal a partir de c_k = (2/N) sum x e^{-jk th}: x = sum_k Re(c_k e^{jk th})."""
    return sum((ck * np.exp(1j * k * th)).real for k, ck in enumerate(c, start=1))


def forma_alineada(x, th, H=40):
    """Coeficientes de Fourier de x, rotados para que el fundamental sea sin(th) con amplitud 1."""
    c = np.array([2 / len(x) * np.sum(x * np.exp(-1j * k * th)) for k in range(1, H + 1)])
    c_sin = 2 / len(th) * np.sum(np.sin(th) * np.exp(-1j * th))
    rot = np.angle(c[0] / c_sin)
    c = c * np.exp(-1j * rot * np.arange(1, H + 1)) / np.abs(c[0])
    return c


# ═════════════ F1 (6.3): la BEMF del banco es una senoide; la planta simulada no ═════════════
def fig_bemf_forma():
    th = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    oc = json.load(open(ROOT / "python_scripts/bemf_2804_oc.json"))
    ca, sa = np.array(oc["S_alpha_cos"]), np.array(oc["S_alpha_sin"])
    banco = sum(ca[k - 1] * np.cos(k * th) + sa[k - 1] * np.sin(k * th) for k in range(1, oc["H"] + 1))
    d = sio.loadmat(ROOT / "fcs_m2pc_v2/data/bemf_lut.mat")
    sim = np.interp(th, d["lut_theta"].ravel(), d["lut_alpha_real"].ravel(), period=2 * np.pi)

    cb, cs = forma_alineada(banco, th, 12), forma_alineada(sim, th, 39)
    fb, fs = serie_fourier(cb, th), serie_fourier(cs, th)
    rb = fb - np.sin(th)
    rs = fs - np.sin(th)
    thd_b = [float(np.sqrt(np.sum(np.array(h[1:]) ** 2))) for h in oc["harm_pct_abc"]]
    thd_s = float(100 * np.sqrt(np.sum(np.abs(cs[1:]) ** 2)))
    NUM.update(thd_banco_pct=thd_b, thd_planta_sim_pct=thd_s,
               resid_banco_max_pct=float(100 * np.abs(rb).max()), resid_sim_max_pct=float(100 * np.abs(rs).max()))

    deg = np.degrees(th)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_TEXTO, 2.1), gridspec_kw=dict(wspace=0.32))
    a1.plot(deg, np.sin(th), color=GRIS, lw=1.0, ls=(0, (4, 2)))
    a1.plot(deg, fs, color=NARANJA)
    a1.plot(deg, fb, color=AZUL)
    a1.set_title("(a) Forma normalizada al fundamental")
    a1.set_ylabel(r"$e_\alpha\,/\,|e_{\alpha,1}|$")
    a1.text(95, 1.13, "planta simulada", color=NARANJA, fontsize=7.5, ha="center")
    a1.text(250, 0.35, "banco y\nsenoide ideal\n(superpuestas)", color=AZUL, fontsize=7.5, ha="center")
    a1.set_ylim(-1.35, 1.35)

    a2.axhline(0, color=GRIS, lw=0.6)
    a2.plot(deg, 100 * rs, color=NARANJA)
    a2.plot(deg, 100 * rb, color=AZUL)
    a2.set_title("(b) Diferencia con la senoide ideal")
    a2.set_ylabel(r"% del pico del fundamental")
    a2.text(6, 100 * rs.max() * 0.92, f"planta simulada, THD {thd_s:.1f} %", color=NARANJA, fontsize=7.5, va="top")
    a2.text(215, 3.0, f"banco 2804, THD {min(thd_b):.1f}–{max(thd_b):.1f} %", color=AZUL, fontsize=7.5,
            ha="center", va="bottom", bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
    for ax in (a1, a2):
        ax.set_xlabel(r"ángulo eléctrico $\theta_e$ [°]")
        ax.set_xticks(range(0, 361, 90))
        ax.set_xlim(0, 360)
    guardar(fig, "c6_bemf_forma")


# ═════════════ F2 (6.4): el rizo de conmutación domina la corriente ═════════════
def fig_escalon():
    rows = []
    for ln in open(DB / "m2pc_s23_exp1_300rpm_b.log", errors="ignore"):
        if ln.startswith("[boot]"):
            rows = []
        m = re.match(r"^C (\d+) (-?\d+) (-?\d+)", ln)
        if m:
            rows.append([int(x) for x in m.groups()])
    a = np.array(rows, float)
    t = (a[:, 0] - 128) / 50.0           # ms, flanco en la muestra 128
    iq, ref = a[:, 1] * MA_PER_Q8, a[:, 2] * MA_PER_Q8
    k = 25                                # media móvil de 0.5 ms
    iq_m = np.convolve(iq, np.ones(k) / k, mode="same")
    post = (t > 1.0) & (t < 18)
    NUM.update(escalon_pp_post_mA=float(iq[post].max() - iq[post].min()),
               escalon_rms_post_mA=float(np.sqrt(np.mean((iq[post] - ref[post]) ** 2))),
               escalon_ref_mA=[float(ref[0]), float(ref[-1])])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_TEXTO, 2.1), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.25))
    for ax in (a1, a2):
        h_raw, = ax.plot(t, iq, color=AZUL, lw=0.6, alpha=0.45)
        h_ref, = ax.plot(t, ref, color=TINTA, lw=1.2, drawstyle="steps-post")
    h_med, = a1.plot(t, iq_m, color=AZUL, lw=1.6)
    a1.legend([h_ref, h_raw, h_med], [r"$i_q^*$", r"$i_q$ muestra a muestra (50 kHz)", r"$i_q$, media de 0.5 ms"],
              loc="upper left", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.01, 1.0), handlelength=1.3,
              columnspacing=0.8)
    a1.set_xlim(t.min(), t.max())
    a1.set_title("(a) Escalón de $+100$ mA en $i_q^*$, $300$ rpm")
    a1.set_xlabel("tiempo desde el escalón [ms]")
    a1.set_ylabel("corriente [mA]")
    a1.set_ylim(-150, 470)
    z = (t >= -0.3) & (t <= 0.6)
    a2.plot(t[z], iq[z], "o", color=AZUL, ms=2.6)
    a2.set_xlim(-0.3, 0.6)
    a2.set_ylim(-150, 470)
    a2.set_title("(b) Detalle del flanco")
    a2.set_xlabel("tiempo [ms]")
    a2.axvspan(0, 0.2, color=GRIS_CLARO, alpha=0.6, lw=0)
    a2.text(0.1, a2.get_ylim()[1] * 0.92, "200 µs", ha="center", fontsize=7, color=GRIS, va="top")
    guardar(fig, "c6_escalon")


# ═════════════ F3 (6.5): ningún modelo mueve el error más allá de la dispersión ═════════════
RUNS = [("Senoidal nominal", "m2pc_s26_B_off_300rpm_ldec100.log", TINTA, "o"),
        ("Equivocado ($0.8\\lambda$, $+10^\\circ$), fijo", "m2pc_s26_E_wrong_mu0_300rpm.log", GRIS, "s"),
        ("ADALINE desde el nominal", "m2pc_s26_B_on_300rpm_H3_mu7.log", AQUA, "o"),
        ("ADALINE desde el equivocado", "m2pc_s26_E_wrong_mu7_300rpm.log", VIOLETA, "o")]


def fig_rms_modelos():
    res = []
    for lab, f, col, mk in RUNS:
        e = load_E(f)
        _, v = block_rms(e["t"], e["eq"])
        res.append((lab, col, mk, float(np.sqrt(np.mean(e["eq"] ** 2))), float(np.std(v, ddof=1)), len(v),
                    float(np.sqrt(np.mean(e["ed"] ** 2)))))
    NUM["rms_runs"] = [dict(nombre=r[0], eq=r[3], sd=r[4], n=r[5], ed=r[6]) for r in res]
    m0, s0 = res[0][3], res[0][4]
    pred_eq = float(np.hypot(m0, 2 * G * abs(1 - 0.8 * np.exp(1j * np.radians(10))) * E_PK_300 * 1e3))
    NUM["rms_pred_equivocado_mA"] = pred_eq

    fig, ax = plt.subplots(figsize=(W_TEXTO, 1.9))
    ax.axvspan(m0 - s0, m0 + s0, color=GRIS_CLARO, alpha=0.7, lw=0)
    ax.text(m0, -0.75, "dispersión de la referencia ($\\pm 1\\sigma$)", ha="center", fontsize=7, color=GRIS)
    ys = np.arange(len(res))[::-1]
    for y, (lab, col, mk, m, s, n, ed) in zip(ys, res):
        ax.errorbar(m, y, xerr=s, fmt=mk, color=col, ms=6, capsize=0, elinewidth=1.6, mec="white", mew=0.8)
        ax.text(61.5, y, f"{m:.1f} ± {s:.1f}", va="center", ha="right", fontsize=7.5, color=TINTA)
        ax.text(66.5, y, f"{n}", va="center", ha="right", fontsize=7.5, color=GRIS)
    ax.text(61.5, len(res) - 0.55, "media ± σ [mA]", ha="right", fontsize=7, color=GRIS)
    ax.text(66.5, len(res) - 0.55, "bloques", ha="right", fontsize=7, color=GRIS)
    ax.plot(pred_eq, ys[1], "D", mfc="white", mec=TINTA, ms=5.5, mew=1.0)
    ax.text(pred_eq, ys[1] + 0.32, "predicción", fontsize=7, color=TINTA, ha="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in res])
    for tl, r in zip(ax.get_yticklabels(), res):
        tl.set_color(TINTA)
    ax.set_xlim(39, 67)
    ax.set_xticks(range(40, 57, 4))
    ax.spines["bottom"].set_bounds(39, 57)
    ax.set_ylim(-1.0, len(res) - 0.4)
    ax.set_xlabel(r"error de seguimiento, RMS de $\varepsilon_q$ [mA]")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    guardar(fig, "c6_rms_modelos")


# ═════════════ F4 (6.5): el LMS corrige la amplitud, no la fase ═════════════
def fig_identificacion():
    ref_amp = 3.487 / 3.5    # lambda medido en circuito abierto / lambda nominal del firmware
    fig, axs = plt.subplots(1, 3, figsize=(W_TEXTO, 2.0), gridspec_kw=dict(wspace=0.45))
    series = [("m2pc_s26_B_on_300rpm_H3_mu7.log", "desde el nominal", AQUA),
              ("m2pc_s26_E_wrong_mu7_300rpm.log", "desde el equivocado", VIOLETA)]
    fin = {}
    for f, lab, col in series:
        t, W = load_W(f)
        amp, ph, neg = seq_decomp(W)
        for ax, y in zip(axs, (amp, ph, 100 * neg)):
            ax.plot(t, y, color=col)
        fin[lab] = (t, amp, ph, 100 * neg)
        NUM[f"ident_{lab.split()[-1]}"] = dict(amp_fin=float(amp[-1]), fase_fin=float(ph[-1]),
                                               neg_fin=float(100 * neg[-1]), t_fin=float(t[-1]))
    for ax, r in zip(axs, (ref_amp, 0, 0)):
        ax.axhline(r, color=AZUL, lw=1.0, ls=(0, (4, 2)))
    axs[0].set_title("(a) Amplitud (1 = nominal)")
    axs[1].set_title("(b) Fase [°]")
    axs[2].set_title("(c) Desbalance $\\alpha/\\beta$ [%]")
    axs[0].set_ylim(0.75, 1.08)
    axs[1].set_ylim(-2, 16)
    axs[2].set_ylim(-1, 10)
    axs[0].text(140, ref_amp - 0.035, "medida en\ncircuito abierto", color=AZUL, fontsize=7, ha="center", va="top")
    t, amp, ph, ng = fin["desde el equivocado"]
    axs[0].text(28, 0.80, "desde el\nequivocado", color=VIOLETA, fontsize=7, va="bottom")
    t, amp, ph, ng = fin["desde el nominal"]
    axs[0].text(5, 1.045, "desde el nominal", color=AQUA, fontsize=7)
    for ax in axs:
        ax.set_xlabel("tiempo de aprendizaje [s]")
    guardar(fig, "c6_identificacion")


# ═════════════ F5 (6.5): estabilidad del aprendizaje ═════════════
def fig_estabilidad():
    base_traj = float(np.sqrt(np.mean(load_E("m2pc_s26_A_on_mu0_traj300.log")["eq"] ** 2)))
    base_cte = float(np.sqrt(np.mean(load_E("m2pc_s26_B_off_300rpm_ldec100.log")["eq"] ** 2)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_TEXTO, 2.1), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax in (a1, a2):
        ax.axhline(1, color=GRIS, lw=0.8, ls=(0, (4, 2)))
    # (a) velocidad variable, H = 7
    ea = load_E("m2pc_s26_adaline_on_traj300_a.log", t0=1.0)
    eb = load_E("m2pc_s26_adaline_on_traj300.log", t0=0.0)
    ca, va = block_rms(ea["t"], ea["eq"], width=5.0, t0=1.0)
    cb, vb = block_rms(eb["t"], eb["eq"], width=5.0, t0=float(eb["t"].min()))
    a1.plot(ca, va / base_traj, color=TINTA, marker="x", ms=4)
    a1.plot(cb, vb / base_traj, color=TINTA, marker="x", ms=4)
    a1.plot([ca[-1], cb[0]], [va[-1] / base_traj, vb[0] / base_traj], color=TINTA, lw=0.8, ls=":")
    a1.text(cb[-1], vb[-1] / base_traj - 0.18, "disparo", fontsize=7, color=TINTA, ha="center", va="top")
    a1.text(60, 3.0, r"$\mu = 2\times10^{-3}$ (sin registro" + "\n" + "entre 20 y 105 s)", color=TINTA, fontsize=7.5, ha="center")
    e2 = load_E("m2pc_s26_adaline_mu1e4.log", t0=1.0)
    c2, v2 = block_rms(e2["t"], e2["eq"], width=5.0, t0=1.0)
    a1.plot(c2, v2 / base_traj, color=AQUA)
    a1.text(60, 1.85, r"$\mu = 10^{-4}$", color=AQUA, fontsize=7.5, ha="center")
    a1.set_title("(a) Velocidad variable ($\\pm300$ rpm), $H = 7$")
    # (b) velocidad constante, H = 3
    e3 = load_E("m2pc_s26_B_on_300rpm_H3_mu7.log", t0=1.0)
    c3, v3 = block_rms(e3["t"], e3["eq"], width=5.0, t0=1.0)
    e4 = load_E("m2pc_s26_E_wrong_mu7_300rpm.log", t0=1.0)
    c4, v4 = block_rms(e4["t"], e4["eq"], width=5.0, t0=1.0)
    a2.plot(c3, v3 / base_cte, color=AQUA)
    a2.plot(c4, v4 / base_cte, color=VIOLETA)
    a2.text(150, 0.82, "desde el nominal", color=AQUA, fontsize=7.5, ha="center", va="top")
    a2.text(105, 1.45, "desde el equivocado", color=VIOLETA, fontsize=7.5, ha="center")
    a2.set_title("(b) $300$ rpm constantes, $H = 3$, $\\mu = 10^{-4}$")
    a1.set_ylabel("RMS de $\\varepsilon_q$ / referencia")
    a1.set_ylim(0.6, 3.5)
    for ax in (a1, a2):
        ax.set_xlabel("tiempo de aprendizaje [s]")
    NUM.update(est_mu_grande_fin=float(vb[-1] / base_traj), est_mu_peq_traj=float(np.mean(v2[len(v2) // 3:]) / base_traj),
               est_mu_peq_cte=float(np.mean(v3[len(v3) // 3:]) / base_cte), est_equivocado_fin=float(np.mean(v4[-4:]) / base_cte))
    guardar(fig, "c6_estabilidad")


# ═════════════ F6 (6.6): dónde cae cada caso en el criterio kappa ═════════════
def fig_kappa():
    k = kap.calcular(verbose=False)
    nom = load_E("m2pc_s26_B_off_300rpm_ldec100.log")
    wr = load_E("m2pc_s26_E_wrong_mu0_300rpm.log")
    _, vb = block_rms(nom["t"], nom["eq"])
    _, vw = block_rms(wr["t"], wr["eq"])
    s0, sd0 = float(np.sqrt(np.mean(nom["eq"] ** 2))), float(np.std(vb, ddof=1))
    sw, sdw = float(np.sqrt(np.mean(wr["eq"] ** 2))), float(np.std(vw, ddof=1))
    ripple = 12.0 * G * 1e3        # V_dc * T_s / L en el banco [mA]
    x = np.logspace(-4, 0, 400)
    curva = np.sqrt(s0 ** 2 + (x * ripple) ** 2)
    NUM["kappa"] = {kk: float(v) for kk, v in k.items()}

    fig, ax = plt.subplots(figsize=(W_TEXTO, 2.4))
    ax.axhspan(s0 - sd0, s0 + sd0, color=GRIS_CLARO, alpha=0.7, lw=0)
    ax.axvspan(min(k["sim_SIN_forma"], k["sim_TRAP_forma"]), max(k["sim_SIN_forma"], k["sim_TRAP_forma"]),
               color=NARANJA, alpha=0.18, lw=0)
    ax.axvspan(k["sim_SIN"], k["sim_TRAP"], color=NARANJA, alpha=0.40, lw=0)
    ax.plot(x, curva, color=TINTA, lw=1.4)
    ax.errorbar(k["banco_real"], s0, yerr=sd0, fmt="o", color=AZUL, ms=6, capsize=0, elinewidth=1.6, mec="white")
    ax.errorbar(k["banco_equivocado"], sw, yerr=sdw, fmt="s", color=GRIS, ms=6, capsize=0, elinewidth=1.6, mec="white")
    ax.text(k["banco_real"] * 1.25, s0 + 3.3, "banco: distorsión\nreal del 2804", color=AZUL, fontsize=7.5, va="bottom")
    ax.text(k["banco_equivocado"], sw - sdw - 0.8, "banco: modelo equivocado", color=GRIS, fontsize=7.5, ha="center",
            va="top", bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
    ax.text(np.sqrt(k["sim_SIN_forma"] * k["sim_TRAP_forma"]), 77, "simulación,\nsolo forma", color=NARANJA,
            fontsize=7.5, ha="right", va="top")
    ax.text(np.sqrt(k["sim_SIN"] * k["sim_TRAP"]), 77, "simulación,\nforma y amplitud", color=NARANJA,
            fontsize=7.5, ha="center", va="top")
    ax.text(2.2e-4, s0 - sd0 - 1.0, "dispersión entre corridas", color=GRIS, fontsize=7, va="top")
    ax.text(0.55, 40.5, "predicción\n$\\sqrt{\\sigma^2 + (\\kappa\\,V_{dc}T_s/L)^2}$", color=TINTA, fontsize=7.5,
            ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(1.5e-4, 0.6)
    ax.set_ylim(36, 80)
    ax.set_xlabel(r"$\kappa = 2\,\delta e / V_{dc}$  (error de BEMF relativo a la tensión del bus)")
    ax.set_ylabel("RMS de $\\varepsilon_q$ en el banco [mA]")
    guardar(fig, "c6_kappa")


# ═════════════ F7 (6.6): dónde sí se nota el modelo y de dónde viene el rizo ═════════════
def fig_sesgo_rizo():
    nom, wr, ln = (load_E(f) for f in ("m2pc_s26_B_off_300rpm_ldec100.log", "m2pc_s26_E_wrong_mu0_300rpm.log",
                                        "m2pc_s26_B_on_300rpm_H3_mu7.log"))

    def med(e):
        return e["id"].mean(), (e["iq"] - e["ref"]).mean()

    n, w, l = med(nom), med(wr), med(ln)
    A, phi = 0.8, np.radians(10)
    teo = (-2 * G * A * E_PK_300 * np.sin(phi) * 1e3, -2 * G * E_PK_300 * (1 - A * np.cos(phi)) * 1e3)
    medido = (w[0] - n[0], w[1] - n[1])
    adal = (l[0] - n[0], l[1] - n[1])

    # rizo a 6 f_e (mismo procedimiento que make_advisor_figs.fig_armonicos)
    rows = []
    for line in open(DB / "m2pc_s26_B_off_300rpm_b.log", errors="ignore"):
        if line.startswith("[boot]"):
            rows = []
        m = re.match(r"^L (\d+) (\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)", line)
        if m:
            rows.append([int(x) for x in m.groups()])
    a = np.array(rows, float)
    a = a[a[:, 0] / 1000 > 10]
    rpm = a[:, 6] / 10
    th = a[:, 1] / 65536 * 2 * np.pi - rpm / 60 * PP * 2 * np.pi * 0.5e-3
    ia, ib = a[:, 2] / (256 * K_LSB_A), a[:, 3] / (256 * K_LSB_A)
    c, s = np.cos(th), np.sin(th)
    iq, idd = -ia * s + ib * c, ia * c + ib * s
    H = 12
    X = np.column_stack([np.ones(len(iq))] + [f(k * th) for k in range(1, H + 1) for f in (np.cos, np.sin)])
    fe = float(np.mean(rpm) / 60 * PP)
    sinc6 = np.sinc(6 * fe * 1e-3)
    amp6 = {}
    for nm, y in (("q", iq), ("d", idd)):
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        amp6[nm] = float(np.hypot(b[11], b[12]) / sinc6 * 1e3)
    oc = json.load(open(ROOT / "python_scripts/bemf_2804_oc.json"))
    hab = np.array(oc["harm_pct_abc"])[:, :7] / 100
    cota = float(2 * G * np.max(hab[:, 4] + hab[:, 6]) * E_PK_300 * 1e3)
    NUM.update(sesgo_teo=list(teo), sesgo_medido=list(medido), sesgo_adaline=list(adal),
               rizo6_q=amp6["q"], rizo6_d=amp6["d"], cota6=cota)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_TEXTO, 2.2), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.38))
    xs, bw = np.array([0.0, 1.0]), 0.25
    b1 = a1.bar(xs - bw, teo, bw * 0.92, color="white", edgecolor=TINTA, hatch="////", lw=0.8)
    b2 = a1.bar(xs, medido, bw * 0.92, color=GRIS)
    b3 = a1.bar(xs + bw, adal, bw * 0.92, color=AQUA)
    for x0, v in zip(np.r_[xs - bw, xs, xs + bw], np.r_[teo, medido, adal]):
        a1.text(x0, v - 0.6, f"{v:.1f}", ha="center", va="top", fontsize=7)
    a1.axhline(0, color=TINTA, lw=0.6)
    a1.set_xticks(xs)
    a1.set_xticklabels([r"eje $d$", r"eje $q$"])
    a1.set_ylim(-16, 9)
    a1.set_ylabel("cambio de la corriente media [mA]")
    a1.set_title("(a) Sesgo frente al modelo nominal")
    a1.legend([b1, b2, b3], ["predicción $-2G\\,\\delta e$", "medido, modelo equivocado", "medido, ADALINE (150 s)"],
              loc="upper center", ncol=1, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0), labelspacing=0.25)
    a1.tick_params(axis="x", length=0)

    vals = [amp6["q"], amp6["d"]]
    a2.bar([0, 1], vals, 0.55, color=[TINTA, GRIS])
    for x0, v in zip([0, 1], vals):
        a2.text(x0, v * 1.25, f"{v:.1f} mA", ha="center", fontsize=7.5)
    a2.plot([-0.5, 1.4], [cota, cota], color=AZUL, lw=1.6)
    a2.text(1.45, cota * 1.6, f"cota por la\nforma de la\nBEMF: {cota:.2f} mA", color=AZUL, fontsize=7, ha="left",
            va="bottom")
    a2.set_yscale("log")
    a2.set_ylim(0.08, 60)
    a2.set_xlim(-0.5, 2.45)
    a2.spines["bottom"].set_bounds(-0.5, 1.4)
    a2.set_xticks([0, 1])
    a2.set_xticklabels([r"$i_q$", r"$i_d$"])
    a2.tick_params(axis="x", length=0)
    a2.set_ylabel("amplitud a $6f_e$ [mA]")
    a2.set_title("(b) Rizo de corriente a $6f_e$")
    guardar(fig, "c6_sesgo_rizo")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_bemf_forma, fig_escalon, fig_rms_modelos, fig_identificacion, fig_estabilidad, fig_kappa, fig_sesgo_rizo):
        f()
    json.dump(NUM, open(OUT / "c6_numeros.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
