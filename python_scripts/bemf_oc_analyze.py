#!/usr/bin/env python3
"""BEMF del 2804 en circuito abierto (app 08_bemf_oc, PLANNING §11 Exp. 2b).

Lee `B seq th w va vb vc` (2.5 kHz): θ_e (1/65536 de vuelta eléctrica), ω̂
(rpm×10) y las tres lecturas del ADC (12 bits) de los nodos de BEMF de la
placa (fase ─10 kΩ─┬─2.2 kΩ─GND, ratio nominal 0.18, GPIO_BEMF en BAJO; modo
común fijado con una resistencia externa).

Con las tres fases en abierto los tres divisores forman un neutro virtual:
e_x − ē = (v_x − mean(v))/ratio. Se trabaja con la componente diferencial,
que elimina el modo común (la polarización externa y cualquier deriva).

Modelo por mínimos cuadrados, por fase y sin ningún parámetro del motor:

    e_x(n) = ω_e(n) · Σ_k [a_xk cos kθ_n + b_xk sin kθ_n] + c_x,   k = 1..H

(la BEMF es ω·S(θ) exactamente; no hay R, L, dead-time ni tensión comandada).
Luego se proyecta a αβ (Clarke amplitud-invariante) para comparar con la
reconstrucción de la Exp. 3a (bemf_2804_fourier.json) y con la hipótesis
sinusoidal del controlador. Intervalos de confianza por bootstrap sobre
bloques de 0.5 s. λ sale del fundamental (en V·s/rad eléctrico) y del
ratio del divisor (RATIO; calibrar con DMM si se quiere λ absoluto).

Uso:
    python3 python_scripts/bemf_oc_analyze.py datos_banco/bemf_oc_s24.log \
        [--harm 12 --wmin-rpm 80 --ratio 0.1803 --png firmware/apps/08_bemf_oc/doc/bemf_oc]
"""

import argparse
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLE_PAIRS = 7
FS         = 2500.0
VREF       = 3.3
ADC_FS     = 4096.0
LINE = re.compile(r"^B (\d+) (\d+) (-?\d+) (\d+) (\d+) (\d+)")


def load(path):
    rows = []
    for ln in open(path, errors="ignore"):
        if ln.startswith("[boot]"):
            rows = []
        m = LINE.match(ln.strip())
        if m:
            rows.append([int(x) for x in m.groups()])
    a = np.array(rows, dtype=float)
    seq = a[:, 0]
    th  = a[:, 1] / 65536.0 * 2 * np.pi
    w_e = a[:, 2] / 10.0 * 2 * np.pi / 60.0 * POLE_PAIRS
    v   = a[:, 3:6] * VREF / ADC_FS
    return seq, th, w_e, v


def basis(th, w_e, H):
    cols = []
    for k in range(1, H + 1):
        cols += [w_e * np.cos(k * th), w_e * np.sin(k * th)]
    cols.append(np.ones_like(th))
    return np.column_stack(cols)


def fit(th, w_e, e, H):
    X = basis(th, w_e, H)
    coef, *_ = np.linalg.lstsq(X, e, rcond=None)
    return coef[:-1], coef[-1], e - X @ coef


def harm(S):
    return np.hypot(S[0::2], S[1::2]), np.arctan2(S[1::2], S[0::2])


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def eval_series(S, th):
    H = S.shape[0] // 2
    out = np.zeros((len(th), S.shape[1]))
    for k in range(1, H + 1):
        out += np.outer(np.cos(k * th), S[2 * k - 2]) + np.outer(np.sin(k * th), S[2 * k - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--harm", type=int, default=12)
    ap.add_argument("--wmin-rpm", type=float, default=80.0)
    ap.add_argument("--ratio", type=float, default=2.2 / 12.2, help="divisor nodo/fase")
    ap.add_argument("--png", default="/tmp/opencode/bemf_oc")
    ap.add_argument("--json", default="python_scripts/bemf_2804_oc.json")
    ap.add_argument("--ref", default="python_scripts/bemf_2804_fourier.json", help="Exp. 3a para comparar")
    ap.add_argument("--boot", type=int, default=200, help="réplicas bootstrap")
    args = ap.parse_args()
    H = args.harm

    seq, th, w_e, v = load(args.log)
    print(f"{len(seq)} muestras; saltos de secuencia: {int(np.sum(np.diff(seq) != 4))}")
    rpm = w_e / POLE_PAIRS / (2 * np.pi) * 60
    print(f"ω: {rpm.min():.0f} … {rpm.max():.0f} rpm;  nodo: {v.min():.2f} … {v.max():.2f} V "
          f"(saturación si toca 0 o 3.3)")
    clip = np.any((v < 0.01) | (v > VREF - 0.01), axis=1)
    e_abc = (v - v.mean(axis=1, keepdims=True)) / args.ratio          # fase − neutro virtual [V]
    ok = (np.abs(rpm) >= args.wmin_rpm) & ~clip
    print(f"útiles: {ok.sum()} ({100*ok.mean():.0f} %), recortadas: {clip.sum()}")
    th, w_e, e_abc, rpm = th[ok], w_e[ok], e_abc[ok], rpm[ok]

    # ── Ajuste por fase ────────────────────────────────────────────────────
    S_abc, c_abc, r_abc = fit(th, w_e, e_abc, H)
    amp, ph = harm(S_abc)
    lam_abc = amp[0]
    print(f"\n=== Circuito abierto, {len(th)} muestras, H = {H}, ratio {args.ratio:.4f} ===")
    print(f"λ por fase (fundamental): a {1e3*lam_abc[0]:.3f}  b {1e3*lam_abc[1]:.3f}  c {1e3*lam_abc[2]:.3f} mWb"
          f"   (Exp. 3a: 3.457; nominal 3.5)")
    print(f"fase del fundamental respecto a −sin(θ_e − k·120°): a {wrap_deg(np.degrees(ph[0,0])+90):+.2f}°  "
          f"b {wrap_deg(np.degrees(ph[0,1])+90-120):+.2f}°  c {wrap_deg(np.degrees(ph[0,2])+90+120):+.2f}°")
    print(f"DC c_x: {1e3*c_abc[0]:+.1f} / {1e3*c_abc[1]:+.1f} / {1e3*c_abc[2]:+.1f} mV   "
          f"residuo RMS: {1e3*r_abc.std(axis=0)[0]:.1f} / {1e3*r_abc.std(axis=0)[1]:.1f} / {1e3*r_abc.std(axis=0)[2]:.1f} mV")

    # ── Bootstrap por bloques (0.5 s) para intervalos de confianza ────────
    nb = int(0.5 * FS)
    blocks = np.arange(0, len(th) - nb, nb)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(args.boot):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([np.arange(b, b + nb) for b in pick])
        Sb, _, _ = fit(th[idx], w_e[idx], e_abc[idx], H)
        ab, _ = harm(Sb)
        boots.append(100 * ab / ab[0])
    boots = np.array(boots)                                  # (B, H, 3)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print("\narmónicos [% del fundamental], fase a | b | c  (IC 95 % bootstrap):")
    for k in range(2, H + 1):
        print(f"  h{k:<2d} " + " | ".join(f"{100*amp[k-1,x]/lam_abc[x]:5.2f} [{lo[k-1,x]:4.2f}–{hi[k-1,x]:4.2f}]" for x in range(3)))
    thd = np.sqrt(np.sum(amp[1:] ** 2, axis=0)) / amp[0] * 100
    print(f"THD: a {thd[0]:.2f} %  b {thd[1]:.2f} %  c {thd[2]:.2f} %")

    # ── αβ para comparar con la Exp. 3a ───────────────────────────────────
    e_ab = np.column_stack([(2 * e_abc[:, 0] - e_abc[:, 1] - e_abc[:, 2]) / 3,
                            (e_abc[:, 1] - e_abc[:, 2]) / np.sqrt(3)])
    S_ab, c_ab, r_ab = fit(th, w_e, e_ab, H)
    amp_ab, ph_ab = harm(S_ab)
    lam = 0.5 * (amp_ab[0, 0] + amp_ab[0, 1])
    print(f"\nαβ: λ = {1e3*lam:.3f} mWb; fase del fundamental α {wrap_deg(np.degrees(ph_ab[0,0])+90):+.2f}°  β {wrap_deg(np.degrees(ph_ab[0,1])):+.2f}°")
    print("    armónicos α | β [%]: " + "  ".join(f"h{k} {100*amp_ab[k-1,0]/amp_ab[0,0]:.2f}|{100*amp_ab[k-1,1]/amp_ab[0,1]:.2f}" for k in range(2, min(H, 8) + 1)))
    ref = None
    try:
        ref = json.load(open(args.ref))
        print(f"    Exp. 3a (tensión comandada): " + "  ".join(f"h{k} {ref['harm_pct_alpha'][k-1]:.2f}|{ref['harm_pct_beta'][k-1]:.2f}" for k in range(2, min(ref["H"], 8) + 1)))
    except (OSError, KeyError):
        pass

    out = {"H": H, "ratio": args.ratio, "lambda_Wb": lam, "lambda_abc_Wb": lam_abc.tolist(),
           "S_abc_cos": S_abc[0::2].T.tolist(), "S_abc_sin": S_abc[1::2].T.tolist(),
           "S_alpha_cos": S_ab[0::2, 0].tolist(), "S_alpha_sin": S_ab[1::2, 0].tolist(),
           "S_beta_cos": S_ab[0::2, 1].tolist(), "S_beta_sin": S_ab[1::2, 1].tolist(),
           "harm_pct_abc": (100 * amp / lam_abc).T.tolist(),
           "harm_ci95_lo": lo.T.tolist(), "harm_ci95_hi": hi.T.tolist(),
           "harm_pct_alpha": (100 * amp_ab[:, 0] / amp_ab[0, 0]).tolist(),
           "harm_pct_beta": (100 * amp_ab[:, 1] / amp_ab[0, 1]).tolist(),
           "n": int(len(th)), "rpm_range": [float(rpm.min()), float(rpm.max())], "log": args.log}
    json.dump(out, open(args.json, "w"), indent=1)
    print(f"→ {args.json}")

    # ── Figuras ───────────────────────────────────────────────────────────
    thg = np.linspace(0, 2 * np.pi, 721)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]
    s_meas = e_abc / (lam_abc * w_e[:, None])
    bins = np.linspace(0, 2 * np.pi, 121); idx = np.digitize(th, bins) - 1
    for x, nm in enumerate("abc"):
        mb = np.array([s_meas[idx == b, x].mean() if np.any(idx == b) else np.nan for b in range(120)])
        a.plot(np.degrees(bins[:-1] + np.pi / 120), mb, ".", ms=3, alpha=.7, label=f"e_{nm}/(λω) bins")
    Sg = eval_series(S_abc / lam_abc, thg)
    for x, nm in enumerate("abc"):
        a.plot(np.degrees(thg), Sg[:, x], "k", lw=1)
    a.plot(np.degrees(thg), -np.sin(thg), "k--", lw=.7, label="−sin θ")
    a.set_title("Forma normalizada por fase (circuito abierto)"); a.set_xlabel("θ_e [°]"); a.grid(alpha=.3); a.legend(fontsize=7)

    a = ax[0, 1]
    for x, nm in enumerate("abc"):
        a.plot(np.degrees(thg), Sg[:, x] + np.sin(thg - x * 2 * np.pi / 3), label=f"e_{nm}: desviación de la sinusoide")
    a.set_title("Desviación respecto a −sin(θ − k·120°)"); a.set_xlabel("θ_e [°]"); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[1, 0]
    ks = np.arange(2, H + 1)
    for x, nm in enumerate("abc"):
        est = 100 * amp[1:, x] / lam_abc[x]
        a.errorbar(ks + (x - 1) * 0.25, est,
                   yerr=[np.clip(est - lo[1:, x], 0, None), np.clip(hi[1:, x] - est, 0, None)],
                   fmt="o", ms=4, capsize=2, label=f"fase {nm} (IC 95 %)")
    if ref is not None:
        kr = min(H, ref["H"])
        a.plot(np.arange(2, kr + 1), ref["harm_pct_alpha"][1:kr], "kx", ms=6, label="Exp. 3a α (tensión comandada)")
    a.set_title("Armónicos [% del fundamental]"); a.set_xlabel("k"); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[1, 1]
    t = np.arange(len(rpm)) / FS
    a.plot(t, rpm, lw=.6); a.set_title("ω̂ durante la captura (muestras útiles)"); a.set_xlabel("s"); a.set_ylabel("rpm"); a.grid(alpha=.3)
    plt.suptitle(f"BEMF del 2804 en circuito abierto — λ = {1e3*lam:.3f} mWb (ratio {args.ratio:.3f}), THD a/b/c = {thd[0]:.2f}/{thd[1]:.2f}/{thd[2]:.2f} %")
    plt.tight_layout(); plt.savefig(args.png + "_forma.png", dpi=110)
    print(f"figura → {args.png}_forma.png")


if __name__ == "__main__":
    main()
