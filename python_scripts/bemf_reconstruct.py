#!/usr/bin/env python3
"""Reconstrucción de la BEMF real del 2804 a partir de los logs de 07_m2pc
(PLANNING §11, Exp. 3a). Sustituye al análisis por bins de analyze_m2pc_log.py
cuando hay datos a velocidad variable y con inversión de giro.

Modelo. Para cada ventana de 1 ms del logger (medias de i_αβ y u_αβ, θ_e al
cierre, ω̂):

    e_αβ(n) = u(n) − R·i(n) − L·di/dt(n)                        [V]
    e_x(n)  = ω_e(n)·S_x(θ_n) + D_x(θ_n) + c_x + ruido,   x ∈ {α, β}

    S_x(θ) = Σ_k [a_xk cos kθ + b_xk sin kθ]   k = 1..H   (BEMF: escala con ω)
    D_x(θ) = Σ_k [p_xk cos kθ + q_xk sin kθ]   k = 1..H   (dead-time, etc.: no escala)

S es la forma de la BEMF (λ es la amplitud de su fundamental). D recoge todo lo
periódico en θ que NO escala con la velocidad; a velocidad constante S y D
son indistinguibles — por eso el dataset de la Exp. 2 (±600 rpm) es el que
permite separarlas. c absorbe el residuo de offset de corriente (R·Δi).

Tres correcciones al logger de 1 ms:
  - θ_n se toma al CENTRO de la ventana: θ_cierre − ω_e·0.5 ms (a 600 rpm el
    cierre va 12.5° por delante del centro).
  - El promedio de 1 ms atenúa el armónico k por sinc(k·f_e·1 ms): la base se
    multiplica por ese factor, muestra a muestra, para que el modelo prediga lo
    que el logger mide. (A 600 rpm, k = 7: 0.66.)
  - Se descartan las ventanas con |ω| < W_MIN (la normalización por ω no tiene
    sentido y el dead-time domina) y los 3 s iniciales.

Se resuelve por mínimos cuadrados (batch, exacto) y después se corre el LMS
secuencial (ADALINE, lms_update.m) sobre la forma normalizada s = e/(λ·ω) para
verificar que converge a lo mismo y elegir μ. Salida: figuras + coeficientes
en JSON para el firmware.

Uso:
    python3 python_scripts/bemf_reconstruct.py datos_banco/m2pc_s23_exp2_*.log \
        datos_banco/m2pc_s22_cordic_fix.log --harm 10 --png /tmp/opencode/bemf_rec
"""

import argparse
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_RAW_PER_A = 32.93
R_OHM       = 2.7
L_H         = 0.86e-3
POLE_PAIRS  = 7
LAMBDA_NOM  = 0.0035
T_WIN       = 1e-3

LINE = re.compile(r"^L (\d+) (\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)")


def load(path):
    """Solo lo que sigue al último [boot] (la cola del firmware anterior queda al principio)."""
    rows = []
    for ln in open(path, errors="ignore"):
        if ln.startswith("[boot]"):
            rows = []
        m = LINE.match(ln.strip())
        if m:
            rows.append([int(x) for x in m.groups()])
    a = np.array(rows, dtype=float)
    t    = a[:, 0] / 1000.0
    th   = a[:, 1] / 65536.0 * 2 * np.pi
    i_ab = a[:, 2:4] / 256.0 / K_RAW_PER_A
    u_ab = a[:, 4:6] / 1000.0
    w_e  = a[:, 6] / 10.0 * 2 * np.pi / 60.0 * POLE_PAIRS      # rad/s eléctricos, con signo
    return t, th, i_ab, u_ab, w_e


def reconstruct(t, th, i_ab, u_ab, w_e, t0, w_min):
    di = np.gradient(i_ab, t, axis=0)
    e  = u_ab - R_OHM * i_ab - L_H * di
    th_c = np.mod(th - w_e * T_WIN / 2, 2 * np.pi)
    ok = (t >= t0) & (np.abs(w_e) >= w_min) & np.isfinite(e).all(axis=1)
    # descartar discontinuidades de tiempo (líneas perdidas) en la derivada
    dt = np.gradient(t)
    ok &= np.abs(dt - T_WIN) < 0.3 * T_WIN
    return e[ok], th_c[ok], w_e[ok], i_ab[ok], di[ok]


I_SIGN_A = 1.0 / K_RAW_PER_A        # 1 LSB: escala del signo suavizado


def deadtime_regressor(i_ab):
    """Distorsión del inversor por dead-time: +V_dt·sgn(i_x) en cada fase (la
    tensión real baja cuando la corriente sale por el low-side durante el
    dead-time). Reconstruye i_abc de i_αβ (amplitud-invariante), suaviza el
    signo con tanh(i/1 LSB) y proyecta a αβ. Devuelve (N, 2) para un único
    coeficiente V_dt [V] por ajustar (mismo para los dos ejes)."""
    ia = i_ab[:, 0]
    ib = -0.5 * i_ab[:, 0] + 0.8660254 * i_ab[:, 1]
    ic = -0.5 * i_ab[:, 0] - 0.8660254 * i_ab[:, 1]
    sa, sb, sc = (np.tanh(x / I_SIGN_A) for x in (ia, ib, ic))
    g_a = (2 * sa - sb - sc) / 3.0
    g_b = (sb - sc) / 1.7320508
    return np.column_stack([g_a, g_b])


def basis(th, w_e, H, sinc_corr=True):
    """Columnas: [ω·sinc_k·cos kθ, ω·sinc_k·sin kθ]_k, [sinc_k·cos kθ, sinc_k·sin kθ]_k, 1."""
    cols = []
    f_e = np.abs(w_e) / (2 * np.pi)
    for k in range(1, H + 1):
        s = np.sinc(k * f_e * T_WIN) if sinc_corr else 1.0
        cols += [w_e * s * np.cos(k * th), w_e * s * np.sin(k * th)]
    for k in range(1, H + 1):
        s = np.sinc(k * f_e * T_WIN) if sinc_corr else 1.0
        cols += [s * np.cos(k * th), s * np.sin(k * th)]
    cols.append(np.ones_like(th))
    return np.column_stack(cols)


def fit_ls(e, th, w_e, H, i_ab, di, fixed=None, fit_rl=False):
    """Mínimos cuadrados apilando los dos ejes. Además de S, D y c se
    identifican tres escalares comunes a α y β: V_dt (dead-time), ΔR y ΔL
    (corrección a R_OHM y L_H: e se reconstruyó con los nominales, así que
    e_real = e − ΔR·i − ΔL·di/dt). Con fixed=(D, c, vdt, dR, dL) se ajusta solo
    S (logs a velocidad constante, donde S y D son colineales)."""
    X = basis(th, w_e, H)
    g = deadtime_regressor(i_ab)
    if fixed is not None:
        D, c, vdt, dR, dL = fixed
        target = e - eval_series(D, th) - c - vdt * g - dR * i_ab - dL * di
        Xs = X[:, :2 * H]
        S, *_ = np.linalg.lstsq(Xs, target, rcond=None)
        return S, D, c, vdt, dR, dL, target - Xs @ S
    n, m = X.shape
    extra = 3 if fit_rl else 1
    Z = np.zeros((2 * n, 2 * m + extra))
    Z[:n, :m] = X; Z[n:, m:2 * m] = X
    Z[:n, 2 * m] = g[:, 0]; Z[n:, 2 * m] = g[:, 1]
    if fit_rl:
        # ΔR y ΔL identificados: NO fiable con medias de 1 ms (el L·di/dt de
        # ventanas promediadas no es el real: sesión 23, L salía 6.7 mH).
        Z[:n, -2] = i_ab[:, 0]; Z[n:, -2] = i_ab[:, 1]
        Z[:n, -1] = di[:, 0];   Z[n:, -1] = di[:, 1]
    y = np.concatenate([e[:, 0], e[:, 1]])
    sol, *_ = np.linalg.lstsq(Z, y, rcond=None)
    ca, cb, vdt = sol[:m], sol[m:2 * m], sol[2 * m]
    dR, dL = (sol[-2], sol[-1]) if fit_rl else (0.0, 0.0)
    coef = np.column_stack([ca, cb])
    resid = e - X @ coef - vdt * g - dR * i_ab - dL * di
    return coef[:2 * H], coef[2 * H:4 * H], coef[4 * H], vdt, dR, dL, resid


def harmonics(S):
    """Amplitud y fase por armónico, por eje. S[(2k-2)] = a_k (cos), S[(2k-1)] = b_k (sin)."""
    H = S.shape[0] // 2
    amp = np.hypot(S[0::2], S[1::2])          # (H, 2)
    ph  = np.arctan2(S[1::2], S[0::2])        # ángulo de (a, b)
    return amp, ph


def eval_series(S, th):
    H = S.shape[0] // 2
    out = np.zeros((len(th), S.shape[1]))
    for k in range(1, H + 1):
        out += np.outer(np.cos(k * th), S[2 * k - 2]) + np.outer(np.sin(k * th), S[2 * k - 1])
    return out


def run_lms(e, th, w_e, H, lam, mu, D=None, c=None, n_epochs=1):
    """ADALINE secuencial sobre la forma normalizada s = (e − D − c)/(λ·ω)."""
    target = e.copy()
    if D is not None:
        target -= eval_series(D, th)
    if c is not None:
        target -= c
    target /= (lam * w_e)[:, None]
    W = np.zeros((2 * H, 2))
    hist = []
    for _ in range(n_epochs):
        for n in range(len(th)):
            x = np.empty(2 * H)
            x[0::2] = np.cos(np.arange(1, H + 1) * th[n])
            x[1::2] = np.sin(np.arange(1, H + 1) * th[n])
            eps = target[n] - W.T @ x
            W += mu * np.outer(x, eps)
            if n % 50 == 0:
                hist.append(W.copy())
    return W, np.array(hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--harm", type=int, default=10)
    ap.add_argument("--t0", type=float, default=3.0)
    ap.add_argument("--wmin-rpm", type=float, default=60.0)
    ap.add_argument("--mu", type=float, default=2e-3)
    ap.add_argument("--png", default="/tmp/opencode/bemf_rec")
    ap.add_argument("--json", default="python_scripts/bemf_2804_fourier.json")
    ap.add_argument("--max-win", type=int, default=150000, help="ventanas máximas por log")
    ap.add_argument("--fit-rl", action="store_true", help="identificar también ΔR, ΔL (no fiable con medias de 1 ms)")
    args = ap.parse_args()
    H = args.harm
    w_min = args.wmin_rpm * 2 * np.pi / 60 * POLE_PAIRS

    sets = []
    for p in args.logs:
        t, th, i_ab, u_ab, w_e = load(p)
        e, thc, we, iab, di = reconstruct(t, th, i_ab, u_ab, w_e, args.t0, w_min)
        e, thc, we, iab, di = (x[:args.max_win] for x in (e, thc, we, iab, di))
        rpm_lo, rpm_hi = we.min() / (2 * np.pi) * 60 / POLE_PAIRS, we.max() / (2 * np.pi) * 60 / POLE_PAIRS
        variable = (rpm_hi - rpm_lo) > 0.5 * max(abs(rpm_lo), abs(rpm_hi))
        sets.append((p.split("/")[-1], e, thc, we, iab, variable, di))
        print(f"{p}: {len(e)} ventanas útiles, ω ∈ [{rpm_lo:.0f}, {rpm_hi:.0f}] rpm "
              f"→ {'velocidad variable (ajusta S+D)' if variable else 'velocidad constante (solo S, D fija)'}")

    # ── Ajuste conjunto (solo logs de velocidad variable: S y D separables) ──
    var = [s for s in sets if s[5]]
    if not var:
        raise SystemExit("hace falta al menos un log a velocidad variable para separar S de D")
    e_all  = np.vstack([s[1] for s in var])
    th_all = np.concatenate([s[2] for s in var])
    w_all  = np.concatenate([s[3] for s in var])
    i_all  = np.vstack([s[4] for s in var])
    di_all = np.vstack([s[6] for s in var])
    S, D, c, vdt, dR, dL, resid = fit_ls(e_all, th_all, w_all, H, i_all, di_all, fit_rl=args.fit_rl)
    amp, ph = harmonics(S)
    lam_a, lam_b = amp[0]
    lam = 0.5 * (lam_a + lam_b)
    print(f"\n=== Ajuste conjunto ({len(e_all)} ventanas, H = {H}) ===")
    print(f"λ = {lam*1e3:.3f} mWb  (α {lam_a*1e3:.3f}, β {lam_b*1e3:.3f}; nominal {LAMBDA_NOM*1e3:.1f})")
    # fase del fundamental: ideal e_α = −λ sin θ → (a1, b1) = (0, −λ) → ángulo −90°; e_β = +λ cos θ → 0°
    print(f"fase del fundamental: α {np.degrees(ph[0,0])+90:+.2f}°  β {np.degrees(ph[0,1]):+.2f}°  "
          f"(desviación respecto a −sin / cos; = error de alineación θ_e)")
    print("armónicos de S [% del fundamental]  (α | β):")
    for k in range(2, H + 1):
        print(f"  h{k:<2d} {100*amp[k-1,0]/lam_a:5.2f} | {100*amp[k-1,1]/lam_b:5.2f}")
    thd = np.sqrt(np.sum(amp[1:] ** 2, axis=0)) / amp[0] * 100
    print(f"THD: α {thd[0]:.2f} %  β {thd[1]:.2f} %")
    ampD, phD = harmonics(D)
    print(f"\nD(θ) — término independiente de ω [mV]: fundamental α {1e3*ampD[0,0]:.1f} β {1e3*ampD[0,1]:.1f};"
          f" h3 {1e3*ampD[2,0]:.1f}/{1e3*ampD[2,1]:.1f}; h5 {1e3*ampD[4,0]:.1f}/{1e3*ampD[4,1]:.1f};"
          f" h7 {1e3*ampD[6,0]:.1f}/{1e3*ampD[6,1]:.1f}")
    print(f"V_dt (dead-time, por fase) = {1e3*vdt:+.1f} mV   c (DC) = {1e3*c[0]:+.1f} / {1e3*c[1]:+.1f} mV")
    print(f"R identificada = {R_OHM + dR:.3f} Ω (nominal {R_OHM})   L identificada = {1e3*(L_H + dL):.3f} mH (nominal {1e3*L_H})"
          f"   RMS residuo = {1e3*resid.std(axis=0)[0]:.1f} / {1e3*resid.std(axis=0)[1]:.1f} mV")

    per = []
    print("\nPor dataset (S con D, c, V_dt fijos del ajuste conjunto):")
    for name, e, thc, we, iab, variable, di in sets:
        Si, _, _, _, _, _, ri = fit_ls(e, thc, we, H, iab, di, fixed=(D, c, vdt, dR, dL))
        ai, pi_ = harmonics(Si)
        per.append((name, Si, ai, ri))
        print(f"  {name:38s} λ = {0.5e3*(ai[0,0]+ai[0,1]):.3f} mWb  fase {np.degrees(pi_[0,0])+90:+.2f}°"
              f"  h2 {100*ai[1,0]/ai[0,0]:.2f}  h3 {100*ai[2,0]/ai[0,0]:.2f}  h5 {100*ai[4,0]/ai[0,0]:.2f}"
              f"  h7 {100*ai[6,0]/ai[0,0]:.2f} %  resid {1e3*ri.std():.1f} mV")

    # ── LMS secuencial (ADALINE) sobre la forma normalizada ──────────────
    e_lms = e_all - vdt * deadtime_regressor(i_all) - dR * i_all - dL * di_all
    W, hist = run_lms(e_lms, th_all, w_all, H, lam, args.mu, D=D, c=c)
    S_norm = S / lam
    err_W = np.abs(W - S_norm).max()
    print(f"\nLMS (μ = {args.mu}, {len(e_all)} muestras, 1 época): máx |W − S/λ| = {err_W:.4f}"
          f"  (fundamental ≈ 1; W_b1α = {W[1,0]:+.4f}, W_a1β = {W[0,1]:+.4f})")
    # tiempo de convergencia: primera vez que |W_b1α − S| < 2 %
    conv = np.argmax(np.abs(hist[:, 1, 0] - S_norm[1, 0]) < 0.02) * 50
    print(f"convergencia del fundamental al 2 %: ~{conv} muestras ({conv/1000:.1f} s de log)")

    # ── JSON para el firmware ────────────────────────────────────────────
    out = {
        "H": H, "lambda_Wb": lam, "R_ohm": R_OHM, "L_H": L_H, "pole_pairs": POLE_PAIRS,
        "note": "e_x(θ,ω) = ω_e·S_x(θ) + D_x(θ) + c_x; S/λ es la forma normalizada (ADALINE W).",
        "S_alpha_cos": S[0::2, 0].tolist(), "S_alpha_sin": S[1::2, 0].tolist(),
        "S_beta_cos":  S[0::2, 1].tolist(), "S_beta_sin":  S[1::2, 1].tolist(),
        "D_alpha_cos_V": D[0::2, 0].tolist(), "D_alpha_sin_V": D[1::2, 0].tolist(),
        "D_beta_cos_V":  D[0::2, 1].tolist(), "D_beta_sin_V":  D[1::2, 1].tolist(),
        "c_V": c.tolist(), "V_deadtime_V": float(vdt), "R_ident_ohm": R_OHM + dR, "L_ident_H": L_H + dL,
        "harm_pct_alpha": (100 * amp[:, 0] / lam_a).tolist(),
        "harm_pct_beta":  (100 * amp[:, 1] / lam_b).tolist(),
        "logs": [s[0] for s in sets],
    }
    json.dump(out, open(args.json, "w"), indent=1)
    print(f"coeficientes → {args.json}")

    # ── Figuras ──────────────────────────────────────────────────────────
    thg = np.linspace(0, 2 * np.pi, 361)
    Sg = eval_series(S_norm, thg)
    Dg = eval_series(D, thg)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    a = ax[0, 0]
    for name, e, thc, we, iab, variable, di in sets:
        s_meas = (e - eval_series(D, thc) - c - vdt * deadtime_regressor(iab) - dR * iab - dL * di) / (lam * we)[:, None]
        bins = np.linspace(0, 2 * np.pi, 73); idx = np.digitize(thc, bins) - 1
        mb = np.array([s_meas[idx == b].mean(axis=0) if np.any(idx == b) else [np.nan, np.nan] for b in range(72)])
        a.plot(np.degrees(bins[:-1] + np.pi / 72), mb[:, 0], ".", ms=3, alpha=.6, label=f"α {name[5:25]}")
    a.plot(np.degrees(thg), Sg[:, 0], "k", lw=1.5, label="S_α/λ ajustada")
    a.plot(np.degrees(thg), -np.sin(thg), "k--", lw=.8, label="−sin θ")
    a.set_title("Forma normalizada (e − D − c)/(λω), eje α"); a.set_xlabel("θ_e [°]"); a.grid(alpha=.3); a.legend(fontsize=7)

    a = ax[0, 1]
    a.plot(np.degrees(thg), Sg[:, 0] + np.sin(thg), label="S_α/λ − (−sin θ)")
    a.plot(np.degrees(thg), Sg[:, 1] - np.cos(thg), label="S_β/λ − cos θ")
    a.set_title("Desviación de la forma respecto a la sinusoide ideal"); a.set_xlabel("θ_e [°]"); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[1, 0]
    ks = np.arange(2, H + 1)
    a.bar(ks - 0.2, 100 * amp[1:, 0] / lam_a, 0.4, label="α (conjunto)")
    a.bar(ks + 0.2, 100 * amp[1:, 1] / lam_b, 0.4, label="β (conjunto)")
    for name, Si, ai, _ in per:
        a.plot(ks, 100 * ai[1:, 0] / ai[0, 0], "x", ms=5, label=f"α {name[5:25]}")
    a.set_title("Armónicos de S [% del fundamental]"); a.set_xlabel("k"); a.grid(alpha=.3); a.legend(fontsize=7)

    a = ax[1, 1]
    a.plot(np.degrees(thg), 1e3 * Dg[:, 0], label="D_α")
    a.plot(np.degrees(thg), 1e3 * Dg[:, 1], label="D_β")
    a.set_title("D(θ): término que no escala con ω (dead-time, etc.) [mV]"); a.set_xlabel("θ_e [°]"); a.grid(alpha=.3); a.legend(fontsize=8)
    plt.suptitle(f"BEMF del 2804 reconstruida — λ = {lam*1e3:.3f} mWb, THD α {thd[0]:.2f} % β {thd[1]:.2f} %, H = {H}")
    plt.tight_layout(); plt.savefig(args.png + "_forma.png", dpi=110)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    n = np.arange(len(hist)) * 50 / 1000
    ax[0].plot(n, hist[:, 1, 0], label="W: b1 α (→ −1)"); ax[0].plot(n, hist[:, 0, 1], label="W: a1 β (→ +1)")
    ax[0].plot(n, hist[:, 5, 0], label="W: b3 α"); ax[0].plot(n, hist[:, 4, 0], label="W: a3 α")
    ax[0].plot(n, hist[:, 9, 0], label="W: b5 α"); ax[0].plot(n, hist[:, 13, 0], label="W: b7 α")
    for v, lab in [(S_norm[1, 0], ""), (S_norm[5, 0], ""), (S_norm[4, 0], ""), (S_norm[9, 0], ""), (S_norm[13, 0], "")]:
        ax[0].axhline(v, color="k", lw=.5, ls=":")
    ax[0].set_title(f"ADALINE/LMS, μ = {args.mu}: pesos vs muestras (línea punteada = mínimos cuadrados)")
    ax[0].set_xlabel("s de log (1 kHz)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
    ax[1].plot(np.degrees(thg), eval_series(W, thg)[:, 0], label="W_α final")
    ax[1].plot(np.degrees(thg), Sg[:, 0], "k--", label="S_α/λ (LS)")
    ax[1].set_title("Forma aprendida por LMS vs mínimos cuadrados"); ax[1].set_xlabel("θ_e [°]"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(args.png + "_lms.png", dpi=110)
    print(f"figuras → {args.png}_forma.png, {args.png}_lms.png")


if __name__ == "__main__":
    main()
