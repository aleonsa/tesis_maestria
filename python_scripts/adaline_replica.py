#!/usr/bin/env python3
"""Réplica bit-exacta del ADALINE online (PLANNING §11, Exp. 3b paso 1).

Propósito: congelar TODOS los formatos de punto fijo y el orden de las
operaciones del ADALINE en línea ANTES de portarlo a C, y verificar con los
logs existentes (Exp. 2 y sesión 22) que la cadena entera converge a la misma
forma de BEMF que el LMS float de la Exp. 3a (`bemf_reconstruct.py`).

La cadena —cada línea es literalmente port-able a C (lazo lento de 1 kHz,
fuera de la ISR)—:

  por ventana de 1 ms (las mismas medias que acumula el logger de 07_m2pc):
    θ_c   = θ_cierre − 7·ω̂·25 ticks          (ω̂ mecánica, θ eléctrica)
    e_obs = ū − R·ī − L·Δī/Δt                (observador algebraico)
    e_pk  = (ω̂·KE_MV_Q18) >> 18              [mV] (modelo nominal λ=3.5 mWb)
    x_k(θ_c): sin/cos por "CORDIC" q1.15 + recurrencia compleja Q31 (>>31)
    ĝ     = (Σ_k W_k·x_k) >> 31              (forma normalizada, Q27)
    ê     = (e_pk·ĝ) >> 27                   [mV]
    ε_q15 = ((e_obs − ê)<<15) / e_pk         (división entera, truncada)
    W_k  += (MU_Q16·x_k·ε_q15) >> 35         (LMS, μ = 2e-3, 1 kHz)

Formatos (constantes al final del fichero, con su derivación a la vista):
  corriente Q8 LSB; tensión mV; θ Q32; W Q27; ε Q15; e_pk mV.
  ε limitado a ±4, pesos a ±2, norma L1 por eje ≤4 (protección numérica).
  Sin corrección sinc: se aprende la forma efectiva promediada a 1 ms.
  Bit-exacto se refiere a la aritmética C con la misma base cuantizada;
  NO a una emulación bit-exacta del periférico CORDIC.
  En la ISR (50 kHz) solo se evalúa ĝ·e_pk: 1 CORDIC (ya existe para Park),
  H−1 multiplicaciones complejas Q31 y 2H productos acumulados — el update
  de W queda en este lazo lento.

Decisiones ya tomadas (PROGRESO sesión 24): H = 7, μ = 2e-3 a 1 kHz,
referencia = `bemf_2804_oc.json` (BEMF en circuito abierto, Exp. 2b).

Salida: reporte en consola (convergencia, |W − W_float|, armónicos aprendidos
vs referencia OC, error de la recurrencia, preview de mejora del residuo),
figuras W(t) y forma aprendida, y JSON con W_q27 para el port a C.

Uso:
    python3 python_scripts/adaline_replica.py \
        datos_banco/m2pc_s23_exp2_100rpm_50mHz.log \
        datos_banco/m2pc_s23_exp2_300rpm_100mHz.log \
        datos_banco/m2pc_s23_exp2_600rpm_200mHz.log \
        datos_banco/m2pc_s22_cordic_fix.log
"""

import argparse
import json
import math
import re

import numpy as np

# ── Constantes (idénticas a motor_params.h / derivadas del lazo lento) ─────
K_RAW_PER_A = 32.93          # sesión 19, DMM
RK_Q8       = 20990          # M2PC_RK_Q8: mV = (i_q8>>4)·RK_Q8 >> 12
KE_MV_Q18   = 470            # KE_MV_Q18 (λ_nom = 3.5 mWb, 7 pp, 50 kHz)
# L·di/dt: mV = Δi_q8 · (L/T_win)·1000/(256·K) = Δi_q8·0.102052 → Q15
LK_Q15      = round(0.86e-3 / 1e-3 * 1000.0 / (256.0 * K_RAW_PER_A) * 32768.0)
# MU_Q16 = 7 (≈1.1e-4). En banco (s26) μ=2e-3 (131) fue INESTABLE en lazo
# cerrado (la forma aprendida alimenta el sector del predictor: ε_q 73→138 mA
# en 25 s hasta TRIP). 7 es estable aunque el aprendizaje empeora ε_q vs el
# modelo sinusoidal — ver PROGRESO sesión 26. Debe coincidir con adaline.h.
MU_Q16      = 7
H_DEFAULT   = 7
LOG_WINDOW  = 50             # ventana del logger: 50 ticks de 20 µs = 1 ms
LOG_DT_MS   = 1              # paso entre líneas L en el log [ms]
EPK_MIN_MV  = 150            # |e_pk| mínimo para actualizar W (≈ 60 rpm mec)
LAMBDA_NOM  = 3.5e-3         # usado por KE_MV_Q18 (e_pk del modelo)
FS_ISR      = 50000.0
POLE_PAIRS  = 7
W_FRAC      = 27
W_ONE       = 1 << W_FRAC

LINE = re.compile(r"^L (\d+) (\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)")

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1


def i32(v):
    """Clamp a int32 (donde el C guarda en int32)."""
    return max(INT32_MIN, min(INT32_MAX, int(v)))


def tdiv(a, b):
    """División entera con truncamiento (semántica de C), b != 0."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def q15(v):
    """Cuantización q1.15 del "CORDIC": modela la salida real del periférico
    (error residual < 2^-18, RM0440 Tabla 103, despreciable frente al LSB q15).
    El port a C usa cordic_sincos_q31() directamente."""
    scaled = v * 32768.0
    rounded = math.floor(abs(scaled) + 0.5) * (1 if scaled >= 0 else -1)
    return max(-32768, min(32767, rounded))


def sincos_modelo(theta_q32):
    """Modelo de cordic_sincos_q31: (sin, cos) en Q31."""
    th = (theta_q32 & 0xFFFFFFFF) / 2 ** 32 * 2 * math.pi
    return q15(math.sin(th)) << 16, q15(math.cos(th)) << 16


def basis_recurrencia(theta_q32, H):
    """sin/cos kθ, k = 1..H, en Q31, con la recurrencia compleja truncante
    que correrá en la ISR:
        z_1 = (c1, s1) del CORDIC;  z_k = z_{k-1} ⊗ z_1 con >>31 por paso.
    Devuelve lista [(c_1, s_1), ..., (c_H, s_H)]."""
    s1, c1 = sincos_modelo(theta_q32)
    out = [(c1, s1)]
    cp, sp = c1, s1
    for _ in range(2, H + 1):
        cn = i32((cp * c1 - sp * s1) >> 31)
        sn = i32((sp * c1 + cp * s1) >> 31)
        out.append((cn, sn))
        cp, sp = cn, sn
    return out


def eval_isr(W, s1_q31, c1_q31, e_pk, H):
    """Espejo de adaline_eval() del firmware: base en Q30 (un bit de margen,
    sin saturación), acumulación Q57 → ĝ Q27 → ê mV. Devuelve (e_α, e_β)."""
    c1, s1 = c1_q31 >> 1, s1_q31 >> 1
    ca, sa = c1, s1
    aa = int(W[0, 0]) * ca + int(W[1, 0]) * sa
    ab = int(W[0, 1]) * ca + int(W[1, 1]) * sa
    for k in range(1, H):
        ca, sa = i32((ca * c1 - sa * s1) >> 30), i32((sa * c1 + ca * s1) >> 30)
        aa += int(W[2 * k, 0]) * ca + int(W[2 * k + 1, 0]) * sa
        ab += int(W[2 * k, 1]) * ca + int(W[2 * k + 1, 1]) * sa
    return i32((e_pk * i32(aa >> 30)) >> W_FRAC), i32((e_pk * i32(ab >> 30)) >> W_FRAC)


def basis_exacta(theta_q32, H):
    """Referencia float64 de la base (para medir el error de la recurrencia)."""
    th = (theta_q32 & 0xFFFFFFFF) / 2 ** 32 * 2 * math.pi
    return [(math.cos(k * th), math.sin(k * th)) for k in range(1, H + 1)]


class AdalineQ:
    """ADALINE en punto fijo, bit-exacto respecto al futuro C."""

    def __init__(self, H, init):
        self.H = H
        # W[(2k-2)+ej*0] layout por eje: [a1,b1,...,aH,bH] con ejes α,β
        self.W = np.zeros((2 * H, 2), dtype=np.int64)
        if init == "modelo":
            # sinusoidal nominal: e_α = −λ sin θ, e_β = +λ cos θ
            self.W[1, 0] = -W_ONE             # b1 α = −1
            self.W[0, 1] = W_ONE              # a1 β = +1
        self.n_upd = 0

    def eval_g(self, x, eje):
        """ĝ = (Σ W_k·x_k) >> 31 en Q27 (productos int64)."""
        acc = 0
        W = self.W[:, eje]
        for k in range(self.H):
            c, s = x[k]
            acc += int(W[2 * k]) * c + int(W[2 * k + 1]) * s
        return i32(acc >> 31)

    def paso(self, x, e_pk, e_obs_a, e_obs_b):
        """Una ventana: evalúa, normaliza el error y actualiza W (los dos ejes)."""
        g_a = self.eval_g(x, 0)
        g_b = self.eval_g(x, 1)
        e_hat_a = i32((e_pk * g_a) >> W_FRAC)
        e_hat_b = i32((e_pk * g_b) >> W_FRAC)
        eps_a = max(-4*32768, min(4*32768, tdiv((e_obs_a - e_hat_a)*32768, e_pk)))
        eps_b = max(-4*32768, min(4*32768, tdiv((e_obs_b - e_hat_b)*32768, e_pk)))
        next_w = self.W.copy()
        for k in range(self.H):
            c, s = x[k]
            next_w[2*k, 0] += (MU_Q16*c*eps_a) >> 35
            next_w[2*k+1, 0] += (MU_Q16*s*eps_a) >> 35
            next_w[2*k, 1] += (MU_Q16*c*eps_b) >> 35
            next_w[2*k+1, 1] += (MU_Q16*s*eps_b) >> 35
        next_w = np.clip(next_w, -2*W_ONE, 2*W_ONE)
        self.accepted = bool(np.all(np.abs(next_w).sum(axis=0) <= 4*W_ONE))
        if self.accepted:
            self.W = next_w
        self.n_upd += 1
        return e_hat_a, e_hat_b


class AdalineF:
    """Mismo algoritmo en float64 (referencia; es el LMS de la Exp. 3a con
    H = 7, sin corrección D/c: eso es lo que el online realmente ve)."""

    def __init__(self, H, init):
        self.H = H
        self.W = np.zeros((2 * H, 2))
        if init == "modelo":
            self.W[1, 0] = -1.0
            self.W[0, 1] = +1.0

    def paso(self, th_c, lam_w, e_obs_a, e_obs_b):
        x = np.empty(2 * self.H)
        x[0::2] = np.cos(np.arange(1, self.H + 1) * th_c)
        x[1::2] = np.sin(np.arange(1, self.H + 1) * th_c)
        eps = np.clip(np.array([e_obs_a, e_obs_b]) / lam_w - self.W.T @ x, -4, 4)
        next_w = np.clip(self.W + MU_Q16 / 65536.0 * np.outer(x, eps), -2, 2)
        if np.all(np.abs(next_w).sum(axis=0) <= 4):
            self.W = next_w
        return None


def ventanas(path, max_rows):
    """Generador de ventanas de 1 ms desde un log 07_m2pc (línea L).
    (t_ms, theta_close_q32, w_q32, ia_q8, ib_q8, ua_mv, ub_mv)."""
    rows = []
    with open(path, errors="ignore") as stream:
        for ln in stream:
            if ln.startswith("[boot]"):
                rows = []
            m = LINE.match(ln.strip())
            if m and len(rows) < max_rows:
                rows.append([int(v) for v in m.groups()])
    out = []
    for t, th16, ia, ib, ua, ub, rpm10, _ref in rows:
        w_q32 = int(round(rpm10 * (2 ** 32) / 30_000_000.0))
        out.append((t, th16 << 16, w_q32, ia, ib, ua, ub))
    return out


def corrida(logs, H, init, t0_s, max_win, hist_dec=25):
    """Corre la cadena bit-exacta y la float sobre los mismos datos.
    Devuelve (hist_t, hist_Wq, hist_Wf, métricas del residuo)."""
    q, f = AdalineQ(H, init), AdalineF(H, init)
    hist_t, hist_q, hist_f = [], [], []
    t_run = 0                     # tiempo continuo de log procesado [ms]
    res_sin, res_ad = [], []

    for path in logs:
        prev_i = None
        prev_t = None
        n = 0
        for (t, th_c, w_q32, ia, ib, ua, ub) in ventanas(path, max_win + 10000):
            t_run += LOG_DT_MS
            dt_ok = (prev_t is not None) and (t - prev_t == LOG_DT_MS)
            di_a = ia - prev_i[0] if dt_ok else 0
            di_b = ib - prev_i[1] if dt_ok else 0
            prev_i = (ia, ib)
            prev_t = t

            # ── observador algebraico (punto fijo, truncante como el C) ──
            ri_a = ((ia >> 4) * RK_Q8) >> 12
            ri_b = ((ib >> 4) * RK_Q8) >> 12
            ldi_a = (di_a * LK_Q15) >> 15
            ldi_b = (di_b * LK_Q15) >> 15
            e_a = i32(ua - ri_a - ldi_a)
            e_b = i32(ub - ri_b - ldi_b)

            # ── e_pk del modelo y puertas ──
            e_pk = i32((w_q32 * KE_MV_Q18) >> 18)
            theta_c = theta_centro(th_c, w_q32)
            ok = (dt_ok and abs(e_pk) >= EPK_MIN_MV and t_run >= t0_s * 1000)

            if ok:
                x = basis_recurrencia(theta_c, H)
                eha, ehb = q.paso(x, e_pk, e_a, e_b)
                # referencia float: MISMO e_obs en V, normalizado por el MISMO
                # e_pk (mV→V): es el LMS de la Exp. 3a con H = 7 y sin D/c.
                ka = 1.0 / (256.0 * K_RAW_PER_A)
                # L·di/dt = (L/T_win)·Δi = 0.86 Ω · Δi_A   (T_win = 1 ms)
                f.paso(theta_c / 2 ** 32 * 2 * math.pi, e_pk / 1000.0,
                       ua / 1000.0 - 2.7 * ia * ka - 0.86 * di_a * ka,
                       ub / 1000.0 - 2.7 * ib * ka - 0.86 * di_b * ka)
                # residuo: modelo sinusoidal vs ADALINE (preview Exp. 3c)
                s1, c1 = sincos_modelo(theta_c)
                res_sin.append((e_a - i32((e_pk * (-s1)) >> 31)) ** 2
                               + (e_b - i32((e_pk * c1) >> 31)) ** 2)
                res_ad.append((e_a - eha) ** 2 + (e_b - ehb) ** 2)
                if q.n_upd % hist_dec == 0:
                    hist_t.append(t_run / 1000.0)
                    hist_q.append(q.W.copy())
                    hist_f.append(f.W.copy())
            n += 1
            if n >= max_win:
                break

    res = {
        "n_upd": q.n_upd,
        "rms_res_sin_mv": math.sqrt(sum(res_sin) / max(1, len(res_sin))),
        "rms_res_ad_mv": math.sqrt(sum(res_ad) / max(1, len(res_ad))),
    }
    return (np.array(hist_t), np.array(hist_q), np.array(hist_f), q, f, res)


def arm(W_col):
    """Amplitud relativa de los armónicos de una columna de W (float)."""
    amp = np.hypot(W_col[0::2], W_col[1::2])
    return amp / amp[0]


def theta_centro(theta_close, omega):
    """θ eléctrica al centro; omega MECÁNICA (Q32 vueltas/tick)."""
    return (theta_close - omega * POLE_PAIRS * (LOG_WINDOW // 2)) & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--H", type=int, default=H_DEFAULT)
    ap.add_argument("--t0", type=float, default=3.0, help="descarte inicial [s]")
    ap.add_argument("--max-win", type=int, default=300000, help="ventanas máx. por log")
    ap.add_argument("--json", default="python_scripts/adaline_replica_w.json")
    ap.add_argument("--png", default="/tmp/opencode/adaline_replica")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    H = args.H
    if not 1 <= H <= 7:
        ap.error("H debe estar entre 1 y 7")

    print(f"constantes: RK_Q8={RK_Q8}  KE_Q18={KE_MV_Q18}  LK_Q15={LK_Q15}  "
          f"MU_Q16={MU_Q16}  H={H}  EPK_MIN={EPK_MIN_MV} mV")
    print(f"logs: {[p.split('/')[-1] for p in args.logs]}")

    resultados = {}
    for init in ("modelo", "cero"):
        ht, hq, hf, q, f, res = corrida(args.logs, H, init, args.t0, args.max_win)
        if not len(hq):
            ap.error("sin ventanas útiles: revise logs, t0 y max-win")
        Wq = q.W / W_ONE
        Wf = f.W
        dif = np.abs(Wq - Wf).max()
        # convergencia del fundamental (b1 α) al 2 % del valor float final
        tgt = Wf[1, 0]
        conv = next((i for i, w in enumerate(hq[:, 1, 0] / W_ONE)
                     if abs(w - tgt) < 0.02 * abs(tgt)), None)
        conv_s = f"primer cruce en t={ht[conv]:.2f} s procesados (no asentamiento)" if conv is not None else "no alcanzada"
        print(f"\n=== init W = {init} ===  ({res['n_upd']} actualizaciones)")
        print(f"max |W_bitexact − W_float| = {dif:.5f}   "
              f"(W_b1α: {Wq[1,0]:+.4f} vs float {Wf[1,0]:+.4f})")
        print(f"convergencia b1α al 2 %: {conv_s}")
        for eje, nom in ((0, "α"), (1, "β")):
            a = arm(Wq[:, eje])
            print(f"armónicos W_{nom} [% fund]: " +
                  "  ".join(f"h{k}={100*a[k-1]:.3f}" for k in range(2, H + 1)))
        print(f"RMS residuo: sinusoidal {res['rms_res_sin_mv']:.1f} mV  "
              f"→ ADALINE {res['rms_res_ad_mv']:.1f} mV  "
              f"(−{100*(1 - res['rms_res_ad_mv']/max(1e-9, res['rms_res_sin_mv'])):.1f} %)")
        resultados[init] = (ht, hq, hf, Wq, Wf, res)

    # ── comparación contra referencias ──────────────────────────────────
    oc = json.load(open("python_scripts/bemf_2804_oc.json"))
    Wq = resultados["modelo"][3]
    lam_oc, lam_nom = oc["lambda_Wb"], LAMBDA_NOM
    print("\n=== forma aprendida (init = modelo) vs referencia OC (Exp. 2b) ===")
    print(f"  (W esperado = (S/λ_oc)·(λ_oc/λ_nom); λ_oc = {lam_oc*1e3:.3f} mWb, "
          f"λ_nom = {lam_nom*1e3:.1f} mWb → factor {lam_oc/lam_nom:.4f})")
    json_out = {"H": H, "W_frac": W_FRAC, "pole_pairs": POLE_PAIRS,
                "MU_Q16": MU_Q16, "LK_Q15": LK_Q15, "EPK_MIN_MV": EPK_MIN_MV,
                "lambda_nom_Wb": lam_nom, "logs": [p.split("/")[-1] for p in args.logs]}
    for eje, nom, pre in ((0, "α", "S_alpha"), (1, "β", "S_beta")):
        aprend = Wq[:, eje]
        # forma de referencia en unidades de W: S_oc/λ_nom (W normaliza por
        # e_pk = λ_nom·ω; el factor λ_oc/λ_nom queda dentro de S/λ_nom)
        ref_cos = np.array(oc[pre + "_cos"][:H]) / lam_nom
        ref_sin = np.array(oc[pre + "_sin"][:H]) / lam_nom
        a_ap = np.hypot(aprend[0::2], aprend[1::2])
        a_rf = np.hypot(ref_cos, ref_sin)
        print(f"  eje {nom}: fund aprendido {a_ap[0]:.4f} vs ref {a_rf[0]:.4f}  |  "
              + "  ".join(f"h{k}: {100*a_ap[k-1]/a_ap[0]:.3f} vs {100*a_rf[k-1]/a_rf[0]:.3f}"
                          for k in (2, 3, 5, 7) if k <= H))
        json_out[f"W_{nom}_cos_q27"] = [int(v) for v in (aprend[0::2] * W_ONE)]
        json_out[f"W_{nom}_sin_q27"] = [int(v) for v in (aprend[1::2] * W_ONE)]
        json_out[f"ref_{nom}_cos"] = ref_cos.tolist()
        json_out[f"ref_{nom}_sin"] = ref_sin.tolist()

    # ── error de la recurrencia (lo que se evaluará en la ISR a 50 kHz) ──
    thg = np.linspace(0, 2 ** 32, 4096, endpoint=False, dtype=np.uint64)
    err_max = 0.0
    for tq in thg[::4]:
        xr = basis_recurrencia(int(tq), H)
        xe = basis_exacta(int(tq), H)
        for (cr, sr), (ce, se) in zip(xr, xe):
            err_max = max(err_max, abs(cr / 2 ** 31 - ce), abs(sr / 2 ** 31 - se))
    print(f"\nerror máximo de la recurrencia Q31 vs sin/cos exacto: {err_max:.2e} "
          f"(≈ {err_max*1540:.2f} mV de e a 600 rpm)")

    json.dump(json_out, open(args.json, "w"), indent=1)
    print(f"\nW_q27 → {args.json}")

    # ── figuras ─────────────────────────────────────────────────────────
    if not args.no_figs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ht, hq, hf, Wq, Wf, _ = resultados["cero"]
        fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        coeffs = [(1, 0, "b1 α → −1"), (0, 1, "a1 β → +1"),
                  (4, 0, "a3 α"), (5, 0, "b3 α"),
                  (8, 0, "a5 α"), (13, 0, "b7 α")]
        for a, (j, eje, nom) in zip(ax.flat, coeffs):
            if j >= 2*H:
                a.set_visible(False)
                continue
            a.plot(ht, hq[:, j, eje] / W_ONE, lw=0.8, label="aritmética Q27")
            a.plot(ht, hf[:, j, eje], lw=0.8, ls="--", label="float64")
            a.axhline(Wf[j, eje], color="k", lw=0.5, ls=":")
            a.set_title(nom); a.grid(alpha=.3); a.legend(fontsize=7)
        ax[-1, 0].set_xlabel("t de log [s]")
        fig.suptitle(f"ADALINE online — réplica bit-exacta (init W = 0, H = {H}, μ = 2e-3 @1 kHz)")
        fig.tight_layout(); fig.savefig(args.png + "_w.png", dpi=110)

        th = np.linspace(0, 2 * np.pi, 361)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for eje, nom, pre, ideal in ((0, "α", "S_alpha", -np.sin(th)),
                                     (1, "β", "S_beta", np.cos(th))):
            g = np.zeros_like(th)
            for k in range(1, H + 1):
                g += Wq[2 * k - 2, eje] * np.cos(k * th) + Wq[2 * k - 1, eje] * np.sin(k * th)
            ref = np.zeros_like(th)
            for k in range(1, H + 1):
                ref += (json_out[f"ref_{nom}_cos"][k - 1] * np.cos(k * th)
                        + json_out[f"ref_{nom}_sin"][k - 1] * np.sin(k * th))
            ax[eje].plot(np.degrees(th), g, lw=1.4, label="aprendida online (Q)")
            ax[eje].plot(np.degrees(th), ref, "--", lw=1.2, label="referencia OC (2b)")
            ax[eje].plot(np.degrees(th), ideal, "k:", lw=0.8, label="sinusoide ideal")
            ax[eje].set_title(f"forma e_{nom}/(λ_nom·ω)"); ax[eje].set_xlabel("θ_e [°]")
            ax[eje].grid(alpha=.3); ax[eje].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.png + "_forma.png", dpi=110)
        print(f"figuras → {args.png}_w.png, {args.png}_forma.png")


if __name__ == "__main__":
    main()
