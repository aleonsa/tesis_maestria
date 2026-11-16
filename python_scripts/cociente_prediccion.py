"""Criterio de la vía de la predicción: kappa = 2*delta_e / V_dc (Cap. 6, sec. 6.6).

Uso: uv run --no-project --with scipy --with numpy cociente_prediccion.py

El efecto de un error de BEMF delta_e en la predicción de dos pasos es 2*G*delta_e,
con G = T_s/L. Dividido por el rizo de conmutación V_dc*T_s/L queda kappa = 2*delta_e/V_dc.
delta_e se toma como amplitud equivalente de pico del error de modelo en el eje alfa:
sqrt(2) * RMS_theta( e_modelo_alfa - e_real_alfa ). Con la planta del banco y el modelo
senoidal de amplitud correcta, esto coincide con THD * e_pk (make_advisor_figs.py).

Simulación (Cap. 5): e = K_e * w_m * s(theta_e), K_e = 0.40355, w_m = 80 rad/s,
V_dc = 240 V; planta = bemf_lut.mat/lut_alpha_real (USE_REAL_BEMF=false);
modelos SIN = sin(theta) y TRAP = clarke(trapecio) de fcs_m2pc_v2/models/.
Banco: lambda = 3.487 mWb, p = 7, 300 rpm, V_dc = 12 V, THD medio de bemf_2804_oc.json.
"""
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
N = 1000


def trap_alpha(th):
    def ph(t):
        t = np.mod(t, 2 * np.pi)
        return np.where(t < np.pi / 6, t * 6 / np.pi,
               np.where(t < 5 * np.pi / 6, 1.0,
               np.where(t < 7 * np.pi / 6, 1 - (t - 5 * np.pi / 6) * 6 / np.pi,
               np.where(t < 11 * np.pi / 6, -1.0, -1 + (t - 11 * np.pi / 6) * 6 / np.pi))))
    a, b, c = ph(th), ph(th - 2 * np.pi / 3), ph(th - 4 * np.pi / 3)
    return (2 * a - b - c) / 3  # Clarke de amplitud invariante, eje alfa


def fund(x, th):
    return 2 / len(x) * np.sum(x * np.exp(-1j * th))


def dpk(model, real):
    return np.sqrt(2) * np.sqrt(np.mean((model - real) ** 2))


def calcular(verbose=True):
    """Devuelve los kappa del cuadro tab:kappa (y los imprime si verbose)."""
    out = {}
    # --- simulación ---
    d = sio.loadmat(ROOT / "fcs_m2pc_v2/data/bemf_lut.mat")
    th = d["lut_theta"].ravel()
    s_real = d["lut_alpha_real"].ravel()
    KE, WM, VDC_SIM = 0.40355, 80.0, 240.0
    esc = KE * WM
    c_real, c_sin = fund(s_real, th), fund(np.sin(th), th)
    out['fund_planta'] = abs(c_real)
    if verbose:
        print(f"[sim] fundamental planta |c1| = {abs(c_real):.4f}, fase vs sin(theta) = {np.degrees(np.angle(c_real / c_sin)):+.2f} deg")
    for nombre, s_mod in [("SIN", np.sin(th)), ("TRAP", trap_alpha(th))]:
        de = dpk(s_mod, s_real) * esc
        # parte de forma: modelo reescalado al fundamental de la planta
        s_esc = s_mod * abs(c_real) / abs(fund(s_mod, th))
        de_forma = dpk(s_esc, s_real) * esc
        out[f'sim_{nombre}'] = 2 * de / VDC_SIM
        out[f'fund_{nombre}'] = abs(fund(s_mod, th))
        out[f'sim_{nombre}_forma'] = 2 * de_forma / VDC_SIM
        if verbose:
            print(f"[sim] {nombre:4s}: delta_e = {de:6.2f} V (solo forma {de_forma:5.2f} V)  "
              f"kappa = {2 * de / VDC_SIM:.3f} (solo forma {2 * de_forma / VDC_SIM:.3f})")

    # --- banco ---
    oc = json.load(open(ROOT / "python_scripts/bemf_2804_oc.json"))
    thd = float(np.mean([np.sqrt(np.sum(np.array(h[1:]) ** 2)) for h in oc["harm_pct_abc"]])) / 100
    e_pk = 3.487e-3 * 7 * 2 * np.pi * 300 / 60
    de_b = thd * e_pk
    out['banco_real'] = 2 * de_b / 12.0
    if verbose:
        print(f"[banco] e_pk = {e_pk:.3f} V, THD medio = {100 * thd:.2f} %, delta_e = {1e3 * de_b:.2f} mV, "
          f"kappa = {2 * de_b / 12.0:.2e}")

    # modelo equivocado de la sec. 6.5 (0.8*lambda, +10 deg) y error que subiría el RMS un 10 %
    wrong = abs(1 - 0.8 * np.exp(1j * np.radians(10)))
    out['banco_equivocado'] = 2 * wrong * e_pk / 12.0
    if verbose:
        print(f"[banco] modelo equivocado: delta_e = {wrong * e_pk:.3f} V, kappa = {2 * wrong * e_pk / 12.0:.3f}")
    num = json.load(open(ROOT / "presentation/fig/asesor_numeros.json"))
    p10 = num["error_para_10pct_en_rms_pct"] / 100
    out['banco_10pct'] = 2 * p10 * e_pk / 12.0
    if verbose:
        print(f"[banco] error para +10 % de RMS: {100 * p10:.0f} % de e_pk -> kappa = {2 * p10 * e_pk / 12.0:.3f}")

    return out


if __name__ == "__main__":
    calcular()
