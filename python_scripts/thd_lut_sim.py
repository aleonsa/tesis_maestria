"""THD de la BEMF de las tablas de simulación (Cap. 5) frente a la medida en el banco (Cap. 6).

Uso: uv run --no-project --with scipy --with numpy thd_lut_sim.py
- bemf_lut.mat / lut_alpha: planta sintética del Cap. 5 (main_comparison.m con USE_REAL_BEMF=false).
- bemf_real_lut.mat / lut_alpha_real: motor real de dSPACE (Anaheim), no usado como planta en el Cap. 5.
- bemf_2804_oc.json: BEMF del 2804 en circuito abierto (Exp. 2b), armónicos en % del fundamental.
THD = sqrt(sum_{k>=2} h_k^2) / h_1, con h_k de la FFT de la tabla (un periodo eléctrico, 1000 puntos).
"""
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent


def thd_tabla(x, kmax=39):
    c = np.abs(np.fft.rfft(np.asarray(x).ravel()))[: kmax + 1]
    return 100 * np.sqrt(np.sum(c[2:] ** 2)) / c[1]


sint = sio.loadmat(ROOT / "fcs_m2pc_v2/data/bemf_lut.mat")
real = sio.loadmat(ROOT / "fcs_m2pc_v2/data/bemf_real_lut.mat")
oc = json.load(open(ROOT / "python_scripts/bemf_2804_oc.json"))
h = np.array(oc["harm_pct_alpha"])

print(f"planta sintética del Cap. 5 (alfa): THD = {thd_tabla(sint['lut_alpha']):.2f} %")
print(f"planta sintética del Cap. 5 (beta): THD = {thd_tabla(sint['lut_beta']):.2f} %")
print(f"motor real dSPACE (alfa):           THD = {thd_tabla(real['lut_alpha_real']):.2f} %")
print(f"banco 2804, circuito abierto (alfa): THD = {np.sqrt(np.sum(h[1:] ** 2)):.2f} %  (h normalizado: h[0]={h[0]:.1f})")
