#!/usr/bin/env python3
"""Resumen rápido de un log de 07_m2pc (sesión 26): ε_q/ε_d por tramos de
10 s (línea E), estado de la ISR, contadores del ADALINE y evolución de W."""
import re
import sys

import numpy as np

K = 256 * 32.93 / 1e3          # Q8 LSB → mA
path = sys.argv[1]
E, W, boot, trip, m2pc, ada = [], [], None, None, [], []
for ln in open(path, errors="ignore"):
    if ln.startswith("[boot]"):
        E, W, m2pc, ada = [], [], [], []
    elif ln.startswith("[TRIP"):
        trip = ln.strip()
    elif ln.startswith("[adaline] enabled"):
        boot = ln.strip()
    elif ln.startswith("[adaline] updates"):
        ada.append(ln.strip())
    elif ln.startswith("[m2pc] corr"):
        m2pc.append(ln.strip())
    elif ln.startswith("W "):
        W.append([int(x) for x in ln.split()[1:]])
    else:
        m = re.match(r"^E (\d+) (-?\d+) (-?\d+) (-?\d+) (\d+) (\d+) (-?\d+) (-?\d+)", ln)
        if m:
            E.append([int(x) for x in m.groups()])

print(f"== {path}")
print(boot or "(sin banner adaline)")
e = np.array(E, float)
if len(e):
    t = e[:, 0] / 1000
    print(f"E: {len(e)} líneas, t = {t[0]:.0f}..{t[-1]:.0f} s")
    for t0 in range(0, int(t[-1]) + 1, 10):
        s = (t >= t0) & (t < t0 + 10)
        if s.sum() > 20:
            print(f"  t={t0:3d}-{t0+10:3d}: eq={e[s,4].mean()/K:6.1f} mA  ed={e[s,5].mean()/K:6.1f} mA  "
                  f"id={e[s,1].mean()/K:+6.1f}  iq={e[s,2].mean()/K:+6.1f}  pkpk={(e[s,7]-e[s,6]).mean()/K:5.0f} mA")
    s = t >= max(10, t[-1] - 30)
    print(f"  ÚLTIMOS 30 s: eq={e[s,4].mean()/K:.1f} mA  ed={e[s,5].mean()/K:.1f} mA")
for ln in m2pc[-2:]:
    print(" ", ln)
for ln in ada[-2:]:
    print(" ", ln)
w = np.array(W, float)
if len(w):
    H = (w.shape[1] - 1) // 4
    wl = w[:, 1:].reshape(len(w), 2 * H, 2) / 2 ** 27
    for i in sorted({0, len(w) // 2, len(w) - 1}):
        b1a, a1b = wl[i, 1, 0], wl[i, 0, 1]
        fund = np.hypot(wl[i, 0, 0], wl[i, 1, 0])
        h = "  ".join(f"h{k}={100*np.hypot(wl[i,2*(k-1),0], wl[i,2*(k-1)+1,0])/fund:.2f}" for k in range(2, H + 1))
        print(f"  W t={w[i,0]/1000:5.1f}: b1α={b1a:+.4f} a1β={a1b:+.4f} a1α={wl[i,0,0]:+.4f} b1β={wl[i,1,1]:+.4f}  {h} %")
if trip:
    print("  !!!", trip)
