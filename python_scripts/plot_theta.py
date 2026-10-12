#!/usr/bin/env python3
"""Plot en tiempo real de la posición del AS5600 (como el Serial Plotter de Arduino).

Lee /dev/ttyACM0, parsea las líneas del loop de 03_as5600:

    raw=4089  ang=359.3 deg

y grafica el ángulo contra tiempo en una ventana deslizante. Desenvuelve el
wrap 4095->0 para que la rampa se vea continua al girar el rotor.

Uso:
    python3 plot_theta.py                      # plot en vivo
    python3 plot_theta.py --save vuelta.csv    # además guarda CSV (t, raw, ang)
    python3 plot_theta.py --window 10          # ventana de 10 s (default 5)
    python3 plot_theta.py --port /dev/ttyACM1  # otro puerto

El CSV guardado sirve después para análisis offline (monotonía, ruido en
reposo, cuentas por vuelta) y como referencia al validar el observador PLL
(app 04_theta_pll): misma herramienta, misma métrica.

Ctrl+C para salir.
"""

import argparse
import re
import sys
import time
from collections import deque

import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LINE_RE = re.compile(r"raw=(\d+)\s+ang=([\d.]+)")
# Stream de 04_theta_pll: "S t=12345 raw=2048 th=1234.56 om=45.2"
# th ya viene desenvuelto del MCU; se grafica tal cual.
PLL_RE = re.compile(r"^S t=(\d+) raw=(\d+) th=(-?[\d.]+) om=(-?[\d.]+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--window", type=float, default=5.0,
                    help="segundos visibles en el plot (default 5)")
    ap.add_argument("--save", metavar="CSV", default=None,
                    help="guardar todo lo recibido a CSV")
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        sys.exit(f"No pude abrir {args.port}: {e}")

    print(f"Escuchando {args.port} @ {args.baud}. Ctrl+C para salir.")
    if args.save:
        print(f"Guardando CSV en {args.save}")

    t0 = time.monotonic()
    ts = deque()      # tiempo [s]
    angs = deque()    # ángulo desenvuelto [deg]
    csv_f = open(args.save, "w") if args.save else None
    if csv_f:
        csv_f.write("t_s,raw,ang_deg\n")

    ang_prev = None
    ang_unwrapped = 0.0

    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("ángulo desenvuelto [deg]")
    ax.set_title("AS5600 en vivo")
    ax.grid(True, alpha=0.3)
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def poll():
        """Drena el buffer serie; devuelve True si llegó al menos un dato."""
        nonlocal ang_prev, ang_unwrapped
        got = False
        while ser.in_waiting:
            try:
                line_b = ser.readline()
            except serial.SerialException:
                continue
            m_pll = PLL_RE.search(line_b.decode("utf-8", errors="ignore"))
            if m_pll:
                # Stream del PLL: th desenvuelto directo del MCU.
                ang = float(m_pll.group(3))
                raw = int(m_pll.group(2))
                t = time.monotonic() - t0
                ts.append(t)
                angs.append(ang)
                if csv_f:
                    csv_f.write(f"{t:.6f},{raw},{ang:.3f}\n")
                ang_prev = ang
                got = True
                continue
            m = LINE_RE.search(line_b.decode("utf-8", errors="ignore"))
            if not m:
                continue
            raw, ang = int(m.group(1)), float(m.group(2))
            if ang_prev is not None:
                d = ang - ang_prev
                if d > 180.0:      # wrap hacia abajo (4095 -> 0)
                    d -= 360.0
                elif d < -180.0:   # wrap hacia arriba (0 -> 4095)
                    d += 360.0
                ang_unwrapped += d
            else:
                ang_unwrapped = ang
            ang_prev = ang
            t = time.monotonic() - t0
            ts.append(t)
            angs.append(ang_unwrapped)
            if csv_f:
                csv_f.write(f"{t:.6f},{raw},{ang:.3f}\n")
            got = True
        return got

    def update(_frame):
        if not poll() and not ts:
            return line, text
        t_min = ts[-1] - args.window
        while ts and ts[0] < t_min:
            ts.popleft()
            angs.popleft()
        line.set_data(ts, angs)
        ax.set_xlim(max(0.0, t_min), max(args.window, ts[-1]))
        if angs:
            lo, hi = min(angs), max(angs)
            pad = max(1.0, 0.05 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        text.set_text(f"ang={ang_prev:6.2f}°  n={len(ts)} pts en ventana"
                      if ang_prev is not None else "esperando datos...")
        return line, text

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        if csv_f:
            csv_f.close()
            print(f"CSV cerrado: {args.save}")


if __name__ == "__main__":
    main()
