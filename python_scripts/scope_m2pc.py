#!/usr/bin/env python3
"""Scopes de 07_m2pc, sin modificar ni enviar comandos al firmware.

python3 python_scripts/scope_m2pc.py --window 3
python3 python_scripts/scope_m2pc.py --replay datos_banco/m2pc_s21_valle_on.log
"""

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import re
import threading
import time


K_RAW_PER_A = 32.93
SAMPLE_RE = re.compile(r"^L (\d+) (\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)\s*$")
SPEED_RE = re.compile(r"w\* = (-?\d+) rpm")


@dataclass(frozen=True)
class Sample:
    t: float
    rpm: float
    alpha: float
    beta: float
    u_alpha: float
    u_beta: float
    id_ma: float
    iq_ma: float
    ref_ma: float


def parse_sample(line):
    """Unidades SI/mA; dq usa el ángulo al cierre y es aproximado."""
    match = SAMPLE_RE.fullmatch(line)
    if match is None:
        return None
    t, theta, ia, ib, ua, ub, rpm, ref = map(int, match.groups())
    angle = theta * math.tau / 65536
    alpha = ia * 1000 / (256 * K_RAW_PER_A)
    beta = ib * 1000 / (256 * K_RAW_PER_A)
    c, s = math.cos(angle), math.sin(angle)
    return Sample(t / 1000, rpm / 10, alpha, beta, ua / 1000, ub / 1000,
                  alpha * c + beta * s, -alpha * s + beta * c, ref)


class Reader(threading.Thread):
    """Único lector de serie; guarda el log fuera del hilo gráfico."""

    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.pending = deque(maxlen=20000)
        self.dropped = 0
        self.speed_ref = args.speed_ref
        self.speed_source = "manual/supuesta"
        self.status = "Conectando…"
        self.health = "Sin diagnóstico todavía"
        self.offset = ""
        self.trip = False
        self.last_rx = None
        self.epoch = 0
        self.previous_t = None

    def consume(self, line):
        sample = parse_sample(line)
        with self.lock:
            if line.startswith("[boot]"):
                self.epoch += 1
                self.pending.clear()
                self.previous_t = None
                self.trip = False
                self.speed_ref = self.args.speed_ref
                self.speed_source = "manual/supuesta"
                self.health = "Reinicio detectado"
                self.offset = ""
            speed = SPEED_RE.search(line)
            if speed:
                self.speed_ref = float(speed.group(1))
                self.speed_source = "banner firmware"
            if line.startswith("[m2pc] corr/s="):
                self.health = line
            if line.startswith("[off]"):
                self.offset = line
            if "[TRIP]" in line:
                self.trip = True
                self.health = line
            if sample is not None:
                if self.previous_t is not None and sample.t < self.previous_t:
                    self.epoch += 1
                    self.pending.clear()
                self.previous_t = sample.t
                if len(self.pending) == self.pending.maxlen:
                    self.dropped += 1
                self.pending.append((self.epoch, sample))
                self.last_rx = time.monotonic()

    def run(self):
        try:
            if self.args.replay:
                with Path(self.args.replay).open(errors="replace") as source:
                    self.status = "Reproduciendo archivo"
                    start = time.monotonic()
                    first_t = previous_t = None
                    for line in source:
                        if self.stop_event.is_set():
                            break
                        line = line.strip()
                        sample = parse_sample(line)
                        if sample is not None:
                            if first_t is None or sample.t < previous_t:
                                first_t = sample.t
                                start = time.monotonic()
                            previous_t = sample.t
                            delay = start + (sample.t - first_t) / self.args.rate - time.monotonic()
                            if delay > 0 and self.stop_event.wait(delay):
                                break
                        self.consume(line)
                self.status = "Reproducción terminada"
            else:
                import serial

                # Bloqueo cooperativo: no ejecutar cat ni otro monitor a la vez.
                with serial.Serial(self.args.port, self.args.baud, timeout=0.2,
                                   exclusive=True) as source, Path(self.args.save).open("x") as log:
                    self.status = f"Serie {self.args.port} @ {self.args.baud}"
                    fragment = b""
                    flushed = time.monotonic()
                    while not self.stop_event.is_set():
                        chunk = source.read(max(1, min(source.in_waiting, 65536)))
                        fragment += chunk
                        lines = fragment.split(b"\n")
                        fragment = lines.pop()
                        for raw in lines:
                            line = raw.decode("utf-8", errors="replace").rstrip("\r")
                            log.write(line + "\n")
                            self.consume(line)
                        if len(fragment) > 65536:
                            log.write(fragment.decode("utf-8", errors="replace") + "\n")
                            fragment = b""
                        if time.monotonic() - flushed > 1:
                            log.flush()
                            flushed = time.monotonic()
                    if fragment:
                        log.write(fragment.decode("utf-8", errors="replace"))
        except Exception as exc:
            self.status = f"ERROR: {exc}"
            print(self.status, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--window", type=float, default=3, help="segundos visibles (0.05–30)")
    ap.add_argument("--speed-ref", type=float, default=300, help="rpm supuestas hasta recibir el banner")
    ap.add_argument("--save", help="log nuevo; por defecto datos_banco/m2pc_live_<fecha>.log")
    ap.add_argument("--replay", help="reproducir un log sin abrir el puerto")
    ap.add_argument("--rate", type=float, default=1, help="velocidad de reproducción")
    ap.add_argument("--duration", type=float, help="cerrar tras N segundos (pruebas)")
    ap.add_argument("--snapshot", help="guardar imagen al cerrar")
    args = ap.parse_args()
    if not 0.05 <= args.window <= 30 or args.rate <= 0:
        ap.error("--window debe estar entre 0.05 y 30; --rate debe ser positivo")
    if not args.save:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.save = str(Path(__file__).resolve().parents[1] / "datos_banco" / f"m2pc_live_{stamp}.log")

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button, RadioButtons

    reader = Reader(args)
    history = deque(maxlen=35000)
    current_epoch = -1
    paused = False
    window = args.window
    fig, grid = plt.subplots(4, 2, figsize=(15, 10), sharex=True)
    fig.canvas.manager.set_window_title("Banco FCS-M²PC — scopes en vivo")
    fig.subplots_adjust(left=.07, right=.98, bottom=.21, top=.91, hspace=.6, wspace=.22)
    fig.suptitle("FCS-M²PC · stream de medias a ~1 kHz · NO muestra rizado PWM de 50 kHz", fontsize=13)
    axes = grid.ravel()
    specs = [
        ("Velocidad", "rpm", ("ω medida", "ω* (banner o --speed-ref)")),
        ("Error de velocidad (referencia nominal)", "rpm", ("ω* − ω",)),
        ("Corriente d — aproximada", "mA", ("id ≈", "id* = 0")),
        ("Corriente q — aproximada", "mA", ("iq ≈", "iq* (al imprimir)")),
        ("Errores de corriente — aprox., NO RMSE del control", "mA", ("−id ≈", "iq* − iq ≈")),
        ("Corrientes αβ — medias de 1 ms", "mA", ("iα", "iβ")),
        ("Tensión comandada — no tensión medida", "V", ("uα", "uβ")),
        ("Salida del PI / referencia del lazo interno", "mA", ("iq*",)),
    ]
    curves = []
    for ax, (title, unit, labels) in zip(axes, specs):
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(unit)
        ax.grid(alpha=.25)
        curves.append([ax.plot([], [], lw=1, label=label)[0] for label in labels])
        ax.legend(loc="upper right", fontsize=7)
    for ax in grid[-1]:
        ax.set_xlabel("tiempo del logger [s]")
    footer = fig.text(.02, .09, "Esperando datos…", fontsize=8, va="top", family="monospace")
    fig.text(.02, .012, "dq usa corrientes promediadas + ángulo final: desfase aparente. Pausa congela solo la vista, no el log.", fontsize=8)
    button = Button(fig.add_axes([.02, .125, .12, .04]), "Pausar vista")
    radio_labels = tuple(dict.fromkeys((f"{window:g} s", "0.1 s", "1 s", "3 s", "10 s")))
    radios = RadioButtons(fig.add_axes([.79, .015, .19, .12]), radio_labels, active=0)

    def pause(_):
        nonlocal paused
        paused = not paused
        button.label.set_text("Seguir" if paused else "Pausar vista")

    def select(label):
        nonlocal window
        window = float(label.split()[0])

    button.on_clicked(pause)
    radios.on_clicked(select)
    started = time.monotonic()

    def update(_):
        nonlocal current_epoch
        with reader.lock:
            pending = list(reader.pending)
            reader.pending.clear()
            speed_ref, speed_source = reader.speed_ref, reader.speed_source
            health, offset, trip = reader.health, reader.offset, reader.trip
            last_rx, dropped, epoch = reader.last_rx, reader.dropped, reader.epoch
        if epoch != current_epoch:
            history.clear()
            current_epoch = epoch
        history.extend(sample for e, sample in pending if e == current_epoch)
        age = "sin muestras" if last_rx is None else f"última muestra hace {time.monotonic() - last_rx:.1f} s"
        footer.set_text(f"{reader.status} | {age} | descartes visor={dropped}\n"
                        f"ω*={speed_ref:g} rpm ({speed_source}) | {health}\n"
                        f"{offset}\n" + (f"Log: {args.save}" if not args.replay else f"Archivo: {args.replay}"))
        footer.set_color("red" if trip or reader.status.startswith("ERROR") else "black")
        # El zoom de la barra se usa con la vista pausada.
        if history and not paused:
            end = history[-1].t
            visible = [s for s in history if s.t >= end - window]
            t = np.array([s.t for s in visible])
            rpm = np.array([s.rpm for s in visible])
            d = np.array([s.id_ma for s in visible])
            q = np.array([s.iq_ma for s in visible])
            ref = np.array([s.ref_ma for s in visible])
            zero = np.zeros(len(t))
            values = [(rpm, zero + speed_ref), (speed_ref - rpm,), (d, zero), (q, ref),
                      (-d, ref - q), ([s.alpha for s in visible], [s.beta for s in visible]),
                      ([s.u_alpha for s in visible], [s.u_beta for s in visible]), (ref,)]
            for ax, lines, ys in zip(axes, curves, values):
                for line, y in zip(lines, ys):
                    line.set_data(t, y)
                ax.relim()
                ax.autoscale_view(scalex=False)
            axes[0].set_xlim(max(0, end - window), max(window, end))
        if args.duration and time.monotonic() - started >= args.duration:
            if args.snapshot:
                fig.savefig(args.snapshot, dpi=120)
            plt.close(fig)

    reader.start()
    animation = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    try:
        plt.show()
    finally:
        reader.stop_event.set()
        reader.join(timeout=2)
        if args.snapshot and not args.duration:
            fig.savefig(args.snapshot, dpi=120)
    return animation


if __name__ == "__main__":
    main()
