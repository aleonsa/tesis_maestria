#!/usr/bin/env python3
"""Genera docs/banco_2804.html: resumen visual autocontenido (imágenes embebidas,
sin red) de lo hecho y medido en el banco B-G431B-ESC1 + 2804 hasta la sesión 24.
La animación de la BEMF usa los coeficientes de Fourier medidos en circuito
abierto (python_scripts/bemf_2804_oc.json) y los de la reconstrucción desde el
controlador (bemf_2804_fourier.json).

    python3 python_scripts/make_bench_report.py && xdg-open docs/banco_2804.html
"""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "banco_2804.html"

FIGS = {
    "pll":   "firmware/apps/04_theta_pll/doc/validacion_pll_2026-10-12.png",
    "bemf300": "firmware/apps/07_m2pc/doc/bemf_s22_cordic_fix.png",
    "exp1":  "firmware/apps/07_m2pc/doc/exp1_escalon_iq.png",
    "exp2":  "firmware/apps/07_m2pc/doc/exp2_trayectoria_seno.png",
    "rec":   "firmware/apps/07_m2pc/doc/bemf_rec_s23_forma.png",
    "lms":   "firmware/apps/07_m2pc/doc/bemf_rec_s23_lms.png",
    "oc":    "firmware/apps/08_bemf_oc/doc/bemf_oc_s24_forma.png",
}


def b64(path):
    return "data:image/png;base64," + base64.b64encode((ROOT / path).read_bytes()).decode()


oc = json.load(open(ROOT / "python_scripts/bemf_2804_oc.json"))
rec = json.load(open(ROOT / "python_scripts/bemf_2804_fourier.json"))
imgs = {k: b64(v) for k, v in FIGS.items()}

harm_oc = [[oc["harm_pct_abc"][x][k] for x in range(3)] for k in range(1, 8)]   # h2..h8, por fase
harm_rec_a = rec["harm_pct_alpha"][1:8]
harm_rec_b = rec["harm_pct_beta"][1:8]

HTML = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Banco FCS‑M²PC · 2804 — lo hecho y medido</title>
<style>
:root{--bg:#0b0f1a;--bg2:#121a2b;--card:#161f33;--ink:#e8edf7;--mut:#8a97b3;--a:#4fd1c5;--b:#f6ad55;--c:#f687b3;--g:#68d391;--r:#fc8181;--line:#243049}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;font:16px/1.55 system-ui,-apple-system,"Segoe UI",Inter,Roboto,sans-serif;color:var(--ink);background:radial-gradient(1200px 600px at 10% -10%,#1b2a4a 0%,transparent 60%),radial-gradient(900px 500px at 100% 0%,#143a3a 0%,transparent 55%),var(--bg)}
a{color:var(--a);text-decoration:none}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
header{padding:64px 0 24px}
h1{font-size:clamp(30px,5vw,54px);line-height:1.05;margin:0 0 10px;letter-spacing:-.02em}
h1 span{background:linear-gradient(90deg,var(--a),var(--g));-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mut);font-size:18px;max-width:820px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:34px 0 10px}
.kpi{background:linear-gradient(160deg,var(--card),#0f1627);border:1px solid var(--line);border-radius:16px;padding:18px 18px 14px;position:relative;overflow:hidden}
.kpi::after{content:"";position:absolute;inset:auto -30% -60% auto;width:160px;height:160px;border-radius:50%;background:radial-gradient(circle,rgba(79,209,197,.18),transparent 70%)}
.kpi b{display:block;font-size:34px;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.kpi small{color:var(--mut)}
.kpi .u{font-size:18px;color:var(--mut);margin-left:3px}
section{padding:40px 0;border-top:1px solid var(--line)}
h2{font-size:28px;margin:0 0 6px;letter-spacing:-.01em}
h2 .n{color:var(--a);font-family:ui-monospace,Menlo,monospace;font-size:16px;margin-right:10px;vertical-align:middle}
.lead{color:var(--mut);margin:0 0 22px;max-width:900px}
.grid{display:grid;gap:18px}.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:repeat(3,1fr)}
@media(max-width:900px){.g2,.g3{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;transition:transform .25s,border-color .25s}
.card:hover{transform:translateY(-2px);border-color:#34507a}
.card h3{margin:0 0 8px;font-size:17px}
.card p{margin:6px 0;color:var(--mut);font-size:15px}
.fig{border-radius:12px;overflow:hidden;border:1px solid var(--line);background:#fff}
.fig img{display:block;width:100%;height:auto}
.cap{color:var(--mut);font-size:13px;margin:8px 2px 0}
table{width:100%;border-collapse:collapse;font-size:14.5px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.04em}
.hl{color:var(--a);font-weight:700}.bad{color:var(--r)}.ok{color:var(--g)}
.tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;border:1px solid var(--line);color:var(--mut);margin-right:6px}
.tag.a{border-color:var(--a);color:var(--a)}.tag.r{border-color:var(--r);color:var(--r)}.tag.g{border-color:var(--g);color:var(--g)}
.timeline{position:relative;margin:26px 0 0;padding-left:26px}
.timeline::before{content:"";position:absolute;left:8px;top:6px;bottom:6px;width:2px;background:linear-gradient(var(--a),var(--c))}
.tl{position:relative;margin:0 0 18px;opacity:0;transform:translateX(-10px);transition:all .6s}
.tl.show{opacity:1;transform:none}
.tl::before{content:"";position:absolute;left:-23px;top:7px;width:12px;height:12px;border-radius:50%;background:var(--bg);border:3px solid var(--a)}
.tl b{color:var(--ink)}.tl span{color:var(--mut)}
.bars{display:grid;grid-template-columns:40px 1fr;gap:6px 10px;align-items:center;font-size:13px}
.bar{height:16px;border-radius:5px;background:linear-gradient(90deg,var(--a),var(--g));transform-origin:left;transform:scaleX(0);transition:transform 1.2s cubic-bezier(.2,.8,.2,1)}
.bar.r{background:linear-gradient(90deg,var(--b),var(--r))}
.bars.show .bar{transform:scaleX(1)}
.bar-row{display:flex;gap:8px;align-items:center;margin:2px 0}.bar-row em{font-style:normal;color:var(--mut);font-size:12px;white-space:nowrap}
canvas{width:100%;height:auto;display:block;border-radius:12px;background:#0a0e18;border:1px solid var(--line)}
.ctrl{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:10px;color:var(--mut);font-size:14px}
input[type=range]{accent-color:var(--a)}
footer{padding:36px 0 60px;color:var(--mut);font-size:13px;border-top:1px solid var(--line)}
code{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#b8c7e6;background:#0e1424;padding:1px 6px;border-radius:5px}
.pulse{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--g);box-shadow:0 0 0 0 rgba(104,211,145,.6);animation:p 1.6s infinite}
@keyframes p{0%{box-shadow:0 0 0 0 rgba(104,211,145,.6)}70%{box-shadow:0 0 0 12px rgba(104,211,145,0)}100%{box-shadow:0 0 0 0 rgba(104,211,145,0)}}
</style></head><body>
<div class="wrap">
<header>
  <div><span class="tag a">B‑G431B‑ESC1</span><span class="tag">2804 · 12N14P · 7 pp · delta</span><span class="tag">AS5600</span><span class="tag">bare‑metal CMSIS</span><span class="tag g"><span class="pulse"></span>&nbsp;sesiones 16–24 · oct 2026</span></div>
  <h1>FCS‑M²PC en el banco:<br><span>lo que se construyó y lo que se midió</span></h1>
  <p class="sub">Control predictivo de corriente de conjunto finito con modulación de dos vectores, a 50 kHz en un Cortex‑M4, cerrando el lazo sobre un motor gimbal con encoder magnético. Este documento resume el estado experimental: cadena de sensado, desempeño del controlador, trayectorias con inversión, y la forma real de la BEMF del motor — medida, no supuesta.</p>
  <div class="kpis">
    <div class="kpi"><b><span data-n="0.42" data-d="2"></span><span class="u">%</span></b><small>THD de la BEMF real (fase a)</small></div>
    <div class="kpi"><b><span data-n="3.49" data-d="2"></span><span class="u">mWb</span></b><small>λ del imán (circuito abierto)</small></div>
    <div class="kpi"><b><span data-n="50" data-d="0"></span><span class="u">kHz</span></b><small>lazo de corriente, 66 candidatos</small></div>
    <div class="kpi"><b><span data-n="15.8" data-d="1"></span><span class="u">µs</span></b><small>paso completo de 20 µs · 0 overruns</small></div>
    <div class="kpi"><b>&lt;<span data-n="200" data-d="0"></span><span class="u">µs</span></b><small>respuesta a escalón de i_q</small></div>
    <div class="kpi"><b>±<span data-n="600" data-d="0"></span><span class="u">rpm</span></b><small>trayectoria senoidal con inversión</small></div>
  </div>
</header>

<section id="bemf">
  <h2><span class="n">01</span>La BEMF real del 2804, animada con sus coeficientes medidos</h2>
  <p class="lead">Las tres tensiones de fase en circuito abierto y el vector αβ, reconstruidos con la serie de Fourier identificada en la sesión 24 (app <code>08_bemf_oc</code>, llanta girada a mano, 300 k muestras). A escala real la desviación de la sinusoide es invisible; el deslizador exagera los armónicos para que se vean. </p>
  <div class="grid g2">
    <div>
      <canvas id="cv" width="640" height="420"></canvas>
      <div class="ctrl">
        <label>exagerar armónicos ×<b id="xv">1</b> <input id="xg" type="range" min="1" max="40" value="1"></label>
        <label>rpm <b id="rv">300</b> <input id="rg" type="range" min="60" max="1200" value="300"></label>
        <label><input id="cmp" type="checkbox"> comparar con la BEMF "vista por el controlador" (Exp. 3a)</label>
      </div>
    </div>
    <div class="card">
      <h3>Armónicos medidos [% del fundamental]</h3>
      <p>Barras: circuito abierto (fase a, IC 95 % ±0.05). Naranja: lo que el controlador "ve" al reconstruir e = u − R·i − L·di/dt — el dead‑time disfrazado de armónicos.</p>
      <div class="bars" id="bars"></div>
      <table style="margin-top:14px"><tr><th></th><th>a</th><th>b</th><th>c</th></tr>
      <tr><td>λ [mWb]</td><td>3.488</td><td>3.488</td><td>3.485</td></tr>
      <tr><td>h3 / h5</td><td>0.29 / 0.28</td><td>0.61 / 0.27</td><td>0.37 / 0.52</td></tr>
      <tr><td>THD</td><td class="hl">0.42 %</td><td class="hl">0.67 %</td><td class="hl">0.64 %</td></tr>
      <tr><td>fase vs encoder</td><td colspan="3">+2.85° (las tres) → <code>PHASE_A_OFFSET_RAW</code> 2558 → 2563</td></tr></table>
    </div>
  </div>
  <div class="grid g2" style="margin-top:18px">
    <div><div class="fig"><img src="{oc}"></div><p class="cap">Exp. 2b — circuito abierto, sin modelo. Red de BEMF de la placa (PA4/PC4/PB11) + 10 kΩ externa para fijar el modo común. Bootstrap por bloques para los IC.</p></div>
    <div><div class="fig"><img src="{rec}"></div><p class="cap">Exp. 3a — reconstrucción desde la tensión comandada con los logs de ±600 rpm: separa ω·S(θ) de D(θ) + dead‑time. Sobreestima h2–h6 entre 2 y 5×: la literatura lo predice.</p></div>
  </div>
</section>

<section id="loop">
  <h2><span class="n">02</span>El lazo de corriente, medido donde debe medirse</h2>
  <p class="lead">Park muestra a muestra en la ISR, RMS de error por ventana de 10 ms y captura a 50 kHz alrededor de un escalón con el PI de velocidad congelado (la planta mecánica es tan rápida — 100 mA ≈ 2400 rpm/s — que un PI activo cancela el escalón en 15 ms).</p>
  <div class="grid g2">
    <div><div class="fig"><img src="{exp1}"></div><p class="cap">Exp. 1 — escalones de ±100 mA a 300 y 600 rpm. Arriba, i_q a 50 kHz; en medio, medias de 10 ms; abajo, la velocidad cae ~50 rpm y el PI la recupera en 40 ms.</p></div>
    <div class="card">
      <h3>M²PC con modelo sinusoidal de BEMF (baseline del ADALINE)</h3>
      <table><tr><th></th><th>300 rpm</th><th>600 rpm</th></tr>
      <tr><td>respuesta al escalón</td><td class="ok">&lt; 200 µs</td><td class="ok">&lt; 200 µs</td></tr>
      <tr><td>RMS(ε_q) / RMS(ε_d)</td><td>37 / 37 mA</td><td>44 / 54 mA</td></tr>
      <tr><td>rizo pk‑pk de i_q</td><td>225 mA</td><td>377 mA</td></tr>
      <tr><td>sesgo de i_q</td><td>+13 mA</td><td>+12 mA</td></tr>
      <tr><td>i_d estacionario</td><td>+5 mA</td><td>+17…30 mA</td></tr>
      <tr><td>paso en la ISR</td><td colspan="2">15.8 µs de 20 (sensado 3.6 + M²PC 10.0 + telemetría)</td></tr></table>
      <p style="margin-top:12px"><b>Lectura.</b> El rizo es el suelo físico del banco: V·T_s/L = 12 V·20 µs/0.86 mH = 280 mA por periodo. El sesgo y el i_d estacionario son el error de régimen de un predictor sin integrador frente a lo que no modela (dead‑time, offset): exactamente lo que un modelo de BEMF aprendido puede absorber.</p>
      <p>Inercia de regalo: K_t·Δi / α → <b>J ≈ 1.5·10⁻⁵ kg·m²</b> (rotor + llanta).</p>
    </div>
  </div>
</section>

<section id="traj">
  <h2><span class="n">03</span>Trayectoria senoidal con inversión de giro</h2>
  <p class="lead">ω* = A·sin(2πft) generada en el lazo de 1 kHz. Pasa por cero y cambia de sentido. Fue también el dataset de la reconstrucción de BEMF (barre velocidad y signo: lo que escala con ω se separa de lo que no).</p>
  <div class="grid g2">
    <div><div class="fig"><img src="{exp2}"></div><p class="cap">Exp. 2 — ±100 / ±300 / ±600 rpm. RMS(ω̂ − ω*) ≈ 6–11 rpm (es el rizo a f_e que el PI de 3 Hz no persigue), retraso 5–18 ms, |i_q*| ≤ 60–114 mA.</p></div>
    <div class="card">
      <h3>Lo que destapó: el M²PC del paper no sabía frenar</h3>
      <p>La eq. 4 del paper elige el sector por la dirección de la BEMF (θ_e + 90°). Con par negativo a baja velocidad, la tensión necesaria apunta a −q y <b>no hay candidato que la sintetice</b>: el motor se paraba a 0 rpm con i_q* saturado y no invertía. La simulación nunca giró en negativo.</p>
      <p><b>Extensión adoptada:</b> sector por la tensión de régimen <code>u_ss = e + R·i*</code> — igual que el paper cuando e domina, sigue a la referencia cuando no. A/B a 600 rpm:</p>
      <table><tr><th>regla</th><th>RMS ε_q</th><th>pk‑pk</th><th>invierte</th></tr>
      <tr><td>BEMF (paper)</td><td>56</td><td>432</td><td class="bad">no</td></tr>
      <tr><td>deadbeat (con término de L)</td><td>72</td><td class="bad">667</td><td class="ok">sí</td></tr>
      <tr><td class="hl">régimen e + R·i*</td><td class="hl">44</td><td class="hl">377</td><td class="ok">sí</td></tr></table>
      <p>El término (L/T_s)(i* − i) con 1 LSB de ruido vale 1.3 V — comparable a la BEMF — y hacía saltar el sector al azar.</p>
    </div>
  </div>
</section>

<section id="chain">
  <h2><span class="n">04</span>La cadena de sensado que lo sostiene</h2>
  <div class="grid g3">
    <div class="card"><h3>Posición: PLL tipo II sobre el AS5600</h3><p>El bus I²C entrega una muestra cada ~470 µs; la ISR corre cada 20 µs. Observador en Q32 puro: extrapola cada tick, corrige cada muestra. Error máx 0.27° mecánicos; 2.1 µs.</p><div class="fig"><img src="{pll}"></div></div>
    <div class="card"><h3>Corriente: tres shunts, OPAMP ×16, offset medido en el valle</h3><p>K = 32.93 raw/A calibrado con DMM. El offset del OPAMP interno deriva ±2 LSB: se mide 50 000 veces por segundo en el valle del PWM (los shunts ven corriente cero). Invariante de control: media(i_a + i_b + i_c) = +0.5 ± 0.4 LSB.</p><p>Referencia Kelvin del PGA: eliminó una anomalía del 17 % en la fase C.</p></div>
    <div class="card"><h3>Tiempo: UEV en el pico, retardo entero</h3><p>Muestreo en el pico del PWM (low‑side ON), comando latcheado en el pico siguiente: i(k+1) con el comando en curso, i(k+2) con cada candidato (Cortés 2012). 66 candidatos en 10 µs con incrementos precalculados por sector — sin división ni trigonometría en el camino caliente.</p><div class="fig"><img src="{bemf300}"></div><p class="cap">BEMF reconstruida del log a 300 rpm, con la Park sana.</p></div>
  </div>
</section>

<section id="bugs">
  <h2><span class="n">05</span>Tres bugs que no se veían — y cómo se cazaron</h2>
  <p class="lead">Ninguno producía un fallo visible: el motor giraba a la velocidad pedida. Todos se destaparon con una prueba de regresión de un minuto que mira la cadena sin un lazo externo que la tape.</p>
  <div class="grid g3">
    <div class="card"><span class="tag r">sesión 22</span><h3>El CORDIC devolvía sin ≡ cos</h3><p>La escritura del CSR tras habilitar el reloj se perdía con <code>-O2</code> + prefetch (CSR en reset: un solo resultado). Y con dos lecturas de 32 bits, la segunda se adelantaba el 3 % de las veces. Park y modelo de BEMF rotos durante dos sesiones; el PI de velocidad lo tapaba.</p><p><b>Fix:</b> relectura del CSR + resultados de 16 bits en una sola lectura. i_dq en open‑loop: 169 ± 160 → 323 ± 16 mA.</p></div>
    <div class="card"><span class="tag r">sesión 23</span><h3>El FPU nunca se habilitó</h3><p>Hard‑float desde la Fase 0; el primer <code>float</code> del proyecto (<code>sinf</code> de la trayectoria) cayó en HardFault NOCP en la primera instrucción de <code>main</code>.</p><p><b>Fix:</b> <code>CPACR</code> en <code>SystemInit</code>.</p></div>
    <div class="card"><span class="tag r">sesión 23–24</span><h3>El sector solo conocía el par positivo</h3><p>Ver §03. Primera corrección (deadbeat) invertía pero metía el ruido de medida en el sector. La definitiva (régimen) invierte y mejora el rizo.</p><p><b>Regla heredada:</b> cada cambio en <code>lib/</code> o en los flags → correr <code>06_openloop_sync</code> (autotest del CORDIC + i_dq DC).</p></div>
  </div>
</section>

<section id="tl">
  <h2><span class="n">06</span>Línea de tiempo</h2>
  <div class="timeline">
    <div class="tl"><b>s16–17 · 11–12 oct</b> <span>— I²C por IRQ validado (2139 lecturas/s, 0 errores). PLL tipo II: 0.27° máx, 2.1 µs.</span></div>
    <div class="tl"><b>s18–19 · 13–14 oct</b> <span>— La paradoja de corriente cerrada (TRGO en el valle, red 4/7, polarización 128 mV). K = 32.93 raw/A con DMM. Cadena completa en la ISR: i_dq DC.</span></div>
    <div class="tl"><b>s20 · 14 oct</b> <span>— Primer lazo cerrado FCS‑M²PC: 66 candidatos, retardo entero, 0 overruns. PI de velocidad. λ = 3.46 mWb del log.</span></div>
    <div class="tl"><b>s21 · 16–17 oct</b> <span>— Offset de corriente en línea desde el valle del PWM: deriva ±2 LSB → +0.5 ± 0.4.</span></div>
    <div class="tl"><b>s22 · 18 oct</b> <span>— Setup final (llanta + base). Scope en vivo. <b>Bug del CORDIC</b> cazado y corregido.</span></div>
    <div class="tl"><b>s23 · 18 oct</b> <span>— Exp. 1 (telemetría en la ISR, escalones), Exp. 2 (trayectoria ±600 con inversión; sector corregido), Exp. 3a (BEMF reconstruida, LMS converge en 2 s).</span></div>
    <div class="tl"><b>s24 · 18 oct</b> <span>— <b>Exp. 2b: BEMF real en circuito abierto, THD 0.4–0.7 %.</b> Offset del encoder +2.85° medido y corregido. Regla de sector definitiva.</span></div>
  </div>
</section>

<section id="next">
  <h2><span class="n">07</span>Lo que sigue</h2>
  <div class="grid g3">
    <div class="card"><h3>Exp. 3b — ADALINE en línea</h3><p>Observador algebraico sobre medias de 1 ms, LMS a 1 kHz fuera de la ISR, evaluación W·x(θ) a 50 kHz sustituyendo la línea λω·[−sin, cos]. Réplica bit‑exacta en Python antes de flashear.</p></div>
    <div class="card"><h3>Exp. 3c — la comparativa</h3><p>Exp. 1 repetida con <code>ADALINE = 1</code>: RMS de error, rizo, sesgo, i_d, a 300 y 600 rpm; curva W(t); forma aprendida vs la forma real de la sesión 24.</p></div>
    <div class="card"><h3>Lo que ya se puede escribir</h3><p>El punto de decisión 2 del plan está cerrado con número: BEMF sinusoidal al 99.5 %. El ADALINE en este motor es demostrativo; lo que aprende es el dead‑time, el offset y la fase — y la BEMF "vista por el controlador" no es la del imán.</p></div>
  </div>
</section>

<footer>
  Generado por <code>python_scripts/make_bench_report.py</code> a partir de <code>PROGRESO_HARDWARE.md</code>, <code>bemf_2804_oc.json</code> y <code>bemf_2804_fourier.json</code>. Logs en <code>datos_banco/</code>. Repo: <code>aleonsa/tesis_maestria</code>.
</footer>
</div>

<script>
const OC = __OC__;      // coeficientes circuito abierto (por fase)
const REC = __REC__;    // reconstrucción desde el controlador (αβ)
const HARM_OC = __HOC__, HARM_RA = __HRA__;

// ── contadores ──
const io = new IntersectionObserver(es=>es.forEach(e=>{if(!e.isIntersecting)return;const el=e.target,n=+el.dataset.n,d=+el.dataset.d;let t0=null;const f=ts=>{if(!t0)t0=ts;const p=Math.min(1,(ts-t0)/1300),q=1-Math.pow(1-p,3);el.textContent=(n*q).toFixed(d);if(p<1)requestAnimationFrame(f)};requestAnimationFrame(f);io.unobserve(el)}),{threshold:.5});
document.querySelectorAll('[data-n]').forEach(e=>{e.textContent='0';io.observe(e)});
// ── timeline / barras ──
const io2=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('show');io2.unobserve(e.target)}}),{threshold:.2});
document.querySelectorAll('.tl,.bars').forEach(e=>io2.observe(e));
// barras de armónicos h2..h8
const bars=document.getElementById('bars');const mx=Math.max(...HARM_RA,...HARM_OC.map(r=>r[0]))*1.05;
for(let k=0;k<7;k++){const l=document.createElement('div');l.textContent='h'+(k+2);const w=document.createElement('div');
 w.innerHTML=`<div class="bar-row"><div class="bar" style="width:${62*HARM_OC[k][0]/mx}%"></div><em>${HARM_OC[k][0].toFixed(2)} % medido</em></div><div class="bar-row"><div class="bar r" style="width:${62*HARM_RA[k]/mx}%"></div><em>${HARM_RA[k].toFixed(2)} % "visto"</em></div>`;
 bars.appendChild(l);bars.appendChild(w)}
// ── animación ──
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');const W=cv.width,H=cv.height;
const xg=document.getElementById('xg'),rg=document.getElementById('rg'),cmp=document.getElementById('cmp');
xg.oninput=()=>document.getElementById('xv').textContent=xg.value;rg.oninput=()=>document.getElementById('rv').textContent=rg.value;
const lam=OC.lambda_Wb;
function phase(x,th,X){ // forma normalizada de la fase x con armónicos ×X
  let s=0;for(let k=1;k<=OC.H;k++){const a=OC.S_abc_cos[x][k-1]/lam,b=OC.S_abc_sin[x][k-1]/lam;const g=k===1?1:X;s+=g*(a*Math.cos(k*th)+b*Math.sin(k*th))}return s}
function rec(ax,th,X){let s=0;const c=ax?REC.S_beta_cos:REC.S_alpha_cos,b=ax?REC.S_beta_sin:REC.S_alpha_sin;for(let k=1;k<=REC.H;k++){const g=k===1?1:X;s+=g*(c[k-1]*Math.cos(k*th)+b[k-1]*Math.sin(k*th))/REC.lambda_Wb}return s}
let t=0,last=null;
function draw(ts){if(!last)last=ts;const dt=(ts-last)/1000;last=ts;const rpm=+rg.value,we=rpm/60*2*Math.PI*7;t+=dt;const X=+xg.value;
 ctx.clearRect(0,0,W,H);ctx.fillStyle='#0a0e18';ctx.fillRect(0,0,W,H);
 // panel izquierdo: ondas
 const L=20,R=400,mid=H*0.5,amp=H*0.36,span=4*Math.PI,Vpk=lam*we;
 ctx.strokeStyle='#243049';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(L,mid);ctx.lineTo(R,mid);ctx.stroke();
 const cols=['#4fd1c5','#f6ad55','#f687b3'];
 for(let x=0;x<3;x++){ctx.strokeStyle=cols[x];ctx.lineWidth=2;ctx.beginPath();for(let i=0;i<=200;i++){const th=we*t-span+span*i/200;const y=mid-amp*phase(x,th,X);const px=L+(R-L)*i/200;i?ctx.lineTo(px,y):ctx.moveTo(px,y)}ctx.stroke()}
 if(cmp.checked){ctx.setLineDash([4,4]);ctx.strokeStyle='#e8edf7';ctx.lineWidth=1.2;ctx.beginPath();for(let i=0;i<=200;i++){const th=we*t-span+span*i/200;const y=mid-amp*(-rec(0,th,X));const px=L+(R-L)*i/200;i?ctx.lineTo(px,y):ctx.moveTo(px,y)}ctx.stroke();ctx.setLineDash([])}
 ctx.fillStyle='#8a97b3';ctx.font='12px system-ui';ctx.fillText(`e_a, e_b, e_c  ·  ${rpm} rpm  ·  f_e = ${(we/2/Math.PI).toFixed(1)} Hz  ·  pico ${(Vpk).toFixed(2)} V`,L,18);
 ctx.fillText(`armónicos ×${X}${cmp.checked?'  ·  línea punteada: −e_α "vista por el controlador"':''}`,L,H-10);
 // panel derecho: vector αβ y rotor
 const cx=520,cy=mid,r=100;ctx.strokeStyle='#243049';ctx.beginPath();ctx.arc(cx,cy,r,0,2*Math.PI);ctx.stroke();
 ctx.beginPath();ctx.moveTo(cx-r-10,cy);ctx.lineTo(cx+r+10,cy);ctx.moveTo(cx,cy-r-10);ctx.lineTo(cx,cy+r+10);ctx.stroke();
 const th=we*t;const ea=phase(0,th,X),eb=phase(1,th,X),ec=phase(2,th,X);const al=(2*ea-eb-ec)/3,be=(eb-ec)/Math.sqrt(3);
 // traza
 ctx.strokeStyle='rgba(79,209,197,.55)';ctx.lineWidth=1.5;ctx.beginPath();for(let i=0;i<=360;i++){const p=i/360*2*Math.PI;const a=phase(0,p,X),b=phase(1,p,X),c=phase(2,p,X);const A=(2*a-b-c)/3,B=(b-c)/Math.sqrt(3);const px=cx+r*A,py=cy-r*B;i?ctx.lineTo(px,py):ctx.moveTo(px,py)}ctx.stroke();
 ctx.strokeStyle='#68d391';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+r*al,cy-r*be);ctx.stroke();
 // rotor (eje d = θ_e)
 ctx.strokeStyle='#f6ad55';ctx.lineWidth=2;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+r*0.8*Math.cos(th),cy-r*0.8*Math.sin(th));ctx.stroke();ctx.setLineDash([]);
 ctx.fillStyle='#8a97b3';ctx.fillText('e_αβ (verde) · eje d del rotor (naranja)',cx-r-10,cy+r+34);ctx.fillText('traza: 1 vuelta eléctrica',cx-r-10,cy+r+50);
 requestAnimationFrame(draw)}
requestAnimationFrame(draw);
</script>
</body></html>
"""

html = (HTML.replace("{oc}", imgs["oc"]).replace("{rec}", imgs["rec"]).replace("{exp1}", imgs["exp1"])
        .replace("{exp2}", imgs["exp2"]).replace("{pll}", imgs["pll"]).replace("{bemf300}", imgs["bemf300"])
        .replace("__OC__", json.dumps({k: oc[k] for k in ("H", "lambda_Wb", "S_abc_cos", "S_abc_sin")}))
        .replace("__REC__", json.dumps({k: rec[k] for k in ("H", "lambda_Wb", "S_alpha_cos", "S_alpha_sin", "S_beta_cos", "S_beta_sin")}))
        .replace("__HOC__", json.dumps(harm_oc)).replace("__HRA__", json.dumps(harm_rec_a)))
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html)
print(f"→ {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
