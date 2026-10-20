# datos_banco — logs del banco propio (B-G431B-ESC1 + 2804 12N14P)

Logs serie de `07_m2pc` capturados en este banco. No confundir con `datos_axel/`,
que son grabaciones de otra persona con otro motor (base de `generate_real_bemf_lut.py`).

Los `.log` no se versionan (pesan hasta 25 MB); solo este README. Se nombran
`m2pc_<descripción>_<fecha>.log`; `scope_m2pc.py` guarda por defecto como
`m2pc_live_<fecha>.log`.

| Archivo | Sesión | Condición |
|---|---|---|
| `m2pc_300rpm_2026-10-14.log` | 20 | primer lazo cerrado, 300 rpm, offset solo al arranque |
| `m2pc_s21_frio_real.log` | 21 | 300 rpm, motor frío, offset al arranque (deriva) |
| `m2pc_s21_caliente_real.log` | 21 | 300 rpm, motor caliente, offset al arranque |
| `m2pc_s21_valle_rep.log` | 21 | offset del valle en modo solo reporte (validación cruzada) |
| `m2pc_s21_valle_on.log` | 21 | 300 rpm, offset en línea activo |
| `m2pc_live_20261018_111051_678526.log` | 22 | 300 rpm, **llanta + base**, CORDIC aún roto (sin ≡ cos) |
| `m2pc_s22_cordic_fix_a.log` | 22 | 30 s, primer arranque con el CORDIC corregido |
| `m2pc_s22_cordic_fix.log` | 22 | 300 rpm, llanta, CORDIC corregido (**referencia para BEMF**) |
| `m2pc_s23_exp1_300rpm_a.log` | 23 | Exp. 1, PI activo y escalón de 200 ms: el PI lo cancela (no usar) |
| `m2pc_s23_exp1_300rpm_b.log` | 23 | **Exp. 1**, 300 rpm, escalones ±100 mA/20 ms con PI congelado, líneas `E` y captura `C` |
| `m2pc_s23_exp1_600rpm.log` | 23 | **Exp. 1**, 600 rpm, ídem |
| `m2pc_s23_exp2_100rpm_50mHz.log` | 23 | **Exp. 2 / dataset**, ω* = ±100 rpm·sin(2π·0.05·t), sector por tensión deadbeat |
| `m2pc_s23_exp2_300rpm_100mHz.log` | 23 | **Exp. 2 / dataset**, ±300 rpm, 0.1 Hz |
| `m2pc_s23_exp2_600rpm_200mHz.log` | 23 | **Exp. 2 / dataset**, ±600 rpm, 0.2 Hz |
| `m2pc_s24_off2563_600rpm.log`, `m2pc_s24_sector_{bemf,deadbeat,uss}_600rpm.log`, `m2pc_s24_sector_uss_traj300.log` | 24 | offset 2563; A/B de la regla de sector a 600 rpm; `u_ss` con trayectoria ±300 |
| `bemf_oc_s24_run1.log` | 24 | **Exp. 2b**, app `08_bemf_oc`: BEMF en circuito abierto, llanta a mano, ±730 rpm, 2.5 kHz (`B seq th w va vb vc`) |
| `m2pc_s26_adaline_off_traj300.log` | 26 | **Exp. 3b/3c baseline**, `ADALINE=0`, trayectoria ±300 rpm (0.1 Hz), 45 s (`ε_q ≈ 45 mA`) |
| `m2pc_s26_adaline_on_traj300_a.log`, `m2pc_s26_adaline_on_traj300.log` | 26 | **Exp. 3b**, `ADALINE=1`, `MU_Q16=131` (μ = 2e-3): arranque t=0..25 s (`_a`) y continuación t=106..138 s hasta `[TRIP]` (espiral inestable en lazo cerrado) |
| `m2pc_s26_A_on_mu0_traj300.log` | 26 | control: ON, μ=0, trayectoria ±300 (= OFF: implementación inocua) |
| `m2pc_s26_B_off_300rpm{,_b}.log`, `m2pc_s26_B_off_300rpm_ldec100.log` | 26 | OFF 300 rpm constante con L a 1 kHz (55 mA) y con L decimado 1/100 (**46.7 mA**, baseline válido) |
| `m2pc_s26_B_on_300rpm_mu7_gate500.log`, `m2pc_s26_B_on_600rpm_mu7_gate500.log` | 26 | ON H=7: **inválidos** (ISR ≈19 µs, foreground al 5 %, PI a 100 Hz); evidencia del problema de presupuesto |
| `m2pc_s26_B_on_300rpm_H3_mu7.log` | 26 | **ON válido**: H=3, μ=1e-4, puerta 500 mV, 150 s: 44.8/39.3 mA, W deriva lento |
| `m2pc_s26_E_wrong_mu0_300rpm.log`, `m2pc_s26_E_wrong_mu7_300rpm.log` | 26 | **Exp. E**: modelo inicial λ·0.8 +10°; sin aprender 44.1 mA (insensibilidad); con μ=1e-4 recupera λ en ~90 s y deriva |
| `m2pc_s26_adaline_mu1e4.log` | 26 | **Exp. 3b**, `ADALINE=1`, `MU_Q16=7` (μ ≈ 1e-4), 100 s estables pero `ε_q ≈ 102 mA` y `h2_α` 0 → 17.6 % (`W` a 1 Hz) |

Los archivos pueden empezar con la cola del firmware anterior (el `cat` arranca antes del
flash): **usar solo lo que sigue al último `[boot]`**.

Los logs de las sesiones 20–21 y el primero de la 22 se tomaron con el CORDIC entregando
`sin ≡ cos` (N1.27): sirven para identificar λ y la forma de la BEMF (offline, en αβ) pero
**no** como datos de desempeño del lazo cerrado.

Análisis: `python3 python_scripts/analyze_m2pc_log.py datos_banco/<log> --win 30` (un log a
velocidad constante) o `python3 python_scripts/bemf_reconstruct.py datos_banco/m2pc_s23_exp2_*.log
datos_banco/m2pc_s22_cordic_fix.log` (reconstrucción conjunta S + D + dead‑time, LMS).

Líneas desde la sesión 23 (`07_m2pc`): `L` a 1 kHz (medias de i_αβ, u_αβ, θ_e, ω̂, i_q*),
`E` a 100 Hz (`t id iq ref eq_rms ed_rms iq_min iq_max`, Q8 LSB, Park en la ISR),
`C k iq ref` captura de 1024 muestras a 50 kHz alrededor del primer escalón.
