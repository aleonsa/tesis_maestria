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

Los logs de las sesiones 20–21 y el primero de la 22 se tomaron con el CORDIC entregando
`sin ≡ cos` (N1.27): sirven para identificar λ y la forma de la BEMF (offline, en αβ) pero
**no** como datos de desempeño del lazo cerrado.

Análisis: `python3 python_scripts/analyze_m2pc_log.py datos_banco/<log> --win 30`.

Líneas desde la sesión 23 (`07_m2pc`): `L` a 1 kHz (medias de i_αβ, u_αβ, θ_e, ω̂, i_q*),
`E` a 100 Hz (`t id iq ref eq_rms ed_rms iq_min iq_max`, Q8 LSB, Park en la ISR),
`C k iq ref` captura de 1024 muestras a 50 kHz alrededor del primer escalón.
