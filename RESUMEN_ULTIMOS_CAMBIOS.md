# Resumen de los últimos cambios (al 2026-10-11)

Cubre el trabajo posterior al commit `dee0967` (2026-08-23) que estaba pendiente de commit:
redacción del Cap. 1 de la tesis (2026-08-17) y la sesión 15 de firmware (2026-08-24).

---

## 1. Tesis — Cap. 1 Introducción escrito (Bloque B cerrado)

Archivo: `tesis_documento/Documentos/Chapters/Capitulo1/Cap1.tex` (reemplaza los `\lipsum`).

- **1.1 Motivación**: ubicuidad del BLDC → rizo de par → la BEMF trapezoidal ideal no se cumple
  → los controladores modernos cargan una hipótesis de forma de onda en su modelo interno
  → caracterizar offline resuelve un motor, no el problema → ¿puede el controlador aprender
  la forma de onda mientras opera?
- **1.2 Estado del arte**, cuatro subsecciones:
  - Reducción de rizo (diseño de máquina vs. control; referencia óptima de Le-Huy; rizo de
    conmutación; DTC; control repetitivo e ILC).
  - FCS-MPC: sus dos debilidades (rizo intra-periodo, sensibilidad al modelo). El M2PC se
    atribuye a Tarisciotti et al. 2015; Coronado 2025 lo adapta al BLDC.
  - Estimación de BEMF y aprendizaje: observadores orientados a posición, NN profundas y sus
    tres inconvenientes, ADALINE; zhao2025 como lo más cercano (compensador aditivo, no toca
    el predictor).
  - La brecha que ocupa la tesis.
- **1.3 Contribuciones**: 5 párrafos alineados 1:1 con los objetivos específicos. Números:
  69 % / 61 % de reducción de rizo vs. TRAP, 40 parámetros vs. ~25k.
- **1.4 Estructura de la tesis**: un párrafo por capítulo con `\ref`.

Otros cambios:
- `Cap6.tex` ahora lleva `\label{cap:conclusiones}` (lo referencia 1.4).
- `graphicspath` de Cap. 1 corregido.
- PDF regenerado: **56 páginas**, 0 referencias/citas indefinidas, 0 overfull boxes.
- `TESIS_ESCRITURA.md` actualizado.

**⚠ Pendiente de verificar:** 7 entradas nuevas en `Bibliografia.bib` con metadatos escritos de
memoria (sin PDF en `papers/`): `jahns1996pulsating`, `carlson1992commutation`, `liu2005dtc`,
`qian2004ilc`, `rodriguez2013stateofart`, `cortes2012delay`, `tarisciotti2015m2pc`.

---

## 2. Firmware — Sesión 15: errata de silicio en el I²C por IRQ

Archivos: `firmware/lib/i2c.c`, `firmware/lib/i2c.h`, `firmware/apps/03_as5600/src/main.c`.
Documentación: `FIELD_NOTES.md` N1.20, `PROGRESO_HARDWARE.md` sesión 15.

**Causa raíz.** ES0431 Rev. 9 §2.11.5 (`I2C2-190208`, "Transmission stalled after first byte"):
el workaround exige precargar el primer byte en `TXDR` **antes** de generar `START`. El driver
hacía lo contrario (`START` → esperar `TXIS` → escribir registro). Las release notes de
STM32CubeG4 confirman que el HAL ya aplica esa precarga.

**Cambios en el driver:**
- `i2c1_read_regs()` e `i2c1_read_regs_start()` precargan el registro en `TXDR` antes de `CR2.START`.
- Camino IRQ nominal: 4 eventos para 2 bytes (`TC`, `RXNE`, `RXNE`, `STOPF`). Un `TXIS`
  inesperado aborta, nunca reenvía el registro.
- Fuentes de IRQ habilitadas por transferencia y deshabilitadas antes de publicar estado terminal.
- `i2c1_nb_abort()` es asíncrono: pide STOP y retorna; `STOPF` autoriza el cierre. Sin polling
  ni reset de `PE` dentro de la ISR.
- Flag `closing` + secciones críticas cortas contra la ISR de control (prio 1 > I²C prio 3).
- Flags de bus capturados antes de limpiarlos; `seq/errors` y `DONE/ERR` se publican juntos al final.

**Cambios en la app `03_as5600`:**
- Timeout de `nb_pump()` conserva su timestamp entre llamadas; si `STOPF` no llega, recuperación
  con `i2c1_reset_peripheral()` desde foreground.
- Prueba de arranque reducida a una sola lectura IRQ aislada, que imprime la traza de eventos.
- `nb_report_failure()` compara contra el contador previo en lugar de contra cero.

**Verificación hecha:** los tres targets (`blink`, `pwm_adc`, `as5600`) compilan sin warnings
(GCC ARM 15.2.1). `objdump` confirma el store a `TXDR` antes de `CR2.START`.
`as5600.elf`: 15 644 B.

**Pendiente en banco:**
1. Flashear `03_as5600`.
2. Confirmar `[shot] 0: OK` y secuencia sin `TXIS`.
3. Medir `[perf] N lecturas/s` con `errores=0`, `abortadas=0`.

---

## 3. Qué sigue

- Validar en banco el I²C por IRQ (arriba) y luego extrapolación de θ dentro de la ISR.
- Offset θ_e ↔ fase A (alineación con eje d).
- Verificar las 7 citas nuevas del bib.
- Cap. 6 Conclusiones y capítulo de hardware.
