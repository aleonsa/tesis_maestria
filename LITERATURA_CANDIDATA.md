# Literatura candidata (búsqueda del 16-nov-2026)

Metadatos y DOI verificados contra Crossref. **No leí los artículos completos**: la relevancia sale de títulos y resúmenes. Antes de citar cada uno hay que bajar el PDF (IEEE Xplore) y confirmar que dice lo que le atribuimos.

Prioridad: **A** = hay que citarlo o un sinodal lo va a echar de menos; **B** = fortalece; **C** = opcional.

## 1. Trabajos más cercanos a la propuesta (revisar contra las contribuciones del Cap. 1)

| P | Referencia | DOI | Por qué importa |
|---|---|---|---|
| A | Guo, Zhou, Song, Zhu, Mu (2024). An Enhanced Parallel-Observer-Based Deadbeat Predictive Control Method for PMSM With Non-Sinusoidal Back-EMF. *IEEE Access* 12, 47607–47618 | 10.1109/access.2023.3347590 | Predictivo (deadbeat) + ADALINE + observador para BEMF no senoidal. Es el antecedente más parecido; hay que decir en qué diferimos (FCS-M2PC, BLDC, base de Fourier completa, LMS sobre la BEMF y no sobre el error de predicción). |
| A | Flieller, Nguyen, Wira, Sturtzer, Abdeslam, Mercklé (2014). A Self-Learning Solution for Torque Ripple Reduction for Nonsinusoidal Permanent-Magnet Motor Drives Based on Artificial Neural Networks. *IEEE TIE* 61(2), 655–666 | 10.1109/tie.2013.2257136 | ADALINE que aprende corrientes óptimas para BEMF no senoidal. Antecedente directo del estimador. |
| A | Xia, Wang, Shi (2013). Implementation of Finite-State Model Predictive Control for Commutation Torque Ripple Minimization of Permanent-Magnet Brushless DC Motor. *IEEE TIE* 60(3), 896–905 | 10.1109/tie.2012.2189536 | Referencia canónica de FCS-MPC en BLDC para rizo de par. |
| A | Li, Fan, Kong, Liu, Zhang (2024). Torque Ripple Suppression of BLDCM With Optimal Duty Cycle and Switch State by FCS-MPC. *IEEE OJPEL* 5, 381–391 | 10.1109/ojpel.2024.3368221 | FCS-MPC con ciclo de trabajo óptimo en BLDC; misma revista que coronado2025. |
| B | Inan (2024). An Improved Model Predictive Current Control of BLDC Motor With a Novel Adaptive Extended Kalman Filter–Based Back EMF Estimator… *Int. J. Circuit Theory Appl.* 53(2), 1135–1150 | 10.1002/cta.4407 | MPCC en BLDC con estimador de BEMF (EKF en vez de ADALINE). |
| B | Lahooti Eshkevari, Torkaman (2020). Simplified model predictive current control of non-sinusoidal low power brushless DC machines. *Turk. J. Elec. Eng. & Comp. Sci.* 28(4), 2288–2302 | 10.3906/elk-2001-141 | MPCC para BLDC de baja potencia con BEMF no senoidal. |
| B | Zhang, Wang, Xu, Zhao (2016). ADALINE-Network-Based PLL for Position Sensorless Interior Permanent Magnet Synchronous Motor Drives. *IEEE TPEL* 31(2), 1450–1460 | 10.1109/tpel.2015.2424256 | ADALINE + observador de BEMF para armónicos; apoya la elección de ADALINE (Cap. 4). |
| B | Qamar, Hatziadoniu (2019). Cancelation of Torque Ripples in PMSM via a Novel Minimal Parameter Harmonic Flux Estimator. *IEEE TPEL* 34(3), 2553–2562 | 10.1109/tpel.2018.2842032 | Estima flujo armónico a partir de BEMF armónica estimada. |
| B | Mora, Orellana, Juliet, Cárdenas (2016). Model Predictive Torque Control for Torque Ripple Compensation in Variable-Speed PMSMs. *IEEE TIE* 63(7), 4584–4592 | 10.1109/tie.2016.2536586 | MPC que compensa rizo de par con modelo de armónicos. |

## 2. Sensibilidad del predictivo al modelo (antecedentes del criterio κ, Cap. 6)

| P | Referencia | DOI | Por qué importa |
|---|---|---|---|
| A | Young, Perez, Rodriguez (2016). Analysis of Finite-Control-Set Model Predictive Current Control With Model Parameter Mismatch in a Three-Phase Inverter. *IEEE TIE* 63(5), 3100–3107 | 10.1109/tie.2016.2515072 | Análisis del error de predicción por desajuste de parámetros. Hay que posicionar κ frente a este análisis. |
| A | Siami, Khaburi, Abbaszadeh, Rodriguez (2016). Robustness Improvement of Predictive Current Control Using Prediction Error Correction for PMSMs. *IEEE TIE* 63(6), 3458–3466 | 10.1109/tie.2016.2521734 | Corrección del error de predicción; alternativa a estimar la BEMF. |
| B | Karamanakos, Geyer (2020). Guidelines for the Design of Finite Control Set Model Predictive Controllers. *IEEE TPEL* 35(7), 7434–7450 | 10.1109/tpel.2019.2954357 | Guía de diseño FCS-MPC; rizo, frecuencia de muestreo. |
| B | Zhang, Jin, Huang (2021). Model-Free Predictive Current Control of PMSM Drives Based on Extended State Observer Using Ultralocal Model. *IEEE TIE* 68(2), 993–1003 | 10.1109/tie.2020.2970660 | Línea "sin modelo": contraste con aprender la BEMF. |
| C | Lin, Liu, Yu, Fu, Hsiao (2014). Model-Free Predictive Current Control for IPMSM Drives Based on Current Difference Detection Technique. *IEEE TIE* 61(2), 667–681 | 10.1109/tie.2013.2253065 | Ídem. |

## 3. Motores de baja inductancia y piso de rizo (explicación del banco, Cap. 6)

| P | Referencia | DOI | Por qué importa |
|---|---|---|---|
| A | Dai, Wang, Sun, Chong (2022). Multiple Current Harmonics Suppression for Low-Inductance PMSM Drives With Deadbeat Predictive Current Control. *IEEE TIE* 69(10), 9817–9826 | 10.1109/tie.2022.3144577 | Baja inductancia + no linealidad del inversor + BEMF no senoidal con predictivo. Es el mismo régimen que el 2804. |
| A | Fang, Zhou, Liu (2012). Instantaneous Torque Control of Small Inductance Brushless DC Motor. *IEEE TPEL* 27(12), 4952–4964 | 10.1109/tpel.2012.2193420 | BLDC de inductancia pequeña: el rizo de conmutación domina. |
| B | Fang, Zhou, Liu (2013). Precise Accelerated Torque Control for Small Inductance Brushless DC Motor. *IEEE TPEL* 28(3), 1400–1412 | 10.1109/tpel.2012.2210251 | Complemento del anterior. |

## 4. Inversor: tiempos muertos y no linealidad (señal de aprendizaje contaminada, Cap. 6 y 7)

| P | Referencia | DOI |
|---|---|---|
| A | Munoz, Lipo (1999). On-line dead-time compensation technique for open-loop PWM-VSI drives. *IEEE TPEL* 14(4), 683–689 | 10.1109/63.774205 |
| B | Hwang, Kim (2010). Dead Time Compensation Method for Voltage-Fed PWM Inverter. *IEEE TEC* 25(1), 1–10 | 10.1109/tec.2009.2031811 |
| B | Kim (2013). Improved estimation scheme for disturbance due to dead time and inverter nonlinearity in inverter-fed PMSM drive. *Electronics Letters* 49(15), 941–943 | 10.1049/el.2013.1704 |
| B | Chen, Yang, Guo, Li (2016). Disturbance-Observer-Based Control and Related Methods—An Overview. *IEEE TIE* 63(2), 1083–1095 | 10.1109/tie.2015.2478397 |

## 5. Referencia de corriente óptima con BEMF no senoidal (eq:iref-optima, Cap. 3)

| P | Referencia | DOI |
|---|---|---|
| A | Hung, Ding (1993). Design of currents to reduce torque ripple in brushless permanent magnet motors. *IEE Proc. B* 140(4), 260–266 | 10.1049/ip-b.1993.0032 |
| A | Park, Park, Lee, Harashima (2000). A new approach for minimum-torque-ripple maximum-efficiency control of BLDC motor. *IEEE TIE* 47(1), 109–114 | 10.1109/41.824132 |
| B | Chapman, Sudhoff, Whitcomb (1999). Optimal current control strategies for surface-mounted permanent-magnet synchronous machine drives. *IEEE TEC* 14(4), 1043–1050 | 10.1109/60.815026 |
| B | Lu, Zhang, Qu (2008). A New Torque Control Method for Torque Ripple Minimization of BLDC Motors With Un-Ideal Back EMF. *IEEE TPEL* 23(2), 950–958 | 10.1109/tpel.2007.915667 |
| B | Petrović, Ortega, Stanković, Tadmor (2000). Design and implementation of an adaptive controller for torque ripple minimization in PM synchronous motors. *IEEE TPEL* 15(5), 871–880 | 10.1109/63.867676 |
| B | Holtz, Springob (1996). Identification and compensation of torque ripple in high-precision permanent magnet motor drives. *IEEE TIE* 43(2), 309–320 | 10.1109/41.491355 |
| C | Mattavelli, Tubiana, Zigliotto (2005). Torque-Ripple Reduction in PM Synchronous Motor Drives Using Repetitive Current Control. *IEEE TPEL* 20(6), 1423–1431 | 10.1109/tpel.2005.857559 (es control repetitivo: fuera de alcance salvo en estado del arte) |

## 6. Control predictivo: fundamentos (Cap. 2 y 4)

| P | Referencia | DOI |
|---|---|---|
| A | Rodriguez, Pontt, Silva, Correa, Lezana, Cortés, Ammann (2007). Predictive Current Control of a Voltage Source Inverter. *IEEE TIE* 54(1), 495–503 | 10.1109/tie.2006.888802 |
| A | Vazquez, Rodriguez, Rivera, Franquelo, Norambuena (2017). Model Predictive Control for Power Converters and Drives: Advances and Trends. *IEEE TIE* 64(2), 935–947 | 10.1109/tie.2016.2625238 |
| A | Tarisciotti, Zanchetta, Watson, Bifaretti, Clare (2014). Modulated Model Predictive Control for a Seven-Level Cascaded H-Bridge Back-to-Back Converter. *IEEE TIE* 61(10), 5375–5383 | 10.1109/tie.2014.2300056 |
| B | Zhang, Yang (2015). Generalized Two-Vector-Based Model-Predictive Torque Control of Induction Motor Drives. *IEEE TPEL* 30(7), 3818–3829 | 10.1109/tpel.2014.2349508 |
| B | Morel, Lin-Shi, Rétif, Allard, Buttay (2009). A Comparative Study of Predictive Current Control Schemes for a PMSM Drive. *IEEE TIE* 56(7), 2715–2728 | 10.1109/tie.2009.2018429 |
| B | Bolognani, Bolognani, Peretti, Zigliotto (2009). Design and Implementation of Model Predictive Control for Electrical Motor Drives. *IEEE TIE* 56(6), 1925–1936 | 10.1109/tie.2008.2007547 |
| C | Geyer (2016). *Model Predictive Control of High Power Converters and Industrial Drives*. Wiley | 10.1002/9781119010883 |

## 7. Modelo del motor, LMS y devanado

| P | Referencia | DOI | Dónde |
|---|---|---|---|
| A | Pillay, Krishnan (1989). Modeling, simulation, and analysis of permanent-magnet motor drives. II. The brushless DC motor drive. *IEEE TIA* 25(2), 274–279 | 10.1109/28.25542 | Cap. 2, modelo abc |
| A | Widrow, McCool, Larimore, Johnson (1976). Stationary and nonstationary learning characteristics of the LMS adaptive filter. *Proc. IEEE* 64(8), 1151–1162 | 10.1109/proc.1976.10286 | Cap. 4, desajuste y convergencia |
| A | El-Refaie (2010). Fractional-Slot Concentrated-Windings Synchronous Permanent Magnet Machines: Opportunities and Challenges. *IEEE TIE* 57(1), 107–121 | 10.1109/tie.2009.2030211 | Cap. 6, factores de devanado 12 ranuras/14 polos |
| B | Ishak, Zhu, Howe (2006). Comparison of PM brushless motors, having either all teeth or alternate teeth wound. *IEEE TEC* 21(1), 95–103 | 10.1109/tec.2005.853765 | Ídem; estudia justo 12 ranuras con 10 y 14 polos según el resumen: confirmar |
| B | Zhu, Howe (2000). Influence of design parameters on cogging torque in permanent magnet machines. *IEEE TEC* 15(4), 407–412 | 10.1109/60.900501 | Cap. 2, par de cogging |
| C | Xia (2012). *Permanent Magnet Brushless DC Motor Drives and Controls*. Wiley | 10.1002/9781118188347 | Cap. 2 |

## Pendiente de Libert y Soulard (2004)

El artículo que queríamos ("Investigation on pole-slot combinations for permanent-magnet machines with concentrated windings", ICEM 2004) **no está en Crossref** y no pude verificarlo. Lo que sí aparece de esos autores es de 2006 y trata de manufactura de estatores (10.1049/cp:20060194), que no sirve para el argumento. Recomiendo sustituirlo por El-Refaie (2010) o Ishak et al. (2006), ambos verificables, después de confirmar en el PDF que dan los factores de devanado de 12 ranuras/14 polos.

## Las 7 entradas "sin verificar" del .bib

Todas coinciden con Crossref en autores, revista, volumen, número y año, con dos diferencias:

- `tarisciotti2015m2pc`: Crossref lista seis autores (Tarisciotti, Zanchetta, Watson, Clare, **Degano**, Bifaretti). El .bib trae cinco y omite a Degano.
- `li2019highstability` (no era de las 7): el .bib dice páginas 4895–4907; Crossref dice 4895–4903.

DOI de las siete: jahns 10.1109/41.491356 · carlson 10.1109/28.137450 · liu 10.1109/tia.2005.844853 · qian 10.1109/tpel.2003.820537 · rodriguez2013 10.1109/tii.2012.2221469 · cortes 10.1109/tie.2011.2157284 · tarisciotti2015 10.1109/tia.2014.2339397.
