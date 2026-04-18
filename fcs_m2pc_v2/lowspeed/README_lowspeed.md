# Fase 3 — Control FCS-M²PC a Baja Velocidad con ADALINE de Tiempo Finito

**Estado**: Propuesta de investigación — pendiente de implementación  
**Fecha**: Abril 2026  
**Motor**: Anaheim BLY-344S-240V-3000 (P=8, Ke=0.404 V·s/rad, SPMSM)

---

## 1. El Problema

El sistema actual (Fases 1 y 2) funciona correctamente a velocidad nominal (ω_ref = 80 rad/s),
pero tiene un límite fundamental a baja velocidad:

```
e_αβ = Ke · ω · shape(θ_e)
```

Cuando ω → 0, la BEMF se vuelve indistinguible del ruido de corriente (σ ≈ 50 mA, ADC 12-bit).
El observador de BEMF deja de funcionar, θ_e no es estimable, y el FCS-M²PC pierde referencia.

**Límite práctico estimado**: ω < 15–20 rad/s (basado en SNR del ADC y Ke del motor).

A baja velocidad, los efectos que antes eran despreciables dominan la dinámica:
- **Cogging torque**: T_cog(θ) — periódico en θ, independiente de ω
- **Reluctancia variable**: similar perfil a cogging
- **Dead-time del inversor**: introduce perturbaciones periódicas en θ_e

**Observación clave**: estas perturbaciones son función de θ, *no de ω*. Pueden aprenderse
incluso cuando la BEMF es demasiado pequeña para observarse directamente.

---

## 2. Contexto: Estado del Arte Relevante

### 2.1 Li & Zhou (IEEE Trans. Power Electronics, 2019)
Sensorless estable a baja velocidad via detección de cruces por cero (ZCP) de BEMF.
Usan la función G = E_ca / E_bc (cociente normalizado) que cancela ruido común.
Compensan el delay del LPF con una frecuencia de corte adaptativa ∝ ω.

**Límite**: sigue necesitando que |e| sea detectable. No llega a ω = 0.  
**Aplicable aquí**: la idea del cociente normalizado puede robustificar nuestro observador
en la región de transición (ω ≈ 15–30 rad/s).

### 2.2 Zhao et al. (ICEMS, 2025)
ADALINE para supresión de ripple de par en PMSM a velocidad variable. Su aportación:
ley de adaptación de **tiempo finito** que garantiza convergencia en T* finito (vs asintótica del LMS).

Ley propuesta:
```
Ẇ = -k_a · (e · φ + σ · sign(W) · |W|^(2β-1))
```
donde β ∈ (0.5, 1) controla la velocidad de convergencia, y el término `σ·|W|^(2β-1)`
actúa como regularización tipo leaky: evita drift de pesos cuando la excitación es baja
(exactamente el problema que surge a ω pequeño).

**Relevancia directa**: podemos adoptar esta ley en `lms_update.m`.

### 2.3 Ahn (ICCA, 2003)
Control repetitivo adaptativo discreto para motor lineal BLDC. Actualiza online los coeficientes
del controlador feedforward usando identificación de sistema por mínimos cuadrados.
Converge más rápido que RC fijo incluso con 20% de error de modelado.

**Relevancia**: aplicable a la sub-fase de Control Repetitivo (más adelante). No para baja velocidad directamente.

---

## 3. Propuesta: ADALINE-FT para Perturbaciones Periódicas

### 3.1 Hipótesis central

> En lugar de estimar la BEMF (no observable a ω bajo), el ADALINE aprende el **torque
> de perturbación periódico total** directamente desde el error de corriente. Esto incluye
> cogging, reluctancia variable, y dead-time, todos los cuales son función de θ_e únicamente.

El aprendizaje es posible incluso a ω≈0 porque el error de corriente `ε = i_ref - i_medida`
*sí es observable*, y las perturbaciones periódicas generan un patrón en ε(θ).

### 3.2 Arquitectura propuesta

```
                     ┌──────────────┐
ω > ω_th ─────────►  │  Observador  │──► θ_e, e_obs
                     │  BEMF (Fase2)│
                     └──────────────┘
                              │ α(ω)=1
                              ▼
i_ref ──►[ FCS-M²PC ]──► u_ab ──►[ Planta ]──► i_ab, ω
              ▲                        │
              │ T_ff(θ)                │ ε = i_ref - i_ab
              │                        ▼
              └────[ ADALINE-FT ]◄────[ φ(θ) ]
                     W ∈ ℝ^21
                     
ω < ω_th: θ_e del estimador de posición alternativo (a definir)
          T_ff(θ) como feedforward al controlador
```

### 3.3 Modelo del ADALINE de perturbaciones

Base de Fourier (H=10, misma que BEMF ADALINE):
```
φ(θ) = [sin(θ), cos(θ), sin(2θ), cos(2θ), ..., sin(10θ), cos(10θ), 1]^T   ∈ ℝ^21
```

Salida (torque de perturbación estimado):
```
T_dist(θ; W) = W^T · φ(θ)
```

Error de adaptación (basado en error de par inferido desde corriente):
```
ε_T(k) = T_ref(k) - T_e_estimado(k)
```

### 3.4 Ley de adaptación LMS-FT (propuesta)

```matlab
% Actual (LMS estándar):
W = W + mu * eps * phi

% Propuesto (tiempo finito, inspirado en Zhao 2025):
beta  = 0.7;          % exponente: 0.5 < beta < 1
sigma = 1e-4;         % coeficiente de regularización
W = W + mu * eps * phi ...
      - mu * sigma * sign(W) .* abs(W).^(2*beta - 1);
```

El término de regularización:
- Amortigua pesos que no son excitados (importante a baja velocidad)
- Garantiza convergencia en tiempo finito T* (Lyapunov, Zhao 2025, Theorem 1)
- Con β=0.7 y σ pequeño, el comportamiento a velocidad nominal es casi idéntico al LMS estándar

### 3.5 Transición suave entre modos

```matlab
alpha = min(max(w_m / w_threshold, 0), 1);   % 0=baja vel, 1=alta vel

e_control = alpha * e_observer + (1-alpha) * (Ke * shape_adaline(theta_e));
T_ff      = (1 - alpha) * adaline_ft_output(theta_e, W);
```

---

## 4. Hipótesis de Mejora (a validar en simulación)

| Condición | Métrica | Actual (Fase 2) | Hipótesis Fase 3 | Base |
|-----------|---------|-----------------|------------------|------|
| ω = 80 rad/s, estado estable | Ripple Te [%] | 3.65% | ≤ 3.65% (sin regresión) | LMS estándar ya convergió |
| ω = 80 rad/s | NMSE α | 0.0028 | ≤ 0.0028 | sin cambio en alta vel |
| ω = 15 rad/s | Ripple Te [%] | ~15–30% (obs falla) | < 10% | ADALINE-FT aprende T_dist(θ) |
| ω = 15 rad/s | θ_e error [°] | > 30° (obs diverge) | < 15° | feedforward reduce efecto del error |
| Arranque 0→80 rad/s | Tiempo hasta ripple < 5% | no medido | < 200 ms | pre-entrenamiento offline |
| Perturbación de carga ΔT=0.5 Nm | Desviación ω [rad/s] | ~0.4 | ≤ 0.4 | igual regulación de velocidad |

**Fundamentos de las hipótesis**:
- **Sin regresión a alta vel**: el término de regularización es despreciable cuando ||W|| grande (pesos bien adaptados). La ley converge al mismo mínimo que LMS.
- **Mejora a baja vel**: el cogging del Anaheim es pequeño (<1% de Iq nominal, medido en datos dSPACE), pero el dead-time y la asimetría del inversor producen perturbaciones 6ω_e y 12ω_e que sí son aprendibles.
- **θ_e a baja vel**: requiere un estimador alternativo (no BEMF). Opciones: (a) integración de modelo cinemático puro `θ += ω·Ts·(P/2)` con corrección periódica, (b) uso de corriente de fase para estimar posición via reluctancia (solo si el motor tiene algo de saliencia).

---

## 5. Plan de Implementación

### Paso 1: Extender la simulación para probar a baja velocidad
- Modificar `motor_params.m`: agregar `p.w_ref_low = 15` y `p.w_threshold = 20`
- Modificar `main_comparison.m`: agregar rampa de velocidad 0→80 rad/s o test a ω_ref=15

### Paso 2: Medir el baseline actual a baja velocidad
- Ejecutar simulación con ω_ref=15 rad/s para los 6 métodos actuales
- Documentar dónde falla el observador (θ_e error, ripple)

### Paso 3: Implementar `lms_update_ft.m` (variante FT del LMS)
```
fcs_m2pc_v2/adaptation/lms_update_ft.m
```
Parámetros nuevos en `motor_params.m`:
```matlab
p.beta_ft    = 0.7;     % exponente tiempo finito
p.sigma_ft   = 1e-4;    % coeficiente regularización
```

### Paso 4: Implementar modo dual en el loop de simulación
- Archivo nuevo: `fcs_m2pc_v2/lowspeed/lowspeed_controller.m`
- Integrar en `simulate_all_methods.m` con flag `USE_LOWSPEED_MODE`

### Paso 5: Validar y comparar
- Métrica principal: ripple Te y error θ_e para ω ∈ {15, 30, 50, 80} rad/s
- Comparar LMS-estándar vs LMS-FT en velocidad de convergencia (inicio en frío vs pre-entrenado)

### Paso 6: Analizar datos dSPACE para validación parcial
- El CSV `rec1_002.csv` (ω≈35 rad/s, transitorio) es lo más cercano a baja velocidad disponible
- Extraer perfil de perturbaciones desde error de corriente medido

---

## 6. Preguntas Abiertas

1. **¿Cómo estimar θ_e cuando ω < ω_threshold y el observador BEMF falla?**
   - Opción A: integración cinemática pura (deriva en el tiempo)
   - Opción B: usar corriente de fase para inferir posición (requiere saliencia)
   - Opción C: asumir que el ADALINE-FT es suficientemente robusto al error de θ_e

2. **¿El cogging del Anaheim es suficientemente grande para beneficiarse del feedforward?**
   - Datos dSPACE sugieren <1% de Iq nominal → el beneficio puede ser marginal
   - El dead-time del inversor DS1104 puede ser más significativo

3. **¿Hay suficiente persistencia de excitación (PE) a baja velocidad para que el ADALINE converja?**
   - A ω muy bajo, la base φ(θ) varía lentamente → PE puede ser insuficiente
   - El término de regularización mitiga esto, pero no lo elimina completamente

4. **¿Pre-entrenamiento offline del ADALINE-FT vs arranque en frío?**
   - Pre-entrenar con datos sintéticos (modelo de cogging idealizado) puede acelerar convergencia

---

## 7. Referencias Clave

- **Zhao et al., "Finite-Time Adaptive Linear Neuron Control for Torque Ripple Suppression of PMSMs"**, ICEMS 2025 — ley de adaptación FT, Theorem 1 (Lyapunov)
- **Li & Zhou, "High-Stability Position-Sensorless Control Method for BLDCs at Low Speed"**, IEEE Trans. Power Electronics, 2019 — ZCP + LPF adaptativo
- **Coronado, "FCS-M²PC for BLDC"**, 2025 — framework de control base
- `papers/` en este repositorio contiene los PDFs
