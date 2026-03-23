"""
Fase 0: ADALINE para Aprendizaje de BEMF
=========================================
Reemplaza la NN profunda (BEMFNet: 4 capas, ~25k params) por un ADALINE
(red lineal en base de Fourier, ~40 params) que:
  1. Tiene solución óptima en forma cerrada (pseudoinversa)
  2. Los pesos SON los coeficientes de Fourier (interpretables)
  3. Es trivialmente actualizable online con LMS (Fase 1)

Genera: bemf_adaline_lut.mat compatible con FSC_M2PC_ClosedLoop.m
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.linalg import lstsq
import time

# ============================================================================
# 1. GENERACIÓN DE LA BEMF "REAL" (idéntica al script original)
# ============================================================================

def generate_complex_bemf(theta, harmonics=True):
    """
    BEMF real con armónicos + distorsión trapezoidal.
    Idéntica a la versión anterior para comparación justa.
    """
    theta_a = theta
    theta_b = theta - 2*np.pi/3
    theta_c = theta - 4*np.pi/3

    def phase_shape(th):
        y = 0.70 * np.sin(th)
        if harmonics:
            y += 0.20 * np.sin(3*th + np.pi/6)
            y += 0.07 * np.sin(5*th - np.pi/4)
            y += 0.03 * np.sin(7*th + np.pi/3)
            y += 0.05 * np.sign(np.sin(th)) * (1 - np.exp(-5*np.abs(np.sin(th))))
        return y

    phase_a = phase_shape(theta_a)
    phase_b = phase_shape(theta_b)
    phase_c = phase_shape(theta_c)
    return np.array([phase_a, phase_b, phase_c])


def clarke_transform(abc):
    """Clarke: ABC -> Alpha-Beta"""
    T_clarke = (2/3) * np.array([
        [1, -0.5, -0.5],
        [0, np.sqrt(3)/2, -np.sqrt(3)/2]
    ])
    return T_clarke @ abc


# ============================================================================
# 2. CLASE ADALINE
# ============================================================================

class ADALINE_BEMF:
    """
    ADALINE para aprendizaje de BEMF en base de Fourier.
    
    Modelo:
        ê_α(θ) = Σ_{n=1}^{H} [ w_n^c·cos(nθ) + w_n^s·sin(nθ) ]
        ê_β(θ) = Σ_{n=1}^{H} [ v_n^c·cos(nθ) + v_n^s·sin(nθ) ]
    
    Parámetros: 2H por eje × 2 ejes = 4H total
    
    Entrenamiento offline: pseudoinversa (solución óptima global)
    Entrenamiento online:  LMS / NLMS (para Fase 1)
    """

    def __init__(self, H=10):
        """
        Args:
            H: número de armónicos en la base de Fourier
        """
        self.H = H
        self.n_features = 2 * H   # [cos θ, sin θ, cos 2θ, sin 2θ, ..., cos Hθ, sin Hθ]
        # Pesos: columna 0 = alpha, columna 1 = beta
        self.W = np.zeros((self.n_features, 2))

    def _build_features(self, theta):
        """
        Construye la matriz de regresores X para un vector de ángulos.
        
        Para un solo θ:  x(θ) = [cos θ, sin θ, cos 2θ, sin 2θ, ..., cos Hθ, sin Hθ]
        Para N ángulos:   X ∈ R^{N × 2H}
        
        Args:
            theta: array de ángulos (N,) o escalar
        Returns:
            X: matriz de regresores (N, 2H) o vector (2H,)
        """
        theta = np.atleast_1d(theta)
        N = len(theta)
        X = np.zeros((N, self.n_features))
        for n in range(1, self.H + 1):
            X[:, 2*(n-1)]     = np.cos(n * theta)  # cos(n·θ)
            X[:, 2*(n-1) + 1] = np.sin(n * theta)  # sin(n·θ)
        return X

    def fit_offline(self, theta, e_alpha, e_beta):
        """
        Entrenamiento offline por mínimos cuadrados (pseudoinversa).
        
        Resuelve:  min_W ||X·W - E||²_F
        Solución:  W* = (X^T X)^{-1} X^T E = X† E
        
        Esta es la solución ÓPTIMA GLOBAL — no hay mínimos locales,
        no hay hiperparámetros (learning rate, epochs, batch size).
        
        Args:
            theta: ángulos de entrenamiento (N,)
            e_alpha: BEMF alpha real (N,)
            e_beta:  BEMF beta real (N,)
        """
        X = self._build_features(theta)
        E = np.column_stack([e_alpha, e_beta])  # (N, 2)

        # Solución por mínimos cuadrados (usa SVD internamente, más estable que normal equations)
        self.W, residuals, rank, sv = lstsq(X, E)

        # Diagnóstico
        E_pred = X @ self.W
        mse_alpha = np.mean((E_pred[:, 0] - e_alpha)**2)
        mse_beta  = np.mean((E_pred[:, 1] - e_beta)**2)
        return mse_alpha, mse_beta

    def predict(self, theta):
        """
        Predicción: ê(θ) = X(θ) · W
        
        Args:
            theta: ángulos (N,) o escalar
        Returns:
            e_pred: (N, 2) con columnas [alpha, beta]
        """
        X = self._build_features(theta)
        return X @ self.W

    def predict_single(self, theta_scalar):
        """
        Predicción para un solo ángulo (eficiente para DSP).
        Evita crear matrices — solo producto punto.
        
        Args:
            theta_scalar: un ángulo escalar
        Returns:
            (e_alpha, e_beta) tupla
        """
        x = np.zeros(self.n_features)
        for n in range(1, self.H + 1):
            x[2*(n-1)]     = np.cos(n * theta_scalar)
            x[2*(n-1) + 1] = np.sin(n * theta_scalar)
        e_alpha = x @ self.W[:, 0]
        e_beta  = x @ self.W[:, 1]
        return e_alpha, e_beta

    def get_fourier_coefficients(self):
        """
        Extrae los coeficientes de Fourier de forma interpretable.
        
        Returns:
            dict con claves 'alpha' y 'beta', cada uno conteniendo
            listas de (n, amplitude, phase) para cada armónico.
        """
        result = {'alpha': [], 'beta': []}
        for axis, name in enumerate(['alpha', 'beta']):
            for n in range(1, self.H + 1):
                wc = self.W[2*(n-1), axis]      # coef coseno
                ws = self.W[2*(n-1) + 1, axis]  # coef seno
                amplitude = np.sqrt(wc**2 + ws**2)
                phase = np.arctan2(ws, wc)
                result[name].append({
                    'harmonic': n,
                    'cos_coeff': wc,
                    'sin_coeff': ws,
                    'amplitude': amplitude,
                    'phase_rad': phase,
                    'phase_deg': np.degrees(phase)
                })
        return result

    def get_weight_count(self):
        """Número total de parámetros."""
        return self.W.size


# ============================================================================
# 3. GENERACIÓN DEL DATASET
# ============================================================================

print("=" * 70)
print("FASE 0: ADALINE para Aprendizaje de BEMF")
print("=" * 70)

print("\n--- Paso 1: Generando Dataset ---")

n_samples = 10000
theta_train = np.linspace(0, 2*np.pi, n_samples, endpoint=False)

# BEMF real en ABC y luego Clarke
bemf_abc = np.array([generate_complex_bemf(th) for th in theta_train])
bemf_ab  = np.array([clarke_transform(abc) for abc in bemf_abc])

alpha_real = bemf_ab[:, 0]
beta_real  = bemf_ab[:, 1]

print(f"  Muestras: {n_samples}")
print(f"  Rango α: [{alpha_real.min():.4f}, {alpha_real.max():.4f}]")
print(f"  Rango β: [{beta_real.min():.4f}, {beta_real.max():.4f}]")

# ============================================================================
# 4. ENTRENAMIENTO ADALINE OFFLINE
# ============================================================================

print("\n--- Paso 2: Entrenamiento ADALINE (Pseudoinversa) ---")

H = 10  # Armónicos en la base de Fourier

adaline = ADALINE_BEMF(H=H)

t_start = time.time()
mse_alpha, mse_beta = adaline.fit_offline(theta_train, alpha_real, beta_real)
t_train = time.time() - t_start

print(f"  Armónicos H = {H}")
print(f"  Parámetros totales: {adaline.get_weight_count()} (vs ~25,000 de la NN)")
print(f"  Tiempo de entrenamiento: {t_train*1000:.2f} ms (vs ~30 s de la NN)")
print(f"  MSE α: {mse_alpha:.2e}")
print(f"  MSE β: {mse_beta:.2e}")
print(f"  RMSE α: {np.sqrt(mse_alpha):.6f}")
print(f"  RMSE β: {np.sqrt(mse_beta):.6f}")

# ============================================================================
# 5. ANÁLISIS DE COEFICIENTES DE FOURIER
# ============================================================================

print("\n--- Paso 3: Coeficientes de Fourier (interpretabilidad) ---")

coeffs = adaline.get_fourier_coefficients()

print(f"\n  {'Arm.':<6} {'|A_α|':<10} {'φ_α [°]':<10} {'|A_β|':<10} {'φ_β [°]':<10}")
print(f"  {'-'*46}")
for i in range(H):
    ca = coeffs['alpha'][i]
    cb = coeffs['beta'][i]
    marker = " ◄" if ca['amplitude'] > 0.01 else ""
    print(f"  {ca['harmonic']:<6} {ca['amplitude']:<10.4f} {ca['phase_deg']:<10.1f} "
          f"{cb['amplitude']:<10.4f} {cb['phase_deg']:<10.1f}{marker}")

print(f"\n  ◄ = armónico significativo (|A| > 0.01)")
print(f"  Nota: estos pesos SON la representación del motor.")
print(f"        En la NN profunda, esta información estaba oculta en ~25k params.")

# ============================================================================
# 6. ESTUDIO DE SENSIBILIDAD: ¿CUÁNTOS ARMÓNICOS NECESITAMOS?
# ============================================================================

print("\n--- Paso 4: Barrido de H (¿cuántos armónicos bastan?) ---")

H_values = [1, 2, 3, 5, 7, 10, 15, 20]
rmse_vs_H = {'H': [], 'rmse_alpha': [], 'rmse_beta': [], 'n_params': []}

for h in H_values:
    ada_h = ADALINE_BEMF(H=h)
    mse_a, mse_b = ada_h.fit_offline(theta_train, alpha_real, beta_real)
    rmse_vs_H['H'].append(h)
    rmse_vs_H['rmse_alpha'].append(np.sqrt(mse_a))
    rmse_vs_H['rmse_beta'].append(np.sqrt(mse_b))
    rmse_vs_H['n_params'].append(ada_h.get_weight_count())
    print(f"  H={h:>2d}  params={ada_h.get_weight_count():>3d}  "
          f"RMSE_α={np.sqrt(mse_a):.2e}  RMSE_β={np.sqrt(mse_b):.2e}")

# ============================================================================
# 7. COMPARACIÓN CON NN PROFUNDA (entrenamos la NN original para comparar)
# ============================================================================

print("\n--- Paso 5: Entrenamiento NN profunda (para comparación) ---")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True

    class BEMFNet(nn.Module):
        """Red original del script anterior (corregida: ahora SÍ usa codificación trig)."""
        def __init__(self, H_enc=10):
            super().__init__()
            self.H_enc = H_enc
            input_dim = 2 * H_enc  # [cos θ, sin θ, ..., cos Hθ, sin Hθ]
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 2)
            )

        def forward(self, theta):
            # Codificación trigonométrica (AHORA SÍ SE USA)
            features = []
            for n in range(1, self.H_enc + 1):
                features.append(torch.cos(n * theta))
                features.append(torch.sin(n * theta))
            x = torch.cat(features, dim=1)
            return self.net(x)

    # Preparar datos
    X_t = torch.FloatTensor(theta_train.reshape(-1, 1))
    y_t = torch.FloatTensor(bemf_ab)

    model_nn = BEMFNet(H_enc=H)
    n_params_nn = sum(p.numel() for p in model_nn.parameters())
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

    # Entrenamiento
    epochs = 300
    batch_size = 256
    nn_train_losses = []

    t_start = time.time()
    for epoch in range(epochs):
        model_nn.train()
        indices = torch.randperm(X_t.size(0))
        epoch_loss = 0
        n_batches = 0
        for i in range(0, X_t.size(0), batch_size):
            batch_idx = indices[i:i+batch_size]
            outputs = model_nn(X_t[batch_idx])
            loss = criterion(outputs, y_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        nn_train_losses.append(epoch_loss / n_batches)
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {nn_train_losses[-1]:.6f}")

    t_nn = time.time() - t_start

    # Predicción
    model_nn.eval()
    with torch.no_grad():
        nn_pred = model_nn(X_t).numpy()

    nn_rmse_alpha = np.sqrt(np.mean((nn_pred[:, 0] - alpha_real)**2))
    nn_rmse_beta  = np.sqrt(np.mean((nn_pred[:, 1] - beta_real)**2))

    print(f"  Parámetros NN: {n_params_nn}")
    print(f"  Tiempo NN: {t_nn:.2f} s")
    print(f"  RMSE NN α: {nn_rmse_alpha:.6f}")
    print(f"  RMSE NN β: {nn_rmse_beta:.6f}")

except ImportError:
    HAS_TORCH = False
    print("  [PyTorch no disponible — saltando comparación con NN]")
    nn_pred = None
    nn_rmse_alpha = None
    nn_rmse_beta = None
    n_params_nn = "N/A"
    t_nn = "N/A"
    nn_train_losses = []

# ============================================================================
# 8. GENERACIÓN DE LUT PARA MATLAB
# ============================================================================

print("\n--- Paso 6: Generación de LUT ---")

lut_resolution = 1000
lut_theta = np.linspace(0, 2*np.pi, lut_resolution, endpoint=False)

# Predicción ADALINE
ada_pred = adaline.predict(lut_theta)
lut_alpha = ada_pred[:, 0]
lut_beta  = ada_pred[:, 1]

# BEMF real para referencia
lut_real = np.array([clarke_transform(generate_complex_bemf(th)) for th in lut_theta])

# Error ADALINE
err_a = np.abs(lut_alpha - lut_real[:, 0])
err_b = np.abs(lut_beta - lut_real[:, 1])

print(f"  LUT: {lut_resolution} puntos")
print(f"  Error máx α: {err_a.max():.6f}")
print(f"  Error máx β: {err_b.max():.6f}")

# Guardar LUT (mismas variables que el script original para compatibilidad con MATLAB)
save_dict = {
    'lut_theta': lut_theta,
    'lut_alpha': lut_alpha,
    'lut_beta': lut_beta,
    'lut_alpha_real': lut_real[:, 0],
    'lut_beta_real': lut_real[:, 1],
    # Pesos del ADALINE (para Fase 1: inicialización online)
    'adaline_W': adaline.W,
    'adaline_H': H,
    # Modelo TRAP y SIN para comparación en MATLAB
    'method': 'ADALINE'
}
savemat('bemf_adaline_lut.mat', save_dict)
print(f"  Guardado: bemf_adaline_lut.mat")

# También guardar la versión compatible con el nombre original
savemat('bemf_lut.mat', {
    'lut_theta': lut_theta,
    'lut_alpha': lut_alpha,
    'lut_beta': lut_beta,
    'lut_alpha_real': lut_real[:, 0],
    'lut_beta_real': lut_real[:, 1]
})
print(f"  Guardado: bemf_lut.mat (compatible con scripts MATLAB existentes)")

# ============================================================================
# 9. VISUALIZACIÓN COMPLETA
# ============================================================================

print("\n--- Paso 7: Generando figuras ---")

fig = plt.figure(figsize=(18, 22))
fig.suptitle('Fase 0: ADALINE vs NN Profunda para Aprendizaje de BEMF',
             fontsize=16, fontweight='bold', y=0.98)

# ---- Subplot 1: BEMF Real (ABC) ----
ax1 = fig.add_subplot(4, 2, 1)
idx = np.linspace(0, n_samples-1, 500, dtype=int)
ax1.plot(theta_train[idx], bemf_abc[idx, 0], 'r-', lw=1.5, label='Phase A')
ax1.plot(theta_train[idx], bemf_abc[idx, 1], 'g-', lw=1.5, label='Phase B')
ax1.plot(theta_train[idx], bemf_abc[idx, 2], 'b-', lw=1.5, label='Phase C')
ax1.set_xlabel('θ [rad]')
ax1.set_ylabel('Amplitude')
ax1.set_title('BEMF Real — ABC')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ---- Subplot 2: Comparación Alpha ----
ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(lut_theta, lut_real[:, 0], 'k-', lw=2.5, label='Real', alpha=0.5)
ax2.plot(lut_theta, lut_alpha, 'r--', lw=2, label=f'ADALINE (H={H})')
if nn_pred is not None:
    lut_nn = np.array([clarke_transform(generate_complex_bemf(th)) for th in lut_theta])
    # Re-predict with NN at lut points
    model_nn.eval()
    with torch.no_grad():
        nn_lut_pred = model_nn(torch.FloatTensor(lut_theta.reshape(-1, 1))).numpy()
    ax2.plot(lut_theta, nn_lut_pred[:, 0], 'b:', lw=1.5, label='NN profunda')
ax2.set_xlabel('θ [rad]')
ax2.set_ylabel('e_α')
ax2.set_title('Comparación Alpha: Real vs ADALINE vs NN')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ---- Subplot 3: Comparación Beta ----
ax3 = fig.add_subplot(4, 2, 3)
ax3.plot(lut_theta, lut_real[:, 1], 'k-', lw=2.5, label='Real', alpha=0.5)
ax3.plot(lut_theta, ada_pred[:, 1], 'r--', lw=2, label=f'ADALINE (H={H})')
if nn_pred is not None:
    ax3.plot(lut_theta, nn_lut_pred[:, 1], 'b:', lw=1.5, label='NN profunda')
ax3.set_xlabel('θ [rad]')
ax3.set_ylabel('e_β')
ax3.set_title('Comparación Beta: Real vs ADALINE vs NN')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ---- Subplot 4: Error de Aproximación ----
ax4 = fig.add_subplot(4, 2, 4)
ax4.plot(lut_theta, err_a * 1000, 'r-', lw=1.5, label='ADALINE |Δα|')
ax4.plot(lut_theta, err_b * 1000, 'b-', lw=1.5, label='ADALINE |Δβ|')
if nn_pred is not None:
    nn_err_a = np.abs(nn_lut_pred[:, 0] - lut_real[:, 0])
    nn_err_b = np.abs(nn_lut_pred[:, 1] - lut_real[:, 1])
    ax4.plot(lut_theta, nn_err_a * 1000, 'r:', lw=1, alpha=0.7, label='NN |Δα|')
    ax4.plot(lut_theta, nn_err_b * 1000, 'b:', lw=1, alpha=0.7, label='NN |Δβ|')
ax4.set_xlabel('θ [rad]')
ax4.set_ylabel('Error [×10⁻³]')
ax4.set_title('Error de Aproximación')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ---- Subplot 5: Espectro de Fourier (coeficientes ADALINE) ----
ax5 = fig.add_subplot(4, 2, 5)
harmonics_n = np.arange(1, H+1)
amp_alpha = [coeffs['alpha'][i]['amplitude'] for i in range(H)]
amp_beta  = [coeffs['beta'][i]['amplitude'] for i in range(H)]
bar_width = 0.35
ax5.bar(harmonics_n - bar_width/2, amp_alpha, bar_width, color='tab:red',
        alpha=0.7, label='|w_α|')
ax5.bar(harmonics_n + bar_width/2, amp_beta, bar_width, color='tab:blue',
        alpha=0.7, label='|w_β|')
ax5.set_xlabel('Armónico n')
ax5.set_ylabel('Amplitud')
ax5.set_title('Coeficientes de Fourier del ADALINE (= espectro de BEMF)')
ax5.set_xticks(harmonics_n)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ---- Subplot 6: RMSE vs H (¿cuántos armónicos necesitamos?) ----
ax6 = fig.add_subplot(4, 2, 6)
ax6.semilogy(rmse_vs_H['H'], rmse_vs_H['rmse_alpha'], 'ro-', lw=2,
             markersize=6, label='RMSE α')
ax6.semilogy(rmse_vs_H['H'], rmse_vs_H['rmse_beta'], 'bs-', lw=2,
             markersize=6, label='RMSE β')
if nn_rmse_alpha is not None:
    ax6.axhline(nn_rmse_alpha, color='r', ls=':', alpha=0.5, label=f'NN α ({nn_rmse_alpha:.2e})')
    ax6.axhline(nn_rmse_beta, color='b', ls=':', alpha=0.5, label=f'NN β ({nn_rmse_beta:.2e})')
ax6.set_xlabel('Armónicos H')
ax6.set_ylabel('RMSE')
ax6.set_title('RMSE vs. Número de Armónicos')
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3)

# ---- Subplot 7: Trayectoria αβ ----
ax7 = fig.add_subplot(4, 2, 7)
ax7.plot(lut_real[:, 0], lut_real[:, 1], 'k-', lw=2, alpha=0.4, label='Real')
ax7.plot(lut_alpha, lut_beta, 'r--', lw=1.5, label='ADALINE')
if nn_pred is not None:
    ax7.plot(nn_lut_pred[:, 0], nn_lut_pred[:, 1], 'b:', lw=1, label='NN')
ax7.set_xlabel('e_α')
ax7.set_ylabel('e_β')
ax7.set_title('Trayectoria en plano αβ')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)
ax7.axis('equal')

# ---- Subplot 8: Tabla comparativa ----
ax8 = fig.add_subplot(4, 2, 8)
ax8.axis('off')

ada_rmse_a = np.sqrt(mse_alpha)
ada_rmse_b = np.sqrt(mse_beta)

table_data = [
    ['Métrica', 'ADALINE', 'NN Profunda'],
    ['Parámetros', f'{adaline.get_weight_count()}', f'{n_params_nn}'],
    ['Tiempo entrenamiento', f'{t_train*1000:.1f} ms', f'{t_nn:.1f} s' if isinstance(t_nn, float) else 'N/A'],
    ['RMSE α', f'{ada_rmse_a:.6f}', f'{nn_rmse_alpha:.6f}' if nn_rmse_alpha else 'N/A'],
    ['RMSE β', f'{ada_rmse_b:.6f}', f'{nn_rmse_beta:.6f}' if nn_rmse_beta else 'N/A'],
    ['Error máx α', f'{err_a.max():.6f}', f'{nn_err_a.max():.6f}' if nn_pred is not None else 'N/A'],
    ['Error máx β', f'{err_b.max():.6f}', f'{nn_err_b.max():.6f}' if nn_pred is not None else 'N/A'],
    ['Interpretable', 'SÍ (Fourier)', 'NO (caja negra)'],
    ['Online update', 'LMS trivial', 'Backprop (costoso)'],
    ['Solución', 'Óptima global', 'Mínimo local'],
]

table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.35, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Formatear header
for j in range(3):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Colorear filas donde ADALINE gana
wins = {1, 2, 7, 8, 9}  # filas donde ADALINE es claramente superior
for row_idx in wins:
    for col_idx in range(3):
        if col_idx == 1:
            table[row_idx, col_idx].set_facecolor('#d5f5e3')

ax8.set_title('Comparación cuantitativa', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('adaline_vs_nn_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Guardado: adaline_vs_nn_comparison.png")

# ============================================================================
# 10. RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 70)
print("RESUMEN FASE 0")
print("=" * 70)

# ¿El ADALINE iguala o supera a la NN?
if nn_rmse_alpha is not None:
    ratio_a = ada_rmse_a / nn_rmse_alpha
    ratio_b = ada_rmse_b / nn_rmse_beta
    print(f"\n  ADALINE RMSE α / NN RMSE α = {ratio_a:.2f}x")
    print(f"  ADALINE RMSE β / NN RMSE β = {ratio_b:.2f}x")

    if ratio_a <= 1.1 and ratio_b <= 1.1:
        print(f"\n  ✓ CRITERIO DE ÉXITO: ADALINE iguala o supera a la NN")
        print(f"    → Podemos reemplazar la NN por ADALINE sin pérdida")
        print(f"    → Ganamos: interpretabilidad, velocidad, y path a online (Fase 1)")
    elif ratio_a <= 2.0 and ratio_b <= 2.0:
        print(f"\n  ~ ADALINE es ligeramente peor pero aceptable")
        print(f"    → La ventaja de online learning compensa la pérdida marginal")
        print(f"    → Considerar aumentar H para cerrar la brecha")
    else:
        print(f"\n  ✗ ADALINE significativamente peor que NN")
        print(f"    → La BEMF tiene componentes no capturables por Fourier")
        print(f"    → Revisar: ¿la distorsión trapezoidal necesita más armónicos?")
        # Buscar el H que iguala a la NN
        for h_try in [15, 20, 30, 50]:
            ada_try = ADALINE_BEMF(H=h_try)
            mse_a_try, _ = ada_try.fit_offline(theta_train, alpha_real, beta_real)
            if np.sqrt(mse_a_try) <= nn_rmse_alpha * 1.1:
                print(f"    → Con H={h_try} se iguala la NN ({ada_try.get_weight_count()} params)")
                break

print(f"\n  Archivos generados:")
print(f"    1. bemf_adaline_lut.mat  — LUT + pesos ADALINE para MATLAB")
print(f"    2. bemf_lut.mat          — LUT compatible con scripts existentes")
print(f"    3. adaline_vs_nn_comparison.png — Figura comparativa completa")
print(f"\n  Siguiente paso: Fase 1 — ADALINE Online con LMS")
print("=" * 70)

plt.show()