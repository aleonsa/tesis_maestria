"""
Script Python: Generación de BEMF Real y Entrenamiento de LUT Neural
======================================================================
Crea una BEMF compleja (senoidal + armónicos) que no es ni puramente
senoidal ni trapezoidal, luego entrena una red neuronal para aprenderla
y genera un lookup table para usar en MATLAB.

CORRECCIONES RESPECTO A VERSIÓN ANTERIOR:
  - La red neuronal ahora USA la codificación trigonométrica (antes se
    calculaba theta_encoded pero se pasaba theta crudo)
  - Input dimension corregida: 6 features (sin/cos de theta, 2*theta, 3*theta)
    en lugar de 1 (theta escalar)
  - Agregado LR scheduler y early stopping
  - Validación con puntos equiespaciados desplazados (no random split)
  - Arquitectura más limpia y apropiada para función periódica
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# 1. GENERACIÓN DE LA BEMF "REAL" (Compleja con Armónicos)
# ============================================================================

def generate_complex_bemf(theta, harmonics=True):
    """
    Genera una BEMF compleja que no es ni senoidal pura ni trapezoidal.
    
    Componentes:
    - Fundamental senoidal (70%)
    - 3er armónico (20%) - característico de motores reales
    - 5to armónico (7%)
    - 7mo armónico (3%)
    - Pequeña distorsión trapezoidal (saturación magnética)
    
    Returns: shape_abc (3 fases)
    """
    theta_a = theta
    theta_b = theta - 2*np.pi/3
    theta_c = theta - 4*np.pi/3
    
    def phase_shape(th):
        # Fundamental senoidal
        y = 0.70 * np.sin(th)
        
        if harmonics:
            # 3er armónico (común en motores trifásicos)
            y += 0.20 * np.sin(3*th + np.pi/6)
            
            # 5to armónico
            y += 0.07 * np.sin(5*th - np.pi/4)
            
            # 7mo armónico
            y += 0.03 * np.sin(7*th + np.pi/3)
            
            # Pequeña componente trapezoidal (saturación)
            y += 0.05 * np.sign(np.sin(th)) * (1 - np.exp(-5*np.abs(np.sin(th))))
        
        return y
    
    phase_a = phase_shape(theta_a)
    phase_b = phase_shape(theta_b)
    phase_c = phase_shape(theta_c)
    
    return np.array([phase_a, phase_b, phase_c])

def clarke_transform(abc):
    """Transformación de Clarke: ABC -> Alpha-Beta"""
    T_clarke = (2/3) * np.array([
        [1, -0.5, -0.5],
        [0, np.sqrt(3)/2, -np.sqrt(3)/2]
    ])
    return T_clarke @ abc

# ============================================================================
# 2. GENERACIÓN DEL DATASET
# ============================================================================

print("="*70)
print("PASO 1: Generando Dataset de BEMF Real")
print("="*70)

# Dataset de entrenamiento: puntos equiespaciados
n_train = 8000
theta_train = np.linspace(0, 2*np.pi, n_train, endpoint=False)

# Dataset de validación: puntos equiespaciados DESPLAZADOS medio paso
# Esto valida la interpolación en puntos que la red nunca vio,
# y es más informativo que random split para funciones periódicas
n_val = 2000
delta = (2*np.pi / n_train) / 2  # Medio paso del training grid
theta_val = np.linspace(delta, 2*np.pi + delta, n_val, endpoint=False)
theta_val = np.mod(theta_val, 2*np.pi)

# Generar formas de onda ABC y transformar a Alpha-Beta
bemf_abc_train = np.array([generate_complex_bemf(th) for th in theta_train])
bemf_ab_train = np.array([clarke_transform(abc) for abc in bemf_abc_train])

bemf_abc_val = np.array([generate_complex_bemf(th) for th in theta_val])
bemf_ab_val = np.array([clarke_transform(abc) for abc in bemf_abc_val])

print(f"✓ Dataset generado: {n_train} train + {n_val} validation")
print(f"  Rango Alpha: [{bemf_ab_train[:, 0].min():.3f}, {bemf_ab_train[:, 0].max():.3f}]")
print(f"  Rango Beta:  [{bemf_ab_train[:, 1].min():.3f}, {bemf_ab_train[:, 1].max():.3f}]")

# ============================================================================
# 3. VISUALIZACIÓN DE LA BEMF REAL
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ABC Original
ax = axes[0, 0]
sample_points = 500
idx = np.linspace(0, n_train-1, sample_points, dtype=int)
ax.plot(theta_train[idx], bemf_abc_train[idx, 0], 'r-', label='Phase A', linewidth=1.5)
ax.plot(theta_train[idx], bemf_abc_train[idx, 1], 'g-', label='Phase B', linewidth=1.5)
ax.plot(theta_train[idx], bemf_abc_train[idx, 2], 'b-', label='Phase C', linewidth=1.5)
ax.set_xlabel('Theta [rad]', fontsize=11)
ax.set_ylabel('Normalized Amplitude', fontsize=11)
ax.set_title('BEMF Real - ABC Frame (con armónicos)', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Alpha-Beta
ax = axes[0, 1]
ax.plot(theta_train[idx], bemf_ab_train[idx, 0], 'b-', label='Alpha', linewidth=2)
ax.plot(theta_train[idx], bemf_ab_train[idx, 1], 'r-', label='Beta', linewidth=2)
ax.set_xlabel('Theta [rad]', fontsize=11)
ax.set_ylabel('Normalized Amplitude', fontsize=11)
ax.set_title('BEMF Real - Alpha-Beta Frame', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Espacio Alpha-Beta (trayectoria)
ax = axes[1, 0]
ax.plot(bemf_ab_train[:, 0], bemf_ab_train[:, 1], 'b-', linewidth=1.5, alpha=0.7)
ax.plot(bemf_ab_train[0, 0], bemf_ab_train[0, 1], 'go', markersize=10, label='Start')
ax.set_xlabel('Alpha', fontsize=11)
ax.set_ylabel('Beta', fontsize=11)
ax.set_title('Trayectoria en Plano Alpha-Beta', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# FFT de Phase A (para ver armónicos)
ax = axes[1, 1]
fft_a = np.fft.fft(bemf_abc_train[:, 0])
freqs = np.fft.fftfreq(n_train, d=2*np.pi/n_train)
magnitude = np.abs(fft_a[:n_train//2])
magnitude = magnitude / magnitude.max()  # Normalizar

ax.stem(freqs[:20], magnitude[:20], linefmt='b-', markerfmt='bo', basefmt='k-')
ax.set_xlabel('Harmonic Order', fontsize=11)
ax.set_ylabel('Normalized Magnitude', fontsize=11)
ax.set_title('Espectro de Armónicos (Phase A)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 15])

plt.tight_layout()
plt.savefig('bemf_real_shape.png', dpi=150, bbox_inches='tight')
print("✓ Gráfica guardada: bemf_real_shape.png")

# ============================================================================
# 4. RED NEURONAL PARA APRENDER LA BEMF
# ============================================================================

print("\n" + "="*70)
print("PASO 2: Entrenamiento de Red Neuronal")
print("="*70)

def trigonometric_encoding(theta, n_harmonics=7):
    """
    Codifica theta en features trigonométricos para garantizar periodicidad.
    
    Para una función f(theta) = f(theta + 2*pi), la representación óptima
    es en términos de sin(n*theta) y cos(n*theta). Esto le da a la red
    la base de Fourier como input — solo tiene que aprender los coeficientes.
    
    Args:
        theta: tensor [N, 1] de ángulos
        n_harmonics: número de armónicos a incluir (default 7 para cubrir
                     hasta el 7mo armónico de la BEMF)
    
    Returns:
        tensor [N, 2*n_harmonics] de features trigonométricos
    """
    features = []
    for n in range(1, n_harmonics + 1):
        features.append(torch.sin(n * theta))
        features.append(torch.cos(n * theta))
    return torch.cat(features, dim=1)


# Número de armónicos en la codificación.
# Nuestra BEMF tiene hasta el 7mo armónico, pero la componente trapezoidal
# (sign * exp) genera armónicos de orden superior. Usamos algunos extras.
N_HARMONICS = 10
N_FEATURES = 2 * N_HARMONICS  # = 20 features (sin y cos de cada armónico)


class BEMFNet(nn.Module):
    """
    Red neuronal para aprender la forma de BEMF en alpha-beta.
    
    Arquitectura: codificación trigonométrica → MLP → [alpha, beta]
    
    La codificación trigonométrica resuelve dos problemas:
    1. Periodicidad: sin(theta) = sin(theta + 2*pi) automáticamente
    2. Base de Fourier: la red solo necesita aprender combinaciones
       lineales/no-lineales de sin(n*theta), cos(n*theta)
    """
    def __init__(self, n_harmonics=10):
        super(BEMFNet, self).__init__()
        n_features = 2 * n_harmonics
        self.n_harmonics = n_harmonics
        
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # Output: [alpha, beta]
        )
    
    def forward(self, theta):
        # Codificar theta con sin/cos para periodicidad
        x = trigonometric_encoding(theta, self.n_harmonics)
        # Pasar por la red
        return self.net(x)


# Preparar datos como tensores
X_train_t = torch.FloatTensor(theta_train.reshape(-1, 1))
y_train_t = torch.FloatTensor(bemf_ab_train)
X_val_t = torch.FloatTensor(theta_val.reshape(-1, 1))
y_val_t = torch.FloatTensor(bemf_ab_val)

# Crear modelo
model = BEMFNet(n_harmonics=N_HARMONICS)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler: reduce LR cuando val_loss se estanca
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20
)

# Entrenamiento con early stopping
epochs = 500
batch_size = 128
train_losses = []
val_losses = []

# Early stopping
best_val_loss = float('inf')
best_model_state = None
patience = 50
patience_counter = 0

print(f"Entrenando red neuronal (max {epochs} épocas, early stopping patience={patience})...")
print(f"  Arquitectura: {N_FEATURES} inputs → 64 → 64 → 2 outputs")
print(f"  Codificación: {N_HARMONICS} armónicos (sin/cos)")

for epoch in range(epochs):
    model.train()
    
    # Mini-batches
    indices = torch.randperm(X_train_t.size(0))
    epoch_loss = 0
    n_batches = 0
    
    for i in range(0, X_train_t.size(0), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_X = X_train_t[batch_idx]
        batch_y = y_train_t[batch_idx]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    # Validación
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
    
    avg_train_loss = epoch_loss / n_batches
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    
    # Learning rate scheduler
    scheduler.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 50 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Época {epoch+1}/{epochs} - Train: {avg_train_loss:.8f}, "
              f"Val: {val_loss:.8f}, LR: {current_lr:.6f}")
    
    if patience_counter >= patience:
        print(f"\n  ⚡ Early stopping en época {epoch+1} (val_loss no mejoró en {patience} épocas)")
        break

# Restaurar mejor modelo
model.load_state_dict(best_model_state)
print(f"✓ Entrenamiento completado. Mejor val_loss: {best_val_loss:.8f}")

# ============================================================================
# 5. GENERACIÓN DEL LOOKUP TABLE
# ============================================================================

print("\n" + "="*70)
print("PASO 3: Generación de Lookup Table")
print("="*70)

# Crear LUT de alta resolución
lut_resolution = 1000
lut_theta = np.linspace(0, 2*np.pi, lut_resolution)

# Predicción con la red neuronal
model.eval()
with torch.no_grad():
    lut_theta_t = torch.FloatTensor(lut_theta.reshape(-1, 1))
    lut_ab_pred = model(lut_theta_t).numpy()

lut_alpha = lut_ab_pred[:, 0]
lut_beta = lut_ab_pred[:, 1]

# Calcular BEMF real para comparación
lut_ab_real = np.array([clarke_transform(generate_complex_bemf(th)) for th in lut_theta])
error_alpha = np.abs(lut_alpha - lut_ab_real[:, 0])
error_beta = np.abs(lut_beta - lut_ab_real[:, 1])

rmse_alpha = np.sqrt(np.mean((lut_alpha - lut_ab_real[:, 0])**2))
rmse_beta = np.sqrt(np.mean((lut_beta - lut_ab_real[:, 1])**2))

print(f"✓ LUT generada: {lut_resolution} puntos")
print(f"  Error máximo Alpha: {error_alpha.max():.6f}")
print(f"  Error máximo Beta:  {error_beta.max():.6f}")
print(f"  RMSE Alpha: {rmse_alpha:.6f}")
print(f"  RMSE Beta:  {rmse_beta:.6f}")

# Verificación de periodicidad: error en los bordes
boundary_err_alpha = abs(lut_alpha[0] - lut_alpha[-1])
boundary_err_beta = abs(lut_beta[0] - lut_beta[-1])
print(f"  Boundary continuity Alpha: {boundary_err_alpha:.6f}")
print(f"  Boundary continuity Beta:  {boundary_err_beta:.6f}")

# ============================================================================
# 6. GUARDAR PARA MATLAB
# ============================================================================

savemat('bemf_lut.mat', {
    'lut_theta': lut_theta,
    'lut_alpha': lut_alpha,
    'lut_beta': lut_beta,
    'lut_alpha_real': lut_ab_real[:, 0],
    'lut_beta_real': lut_ab_real[:, 1]
})

print("\n✓ Archivo guardado: bemf_lut.mat")

# ============================================================================
# 7. VISUALIZACIÓN DE RESULTADOS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Comparación Alpha
ax = axes[0, 0]
ax.plot(lut_theta, lut_ab_real[:, 0], 'b-', linewidth=2, label='BEMF Real', alpha=0.7)
ax.plot(lut_theta, lut_alpha, 'r--', linewidth=2, label='Red Neuronal (LUT)', alpha=0.8)
ax.set_xlabel('Theta [rad]', fontsize=11)
ax.set_ylabel('Alpha Component', fontsize=11)
ax.set_title('Comparación Alpha: Real vs Neural Network', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Comparación Beta
ax = axes[0, 1]
ax.plot(lut_theta, lut_ab_real[:, 1], 'b-', linewidth=2, label='BEMF Real', alpha=0.7)
ax.plot(lut_theta, lut_beta, 'r--', linewidth=2, label='Red Neuronal (LUT)', alpha=0.8)
ax.set_xlabel('Theta [rad]', fontsize=11)
ax.set_ylabel('Beta Component', fontsize=11)
ax.set_title('Comparación Beta: Real vs Neural Network', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Error de aproximación
ax = axes[1, 0]
ax.plot(lut_theta, error_alpha*1000, 'b-', linewidth=1.5, label='Error Alpha')
ax.plot(lut_theta, error_beta*1000, 'r-', linewidth=1.5, label='Error Beta')
ax.set_xlabel('Theta [rad]', fontsize=11)
ax.set_ylabel('Error [×10⁻³]', fontsize=11)
ax.set_title(f'Error de Aproximación (RMSE α={rmse_alpha:.5f}, β={rmse_beta:.5f})',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Curva de aprendizaje
ax = axes[1, 1]
ax.plot(train_losses, 'b-', linewidth=1.5, label='Train Loss', alpha=0.7)
ax.plot(val_losses, 'r-', linewidth=1.5, label='Validation Loss', alpha=0.7)
if best_val_loss < float('inf'):
    best_epoch = val_losses.index(min(val_losses))
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5,
               label=f'Best epoch ({best_epoch})')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('MSE Loss', fontsize=11)
ax.set_title('Curva de Aprendizaje', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('neural_network_results.png', dpi=150, bbox_inches='tight')
print("✓ Gráfica guardada: neural_network_results.png")

# ============================================================================
# 8. COMPARACIÓN CON MODELOS IDEALES (nuevo - para validar en MATLAB)
# ============================================================================

print("\n" + "="*70)
print("PASO 4: Comparación con modelos ideales")
print("="*70)

# Generar modelos ideales para comparación
sin_alpha = np.sin(lut_theta)
sin_beta = -np.cos(lut_theta)

# Trapezoidal (misma función que en MATLAB, con fase alineada a sin)
def trap_abc(theta):
    phases = [0, 2*np.pi/3, 4*np.pi/3]
    abc = np.zeros(3)
    for i in range(3):
        ti = np.mod(theta - phases[i] + np.pi/6, 2*np.pi)
        if ti < np.pi/6:
            val = ti * (6/np.pi)
        elif ti < 5*np.pi/6:
            val = 1.0
        elif ti < 7*np.pi/6:
            val = 1.0 - (ti - 5*np.pi/6) * (6/np.pi)
        elif ti < 11*np.pi/6:
            val = -1.0
        else:
            val = -1.0 + (ti - 11*np.pi/6) * (6/np.pi)
        abc[i] = val
    return abc

trap_ab = np.array([clarke_transform(trap_abc(th)) for th in lut_theta])

# Calcular RMSE de cada modelo vs BEMF real
rmse_sin = np.sqrt(np.mean((sin_alpha - lut_ab_real[:, 0])**2 +
                            (sin_beta - lut_ab_real[:, 1])**2))
rmse_trap = np.sqrt(np.mean((trap_ab[:, 0] - lut_ab_real[:, 0])**2 +
                             (trap_ab[:, 1] - lut_ab_real[:, 1])**2))
rmse_nn = np.sqrt(np.mean((lut_alpha - lut_ab_real[:, 0])**2 +
                           (lut_beta - lut_ab_real[:, 1])**2))

print(f"  RMSE vs Real:")
print(f"    Sinusoidal:   {rmse_sin:.6f}")
print(f"    Trapezoidal:  {rmse_trap:.6f}")
print(f"    Red Neuronal: {rmse_nn:.6f}")
print(f"  Mejora NN vs SIN:  {100*(1 - rmse_nn/rmse_sin):.1f}%")
print(f"  Mejora NN vs TRAP: {100*(1 - rmse_nn/rmse_trap):.1f}%")

print("\n" + "="*70)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("="*70)
print("\nArchivos generados:")
print("  1. bemf_lut.mat            - Lookup table para MATLAB")
print("  2. bemf_real_shape.png     - Visualización de BEMF real")
print("  3. neural_network_results.png - Resultados del entrenamiento")
print("\n¡Ahora puedes ejecutar el script de MATLAB!")
print("="*70)

plt.show()