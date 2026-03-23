%% main_comparison.m
% FCS-M2PC — Comparación de 6 modelos de BEMF (single run)
% =========================================================================
%
% Flags de configuración:
%   USE_OBSERVER  false → Fase 1: LMS usa BEMF real de la planta (ideal)
%                 true  → Fase 2: LMS usa BEMF estimado por observador
%
%   USE_REAL_BEMF false → BEMF de planta = sintética (LUT trapezoidal+NN)
%                 true  → BEMF de planta = LUT medida experimentalmente
%                         (requiere data/bemf_real_lut.mat)
%
% Requiere: data/bemf_lut.mat, data/bemf_adaline_lut.mat
%           (+ data/bemf_real_lut.mat si USE_REAL_BEMF=true)
% =========================================================================
clear; close all; clc;
addpath(genpath('.'));

%% ── Configuración ────────────────────────────────────────────────────────

USE_OBSERVER  = true;    % false = Fase 1 | true = Fase 2
USE_REAL_BEMF = true;   % false = sintético | true = experimental (dSPACE)

%% ── Parámetros ───────────────────────────────────────────────────────────

p = motor_params();
Total_Steps = round(p.t_final / p.Ts);

%% ── Cargar datos ─────────────────────────────────────────────────────────

if ~isfile('data/bemf_lut.mat') || ~isfile('data/bemf_adaline_lut.mat')
    error('Faltan archivos en data/. Ejecutar generate_lut_adaline.py primero.');
end

if USE_REAL_BEMF
    if ~isfile('data/bemf_real_lut.mat') || ~isfile('data/bemf_real_adaline_lut.mat')
        error('Faltan archivos reales. Ejecutar generate_real_bemf_lut.py primero.');
    end
    raw_lut = load('data/bemf_real_lut.mat');
    raw_ada = load('data/bemf_real_adaline_lut.mat');
    % Update Ke from measured value; scale Kt proportionally (Kt ∝ Ke)
    ke_ratio = raw_lut.Ke_measured / p.Ke;
    p.Ke = raw_lut.Ke_measured;
    p.Kt = p.Kt * ke_ratio;
else
    raw_lut = load('data/bemf_lut.mat');
    raw_ada = load('data/bemf_adaline_lut.mat');
end

lut.theta      = raw_lut.lut_theta;
lut.alpha_real = raw_lut.lut_alpha_real;
lut.beta_real  = raw_lut.lut_beta_real;
lut.alpha_nn   = raw_lut.lut_alpha;
lut.beta_nn    = raw_lut.lut_beta;

W_optimal = raw_ada.adaline_W;
H_adaline = double(raw_ada.adaline_H);

bemf_source = ternary(USE_REAL_BEMF, 'experimental (dSPACE)', 'sintética (NN+TRAP)');

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  FCS-M2PC — 6 modelos de BEMF                              ║\n');
fprintf('║  Fase:  %-51s║\n', ...
        ternary(USE_OBSERVER, '2 — Observador de BEMF (realista)', ...
                              '1 — BEMF real (ideal)'));
fprintf('║  BEMF:  %-51s║\n', bemf_source);
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
fprintf('LUT: %d pts | ADALINE: H=%d (%d params) | USE_OBSERVER=%d\n', ...
        length(lut.theta), H_adaline, 4*H_adaline, USE_OBSERVER);
fprintf('Ke=%.5f V·s/rad | Simulación: %.0f ms, Ts=%.0f µs, ω_ref=%.0f rad/s\n\n', ...
        p.Ke, p.t_final*1e3, p.Ts*1e6, p.w_ref);

%% ── Inicializaciones ADALINE ─────────────────────────────────────────────

% Trapezoidal: ajuste de Fourier por mínimos cuadrados sobre una grilla densa
theta_grid = linspace(0, 2*pi, 5000)';
X_grid     = zeros(5000, 2*H_adaline);
for n = 1:H_adaline
    X_grid(:, 2*(n-1)+1) = cos(n * theta_grid);
    X_grid(:, 2*(n-1)+2) = sin(n * theta_grid);
end
trap_ab = zeros(5000, 2);
for i = 1:5000
    trap_ab(i, :) = bemf_trapezoidal(theta_grid(i))';
end
W_trap_init = X_grid \ trap_ab;

% Sinusoidal: solo el fundamental
W_sin_init       = zeros(2*H_adaline, 2);
W_sin_init(2, 1) =  1.0;   % sin(θ) → α
W_sin_init(1, 2) = -1.0;   % -cos(θ) → β

%% ── Colores por método ───────────────────────────────────────────────────

colors.TRAP         = [0.85, 0.33, 0.10];
colors.SIN          = [0.00, 0.45, 0.74];
colors.LEARNED      = [0.47, 0.67, 0.19];
colors.ADALINE_OFF  = [0.56, 0.27, 0.68];
colors.ADALINE_ON_T = [0.93, 0.69, 0.13];
colors.ADALINE_ON_S = [0.30, 0.75, 0.93];

methods = {'TRAP','SIN','LEARNED','ADALINE_OFF','ADALINE_ON_T','ADALINE_ON_S'};

%% ── Simulación ───────────────────────────────────────────────────────────

fprintf('═══════════════════════════════════════════════\n');
results = simulate_all_methods(USE_OBSERVER, p, H_adaline, W_optimal, ...
                               lut, W_trap_init, W_sin_init);
fprintf('═══════════════════════════════════════════════\n\n');

%% ── Gráficas ─────────────────────────────────────────────────────────────

t_ms = (1:Total_Steps) * p.Ts * 1000;
plot_results(results, methods, colors, p, t_ms);

%% ── Reporte ──────────────────────────────────────────────────────────────

fprintf('╔═════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                        RESULTADOS — 6 MÉTODOS                         ║\n');
fprintf('╠═════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Método         │ Ripple [%%] │ Te P-P [Nm] │ ω P-P [rad/s] │ ω err   ║\n');
fprintf('╠═════════════════════════════════════════════════════════════════════════╣\n');
for idx = 1:length(methods)
    m = methods{idx};
    r = results.(m);
    fprintf('║ %-14s │   %6.2f   │    %.4f   │    %.4f     │ %.4f  ║\n', ...
            m, r.Te_ripple, r.Te_pp, r.w_pp, r.w_ss_error);
end
fprintf('╚═════════════════════════════════════════════════════════════════════════╝\n\n');

r_trap  = results.TRAP.Te_ripple;
r_nn    = results.LEARNED.Te_ripple;
r_off   = results.ADALINE_OFF.Te_ripple;
r_on_t  = results.ADALINE_ON_T.Te_ripple;
r_on_s  = results.ADALINE_ON_S.Te_ripple;

fprintf('COMPARACIONES CLAVE:\n\n');
fprintf('  ADALINE offline vs NN (LEARNED):\n');
fprintf('    Ripple: %.2f%% vs %.2f%% (Δ = %+.2f%%)\n', r_off, r_nn, r_off - r_nn);
if abs(r_off - r_nn) < 0.5
    fprintf('    → Rendimiento equivalente con 40 vs ~25k parámetros ✓\n');
end

fprintf('\n  ADALINE online (desde TRAP) vs offline:\n');
fprintf('    Ripple: %.2f%% vs %.2f%% (Δ = %+.2f%%)\n', r_on_t, r_off, r_on_t - r_off);
fprintf('    → Online convergió al nivel del offline: %s\n', ...
        ternary(abs(r_on_t - r_off) < 1.0, 'SÍ ✓', 'NO ✗'));

fprintf('\n  ADALINE online (desde SIN) vs offline:\n');
fprintf('    Ripple: %.2f%% vs %.2f%% (Δ = %+.2f%%)\n', r_on_s, r_off, r_on_s - r_off);
fprintf('    → Online convergió al nivel del offline: %s\n', ...
        ternary(abs(r_on_s - r_off) < 1.0, 'SÍ ✓', 'NO ✗'));

fprintf('\n  Mejora global (ADALINE online desde TRAP vs TRAP fijo):\n');
fprintf('    Reducción de ripple: %.1f%%\n\n', 100*(r_trap - r_on_t)/r_trap);
