%% main_compare_bemf_sources.m
% FCS-M2PC — Comparación BEMF sintética vs BEMF real (experimental dSPACE)
% =========================================================================
%
% Ejecuta la simulación completa dos veces (Fase 2 en ambos casos):
%   Run 1: USE_REAL_BEMF = false  (BEMF sintética — LUT trapezoidal + NN)
%   Run 2: USE_REAL_BEMF = true   (BEMF real experimental — dSPACE)
%
% Genera las Figuras 1-3 para cada fuente y la Figura 4 con el bar chart
% comparativo de torque ripple (6 métodos × 2 fuentes).
%
% Requiere: data/bemf_lut.mat, data/bemf_adaline_lut.mat
%           data/bemf_real_lut.mat, data/bemf_real_adaline_lut.mat
% =========================================================================
clear; close all; clc;
addpath(genpath('.'));

%% ── Parámetros base ──────────────────────────────────────────────────────

p0 = motor_params();    % nominal params (will be cloned per run)
Total_Steps = round(p0.t_final / p0.Ts);
t_ms = (1:Total_Steps) * p0.Ts * 1000;

%% ── Verificar archivos ───────────────────────────────────────────────────

required = {'data/bemf_lut.mat', 'data/bemf_adaline_lut.mat', ...
            'data/bemf_real_lut.mat', 'data/bemf_real_adaline_lut.mat'};
for k = 1:length(required)
    if ~isfile(required{k})
        error('Falta: %s\nEjecutar los scripts Python primero.', required{k});
    end
end

%% ── Colores y métodos ────────────────────────────────────────────────────

colors.TRAP         = [0.85, 0.33, 0.10];
colors.SIN          = [0.00, 0.45, 0.74];
colors.LEARNED      = [0.47, 0.67, 0.19];
colors.ADALINE_OFF  = [0.56, 0.27, 0.68];
colors.ADALINE_ON_T = [0.93, 0.69, 0.13];
colors.ADALINE_ON_S = [0.30, 0.75, 0.93];

methods = {'TRAP','SIN','LEARNED','ADALINE_OFF','ADALINE_ON_T','ADALINE_ON_S'};

%% ── Helper: load LUT + ADALINE ──────────────────────────────────────────

function [p, lut, W_optimal, H_adaline] = load_run_data(p_base, use_real)
    p = p_base;
    if use_real
        raw_lut = load('data/bemf_real_lut.mat');
        raw_ada = load('data/bemf_real_adaline_lut.mat');
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
end

%% ── Helper: compute ADALINE initial weights ─────────────────────────────

function [W_trap_init, W_sin_init] = make_adaline_inits(H_adaline)
    theta_grid = linspace(0, 2*pi, 5000)';
    X_grid = zeros(5000, 2*H_adaline);
    for n = 1:H_adaline
        X_grid(:, 2*(n-1)+1) = cos(n * theta_grid);
        X_grid(:, 2*(n-1)+2) = sin(n * theta_grid);
    end
    trap_ab = zeros(5000, 2);
    for i = 1:5000
        trap_ab(i, :) = bemf_trapezoidal(theta_grid(i))';
    end
    W_trap_init = X_grid \ trap_ab;

    W_sin_init       = zeros(2*H_adaline, 2);
    W_sin_init(2, 1) =  1.0;
    W_sin_init(1, 2) = -1.0;
end

%% ── Run 1: BEMF sintética ────────────────────────────────────────────────

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  RUN 1 — BEMF sintética (LUT trapezoidal + NN)             ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
fprintf('═══════════════════════════════════════════════\n');

[p_syn, lut_syn, W_opt_syn, H_syn] = load_run_data(p0, false);
[W_trap_syn, W_sin_syn] = make_adaline_inits(H_syn);

results_syn = simulate_all_methods(true, p_syn, H_syn, W_opt_syn, ...
                                   lut_syn, W_trap_syn, W_sin_syn);

fprintf('═══════════════════════════════════════════════\n\n');
plot_results(results_syn, methods, colors, p_syn, t_ms);

%% ── Run 2: BEMF real (experimental) ─────────────────────────────────────

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  RUN 2 — BEMF real experimental (dSPACE)                   ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
fprintf('═══════════════════════════════════════════════\n');

[p_real, lut_real, W_opt_real, H_real] = load_run_data(p0, true);
[W_trap_real, W_sin_real] = make_adaline_inits(H_real);

results_real = simulate_all_methods(true, p_real, H_real, W_opt_real, ...
                                    lut_real, W_trap_real, W_sin_real);

fprintf('═══════════════════════════════════════════════\n\n');
plot_results(results_real, methods, colors, p_real, t_ms);

%% ── Figura comparativa: bar chart 6 métodos × 2 fuentes ─────────────────

figure('Color', 'w', 'Position', [200, 200, 1000, 520]);

ripple_syn  = cellfun(@(m) results_syn.(m).Te_ripple,  methods);
ripple_real = cellfun(@(m) results_real.(m).Te_ripple, methods);

x = 1:length(methods);
w = 0.35;

b1 = bar(x - w/2, ripple_syn,  w, 'FaceColor', [0.55 0.55 0.55]);
hold on;
b2 = bar(x + w/2, ripple_real, w, 'FaceColor', [0.20 0.60 0.90]);

set(gca, 'XTick', x, 'XTickLabel', methods, 'XTickLabelRotation', 25, ...
    'FontSize', 9);
ylabel('Torque Ripple [%]');
title('Figura 4 — Comparación de Ripple: BEMF Sintética vs Real', ...
      'FontSize', 13, 'FontWeight', 'bold');
legend([b1 b2], {'Sintética (LUT trap+NN)', 'Real (dSPACE)'}, ...
       'Location', 'northwest');
grid on; box on;

% Value labels
for i = 1:length(methods)
    text(i - w/2, ripple_syn(i),  sprintf('%.2f', ripple_syn(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 7, 'FontWeight', 'bold');
    text(i + w/2, ripple_real(i), sprintf('%.2f', ripple_real(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 7, 'FontWeight', 'bold', 'Color', [0.0 0.30 0.60]);
end

%% ── Reporte comparativo ──────────────────────────────────────────────────

fprintf('╔═══════════════════════════════════════════════════════════════════════╗\n');
fprintf('║           IMPACTO DE LA FUENTE DE BEMF EN EL RIPPLE               ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Método         │ Sintética [%%] │  Real [%%]  │  Δ [%%]  │  Δ rel.  ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════════════╣\n');

for idx = 1:length(methods)
    m   = methods{idx};
    rs  = results_syn.(m).Te_ripple;
    rr  = results_real.(m).Te_ripple;
    d   = rr - rs;
    drel = 100 * d / rs;
    fprintf('║ %-14s │    %6.2f    │   %6.2f   │  %+.2f  │  %+.1f%%  ║\n', ...
            m, rs, rr, d, drel);
end

fprintf('╠═══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Ke sintético: %.5f  →  Ke real: %.5f V·s/rad  (Δ = %.2f%%)       ║\n', ...
        p_syn.Ke, p_real.Ke, 100*(p_real.Ke - p_syn.Ke)/p_syn.Ke);
fprintf('╚═══════════════════════════════════════════════════════════════════════╝\n\n');
