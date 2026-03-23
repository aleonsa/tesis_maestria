%% main_compare_phases.m
% FCS-M2PC — Comparación Fase 1 (ideal) vs Fase 2 (observador)
% =========================================================================
%
% Ejecuta la simulación completa dos veces:
%   Fase 1: USE_OBSERVER = false (señal de error LMS = BEMF real)
%   Fase 2: USE_OBSERVER = true  (señal de error LMS = BEMF observado)
%
% Genera las Figuras 1–3 para cada fase y la Figura 4 que compara
% directamente la convergencia de pesos de los métodos online.
%
% Requiere: data/bemf_lut.mat, data/bemf_adaline_lut.mat
% =========================================================================
clear; close all; clc;
addpath(genpath('.'));

%% ── Parámetros ───────────────────────────────────────────────────────────

p = motor_params();
Total_Steps = round(p.t_final / p.Ts);

%% ── Cargar datos ─────────────────────────────────────────────────────────

if ~isfile('data/bemf_lut.mat') || ~isfile('data/bemf_adaline_lut.mat')
    error('Faltan archivos en data/. Ejecutar generate_lut_adaline.py primero.');
end

raw_lut = load('data/bemf_lut.mat');
raw_ada = load('data/bemf_adaline_lut.mat');

lut.theta      = raw_lut.lut_theta;
lut.alpha_real = raw_lut.lut_alpha_real;
lut.beta_real  = raw_lut.lut_beta_real;
lut.alpha_nn   = raw_lut.lut_alpha;
lut.beta_nn    = raw_lut.lut_beta;

W_optimal = raw_ada.adaline_W;
H_adaline = double(raw_ada.adaline_H);

%% ── Inicializaciones ADALINE ─────────────────────────────────────────────

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

W_sin_init       = zeros(2*H_adaline, 2);
W_sin_init(2, 1) =  1.0;
W_sin_init(1, 2) = -1.0;

%% ── Colores y métodos ────────────────────────────────────────────────────

colors.TRAP         = [0.85, 0.33, 0.10];
colors.SIN          = [0.00, 0.45, 0.74];
colors.LEARNED      = [0.47, 0.67, 0.19];
colors.ADALINE_OFF  = [0.56, 0.27, 0.68];
colors.ADALINE_ON_T = [0.93, 0.69, 0.13];
colors.ADALINE_ON_S = [0.30, 0.75, 0.93];

methods = {'TRAP','SIN','LEARNED','ADALINE_OFF','ADALINE_ON_T','ADALINE_ON_S'};

t_ms = (1:Total_Steps) * p.Ts * 1000;

%% ── Fase 1: LMS ideal ────────────────────────────────────────────────────

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  FASE 1 — LMS con BEMF real (ideal)                        ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
fprintf('═══════════════════════════════════════════════\n');

results_p1 = simulate_all_methods(false, p, H_adaline, W_optimal, ...
                                   lut, W_trap_init, W_sin_init);

fprintf('═══════════════════════════════════════════════\n\n');
plot_results(results_p1, methods, colors, p, t_ms);

%% ── Fase 2: LMS con observador ───────────────────────────────────────────

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  FASE 2 — LMS con observador de BEMF (realista)            ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
fprintf('═══════════════════════════════════════════════\n');

results_p2 = simulate_all_methods(true, p, H_adaline, W_optimal, ...
                                   lut, W_trap_init, W_sin_init);

fprintf('═══════════════════════════════════════════════\n\n');
plot_results(results_p2, methods, colors, p, t_ms);

%% ── Figura 4: Comparación de convergencia ideal vs observador ────────────

figure('Color', 'w', 'Position', [200, 200, 1200, 450]);

% -- Left subplot: ADALINE_ON_T --
subplot(1, 2, 1);

plot(t_ms, results_p1.ADALINE_ON_T.W_error * 100, ...
     'Color', colors.ADALINE_ON_T, 'LineWidth', 2.0, ...
     'DisplayName', 'Ideal (Fase 1)');
hold on;
plot(t_ms, results_p2.ADALINE_ON_T.W_error * 100, ...
     'Color', colors.ADALINE_ON_T, 'LineWidth', 2.0, ...
     'LineStyle', '--', 'DisplayName', 'Observador (Fase 2)');
yline(5, 'k:', '5%', 'LineWidth', 1, 'HandleVisibility', 'off');

ylabel('||W - W^*|| / ||W^*|| [%]');
xlabel('Tiempo [ms]');
title('ADALINE\_ON\_T: Ideal vs Observador', 'FontWeight', 'bold', ...
      'Interpreter', 'none');
legend('Location', 'best');
grid on; box on;

% Add final ripple annotation
r1t = results_p1.ADALINE_ON_T.Te_ripple;
r2t = results_p2.ADALINE_ON_T.Te_ripple;
text(0.97, 0.92, sprintf('Ripple P1: %.2f%%\nRipple P2: %.2f%%', r1t, r2t), ...
     'Units', 'normalized', 'HorizontalAlignment', 'right', ...
     'FontSize', 8, 'BackgroundColor', [1 1 1 0.7]);

% -- Right subplot: ADALINE_ON_S --
subplot(1, 2, 2);

plot(t_ms, results_p1.ADALINE_ON_S.W_error * 100, ...
     'Color', colors.ADALINE_ON_S, 'LineWidth', 2.0, ...
     'DisplayName', 'Ideal (Fase 1)');
hold on;
plot(t_ms, results_p2.ADALINE_ON_S.W_error * 100, ...
     'Color', colors.ADALINE_ON_S, 'LineWidth', 2.0, ...
     'LineStyle', '--', 'DisplayName', 'Observador (Fase 2)');
yline(5, 'k:', '5%', 'LineWidth', 1, 'HandleVisibility', 'off');

ylabel('||W - W^*|| / ||W^*|| [%]');
xlabel('Tiempo [ms]');
title('ADALINE\_ON\_S: Ideal vs Observador', 'FontWeight', 'bold', ...
      'Interpreter', 'none');
legend('Location', 'best');
grid on; box on;

r1s = results_p1.ADALINE_ON_S.Te_ripple;
r2s = results_p2.ADALINE_ON_S.Te_ripple;
text(0.97, 0.92, sprintf('Ripple P1: %.2f%%\nRipple P2: %.2f%%', r1s, r2s), ...
     'Units', 'normalized', 'HorizontalAlignment', 'right', ...
     'FontSize', 8, 'BackgroundColor', [1 1 1 0.7]);

sgtitle('Figura 4 — Convergencia: LMS Ideal vs LMS con Observador', ...
        'FontSize', 13, 'FontWeight', 'bold');

%% ── Reporte comparativo ──────────────────────────────────────────────────

fprintf('╔══════════════════════════════════════════════════════════════════════╗\n');
fprintf('║              IMPACTO DEL OBSERVADOR EN LOS MÉTODOS ONLINE          ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Método         │ Ripple P1 [%%] │ Ripple P2 [%%] │ Degradación [%%] ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════╣\n');

online_methods = {'ADALINE_ON_T', 'ADALINE_ON_S'};
for idx = 1:length(online_methods)
    m   = online_methods{idx};
    r1  = results_p1.(m).Te_ripple;
    r2  = results_p2.(m).Te_ripple;
    deg = r2 - r1;
    fprintf('║ %-14s │     %6.2f     │     %6.2f     │    %+.2f       ║\n', ...
            m, r1, r2, deg);
end
fprintf('╠══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ (métodos no adaptativos son idénticos en ambas fases)              ║\n');
fprintf('╚══════════════════════════════════════════════════════════════════════╝\n\n');

% Final W_error at end of simulation
fprintf('Error de pesos final ||W-W*||/||W*||:\n');
for idx = 1:length(online_methods)
    m  = online_methods{idx};
    e1 = results_p1.(m).W_error(end) * 100;
    e2 = results_p2.(m).W_error(end) * 100;
    fprintf('  %-16s  Fase1: %.2f%%   Fase2: %.2f%%\n', m, e1, e2);
end
fprintf('\n');
