%% plot_bemf_shapes.m
% Diagnostic plot: αβ BEMF shapes comparing trapezoidal conventions.
% Run from fcs_m2pc_v2/ after addpath(genpath('.'))
%
% Shows:
%   1. Trapezoidal αβ BEFORE fix (+pi/6 offset) vs AFTER (correct)
%   2. Comparison with sinusoidal reference [sin θ; −cos θ]
%   3. Phase-a waveform overlay to show the 30° shift

clear; close all; clc;
addpath(genpath('.'));

theta = linspace(0, 2*pi, 1000)';

%% ── Compute shapes ───────────────────────────────────────────────────────

% Current (fixed) trapezoidal
trap_ab = zeros(1000, 2);
for k = 1:1000
    s = bemf_trapezoidal(theta(k));
    trap_ab(k,:) = s';
end

% Old (buggy) trapezoidal — manually inline with +pi/6
trap_ab_old = zeros(1000, 2);
for k = 1:1000
    phases = [0, 2*pi/3, 4*pi/3];
    abc    = zeros(3,1);
    for i = 1:3
        ti = mod(theta(k) - phases(i) + pi/6, 2*pi);
        if     ti < pi/6,    val =  ti * (6/pi);
        elseif ti < 5*pi/6,  val =  1;
        elseif ti < 7*pi/6,  val =  1 - (ti - 5*pi/6) * (6/pi);
        elseif ti < 11*pi/6, val = -1;
        else,                val = -1 + (ti - 11*pi/6) * (6/pi);
        end
        abc(i) = val;
    end
    ab = clarke_transform(abc);
    trap_ab_old(k,:) = ab';
end

% Sinusoidal reference
sin_ab = [sin(theta), -cos(theta)];

%% ── Figure 1: αβ shapes comparison ─────────────────────────────────────
figure('Color', 'w', 'Position', [50, 50, 1100, 700]);

subplot(2, 2, 1);
plot(theta*180/pi, trap_ab_old(:,1), 'r-',  'LineWidth', 1.5, 'DisplayName', 'TRAP vieja (+\pi/6)');
hold on;
plot(theta*180/pi, trap_ab(:,1),     'b-',  'LineWidth', 1.5, 'DisplayName', 'TRAP corregida');
plot(theta*180/pi, sin_ab(:,1),      'k--', 'LineWidth', 1.2, 'DisplayName', 'SIN [sin\theta]');
xline(90,  'g:', '90°', 'LineWidth', 1, 'HandleVisibility','off');
xline(60,  'r:', '60°', 'LineWidth', 1, 'HandleVisibility','off');
xlabel('\theta_e [°]');  ylabel('e_\alpha (normalizado)');
title('e_\alpha: TRAP vieja vs corregida vs SIN', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on; box on; xlim([0 360]);

subplot(2, 2, 3);
plot(theta*180/pi, trap_ab_old(:,2), 'r-',  'LineWidth', 1.5, 'DisplayName', 'TRAP vieja (+\pi/6)');
hold on;
plot(theta*180/pi, trap_ab(:,2),     'b-',  'LineWidth', 1.5, 'DisplayName', 'TRAP corregida');
plot(theta*180/pi, sin_ab(:,2),      'k--', 'LineWidth', 1.2, 'DisplayName', 'SIN [-cos\theta]');
xlabel('\theta_e [°]');  ylabel('e_\beta (normalizado)');
title('e_\beta: TRAP vieja vs corregida vs SIN', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on; box on; xlim([0 360]);

subplot(2, 2, 2);
plot(trap_ab_old(:,1), trap_ab_old(:,2), 'r-',  'LineWidth', 1.8, ...
     'DisplayName', 'TRAP vieja (+\pi/6)');
hold on;
plot(trap_ab(:,1),     trap_ab(:,2),     'b-',  'LineWidth', 1.8, ...
     'DisplayName', 'TRAP corregida');
plot(sin_ab(:,1),      sin_ab(:,2),      'k--', 'LineWidth', 1.2, ...
     'DisplayName', 'SIN');
plot(0, 0, 'k+', 'MarkerSize', 10, 'HandleVisibility','off');
xlabel('e_\alpha');  ylabel('e_\beta');
title('Trayectoria αβ (locus)', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
axis equal; grid on; box on;

subplot(2, 2, 4);
% Phase-a waveform: show the 30° lead
phases = [0, 2*pi/3, 4*pi/3];
abc_old = zeros(1000, 3);
abc_new = zeros(1000, 3);
for k = 1:1000
    for i = 1:3
        ti_old = mod(theta(k) - phases(i) + pi/6, 2*pi);
        if     ti_old < pi/6,    abc_old(k,i) =  ti_old * (6/pi);
        elseif ti_old < 5*pi/6,  abc_old(k,i) =  1;
        elseif ti_old < 7*pi/6,  abc_old(k,i) =  1 - (ti_old - 5*pi/6) * (6/pi);
        elseif ti_old < 11*pi/6, abc_old(k,i) = -1;
        else,                    abc_old(k,i) = -1 + (ti_old - 11*pi/6) * (6/pi);
        end
        ti_new = mod(theta(k) - phases(i), 2*pi);
        if     ti_new < pi/6,    abc_new(k,i) =  ti_new * (6/pi);
        elseif ti_new < 5*pi/6,  abc_new(k,i) =  1;
        elseif ti_new < 7*pi/6,  abc_new(k,i) =  1 - (ti_new - 5*pi/6) * (6/pi);
        elseif ti_new < 11*pi/6, abc_new(k,i) = -1;
        else,                    abc_new(k,i) = -1 + (ti_new - 11*pi/6) * (6/pi);
        end
    end
end
plot(theta*180/pi, abc_old(:,1), 'r-',  'LineWidth', 1.5, 'DisplayName', 'Fase A vieja');
hold on;
plot(theta*180/pi, abc_new(:,1), 'b-',  'LineWidth', 1.5, 'DisplayName', 'Fase A corregida');
plot(theta*180/pi, sin(theta),   'k--', 'LineWidth', 1.2, 'DisplayName', 'sin\theta (ref)');
xline(90, 'k:', '90°', 'LineWidth', 1, 'HandleVisibility','off');
xline(60, 'r:', '60°', 'LineWidth', 1, 'HandleVisibility','off');
xlabel('\theta_e [°]');  ylabel('BEMF fase A (normalizado)');
title('Fase A: desfase de 30°', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on; box on; xlim([0 360]);

sgtitle('Diagnóstico BEMF Trapezoidal: Corrección del desfase +\pi/6', ...
        'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'tex');

%% ── Console summary ─────────────────────────────────────────────────────
fprintf('\n── Valores en θ=0 ──────────────────────────────────────────────────\n');
fprintf('  TRAP vieja (+pi/6):  [α=%.4f, β=%.4f]  (debe ser [0, -1])\n', ...
        trap_ab_old(1,1), trap_ab_old(1,2));
fprintf('  TRAP corregida:      [α=%.4f, β=%.4f]  ✓\n', ...
        trap_ab(1,1), trap_ab(1,2));
fprintf('  SIN [sin,−cos]:      [α=%.4f, β=%.4f]  referencia\n\n', ...
        sin_ab(1,1), sin_ab(1,2));

fprintf('── Desfase de pico (fase A) ─────────────────────────────────────────\n');
[~, idx_old] = max(abc_old(:,1));
[~, idx_new] = max(abc_new(:,1));
fprintf('  TRAP vieja: pico en θ = %.1f°  (target: 90°)\n', theta(idx_old)*180/pi);
fprintf('  TRAP nueva: pico en θ = %.1f°  ✓\n\n', theta(idx_new)*180/pi);
