function plot_results(results, methods, colors, p, t_ms)
%PLOT_RESULTS  Generates Figures 1–3 for a single simulation run.
%
%   plot_results(results, methods, colors, p, t_ms)
%
%   results : struct with one field per method (output of simulate_all_methods)
%   methods : cell array of method name strings
%   colors  : struct mapping method names to RGB triples
%   p       : params struct (uses p.w_ref, p.P, p.Ts)
%   t_ms    : [1 x N] time vector in milliseconds
%
%   Figure 1: Full-run overview (speed, PI output, torque, current magnitude)
%   Figure 2: Steady-state zoom + bar charts (torque ripple, speed P-P)
%   Figure 3: ADALINE convergence (weight error + instantaneous ripple)

n_methods   = length(methods);
Total_Steps = length(t_ms);
t_final_ms  = t_ms(end);

% ── Figure 1: Full run overview ─────────────────────────────────────────
figure('Color', 'w', 'Position', [50, 50, 1400, 900]);

subplot(4, 1, 1);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_ms, results.(m).w_m, 'Color', colors.(m), ...
         'LineWidth', 1.3, 'DisplayName', m);
    hold on;
end
yline(p.w_ref, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(t_final_ms/2, 'r:', 'Load step', 'LineWidth', 1, ...
      'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
ylabel('\omega_m [rad/s]', 'Interpreter', 'tex');
title('Velocidad', 'FontWeight', 'bold');
legend('Location', 'best', 'NumColumns', 3, 'FontSize', 7);
grid on; box on;

subplot(4, 1, 2);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_ms, results.(m).T_ref, 'Color', colors.(m), 'LineWidth', 1);
    hold on;
end
xline(t_final_ms/2, 'r:', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('T_{ref} [Nm]', 'Interpreter', 'tex');
title('Salida PI', 'FontWeight', 'bold');
grid on; box on;

subplot(4, 1, 3);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_ms, results.(m).Te, 'Color', colors.(m), 'LineWidth', 0.8);
    hold on;
end
plot(t_ms, results.(methods{1}).T_load_actual, 'k--', 'LineWidth', 1.5, ...
     'DisplayName', 'T_{load}');
xline(t_final_ms/2, 'r:', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('T_e [Nm]', 'Interpreter', 'tex');
title('Torque Electromagnético', 'FontWeight', 'bold');
grid on; box on;

subplot(4, 1, 4);
for idx = 1:n_methods
    m = methods{idx};
    i_mag = sqrt(results.(m).i(1,:).^2 + results.(m).i(2,:).^2);
    plot(t_ms, i_mag, 'Color', colors.(m), 'LineWidth', 0.8);
    hold on;
end
xline(t_final_ms/2, 'r:', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('|i| [A]');
xlabel('Tiempo [ms]');
title('Magnitud de Corriente', 'FontWeight', 'bold');
grid on; box on;

sgtitle('Comparación Completa: 6 Modelos de BEMF con FCS-M2PC', ...
        'FontSize', 14, 'FontWeight', 'bold');

% ── Figure 2: Steady-state zoom + bar charts ────────────────────────────
figure('Color', 'w', 'Position', [100, 100, 1400, 700]);

w_final      = abs(results.(methods{1}).w_m(end));
if w_final < 1, w_final = p.w_ref; end
samples_2rev = round(2 * 2*pi / (w_final * p.Ts * (p.P/2)));
samples_2rev = min(samples_2rev, Total_Steps - 1);
z_start      = max(1, Total_Steps - samples_2rev);
t_zoom       = t_ms(z_start:end);

subplot(2, 2, 1);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_zoom, results.(m).Te(z_start:end), 'Color', colors.(m), ...
         'LineWidth', 1.3, 'DisplayName', m);
    hold on;
end
ylabel('T_e [Nm]', 'Interpreter', 'tex');
title('Torque — Zoom (2 rev)', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 7);
grid on; box on;

subplot(2, 2, 2);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_zoom, results.(m).w_m(z_start:end), 'Color', colors.(m), ...
         'LineWidth', 1.3, 'DisplayName', m);
    hold on;
end
yline(p.w_ref, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('\omega_m [rad/s]', 'Interpreter', 'tex');
title('Velocidad — Zoom (2 rev)', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 7);
grid on; box on;

subplot(2, 2, 3);
ripple_vals = cellfun(@(m) results.(m).Te_ripple, methods);
b = bar(ripple_vals);
b.FaceColor = 'flat';
for i = 1:n_methods
    b.CData(i,:) = colors.(methods{i});
end
set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 30, 'FontSize', 8);
ylabel('Torque Ripple [%]');
title('Torque Ripple (STD/mean)', 'FontWeight', 'bold');
grid on; box on;
for i = 1:n_methods
    text(i, ripple_vals(i), sprintf('%.2f%%', ripple_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontWeight', 'bold', 'FontSize', 8);
end

subplot(2, 2, 4);
w_pp_vals = cellfun(@(m) results.(m).w_pp, methods);
b = bar(w_pp_vals);
b.FaceColor = 'flat';
for i = 1:n_methods
    b.CData(i,:) = colors.(methods{i});
end
set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 30, 'FontSize', 8);
ylabel('\omega P-P [rad/s]', 'Interpreter', 'tex');
title('Velocity Ripple P-P', 'FontWeight', 'bold');
grid on; box on;
for i = 1:n_methods
    text(i, w_pp_vals(i), sprintf('%.4f', w_pp_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontWeight', 'bold', 'FontSize', 8);
end

sgtitle('Estado Estable: 6 Modelos de BEMF', ...
        'FontSize', 14, 'FontWeight', 'bold');

% ── Figure 3: ADALINE convergence ───────────────────────────────────────
if ~isfield(results.ADALINE_ON_T, 'W_error')
    return;
end

figure('Color', 'w', 'Position', [150, 150, 1200, 400]);

subplot(1, 2, 1);
plot(t_ms, results.ADALINE_ON_T.W_error * 100, ...
     'Color', colors.ADALINE_ON_T, 'LineWidth', 1.8, ...
     'DisplayName', 'Online desde TRAP');
hold on;
plot(t_ms, results.ADALINE_ON_S.W_error * 100, ...
     'Color', colors.ADALINE_ON_S, 'LineWidth', 1.8, ...
     'DisplayName', 'Online desde SIN');
yline(5, 'k--', '5%', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('||W - W^*|| / ||W^*|| [%]');
xlabel('Tiempo [ms]');
title('Convergencia de Pesos hacia Óptimo Offline', 'FontWeight', 'bold');
legend('Location', 'best');
grid on; box on;

subplot(1, 2, 2);
win = round(2*pi / (p.w_ref * (p.P/2) * p.Ts));   % samples per electrical rev
for idx = 1:n_methods
    m = methods{idx};
    Te_roll_std  = movstd(results.(m).Te, win);
    Te_roll_mean = movmean(results.(m).Te, win);
    ripple_inst  = 100 * Te_roll_std ./ max(abs(Te_roll_mean), 1e-6);
    plot(t_ms, ripple_inst, 'Color', colors.(m), ...
         'LineWidth', 1, 'DisplayName', m);
    hold on;
end
ylabel('Torque Ripple Instantáneo [%]');
xlabel('Tiempo [ms]');
title('Evolución del Ripple (ventana = 1 rev eléctrica)', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 7);
grid on; box on;

sgtitle('Convergencia del ADALINE Online', ...
        'FontSize', 14, 'FontWeight', 'bold');

% ── Figure 4: Coronado-style αβ current tracking ────────────────────────
% Last ~100 ms of simulation (steady-state after load step), both iα and iβ
% per method, reference (black dashed) vs measured (method color).
samples_win = min(round(0.10 / p.Ts), Total_Steps - 1);
s0          = max(1, Total_Steps - samples_win);
t_win       = t_ms(s0:end);

figure('Color', 'w', 'Position', [200, 50, 700, 140*n_methods]);

for idx = 1:n_methods
    m  = methods{idx};
    ia = results.(m).i(1, s0:end);
    ib = results.(m).i(2, s0:end);
    ia_ref = results.(m).i_ref(1, s0:end);
    ib_ref = results.(m).i_ref(2, s0:end);

    subplot(n_methods, 1, idx);
    plot(t_win, ia_ref, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    hold on;
    plot(t_win, ib_ref, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    plot(t_win, ia, 'Color', colors.(m), 'LineWidth', 0.9, ...
         'DisplayName', [m ' i_\alpha']);
    plot(t_win, ib, 'Color', colors.(m) * 0.6, 'LineWidth', 0.9, ...
         'DisplayName', [m ' i_\beta']);
    ylabel('i_{\alpha\beta} [A]', 'Interpreter', 'tex', 'FontSize', 8);
    title(sprintf('%s  |  RMSE α=%.4f A   β=%.4f A   NMSE α=%.4f   β=%.4f', ...
          m, results.(m).rmse_alpha, results.(m).rmse_beta, ...
          results.(m).nmse_alpha,    results.(m).nmse_beta), ...
          'FontSize', 8, 'FontWeight', 'bold');
    legend('Location', 'northeast', 'FontSize', 7, 'NumColumns', 2);
    grid on; box on;
    if idx < n_methods
        set(gca, 'XTickLabel', []);
    else
        xlabel('Tiempo [ms]');
    end
end

sgtitle('Grupo 1 — Seguimiento de Corriente \alpha\beta (estado estable, 100 ms)', ...
        'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'tex');

% ── Console RMSE/NMSE table ──────────────────────────────────────────────
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║         GRUPO 1 — Tracking de Corriente (estado estable)       ║\n');
fprintf('╠══════════════╦══════════════╦══════════════╦══════════╦══════════╣\n');
fprintf('║ Método       ║  RMSE α [A]  ║  RMSE β [A]  ║  NMSE α  ║  NMSE β  ║\n');
fprintf('╠══════════════╬══════════════╬══════════════╬══════════╬══════════╣\n');
for idx = 1:n_methods
    m = methods{idx};
    fprintf('║ %-12s ║   %8.4f   ║   %8.4f   ║  %6.4f  ║  %6.4f  ║\n', ...
            m, results.(m).rmse_alpha, results.(m).rmse_beta, ...
            results.(m).nmse_alpha,    results.(m).nmse_beta);
end
fprintf('╚══════════════╩══════════════╩══════════════╩══════════╩══════════╝\n\n');

end
