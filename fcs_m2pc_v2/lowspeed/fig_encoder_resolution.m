function fig_encoder_resolution()
%FIG_ENCODER_RESOLUTION  Resolucion del encoder vs velocidad de rotacion.
%
%   Responde a la pregunta del asesor (reunion 2026-04-20):
%     "comparar la resolucion del sensor con velocidad de rotacion,
%      de tal forma que se sepa si su resolucion es conmensurable
%      con los desplazamientos entre conmutaciones, para saber
%      cuando y como es mas necesario el observador"
%
%   Definiciones:
%     LSB_elec = (2π / PPR) · (P/2)               [rad elec]
%     dθ_e/Ts  = ω_e · Ts = (P/2) · ω_m · Ts      [rad por muestra]
%     Cruce ω* donde dθ_e/Ts = LSB_elec
%
%   Por debajo de ω*: el encoder queda "congelado" por varias muestras
%   (la cuenta no cambia entre lecturas consecutivas). El observador
%   aporta θ interpolado y suaviza el control.
%
%   Por encima de ω*: el encoder actualiza cuenta cada muestra o mas
%   (hasta ~5 veces a velocidad nominal). La resolucion del encoder
%   deja de ser limitante; el observador aporta redundancia.
%
%   Uso:
%       cd fcs_m2pc_v2/lowspeed
%       fig_encoder_resolution()

this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
addpath(genpath(parent_dir));

p   = motor_params();
PPR = 2048;
P   = p.P;
Ts  = p.Ts;

% ── Geometria del encoder ─────────────────────────────────────────────────
LSB_elec_rad = (2*pi / PPR) * (P/2);
LSB_elec_deg = LSB_elec_rad * 180/pi;

% ── Barrido de velocidad ──────────────────────────────────────────────────
rpm_axis = logspace(0, log10(3000), 400);   % 1..3000 rpm
w_m_axis = rpm_axis * 2*pi/60;              % rad/s
dtheta_e_per_Ts_rad = (P/2) * w_m_axis * Ts;
dtheta_e_per_Ts_deg = dtheta_e_per_Ts_rad * 180/pi;

% Muestras Ts entre transiciones consecutivas del encoder
N_Ts_per_LSB = LSB_elec_rad ./ dtheta_e_per_Ts_rad;

% Cruce
rpm_cross = LSB_elec_rad / ((P/2) * Ts * 2*pi/60);

% ── Consola ───────────────────────────────────────────────────────────────
fprintf('\n══ Resolución encoder vs velocidad ═══════════════════════════\n');
fprintf('  Encoder PPR         : %d\n',      PPR);
fprintf('  Polos (pares)       : %d (%d)\n', P, P/2);
fprintf('  Ts controlador      : %.0f us\n', Ts*1e6);
fprintf('  LSB electrico       : %.5f rad  = %.4f°\n', LSB_elec_rad, LSB_elec_deg);
fprintf('  Velocidad de cruce  : %.1f rpm (dθ_e/Ts = LSB)\n', rpm_cross);
fprintf('──────────────────────────────────────────────────────────────\n');
fprintf('      ω [rpm] │ dθ_e/Ts [°] │ LSB/Ts ratio │ Ts por LSB\n');
fprintf('      ────────┼─────────────┼──────────────┼────────────\n');
for rpm_test = [1, 5, 10, 60, 500, rpm_cross, 1000, 3000]
    w_m    = rpm_test * 2*pi/60;
    dth    = (P/2) * w_m * Ts * 180/pi;
    ratio  = dth / LSB_elec_deg;
    N_Ts   = 1 / ratio;
    fprintf('      %7.1f │   %7.4f   │   %8.3f   │  %8.2f\n', ...
            rpm_test, dth, ratio, N_Ts);
end
fprintf('══════════════════════════════════════════════════════════════\n\n');

% ── Figura (dos paneles) ──────────────────────────────────────────────────
fig = figure('Color','w','Position',[80 80 1100 420]);

% Panel 1 — desplazamiento eléctrico por Ts vs velocidad
subplot(1,2,1);
loglog(rpm_axis, dtheta_e_per_Ts_deg, 'LineWidth', 2.2, ...
       'Color', [0 0.447 0.741]);
hold on;
yline(LSB_elec_deg, '--', ...
      sprintf('LSB_{elec} = %.3f°', LSB_elec_deg), ...
      'Color', [0.85 0 0], 'LineWidth', 1.3, ...
      'LabelHorizontalAlignment', 'left');
xline(rpm_cross, ':', sprintf('\\omega^* = %.0f rpm', rpm_cross), ...
      'Color', [0.25 0.25 0.25], 'LineWidth', 1.3, ...
      'LabelVerticalAlignment', 'bottom');

% Marcar velocidades del baseline (5–60 rpm)
ylim_curr = ylim;
patch([5 60 60 5], [ylim_curr(1) ylim_curr(1) ylim_curr(2) ylim_curr(2)], ...
      [0.9 0.9 0.6], 'EdgeColor','none','FaceAlpha', 0.2);
% Re-plot encima del patch
loglog(rpm_axis, dtheta_e_per_Ts_deg, 'LineWidth', 2.2, ...
       'Color', [0 0.447 0.741]);

xlabel('\omega_m  [rpm]');
ylabel('d\theta_e / T_s   [°]');
title('Desplazamiento eléctrico por muestra');
grid on; grid minor;
xlim([rpm_axis(1), rpm_axis(end)]);
legend({'d\theta_e/T_s', 'LSB encoder', '\omega^* cruce', 'Rango baseline 5–60 rpm'}, ...
       'Location', 'northwest', 'FontSize', 8);

% Panel 2 — "muestras congeladas" vs velocidad
subplot(1,2,2);
loglog(rpm_axis, N_Ts_per_LSB, 'LineWidth', 2.2, ...
       'Color', [0.466 0.674 0.188]);
hold on;
yline(1, '--', 'Encoder actualiza cada T_s', ...
      'Color', [0.85 0 0], 'LineWidth', 1.3, ...
      'LabelHorizontalAlignment', 'left');
xline(rpm_cross, ':', sprintf('\\omega^* = %.0f rpm', rpm_cross), ...
      'Color', [0.25 0.25 0.25], 'LineWidth', 1.3, ...
      'LabelVerticalAlignment', 'bottom');

ylim_curr = ylim;
patch([5 60 60 5], [ylim_curr(1) ylim_curr(1) ylim_curr(2) ylim_curr(2)], ...
      [0.9 0.9 0.6], 'EdgeColor','none','FaceAlpha', 0.2);
loglog(rpm_axis, N_Ts_per_LSB, 'LineWidth', 2.2, ...
       'Color', [0.466 0.674 0.188]);

xlabel('\omega_m  [rpm]');
ylabel('T_s entre transiciones del encoder');
title('Tiempo muerto del encoder (en unidades de T_s)');
grid on; grid minor;
xlim([rpm_axis(1), rpm_axis(end)]);
legend({'N_{Ts}/LSB', 'Encoder vivo', '\omega^* cruce', 'Rango baseline 5–60 rpm'}, ...
       'Location', 'northeast', 'FontSize', 8);

sgtitle(sprintf(['Resolución encoder (PPR=%d, P=%d, T_s=%.0f\\mus) vs velocidad ' ...
                 '— cruce a %.0f rpm'], PPR, P, Ts*1e6, rpm_cross), ...
        'FontWeight','bold', 'FontSize', 11);

% ── Guardar ───────────────────────────────────────────────────────────────
saveas(fig, fullfile(this_dir, 'encoder_resolution.png'));
fprintf('Guardado: encoder_resolution.png\n');

end
