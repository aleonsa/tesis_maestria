function sim_lowspeed_snr()
%SIM_LOWSPEED_SNR  Demonstrates BEMF observer SNR collapse at low speed.
%
%   Speed profile: 35 rpm → 0 rpm (linear ramp, 1.5 s)
%
%   The algebraic observer:
%       e_obs(k) = u(k-1) - R·i(k-1) - (L/Ts)·[i(k) - i(k-1)]
%   amplifies current measurement noise by the factor L/Ts = 41 V/A.
%   The BEMF signal itself scales as Ke·ω → 0. Below ω_crit the noise
%   floor permanently dominates and θ_e cannot be recovered.
%
%   Run from fcs_m2pc_v2/ :
%       addpath(genpath('.')); cd lowspeed; sim_lowspeed_snr()
%   or from anywhere:
%       run('/path/to/fcs_m2pc_v2/lowspeed/sim_lowspeed_snr.m')

% ── Add parent directories to path ────────────────────────────────────────
this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
addpath(genpath(parent_dir));

p   = motor_params();
rng(42);  % reproducibility

% ── Speed ramp ────────────────────────────────────────────────────────────
t_final   = 1.5;                         % [s]
Nsim      = round(t_final / p.Ts);
t         = (0:Nsim-1)' * p.Ts;          % [s]

rpm_start = 35;
rpm_end   = 0;
w_m = linspace(rpm_start, rpm_end, Nsim)' * (2*pi/60);  % mech [rad/s]
w_e = (p.P/2) * w_m;                                    % elec [rad/s]

% Integrate electrical angle from zero
theta_e = cumsum(w_e) * p.Ts;   % [rad], monotonically increasing

% ── True BEMF in αβ ───────────────────────────────────────────────────────
e_true = zeros(Nsim, 2);
for k = 1:Nsim
    shape       = bemf_trapezoidal(theta_e(k));    % [2x1]
    e_true(k,:) = (p.Ke * w_m(k) * shape)';
end

% ── Synthetic current (d=0, perfect tracking) ─────────────────────────────
%   i_αβ = I_peak · [cos(θ_e), −sin(θ_e)]
%   This is a reasonable steady-state current for MTPA at d=0.
I_peak = 2.0;   % [A]
i_true = I_peak * [cos(theta_e), -sin(theta_e)];

% Voltage from plant equation:  u = e + R·i + L·(di/dt)
di_dt  = [zeros(1,2); diff(i_true, 1, 1)] / p.Ts;
u_true = e_true + p.R * i_true + p.L * di_dt;

% ── Noisy current measurement (Gaussian + 12-bit ADC) ─────────────────────
noise   = p.sigma_noise * randn(Nsim, 2);
i_noisy = i_true + noise;
for k = 1:Nsim
    i_noisy(k,:) = adc_quantize(i_noisy(k,:)', p.adc_bits, p.adc_range)';
end

% ── Algebraic BEMF observer ───────────────────────────────────────────────
e_obs = zeros(Nsim, 2);
for k = 2:Nsim
    e_obs(k,:) = u_true(k-1,:) ...
                 - p.R  * i_noisy(k-1,:) ...
                 - (p.L/p.Ts) * (i_noisy(k,:) - i_noisy(k-1,:));
end

% Moving-average filter on e_obs (100-step window ≈ 5 ms)
win          = 100;
e_obs_filt   = movmean(e_obs, win, 1);

% ── SNR analysis ──────────────────────────────────────────────────────────
e_true_mag     = sqrt(sum(e_true.^2, 2));            % true BEMF magnitude
noise_floor_th = (p.L/p.Ts) * sqrt(2) * p.sigma_noise;  % theoretical noise RMS [V]
SNR_linear     = e_true_mag / noise_floor_th;
SNR_dB         = 20 * log10(max(SNR_linear, 1e-4));

% ── Angle estimation ─────────────────────────────────────────────────────
% From observer: θ_obs = atan2(−e_β, e_α)  (standard sensorless convention)
theta_obs      = atan2(-e_obs(:,2),      e_obs(:,1));
theta_obs_filt = atan2(-e_obs_filt(:,2), e_obs_filt(:,1));

% Angle error in [−π, π]
theta_err_raw  = angle_diff(theta_obs,      theta_e);
theta_err_filt = angle_diff(theta_obs_filt, theta_e);

% ── Print summary ─────────────────────────────────────────────────────────
w_crit_rads = noise_floor_th / p.Ke;
w_crit_rpm  = w_crit_rads * 60/(2*pi);

fprintf('\n══ BEMF Observer SNR Analysis ══════════════════════════════════\n');
fprintf('  L/Ts amplification factor :  %.1f  V/A\n',      p.L/p.Ts);
fprintf('  Current noise (1σ)        :  %.0f  mA\n',       p.sigma_noise*1e3);
fprintf('  Theoretical noise floor   :  %.2f V RMS\n',     noise_floor_th);
fprintf('  ─────────────────────────────────────────────────────────────\n');
fprintf('  Critical speed (SNR=1)    :  %.1f rad/s  =  %.0f rpm\n', ...
        w_crit_rads, w_crit_rpm);
fprintf('  ─────────────────────────────────────────────────────────────\n');
fprintf('  At %.0f rpm  →  |e_true| = %.3f V,  SNR = %.2f dB\n', ...
        rpm_start, e_true_mag(1), SNR_dB(1));
fprintf('  At  5 rpm  →  |e_true| = %.3f V,  SNR = %.2f dB\n', ...
        p.Ke * 5*(2*pi/60), 20*log10(p.Ke*5*(2*pi/60)/noise_floor_th));
fprintf('  At  1 rpm  →  |e_true| = %.4f V,  SNR = %.2f dB\n', ...
        p.Ke * 1*(2*pi/60), 20*log10(p.Ke*1*(2*pi/60)/noise_floor_th));
fprintf('══════════════════════════════════════════════════════════════════\n\n');

% ── Figure 1: Main demonstration ──────────────────────────────────────────
t_ms   = t * 1e3;
w_rpm  = w_m * 60/(2*pi);
n_zoom = min(Nsim, round(0.1/p.Ts));   % first 100 ms

fig1 = figure('Name','BEMF Observer — SNR Demo', ...
              'Position',[80 80 1060 780], 'Color','w');

% Panel 1 — Speed profile
ax1 = subplot(3,2,1);
plot(t_ms, w_rpm, 'b', 'LineWidth', 1.8);
hold on;
yline(0, 'k--', 'LineWidth', 0.8);
xlabel('Time [ms]'); ylabel('\omega_m  [rpm]');
title('(a)  Speed profile'); grid on;
xlim([0 t_ms(end)]); ylim([-2 40]);

% Panel 2 — BEMF magnitude vs noise floor
ax2 = subplot(3,2,2);
plot(t_ms, e_true_mag, 'b', 'LineWidth', 1.8, 'DisplayName','|e_{true}|'); hold on;
yline(noise_floor_th,   'r--', 'LineWidth', 1.5);
yline(3*noise_floor_th, 'r:',  'LineWidth', 1.2);
text(t_ms(end)*0.55, noise_floor_th*1.15, ...
     sprintf('\\sigma_{obs} = %.1f V', noise_floor_th), ...
     'Color','r', 'FontSize',9);
text(t_ms(end)*0.55, 3*noise_floor_th*1.08, ...
     sprintf('3\\sigma_{obs} = %.1f V', 3*noise_floor_th), ...
     'Color','r', 'FontSize',9);
legend('|e_{true}|', 'Location','northeast', 'FontSize',8);
xlabel('Time [ms]'); ylabel('Voltage [V]');
title('(b)  BEMF signal vs noise floor'); grid on;
xlim([0 t_ms(end)]); ylim([0 10]);

% Panel 3 — SNR in dB
ax3 = subplot(3,2,3);
plot(t_ms, SNR_dB, 'k', 'LineWidth', 1.8); hold on;
yline(0, 'r--', 'LineWidth', 1.5);
text(t_ms(end)*0.05, 1.5, 'SNR = 0 dB  (signal = noise)', ...
     'Color','r', 'FontSize', 8);
fill([0 t_ms(end) t_ms(end) 0], [-40 -40 0 0], ...
     'r', 'FaceAlpha',0.07, 'EdgeColor','none');
xlabel('Time [ms]'); ylabel('SNR [dB]');
title('(c)  Instantaneous SNR'); grid on;
xlim([0 t_ms(end)]); ylim([-40 5]);

% Panel 4 — e_obs vs e_true (alpha component, first 100 ms)
ax4 = subplot(3,2,4);
plot(t_ms(1:n_zoom), e_true(1:n_zoom,1), 'b', 'LineWidth',1.5, ...
     'DisplayName','e_{\alpha} true'); hold on;
plot(t_ms(1:n_zoom), e_obs(1:n_zoom,1), ...
     'Color',[0.85 0.2 0.2 0.35], 'LineWidth',0.6, ...
     'DisplayName','e_{\alpha} raw obs');
plot(t_ms(1:n_zoom), e_obs_filt(1:n_zoom,1), 'r', 'LineWidth',1.5, ...
     'DisplayName','e_{\alpha} filtered');
legend('Location','northeast','FontSize',8);
xlabel('Time [ms]'); ylabel('e_{\alpha}  [V]');
title('(d)  BEMF \alpha  —  first 100 ms'); grid on;
xlim([t_ms(1) t_ms(n_zoom)]);

% Panel 5 — Angle error (raw observer)
ax5 = subplot(3,2,5);
plot(t_ms, rad2deg(theta_err_raw), 'Color',[0.85 0.2 0.2 0.5], ...
     'LineWidth',0.6, 'DisplayName','raw'); hold on;
plot(t_ms, rad2deg(theta_err_filt), 'r', 'LineWidth',1.5, ...
     'DisplayName','filtered');
yline(0,'k--','LineWidth',0.8);
yline( 10,'b:','LineWidth',1.0);
yline(-10,'b:','LineWidth',1.0);
text(t_ms(end)*0.6, 12, '±10° target', 'Color','b','FontSize',8);
legend('Location','northwest','FontSize',8);
xlabel('Time [ms]'); ylabel('\Delta\theta_e  [°]');
title('(e)  Angle estimation error'); grid on;
xlim([0 t_ms(end)]); ylim([-200 200]);

% Panel 6 — RMS angle error vs speed (running window)
win_rms = round(0.05/p.Ts);   % 50 ms window
rms_err = sqrt(movmean(theta_err_filt.^2, win_rms));
ax6 = subplot(3,2,6);
plot(w_rpm, rad2deg(rms_err), 'r', 'LineWidth',1.5); hold on;
yline(10, 'b--', 'LineWidth',1.2);
text(20, 11.5, '10° threshold', 'Color','b','FontSize',8);
xlabel('\omega_m  [rpm]'); ylabel('RMS  \Delta\theta_e  [°]');
title('(f)  Angle RMS error vs speed'); grid on;
xlim([0 rpm_start]); set(gca,'XDir','reverse');

sgtitle('BEMF Algebraic Observer — SNR Collapse at Low Speed', ...
        'FontWeight','bold','FontSize',13);

% ── Figure 2: Theoretical SNR vs speed (context) ──────────────────────────
fig2 = figure('Name','Theoretical SNR vs Speed', ...
              'Position',[1160 80 580 420], 'Color','w');

rpm_axis  = linspace(0, 300, 2000);
w_axis    = rpm_axis * (2*pi/60);
snr_curve = 20 * log10(max(p.Ke * w_axis / noise_floor_th, 1e-4));

plot(rpm_axis, snr_curve, 'b', 'LineWidth', 2); hold on;
yline(0, 'r--', 'LineWidth', 1.5);
xline(w_crit_rpm, 'r:', 'LineWidth', 1.5);

% Shade the simulation range (0 – 35 rpm)
xlims = [0 35];
fill([xlims(1) xlims(2) xlims(2) xlims(1)], [-60 -60 30 30], ...
     'r', 'FaceAlpha',0.10, 'EdgeColor','none');
text(17, -50, sprintf('Simulation\nrange\n(0–35 rpm)'), ...
     'Color',[0.7 0 0],'FontSize',9,'HorizontalAlignment','center');
text(w_crit_rpm+3, -8, sprintf('\\omega_{crit} = %.0f rpm', w_crit_rpm), ...
     'Color','r','FontSize',10);

xlabel('Speed [rpm]'); ylabel('SNR [dB]');
title({'Theoretical BEMF Observer SNR vs Speed', ...
       sprintf('Noise floor = (L/T_s)·\\surd2·\\sigma_i = %.2f V RMS', ...
               noise_floor_th)});
grid on;
ylim([-60 30]); xlim([0 300]);
legend('SNR(\omega)', 'SNR = 0 dB (signal = noise)', 'FontSize',9, ...
       'Location','southeast');

% ── Save ──────────────────────────────────────────────────────────────────
saveas(fig1, fullfile(this_dir, 'lowspeed_snr_demo.png'));
saveas(fig2, fullfile(this_dir, 'lowspeed_snr_theory.png'));
fprintf('Saved:  lowspeed_snr_demo.png\n');
fprintf('Saved:  lowspeed_snr_theory.png\n');

end   % sim_lowspeed_snr

% ── Local helper ──────────────────────────────────────────────────────────
function d = angle_diff(a, b)
%ANGLE_DIFF  Wrapped angular difference in (−π, π].  No toolbox required.
d = mod(a - b + pi, 2*pi) - pi;
end
