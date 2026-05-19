function sim_lowspeed_rhodivs_sweep()
%SIM_LOWSPEED_RHODIVS_SWEEP  Diagnostico — ¿el 16% de ripple es granularidad
%   del simulador o piso intrinseco?
%
%   Barrido de rho_divs a velocidad fija (5 rpm, 1 periodo electrico).
%
%     * Si el ripple CAE monotonamente con rho_divs → es cuantizacion
%       temporal del simulador. El M2PC continuo no tiene este rizo.
%     * Si el ripple SE ESTANCA en un piso → hay rizo intrinseco
%       (shape mismatch, Euler, etc) que no se resuelve refinando
%       la busqueda de duty.
%
%   Referencia hardware (F28379D):
%     * ePWM 16-bit, PWM @ 33.3 kHz (Ts_pwm = 30 μs)
%     * Resolucion de duty ≈ 6000 pasos
%     * Equivale a rho_divs ≈ 80 en este simulador
%
%   Uso:
%       cd fcs_m2pc_v2/lowspeed
%       sim_lowspeed_rhodivs_sweep()

this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
addpath(genpath(parent_dir));

p = motor_params();
rng(42);

% ── Config del diagnostico ────────────────────────────────────────────────
rpm           = 5;                   % velocidad fija [rpm]
N_cycles      = 1;                   % 1 periodo electrico (contener costo)
enc_PPR       = 2048;
T_load        = 0.3;                 % [Nm]
rho_divs_list = [10, 20, 40, 80, 120];

fprintf('\n══ Diagnostico — Ripple vs rho_divs @ %d rpm ═════════════════\n', rpm);
fprintf('  Hardware F28379D: PWM 33.3 kHz, ePWM 16-bit\n');
fprintf('  Resolucion de duty ≈ 6000 pasos → rho_divs ≈ 80\n');
fprintf('  Barrido: [%s]\n', num2str(rho_divs_list));
fprintf('══════════════════════════════════════════════════════════════\n\n');

% ── Setup comun ───────────────────────────────────────────────────────────
Vdc    = 200;
v_mag  = 2/3 * Vdc;
s3     = sqrt(3);
V_ab   = [[0;0], [v_mag;0], [v_mag/2; v_mag*s3/2], ...
          [-v_mag/2; v_mag*s3/2], [-v_mag;0], ...
          [-v_mag/2; -v_mag*s3/2], [v_mag/2; -v_mag*s3/2], [0;0]];

if ~isfile(fullfile(parent_dir, 'data', 'bemf_lut.mat'))
    error('Falta bemf_lut.mat en data/. Correr generate_lut.py');
end
lut = load(fullfile(parent_dir, 'data', 'bemf_lut.mat'));

w_m_ref = rpm * 2*pi/60;
w_e_ref = (p.P/2) * w_m_ref;
T_elec  = 2*pi / w_e_ref;
t_final = N_cycles * T_elec;
N       = round(t_final / p.Ts);

% ── Barrido ───────────────────────────────────────────────────────────────
n_rho       = length(rho_divs_list);
res_ripple  = zeros(1, n_rho);
res_Te_pp   = zeros(1, n_rho);
res_Te_mean = zeros(1, n_rho);
res_wall    = zeros(1, n_rho);

for r = 1:n_rho
    rho_divs = rho_divs_list(r);
    fprintf('► rho_divs = %4d ...', rho_divs);
    tic;

    i_ab    = [0; 0];
    w_m     = w_m_ref;
    theta_e = 0;
    log_Te  = zeros(1, N);

    for k = 1:N
        theta_meas = quantize_encoder(theta_e, enc_PPR, p.P);

        s_real = [interp1(lut.lut_theta, lut.lut_alpha_real, ...
                          mod(theta_e, 2*pi), 'linear', 'extrap');
                  interp1(lut.lut_theta, lut.lut_beta_real, ...
                          mod(theta_e, 2*pi), 'linear', 'extrap')];
        e_real = p.Ke * w_m * s_real;

        s_model = [interp1(lut.lut_theta, lut.lut_alpha, ...
                           mod(theta_meas, 2*pi), 'linear', 'extrap');
                   interp1(lut.lut_theta, lut.lut_beta, ...
                           mod(theta_meas, 2*pi), 'linear', 'extrap')];
        e_model = p.Ke * w_m * s_model;

        T_ref   = T_load;
        norm_sq = s_model' * s_model;
        if norm_sq > 1e-6
            i_ref = (T_ref / (p.Kt * norm_sq)) * s_model;
        else
            i_ref = [0; 0];
        end

        ang_s  = mod(atan2(s_model(2), s_model(1)), 2*pi);
        sector = min(floor(ang_s / (pi/3)) + 1, 6);
        vec_pairs = [1,2; 2,3; 3,4; 4,5; 5,6; 6,1];
        av = vec_pairs(sector, :);
        u1 = V_ab(:, av(1)+1);
        u2 = V_ab(:, av(2)+1);
        u0 = [0; 0];

        f1 = (1/p.L) * (u1 - e_model - p.R * i_ab);
        f2 = (1/p.L) * (u2 - e_model - p.R * i_ab);
        f0 = (1/p.L) * (u0 - e_model - p.R * i_ab);

        min_cost = inf;
        t1_opt = 0; t2_opt = 0;
        for nn = 0:rho_divs
            t1 = (nn/rho_divs) * p.Ts;
            for mm = 0:(rho_divs - nn)
                t2 = (mm/rho_divs) * p.Ts;
                t0 = p.Ts - t1 - t2;
                i_pred = i_ab + f1*t1 + f2*t2 + f0*t0;
                cost   = sum((i_ref - i_pred).^2);
                if cost < min_cost
                    min_cost = cost;
                    t1_opt   = t1;
                    t2_opt   = t2;
                end
            end
        end
        u_applied = (u1*t1_opt + u2*t2_opt + u0*(p.Ts - t1_opt - t2_opt)) / p.Ts;

        di_dt = (1/p.L) * (u_applied - e_real - p.R * i_ab);
        i_ab  = i_ab + di_dt * p.Ts;

        if abs(w_m) > 0.5
            Te = (3/2) * (e_real' * i_ab) / w_m;
        else
            nr = norm(s_real);
            if nr > 1e-6
                Te = p.Kt * (s_real' * i_ab) / nr;
            else
                Te = 0;
            end
        end

        theta_e   = theta_e + w_m * p.Ts * (p.P/2);
        log_Te(k) = Te;
    end

    idx             = round(N*0.2):N;
    Te_mean         = mean(log_Te(idx));
    Te_std          = std(log_Te(idx));
    res_ripple(r)   = 100 * Te_std / max(abs(Te_mean), 1e-6);
    res_Te_pp(r)    = max(log_Te(idx)) - min(log_Te(idx));
    res_Te_mean(r)  = Te_mean;
    res_wall(r)     = toc;

    fprintf(' ✓ %6.2fs | Te=%.3f Nm | Ripple=%5.2f%% | Te_pp=%.4f\n', ...
            res_wall(r), Te_mean, res_ripple(r), res_Te_pp(r));
end

fprintf('\n══ Resumen ripple vs rho_divs (5 rpm) ════════════════════════\n');
fprintf('  rho_divs │ Te_mean [Nm] │ Ripple [%%] │ Te_pp [Nm] │ Wall [s]\n');
fprintf('  ─────────┼──────────────┼────────────┼────────────┼─────────\n');
for r = 1:n_rho
    fprintf('  %7d  │    %6.4f    │    %5.2f   │   %6.4f   │  %6.2f\n', ...
            rho_divs_list(r), res_Te_mean(r), res_ripple(r), ...
            res_Te_pp(r), res_wall(r));
end
fprintf('══════════════════════════════════════════════════════════════\n\n');

% ── Figura ────────────────────────────────────────────────────────────────
fig = figure('Color','w','Position',[80 80 1000 380]);

subplot(1,2,1);
loglog(rho_divs_list, res_ripple, 'o-', 'LineWidth', 1.8, 'MarkerSize', 8);
hold on;
xl = xline(80, '--', 'F28379D ≈ 80', 'Color', [0.85 0 0], ...
           'LineWidth', 1.2, 'LabelVerticalAlignment', 'middle');
xlabel('\rho_{divs}'); ylabel('Torque Ripple [%]');
title('Ripple vs granularidad temporal'); grid on; grid minor;

subplot(1,2,2);
loglog(rho_divs_list, res_Te_pp, 's-', 'LineWidth', 1.8, 'MarkerSize', 8, ...
       'Color', [0.85 0.33 0.10]);
hold on;
xline(80, '--', 'F28379D ≈ 80', 'Color', [0.85 0 0], ...
      'LineWidth', 1.2, 'LabelVerticalAlignment', 'middle');
xlabel('\rho_{divs}'); ylabel('T_e pico-pico [Nm]');
title('Rizo absoluto vs granularidad'); grid on; grid minor;

sgtitle(sprintf('Diagnostico — ¿ripple = granularidad o piso intrinseco? (%d rpm)', rpm), ...
        'FontWeight','bold');

saveas(fig, fullfile(this_dir, 'rhodivs_sweep.png'));
save(fullfile(this_dir, 'rhodivs_sweep.mat'), ...
     'rho_divs_list', 'res_ripple', 'res_Te_pp', 'res_Te_mean', 'res_wall', 'rpm');

fprintf('Guardado: rhodivs_sweep.png\n');
fprintf('Guardado: rhodivs_sweep.mat\n');

end   % sim_lowspeed_rhodivs_sweep

% ─────────────────────────────────────────────────────────────────────────
function theta_q = quantize_encoder(theta_e, PPR, P)
%QUANTIZE_ENCODER  Encoder electrico cuantizado (ver sim_lowspeed_baseline.m).
theta_mech       = theta_e / (P/2);
lsb              = 2*pi / PPR;
theta_mech_q     = round(theta_mech / lsb) * lsb;
theta_q          = theta_mech_q * (P/2);
end
