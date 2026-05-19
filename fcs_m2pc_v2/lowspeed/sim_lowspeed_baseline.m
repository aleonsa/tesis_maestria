function sim_lowspeed_baseline(rho_divs_override)
%SIM_LOWSPEED_BASELINE  Paso 1 — Baseline del FCS-M2PC a baja velocidad.
%
%   Objetivo: establecer el punto de comparacion antes de agregar
%   penalizacion de conmutacion gradual (Paso 2). Usa encoder ideal
%   (solo cuantizacion a 2048 PPR, sin latencia) y tu M2PC actual
%   sin modificar.
%
%   Pregunta que responde:
%       "¿Cuanto ripple de par produce el M2PC tal cual a {1, 5, 10,
%        20, 60} rpm cuando theta se conoce con encoder?"
%
%   Si el ripple ya es bajo a baja velocidad, la hipotesis de
%   conmutacion gradual se debilita. Si es alto, hay margen claro
%   para que el Paso 2 mejore.
%
%   rho_divs_override (opcional):
%       10  → rapido, exploratorio (default). Ripple dominado por
%             cuantizacion del simulador.
%       80  → hardware-realista (F28379D, PWM 33.3 kHz, ePWM 16-bit
%             → ~6000 pasos de duty). Tardara ~15 min total.
%
%   Uso:
%       cd fcs_m2pc_v2
%       addpath(genpath('.'))
%       cd lowspeed
%       sim_lowspeed_baseline()        % rho_divs = 10 (default)
%       sim_lowspeed_baseline(80)      % rho_divs = 80 (realista)

% ── Paths ─────────────────────────────────────────────────────────────────
this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
addpath(genpath(parent_dir));

p = motor_params();
rng(42);

% ── Configuracion del barrido ─────────────────────────────────────────────
rpm_list = [5, 10, 20, 60];         % velocidades mecanicas [rpm]
N_cycles = 2;                          % num periodos electricos a simular
enc_PPR  = 2048;                       % resolucion del encoder

% Carga constante (dimensionada para cada velocidad a Te razonable)
T_load = 0.3;                          % [Nm]

% M2PC: resolucion de division temporal del sector.
%   10 = exploratorio rapido. Ripple dominado por cuantizacion.
%   80 = hardware-realista (F28379D, PWM 33.3 kHz, 16-bit ePWM).
if nargin >= 1 && ~isempty(rho_divs_override)
    rho_divs = rho_divs_override;
else
    rho_divs = 10;
end

fprintf('\n══ Paso 1 — Baseline M2PC a baja velocidad ══════════════════\n');
fprintf('  Encoder PPR       : %d\n',     enc_PPR);
fprintf('  Resolucion angular: %.3f deg\n', 360/enc_PPR);
fprintf('  Carga fija        : %.2f Nm\n', T_load);
fprintf('  Velocidades       : [%s] rpm\n', num2str(rpm_list));
if rho_divs >= 80
    rho_tag = '(hardware-realista F28379D)';
else
    rho_tag = '(exploratorio)';
end
fprintf('  rho_divs          : %d %s\n', rho_divs, rho_tag);
fprintf('══════════════════════════════════════════════════════════════\n\n');

% ── Preparar vectores de voltaje (αβ) ─────────────────────────────────────
Vdc    = 200;
v_mag  = 2/3 * Vdc;
s3     = sqrt(3);
V_ab   = [[0;0], [v_mag;0], [v_mag/2; v_mag*s3/2], ...
          [-v_mag/2; v_mag*s3/2], [-v_mag;0], ...
          [-v_mag/2; -v_mag*s3/2], [v_mag/2; -v_mag*s3/2], [0;0]];


% ── Cargar BEMF real (LUT) ────────────────────────────────────────────────
if ~isfile(fullfile(parent_dir, 'data', 'bemf_lut.mat'))
    error('Falta bemf_lut.mat en data/. Correr generate_lut.py');
end
lut = load(fullfile(parent_dir, 'data', 'bemf_lut.mat'));

% ── Almacenamiento de resultados ──────────────────────────────────────────
n_cases = length(rpm_list);
res = struct();
res.rpm          = rpm_list;
res.Te_mean      = zeros(1, n_cases);
res.Te_std       = zeros(1, n_cases);
res.Te_ripple    = zeros(1, n_cases);   % [%] STD/mean
res.Te_pp        = zeros(1, n_cases);   % pico-pico [Nm]
res.w_mean       = zeros(1, n_cases);
res.w_std        = zeros(1, n_cases);

% Guardar series temporales de 2 corridas clave para inspeccion
traces = cell(1, n_cases);

% ── Bucle principal: una simulacion por velocidad ─────────────────────────
for c = 1:n_cases

    rpm       = rpm_list(c);
    w_m_ref   = rpm * 2*pi/60;           % [rad/s] mech
    w_e_ref   = (p.P/2) * w_m_ref;       % [rad/s] elec
    T_elec    = 2*pi / w_e_ref;          % [s]
    t_final   = N_cycles * T_elec;
    N         = round(t_final / p.Ts);

    fprintf('► %2d rpm ...', rpm);
    tic;

    % Estados iniciales
    i_ab    = [0; 0];
    w_m     = w_m_ref;                   % arranca ya en ω_ref (velocidad impuesta)
    theta_e = 0;

    % Buffers
    log_Te    = zeros(1, N);
    log_w     = zeros(1, N);
    log_ia    = zeros(1, N);
    log_ib    = zeros(1, N);
    log_theta = zeros(1, N);
    log_iref_a = zeros(1, N);
    log_iref_b = zeros(1, N);

    for k = 1:N

        % ── Encoder: cuantizar theta_real ────────────────────────────────
        theta_meas = quantize_encoder(theta_e, enc_PPR, p.P);

        % ── BEMF real (planta) con theta real ────────────────────────────
        s_real = [interp1(lut.lut_theta, lut.lut_alpha_real, ...
                          mod(theta_e, 2*pi), 'linear', 'extrap');
                  interp1(lut.lut_theta, lut.lut_beta_real, ...
                          mod(theta_e, 2*pi), 'linear', 'extrap')];
        e_real = p.Ke * w_m * s_real;

        % ── BEMF del modelo con theta medido por encoder ─────────────────
        s_model = [interp1(lut.lut_theta, lut.lut_alpha, ...
                           mod(theta_meas, 2*pi), 'linear', 'extrap');
                   interp1(lut.lut_theta, lut.lut_beta, ...
                           mod(theta_meas, 2*pi), 'linear', 'extrap')];
        e_model = p.Ke * w_m * s_model;   % usamos w_m impuesta

        % ── Referencia de corriente (velocidad impuesta, no PI) ──────────
        % Para aislar el efecto del controlador de corriente, imponemos
        % directamente un T_ref razonable: carga + margen para acelerar.
        T_ref = T_load;                  % en rampa perfecta T_ref=T_load
        norm_sq = s_model' * s_model;
        if norm_sq > 1e-6
            i_ref = (T_ref / (p.Kt * norm_sq)) * s_model;
        else
            i_ref = [0; 0];
        end

        % ── FCS-M2PC: seleccion de 2 vectores adyacentes + t1, t2 ────────
        ang_s  = mod(atan2(s_model(2), s_model(1)), 2*pi);
        sector = min(floor(ang_s / (pi/3)) + 1, 6);
        vec_pairs = [1,2; 2,3; 3,4; 4,5; 5,6; 6,1];
        av = vec_pairs(sector, :);
        u1 = V_ab(:, av(1)+1);
        u2 = V_ab(:, av(2)+1);
        u0 = [0; 0];

        % Dinamicas (derivadas de corriente por cada vector candidato)
        f1 = (1/p.L) * (u1 - e_model - p.R * i_ab);
        f2 = (1/p.L) * (u2 - e_model - p.R * i_ab);
        f0 = (1/p.L) * (u0 - e_model - p.R * i_ab);

        % Busqueda exhaustiva en el triangulo (t1, t2) con t0 = Ts-t1-t2
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
                    t1_opt = t1;
                    t2_opt = t2;
                end
            end
        end
        u_applied = (u1*t1_opt + u2*t2_opt + u0*(p.Ts - t1_opt - t2_opt)) / p.Ts;

        % ── Planta: integracion Euler de la dinamica electrica ───────────
        di_dt = (1/p.L) * (u_applied - e_real - p.R * i_ab);
        i_ab  = i_ab + di_dt * p.Ts;

        % ── Par electromagnetico real ────────────────────────────────────
        % Formula canonica (ver bldc_plant_step.m): factor (3/2) de la
        % transformada de Clarke amplitud-invariante. A baja velocidad
        % (|w_m| < 0.5 rad/s) se usa el fallback via shape para evitar
        % la division por w_m pequeno.
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

        % ── Velocidad mecanica IMPUESTA (rigida) ─────────────────────────
        % En el Paso 1 no hay lazo de velocidad; la velocidad es constante
        % para aislar la dinamica del controlador de corriente.
        % theta se integra con w_m fija:
        theta_e = theta_e + w_m * p.Ts * (p.P/2);

        % Log
        log_Te(k)     = Te;
        log_w(k)      = w_m;
        log_ia(k)     = i_ab(1);
        log_ib(k)     = i_ab(2);
        log_theta(k)  = theta_e;
        log_iref_a(k) = i_ref(1);
        log_iref_b(k) = i_ref(2);
    end

    % Metricas en los ultimos 80% de los datos (descarta transitorio)
    idx = round(N*0.2):N;
    res.Te_mean(c)    = mean(log_Te(idx));
    res.Te_std(c)     = std(log_Te(idx));
    res.Te_ripple(c)  = 100 * res.Te_std(c) / max(abs(res.Te_mean(c)), 1e-6);
    res.Te_pp(c)      = max(log_Te(idx)) - min(log_Te(idx));
    res.w_mean(c)     = mean(log_w(idx));
    res.w_std(c)      = std(log_w(idx));

    traces{c}.t       = (1:N) * p.Ts;
    traces{c}.Te      = log_Te;
    traces{c}.i_ab    = [log_ia; log_ib];
    traces{c}.i_ref   = [log_iref_a; log_iref_b];
    traces{c}.theta_e = log_theta;

    fprintf(' ✓ %.2fs | Te=%.3f Nm | Ripple=%.2f%% | Te_pp=%.4f\n', ...
            toc, res.Te_mean(c), res.Te_ripple(c), res.Te_pp(c));
end

% ── Reporte en consola ────────────────────────────────────────────────────
fprintf('\n══ Resumen Paso 1 ════════════════════════════════════════════\n');
fprintf('  rpm │  Te_mean [Nm] │ Te_ripple [%%] │ Te_pp [Nm]\n');
fprintf('  ────┼───────────────┼───────────────┼────────────\n');
for c = 1:n_cases
    fprintf('  %3d │    %7.4f    │    %6.2f     │   %7.4f\n', ...
            res.rpm(c), res.Te_mean(c), res.Te_ripple(c), res.Te_pp(c));
end
fprintf('══════════════════════════════════════════════════════════════\n\n');

% ── Figura 1: Ripple vs velocidad ─────────────────────────────────────────
fig1 = figure('Color','w','Position',[80 80 900 400]);

subplot(1,2,1);
semilogx(res.rpm, res.Te_ripple, 'o-', 'LineWidth', 1.8, 'MarkerSize', 8);
xlabel('\omega_m [rpm]'); ylabel('Torque Ripple [%]');
title('Ripple vs Velocidad (baseline M2PC)'); grid on;
xlim([0.8, 80]);

subplot(1,2,2);
semilogx(res.rpm, res.Te_pp, 's-', 'LineWidth', 1.8, 'MarkerSize', 8, ...
         'Color', [0.85 0.33 0.10]);
xlabel('\omega_m [rpm]'); ylabel('Te pico-pico [Nm]');
title('Rizo absoluto vs Velocidad'); grid on;
xlim([0.8, 80]);

sgtitle('Paso 1 — Baseline FCS-M2PC con encoder (sin penalizacion)', ...
        'FontWeight','bold');

% ── Figura 2: Series temporales a 1 rpm y 60 rpm ──────────────────────────
fig2 = figure('Color','w','Position',[80 80 1100 700]);

idx_slow = find(rpm_list == 1, 1);
idx_fast = find(rpm_list == 60, 1);
if isempty(idx_slow), idx_slow = 1; end
if isempty(idx_fast), idx_fast = n_cases; end

% 1 rpm — Te
subplot(2,2,1);
plot(traces{idx_slow}.t, traces{idx_slow}.Te, 'LineWidth', 0.8);
xlabel('t [s]'); ylabel('T_e [Nm]');
title(sprintf('Te a %d rpm', rpm_list(idx_slow))); grid on;

% 1 rpm — i_αβ
subplot(2,2,2);
plot(traces{idx_slow}.t, traces{idx_slow}.i_ab(1,:), 'b', ...
     traces{idx_slow}.t, traces{idx_slow}.i_ab(2,:), 'r', ...
     traces{idx_slow}.t, traces{idx_slow}.i_ref(1,:), 'b--', ...
     traces{idx_slow}.t, traces{idx_slow}.i_ref(2,:), 'r--', ...
     'LineWidth', 0.8);
xlabel('t [s]'); ylabel('i [A]');
legend('i_\alpha','i_\beta','i_\alpha ref','i_\beta ref', ...
       'Location','best','FontSize',7);
title(sprintf('Corrientes a %d rpm', rpm_list(idx_slow))); grid on;

% 60 rpm — Te
subplot(2,2,3);
plot(traces{idx_fast}.t, traces{idx_fast}.Te, 'LineWidth', 0.8);
xlabel('t [s]'); ylabel('T_e [Nm]');
title(sprintf('Te a %d rpm', rpm_list(idx_fast))); grid on;

% 60 rpm — i_αβ
subplot(2,2,4);
plot(traces{idx_fast}.t, traces{idx_fast}.i_ab(1,:), 'b', ...
     traces{idx_fast}.t, traces{idx_fast}.i_ab(2,:), 'r', ...
     traces{idx_fast}.t, traces{idx_fast}.i_ref(1,:), 'b--', ...
     traces{idx_fast}.t, traces{idx_fast}.i_ref(2,:), 'r--', ...
     'LineWidth', 0.8);
xlabel('t [s]'); ylabel('i [A]');
legend('i_\alpha','i_\beta','i_\alpha ref','i_\beta ref', ...
       'Location','best','FontSize',7);
title(sprintf('Corrientes a %d rpm', rpm_list(idx_fast))); grid on;

sgtitle('Comparacion de regimenes (baja vs media velocidad)', ...
        'FontWeight','bold');

% ── Guardar ───────────────────────────────────────────────────────────────
saveas(fig1, fullfile(this_dir, 'baseline_ripple_vs_speed.png'));
saveas(fig2, fullfile(this_dir, 'baseline_timeseries.png'));
save(fullfile(this_dir, 'baseline_results.mat'), 'res', 'traces');
fprintf('Guardado: baseline_ripple_vs_speed.png\n');
fprintf('Guardado: baseline_timeseries.png\n');
fprintf('Guardado: baseline_results.mat\n');

end   % sim_lowspeed_baseline

% ─────────────────────────────────────────────────────────────────────────
function theta_q = quantize_encoder(theta_e, PPR, P)
%QUANTIZE_ENCODER  Encoder electrico cuantizado.
%
%   theta_e es angulo electrico. El encoder mide angulo mecanico
%   con resolucion 2π/PPR. Convertimos: theta_mech = theta_e / (P/2).
%   Despues cuantizamos y regresamos.

theta_mech       = theta_e / (P/2);
lsb              = 2*pi / PPR;
theta_mech_q     = round(theta_mech / lsb) * lsb;
theta_q          = theta_mech_q * (P/2);
end
