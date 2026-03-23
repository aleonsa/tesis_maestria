%% FSC-M2PC — Comparación Completa: Todos los Modelos de BEMF
% =======================================================================
% Extiende FSC_M2PC_ClosedLoop.m agregando ADALINE offline y online.
%
% 6 métodos comparados en la MISMA simulación:
%   1. TRAP           — Trapezoidal fijo (baseline)
%   2. SIN            — Senoidal fijo
%   3. LEARNED        — NN profunda / LUT (del script original)
%   4. ADALINE_OFF    — ADALINE offline (Fourier óptimo, 40 params)
%   5. ADALINE_ON_T   — ADALINE online, arranca desde TRAP
%   6. ADALINE_ON_S   — ADALINE online, arranca desde SIN
%
% Los métodos online (#5, #6) actualizan sus pesos en cada paso
% usando LMS. La pregunta: ¿convergen al nivel del offline (#4)?
%
% Misma planta, PI, M2PC, escalón de carga que ClosedLoop.m
% =======================================================================
clear all; close all; clc;

%% 1. CARGA DE DATOS
fprintf('╔═══════════════════════════════════════════════════════════════════╗\n');
fprintf('║  COMPARACIÓN COMPLETA: TRAP/SIN/NN/ADALINE-OFF/ADALINE-ONLINE   ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════════╝\n\n');

if ~isfile('bemf_lut.mat') || ~isfile('bemf_adaline_lut.mat')
    error('Faltan bemf_lut.mat y/o bemf_adaline_lut.mat. Ejecutar generate_adaline_lut.py');
end
load('bemf_lut.mat');
ada_data = load('bemf_adaline_lut.mat');
W_optimal  = ada_data.adaline_W;
H_adaline  = double(ada_data.adaline_H);
fprintf('✓ LUT: %d puntos | ADALINE: H=%d, %d params\n\n', ...
        length(lut_theta), H_adaline, numel(W_optimal));

%% 2. PARÁMETROS (idénticos a ClosedLoop.m)

% Motor
p.R = 1.2;  p.L = 2.05e-3;  p.Ke = 0.40355;  p.Kt = 0.65997;
p.P = 4;    p.J = 0.00027948;  p.d = 0.0006738;

% Inversor y control
Vdc = 240;  Ts = 50e-6;  rho_divs = 10;  w_m_threshold = 5.0;

% PI
pi_ctrl.Kp = 0.5;  pi_ctrl.Ki = 50;
pi_ctrl.T_max = 5.0;  pi_ctrl.T_min = -5.0;

% Operación
w_ref = 80;  T_load = 0.5;  T_load_step = 1.0;

% Tiempo
t_final = 0.3;
Total_Steps = round(t_final / Ts);

% LMS
mu_lms = 5e-3;

% Vectores de voltaje
v_mag = 2/3 * Vdc;  s3 = sqrt(3);
V_ab = [[0;0], [v_mag;0], [v_mag/2;v_mag*s3/2], ...
        [-v_mag/2;v_mag*s3/2], [-v_mag;0], ...
        [-v_mag/2;-v_mag*s3/2], [v_mag/2;-v_mag*s3/2], [0;0]];

%% 3. PREPARAR INICIALIZACIONES ADALINE

% Pesos TRAP: ajuste de Fourier al modelo trapezoidal
theta_grid = linspace(0, 2*pi, 5000)';
X_grid = zeros(5000, 2*H_adaline);
for n = 1:H_adaline
    X_grid(:, 2*(n-1)+1) = cos(n * theta_grid);
    X_grid(:, 2*(n-1)+2) = sin(n * theta_grid);
end
trap_ab = zeros(5000, 2);
for i = 1:5000
    trap_ab(i,:) = clarke_transform(get_trapezoidal_abc(theta_grid(i)))';
end
W_trap_init = X_grid \ trap_ab;

% Pesos SIN: solo fundamental
W_sin_init = zeros(2*H_adaline, 2);
W_sin_init(2, 1) =  1.0;   % sin(θ) → alpha
W_sin_init(1, 2) = -1.0;   % -cos(θ) → beta

%% 4. DEFINIR MÉTODOS

methods = {'TRAP', 'SIN', 'LEARNED', 'ADALINE_OFF', 'ADALINE_ON_T', 'ADALINE_ON_S'};
n_methods = length(methods);

colors.TRAP         = [0.85, 0.33, 0.10];
colors.SIN          = [0.00, 0.45, 0.74];
colors.LEARNED      = [0.47, 0.67, 0.19];
colors.ADALINE_OFF  = [0.56, 0.27, 0.68];
colors.ADALINE_ON_T = [0.93, 0.69, 0.13];
colors.ADALINE_ON_S = [0.30, 0.75, 0.93];

fprintf('Métodos: %s\n', strjoin(methods, ', '));
fprintf('Simulación: %.0f ms, Ts=%.0f μs, ω_ref=%.0f rad/s\n', ...
        t_final*1e3, Ts*1e6, w_ref);
fprintf('LMS: μ=%.0e\n\n', mu_lms);

%% 5. SIMULACIÓN
results = struct();

fprintf('═══════════════════════════════════════════════\n');

for method_idx = 1:n_methods
    method_name = methods{method_idx};
    fprintf('► %s ...', method_name);
    tic;

    % --- Estados iniciales ---
    i_ab = [0;0];  w_m = 0;  theta_e = 0;  pi_int = 0;

    % --- ADALINE online: inicializar pesos ---
    is_online = false;
    switch method_name
        case 'ADALINE_ON_T'
            W_ada = W_trap_init;  is_online = true;
        case 'ADALINE_ON_S'
            W_ada = W_sin_init;   is_online = true;
        case {'ADALINE_OFF'}
            W_ada = W_optimal;
        otherwise
            W_ada = [];  % No se usa
    end

    % --- Almacenamiento ---
    log.i     = zeros(2, Total_Steps);
    log.i_ref = zeros(2, Total_Steps);
    log.Te    = zeros(1, Total_Steps);
    log.w_m   = zeros(1, Total_Steps);
    log.T_ref = zeros(1, Total_Steps);
    log.T_load_actual = zeros(1, Total_Steps);
    if is_online
        log.W_error = zeros(1, Total_Steps);
    end

    % --- Bucle principal ---
    for k = 1:Total_Steps

        % Carga: escalón a la mitad
        if k < Total_Steps/2
            T_load_k = T_load;
        else
            T_load_k = T_load + T_load_step;
        end

        % ── A. BEMF REAL (planta) ──
        theta_mod = mod(theta_e, 2*pi);
        l_a = interp1(lut_theta, lut_alpha_real, theta_mod, 'linear', 'extrap');
        l_b = interp1(lut_theta, lut_beta_real,  theta_mod, 'linear', 'extrap');
        shape_real = [l_a; l_b];
        e_real = p.Ke * w_m * shape_real;

        % ── B. BEMF del MODELO ──
        switch method_name
            case 'SIN'
                shape_model = [sin(theta_e); -cos(theta_e)];

            case 'TRAP'
                shape_model = clarke_transform(get_trapezoidal_abc(theta_e));

            case 'LEARNED'
                nn_a = interp1(lut_theta, lut_alpha, theta_mod, 'linear', 'extrap');
                nn_b = interp1(lut_theta, lut_beta,  theta_mod, 'linear', 'extrap');
                shape_model = [nn_a; nn_b];

            case {'ADALINE_OFF', 'ADALINE_ON_T', 'ADALINE_ON_S'}
                % Evaluar serie de Fourier
                x_f = zeros(2*H_adaline, 1);
                for n = 1:H_adaline
                    x_f(2*(n-1)+1) = cos(n * theta_mod);
                    x_f(2*(n-1)+2) = sin(n * theta_mod);
                end
                shape_model = W_ada' * x_f;
        end
        e_model = p.Ke * w_m * shape_model;

        % ── C. LMS (solo métodos online) ──
        if is_online && abs(w_m) > w_m_threshold
            % Señal de error: shape real vs shape predicha
            s_pred = W_ada' * x_f;
            eps_a = shape_real(1) - s_pred(1);
            eps_b = shape_real(2) - s_pred(2);

            % Actualización
            W_ada(:,1) = W_ada(:,1) + mu_lms * eps_a * x_f;
            W_ada(:,2) = W_ada(:,2) + mu_lms * eps_b * x_f;

            % Recalcular shape con pesos actualizados
            shape_model = W_ada' * x_f;
            e_model = p.Ke * w_m * shape_model;
        end

        % ── D. PI de velocidad ──
        w_error = w_ref - w_m;
        pi_int = pi_int + pi_ctrl.Ki * w_error * Ts;
        T_ref  = pi_ctrl.Kp * w_error + pi_int;
        if T_ref > pi_ctrl.T_max
            T_ref = pi_ctrl.T_max;
            pi_int = pi_int - pi_ctrl.Ki * w_error * Ts;
        elseif T_ref < pi_ctrl.T_min
            T_ref = pi_ctrl.T_min;
            pi_int = pi_int - pi_ctrl.Ki * w_error * Ts;
        end

        % ── E. Referencia de corriente ──
        norm_sq = shape_model' * shape_model;
        if norm_sq > 1e-6 && abs(w_m) > w_m_threshold
            i_ref = (T_ref / (p.Kt * norm_sq)) * shape_model;
        else
            ns = norm(shape_model);
            if ns > 1e-6
                i_ref = (T_ref / p.Kt) * (shape_model / ns);
            else
                i_ref = [0; 0];
            end
        end

        % ── F. FCS-M2PC ──
        if abs(w_m) < w_m_threshold
            min_cost = inf;  u_best = [0;0];
            for v = 1:8
                u_try = V_ab(:,v);
                i_pred = i_ab + (Ts/p.L)*(u_try - e_model - p.R*i_ab);
                cost = sum((i_ref - i_pred).^2);
                if cost < min_cost
                    min_cost = cost;  u_best = u_try;
                end
            end
            u_applied = u_best;
        else
            ang = mod(atan2(shape_model(2), shape_model(1)), 2*pi);
            sector = min(floor(ang/(pi/3)) + 1, 6);
            vec_pairs = [1,2;2,3;3,4;4,5;5,6;6,1];
            av = vec_pairs(sector,:);
            u1 = V_ab(:,av(1)+1);  u2 = V_ab(:,av(2)+1);  u0 = [0;0];

            f1 = (1/p.L)*(u1 - e_model - p.R*i_ab);
            f2 = (1/p.L)*(u2 - e_model - p.R*i_ab);
            f0 = (1/p.L)*(u0 - e_model - p.R*i_ab);

            min_cost = inf;  t1_opt = 0;  t2_opt = 0;
            for nn = 0:rho_divs
                t1 = (nn/rho_divs)*Ts;
                for mm = 0:(rho_divs - nn)
                    t2 = (mm/rho_divs)*Ts;
                    t0 = Ts - t1 - t2;
                    i_pred = i_ab + f1*t1 + f2*t2 + f0*t0;
                    cost = sum((i_ref - i_pred).^2);
                    if cost < min_cost
                        min_cost = cost;
                        t1_opt = t1;  t2_opt = t2;
                    end
                end
            end
            u_applied = (u1*t1_opt + u2*t2_opt + u0*(Ts-t1_opt-t2_opt))/Ts;
        end

        % ── G. Planta ──
        di_dt = (1/p.L)*(u_applied - e_real - p.R*i_ab);
        i_ab = i_ab + di_dt * Ts;

        if abs(w_m) > 0.5
            Te = (p.P/2) * (e_real' * i_ab) / w_m;
        else
            nr = norm(shape_real);
            if nr > 1e-6
                Te = p.Kt * (shape_real' * i_ab) / nr;
            else
                Te = 0;
            end
        end

        dw_dt = (Te - p.d*w_m - T_load_k) / p.J;
        w_m = w_m + dw_dt * Ts;
        theta_e = theta_e + w_m * Ts * (p.P/2);

        % ── H. Registro ──
        log.i(:,k)     = i_ab;
        log.i_ref(:,k) = i_ref;
        log.Te(k)      = Te;
        log.w_m(k)     = w_m;
        log.T_ref(k)   = T_ref;
        log.T_load_actual(k) = T_load_k;
        if is_online
            log.W_error(k) = norm(W_ada - W_optimal,'fro') / ...
                             norm(W_optimal,'fro');
        end
    end

    % --- Métricas ---
    post_idx = round(Total_Steps*0.85):Total_Steps;
    Te_ss = log.Te(post_idx);
    log.Te_mean   = mean(Te_ss);
    log.Te_std    = std(Te_ss);
    log.Te_ripple = 100 * log.Te_std / max(abs(log.Te_mean), 1e-6);
    log.Te_pp     = max(Te_ss) - min(Te_ss);

    w_ss = log.w_m(post_idx);
    log.w_pp       = max(w_ss) - min(w_ss);
    log.w_ss_error = abs(w_ref - mean(w_ss));

    results.(method_name) = log;
    fprintf(' ✓ %.2fs | Ripple: %.2f%% | ω_pp: %.4f\n', ...
            toc, log.Te_ripple, log.w_pp);
end
fprintf('═══════════════════════════════════════════════\n\n');

%% 6. GRÁFICAS

t_ms = (1:Total_Steps) * Ts * 1000;

% --- FIGURA 1: Panorama completo ---
figure('Color','w', 'Position', [50, 50, 1400, 900]);

subplot(4,1,1);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_ms, results.(m).w_m, 'Color', colors.(m), ...
         'LineWidth', 1.3, 'DisplayName', m); hold on;
end
yline(w_ref, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(t_final*500, 'r:', 'Load step', 'LineWidth', 1, ...
      'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
ylabel('\omega_m [rad/s]', 'Interpreter', 'tex');
title('Velocidad', 'FontWeight', 'bold');
legend('Location', 'best', 'NumColumns', 3, 'FontSize', 7);
grid on; box on;

subplot(4,1,2);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_ms, results.(m).T_ref, 'Color', colors.(m), 'LineWidth', 1); hold on;
end
xline(t_final*500, 'r:', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('T_{ref} [Nm]', 'Interpreter', 'tex');
title('Salida PI', 'FontWeight', 'bold');
grid on; box on;

subplot(4,1,3);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_ms, results.(m).Te, 'Color', colors.(m), 'LineWidth', 0.8); hold on;
end
plot(t_ms, results.TRAP.T_load_actual, 'k--', 'LineWidth', 1.5, ...
     'DisplayName', 'T_{load}');
xline(t_final*500, 'r:', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('T_e [Nm]', 'Interpreter', 'tex');
title('Torque Electromagnético', 'FontWeight', 'bold');
grid on; box on;

subplot(4,1,4);
for idx = 1:n_methods
    m = methods{idx};
    i_mag = sqrt(results.(m).i(1,:).^2 + results.(m).i(2,:).^2);
    plot(t_ms, i_mag, 'Color', colors.(m), 'LineWidth', 0.8); hold on;
end
xline(t_final*500, 'r:', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('|i| [A]'); xlabel('Tiempo [ms]');
title('Magnitud de Corriente', 'FontWeight', 'bold');
grid on; box on;

sgtitle('Comparación Completa: 6 Modelos de BEMF con FCS-M2PC', ...
        'FontSize', 14, 'FontWeight', 'bold');

% --- FIGURA 2: Zoom estado estable + barras ---
figure('Color','w', 'Position', [100, 100, 1400, 700]);

w_final = abs(results.TRAP.w_m(end));
if w_final < 1, w_final = w_ref; end
samples_2rev = round(2 * 2*pi / (w_final * Ts * p.P/2));
samples_2rev = min(samples_2rev, Total_Steps - 1);
z_start = max(1, Total_Steps - samples_2rev);
t_zoom = t_ms(z_start:end);

subplot(2,2,1);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_zoom, results.(m).Te(z_start:end), 'Color', colors.(m), ...
         'LineWidth', 1.3, 'DisplayName', m); hold on;
end
ylabel('T_e [Nm]', 'Interpreter', 'tex');
title('Torque — Zoom', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 7); grid on; box on;

subplot(2,2,2);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_zoom, results.(m).w_m(z_start:end), 'Color', colors.(m), ...
         'LineWidth', 1.3, 'DisplayName', m); hold on;
end
yline(w_ref, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('\omega_m [rad/s]', 'Interpreter', 'tex');
title('Velocidad — Zoom', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 7); grid on; box on;

subplot(2,2,3);
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

subplot(2,2,4);
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

% --- FIGURA 3: Convergencia ADALINE online ---
figure('Color','w', 'Position', [150, 150, 1200, 400]);

subplot(1,2,1);
plot(t_ms, results.ADALINE_ON_T.W_error * 100, ...
     'Color', colors.ADALINE_ON_T, 'LineWidth', 1.8, ...
     'DisplayName', 'Online desde TRAP'); hold on;
plot(t_ms, results.ADALINE_ON_S.W_error * 100, ...
     'Color', colors.ADALINE_ON_S, 'LineWidth', 1.8, ...
     'DisplayName', 'Online desde SIN');
yline(5, 'k--', '5%', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('||W - W^*|| / ||W^*|| [%]');
xlabel('Tiempo [ms]');
title('Convergencia de Pesos hacia Óptimo Offline', 'FontWeight', 'bold');
legend('Location', 'best'); grid on; box on;

subplot(1,2,2);
% Torque ripple instantáneo (ventana deslizante)
win = round(2*pi / (w_ref * (p.P/2) * Ts));  % 1 rev eléctrica
for idx = 1:n_methods
    m = methods{idx};
    Te_roll_std = movstd(results.(m).Te, win);
    Te_roll_mean = movmean(results.(m).Te, win);
    ripple_inst = 100 * Te_roll_std ./ max(abs(Te_roll_mean), 1e-6);
    plot(t_ms, ripple_inst, 'Color', colors.(m), ...
         'LineWidth', 1, 'DisplayName', m); hold on;
end
ylabel('Torque Ripple Instantáneo [%]');
xlabel('Tiempo [ms]');
title('Evolución del Ripple (ventana = 1 rev eléctrica)', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 7); grid on; box on;

sgtitle('Fase 1: Convergencia del ADALINE Online', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% 7. REPORTE

fprintf('╔═════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                        RESULTADOS — 6 MÉTODOS                         ║\n');
fprintf('╠═════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Método         │ Ripple [%%] │ Te P-P [Nm] │ ω P-P [rad/s] │ ω err   ║\n');
fprintf('╠═════════════════════════════════════════════════════════════════════════╣\n');
for idx = 1:n_methods
    m = methods{idx};
    r = results.(m);
    fprintf('║ %-14s │   %6.2f   │    %.4f   │    %.4f     │ %.4f  ║\n', ...
            m, r.Te_ripple, r.Te_pp, r.w_pp, r.w_ss_error);
end
fprintf('╚═════════════════════════════════════════════════════════════════════════╝\n\n');

% Comparaciones clave
fprintf('COMPARACIONES CLAVE:\n\n');

r_trap = results.TRAP.Te_ripple;
r_nn   = results.LEARNED.Te_ripple;
r_off  = results.ADALINE_OFF.Te_ripple;
r_on_t = results.ADALINE_ON_T.Te_ripple;
r_on_s = results.ADALINE_ON_S.Te_ripple;

fprintf('  ADALINE offline vs NN (LEARNED):\n');
fprintf('    Ripple: %.2f%% vs %.2f%% (Δ = %+.2f%%)\n', r_off, r_nn, r_off - r_nn);
if abs(r_off - r_nn) < 0.5
    fprintf('    → Rendimiento equivalente con 40 vs ~25k parámetros ✓\n\n');
end

fprintf('  ADALINE online (desde TRAP) vs offline:\n');
fprintf('    Ripple: %.2f%% vs %.2f%% (Δ = %+.2f%%)\n', r_on_t, r_off, r_on_t - r_off);
fprintf('    → Online convergió al nivel del offline: %s\n\n', ...
        ternary(abs(r_on_t - r_off) < 1.0, 'SÍ ✓', 'NO ✗'));

fprintf('  ADALINE online (desde SIN) vs offline:\n');
fprintf('    Ripple: %.2f%% vs %.2f%% (Δ = %+.2f%%)\n', r_on_s, r_off, r_on_s - r_off);
fprintf('    → Online convergió al nivel del offline: %s\n\n', ...
        ternary(abs(r_on_s - r_off) < 1.0, 'SÍ ✓', 'NO ✗'));

fprintf('  Mejora global (ADALINE online desde TRAP vs TRAP fijo):\n');
imp = 100*(r_trap - r_on_t)/r_trap;
fprintf('    Reducción de ripple: %.1f%%\n', imp);
fprintf('    Sin necesidad de datos offline — el motor se auto-calibra.\n\n');

fprintf('═══════════════════════════════════════════════════════════════\n');

%% FUNCIONES AUXILIARES

function abc = get_trapezoidal_abc(theta)
    phases = [0, 2*pi/3, 4*pi/3];
    abc = zeros(3, 1);
    for i = 1:3
        ti = mod(theta - phases(i) + pi/6, 2*pi);
        if ti < pi/6,         val = ti * (6/pi);
        elseif ti < 5*pi/6,   val = 1;
        elseif ti < 7*pi/6,   val = 1 - (ti - 5*pi/6) * (6/pi);
        elseif ti < 11*pi/6,  val = -1;
        else,                  val = -1 + (ti - 11*pi/6) * (6/pi);
        end
        abc(i) = val;
    end
end

function ab = clarke_transform(abc)
    ab = (2/3) * [1, -0.5, -0.5; 0, sqrt(3)/2, -sqrt(3)/2] * abc;
end

function r = ternary(cond, a, b)
    if cond, r = a; else, r = b; end
end