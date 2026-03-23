%% FSC-M2PC con Lazo Cerrado de Velocidad
% =======================================================================
% Simulación completa del drive BLDC con control en cascada:
%
%   Lazo externo: PI de velocidad  →  genera T_ref
%   Lazo interno: FCS-M2PC         →  genera u_applied
%   Planta:       Motor BLDC       →  dinámica eléctrica + mecánica
%
% Compara 4 modelos de BEMF en el controlador:
%   TRAP    - Modelo trapezoidal ideal
%   SIN     - Modelo senoidal puro
%   LEARNED - Modelo aprendido (NN/LUT con interpolación)
%   ADALINE - Modelo aprendido (Fourier directo, 40 params, Fase 0)
%
% CORRECCIONES vs versión anterior:
%   - Bug de arranque: a baja velocidad e_model ≈ 0 → sector inválido
%     → el M2PC elegía vector nulo → motor no arrancaba.
%     Fix: usar barrido de 8 vectores (FCS-MPC clásico) cuando w_m < umbral.
%   - Sector calculado desde shape_model (no desde e_model)
%     porque shape tiene dirección correcta incluso a baja velocidad.
%   - Cálculo de torque en arranque usa proyección sobre shape_real
%     en lugar de ||i|| (que puede dar torque negativo por desalineación).
%   - Zoom de figuras usa velocidad robusta (no depende de un método).
%   - PI con ganancias razonables (Kp=0.5, Ki=50).
% =======================================================================
clear all; close all; clc;

%% 1. CARGA DE LA BEMF REAL
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  FCS-M2PC + PI VELOCIDAD — LAZO CERRADO COMPLETO         ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

if isfile('bemf_lut.mat')
    load('bemf_lut.mat');
    fprintf('✓ LUT cargada: %d puntos\n', length(lut_theta));
else
    error('Falta bemf_lut.mat. Ejecutar generate_adaline_lut.py primero.');
end

% Cargar pesos ADALINE (generados por generate_adaline_lut.py)
if isfile('bemf_adaline_lut.mat')
    ada_data = load('bemf_adaline_lut.mat');
    adaline_W = ada_data.adaline_W;   % (2H × 2) pesos de Fourier
    adaline_H = double(ada_data.adaline_H);  % número de armónicos
    fprintf('✓ ADALINE cargado: H=%d armónicos, %d parámetros\n\n', ...
            adaline_H, numel(adaline_W));
else
    warning('Falta bemf_adaline_lut.mat — ADALINE no disponible.');
    adaline_W = [];
    adaline_H = 0;
end

%% 2. PARÁMETROS DEL SISTEMA

% --- Motor (BLY-344S-240V-3000) ---
p.R  = 1.2;              % Resistencia [Ohm]
p.L  = 2.05e-3;          % Inductancia [H]
p.Ke = 0.40355;          % Constante BEMF [V·s/rad]
p.Kt = 0.65997;          % Constante torque [Nm/A]
p.P  = 4;                % Polos
p.J  = 0.00027948;       % Inercia [kg·m²]
p.d  = 0.0006738;        % Fricción viscosa [N·m·s]

% --- Inversor ---
Vdc = 240;               % Voltaje DC bus [V]

% --- Control ---
Ts = 50e-6;              % Periodo muestreo [s] (20 kHz)
rho_divs = 10;           % Divisiones M2PC
w_m_threshold = 5.0;     % Umbral arranque→M2PC [rad/s]

% --- PI de velocidad ---
% Sintonización conservadora: BW << freq eléctrica
pi_ctrl.Kp = 0.5;        % Ganancia proporcional
pi_ctrl.Ki = 50;          % Ganancia integral
pi_ctrl.T_max = 5.0;     % Saturación [Nm]
pi_ctrl.T_min = -5.0;

% --- Condiciones de operación ---
w_ref = 80;              % Velocidad de referencia [rad/s]
T_load = 0.5;            % Carga nominal [Nm]
T_load_step = 1.0;       % Escalón de carga adicional [Nm]

% --- Tiempo de simulación ---
t_final = 0.3;           % [s]
Total_Steps = round(t_final / Ts);

fprintf('PARÁMETROS:\n');
fprintf('  Motor: R=%.2f Ω, L=%.2f mH, J=%.5f kg·m², P=%d\n', ...
        p.R, p.L*1e3, p.J, p.P);
fprintf('  Control: Ts=%.0f μs, ρ=%d, PI=[Kp=%.2f, Ki=%.1f]\n', ...
        Ts*1e6, rho_divs, pi_ctrl.Kp, pi_ctrl.Ki);
fprintf('  Operación: ω_ref=%.0f rad/s, T_load=%.1f Nm\n', w_ref, T_load);
fprintf('  Perturbación: +%.1f Nm escalón en t=%.0f ms\n', ...
        T_load_step, t_final*500);
fprintf('  Duración: %.0f ms (%d pasos)\n\n', t_final*1e3, Total_Steps);

%% 3. VECTORES DE VOLTAJE
v_mag = 2/3 * Vdc;
s3 = sqrt(3);
V_ab = [ [0;0], [v_mag;0], [v_mag/2; v_mag*s3/2], ...
         [-v_mag/2; v_mag*s3/2], [-v_mag;0], ...
         [-v_mag/2; -v_mag*s3/2], [v_mag/2; -v_mag*s3/2], [0;0] ];

%% 4. SIMULACIÓN
methods = {'TRAP', 'SIN', 'LEARNED', 'ADALINE'};
n_methods = length(methods);
results = struct();

fprintf('═══════════════════════════════════════════════════════\n');
fprintf('INICIANDO SIMULACIONES\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

for method_idx = 1:n_methods
    method_name = methods{method_idx};
    fprintf('► %s ...\n', method_name);
    tic;

    % ===== ESTADOS INICIALES =====
    i_ab    = [0; 0];
    w_m     = 0;
    theta_e = 0;
    pi_int  = 0;

    % ===== ALMACENAMIENTO =====
    log.i     = zeros(2, Total_Steps);
    log.i_ref = zeros(2, Total_Steps);
    log.e_model = zeros(2, Total_Steps);
    log.e_real  = zeros(2, Total_Steps);
    log.Te    = zeros(1, Total_Steps);
    log.w_m   = zeros(1, Total_Steps);
    log.T_ref = zeros(1, Total_Steps);
    log.T_load_actual = zeros(1, Total_Steps);

    % ===== BUCLE PRINCIPAL =====
    for k = 1:Total_Steps

        % Carga: escalón a la mitad
        if k < Total_Steps/2
            T_load_k = T_load;
        else
            T_load_k = T_load + T_load_step;
        end

        % ──────────────────────────────────────────────────
        % A. BEMF REAL (planta)
        % ──────────────────────────────────────────────────
        theta_mod = mod(theta_e, 2*pi);
        l_a = interp1(lut_theta, lut_alpha_real, theta_mod, 'linear', 'extrap');
        l_b = interp1(lut_theta, lut_beta_real,  theta_mod, 'linear', 'extrap');
        shape_real = [l_a; l_b];
        e_real = p.Ke * w_m * shape_real;

        % ──────────────────────────────────────────────────
        % B. BEMF del MODELO (lo que el controlador cree)
        % ──────────────────────────────────────────────────
        switch method_name
            case 'SIN'
                shape_model = [sin(theta_e); -cos(theta_e)];
            case 'TRAP'
                shape_abc = get_trapezoidal_abc(theta_e);
                shape_model = clarke_transform(shape_abc);
            case 'LEARNED'
                nn_a = interp1(lut_theta, lut_alpha, theta_mod, 'linear', 'extrap');
                nn_b = interp1(lut_theta, lut_beta,  theta_mod, 'linear', 'extrap');
                shape_model = [nn_a; nn_b];
            case 'ADALINE'
                % Evaluación directa de la serie de Fourier:
                %   ê(θ) = Σ [w_n^c·cos(nθ) + w_n^s·sin(nθ)]
                % Sin interpolación de LUT — solo producto punto.
                % Esto es lo que correría en un DSP real.
                x_fourier = zeros(2*adaline_H, 1);
                for n = 1:adaline_H
                    x_fourier(2*(n-1)+1) = cos(n * theta_mod);
                    x_fourier(2*(n-1)+2) = sin(n * theta_mod);
                end
                shape_model = adaline_W' * x_fourier;  % (2×2H)·(2H×1) = (2×1)
        end
        e_model = p.Ke * w_m * shape_model;

        % ──────────────────────────────────────────────────
        % C. PI DE VELOCIDAD → T_ref
        % ──────────────────────────────────────────────────
        w_error = w_ref - w_m;
        pi_int  = pi_int + pi_ctrl.Ki * w_error * Ts;
        T_ref   = pi_ctrl.Kp * w_error + pi_int;

        % Anti-windup: saturación con clamping
        if T_ref > pi_ctrl.T_max
            T_ref = pi_ctrl.T_max;
            pi_int = pi_int - pi_ctrl.Ki * w_error * Ts;
        elseif T_ref < pi_ctrl.T_min
            T_ref = pi_ctrl.T_min;
            pi_int = pi_int - pi_ctrl.Ki * w_error * Ts;
        end

        % ──────────────────────────────────────────────────
        % D. GENERAR REFERENCIA DE CORRIENTE
        % ──────────────────────────────────────────────────
        norm_sq = shape_model' * shape_model;
        if norm_sq > 1e-6 && abs(w_m) > w_m_threshold
            % Régimen normal: referencia óptima para torque constante
            i_ref = (T_ref / (p.Kt * norm_sq)) * shape_model;
        else
            % Arranque: magnitud fija, dirección del modelo
            ns = norm(shape_model);
            if ns > 1e-6
                i_ref = (T_ref / p.Kt) * (shape_model / ns);
            else
                i_ref = [0; 0];
            end
        end

        % ──────────────────────────────────────────────────
        % E. FCS-M2PC (lazo interno de corriente)
        % ──────────────────────────────────────────────────

        if abs(w_m) < w_m_threshold
            % === ARRANQUE: Barrido de 8 vectores ===
            % A baja velocidad, e_model ≈ 0 y el sector es
            % inválido. Usamos FCS-MPC clásico que evalúa
            % todos los vectores sin depender del sector.
            min_cost = inf;
            u_best = [0; 0];
            for v = 1:8
                u_try = V_ab(:, v);
                i_pred = i_ab + (Ts/p.L) * (u_try - e_model - p.R*i_ab);
                cost = sum((i_ref - i_pred).^2);
                if cost < min_cost
                    min_cost = cost;
                    u_best = u_try;
                end
            end
            u_applied = u_best;

        else
            % === RÉGIMEN NORMAL: FCS-M2PC con subdivisiones ===
            
            % Sector desde shape_model (tiene dirección correcta
            % incluso cuando e_model es pequeño)
            ang = mod(atan2(shape_model(2), shape_model(1)), 2*pi);
            sector = min(floor(ang / (pi/3)) + 1, 6);

            vec_pairs = [1,2; 2,3; 3,4; 4,5; 5,6; 6,1];
            av = vec_pairs(sector, :);
            u1 = V_ab(:, av(1)+1);
            u2 = V_ab(:, av(2)+1);
            u0 = [0; 0];

            % Pendientes de predicción
            f1 = (1/p.L) * (u1 - e_model - p.R*i_ab);
            f2 = (1/p.L) * (u2 - e_model - p.R*i_ab);
            f0 = (1/p.L) * (u0 - e_model - p.R*i_ab);

            % Grid search M2PC
            min_cost = inf; t1_opt = 0; t2_opt = 0;
            for n = 0:rho_divs
                t1 = (n / rho_divs) * Ts;
                for m = 0:(rho_divs - n)
                    t2 = (m / rho_divs) * Ts;
                    t0 = Ts - t1 - t2;
                    i_pred = i_ab + f1*t1 + f2*t2 + f0*t0;
                    cost = sum((i_ref - i_pred).^2);
                    if cost < min_cost
                        min_cost = cost;
                        t1_opt = t1; t2_opt = t2;
                    end
                end
            end

            t0_opt = Ts - t1_opt - t2_opt;
            u_applied = (u1*t1_opt + u2*t2_opt + u0*t0_opt) / Ts;
        end

        % ──────────────────────────────────────────────────
        % F. PLANTA: Dinámica eléctrica + mecánica
        % ──────────────────────────────────────────────────

        % Eléctrica
        di_dt = (1/p.L) * (u_applied - e_real - p.R*i_ab);
        i_ab = i_ab + di_dt * Ts;

        % Torque electromagnético
        if abs(w_m) > 0.5
            Te = (p.P/2) * (e_real' * i_ab) / w_m;
        else
            % Arranque: proyección de corriente sobre dirección de BEMF
            nr = norm(shape_real);
            if nr > 1e-6
                Te = p.Kt * (shape_real' * i_ab) / nr;
            else
                Te = 0;
            end
        end

        % Mecánica
        dw_dt = (Te - p.d * w_m - T_load_k) / p.J;
        w_m = w_m + dw_dt * Ts;

        % Ángulo eléctrico
        theta_e = theta_e + w_m * Ts * (p.P/2);

        % ──────────────────────────────────────────────────
        % G. REGISTRO
        % ──────────────────────────────────────────────────
        log.i(:, k)     = i_ab;
        log.i_ref(:, k) = i_ref;
        log.e_model(:, k) = e_model;
        log.e_real(:, k)  = e_real;
        log.Te(k)       = Te;
        log.w_m(k)      = w_m;
        log.T_ref(k)    = T_ref;
        log.T_load_actual(k) = T_load_k;
    end

    % ===== MÉTRICAS =====
    post_idx = round(Total_Steps*0.85):Total_Steps;

    Te_ss = log.Te(post_idx);
    log.Te_mean    = mean(Te_ss);
    log.Te_std     = std(Te_ss);
    log.Te_ripple  = 100 * log.Te_std / max(abs(log.Te_mean), 1e-6);
    log.Te_pp      = max(Te_ss) - min(Te_ss);

    w_ss = log.w_m(post_idx);
    log.w_mean     = mean(w_ss);
    log.w_std      = std(w_ss);
    log.w_ripple   = 100 * log.w_std / max(abs(log.w_mean), 1e-6);
    log.w_pp       = max(w_ss) - min(w_ss);
    log.w_ss_error = abs(w_ref - log.w_mean);

    err = log.i(:, post_idx) - log.i_ref(:, post_idx);
    log.rmse = sqrt(mean(sum(err.^2, 1)));

    bemf_err = log.e_real(:, post_idx) - log.e_model(:, post_idx);
    log.bemf_rmse = sqrt(mean(sum(bemf_err.^2, 1)));

    results.(method_name) = log;

    elapsed = toc;
    fprintf('  ✓ %.2f s | Te ripple: %.2f%% | ω P-P: %.4f rad/s | ω_ss err: %.4f\n\n', ...
            elapsed, log.Te_ripple, log.w_pp, log.w_ss_error);
end

%% 5. GRÁFICAS

t_ms = (1:Total_Steps) * Ts * 1000;

colors.TRAP    = [0.85, 0.33, 0.10];
colors.SIN     = [0.00, 0.45, 0.74];
colors.LEARNED = [0.47, 0.67, 0.19];
colors.ADALINE = [0.56, 0.27, 0.68];

% =========================================================================
% FIGURA 1: Panorama completo
% =========================================================================
figure('Color','w', 'Position', [50, 50, 1400, 900]);

subplot(4, 1, 1);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_ms, results.(m).w_m, 'Color', colors.(m), ...
         'LineWidth', 1.5, 'DisplayName', m); hold on;
end
yline(w_ref, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(t_final*500, 'r:', 'Load step', 'LineWidth', 1, ...
      'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
ylabel('\omega_m [rad/s]', 'Interpreter', 'tex');
title('Velocidad Mecánica', 'FontWeight', 'bold');
legend('Location', 'best'); grid on; box on;

subplot(4, 1, 2);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_ms, results.(m).T_ref, 'Color', colors.(m), 'LineWidth', 1.2); hold on;
end
xline(t_final*500, 'r:', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('T_{ref} [Nm]', 'Interpreter', 'tex');
title('Salida del PI (Referencia de Torque)', 'FontWeight', 'bold');
grid on; box on;

subplot(4, 1, 3);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_ms, results.(m).Te, 'Color', colors.(m), 'LineWidth', 1.0); hold on;
end
plot(t_ms, results.TRAP.T_load_actual, 'k--', 'LineWidth', 1.5, ...
     'DisplayName', 'T_{load}');
xline(t_final*500, 'r:', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('T_e [Nm]', 'Interpreter', 'tex');
title('Torque Electromagnético Real', 'FontWeight', 'bold');
legend([methods, {'T_{load}'}], 'Location', 'best'); grid on; box on;

subplot(4, 1, 4);
for idx = 1:n_methods
    m = methods{idx};
    i_mag = sqrt(results.(m).i(1,:).^2 + results.(m).i(2,:).^2);
    plot(t_ms, i_mag, 'Color', colors.(m), 'LineWidth', 1.0); hold on;
end
xline(t_final*500, 'r:', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('|i| [A]', 'Interpreter', 'tex');
xlabel('Tiempo [ms]');
title('Magnitud de Corriente', 'FontWeight', 'bold');
grid on; box on;

sgtitle('Respuesta Completa del Drive BLDC con PI + FCS-M2PC', ...
        'FontSize', 14, 'FontWeight', 'bold');

% =========================================================================
% FIGURA 2: Zoom estado estable (post-escalón)
% =========================================================================
figure('Color','w', 'Position', [100, 100, 1400, 700]);

% Calcular zoom robusto
w_final = 0;
for idx = 1:n_methods
    w_test = abs(results.(methods{idx}).w_m(end));
    if w_test > w_final
        w_final = w_test;
    end
end
if w_final < 1, w_final = w_ref; end

samples_2rev = round(2 * 2*pi / (w_final * Ts * p.P/2));
samples_2rev = min(samples_2rev, Total_Steps - 1);
z_start = max(1, Total_Steps - samples_2rev);
t_zoom = t_ms(z_start:end);

subplot(2, 2, 1);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_zoom, results.(m).Te(z_start:end), 'Color', colors.(m), ...
         'LineWidth', 1.5, 'DisplayName', m); hold on;
end
ylabel('T_e [Nm]', 'Interpreter', 'tex');
title('Torque — Zoom Estado Estable', 'FontWeight', 'bold');
legend('Location', 'best'); grid on; box on;

subplot(2, 2, 2);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_zoom, results.(m).w_m(z_start:end), 'Color', colors.(m), ...
         'LineWidth', 1.5, 'DisplayName', m); hold on;
end
yline(w_ref, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
ylabel('\omega_m [rad/s]', 'Interpreter', 'tex');
title('Velocidad — Zoom Estado Estable', 'FontWeight', 'bold');
legend('Location', 'best'); grid on; box on;

subplot(2, 2, 3);
ripple_vals = cellfun(@(m) results.(m).Te_ripple, methods);
b = bar(ripple_vals);
b.FaceColor = 'flat';
b.CData = cell2mat(cellfun(@(m) colors.(m), methods, 'UniformOutput', false)');
set(gca, 'XTickLabel', methods);
ylabel('Torque Ripple [%]');
title('Torque Ripple (STD/mean)', 'FontWeight', 'bold');
grid on; box on;
for i = 1:n_methods
    text(i, ripple_vals(i), sprintf('%.2f%%', ripple_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontWeight', 'bold');
end

subplot(2, 2, 4);
w_pp_vals = cellfun(@(m) results.(m).w_pp, methods);
b = bar(w_pp_vals);
b.FaceColor = 'flat';
b.CData = cell2mat(cellfun(@(m) colors.(m), methods, 'UniformOutput', false)');
set(gca, 'XTickLabel', methods);
ylabel('\omega P-P [rad/s]', 'Interpreter', 'tex');
title('Velocity Ripple Peak-to-Peak', 'FontWeight', 'bold');
grid on; box on;
for i = 1:n_methods
    text(i, w_pp_vals(i), sprintf('%.4f', w_pp_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontWeight', 'bold');
end

sgtitle('Efecto del Modelo de BEMF en Torque y Velocidad', ...
        'FontSize', 14, 'FontWeight', 'bold');

% =========================================================================
% FIGURA 3: Referencias de corriente (zoom estado estable)
% =========================================================================
figure('Color','w', 'Position', [150, 150, 1200, 400]);

subplot(1, 2, 1);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_zoom, results.(m).i_ref(1, z_start:end), 'Color', colors.(m), ...
         'LineWidth', 1.5, 'DisplayName', m); hold on;
end
ylabel('i^*_\alpha [A]', 'Interpreter', 'tex');
xlabel('Tiempo [ms]');
title('Referencias \alpha (cada método usa su BEMF)', 'FontWeight', 'bold');
legend('Location', 'best'); grid on; box on;

subplot(1, 2, 2);
for idx = 1:n_methods
    m = methods{idx};
    plot(t_zoom, results.(m).i_ref(2, z_start:end), 'Color', colors.(m), ...
         'LineWidth', 1.5, 'DisplayName', m); hold on;
end
ylabel('i^*_\beta [A]', 'Interpreter', 'tex');
xlabel('Tiempo [ms]');
title('Referencias \beta', 'FontWeight', 'bold');
legend('Location', 'best'); grid on; box on;

%% 6. REPORTE FINAL

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    RESULTADOS — LAZO CERRADO                        ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Método  │ Te Ripple │ Te P-P  │ ω Ripple │ ω P-P   │ ω SS err  ║\n');
fprintf('║         │   [%%]    │  [Nm]   │   [%%]    │ [rad/s] │ [rad/s]   ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════╣\n');
for idx = 1:n_methods
    m = methods{idx};
    fprintf('║ %-7s │  %6.2f  │ %.4f  │  %6.4f │ %.4f  │ %.4f    ║\n', ...
            m, results.(m).Te_ripple, results.(m).Te_pp, ...
            results.(m).w_ripple, results.(m).w_pp, results.(m).w_ss_error);
end
fprintf('╚══════════════════════════════════════════════════════════════════════╝\n\n');

fprintf('MEJORA DE LEARNED vs TRAP:\n');
imp_te = 100*(results.TRAP.Te_ripple - results.LEARNED.Te_ripple)/results.TRAP.Te_ripple;
imp_tp = 100*(results.TRAP.Te_pp - results.LEARNED.Te_pp)/results.TRAP.Te_pp;
imp_wp = 100*(results.TRAP.w_pp - results.LEARNED.w_pp)/results.TRAP.w_pp;
fprintf('  • Torque ripple:    %+.1f%%\n', imp_te);
fprintf('  • Torque P-P:       %+.1f%%\n', imp_tp);
fprintf('  • Velocidad P-P:    %+.1f%%\n\n', imp_wp);

fprintf('MEJORA DE ADALINE vs TRAP:\n');
imp_te_a = 100*(results.TRAP.Te_ripple - results.ADALINE.Te_ripple)/results.TRAP.Te_ripple;
imp_tp_a = 100*(results.TRAP.Te_pp - results.ADALINE.Te_pp)/results.TRAP.Te_pp;
imp_wp_a = 100*(results.TRAP.w_pp - results.ADALINE.w_pp)/results.TRAP.w_pp;
fprintf('  • Torque ripple:    %+.1f%%\n', imp_te_a);
fprintf('  • Torque P-P:       %+.1f%%\n', imp_tp_a);
fprintf('  • Velocidad P-P:    %+.1f%%\n\n', imp_wp_a);

fprintf('ADALINE vs LEARNED (NN):\n');
diff_te = results.ADALINE.Te_ripple - results.LEARNED.Te_ripple;
diff_wp = results.ADALINE.w_pp - results.LEARNED.w_pp;
fprintf('  • ΔTorque ripple:   %+.3f%% (positivo = ADALINE peor)\n', diff_te);
fprintf('  • Δω P-P:           %+.6f rad/s\n', diff_wp);
if abs(diff_te) < 0.5
    fprintf('  → ADALINE ≈ LEARNED: rendimiento equivalente con 40 params vs ~25k\n\n');
elseif diff_te < 0
    fprintf('  → ADALINE SUPERA a LEARNED (solución óptima global > mínimo local)\n\n');
else
    fprintf('  → LEARNED ligeramente mejor (NN captura no-linealidades de Gibbs)\n\n');
end

fprintf('MEJORA DE LEARNED vs SIN:\n');
imp_te2 = 100*(results.SIN.Te_ripple - results.LEARNED.Te_ripple)/results.SIN.Te_ripple;
imp_wp2 = 100*(results.SIN.w_pp - results.LEARNED.w_pp)/results.SIN.w_pp;
fprintf('  • Torque ripple:    %+.1f%%\n', imp_te2);
fprintf('  • Velocidad P-P:    %+.1f%%\n\n', imp_wp2);

fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('SIMULACIÓN COMPLETADA\n');
fprintf('═══════════════════════════════════════════════════════════\n');

%% FUNCIONES AUXILIARES

function abc = get_trapezoidal_abc(theta)
    phases = [0, 2*pi/3, 4*pi/3];
    abc = zeros(3, 1);
    for i = 1:3
        ti = mod(theta - phases(i) + pi/6, 2*pi);
        if ti < pi/6
            val = ti * (6/pi);
        elseif ti < 5*pi/6
            val = 1;
        elseif ti < 7*pi/6
            val = 1 - (ti - 5*pi/6) * (6/pi);
        elseif ti < 11*pi/6
            val = -1;
        else
            val = -1 + (ti - 11*pi/6) * (6/pi);
        end
        abc(i) = val;
    end
end

function ab = clarke_transform(abc)
    T_clarke = (2/3) * [1, -0.5, -0.5;
                        0, sqrt(3)/2, -sqrt(3)/2];
    ab = T_clarke * abc;
end