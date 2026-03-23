%% Simulación FCS-M2PC: Comparación TRAP vs SIN vs LEARNED
% =======================================================================
% Compara el desempeño del controlador FCS-M2PC con 3 modelos de BEMF:
%   1. TRAPEZOIDAL - Asume BEMF trapezoidal ideal
%   2. SENOIDAL    - Asume BEMF senoidal pura
%   3. LEARNED     - Usa LUT aprendida con red neuronal
%
% + Baseline FCS-MPC clásico con modelo trapezoidal para referencia.
%
% CORRECCIONES RESPECTO A VERSIÓN ANTERIOR:
%   - Cada método genera su referencia con SU propio modelo de BEMF
%     (antes todos usaban la BEMF real → anulaba la hipótesis)
%   - Torque real calculado como Te = (P/2w) * e_real' * i
%     (antes se usaba ||i|| como proxy, incorrecto para BEMF no-sinusoidal)
%   - Fase del modelo trapezoidal alineada con convención sin(theta)
%     (antes había desfase de 30° que sesgaba la comparación)
%   - Baseline FCS-MPC clásico agregado para validar el framework M2PC
%
% IMPORTANTE: La planta siempre usa la BEMF REAL (con armónicos)
%             generada por el script de Python
% =======================================================================
clear all; close all; clc;

%% 1. CARGA DE LA BEMF REAL (Lookup Table)
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║     SIMULACIÓN FCS-M2PC - COMPARACIÓN DE MÉTODOS          ║\n');
fprintf('║     (Correcciones: referencia por modelo, torque real)     ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

if isfile('bemf_lut.mat')
    load('bemf_lut.mat'); 
    fprintf('✓ LUT cargada exitosamente: %d puntos\n', length(lut_theta));
    fprintf('  • Alpha range: [%.3f, %.3f]\n', min(lut_alpha), max(lut_alpha));
    fprintf('  • Beta range:  [%.3f, %.3f]\n\n', min(lut_beta), max(lut_beta));
else
    error(['⚠ ERROR: No se encuentra bemf_lut.mat\n', ...
           'Por favor ejecuta primero el script de Python para generar la LUT.']);
end

%% 2. PARÁMETROS DEL SISTEMA
% Motor (Tabla III - Paper de referencia: BLY-344S-240V-3000)
p.R = 1.2;           % Resistencia de fase [Ohm]
p.L = 2.05e-3;       % Inductancia de fase [H]
p.Ke = 0.40355;      % Constante de BEMF [V·s/rad]
p.P = 4;             % Número de polos

% Sistema de control
Vdc = 240;           % Voltaje DC del bus [V]
Ts = 50e-6;          % Periodo de muestreo [s] = 20 kHz
rho_divs = 10;       % Divisiones para optimización M2PC

% Condiciones de operación
w_m = 80;            % Velocidad mecánica [rad/s]
I_target = 3.0;      % Corriente objetivo [A] (magnitud de referencia)
Total_Steps = 2000;  % Pasos de simulación

fprintf('PARÁMETROS DE SIMULACIÓN:\n');
fprintf('  • Velocidad:    %.1f rad/s (%.1f RPM)\n', w_m, w_m*60/(2*pi));
fprintf('  • Corriente:    %.1f A\n', I_target);
fprintf('  • Periodo Ts:   %.1f μs (%.1f kHz)\n', Ts*1e6, 1/(Ts*1e3));
fprintf('  • Divisiones ρ: %d (M2PC)\n', rho_divs);
fprintf('  • Duración:     %.1f ms\n\n', Total_Steps*Ts*1000);

%% 3. VECTORES DE VOLTAJE (Alpha-Beta Frame)
% Inversores VSI: V0(000) a V7(111)
v_mag = 2/3 * Vdc;
s3 = sqrt(3);

% Matriz de vectores [V0, V1, V2, V3, V4, V5, V6, V7]
V_ab = [ [0;0], ...                              % V0 (000)
         [v_mag;0], ...                          % V1 (100)
         [v_mag/2; v_mag*s3/2], ...             % V2 (110)
         [-v_mag/2; v_mag*s3/2], ...            % V3 (010)
         [-v_mag;0], ...                         % V4 (011)
         [-v_mag/2; -v_mag*s3/2], ...           % V5 (001)
         [v_mag/2; -v_mag*s3/2], ...            % V6 (101)
         [0;0] ];                                % V7 (111)

%% 4. SIMULACIÓN DE LOS MÉTODOS
% CLASSIC = FCS-MPC clásico (1 vector por Ts) con modelo TRAP → baseline
% TRAP    = FCS-M2PC con modelo trapezoidal
% SIN     = FCS-M2PC con modelo senoidal
% LEARNED = FCS-M2PC con modelo aprendido (NN/LUT)
methods = {'CLASSIC', 'TRAP', 'SIN', 'LEARNED'};
n_methods = length(methods);
results = struct();

fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('INICIANDO SIMULACIONES\n');
fprintf('═══════════════════════════════════════════════════════════\n\n');

for method_idx = 1:n_methods
    method_name = methods{method_idx};
    fprintf('► Simulando método: %s\n', method_name);
    tic;
    
    % ===== RESET DE CONDICIONES INICIALES =====
    i_ab = [0; 0];        % Corrientes Alpha-Beta iniciales
    theta_e = 0;          % Ángulo eléctrico inicial
    
    % ===== ARRAYS DE ALMACENAMIENTO =====
    log.i = zeros(2, Total_Steps);           % Corrientes medidas
    log.i_ref = zeros(2, Total_Steps);       % Corrientes de referencia
    log.e_model = zeros(2, Total_Steps);     % BEMF del modelo (controlador)
    log.e_real = zeros(2, Total_Steps);      % BEMF real (planta)
    log.error = zeros(2, Total_Steps);       % Error de seguimiento
    log.sector = zeros(1, Total_Steps);      % Sector activo
    log.Te = zeros(1, Total_Steps);          % Torque electromagnético real
    
    % ===== BUCLE PRINCIPAL DE SIMULACIÓN =====
    for k = 1:Total_Steps
        
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        % A. BEMF REAL DEL MOTOR (Siempre usa la LUT real)
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        theta_mod = mod(theta_e, 2*pi);
        
        % Interpolación de la LUT real
        l_alpha = interp1(lut_theta, lut_alpha_real, theta_mod, 'linear', 'extrap');
        l_beta  = interp1(lut_theta, lut_beta_real,  theta_mod, 'linear', 'extrap');
        
        shape_real = [l_alpha; l_beta];
        e_real = p.Ke * w_m * shape_real;  % BEMF real [V]
        
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        % B. BEMF DEL MODELO (Depende del método del controlador)
        %    Esto es lo que el controlador "cree" que es la BEMF.
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        switch method_name
            case 'SIN'
                % Modelo senoidal puro: e_alpha = sin(theta), e_beta = -cos(theta)
                shape_model = [sin(theta_e); -cos(theta_e)];
                e_model = p.Ke * w_m * shape_model;
                
            case {'TRAP', 'CLASSIC'}
                % Modelo trapezoidal (fase alineada con convención sin)
                shape_abc = get_trapezoidal_abc(theta_e);
                shape_model = clarke_transform(shape_abc);
                e_model = p.Ke * w_m * shape_model;
                
            case 'LEARNED'
                % Modelo aprendido (usa LUT de la red neuronal)
                l_alpha_nn = interp1(lut_theta, lut_alpha, theta_mod, 'linear', 'extrap');
                l_beta_nn  = interp1(lut_theta, lut_beta,  theta_mod, 'linear', 'extrap');
                shape_model = [l_alpha_nn; l_beta_nn];
                e_model = p.Ke * w_m * shape_model;
        end
        
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        % C. GENERACIÓN DE REFERENCIA DE CORRIENTE
        %    CADA MÉTODO USA SU PROPIO MODELO DE BEMF.
        %    Referencia óptima para torque constante:
        %       i* = I_target * e_hat / ||e_hat||^2
        %    El controlador genera i* usando e_hat (su modelo),
        %    NO la BEMF real. Si e_hat ≠ e_real, la referencia
        %    es subóptima y produce torque ripple.
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        norm_model_sq = shape_model' * shape_model;  % ||shape||^2
        if norm_model_sq > 1e-6
            i_ref = I_target * (shape_model / norm_model_sq);
        else
            i_ref = [0; 0];
        end
        
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        % D. ALGORITMO DE CONTROL
        %    CLASSIC: FCS-MPC clásico (1 vector, barrido de 8)
        %    TRAP/SIN/LEARNED: FCS-M2PC (Algoritmo 3 del paper)
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if strcmp(method_name, 'CLASSIC')
            % ---- FCS-MPC CLÁSICO: Barrido de los 8 vectores ----
            min_cost = inf;
            u_opt_alpha = 0; u_opt_beta = 0;
            
            for v_idx = 1:8
                u_try = V_ab(:, v_idx);
                
                % Predicción: i(k+1) = i(k) + (Ts/L)*(u - e_model - R*i)
                i_pred = i_ab + (Ts/p.L) * (u_try - e_model - p.R*i_ab);
                
                % Función de costo
                cost = sum((i_ref - i_pred).^2);
                if cost < min_cost
                    min_cost = cost;
                    u_opt_alpha = u_try(1);
                    u_opt_beta = u_try(2);
                end
            end
            
            u_applied = [u_opt_alpha; u_opt_beta];
            sector = 0;  % No aplica para clásico
            
        else
            % ---- FCS-M2PC: Optimización con subdivisiones ----
            
            % D.1. Determinar sector (basado en BEMF del modelo)
            vector_angle = atan2(e_model(2), e_model(1));
            vector_angle = mod(vector_angle, 2*pi);
            sector = floor(vector_angle / (pi/3)) + 1;
            if sector > 6, sector = 6; end
            
            % D.2. Seleccionar vectores adyacentes Vi y Vj
            vec_pairs = [1,2; 2,3; 3,4; 4,5; 5,6; 6,1];
            active_vecs = vec_pairs(sector, :);
            
            % Obtener vectores de voltaje (índice +1 por indexación MATLAB)
            u1 = V_ab(:, active_vecs(1)+1);  % Vector activo 1
            u2 = V_ab(:, active_vecs(2)+1);  % Vector activo 2
            u0 = [0; 0];                      % Vector nulo
            
            % D.3. Calcular pendientes
            % di/dt = (1/L) * (u - e_model - R*i)
            f1 = (1/p.L) * (u1 - e_model - p.R*i_ab);
            f2 = (1/p.L) * (u2 - e_model - p.R*i_ab);
            f0 = (1/p.L) * (u0 - e_model - p.R*i_ab);
            
            % D.4. Optimización de tiempos (Grid Search)
            min_cost = inf;
            t1_opt = 0;
            t2_opt = 0;
            
            for n = 0:rho_divs
                t1_test = (n / rho_divs) * Ts;
                max_m = rho_divs - n;
                
                for m = 0:max_m
                    t2_test = (m / rho_divs) * Ts;
                    t0_test = Ts - t1_test - t2_test;
                    
                    % Predicción M2PC: i(k+1) = i(k) + Σ(f_j * t_j)
                    i_pred = i_ab + f1*t1_test + f2*t2_test + f0*t0_test;
                    
                    % Función de costo: ||i_ref - i_pred||²
                    cost = sum((i_ref - i_pred).^2);
                    
                    if cost < min_cost
                        min_cost = cost;
                        t1_opt = t1_test;
                        t2_opt = t2_test;
                    end
                end
            end
            
            % Voltaje promedio aplicado durante Ts
            t0_opt = Ts - t1_opt - t2_opt;
            u_applied = (u1*t1_opt + u2*t2_opt + u0*t0_opt) / Ts;
        end
        
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        % E. PLANTA (Motor Real - Usa BEMF Real)
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        di_dt = (1/p.L) * (u_applied - e_real - p.R*i_ab);
        i_ab = i_ab + di_dt * Ts;
        
        % Actualizar ángulo eléctrico
        theta_e = theta_e + w_m * Ts * (p.P/2);
        
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        % F. REGISTRO DE DATOS
        % ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        log.i(:, k) = i_ab;
        log.i_ref(:, k) = i_ref;
        log.e_model(:, k) = e_model;
        log.e_real(:, k) = e_real;
        log.error(:, k) = i_ab - i_ref;
        log.sector(k) = sector;
        
        % Torque electromagnético real: Te = (P/2) * (e_real' * i) / w_m
        log.Te(k) = (p.P/2) * (e_real' * i_ab) / w_m;
    end
    
    % ===== CÁLCULO DE MÉTRICAS DE DESEMPEÑO =====
    % Usar últimos 50% de pasos (régimen permanente)
    steady_idx = (Total_Steps/2):Total_Steps;
    
    % RMSE de seguimiento de corriente
    rmse_alpha = sqrt(mean(log.error(1, steady_idx).^2));
    rmse_beta = sqrt(mean(log.error(2, steady_idx).^2));
    log.rmse = sqrt(rmse_alpha^2 + rmse_beta^2);
    
    % Error máximo
    error_mag = sqrt(sum(log.error.^2, 1));
    log.max_error = max(error_mag(steady_idx));
    
    % THD (Total Harmonic Distortion)
    log.thd = calculate_thd(log.i(:, steady_idx));
    
    % Error de modelo (BEMF real vs BEMF modelo)
    bemf_error = log.e_real - log.e_model;
    log.bemf_rmse = sqrt(mean(sum(bemf_error(:, steady_idx).^2, 1)));
    
    % ===== CÁLCULO DE TORQUE RIPPLE (Métrica real) =====
    % Torque real: Te = (P/2w) * e_real' * i
    Te_steady = log.Te(steady_idx);
    Te_mean = mean(Te_steady);
    Te_std = std(Te_steady);
    log.torque_ripple = 100 * (Te_std / abs(Te_mean));  % [%]
    log.Te_pp = max(Te_steady) - min(Te_steady);         % [Nm]
    log.Te_mean = Te_mean;
    
    % Magnitud de corriente (métrica secundaria)
    i_magnitude = sqrt(log.i(1,:).^2 + log.i(2,:).^2);
    log.i_mag = i_magnitude;
    log.i_mag_pp = max(i_magnitude(steady_idx)) - min(i_magnitude(steady_idx));
    
    % Guardar resultados
    results.(method_name) = log;
    
    elapsed = toc;
    fprintf('  ✓ Completado en %.2f s\n', elapsed);
    fprintf('    • RMSE:          %.4f A\n', log.rmse);
    fprintf('    • Max Error:     %.4f A\n', log.max_error);
    fprintf('    • THD:           %.2f %%\n', log.thd);
    fprintf('    • Te medio:      %.4f Nm\n', log.Te_mean);
    fprintf('    • Torque Ripple: %.2f %% (STD/mean)\n', log.torque_ripple);
    fprintf('    • Te P-P:        %.4f Nm\n', log.Te_pp);
    fprintf('    • BEMF Error:    %.4f V\n\n', log.bemf_rmse);
end

%% 5. VISUALIZACIÓN DE RESULTADOS

fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('GENERANDO GRÁFICAS\n');
fprintf('═══════════════════════════════════════════════════════════\n\n');

t_ms = (1:Total_Steps) * Ts * 1000;  % Tiempo en milisegundos

% Paleta de colores
colors.CLASSIC = [0.5, 0.5, 0.5];               % Gris (baseline)
colors.TRAP = [0.8500, 0.3250, 0.0980];         % Naranja
colors.SIN = [0, 0.4470, 0.7410];               % Azul
colors.LEARNED = [0.4660, 0.6740, 0.1880];      % Verde

% Métodos M2PC solamente (para gráficas de seguimiento)
m2pc_methods = {'TRAP', 'SIN', 'LEARNED'};
n_m2pc = length(m2pc_methods);

% =========================================================================
% FIGURA 1: Seguimiento de Corrientes (solo métodos M2PC)
% =========================================================================
figure('Color','w', 'Position', [50, 50, 1400, 900]);

for idx = 1:n_m2pc
    method = m2pc_methods{idx};
    
    % Subplot Alpha
    subplot(3, 3, (idx-1)*3 + 1);
    plot(t_ms, results.(method).i(1,:), 'Color', colors.(method), ...
         'LineWidth', 1.2); hold on;
    plot(t_ms, results.(method).i_ref(1,:), 'k--', 'LineWidth', 1.0);
    ylabel('i_\alpha [A]', 'Interpreter', 'tex', 'FontSize', 10);
    title(sprintf('%s - \\alpha', method), 'FontWeight', 'bold', 'FontSize', 11);
    grid on; box on; xlim([0 max(t_ms)]); ylim([-6 6]);
    set(gca, 'FontSize', 9);
    if idx == 1
        legend({'Measured', 'Reference'}, 'Location', 'northeast', 'FontSize', 8);
    end
    
    % Subplot Beta
    subplot(3, 3, (idx-1)*3 + 2);
    plot(t_ms, results.(method).i(2,:), 'Color', colors.(method), ...
         'LineWidth', 1.2); hold on;
    plot(t_ms, results.(method).i_ref(2,:), 'k--', 'LineWidth', 1.0);
    ylabel('i_\beta [A]', 'Interpreter', 'tex', 'FontSize', 10);
    title(sprintf('%s - \\beta', method), 'FontWeight', 'bold', 'FontSize', 11);
    grid on; box on; xlim([0 max(t_ms)]); ylim([-6 6]);
    set(gca, 'FontSize', 9);
    
    % Subplot Torque Real
    subplot(3, 3, (idx-1)*3 + 3);
    plot(t_ms, results.(method).Te, 'Color', colors.(method), ...
         'LineWidth', 1.2);
    ylabel('T_e [Nm]', 'Interpreter', 'tex', 'FontSize', 10);
    title(sprintf('%s - Torque (Ripple: %.2f%%)', method, results.(method).torque_ripple), ...
          'FontWeight', 'bold', 'FontSize', 11);
    grid on; box on; xlim([0 max(t_ms)]);
    set(gca, 'FontSize', 9);
    
    if idx == 3
        xlabel('Time [ms]', 'FontSize', 10);
    end
end

sgtitle('FCS-M2PC Current Tracking & Torque Performance', 'FontSize', 15, 'FontWeight', 'bold');

% =========================================================================
% FIGURA 2: Torque Ripple Analysis (TODAS las técnicas)
% =========================================================================
figure('Color','w', 'Position', [100, 100, 1400, 900]);

% Subplot 1-2: Torque zoom en estado estable (2 rev eléctricas)
subplot(3, 3, [1 2]);
zoom_samples = min(400, Total_Steps);
zoom_start = Total_Steps - zoom_samples;
t_zoom = t_ms(zoom_start:end);
for idx = 1:n_methods
    method = methods{idx};
    plot(t_zoom, results.(method).Te(zoom_start:end), ...
         'Color', colors.(method), 'LineWidth', 1.5, 'DisplayName', method); hold on;
end
ylabel('Torque T_e [Nm]', 'Interpreter', 'tex', 'FontSize', 11);
xlabel('Time [ms]', 'FontSize', 11);
title('Electromagnetic Torque (Steady State Zoom)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on; xlim([min(t_zoom) max(t_zoom)]);
legend('Location', 'best', 'FontSize', 10);
set(gca, 'FontSize', 10);

% Subplot 3: Magnitud de corriente zoom (para ver que no es constante)
subplot(3, 3, 3);
for idx = 1:n_methods
    method = methods{idx};
    plot(t_zoom, results.(method).i_mag(zoom_start:end), ...
         'Color', colors.(method), 'LineWidth', 1.2, 'DisplayName', method); hold on;
end
ylabel('|i| [A]', 'Interpreter', 'tex', 'FontSize', 11);
xlabel('Time [ms]', 'FontSize', 11);
title('Current Magnitude (varies for torque=const)', 'FontSize', 11, 'FontWeight', 'bold');
grid on; box on; xlim([min(t_zoom) max(t_zoom)]);
legend('Location', 'best', 'FontSize', 9);
set(gca, 'FontSize', 10);

% Subplot 4: RMSE de seguimiento
subplot(3, 3, 4);
rmse_vals = cellfun(@(m) results.(m).rmse, methods);
b = bar(rmse_vals);
b.FaceColor = 'flat';
b.CData = cell2mat(cellfun(@(m) colors.(m), methods, 'UniformOutput', false)');
set(gca, 'XTickLabel', methods, 'FontSize', 10);
ylabel('RMSE [A]', 'FontSize', 11);
title('Current Tracking RMSE', 'FontWeight', 'bold', 'FontSize', 11);
grid on; box on;
for i = 1:n_methods
    text(i, rmse_vals(i), sprintf('%.4f', rmse_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9, 'FontWeight', 'bold');
end

% Subplot 5: TORQUE RIPPLE (Métrica clave de la tesis)
subplot(3, 3, 5);
ripple_vals = cellfun(@(m) results.(m).torque_ripple, methods);
b = bar(ripple_vals);
b.FaceColor = 'flat';
b.CData = cell2mat(cellfun(@(m) colors.(m), methods, 'UniformOutput', false)');
set(gca, 'XTickLabel', methods, 'FontSize', 10);
ylabel('Torque Ripple [%]', 'FontSize', 11);
title('TORQUE RIPPLE (STD/mean of T_e)', 'FontWeight', 'bold', 'FontSize', 11);
grid on; box on;
for i = 1:n_methods
    text(i, ripple_vals(i), sprintf('%.2f%%', ripple_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9, 'FontWeight', 'bold');
end

% Subplot 6: Torque Peak-to-Peak
subplot(3, 3, 6);
te_pp_vals = cellfun(@(m) results.(m).Te_pp, methods);
b = bar(te_pp_vals);
b.FaceColor = 'flat';
b.CData = cell2mat(cellfun(@(m) colors.(m), methods, 'UniformOutput', false)');
set(gca, 'XTickLabel', methods, 'FontSize', 10);
ylabel('Te P-P [Nm]', 'FontSize', 11);
title('Torque Peak-to-Peak', 'FontWeight', 'bold', 'FontSize', 11);
grid on; box on;
for i = 1:n_methods
    text(i, te_pp_vals(i), sprintf('%.4f', te_pp_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9, 'FontWeight', 'bold');
end

% Subplot 7: BEMF Model Error
subplot(3, 3, 7);
bemf_err_vals = cellfun(@(m) results.(m).bemf_rmse, methods);
b = bar(bemf_err_vals);
b.FaceColor = 'flat';
b.CData = cell2mat(cellfun(@(m) colors.(m), methods, 'UniformOutput', false)');
set(gca, 'XTickLabel', methods, 'FontSize', 10);
ylabel('BEMF Error [V]', 'FontSize', 11);
title('BEMF Model Error (RMSE)', 'FontWeight', 'bold', 'FontSize', 11);
grid on; box on;
for i = 1:n_methods
    text(i, bemf_err_vals(i), sprintf('%.4f', bemf_err_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9, 'FontWeight', 'bold');
end

% Subplot 8: THD
subplot(3, 3, 8);
thd_vals = cellfun(@(m) results.(m).thd, methods);
b = bar(thd_vals);
b.FaceColor = 'flat';
b.CData = cell2mat(cellfun(@(m) colors.(m), methods, 'UniformOutput', false)');
set(gca, 'XTickLabel', methods, 'FontSize', 10);
ylabel('THD [%]', 'FontSize', 11);
title('Current THD', 'FontWeight', 'bold', 'FontSize', 11);
grid on; box on;
for i = 1:n_methods
    text(i, thd_vals(i), sprintf('%.2f%%', thd_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9, 'FontWeight', 'bold');
end

% Subplot 9: Histograma de Torque (distribución)
subplot(3, 3, 9);
steady_idx = (Total_Steps/2):Total_Steps;
for idx = 1:n_methods
    method = methods{idx};
    histogram(results.(method).Te(steady_idx), 30, ...
              'FaceColor', colors.(method), 'FaceAlpha', 0.5, ...
              'EdgeColor', 'none', 'DisplayName', method); hold on;
end
xlabel('Torque T_e [Nm]', 'Interpreter', 'tex', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title('Torque Distribution (Narrower = Less Ripple)', 'FontWeight', 'bold', 'FontSize', 11);
legend('Location', 'best', 'FontSize', 9);
grid on; box on;
set(gca, 'FontSize', 10);

sgtitle('Torque Ripple Analysis: Effect of BEMF Model Accuracy', 'FontSize', 15, 'FontWeight', 'bold');

% =========================================================================
% FIGURA 3: Comparación de BEMF (1 Revolución Eléctrica)
% =========================================================================
figure('Color','w', 'Position', [150, 150, 1200, 700]);

% Calcular índices para 1 revolución eléctrica en estado estable
samples_per_rev = round((2*pi / (w_m * Ts * p.P/2)));
start_idx = Total_Steps - samples_per_rev;
end_idx = Total_Steps;
t_rev_ms = (0:(end_idx-start_idx)) * Ts * 1000;

% Alpha component
subplot(2, 1, 1);
plot(t_rev_ms, results.TRAP.e_model(1, start_idx:end_idx), ...
     'Color', colors.TRAP, 'LineWidth', 1.5, 'DisplayName', 'TRAP Model'); hold on;
plot(t_rev_ms, results.SIN.e_model(1, start_idx:end_idx), ...
     'Color', colors.SIN, 'LineWidth', 1.5, 'DisplayName', 'SIN Model');
plot(t_rev_ms, results.LEARNED.e_model(1, start_idx:end_idx), ...
     'Color', colors.LEARNED, 'LineWidth', 1.5, 'DisplayName', 'LEARNED Model');
plot(t_rev_ms, results.TRAP.e_real(1, start_idx:end_idx), ...
     'k-', 'LineWidth', 2.5, 'DisplayName', 'Real BEMF');
ylabel('e_\alpha [V]', 'Interpreter', 'tex', 'FontSize', 12);
title('BEMF Comparison - Alpha Component (One Electrical Revolution)', ...
      'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on; box on;
set(gca, 'FontSize', 10);

% Beta component
subplot(2, 1, 2);
plot(t_rev_ms, results.TRAP.e_model(2, start_idx:end_idx), ...
     'Color', colors.TRAP, 'LineWidth', 1.5, 'DisplayName', 'TRAP Model'); hold on;
plot(t_rev_ms, results.SIN.e_model(2, start_idx:end_idx), ...
     'Color', colors.SIN, 'LineWidth', 1.5, 'DisplayName', 'SIN Model');
plot(t_rev_ms, results.LEARNED.e_model(2, start_idx:end_idx), ...
     'Color', colors.LEARNED, 'LineWidth', 1.5, 'DisplayName', 'LEARNED Model');
plot(t_rev_ms, results.TRAP.e_real(2, start_idx:end_idx), ...
     'k-', 'LineWidth', 2.5, 'DisplayName', 'Real BEMF');
ylabel('e_\beta [V]', 'Interpreter', 'tex', 'FontSize', 12);
xlabel('Time [ms]', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on; box on;
set(gca, 'FontSize', 10);

% =========================================================================
% FIGURA 4: Comparación de Referencias (nueva - muestra el efecto clave)
% =========================================================================
figure('Color','w', 'Position', [200, 200, 1200, 500]);

subplot(1, 2, 1);
plot(t_rev_ms, results.TRAP.i_ref(1, start_idx:end_idx), ...
     'Color', colors.TRAP, 'LineWidth', 1.5, 'DisplayName', 'TRAP ref'); hold on;
plot(t_rev_ms, results.SIN.i_ref(1, start_idx:end_idx), ...
     'Color', colors.SIN, 'LineWidth', 1.5, 'DisplayName', 'SIN ref');
plot(t_rev_ms, results.LEARNED.i_ref(1, start_idx:end_idx), ...
     'Color', colors.LEARNED, 'LineWidth', 1.5, 'DisplayName', 'LEARNED ref');
ylabel('i^*_\alpha [A]', 'Interpreter', 'tex', 'FontSize', 12);
xlabel('Time [ms]', 'FontSize', 12);
title('Current References - \alpha (each method uses its own BEMF model)', ...
      'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on; box on;
set(gca, 'FontSize', 10);

subplot(1, 2, 2);
plot(t_rev_ms, results.TRAP.i_ref(2, start_idx:end_idx), ...
     'Color', colors.TRAP, 'LineWidth', 1.5, 'DisplayName', 'TRAP ref'); hold on;
plot(t_rev_ms, results.SIN.i_ref(2, start_idx:end_idx), ...
     'Color', colors.SIN, 'LineWidth', 1.5, 'DisplayName', 'SIN ref');
plot(t_rev_ms, results.LEARNED.i_ref(2, start_idx:end_idx), ...
     'Color', colors.LEARNED, 'LineWidth', 1.5, 'DisplayName', 'LEARNED ref');
ylabel('i^*_\beta [A]', 'Interpreter', 'tex', 'FontSize', 12);
xlabel('Time [ms]', 'FontSize', 12);
title('Current References - \beta', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on; box on;
set(gca, 'FontSize', 10);

%% 6. REPORTE FINAL

fprintf('╔══════════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                           RESUMEN DE RESULTADOS                                  ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Method   │ RMSE  │ Te Ripple │ Te P-P  │ THD   │ BEMF Err │ Te Mean       ║\n');
fprintf('║          │  [A]  │    [%%]    │  [Nm]   │  [%%]  │   [V]    │  [Nm]         ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════════════════╣\n');
for idx = 1:n_methods
    method = methods{idx};
    fprintf('║ %-8s │ %.4f│  %6.2f   │ %.4f  │ %5.2f │  %.4f  │  %.4f       ║\n', ...
            method, results.(method).rmse, results.(method).torque_ripple, ...
            results.(method).Te_pp, results.(method).thd, ...
            results.(method).bemf_rmse, results.(method).Te_mean);
end
fprintf('╚══════════════════════════════════════════════════════════════════════════════════╝\n\n');

% Calcular mejoras relativas
baseline = 'CLASSIC';
fprintf('MEJORA RELATIVA (respecto a %s - FCS-MPC clásico):\n\n', baseline);

for idx = 2:n_methods
    method = methods{idx};
    
    imp_rmse = 100 * (results.(baseline).rmse - results.(method).rmse) / results.(baseline).rmse;
    imp_ripple = 100 * (results.(baseline).torque_ripple - results.(method).torque_ripple) / results.(baseline).torque_ripple;
    imp_tepp = 100 * (results.(baseline).Te_pp - results.(method).Te_pp) / results.(baseline).Te_pp;
    imp_thd = 100 * (results.(baseline).thd - results.(method).thd) / results.(baseline).thd;
    imp_bemf = 100 * (results.(baseline).bemf_rmse - results.(method).bemf_rmse) / results.(baseline).bemf_rmse;
    
    fprintf('► %s vs %s:\n', method, baseline);
    fprintf('  • RMSE:          %+.1f%%\n', imp_rmse);
    fprintf('  • Torque Ripple: %+.1f%% ⚡\n', imp_ripple);
    fprintf('  • Te P-P:        %+.1f%%\n', imp_tepp);
    fprintf('  • THD:           %+.1f%%\n', imp_thd);
    fprintf('  • BEMF Error:    %+.1f%%\n\n', imp_bemf);
end

% Mejora M2PC LEARNED vs M2PC TRAP (efecto puro del modelo aprendido)
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('EFECTO PURO DEL MODELO APRENDIDO (LEARNED vs TRAP, ambos M2PC):\n\n');
imp_ripple_lt = 100 * (results.TRAP.torque_ripple - results.LEARNED.torque_ripple) / results.TRAP.torque_ripple;
imp_tepp_lt = 100 * (results.TRAP.Te_pp - results.LEARNED.Te_pp) / results.TRAP.Te_pp;
imp_rmse_lt = 100 * (results.TRAP.rmse - results.LEARNED.rmse) / results.TRAP.rmse;
fprintf('  • Torque Ripple: %+.1f%% ⚡⚡\n', imp_ripple_lt);
fprintf('  • Te P-P:        %+.1f%%\n', imp_tepp_lt);
fprintf('  • RMSE:          %+.1f%%\n\n', imp_rmse_lt);

fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('SIMULACIÓN COMPLETADA EXITOSAMENTE\n');
fprintf('═══════════════════════════════════════════════════════════\n\n');

%% FUNCIONES AUXILIARES

function abc = get_trapezoidal_abc(theta)
    % Genera forma trapezoidal ideal para 3 fases.
    % CONVENCIÓN: Fase alineada con sin(theta).
    %   La fundamental del trapecio tiene máximo en theta = pi/2,
    %   cero en theta = 0, consistente con e_a ~ sin(theta_e).
    %   Se logra con offset de pi/6 en la definición de la rampa.
    phases = [0, 2*pi/3, 4*pi/3];
    abc = zeros(3, 1);
    
    for i = 1:3
        % Offset de pi/6 para alinear fundamental con sin(theta)
        ti = mod(theta - phases(i), 2*pi);
        
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
    % Transformación de Clarke: ABC -> Alpha-Beta
    T_clarke = (2/3) * [1, -0.5, -0.5; 
                        0, sqrt(3)/2, -sqrt(3)/2];
    ab = T_clarke * abc;
end

function thd = calculate_thd(currents)
    % Calcula THD (Total Harmonic Distortion) de las corrientes
    % Input: currents [2 x N] - corrientes alpha-beta
    
    i_alpha = currents(1, :);
    i_beta = currents(2, :);
    
    N = length(i_alpha);
    
    % FFT de ambas componentes
    Y_alpha = fft(i_alpha);
    Y_beta = fft(i_beta);
    
    % Magnitudes (solo mitad positiva del espectro)
    mag_alpha = abs(Y_alpha(1:floor(N/2)));
    mag_beta = abs(Y_beta(1:floor(N/2)));
    
    % Fundamental (segundo componente, índice 2, ignorando DC)
    if length(mag_alpha) > 1
        fund_alpha = mag_alpha(2);
        fund_beta = mag_beta(2);
        
        % Suma de armónicos (del 3 en adelante)
        if length(mag_alpha) > 2
            harmonics_alpha = sum(mag_alpha(3:end).^2);
            harmonics_beta = sum(mag_beta(3:end).^2);
        else
            harmonics_alpha = 0;
            harmonics_beta = 0;
        end
        
        % THD = sqrt(sum(harmonics^2)) / sqrt(sum(fundamental^2))
        fundamental_power = fund_alpha^2 + fund_beta^2;
        harmonic_power = harmonics_alpha + harmonics_beta;
        
        if fundamental_power > 0
            thd = 100 * sqrt(harmonic_power / fundamental_power);
        else
            thd = 0;
        end
    else
        thd = 0;
    end
end