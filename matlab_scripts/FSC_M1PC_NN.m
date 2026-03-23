%% Tesis Maestría: Validación de Hipótesis (Step 1)
% Referencia: Apunte Técnico - 15 Dic 2025
% Objetivo: Cuantificar reducción de rizo por aprendizaje de BEMF
% Comparativa: Modelo Trapezoidal (Shao Ideal) vs Modelo Aprendido (Kernel/NN)

clear all; close all; clc;

%% 1. CARGA DEL "CEREBRO" (Core 2: Learning Loop)
% Cargamos la BEMF aprendida en Python (simulando el Core 2 del Delfino)
if isfile('bemf_lut.mat')
    load('bemf_lut.mat'); % Contiene: lut_theta, lut_alpha, lut_beta
    LEARNED_MODEL.theta = lut_theta;
    LEARNED_MODEL.alpha = lut_alpha;
    LEARNED_MODEL.beta  = lut_beta;
else
    error('Falta el modelo aprendido (bemf_lut.mat). Ejecutar script Python primero.');
end

%% 2. PARAMETROS DE LA PLANTA (El "Caso Axel")
p.R = 1.2; p.L = 2.05e-3; p.Ke = 0.40; p.P = 4;
Vdc = 240; Ts = 50e-6; 
w_m = 100; % Rad/s
I_target = 3.0; 
Total_Steps = 1000;

% Vectores FCS-MPC
v_mag = 2/3 * Vdc; s3 = sqrt(3);
V_ab = [ [0;0], [v_mag;0], [v_mag/2; v_mag*s3/2], [-v_mag/2; v_mag*s3/2], ...
         [-v_mag;0], [-v_mag/2; -v_mag*s3/2], [v_mag/2; -v_mag*s3/2], [0;0] ];

%% 3. SIMULACIÓN COMPARATIVA
% Escenario A: Controlador asume Trapezoidal (Situación Actual)
% Escenario B: Controlador usa BEMF Aprendida (Propuesta Tesis)
scenarios = {'TRAP_MODEL', 'LEARNED_MODEL'};
results = struct();

for i = 1:length(scenarios)
    sim_type = scenarios{i};
    fprintf('Ejecutando Simulación: %s ...\n', sim_type);
    
    % Inicialización
    i_ab = [0;0]; theta_e = 0;
    log_i = zeros(2, Total_Steps);
    log_Te = zeros(1, Total_Steps);
    
    for k = 1:Total_Steps
        % --- A. PLANTA FÍSICA (REALIDAD) ---
        % La planta SIEMPRE responde con la BEMF real (aprendida de los datos)
        % Esto simula el comportamiento físico verdadero del motor.
        theta_mod = mod(theta_e, 2*pi);
        e_real_alpha = interp1(LEARNED_MODEL.theta, LEARNED_MODEL.alpha, theta_mod);
        e_real_beta  = interp1(LEARNED_MODEL.theta, LEARNED_MODEL.beta, theta_mod);
        e_real = p.Ke * w_m * [e_real_alpha; e_real_beta];
        
        % --- B. MODELO DEL CONTROLADOR (PREDICCIÓN) ---
        if strcmp(sim_type, 'TRAP_MODEL')
            % El controlador "cree" que el motor es Trapezoidal Ideal (Error de Modelo)
            % Eq 2.9 Shao: Ignoramos armónicos -> Error en v_c
            [trap_a, trap_b] = get_ideal_trap(theta_mod);
            e_pred = p.Ke * w_m * [trap_a; trap_b];
            i_ref = I_target * [trap_a; trap_b]; % Referencia subóptima
        else
            % El controlador "sabe" la verdad (Kernel/NN)
            e_pred = e_real; 
            % Referencia optimizada para la forma real
            norm_e = norm([e_real_alpha; e_real_beta]);
            i_ref = I_target * ([e_real_alpha; e_real_beta] / norm_e);
        end
        
        % --- C. FCS-MPC (Core 1 Loop) ---
        min_cost = inf; u_opt = [0;0];
        
        % Barrido de vectores (Simplificado para M1PC por ahora)
        for v_idx = 0:7
            if v_idx==0 || v_idx==7, u_try=[0;0]; else, u_try=V_ab(:, v_idx); end
            
            % Ecuación de Predicción (Eq 2a, Coronado-Andrade et al.)
            % i(k+1) = i(k) + Ts/L * (v - e_pred - R*i)
            i_next = i_ab + (Ts/p.L)*(u_try - e_pred - p.R*i_ab);
            
            cost = sum((i_ref - i_next).^2);
            if cost < min_cost, min_cost = cost; u_opt = u_try; end
        end
        
        % --- D. EVOLUCIÓN PLANTA ---
        i_ab = i_ab + (Ts/p.L)*(u_opt - e_real - p.R*i_ab);
        theta_e = theta_e + w_m * Ts * (p.P/2);
        
        % Torque Real
        Te = (e_real' * i_ab) / (w_m * 2/p.P);
        
        log_i(:,k) = i_ab; log_Te(k) = Te;
    end
    results.(sim_type).i = log_i;
    results.(sim_type).Te = log_Te;
end

%% 4. REPORTE DE RESULTADOS
t = (1:Total_Steps)*Ts*1000;

figure('Color','w');
subplot(2,1,1); 
plot(t, results.TRAP_MODEL.i(1,:), 'r'); hold on;
plot(t, results.LEARNED_MODEL.i(1,:), 'b', 'LineWidth', 1.5);
title('Corriente Alpha: Modelo Trapezoidal (Rojo) vs Aprendido (Azul)');
ylabel('Amperes'); legend('Actual (Trap)', 'Propuesto (Learned)'); grid on;

subplot(2,1,2);
plot(t, results.TRAP_MODEL.Te, 'r'); hold on;
plot(t, results.LEARNED_MODEL.Te, 'b', 'LineWidth', 1.5);
title('Impacto en el Rizo de Par');
ylabel('Nm'); legend('Rizo Actual', 'Rizo Minimizado'); grid on;

% Cálculo de reducción de rizo
ripple_trap = max(results.TRAP_MODEL.Te(200:end)) - min(results.TRAP_MODEL.Te(200:end));
ripple_learn = max(results.LEARNED_MODEL.Te(200:end)) - min(results.LEARNED_MODEL.Te(200:end));
fprintf('Rizo Original: %.4f Nm\n', ripple_trap);
fprintf('Rizo con Aprendizaje: %.4f Nm\n', ripple_learn);
fprintf('REDUCCIÓN: %.2f%%\n', (1 - ripple_learn/ripple_trap)*100);

%% Auxiliar: Trapezoidal Ideal
function [fa, fb] = get_ideal_trap(theta)
    % Aproximación armónica de trapecio para suavidad numérica
    f1 = [-sin(theta); cos(theta)];
    f3 = 0.15 * [-sin(3*theta); cos(3*theta)]; 
    shape = f1 + f3; shape = shape/max(abs(shape));
    fa = shape(1,:); fb = shape(2,:);
end