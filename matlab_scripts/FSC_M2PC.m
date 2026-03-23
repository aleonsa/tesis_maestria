%% Simulación FCS-M2PC (Algoritmo 3): Multimodo
% Referencia: Coronado-Andrade et al. (2025) - Sección III.B
% Implementa Ecuaciones 6a, 6b y optimización de dos vectores activos
clear all; close all; clc;

%% 1. CONFIGURACIÓN
BEMF_TYPE = 'TRAP';   % 'TRAP' o 'SIN'

% Parámetros (Tabla III)
p.R = 1.2; p.L = 2.05e-3; p.Ke = 0.40355; p.P = 4;
Vdc = 240; Ts = 50e-6; 

% Configuración Algoritmo 3
rho_divs = 7; % Divisiones para la malla de búsqueda (t1, t2)

%% 2. VECTORES (Alpha-Beta)
v_mag = 2/3 * Vdc; s3 = sqrt(3);
% V0..V7 (V0 y V7 nulos)
% Índices Matlab: 1=V0, 2=V1, ..., 8=V7
V_ab = [ [0;0], [v_mag;0], [v_mag/2; v_mag*s3/2], [-v_mag/2; v_mag*s3/2], ...
         [-v_mag;0], [-v_mag/2; -v_mag*s3/2], [v_mag/2; -v_mag*s3/2], [0;0] ];

%% 3. CONDICIONES
w_m = 40; I_target = 3.0; Total_Steps = 4000;
i_ab = [0; 0]; theta_e = 0;

% Logs
log_i = zeros(2, Total_Steps);
log_i_ref = zeros(2, Total_Steps);

%% 4. BUCLE DE SIMULACIÓN
for k = 1:Total_Steps
    
    % --- A. GENERACIÓN DE SEÑALES ---
    if strcmp(BEMF_TYPE, 'SIN')
        s_wave = [-sin(theta_e); cos(theta_e)];
        e_ab = p.Ke * w_m * s_wave;
        i_ref = I_target * s_wave;
    else
        shape_abc = get_trapezoidal_abc(theta_e);
        shape_ab = clarke_transform(shape_abc);
        e_ab = p.Ke * w_m * shape_ab;
        i_ref = I_target * shape_ab; 
    end
    
    % --- B. ALGORITMO FCS-M2PC (Algoritmo 3) ---
    
    % 1. Angulo y Sector
    vector_angle = atan2(e_ab(2), e_ab(1));
    vector_angle = mod(vector_angle, 2*pi);
    sector = floor(vector_angle / (pi/3)) + 1;
    if sector > 6, sector = 6; end
    
    % 2. Selección de Vectores Adyacentes (Vi, Vj)
    % Sector 1 -> V1, V2; Sector 2 -> V2, V3, etc.
    vec_pairs = [1,2; 2,3; 3,4; 4,5; 5,6; 6,1];
    curr_vecs = vec_pairs(sector, :); % Índices [Vi, Vj]
    
    % Obtener voltajes
    u1 = V_ab(:, curr_vecs(1)+1); % +1 por índice Matlab
    u2 = V_ab(:, curr_vecs(2)+1);
    
    % 3. Cálculo de Pendientes (Implementación Eq. 6a, 6b)
    % f1: Pendiente vector 1, f2: Pendiente vector 2, f0: Pendiente nulo
    f1 = (1/p.L) * (u1 - e_ab - p.R*i_ab); 
    f2 = (1/p.L) * (u2 - e_ab - p.R*i_ab);
    f0 = (1/p.L) * ([0;0] - e_ab - p.R*i_ab);
    
    min_cost = inf;
    t1_opt = 0; t2_opt = 0;
    
    % 4. Optimización de Tiempos (Grid Search t1, t2)
    % Restricción: t1 + t2 <= Ts
    for n = 0:rho_divs
        t1_test = (n / rho_divs) * Ts;
        
        % Solo iterar lo que queda de tiempo para t2
        max_m = rho_divs - n; 
        
        for m = 0:max_m
            t2_test = (m / rho_divs) * Ts;
            t0_test = Ts - t1_test - t2_test;
            
            % Predicción M2PC (Suma de contribuciones)
            % i(k+1) = i(k) + f1*t1 + f2*t2 + f0*t0
            i_pred = i_ab + f1*t1_test + f2*t2_test + f0*t0_test;
            
            % Costo
            cost = sum((i_ref - i_pred).^2);
            
            if cost < min_cost
                min_cost = cost;
                t1_opt = t1_test;
                t2_opt = t2_test;
            end
        end
    end
    
    % --- C. PLANTA ---
    % Voltaje promedio aplicado (combinación lineal de V1, V2 y V0)
    u_applied = (u1*t1_opt + u2*t2_opt) / Ts; % V0 es cero
    
    i_ab = i_ab + (1/p.L)*(u_applied - e_ab - p.R*i_ab)*Ts;
    theta_e = theta_e + w_m * Ts * (p.P/2);
    
    log_i(:, k) = i_ab; log_i_ref(:, k) = i_ref;
end

%% 5. VISUALIZACIÓN
t = (1:Total_Steps) * Ts * 1000; 

% Colores
c_meas_a = [0, 0.4470, 0.7410];      % Azul
c_meas_b = [0.8500, 0.3250, 0.0980]; % Naranja
c_ref_a  = [0.9290, 0.6940, 0.1250]; % Amarillo
c_ref_b  = [0.4940, 0.1840, 0.5560]; % Morado

figure('Color','w', 'Position', [100, 100, 800, 600]);

% --- Subplot 1 ---
subplot(2,1,1);
p1 = plot(t, log_i(1,:), 'Color', c_meas_a, 'LineWidth', 0.8); hold on;
p2 = plot(t, log_i(2,:), 'Color', c_meas_b, 'LineWidth', 0.8);
p3 = plot(t, log_i_ref(1,:), 'Color', c_ref_a, 'LineWidth', 2);
p4 = plot(t, log_i_ref(2,:), 'Color', c_ref_b, 'LineWidth', 2);

ylabel('Currents \alpha\beta [A]', 'Interpreter', 'tex', 'FontSize', 12);
grid on; box on; 
set(gca, 'FontSize', 11, 'FontName', 'Times New Roman');
xlim([0 max(t)]); ylim([-1.5 1.5]*max(abs(I_target)));

legend([p1 p2 p3 p4], ...
       {'i_{\alpha} (meas)', 'i_{\beta} (meas)', 'i_{\alpha} (ref)', 'i_{\beta} (ref)'}, ...
       'Orientation', 'horizontal', 'Location', 'northoutside', 'Box', 'off');

title(['FCS-M2PC Response - Mode: ' BEMF_TYPE], 'FontWeight', 'normal');

% --- Subplot 2 ---
subplot(2,1,2);
plot(t, log_i(1,:) - log_i_ref(1,:), 'b', 'LineWidth', 0.8); hold on;
plot(t, log_i(2,:) - log_i_ref(2,:), 'r', 'LineWidth', 0.8);

ylabel('Error [A]', 'FontSize', 12); 
xlabel('Time [ms]', 'FontSize', 12);
grid on; box on;
set(gca, 'FontSize', 11, 'FontName', 'Times New Roman');
xlim([0 max(t)]);
legend({'Error \alpha', 'Error \beta'}, 'Location', 'best', 'Box', 'off');

%% FUNCIONES
function abc = get_trapezoidal_abc(theta)
    phases = [0, 2*pi/3, 4*pi/3];
    abc = zeros(3,1);
    for i=1:3
        ti = mod(theta - phases(i), 2*pi);
        if ti < pi/6, val = ti*(6/pi);
        elseif ti < 5*pi/6, val = 1;
        elseif ti < 7*pi/6, val = 1 - (ti - 5*pi/6)*(6/pi);
        elseif ti < 11*pi/6, val = -1;
        else, val = -1 + (ti - 11*pi/6)*(6/pi);
        end
        abc(i) = val;
    end
end
function ab = clarke_transform(abc)
    ab = (2/3) * [1, -0.5, -0.5; 0, sqrt(3)/2, -sqrt(3)/2] * abc;
end