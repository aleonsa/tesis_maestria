%% Simulación FCS-M1PC: Multimodo (Trapezoidal / Sinusoidal)
clear all; close all; clc;

%% 1. CONFIGURACIÓN
BEMF_TYPE = 'TRAP';   % 'TRAP' o 'SIN'

% Parámetros (Tabla III)
p.R = 1.2; p.L = 2.05e-3; p.Ke = 0.40355; p.P = 4;
Vdc = 240; Ts = 50e-6; rho_divs = 7;

%% 2. VECTORES (Alpha-Beta)
v_mag = 2/3 * Vdc; s3 = sqrt(3);
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
    
    % --- A. GENERACIÓN DE SEÑALES (SWITCH) ---
    if strcmp(BEMF_TYPE, 'SIN')
        % MODO SINUSOIDAL (Círculo perfecto)
        % e_alpha = -sin, e_beta = cos
        s_wave = [-sin(theta_e); cos(theta_e)];
        e_ab = p.Ke * w_m * s_wave;
        i_ref = I_target * s_wave;
        
    else
        % MODO TRAPEZOIDAL (Hexágono)
        shape_abc = get_trapezoidal_abc(theta_e);
        shape_ab = clarke_transform(shape_abc); % Vector de forma
        
        e_ab = p.Ke * w_m * shape_ab;
        i_ref = I_target * shape_ab; 
    end
    
    % --- B. ALGORITMO FCS-M1PC ---
    
    % 1. Angulo del Vector (Funciona para ambos casos gracias al atan2 real)
    vector_angle = atan2(e_ab(2), e_ab(1));
    vector_angle = mod(vector_angle, 2*pi);
    
    % 2. Sector
    sector = floor(vector_angle / (pi/3)) + 1;
    if sector > 6, sector = 6; end
    
    % 3. Candidatos
    vec_pairs = [1,2; 2,3; 3,4; 4,5; 5,6; 6,1];
    active_vectors = vec_pairs(sector, :);
    candidate_indices = [0, active_vectors]; 
    
    min_cost = inf; u_opt = [0;0]; t1_opt = 0;
    
    % 4. Optimización
    for v_idx = candidate_indices
        if v_idx == 0
            u_act = [0;0]; loop_range = 0;
        else
            u_act = V_ab(:, v_idx+1); loop_range = 1:rho_divs;
        end
        
        f1 = (1/p.L) * (u_act - e_ab - p.R*i_ab); 
        f0 = (1/p.L) * ([0;0] - e_ab - p.R*i_ab); 
        
        for r = loop_range
            if v_idx == 0, t1 = 0; else, t1 = (r / rho_divs) * Ts; end
            t0 = Ts - t1;
            i_pred = i_ab + f1*t1 + f0*t0;
            
            cost = sum((i_ref - i_pred).^2);
            
            if cost < min_cost
                min_cost = cost; u_opt = u_act; t1_opt = t1;
            end
        end
    end
    
    % --- C. PLANTA ---
    u_applied = u_opt * (t1_opt/Ts);
    i_ab = i_ab + (1/p.L)*(u_applied - e_ab - p.R*i_ab)*Ts;
    theta_e = theta_e + w_m * Ts * (p.P/2);
    
    log_i(:, k) = i_ab; log_i_ref(:, k) = i_ref;
end

%% 5. VISUALIZACIÓN
t = (1:Total_Steps) * Ts * 1000; % Tiempo en ms

% --- Definición de Colores (Estilo de la imagen de referencia) ---
c_meas_a = [0, 0.4470, 0.7410];      % Azul (Medida Alpha)
c_meas_b = [0.8500, 0.3250, 0.0980]; % Naranja (Medida Beta)
c_ref_a  = [0.9290, 0.6940, 0.1250]; % Amarillo Ocre (Ref Alpha)
c_ref_b  = [0.4940, 0.1840, 0.5560]; % Morado (Ref Beta)

figure('Color','w', 'Position', [100, 100, 800, 600]);

% --- Subplot 1: Corrientes ---
subplot(2,1,1);
% 1. Graficar Medidas (Fondo) - Líneas finas
p1 = plot(t, log_i(1,:), 'Color', c_meas_a, 'LineWidth', 0.8); hold on;
p2 = plot(t, log_i(2,:), 'Color', c_meas_b, 'LineWidth', 0.8);

% 2. Graficar Referencias (Frente) - Líneas sólidas y más gruesas
p3 = plot(t, log_i_ref(1,:), 'Color', c_ref_a, 'LineWidth', 2);
p4 = plot(t, log_i_ref(2,:), 'Color', c_ref_b, 'LineWidth', 2);

% Estética
ylabel('Currents \alpha\beta [A]', 'Interpreter', 'tex', 'FontSize', 12);
grid on; box on; 
set(gca, 'FontSize', 11, 'FontName', 'Times New Roman'); % Fuente tipo Paper
xlim([0 max(t)]); % Ajustar eje X al tiempo total
ylim([-1.5 1.5]*max(abs(I_target))); % Margen vertical dinámico

% Leyenda (Opcional: estilo horizontal arriba para ahorrar espacio)
legend([p1 p2 p3 p4], ...
       {'i_{\alpha} (meas)', 'i_{\beta} (meas)', 'i_{\alpha} (ref)', 'i_{\beta} (ref)'}, ...
       'Orientation', 'horizontal', 'Location', 'northoutside', 'Box', 'off');

title(['FCS-M1PC Response - Mode: ' BEMF_TYPE], 'FontWeight', 'normal');

% --- Subplot 2: Error ---
subplot(2,1,2);
plot(t, log_i(1,:) - log_i_ref(1,:), 'b', 'LineWidth', 0.8); hold on;
% Opcional: Graficar también error beta en gris
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