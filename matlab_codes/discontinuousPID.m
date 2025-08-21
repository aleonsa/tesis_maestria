%% Configuración inicial y parámetros
clear all; close all; clc;       

% Parámetros del sistema
theta1 = 3.0;   
theta2 = 1.0;   

% Parámetros de la simulación
Ts = 3e-5;      
Tfinal = 30;    
t = 0:Ts:Tfinal; 

% CI
x1_0 = 0.5;         
x2_0 = 0.5;         

% Parámetros de la señal senoidal de ref    
w = 1;  
y0 = 1;
ref = y0*sin(w * t); % referencia
ref_dot = y0*w*cos(w*t);
ref_ddot = -y0*(w^2)*sin(w*t);

%% Controlador PID discontinuo

k1 = 23.4;
k2 = 31.6;
k3 = 20;

% Condiciones iniciales en coordenadas de error
e1 = x1_0 - ref(1);
e2 = x2_0 - ref_dot(1);

sigma = 0;

%% Bucle de simulación

% Vectores para guardar los resultados
e1_hist = zeros(size(t));
e2_hist = zeros(size(t));
u_hist = zeros(size(t));

% Loop de simulacion
for i = 1:length(t)    
    % -- variable deslizante ---
    sigma_dot = sign(e1);
    
    % --- calcular la acción de control ---
    u = PID_disc(e1,e2,sigma,k1,k2,k3); u = -u;

    e1dot = e2;
    e2dot = -theta1 * sin(e1 + ref(i)) - ref_ddot(i)  + theta2*u;

    % --- integrar para obtener los nuevos estados  ---
    e1 = e1 + e1dot * Ts;
    e2 = e2 + e2dot * Ts;
    sigma = sigma + sigma_dot*Ts;
    
    % --- Paso 4.5: Guardar los valores actuales para graficar  ---
    e1_hist(i) = e1;
    e2_hist(i) = e2;
    u_hist(i) = u;
    
end

%% Graficas
figure; 

% Gráfica del seguimiento de la trayectoria
subplot(2,1,1);
plot(t, e1_hist, 'b', 'LineWidth', 2);
title('Estado e_1 en el tiempo');
xlabel('Tiempo (s)');
ylabel('e1');
legend('Error de seguimiento');
grid on;

% Gráfica de la señal de control
subplot(2,1,2);
plot(t, u_hist, 'm', 'LineWidth', 2);
title('Señal de Control (u)');
xlabel('Tiempo (s)');
ylabel('Amplitud de u');
grid on;

input('Press enter to close');
