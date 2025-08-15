%% Configuración inicial y parámetros
clear all; close all; clc;       

% Parámetros del sistema
theta1 = 3.0;   
theta2 = 1.0;   

% Parámetros de la simulación
Ts = 0.01;      
Tfinal = 20;    
t = 0:Ts:Tfinal; 

% CI
x1 = 0;         
x2 = 0;         

% Parámetros de la señal senoidal de ref    
w = 1;  
ref = sin(w * t); % ref
ref_dot = w*cos(w*t); % derivada de ref

%% Controlador PID discontinuo

k1 = 23.4;
k2 = 31.6;
k3 = 20;

% Variables para el controlador PID-disc
e1 = 0;
e2 = 0;
sigma = 0;

%% Bucle de simulación

% Vectores para guardar los resultados
x1_hist = zeros(size(t));
x2_hist = zeros(size(t));
u_hist = zeros(size(t));
e_hist = zeros(size(t));

for i = 1:length(t)
    
    % --- calcular el error ---
    e1 = x1 - ref(i);
    e2 = x2 - ref_dot(i);

    % -- variable deslizante ---
    sigma_dot = sign(e1);
    
    % --- calcular la acción de control ---
    u = PID_disc(e1,e2,sigma,k1,k2,k3);

    error_anterior = e1;
    
    % --- calcular las derivadas de los estados ---
    x1dot, x2dot = SimulateDynamics(theta1,theta2,u);

    % --- integrar para obtener los nuevos estados  ---
    x1 = x1 + x1dot * Ts;
    x2 = x2 + x2dot * Ts;
    sigma = sigma + sigma_dot*Ts;
    
    % --- Paso 4.5: Guardar los valores actuales para graficar  ---
    x1_hist(i) = x1;
    x2_hist(i) = x2;
    u_hist(i) = u;
    e_hist(i) = e1;
    
end

%% Graficas
figure; 

% Gráfica del seguimiento de la trayectoria
subplot(2,1,1);
plot(t, x1_hist, 'b', 'LineWidth', 2);
hold on;
plot(t, ref, 'r--', 'LineWidth', 1.5); % Graficamos la señal senoidal
title('Seguimiento de Trayectoria Senoidal');
xlabel('Tiempo (s)');
ylabel('x1');
legend('Respuesta del sistema', 'Referencia Senoidal');
grid on;

% Gráfica de la señal de control
subplot(2,1,2);
plot(t, u_hist, 'm', 'LineWidth', 2);
title('Señal de Control (u)');
xlabel('Tiempo (s)');
ylabel('Amplitud de u');
grid on;