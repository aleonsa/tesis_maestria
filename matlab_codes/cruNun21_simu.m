clear; clc;

% Parámetros reales del robot
m1_real = 4;    % kg
m2_real = 2;    % kg
l1_real = 0.4;  % m
l2_real = 0.2;  % m
g = 9.81;       % m/s^2

% Parámetros nominales (para el controlador)
m1 = 4.2;
m2 = 2.1;
l1 = 0.42;
l2 = 0.21;

% Parámetros de simulación
Ts = 0.005;
Tf = 20;
t = 0:Ts:Tf;
N = length(t);

% Parámetros del controlador
% Para C1 (Finite-time control): pU = 1/3, pF = 1/2
pU = 1/3;
pF = 1/2;
P = diag([3, 3]);
D = diag([2, 2]);
PL = zeros(2,2);  % diag([0, 0])
DL = zeros(2,2);  % diag([0, 0])

% Trayectoria deseada
qd = zeros(2, N);
qd_dot = zeros(2, N);
qd_ddot = zeros(2, N);

for i = 1:N
    qd(:,i) = [-2 + 0.5*sin(0.5*t(i)); 3 + 0.5*cos(1.5*t(i))];
    qd_dot(:,i) = [0.25*cos(0.5*t(i)); -0.75*sin(1.5*t(i))];
    qd_ddot(:,i) = [-0.125*sin(0.5*t(i)); -1.125*cos(1.5*t(i))];
end

% Ruido aditivo en mediciones de posición y velocidad
n = [0.05*cos(100*t); 0.05*cos(100*t)];

% Perturbación externa
w = [0.5*cos(2*t); 0.5*cos(2*t)];

% Condiciones iniciales
q = [0; 0];
q_dot = [0; 0];

% Arrays para guardar datos
q_hist = zeros(2, N);
q_dot_hist = zeros(2, N);
q_tilde_hist = zeros(2, N);
q_tilde_dot_hist = zeros(2, N);
tau_hist = zeros(2, N);

% Función para calcular la matriz de inercia
function M = compute_M(q, m1, m2, l1, l2)
    delta1 = l2^2 * m2 + l1^2 * (m1 + m2);
    delta2 = l1 * l2 * m2;
    delta3 = l2^2 * m2;
    c2 = cos(q(2));
    
    M = [delta1 + 2*delta2*c2, delta3 + delta2*c2;
         delta3 + delta2*c2,   delta3];
end

% Función para calcular la matriz de Coriolis
function C = compute_C(q, q_dot, m1, m2, l1, l2)
    delta2 = l1 * l2 * m2;
    s2 = sin(q(2));
    
    C = [-delta2*s2*q_dot(2), -delta2*s2*(q_dot(1) + q_dot(2));
          delta2*s2*q_dot(1),  0];
end

% Función para calcular el vector de gravedad
function g_vec = compute_gravity(q, m1, m2, l1, l2, g)
    delta1 = l2^2 * m2 + l1^2 * (m1 + m2);
    delta3 = l2^2 * m2;
    c1 = cos(q(1));
    c12 = cos(q(1) + q(2));
    
    g_vec = [(1/l2)*g*delta3*c12 + (g/l1)*(delta1-delta3)*c1;
             (1/l2)*g*delta3*c12];
end

% Función para el término no lineal fraccionario
function result = csign(x, p)
    result = sign(x) .* (abs(x).^p);
end

% Loop principal de simulación
for i = 1:N
    % Guardar estado actual
    q_hist(:,i) = q;
    q_dot_hist(:,i) = q_dot;
    
    % Agregar ruido a las mediciones
    q_meas = q + n(:,i);
    q_dot_meas = q_dot + n(:,i);
    
    % Calcular errores
    q_tilde = q_meas - qd(:,i);
    q_tilde_dot = q_dot_meas - qd_dot(:,i);
    
    % Guardar errores
    q_tilde_hist(:,i) = q_tilde;
    q_tilde_dot_hist(:,i) = q_tilde_dot;
    
    % Matrices del modelo nominal (para el controlador)
    M_nom = compute_M(q_meas, m1, m2, l1, l2);
    C_nom = compute_C(q_meas, q_dot_meas, m1, m2, l1, l2);
    g_nom = compute_gravity(q_meas, m1, m2, l1, l2, g);
    
    % Implementación del controlador (Ecuación 19)
    % tau = -∇_{q_tilde}U_c(q_tilde, q_d) - ∇_{q_tilde_dot}F(q_tilde_dot) + M(q)q_ddot_d + C(q,q_dot)q_dot_d
    
    % Término de energía potencial del controlador: E_c(q_tilde) = (1/(pU+1)) * q_tilde^T * P * |q_tilde|^{pU}
    % ∇_{q_tilde}E_c = P * csign(q_tilde, pU)
    grad_Ec = P * csign(q_tilde, pU) + PL * q_tilde;
    
    % Término de disipación: F(q_tilde_dot) = (1/(pF+1)) * q_tilde_dot^T * D * |q_tilde_dot|^{pF} + (1/2) * q_tilde_dot^T * DL * q_tilde_dot
    % ∇_{q_tilde_dot}F = D * csign(q_tilde_dot, pF) + DL * q_tilde_dot
    grad_F = D * csign(q_tilde_dot, pF) + DL * q_tilde_dot;
    
    % Compensación de gravedad (usando el modelo nominal)
    gravity_comp = g_nom;
    
    % Término de compensación de la dinámica nominal
    feedforward = M_nom * qd_ddot(:,i) + C_nom * qd_dot(:,i);
    
    % Ley de control completa (Ecuación 19)
    tau = gravity_comp - grad_Ec - grad_F + feedforward;
    
    % Agregar perturbación externa
    tau = tau + w(:,i);
    
    % Guardar entrada de control
    tau_hist(:,i) = tau;
    
    % Dinámica real del robot (con parámetros reales)
    M_real = compute_M(q, m1_real, m2_real, l1_real, l2_real);
    C_real = compute_C(q, q_dot, m1_real, m2_real, l1_real, l2_real);
    g_real = compute_gravity(q, m1_real, m2_real, l1_real, l2_real, g);
    
    % Ecuación de movimiento: M(q)q_ddot + C(q,q_dot)q_dot + g(q) = tau
    q_ddot = M_real \ (tau - C_real * q_dot - g_real);
    
    % Integración (método de Euler)
    if i < N
        q = q + Ts * q_dot;
        q_dot = q_dot + Ts * q_ddot;
    end
end

% Gráficas de resultados
figure('Position', [100, 100, 1200, 800]);

% Error de posición q1
plot(t, q_tilde_hist(1,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, q_tilde_hist(2,:), 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Tiempo (s)');
ylabel('Error q_1 (rad)');
title('Error de Posición - Articulación 1');

% hold the program (octave)
input('Press enter to close');
