function p = motor_params()
%MOTOR_PARAMS  Returns a struct with all motor, inverter, and control parameters.
%
%   p = motor_params()
%
%   Motor: Anaheim BLY-344S-240V-3000
%   Control: FCS-M2PC in αβ frame, PI speed outer loop

% ── Motor ──────────────────────────────────────────────────────────────
p.R  = 1.2;          % Stator resistance [Ω]
p.L  = 2.05e-3;      % Stator inductance [H]
p.Ke = 0.40355;      % BEMF constant [V·s/rad]
p.Kt = 0.65997;      % Torque constant [N·m/A]
p.P  = 8;            % Number of poles (= 4 pole pairs; pmsm.p = 4 in Coronado's script)
p.J  = 0.00027948;   % Moment of inertia [kg·m²]
p.d  = 0.0006738;    % Viscous damping coefficient [N·m·s/rad]

% ── Inverter & sampling ────────────────────────────────────────────────
p.Vdc      = 240;    % DC bus voltage [V]
p.Ts       = 50e-6;  % Sampling period [s]
p.rho_divs = 10;     % Time subdivisions for M2PC duty-cycle search

% Voltage vectors in αβ (8 states of a 2-level VSI)
v_mag  = 2/3 * p.Vdc;
s3     = sqrt(3);
p.V_ab = [[0;0], [v_mag;0], [v_mag/2; v_mag*s3/2], ...
          [-v_mag/2; v_mag*s3/2], [-v_mag;0], ...
          [-v_mag/2; -v_mag*s3/2], [v_mag/2; -v_mag*s3/2], [0;0]];

% ── Speed PI controller ────────────────────────────────────────────────
p.pi_Kp   = 0.5;     % Proportional gain
p.pi_Ki   = 50;      % Integral gain
p.pi_Tmax =  5.0;    % Output saturation [N·m]
p.pi_Tmin = -5.0;

% ── Operating point ────────────────────────────────────────────────────
p.w_ref         = 80;    % Speed reference [rad/s]
p.T_load        = 0.5;   % Initial load torque [N·m]
p.T_load_step   = 1.0;   % Load step added at t_final/2 [N·m]
p.w_m_threshold = 5.0;   % Minimum speed for i_ref computation [rad/s]

% ── Simulation time ────────────────────────────────────────────────────
p.t_final = 0.3;     % [s]

% ── LMS (online ADALINE) ───────────────────────────────────────────────
p.mu_lms = 5e-3;     % Learning rate

% ── Observer / ADC (Phase 2) ───────────────────────────────────────────
p.sigma_noise = 0.05;  % Current measurement noise std [A]  (≈50 mA)
p.adc_bits    = 12;    % ADC resolution [bits]
p.adc_range   = 10.0;  % ADC full-scale [A]  (±10 A)

end
