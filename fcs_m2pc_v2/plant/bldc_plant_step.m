function [i_ab, w_m, theta_e, Te] = bldc_plant_step(i_ab, w_m, theta_e, ...
                                                    u_applied, e_real, shape_real, ...
                                                    T_load, p)
%BLDC_PLANT_STEP  One discrete Euler step of the BLDC motor plant.
%
%   [i_ab, w_m, theta_e, Te] = bldc_plant_step(i_ab, w_m, theta_e, ...
%                               u_applied, e_real, shape_real, T_load, p)
%
%   Inputs (at step k):
%     i_ab      : [2x1] αβ current [A]
%     w_m       : mechanical speed [rad/s]
%     theta_e   : electrical angle [rad]
%     u_applied : [2x1] voltage vector applied by inverter [V]
%     e_real    : [2x1] real BEMF  = Ke * w_m * shape_real [V]
%     shape_real: [2x1] normalized real BEMF shape (needed for low-speed Te)
%     T_load    : load torque [N·m]
%     p         : params struct
%
%   Returns states at step k+1.

% ── Electrical dynamics ─────────────────────────────────────────────────
di_dt = (1/p.L) * (u_applied - e_real - p.R * i_ab);
i_ab  = i_ab + di_dt * p.Ts;

% ── Electromagnetic torque ──────────────────────────────────────────────
% Factor (3/2) from amplitude-invariant Clarke: P_abc = (3/2)*P_ab, Te = P_abc/w_m
if abs(w_m) > 0.5
    Te = (3/2) * (e_real' * i_ab) / w_m;
else
    % Low-speed: use shape directly to avoid division by near-zero w_m
    nr = norm(shape_real);
    if nr > 1e-6
        Te = p.Kt * (shape_real' * i_ab) / nr;
    else
        Te = 0;
    end
end

% ── Mechanical dynamics ─────────────────────────────────────────────────
dw_dt   = (Te - p.d * w_m - T_load) / p.J;
w_m     = w_m     + dw_dt * p.Ts;
theta_e = theta_e + w_m   * p.Ts * (p.P/2);

end
