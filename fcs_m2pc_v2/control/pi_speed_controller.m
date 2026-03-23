function [T_ref, pi_int] = pi_speed_controller(w_ref, w_m, pi_int, p)
%PI_SPEED_CONTROLLER  Discrete PI speed controller with anti-windup.
%
%   [T_ref, pi_int] = pi_speed_controller(w_ref, w_m, pi_int, p)
%
%   w_ref  : speed reference [rad/s]
%   w_m    : measured speed [rad/s]
%   pi_int : integrator state from previous step [N·m]
%   p      : params struct (uses p.pi_Kp, p.pi_Ki, p.pi_Tmax, p.pi_Tmin, p.Ts)
%
%   Anti-windup: if output saturates, the integration step is reversed.

w_error = w_ref - w_m;
pi_int  = pi_int + p.pi_Ki * w_error * p.Ts;
T_ref   = p.pi_Kp * w_error + pi_int;

if T_ref > p.pi_Tmax
    T_ref  = p.pi_Tmax;
    pi_int = pi_int - p.pi_Ki * w_error * p.Ts;  % undo integration
elseif T_ref < p.pi_Tmin
    T_ref  = p.pi_Tmin;
    pi_int = pi_int - p.pi_Ki * w_error * p.Ts;
end

end
