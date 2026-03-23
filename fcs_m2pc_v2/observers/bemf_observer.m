function e_obs = bemf_observer(u_prev, i_meas_curr, i_meas_prev, p)
%BEMF_OBSERVER  Discrete back-EMF observer from voltage and current measurements.
%
%   e_obs = bemf_observer(u_prev, i_meas_curr, i_meas_prev, p)
%
%   Implements the algebraic BEMF observer:
%     e_obs(k) = u(k-1) - R·i(k-1) - (L/Ts)·[i(k) - i(k-1)]
%
%   u_prev      : [2x1] voltage applied at step k-1 (assumed noise-free) [V]
%   i_meas_curr : [2x1] measured (noisy, quantized) current at step k [A]
%   i_meas_prev : [2x1] measured (noisy, quantized) current at step k-1 [A]
%   p           : params struct (uses p.R, p.L, p.Ts)
%
%   Note: the L/Ts term amplifies current noise. A larger Ts or smoother
%   current measurements reduce this effect.

e_obs = u_prev - p.R * i_meas_prev - (p.L / p.Ts) * (i_meas_curr - i_meas_prev);

end
