function shape = bemf_lut(theta_mod, lut_theta, lut_alpha, lut_beta)
%BEMF_LUT  Normalized αβ BEMF shape from a lookup table via linear interpolation.
%
%   shape = bemf_lut(theta_mod, lut_theta, lut_alpha, lut_beta)
%
%   theta_mod : electrical angle in [0, 2π] [rad]
%   lut_theta : [N x 1] angle grid [rad]
%   lut_alpha : [N x 1] α-axis BEMF shape values
%   lut_beta  : [N x 1] β-axis BEMF shape values
%   shape     : [2x1] normalized αβ BEMF shape vector
%
%   Used for both the real plant BEMF (lut_alpha_real/lut_beta_real)
%   and the deep-NN model BEMF (lut_alpha/lut_beta).

a = interp1(lut_theta, lut_alpha, theta_mod, 'linear', 'extrap');
b = interp1(lut_theta, lut_beta,  theta_mod, 'linear', 'extrap');
shape = [a; b];

end
