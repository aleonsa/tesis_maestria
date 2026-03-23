function shape = bemf_sinusoidal(theta_e)
%BEMF_SINUSOIDAL  Normalized αβ BEMF shape using an ideal sinusoidal waveform.
%
%   shape = bemf_sinusoidal(theta_e)
%
%   theta_e : electrical angle [rad]
%   shape   : [2x1] normalized αβ BEMF shape vector

shape = [sin(theta_e); -cos(theta_e)];

end
