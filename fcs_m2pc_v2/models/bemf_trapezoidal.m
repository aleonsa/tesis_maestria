function shape = bemf_trapezoidal(theta_e)
%BEMF_TRAPEZOIDAL  Normalized αβ BEMF shape using an ideal trapezoidal waveform.
%
%   shape = bemf_trapezoidal(theta_e)
%
%   theta_e : electrical angle [rad]
%   shape   : [2x1] normalized αβ BEMF shape vector

abc   = trap_abc(theta_e);
shape = clarke_transform(abc);

end

% ── Local helper ─────────────────────────────────────────────────────────
function abc = trap_abc(theta)
%TRAP_ABC  Generates the three-phase trapezoidal BEMF waveform.
phases = [0, 2*pi/3, 4*pi/3];
abc    = zeros(3, 1);
for i = 1:3
    ti = mod(theta - phases(i) + pi/6, 2*pi);
    if     ti < pi/6,        val =  ti * (6/pi);
    elseif ti < 5*pi/6,      val =  1;
    elseif ti < 7*pi/6,      val =  1 - (ti - 5*pi/6) * (6/pi);
    elseif ti < 11*pi/6,     val = -1;
    else,                    val = -1 + (ti - 11*pi/6) * (6/pi);
    end
    abc(i) = val;
end
end
