function ab = clarke_transform(abc)
%CLARKE_TRANSFORM  Applies the amplitude-invariant Clarke transformation.
%
%   ab = clarke_transform(abc)
%
%   abc : [3x1] three-phase quantity (current, voltage, or BEMF)
%   ab  : [2x1] αβ stationary-frame quantity

ab = (2/3) * [1,  -0.5,       -0.5; ...
              0,   sqrt(3)/2,  -sqrt(3)/2] * abc;

end
