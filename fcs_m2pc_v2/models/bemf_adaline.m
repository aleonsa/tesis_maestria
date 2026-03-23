function shape = bemf_adaline(theta_mod, W, H)
%BEMF_ADALINE  Evaluates the ADALINE BEMF model at a given angle.
%
%   shape = bemf_adaline(theta_mod, W, H)
%
%   theta_mod : electrical angle in [0, 2π] [rad]
%   W         : [2H x 2] weight matrix  (W(:,1) = α weights, W(:,2) = β)
%   H         : number of harmonics
%   shape     : [2x1] normalized αβ BEMF shape vector
%
%   Computes:  x = build_fourier_basis(theta_mod, H)
%              shape = W' * x
%
%   Note: when the Fourier basis x is also needed (e.g. for LMS updates),
%   call build_fourier_basis directly and compute W'*x inline to avoid
%   computing x twice.

x     = build_fourier_basis(theta_mod, H);
shape = W' * x;

end
