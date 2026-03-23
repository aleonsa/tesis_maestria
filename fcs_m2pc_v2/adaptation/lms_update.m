function W = lms_update(W, x, s_target, mu)
%LMS_UPDATE  One LMS weight update step for the ADALINE BEMF estimator.
%
%   W = lms_update(W, x, s_target, mu)
%
%   W        : [2H x 2] current weight matrix  (W(:,1)=α, W(:,2)=β)
%   x        : [2H x 1] Fourier basis vector at current angle
%   s_target : [2x1] target BEMF shape signal
%              — Phase 1: real plant shape  (shape_real)
%              — Phase 2: observer-estimated shape  (s_obs)
%   mu       : LMS learning rate
%
%   Update rule:  ε = s_target - W'·x
%                 W(:,j) ← W(:,j) + μ·ε(j)·x

eps    = s_target - W' * x;
W(:,1) = W(:,1) + mu * eps(1) * x;
W(:,2) = W(:,2) + mu * eps(2) * x;

end
