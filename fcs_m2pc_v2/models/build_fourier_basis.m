function x = build_fourier_basis(theta, H)
%BUILD_FOURIER_BASIS  Constructs the Fourier feature vector at angle theta.
%
%   x = build_fourier_basis(theta, H)
%
%   Output x is a [2H x 1] vector:
%     x = [cos(θ), sin(θ), cos(2θ), sin(2θ), ..., cos(Hθ), sin(Hθ)]ᵀ
%
%   Used as the regressor for the ADALINE BEMF model.

x = zeros(2*H, 1);
for n = 1:H
    x(2*(n-1)+1) = cos(n * theta);
    x(2*(n-1)+2) = sin(n * theta);
end

end
