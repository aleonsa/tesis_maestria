function corchetes_signados = Csign(x,m)
% Csing devuelve los corchetes signados elevados a la m  .
%   C = Csign(x,m) .
%   C = |x|^m*sign(x).

    corchetes_signados=abs(x)^(m)*sign(x);
end