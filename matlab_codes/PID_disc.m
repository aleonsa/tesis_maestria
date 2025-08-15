function [u] = PID_disc(e1, e2, sigma,k1,k2, k3)
    
    u1 = e1 + k2*Csign(e2,3/2);
    u = -k1*Csign(u1,1/3) - k3*sigma;

end

