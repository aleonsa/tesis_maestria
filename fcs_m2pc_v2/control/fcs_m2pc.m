function u_applied = fcs_m2pc(i_ab, i_ref, e_model, w_m, shape_model, p)
%FCS_M2PC  Finite Control Set Model-2-step Predictive Current Controller.
%
%   u_applied = fcs_m2pc(i_ab, i_ref, e_model, w_m, shape_model, p)
%
%   i_ab        : [2x1] current state in αβ [A]
%   i_ref       : [2x1] current reference in αβ [A]
%   e_model     : [2x1] model BEMF in αβ [V]  (= Ke * w_m * shape_model)
%   w_m         : mechanical speed [rad/s]
%   shape_model : [2x1] normalized BEMF shape (for sector detection)
%   p           : params struct (uses p.V_ab, p.L, p.Ts, p.R, p.rho_divs,
%                                p.w_m_threshold)
%
%   At startup (|w_m| < threshold): exhaustive search over all 8 VSI vectors.
%   At normal speed: M2PC with 3 vectors (2 active + zero) in the BEMF sector,
%   with duty-cycle time-division (rho_divs subdivisions).

if abs(w_m) < p.w_m_threshold
    % ── Startup: search all 8 voltage vectors ──────────────────────────
    min_cost  = inf;
    u_applied = [0; 0];
    for v = 1:8
        u_try  = p.V_ab(:, v);
        i_pred = i_ab + (p.Ts / p.L) * (u_try - e_model - p.R * i_ab);
        cost   = sum((i_ref - i_pred).^2);
        if cost < min_cost
            min_cost  = cost;
            u_applied = u_try;
        end
    end

else
    % ── Normal: 3-vector M2PC within active sector ─────────────────────
    ang    = mod(atan2(shape_model(2), shape_model(1)), 2*pi);
    sector = min(floor(ang / (pi/3)) + 1, 6);

    vec_pairs = [1,2; 2,3; 3,4; 4,5; 5,6; 6,1];
    av = vec_pairs(sector, :);
    u1 = p.V_ab(:, av(1)+1);
    u2 = p.V_ab(:, av(2)+1);
    u0 = [0; 0];

    f1 = (1/p.L) * (u1 - e_model - p.R * i_ab);
    f2 = (1/p.L) * (u2 - e_model - p.R * i_ab);
    f0 = (1/p.L) * (u0 - e_model - p.R * i_ab);

    min_cost = inf;
    t1_opt   = 0;
    t2_opt   = 0;

    for nn = 0:p.rho_divs
        t1 = (nn / p.rho_divs) * p.Ts;
        for mm = 0:(p.rho_divs - nn)
            t2     = (mm / p.rho_divs) * p.Ts;
            t0     = p.Ts - t1 - t2;
            i_pred = i_ab + f1*t1 + f2*t2 + f0*t0;
            cost   = sum((i_ref - i_pred).^2);
            if cost < min_cost
                min_cost = cost;
                t1_opt   = t1;
                t2_opt   = t2;
            end
        end
    end

    u_applied = (u1*t1_opt + u2*t2_opt + u0*(p.Ts - t1_opt - t2_opt)) / p.Ts;
end

end
