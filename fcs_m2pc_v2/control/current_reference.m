function i_ref = current_reference(T_ref, shape_model, w_m, p)
%CURRENT_REFERENCE  Computes the αβ current reference from torque demand.
%
%   i_ref = current_reference(T_ref, shape_model, w_m, p)
%
%   T_ref       : torque reference from PI controller [N·m]
%   shape_model : [2x1] normalized αβ BEMF shape from the active model
%   w_m         : mechanical speed [rad/s]
%   p           : params struct (uses p.Kt, p.w_m_threshold)
%
%   At normal speed:  i_ref = T_ref / (Kt * ||shape||²) * shape
%   At low speed:     i_ref = T_ref / Kt * shape / ||shape||  (unit-shape fallback)

norm_sq = shape_model' * shape_model;

if norm_sq > 1e-6 && abs(w_m) > p.w_m_threshold
    i_ref = (T_ref / (p.Kt * norm_sq)) * shape_model;
else
    ns = sqrt(norm_sq);
    if ns > 1e-6
        i_ref = (T_ref / p.Kt) * (shape_model / ns);
    else
        i_ref = [0; 0];
    end
end

end
