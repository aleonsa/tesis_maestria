function log = compute_metrics(log, p)
%COMPUTE_METRICS  Adds steady-state performance metrics to a simulation log.
%
%   log = compute_metrics(log, p)
%
%   Metrics are computed over the last 15% of the simulation (post-transient).
%
%   Added fields:
%     log.Te_mean    — mean electromagnetic torque [N·m]
%     log.Te_std     — std of torque [N·m]
%     log.Te_ripple  — torque ripple: 100 * std/mean [%]
%     log.Te_pp      — torque peak-to-peak [N·m]
%     log.w_pp       — speed peak-to-peak [rad/s]
%     log.w_ss_error — absolute speed steady-state error [rad/s]

N        = length(log.Te);
post_idx = round(N * 0.85) : N;

Te_ss          = log.Te(post_idx);
log.Te_mean    = mean(Te_ss);
log.Te_std     = std(Te_ss);
log.Te_ripple  = 100 * log.Te_std / max(abs(log.Te_mean), 1e-6);
log.Te_pp      = max(Te_ss) - min(Te_ss);

w_ss           = log.w_m(post_idx);
log.w_pp       = max(w_ss) - min(w_ss);
log.w_ss_error = abs(p.w_ref - mean(w_ss));

end
