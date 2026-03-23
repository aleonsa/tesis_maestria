function results = simulate_all_methods(USE_OBSERVER, p, H_adaline, W_optimal, ...
                                        lut, W_trap_init, W_sin_init)
%SIMULATE_ALL_METHODS  Closed-loop simulation for all 6 BEMF methods.
%
%   results = simulate_all_methods(USE_OBSERVER, p, H_adaline, W_optimal, ...
%                                  lut, W_trap_init, W_sin_init)
%
%   USE_OBSERVER : false → Phase 1: LMS uses real plant BEMF
%                  true  → Phase 2: LMS uses observer-estimated BEMF
%   p            : motor_params struct
%   H_adaline    : number of ADALINE harmonics
%   W_optimal    : [2H x 2] offline-optimal ADALINE weights
%   lut          : struct with fields:
%                    .theta      — angle grid [N x 1]
%                    .alpha_real — real BEMF α shape [N x 1]
%                    .beta_real  — real BEMF β shape [N x 1]
%                    .alpha_nn   — deep-NN BEMF α shape [N x 1]
%                    .beta_nn    — deep-NN BEMF β shape [N x 1]
%   W_trap_init  : [2H x 2] ADALINE init from trapezoidal Fourier fit
%   W_sin_init   : [2H x 2] ADALINE init from sinusoidal (fundamental only)
%
%   Returns a struct with one field per method, each containing the
%   simulation log and steady-state metrics.

methods     = {'TRAP','SIN','LEARNED','ADALINE_OFF','ADALINE_ON_T','ADALINE_ON_S'};
n_methods   = length(methods);
Total_Steps = round(p.t_final / p.Ts);
results     = struct();

for method_idx = 1:n_methods
    method_name = methods{method_idx};
    fprintf('  ► %-16s ...', method_name);
    tic;

    % ── Initial states ──────────────────────────────────────────────────
    i_ab    = [0; 0];
    w_m     = 0;
    theta_e = 0;
    pi_int  = 0;

    % ── ADALINE weights ─────────────────────────────────────────────────
    is_online = false;
    switch method_name
        case 'ADALINE_ON_T',  W_ada = W_trap_init;  is_online = true;
        case 'ADALINE_ON_S',  W_ada = W_sin_init;   is_online = true;
        case 'ADALINE_OFF',   W_ada = W_optimal;
        otherwise,            W_ada = [];
    end

    % ── Observer state (used only in Phase 2 for online methods) ────────
    u_prev      = [0; 0];
    i_meas_prev = [0; 0];
    s_obs_prev  = [0; 0];

    % ── Pre-allocate log ─────────────────────────────────────────────────
    log.i             = zeros(2, Total_Steps);
    log.i_ref         = zeros(2, Total_Steps);
    log.Te            = zeros(1, Total_Steps);
    log.w_m           = zeros(1, Total_Steps);
    log.T_ref         = zeros(1, Total_Steps);
    log.T_load_actual = zeros(1, Total_Steps);
    if is_online
        log.W_error = zeros(1, Total_Steps);
    end

    % ── Simulation loop ──────────────────────────────────────────────────
    for k = 1:Total_Steps

        % Load torque: step at halfway mark
        if k < Total_Steps/2
            T_load_k = p.T_load;
        else
            T_load_k = p.T_load + p.T_load_step;
        end

        % ── A. Real BEMF (plant truth, from LUT) ───────────────────────
        theta_mod  = mod(theta_e, 2*pi);
        shape_real = bemf_lut(theta_mod, lut.theta, lut.alpha_real, lut.beta_real);
        e_real     = p.Ke * w_m * shape_real;

        % ── B. Model BEMF (method-specific) ────────────────────────────
        switch method_name
            case 'SIN'
                shape_model = bemf_sinusoidal(theta_e);
                x_f = [];

            case 'TRAP'
                shape_model = bemf_trapezoidal(theta_e);
                x_f = [];

            case 'LEARNED'
                shape_model = bemf_lut(theta_mod, lut.theta, lut.alpha_nn, lut.beta_nn);
                x_f = [];

            case {'ADALINE_OFF', 'ADALINE_ON_T', 'ADALINE_ON_S'}
                x_f         = build_fourier_basis(theta_mod, H_adaline);
                shape_model = W_ada' * x_f;
        end
        e_model = p.Ke * w_m * shape_model;

        % ── C. LMS update (online methods only) ────────────────────────
        if is_online && abs(w_m) > p.w_m_threshold
            if USE_OBSERVER
                s_err = s_obs_prev;   % Phase 2: observed shape (1-step delay)
            else
                s_err = shape_real;   % Phase 1: ideal real shape
            end
            W_ada       = lms_update(W_ada, x_f, s_err, p.mu_lms);
            shape_model = W_ada' * x_f;
            e_model     = p.Ke * w_m * shape_model;
        end

        % ── D. PI speed controller ──────────────────────────────────────
        [T_ref, pi_int] = pi_speed_controller(p.w_ref, w_m, pi_int, p);

        % ── E. Current reference ────────────────────────────────────────
        i_ref = current_reference(T_ref, shape_model, w_m, p);

        % ── F. FCS-M2PC ─────────────────────────────────────────────────
        u_applied = fcs_m2pc(i_ab, i_ref, e_model, w_m, shape_model, p);

        % ── G. Plant step ────────────────────────────────────────────────
        [i_ab, w_m, theta_e, Te] = bldc_plant_step(i_ab, w_m, theta_e, ...
                                       u_applied, e_real, shape_real, T_load_k, p);

        % ── H. BEMF observer (Phase 2 only, result used at step k+1) ────
        if USE_OBSERVER && is_online
            % Noisy, quantized current measurement
            i_meas = i_ab + p.sigma_noise * randn(2, 1);
            i_meas = adc_quantize(i_meas, p.adc_bits, p.adc_range);

            if k > 1
                e_obs = bemf_observer(u_prev, i_meas, i_meas_prev, p);
                if abs(w_m) > p.w_m_threshold
                    s_obs_prev = e_obs / (p.Ke * w_m);
                else
                    s_obs_prev = [0; 0];
                end
            end
            u_prev      = u_applied;
            i_meas_prev = i_meas;
        end

        % ── I. Log ──────────────────────────────────────────────────────
        log.i(:, k)           = i_ab;
        log.i_ref(:, k)       = i_ref;
        log.Te(k)             = Te;
        log.w_m(k)            = w_m;
        log.T_ref(k)          = T_ref;
        log.T_load_actual(k)  = T_load_k;
        if is_online
            log.W_error(k) = norm(W_ada - W_optimal, 'fro') / norm(W_optimal, 'fro');
        end

    end  % simulation loop

    log = compute_metrics(log, p);
    results.(method_name) = log;

    fprintf(' ✓ %.2fs | Ripple: %.2f%% | ω_pp: %.4f\n', ...
            toc, log.Te_ripple, log.w_pp);
end  % method loop

end
