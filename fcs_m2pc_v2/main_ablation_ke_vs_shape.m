%% main_ablation_ke_vs_shape.m
% FCS-M2PC — Ablation: Ke mismatch vs shape mismatch (4 × 2 matrix)
% =========================================================================
%
% Plant: always uses real BEMF LUT (Ke_plant = 0.370 V·s/rad, from dSPACE).
% Controller: 4 shape models × 2 Ke values → 8 scenarios.
% USE_OBSERVER = false throughout (ideal LMS signal, isolates shape/Ke effects).
%
% Scenarios:
%   A1/A2  TRAP       + Ke_nom / Ke_real
%   B1/B2  SIN        + Ke_nom / Ke_real
%   C1/C2  ADALINE_OFF+ Ke_nom / Ke_real  (weights from bemf_real_adaline_lut)
%   D1/D2  ADALINE_ON + Ke_nom / Ke_real  (online LMS, init from TRAP Fourier)
%
% Requires: data/bemf_real_lut.mat, data/bemf_real_adaline_lut.mat
% =========================================================================
clear; close all; clc;
addpath(genpath('.'));

%% ── Ke constants ─────────────────────────────────────────────────────────

Ke_nom  = 0.40355;   % datasheet / controller (wrong for real motor)
Ke_real = 0.36977;   % measured from dSPACE recordings

%% ── Base parameters ─────────────────────────────────────────────────────

p_base = motor_params();    % all nominal; Ke/Kt overridden below
Total_Steps = round(p_base.t_final / p_base.Ts);
t_ms = (1:Total_Steps) * p_base.Ts * 1000;

% Plant: always uses real Ke and proportionally scaled Kt
p_plant     = p_base;
p_plant.Ke  = Ke_real;
p_plant.Kt  = p_base.Kt * (Ke_real / Ke_nom);

%% ── Load real BEMF data ──────────────────────────────────────────────────

for f = {'data/bemf_real_lut.mat', 'data/bemf_real_adaline_lut.mat'}
    if ~isfile(f{1})
        error('Missing: %s\nRun python_scripts/generate_real_bemf_lut.py first.', f{1});
    end
end

raw_lut = load('data/bemf_real_lut.mat');
raw_ada = load('data/bemf_real_adaline_lut.mat');

lut.theta      = raw_lut.lut_theta;
lut.alpha_real = raw_lut.lut_alpha_real;
lut.beta_real  = raw_lut.lut_beta_real;

W_opt     = raw_ada.adaline_W;     % [2H x 2] offline optimal for real BEMF
H_adaline = double(raw_ada.adaline_H);

%% ── ADALINE initialisations ─────────────────────────────────────────────

theta_grid = linspace(0, 2*pi, 5000)';
X_grid     = zeros(5000, 2*H_adaline);
for n = 1:H_adaline
    X_grid(:, 2*(n-1)+1) = cos(n * theta_grid);
    X_grid(:, 2*(n-1)+2) = sin(n * theta_grid);
end
trap_ab = zeros(5000, 2);
for i = 1:5000
    trap_ab(i, :) = bemf_trapezoidal(theta_grid(i))';
end
W_trap_init = X_grid \ trap_ab;   % Fourier fit to trapezoidal shape

%% ── Scenario table ───────────────────────────────────────────────────────
%
%  Each row: { shape_model_name , Ke_ctrl , short_label }
%  The 8 rows map to indices 1-8; even indices = Ke_real, odd = Ke_nom.

scenarios = {
    'TRAP',        Ke_nom,  'A1: TRAP + Ke_{nom}';
    'TRAP',        Ke_real, 'A2: TRAP + Ke_{real}';
    'SIN',         Ke_nom,  'B1: SIN + Ke_{nom}';
    'SIN',         Ke_real, 'B2: SIN + Ke_{real}';
    'ADALINE_OFF', Ke_nom,  'C1: ADAL\_OFF + Ke_{nom}';
    'ADALINE_OFF', Ke_real, 'C2: ADAL\_OFF + Ke_{real}';
    'ADALINE_ON',  Ke_nom,  'D1: ADAL\_ON + Ke_{nom}';
    'ADALINE_ON',  Ke_real, 'D2: ADAL\_ON + Ke_{real}';
};
n_scen = size(scenarios, 1);

%% ── Run 8 simulations ────────────────────────────────────────────────────

fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║  ABLATION: Ke MISMATCH vs SHAPE MISMATCH — 8 scenarios        ║\n');
fprintf('║  Plant  : Ke_plant = %.5f  (dSPACE measured)              ║\n', Ke_real);
fprintf('║  Observer: OFF (ideal LMS signal)                              ║\n');
fprintf('╚══════════════════════════════════════════════════════════════════╝\n\n');

results = cell(n_scen, 1);

for idx = 1:n_scen
    shape_name = scenarios{idx, 1};
    Ke_ctrl    = scenarios{idx, 2};
    label      = scenarios{idx, 3};

    % Controller params: same as base but with scenario-specific Ke/Kt
    p_ctrl     = p_base;
    p_ctrl.Ke  = Ke_ctrl;
    p_ctrl.Kt  = p_base.Kt * (Ke_ctrl / Ke_nom);

    fprintf('  ► %-32s ...', strrep(label, '\', ''));
    tic;

    res = run_scenario(p_plant, p_ctrl, shape_name, ...
                       W_opt, H_adaline, W_trap_init, lut, Total_Steps);
    res = compute_metrics(res, p_plant);

    res.label      = label;
    res.shape_name = shape_name;
    res.Ke_ctrl    = Ke_ctrl;
    results{idx}   = res;

    fprintf(' ✓ %.2fs | Ripple: %.2f%% | ω_pp: %.4f rad/s\n', ...
            toc, res.Te_ripple, res.w_pp);
end

%% ── Organise results for figures ─────────────────────────────────────────

% Ordered: TRAP, SIN, ADALINE_OFF, ADALINE_ON
% Ke_nom  rows: 1,3,5,7   Ke_real rows: 2,4,6,8
ripple_nom  = [results{1}.Te_ripple, results{3}.Te_ripple, ...
               results{5}.Te_ripple, results{7}.Te_ripple];
ripple_real = [results{2}.Te_ripple, results{4}.Te_ripple, ...
               results{6}.Te_ripple, results{8}.Te_ripple];

shape_labels_x = {'TRAP', 'SIN', 'ADALINE\_OFF', 'ADALINE\_ON'};
x = 1:4;

% ── Zoom window: last 2 electrical revolutions ──────────────────────────
w_ss = abs(results{6}.w_m(end));    % use C2 steady-state speed as reference
if w_ss < 1, w_ss = p_base.w_ref; end
samples_2rev = round(2 * 2*pi / (w_ss * p_base.Ts * (p_base.P/2)));
samples_2rev = min(samples_2rev, Total_Steps - 1);
z_start      = max(1, Total_Steps - samples_2rev);
t_zoom       = t_ms(z_start:end);

%% ── Figure 1: Grouped bar chart (main result) ────────────────────────────

figure('Color','w', 'Position',[50, 80, 900, 460]);

c_nom  = [0.82, 0.25, 0.15];   % warm red  — Ke wrong
c_real = [0.15, 0.50, 0.82];   % cool blue — Ke correct

bw = 0.36;
b1 = bar(x - bw/2, ripple_nom,  bw, 'FaceColor', c_nom,  'EdgeColor','none');
hold on;
b2 = bar(x + bw/2, ripple_real, bw, 'FaceColor', c_real, 'EdgeColor','none');

set(gca, 'XTick', x, 'XTickLabel', shape_labels_x, ...
    'XTickLabelRotation', 0, 'FontSize', 11);
ylabel('Torque Ripple [%]', 'FontSize', 12);
title('Ablation: K_e Mismatch vs Shape Mismatch', ...
      'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'tex');
legend([b1 b2], ...
       {sprintf('K_e = %.3f (nominal — wrong)', Ke_nom), ...
        sprintf('K_e = %.3f (real — correct)',  Ke_real)}, ...
       'Location', 'northwest', 'FontSize', 10, 'Interpreter', 'tex');
grid on; box on; ylim([0, max(ripple_nom)*1.18]);

for s = 1:4
    text(s - bw/2, ripple_nom(s), sprintf('%.1f%%', ripple_nom(s)), ...
         'HorizontalAlignment','center','VerticalAlignment','bottom', ...
         'FontWeight','bold','FontSize',9,'Color', c_nom*0.75);
    text(s + bw/2, ripple_real(s), sprintf('%.1f%%', ripple_real(s)), ...
         'HorizontalAlignment','center','VerticalAlignment','bottom', ...
         'FontWeight','bold','FontSize',9,'Color', c_real*0.70);
end

%% ── Figure 2: Stacked contribution chart ─────────────────────────────────

ke_contrib = max(0, ripple_nom - ripple_real);   % ripple added by Ke mismatch

figure('Color','w', 'Position',[100, 130, 820, 420]);

data_stack = [ripple_real(:), ke_contrib(:)];    % [4 x 2]
b = bar(x, data_stack, 0.55, 'stacked', 'EdgeColor', 'none');
b(1).FaceColor = c_real;
b(2).FaceColor = c_nom;

set(gca, 'XTick', x, 'XTickLabel', shape_labels_x, 'FontSize', 11);
ylabel('Torque Ripple [%]', 'FontSize', 12);
title('Desglose: Ripple de Shape vs Contribución de K_e', ...
      'FontSize', 13, 'FontWeight','bold','Interpreter','tex');
legend([b(1) b(2)], ...
       {'Base (shape + M2PC discretization)', 'Efecto adicional de K_e mismatch'}, ...
       'Location','northwest','FontSize',10,'Interpreter','tex');
grid on; box on;

for s = 1:4
    % base label (inside bar if tall enough)
    if ripple_real(s) > 2
        text(s, ripple_real(s)/2, sprintf('%.1f%%', ripple_real(s)), ...
             'HorizontalAlignment','center','VerticalAlignment','middle', ...
             'FontWeight','bold','FontSize',9,'Color','w');
    else
        text(s, ripple_real(s) + 0.3, sprintf('%.1f%%', ripple_real(s)), ...
             'HorizontalAlignment','center','VerticalAlignment','bottom', ...
             'FontWeight','bold','FontSize',8);
    end
    % Ke contribution label
    if ke_contrib(s) > 2
        text(s, ripple_real(s) + ke_contrib(s)/2, sprintf('+%.1f%%', ke_contrib(s)), ...
             'HorizontalAlignment','center','VerticalAlignment','middle', ...
             'FontWeight','bold','FontSize',9,'Color','k');
    end
end

%% ── Figure 3: Time-domain zoom (4 key scenarios) ─────────────────────────

% A1 = results{1}: TRAP + Ke_nom     (worst)
% A2 = results{2}: TRAP + Ke_real    (Ke fixed, shape wrong)
% B2 = results{4}: SIN  + Ke_real    (Ke correct, shape ~ok)
% C2 = results{6}: ADAL_OFF + Ke_real(best case)

zoom_idx   = [1, 2, 4, 6];
zoom_title = {'A1: TRAP + K_{e,nom} — worst', ...
              'A2: TRAP + K_{e,real} — Ke fixed', ...
              'B2: SIN + K_{e,real} — \approx correct', ...
              'C2: ADALINE\_OFF + K_{e,real} — best'};
zoom_color = {c_nom, [0.93, 0.55, 0.10], [0.10, 0.70, 0.50], c_real};

figure('Color','w','Position',[150,180,1200,520]);

for s = 1:4
    subplot(2, 2, s);
    r   = results{zoom_idx(s)};
    Te_z = r.Te(z_start:end);
    plot(t_zoom, Te_z, 'Color', zoom_color{s}, 'LineWidth', 1.3);
    hold on;
    yline(mean(Te_z), 'k--', 'LineWidth', 0.8, 'HandleVisibility','off');
    ylabel('T_e [Nm]', 'Interpreter','tex');
    title(zoom_title{s}, 'FontSize', 9, 'FontWeight','bold', 'Interpreter','tex');
    grid on; box on;
    text(0.97, 0.91, sprintf('Ripple: %.2f%%', r.Te_ripple), ...
         'Units','normalized','HorizontalAlignment','right', ...
         'FontSize', 9, 'FontWeight','bold', ...
         'BackgroundColor', [1 1 1 0.75]);
    if s > 2, xlabel('Tiempo [ms]'); end
end

sgtitle('Estado Estable — Torque (últimas 2 rev. eléctricas)', ...
        'FontSize',12,'FontWeight','bold');

%% ── Console report ───────────────────────────────────────────────────────

fprintf('\n╔═══════════════════════════════════════════════════════════════════╗\n');
fprintf('║         ABLATION: Ke MISMATCH vs SHAPE MISMATCH                ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Shape       │ Ke_nom [%%] │ Ke_real [%%] │ Δ Ke [%%] │  Δ Ke rel ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════════╣\n');

row_labels = {'TRAP       ','SIN        ','ADALINE_OFF','ADALINE_ON '};
for s = 1:4
    rn = ripple_nom(s);
    rr = ripple_real(s);
    d  = rn - rr;
    fprintf('║ %s │   %6.2f   │    %6.2f   │  %+6.2f  │  %+5.1f%%   ║\n', ...
            row_labels{s}, rn, rr, d, 100*d/max(rn,1e-3));
end

fprintf('╚═══════════════════════════════════════════════════════════════════╝\n\n');

fprintf('INTERPRETACIÓN:\n\n');

fprintf('  [1] Efecto de Ke mismatch aislado (misma shape, cambia Ke):\n');
fprintf('      TRAP:        %5.2f%% → %5.2f%%   (Δ = %+.2f%%)\n', ...
        ripple_nom(1), ripple_real(1), ripple_real(1)-ripple_nom(1));
fprintf('      SIN:         %5.2f%% → %5.2f%%   (Δ = %+.2f%%)\n', ...
        ripple_nom(2), ripple_real(2), ripple_real(2)-ripple_nom(2));
fprintf('      ADALINE_OFF: %5.2f%% → %5.2f%%   (Δ = %+.2f%%)\n', ...
        ripple_nom(3), ripple_real(3), ripple_real(3)-ripple_nom(3));
fprintf('      ADALINE_ON:  %5.2f%% → %5.2f%%   (Δ = %+.2f%%)\n\n', ...
        ripple_nom(4), ripple_real(4), ripple_real(4)-ripple_nom(4));

fprintf('  [2] Efecto de shape error aislado (Ke correcto, cambia shape):\n');
fprintf('      TRAP vs ADALINE_OFF (Ke_real):  %5.2f%% vs %5.2f%%  (shape dominates by %+.2f%%)\n', ...
        ripple_real(1), ripple_real(3), ripple_real(1)-ripple_real(3));
fprintf('      SIN  vs ADALINE_OFF (Ke_real):  %5.2f%% vs %5.2f%%  (shape error = %+.2f%%)\n\n', ...
        ripple_real(2), ripple_real(3), ripple_real(2)-ripple_real(3));

fprintf('  [3] Piso de ripple (ADALINE_OFF + Ke_real — M2PC discretization):\n');
fprintf('      %.2f%%\n\n', ripple_real(3));

fprintf('  [4] ¿ADALINE_ON absorbe el Ke error?\n');
fprintf('      ADALINE_ON Ke_nom: %.2f%%  vs  ADALINE_ON Ke_real: %.2f%%\n', ...
        ripple_nom(4), ripple_real(4));
if abs(ripple_nom(4) - ripple_real(4)) < 1.0
    fprintf('      → SÍ: la adaptación online compensa parcialmente el Ke mismatch ✓\n\n');
else
    fprintf('      → NO: el ripple sigue siendo significativamente mayor con Ke_nom ✗\n\n');
end

fprintf('  [5] Motor parameters used:\n');
fprintf('      Ke_plant = %.5f V·s/rad   Kt_plant = %.5f Nm/A\n', ...
        p_plant.Ke, p_plant.Kt);
fprintf('      Ke_nom   = %.5f V·s/rad   Ke_real  = %.5f V·s/rad  (err = %.2f%%)\n\n', ...
        Ke_nom, Ke_real, 100*(Ke_real-Ke_nom)/Ke_nom);

%% ── Local functions ──────────────────────────────────────────────────────

function log = run_scenario(p_plant, p_ctrl, shape_name, ...
                            W_opt, H_adaline, W_trap_init, lut, Total_Steps)
%RUN_SCENARIO  Closed-loop simulation for one ablation scenario.
%
%   p_plant    : motor params for plant (real Ke/Kt)
%   p_ctrl     : motor params for controller (scenario Ke/Kt)
%   shape_name : 'TRAP' | 'SIN' | 'ADALINE_OFF' | 'ADALINE_ON'
%   W_opt      : [2H x 2] offline-optimal ADALINE weights (real BEMF)
%   H_adaline  : number of Fourier harmonics
%   W_trap_init: [2H x 2] ADALINE init from trapezoidal Fourier fit
%   lut        : struct with .theta, .alpha_real, .beta_real
%   Total_Steps: number of simulation steps

% ── Initial states ──────────────────────────────────────────────────────
i_ab    = [0; 0];
w_m     = 0;
theta_e = 0;
pi_int  = 0;

% ── ADALINE weights ──────────────────────────────────────────────────────
is_online = strcmp(shape_name, 'ADALINE_ON');
switch shape_name
    case 'ADALINE_ON',  W_ada = W_trap_init;
    case 'ADALINE_OFF', W_ada = W_opt;
    otherwise,          W_ada = [];
end

% ── Pre-allocate logs ────────────────────────────────────────────────────
log.Te            = zeros(1, Total_Steps);
log.w_m           = zeros(1, Total_Steps);
log.T_ref         = zeros(1, Total_Steps);
log.T_load_actual = zeros(1, Total_Steps);
if is_online
    log.W_error = zeros(1, Total_Steps);
end

% ── Simulation loop ──────────────────────────────────────────────────────
for k = 1:Total_Steps

    % Load torque: step at halfway
    if k < Total_Steps/2
        T_load_k = p_plant.T_load;
    else
        T_load_k = p_plant.T_load + p_plant.T_load_step;
    end

    % A. Real BEMF from plant LUT (Ke_plant)
    theta_mod  = mod(theta_e, 2*pi);
    shape_real = bemf_lut(theta_mod, lut.theta, lut.alpha_real, lut.beta_real);
    e_real     = p_plant.Ke * w_m * shape_real;

    % B. Model BEMF (controller's estimate, uses Ke_ctrl)
    switch shape_name
        case 'SIN'
            shape_model = bemf_sinusoidal(theta_e);
            x_f = [];

        case 'TRAP'
            shape_model = bemf_trapezoidal(theta_e);
            x_f = [];

        case {'ADALINE_OFF', 'ADALINE_ON'}
            x_f         = build_fourier_basis(theta_mod, H_adaline);
            shape_model = W_ada' * x_f;
    end
    e_model = p_ctrl.Ke * w_m * shape_model;

    % C. LMS update (ADALINE_ON, USE_OBSERVER=false → ideal target = shape_real)
    if is_online && abs(w_m) > p_ctrl.w_m_threshold
        W_ada       = lms_update(W_ada, x_f, shape_real, p_ctrl.mu_lms);
        shape_model = W_ada' * x_f;
        e_model     = p_ctrl.Ke * w_m * shape_model;
    end

    % D. PI speed controller (uses p_ctrl.w_ref, p_ctrl.pi_*)
    [T_ref, pi_int] = pi_speed_controller(p_ctrl.w_ref, w_m, pi_int, p_ctrl);

    % E. Current reference (uses p_ctrl.Kt)
    i_ref = current_reference(T_ref, shape_model, w_m, p_ctrl);

    % F. FCS-M2PC (uses p_ctrl.R, L, Ts, V_ab — no Ke/Kt)
    u_applied = fcs_m2pc(i_ab, i_ref, e_model, w_m, shape_model, p_ctrl);

    % G. Plant step (uses p_plant.L, R, Kt, J, d, P)
    [i_ab, w_m, theta_e, Te] = bldc_plant_step(i_ab, w_m, theta_e, ...
                                    u_applied, e_real, shape_real, T_load_k, p_plant);

    % H. Log
    log.Te(k)            = Te;
    log.w_m(k)           = w_m;
    log.T_ref(k)         = T_ref;
    log.T_load_actual(k) = T_load_k;
    if is_online
        log.W_error(k) = norm(W_ada - W_opt, 'fro') / norm(W_opt, 'fro');
    end

end  % simulation loop

end  % run_scenario
