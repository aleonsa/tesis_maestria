function export_thesis_results(results, methods, p, t_ms, tag)
%EXPORT_THESIS_RESULTS  Export Chapter 5 figures and metrics to files.
%
%   export_thesis_results(results, methods, p, t_ms, tag)
%
%   Saves the four figures created by plot_results (identified by their
%   'Tag' property) as PNG (300 dpi) and PDF (vector), plus the 6-method
%   metrics as metrics.csv and a booktabs LaTeX tabular (metrics.tex), all
%   under data/figures_thesis/<tag>/.
%
%   MUST be called AFTER plot_results so the tagged figures exist.
%
%   tag : string labelling the run/config (e.g. 'fase1_sintetica'). Used as
%         the output subfolder name so different configs don't overwrite.

if nargin < 5 || isempty(tag), tag = 'run'; end

outdir = fullfile('data', 'figures_thesis', tag);
if ~exist(outdir, 'dir'), mkdir(outdir); end

% ── Export the four tagged figures ───────────────────────────────────────
figmap = { 'fig_overview',    'overview';     % Fig 1: full-run overview
           'fig_steadystate', 'steadystate';  % Fig 2: zoom + bar charts
           'fig_convergence', 'convergence';  % Fig 3: ADALINE convergence
           'fig_tracking',    'tracking' };    % Fig 4: αβ current tracking
for k = 1:size(figmap, 1)
    h = findobj('Type', 'figure', 'Tag', figmap{k, 1});
    if isempty(h)
        warning('export:nofig', 'Figura "%s" no encontrada; ¿corriste plot_results?', figmap{k,1});
        continue;
    end
    base = fullfile(outdir, figmap{k, 2});
    exportgraphics(h(1), [base '.png'], 'Resolution', 300);
    exportgraphics(h(1), [base '.pdf'], 'ContentType', 'vector');
end

% ── Collect metrics into a table ─────────────────────────────────────────
nm     = numel(methods);
Method = methods(:);
Ripple = zeros(nm, 1); Te_pp = zeros(nm, 1); w_pp   = zeros(nm, 1);
w_err  = zeros(nm, 1); rmse_a = zeros(nm, 1); rmse_b = zeros(nm, 1);
nmse_a = zeros(nm, 1); nmse_b = zeros(nm, 1);
for i = 1:nm
    r         = results.(methods{i});
    Ripple(i) = r.Te_ripple;  Te_pp(i)  = r.Te_pp;  w_pp(i)   = r.w_pp;
    w_err(i)  = r.w_ss_error; rmse_a(i) = r.rmse_alpha; rmse_b(i) = r.rmse_beta;
    nmse_a(i) = r.nmse_alpha; nmse_b(i) = r.nmse_beta;
end
T = table(Method, Ripple, Te_pp, w_pp, w_err, rmse_a, rmse_b, nmse_a, nmse_b);
writetable(T, fullfile(outdir, 'metrics.csv'));

% ── LaTeX table (booktabs) ───────────────────────────────────────────────
fid = fopen(fullfile(outdir, 'metrics.tex'), 'w');
fprintf(fid, '%% Auto-generado por export_thesis_results.m\n');
fprintf(fid, '%% tag: %s | Ke=%.5f | w_ref=%.0f rad/s | Ts=%.0f us | P=%d polos\n', ...
        tag, p.Ke, p.w_ref, p.Ts*1e6, p.P);
fprintf(fid, '\\begin{tabular}{lrrrr}\n\\toprule\n');
fprintf(fid, ['M\\''etodo & Rizo [\\%%] & $T_e$ P-P [Nm] & ', ...
              '$\\omega$ P-P [rad/s] & RMSE $i_\\alpha$ [A] \\\\\n\\midrule\n']);
for i = 1:nm
    fprintf(fid, '%s & %.2f & %.4f & %.4f & %.4f \\\\\n', ...
            strrep(methods{i}, '_', '\_'), Ripple(i), Te_pp(i), w_pp(i), rmse_a(i));
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
fclose(fid);

fprintf('\n✓ Exportado a %s/\n', outdir);
fprintf('  Figuras: overview, steadystate, convergence, tracking (.png + .pdf)\n');
fprintf('  Tablas:  metrics.csv, metrics.tex\n\n');
end
