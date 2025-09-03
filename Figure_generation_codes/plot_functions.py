#### ----------------------------------------------------------------
#### author: dhrubas2, date: Mar 26, 2025
#### plot functions
#### ----------------------------------------------------------------

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D 
from math import ceil, floor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from compare_auc_delong_xu import delong_roc_test, delong_roc_variance


def RadarChart(num_vars, frame = "circle"):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {"circle", "polygon"}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint = False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        
        if frame == "polygon":
            PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed = True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed = closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, **kwargs)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle(xy = (0.5, 0.5), radius = 0.5,
                              edgecolor = "#000000")
            elif frame == "polygon":
                return RegularPolygon(xy = (0.5, 0.5), numVertices = num_vars, 
                                      radius = 0.5, edgecolor = "#000000")
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be "left"/"right"/"top"/"bottom"/"circle".
                spine = Spine(axes = self, spine_type = "circle", 
                              path = Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5)
                                    + self.transAxes)
                return {"polar": spine}
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

    register_projection(RadarAxes)
    
    return theta


def make_radar_lines(theta, data, ax, rstep = 0.1, labels = None, title = None, 
                     color = "magenta", alpha = 0.25, ls = "-", lw = 1.5, 
                     mrkr = "o", ms = 6, fontdict = None):
    
    rgrids = np.arange(0, 1 + rstep, rstep)
    lprop  = {"linestyle": ls, "linewidth": lw, "color": color}
    mprop  = {"marker": mrkr, "markersize": ms, "markeredgewidth": lw, 
              "markerfacecolor": color}
    fprop  = {"facecolor": color, "alpha": alpha}
    if fontdict is None:
        fontdict  = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}}

    ax.set_rgrids(rgrids)
    ax.plot(theta, data, **lprop, **mprop)
    ax.fill(theta, data, label = "_nolegend_", **fprop, **lprop)
    
    if labels is not None:
        ax.set_varlabels(labels, ma = "center", **fontdict["label"]); 
    
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    if title is not None:
        ax.set_title(f"{title}\n", position = (0.5, 1.2), ha = "center", 
                     va = "center", ma = "center", **fontdict["title"]);
    
    return ax


def pad_radar_ticks(ticks, pads):
    ## add padding to radar tick labels for better visualization.
    
    n_theta   = len(ticks)
    ticks_pad = [[ ]] * n_theta
    for k, tk in enumerate(ticks):
        if (k == 0) | (k == (n_theta - 1)):
            ticks_pad[k] = tk
        elif k >= 1 and k <= floor(n_theta / 2):
            ticks_pad[k] = "\n".join([
                x + " " * pads[0] for x in tk.split("\n")])
        elif k > floor(n_theta / 2) and k < (n_theta - 1):
            ticks_pad[k] = "\n".join([
                " " * pads[1] + x for x in tk.split("\n")])
    
    return ticks_pad


def make_catplot(data, ax, annots = None, annot_fmt = "%s", height = 0.45, ybold = True, 
                 yticks = "left", title = None, colors = None, fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {"label": {"size": 12, "weight": "regular"}, 
                    "sbttl": {"size": 14, "weight": "bold"}, 
                    "title": {"size": 16, "weight": "bold"}, 
                    "super": {"size": 20, "weight": "bold"}, 
                    "panel": {"size": 36, "weight": "bold"}}
    
    if colors is None:
        colors = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                  "#FFC72C", "#A9A9A9", "#DFDFDF", "#000000"]
        colors = colors[:data.shape[1]] 
    colors += ["#000000"]
    
    barprop  = {"linestyle": "-", "linewidth": 2, "edgecolor": colors[-1]}
    lgndprop = {"ncols": data.shape[1], "prop": fontdict["label"], "title": None}
    
    ## main plot.
    if yticks.lower() == "right":
        data, colors = data.iloc[:, ::-1], colors[-2::-1] + colors[-1:]
    for (lbl, dat), clr in zip(data.items(), colors):
        dat  = pd.DataFrame({"ytick": dat.index, "width": dat, 
                             "start": data.cumsum(axis = 1)[lbl] - dat})
        ann  = annots[lbl] if annots is not None else annots
        
        rect = ax.barh(data = dat, y = "ytick", width = "width", left = "start", 
                       height = height, color = clr, alpha = 0.8, **barprop, 
                       label = lbl)
        ax.bar_label(rect, labels = ann, label_type = "center", fmt = annot_fmt, 
                     color = colors[-1], **fontdict["label"])
    
    ax.invert_yaxis()
    if yticks.lower() == "right":    ax.invert_xaxis()
    sns.despine(ax = ax, trim = False, left = (yticks.lower() != "left"), 
                right = (yticks.lower() != "right"));
    
    ## format axis ticks & legends.
    fontdict["sbttl"].update({"weight": "bold" if ybold else "regular"})
    ax.set_yticks(ticks = range(data.shape[0]), labels = data.index, va = "center", 
                  ha = "right" if yticks.lower() == "left" else "left", ma = "center", 
                  **fontdict["sbttl"]);
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.set_title(title, wrap = True, y = 1.01, **fontdict["title"]);
    
    ax.legend(loc = (0.25, 1.03), **lgndprop);
    if yticks.lower() == "right":
        ax.legend(*[x[::-1] for x in ax.get_legend_handles_labels()], 
                  loc = (0.75, 1.03), **lgndprop);
    
    return ax


def make_roc_plot(data, label, pred, group, ax, title = None, legend = "short", 
                  fill = False, alpha = 0.4, colors = None, marker = "o", ms = 6, 
                  mew = 1.5, ls = "-", lw = 2, bs = "--", bw = 1.5, legend_title = None, 
                  legend_loc = None, fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    if colors is None:
        colors = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                  "#FFC72C", "#708090", "#A9A9A9", "#000000"]
        colors = colors[3:6] + [colors[-3]]
    
    # lineprop = {"linestyle": ls, "linewidth": lw}
    # mrkrprop = {"marker": marker, "markersize": ms, "markeredgewidth": mew}
    baseprop = {"linestyle": bs, "linewidth": bw, "color": colors[-1]}
    lgndprop = {"loc": (1.06, 0.25) if legend_loc is None else legend_loc, 
                "title": group.capitalize() if (legend_title is None) else legend_title, 
                "prop": fontdict["label"], "title_fontproperties": fontdict["title"]}
    
    
    ## main plot.
    for k, (grp, data_grp) in enumerate(data.groupby(by = group, sort = False)):
        ## groupwise parameters.
        color    = colors[k]
        lineprop = {"linestyle": ls[k] if isinstance(ls, list) else ls, 
                    "linewidth": lw}
        mrkrprop = {"marker": marker[k] if isinstance(marker, list) else marker, 
                    "markersize": ms[k] if isinstance(ms, list) else ms, 
                    "markeredgewidth": mew}

        ## plot roc curve.
        roc_grp = RocCurveDisplay.from_predictions(
            y_true = data_grp[label], y_pred = data_grp[pred], 
            drop_intermediate = True, pos_label = 1, 
            plot_chance_level = (grp == data[group].iloc[-1]), 
            name = grp, color = color, **lineprop, **mrkrprop, 
            chance_level_kw = baseprop, ax = ax)
        
        if fill:
            ax.fill_between(x = roc_grp.fpr, y1 = roc_grp.fpr, 
                            y2 = roc_grp.tpr, color = clr, alpha = alpha)
        
        ax.set_aspect("auto")                                                  # stop forcing square-sized plots
    
    
    ## format axis ticks & legends.
    ax.axis([-0.05, 1.05, -0.05, 1.05]);
    ax.set_xticks(np.arange(0, 1.2, 0.2));
    ax.set_yticks(np.arange(0, 1.2, 0.2));
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.set_xlabel("1 $-$ Specificity", labelpad = 8, **fontdict["label"]);
    ax.set_ylabel("Sensitivity", labelpad = 8, **fontdict["label"]);
    
    lgnd = ax.legend(**lgndprop)
    for lgndtxt in lgnd.get_texts():
        lgndtxt.set_text( lgndtxt.get_text().replace(") (", ", ") )
        lgndtxt.set_text( lgndtxt.get_text().replace("Chance level", "Random") )
    if legend.lower() == "short":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:-1], labels[:-1], **lgndprop)
        
    ax.set_title(title, wrap = True, y = 1.02, **fontdict["title"]);
    
    return ax


def make_lollipop_plot(data, x, y, ax, size = 150, width = 4, yticks = "left", 
                       colors = None, title = None, xlabel = None, ylabel = None, 
                       offset = 0.035, fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    if colors is None:
        colors   = ["#B075D0", "#708090", "#000000"]
    
    lineprop = {"linestyle": "-", "linewidth": width, "color": colors[-2], 
                "alpha": 0.9}
    baseprop = {"linestyle": "--", "linewidth": 1.5, "color": colors[-1], 
                "alpha": 0.8}
    mrkrprop = {"marker": "o", "linewidths": 1.5, "color": colors[0], 
                "edgecolor": colors[-1], "alpha": 0.8}
    
    ## add offset to horizontal lines [to skip overlap with the dots].
    data[f"{x}_offset"] = data[x].map(
        lambda val: val - (offset if (val > 0) else -offset))
    
    
    ## main plot.
    ax.axvline(x = 0, ymin = 0, ymax = 1, **baseprop)
    ax.hlines(data = data, y = y, xmin = 0, xmax = f"{x}_offset", **lineprop)
    ax.scatter(data = data, x = x, y = y, s = size, **mrkrprop)
    ax.invert_yaxis();                                                         # top-to-bottom by importance
    sns.despine(ax = ax, offset = 0, trim = False, top = False, right = False);   
    
    ## format axis ticks & legends.
    if yticks.lower() == "right":                                              # put yticks to the right
        ax.invert_xaxis()
        ax.tick_params(axis = "y", left = False, right = True, 
                       labelleft = False, labelright = True);
    
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.set_xlabel(xlabel, labelpad = 8, y = 0.05, **fontdict["label"]);
    ax.set_ylabel(ylabel, labelpad = 8, x = 0.05, **fontdict["label"]);
    ax.set_title(title, wrap = True, pad = 8, y = 1.02, **fontdict["title"]);
    
    return ax


def make_dotplot(data, x, y, ax, yticks = "left", center = "mean", ebar = ("ci", 95), 
                 ecap = 0.5, marker = "o", ms = 8, mew = 0, ls = "-", lw = 2, 
                 bs = "--", bw = 1.5, title = None, xlabel = None, ylabel = None, 
                 colors = None, fontdict = None):    
    ## plot parameters.
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    if colors is None:
        colors = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                  "#FFC72C", "#708090", "#A9A9A9", "#000000"]
        colors = [colors[k] for k in [3, 6, -1]]

    lineprop = {"linestyle": ls, "linewidth": lw, "color": colors[1]}
    baseprop = {"linestyle": bs, "linewidth": bw, "color": colors[-1]}
    mrkrprop = {"marker": marker, "markersize": ms, "markeredgewidth": mew, 
                "markerfacecolor": colors[0], "markeredgecolor": colors[1]}
    
    ## main plot.
    ## plot order: errorbar, center.
    ax.axvline(x = 0, ymin = 0, ymax = 1, **baseprop);
    sns.pointplot(data = data, x = x, y = y, errorbar = ebar, capsize = ecap, 
                  err_kws = lineprop, linestyle = "", marker = None, ax = ax)
    sns.pointplot(data = data, x = x, y = y, estimator = center, errorbar = None, 
                  linestyle = "", **mrkrprop, ax = ax)
    sns.despine(ax = ax, left = False, bottom = False, right = False, top = False);
    
    ## format axis ticks & legends.
    if yticks.lower() == "right":
        # ax.invert_xaxis()
        ax.tick_params(axis = "y", left = False, right = True, labelleft = False, 
                       labelright = True);
    
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.set_xlabel(xlabel, labelpad = 8, y = 0.05, **fontdict["label"]);
    ax.set_ylabel(ylabel, labelpad = 8, x = 0.05, **fontdict["label"]);
    ax.set_title(title, wrap = True, pad = 8, y = 1.02, **fontdict["title"]);

    return ax


def make_scatterplot(data, x, y, ax, size = 40, color = None, lw = 2, 
                     fit = True, labels = None, title = None, fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    colors   = ["#B075D0" if color is None else color, "#000000"]
        
    lineprop = {"linestyle": "-", "linewidth": lw, "color": colors[0], 
                "alpha": 0.9}
    ci_alpha = 0.25
    mrkrprop = {"marker": "o", "s": size, "linewidths": ceil(lw / 2), 
                "color": colors[0], "edgecolor": colors[-1], "alpha": 0.8}
    
    labels = [x, y] if labels is None else labels
    
    
    ## main plot.
    ax = sns.regplot(data = data, x = x, y = y, scatter = True, fit_reg = fit, 
                     ci = 95, n_boot = 1000, seed = 86420, scatter_kws = mrkrprop, 
                     line_kws = lineprop, ax = ax)
    plt.setp(ax.collections[1], alpha = ci_alpha);
    sns.despine(ax = ax, offset = 0, trim = False);
    
    ## format axis ticks & legends.
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.set_xlabel(labels[0], labelpad = 8, y = -0.02, **fontdict["label"]);
    ax.set_ylabel(labels[1], labelpad = 8, x = -0.02, **fontdict["label"]);
    ax.set_title(title, y = 0.98, pad = 8, wrap = True, **fontdict["title"]);
    
    return ax


def make_heatmap(data, ax, square = True, fmt = "d", cmap = "tab20b_r", saturation = 0.8, 
                 center = None, cbar = True, cbar_shrink = 0.99, xrot = 0, yrot = 0, lw = 0, 
                 bold = True, labels = None, title = None, fontdict = None):

    ## plot parameters.
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    if labels is None:
        labels = [None, None]
    
    annotprop = {"family": "sans", "size": 13, 
                 "weight": "bold" if bold else "regular", 
                 "ha": "center", "va": "center", "ma": "center", 
                 "rotation": None, "wrap": True}
    cbarprop  = {"location": "right", "fraction": 0.25, "shrink": cbar_shrink}
    lineprop  = {"linecolor": "#000000", "linewidths": lw, "clip_on": False}
    baseprop  = {"edgecolor": "#000000", "linewidth": 1}
    
    
    ## main plot.
    ax = sns.heatmap(data = data, annot = True, fmt = fmt, square = square, 
                     center = center, robust = True, cbar = cbar, cmap = cmap, 
                     alpha = saturation, **lineprop, annot_kws = annotprop, 
                     cbar_kws = cbarprop, ax = ax)
    
    ## keep spines.
    # [ax.figure.axes[0].spines[pos].set(visible = True, **baseprop) 
    #  for pos in ["left", "right", "top", "bottom"]];
    sns.despine(ax = ax, left = False, top = False, right = False, bottom = False);
    
    ## format colorbar.
    if cbar:
        cax = ax.figure.axes[-1]
        # if cbar:
        #     cax.spines["outline"].set(visible = True, **baseprop);
        sns.despine(ax = cax, left = False, top = False, right = False, bottom = False);
        cax.tick_params(labelsize = fontdict["label"]["size"]);
    
    ## format axis ticks & legends.
    ax.set_xticklabels(ax.get_xticklabels(), rotation = xrot, va = "top", 
                       ha = "right" if xrot > 0 else "center", ma = "center");
    ax.set_yticklabels(ax.get_yticklabels(), rotation = yrot, ha = "right", 
                       va = "baseline" if yrot > 0 else "center", ma = "center");
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.set_xlabel(labels[0], labelpad = 8, y = -0.02, wrap = True, **fontdict["label"]);
    ax.set_ylabel(labels[1], labelpad = 8, x = -0.02, wrap = True, **fontdict["label"]);
    ax.set_title(title, y = 1.02, pad = 8, wrap = True, **fontdict["title"]);

    return ax


## make nested barplots for multiple variables.
def make_barplot3_base(data, x, y, hue, ax, fontdict, title, xlabels, 
                       xrot = 0, bar_labels = True, lrot = 0, ylabel = None, 
                       width = 0.8, gap = 0, bline = False, yrefs = None, 
                       legend = True, legend_title = None, colors = None,                        
                       skipcolor = False):
    ## parameters.
    # col_palette = "tab10"
    if data[hue].nunique() > 3:
        if colors is None:
            # col_palette = "CMRmap"
            # col_palette = "crest"
            col_palette = "tab20b_r"
        else:
            col_palette = "tab20b_r" if len(colors) <= 3 else colors
    else:
        if colors is None:
            col_list = ["#75D0B0", "#FFC72C", "#88BECD"]                           # ["green", "yellow", "blue"]-ish
        else:
            col_list = colors
        
        if skipcolor:
            col_list = col_list[1:] + [col_list[0]]
        
        col_palette = dict(zip(data[hue].unique(), col_list))

    bar_prop   = {"edgecolor": "#000000", "linestyle": "-", "linewidth": 2}
    col_bline  = "#000000"
    bline_prop = {"linestyle": "--", "linewidth": 1}
    
    n_bars = data[hue].nunique()
    if bar_labels:    
        bar_lbl = [data[data[hue].eq(bb)][y].round(2) \
                   for bb in data[hue].unique()]                               # bar heights as labels    
    
    if data[y].max() <= 1:
        yticks = np.arange(0, 1.4, 0.2)
    else:
        yticks = np.linspace(
            0, np.round(data[y].max() * 1.2), num = 7).round(1)
        
    ylabels = yticks.round(2).tolist()[:-1] + [""]
    
    # lgnd_loc = [0.8, 0.84]
    # lgnd_loc = "best"
    lgnd_loc  = "lower left"
    lgnd_bbox = (1.0, 0.5, 0.6, 0.6)
    lgnd_font = {pn.replace("font", ""): pv \
                 for pn, pv in fontdict["label"].items()}
    
    ## build plot.
    sns.barplot(data = data, x = x, y = y, hue = hue, orient = "v", 
                width = width, gap = gap, palette = col_palette, 
                saturation = 0.8, dodge = True, ax = ax, **bar_prop);          # nested barplots
    if bar_labels:
        [ax.bar_label(ax.containers[nn], labels = bar_lbl[nn], padding = 0.2, 
                      rotation = lrot, **fontdict["label"]) \
         for nn in range(n_bars)];
    
    ## add baseline.
    if bline & (yrefs is not None):
        [ax.axhline(y = yrefs[nn], xmin = 0, xmax = data.shape[0] / n_bars, 
                    color = col_bline, **bline_prop) \
         for nn in range(n_bars)];
        
    ## format plot ticks & labels.
    ax.set_title(title, y = 1.02, **fontdict["title"]);
    ax.set_xlabel(None);     ax.set_ylabel(ylabel, **fontdict["label"]);
    # ax.set_ylim([yticks.min() - 0.1, yticks.max()]);
    ax.set_yticks(ticks = yticks, labels = ylabels, **fontdict["label"]);
    ax.set_xticks(range(len(xlabels)));
    if xrot != 0:
        ax.set_xticklabels(xlabels, rotation = xrot, rotation_mode = "anchor", 
                           ha = "right", va = "center", ma = "center", 
                           position = (0, -0.02), **fontdict["label"]);
    else:
        ax.set_xticklabels(xlabels, ma = "center", position = (0, -0.02), 
                           **fontdict["label"]);
    
    if legend:
        lgnd_ttl = {pn.replace("font", ""): pv \
                    for pn, pv in fontdict["title"].items()}
        # lgnd_ttl.update({"multialignment": "center"})
        lgnd = ax.legend(loc = lgnd_loc, bbox_to_anchor = lgnd_bbox, 
                  prop = lgnd_font, title = legend_title, 
                  title_fontproperties = lgnd_ttl, frameon = True);
        lgnd.get_title().set_multialignment("center");
    else:
        ax.legend([ ], [ ], frameon = False)
    
    return ax


## make nested barplots with breaks for multiple variables.
def make_barplot3(data, x, y, hue, ax, width = 0.8, gap = 0, bline = False, 
                  yrefs = None, xrot = 0, bar_labels = False, label_rot = 0, 
                  xlabels = None, ylabel = None, title = None, legend = True, 
                  legend_title = None, colors = None, skipcolor = False, 
                  fontdict = None):

    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}

    if xlabels is None:
        xlabels = data[x].unique().tolist()
    
    if np.isinf(data[y].max()):                                                # make nested barplots with breaks
        max_val  = 200
        data_brk = data.replace({y: {np.inf: max_val}})                        # replace infinity with a dummy value
        
        ax_t, ax_b = ax
        ax_t_b, ax_b_t = 60, 45
        
        ax_t = make_barplot3_base(
            data = data_brk, x = x, y = y, hue = hue, width = width, 
            gap = gap, title = title, bar_labels = bar_labels, lrot = label_rot, 
            xlabels = xlabels, xrot = xrot, ylabel = ylabel, bline = bline, 
            yrefs = yrefs, legend = False, legend_title = legend_title, 
            colors = colors, skipcolor = skipcolor, ax = ax_t, 
            fontdict = fontdict)
        ax_t.set_xlim([-0.5, len(xlabels) - 0.5]);
        ax_t.set_ylim([ax_t_b, np.round(max_val * 1.05)]);
        ax_t.spines["bottom"].set_visible(False);
        ax_t.set_yticks(ticks = np.linspace(80, max_val, num = 3), 
                        labels = [100, 1000, "$\infty$"], **fontdict["label"]);
        ax_t.tick_params(axis = "x", which = "both", bottom = False, 
                         labelbottom = False);
        
        # data_brk.loc[data_brk[y].gt(ax_b_t), y] = ax_b_t                       # hack for new py version (?)
        ax_b = make_barplot3_base(
            data = data_brk, x = x, y = y, hue = hue, width = width, 
            gap = gap, title = None, bar_labels = bar_labels, lrot = label_rot, 
            xlabels = xlabels, xrot = xrot, ylabel = ylabel, bline = bline, 
            yrefs = yrefs, legend = legend, legend_title = legend_title, 
            colors = colors, skipcolor = skipcolor, ax = ax_b, 
            fontdict = fontdict)
        ax_b.set_xlim([-0.5, len(xlabels) - 0.5]);
        ax_b.set_ylim([0, ax_b_t]);     ax_b.spines["top"].set_visible(False);
        ax_b.set_yticks(ticks = [0, 20, 40], labels = [0, 20, 40], 
                        **fontdict["label"]);
        
        ## add breakpoint indicators.
        d_brk = 0.005                                                          # axis breakpoint tick size
        brkargs = dict(transform = ax_t.transAxes, color = "#000000", 
                       clip_on = False)
        ax_t.plot((-d_brk, +d_brk), (-d_brk, +d_brk), **brkargs)               # add breakpoint marker tick for top plot
        # ax_t.plot((1 - d_brk, 1 + d_brk), (-d_brk, +d_brk), **brkargs)
        
        brkargs.update(transform = ax_b.transAxes, color = "#000000", 
                       clip_on = False)
        ax_b.plot((-d_brk, +d_brk), (1 - d_brk, 1 + d_brk), **brkargs)         # add breakpoint marker tick for bottom plot
        # ax_b.plot((1 - d_brk, 1 + d_brk), (1 - d_brk, 1 + d_brk), **brkargs)
        
        if legend:
            ax_b.get_legend().set(bbox_to_anchor = (1.0, 0.0, 0.6, 0.6))       # set legend location
        
        ax = ax_t, ax_b
        
    else:
        ax = make_barplot3_base(
            data = data, x = x, y = y, hue = hue, width = width, gap = gap, 
            title = title, bar_labels = bar_labels, lrot = label_rot, 
            xlabels = xlabels, xrot = xrot, ylabel = ylabel, bline = bline, 
            yrefs = yrefs, legend = legend, legend_title = legend_title, 
            colors = colors, skipcolor = skipcolor, ax = ax, fontdict = fontdict)
        
    return ax


def make_donutplots(data, x, outer, inner, ax, labels = False, title = None, 
                    outer_order = None, inner_order = None, donut_size = 0.35, 
                    colors = None, fontdict = None):
    ## plot parameters.
    if colors is None:
        colors = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                  "#FFC72C", "#A9A9A9", "#000000"]
        colors = [colors[0], colors[1], colors[2], colors[5]]
    
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    
    wdgprop = {"edgecolor": "#000000", "linestyle": "-", "linewidth": 2, 
               "antialiased": False, "width": donut_size}
    txtprop = {"size": 14, "weight": "demibold", "ha": "center"}
    
    
    ## prepare data.
    outer_data = data.groupby(by = outer).sum(numeric_only = True)    
    inner_data = data.groupby(by = [outer, inner]).sum(numeric_only = True)
    if outer_order is not None:
        outer_data = outer_data.loc[outer_order]
        if inner_order is not None:
            inner_data = inner_data.loc[list(product(outer_order, inner_order))]
        else:
            inner_data = inner_data.loc[outer_order]
    
    outer_labels, inner_labels = None, None
    if isinstance(labels, bool):    labels = [labels] * 2
    if any(labels):
        if labels[0]:    outer_labels = outer_data.index.tolist()
        if labels[1]:    inner_labels = inner_data.index.get_level_values(1)
    
    
    ## make main plot.
    ax.pie(data = outer_data, x = x, radius = 1 + donut_size, 
           labels = outer_labels, labeldistance = 1.1, autopct = "%0.1f%%", 
           pctdistance = 0.80, colors = colors[:2], counterclock = False, 
           shadow = False, wedgeprops = wdgprop, textprops = txtprop)
    ax.pie(data = inner_data, x = x, radius = 0.95, labels = inner_labels, 
           labeldistance = 0.40, autopct = "%0.1f%%", pctdistance = 0.70, 
           colors = colors[2:], counterclock = False, shadow = False, 
           wedgeprops = wdgprop, textprops = txtprop)

    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    
    ax.set_title(title, y = 1.16, wrap = True, **fontdict["title"]);
    
    return ax

