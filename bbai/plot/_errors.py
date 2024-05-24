from collections import defaultdict, namedtuple
import numpy as np

SummaryStats = namedtuple('SummaryStats', ['min', 'max', 'mean'])

def compute_range(errs):
    low = float("inf")
    high = -low
    for errs_i in errs:
        low = min(np.min(errs_i), low)
        high = max(np.max(errs_i), high)
    return low, high

def bin_errors(errs, bin_size, low):
    res = defaultdict(int)
    for err in errs:
        t = np.rint((err - low) / bin_size)
        res[t] += 1
    return res

def summary_stats(errs):
    min_err = np.min(errs)
    max_err = np.max(errs)
    mean_err = np.mean(errs)
    return SummaryStats(min_err, max_err, mean_err)

def points(bins, a, m):
    res = []
    for err, cnt in bins.items():
        for i in range(cnt):
            res.append((m * i + a, err))
    return np.array(res)

def plot_error_comparison(ax, left_errs, right_errs, nbins):
    left_stats = summary_stats(left_errs)
    right_stats = summary_stats(right_errs)
    left_errs = np.log(left_errs)
    right_errs = np.log(right_errs)
    low, high = compute_range((left_errs, right_errs))
    bin_size = (high - low) / nbins
    left_bins = bin_errors(left_errs, bin_size, low)
    right_bins = bin_errors(right_errs, bin_size, low)
    left_pts = points(left_bins, 0, -1)
    right_pts = points(right_bins, 1, 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks([])

    yticks = [
            left_stats.mean,
            right_stats.mean,
    ]
    for max_err in [left_stats.max, right_stats.max]:
        yticks.append(max_err)
    yticks.append(np.exp(low))
    ylabels = ['%0.2e' % x for x in yticks]
    ax.set_yscale('log')
    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(ylabels)
    ax.set_ylim(np.exp([low, high]))
    ax.yaxis.grid(color='white', linewidth=0.5)
    ax.tick_params(direction='in')

    ax.scatter(left_pts[:, 0], np.exp(left_pts[:, 1] * bin_size + low), s=0.5, c='tab:orange')
    ax.scatter(right_pts[:, 0], np.exp(right_pts[:, 1] * bin_size + low), s=0.5, c='tab:blue')
