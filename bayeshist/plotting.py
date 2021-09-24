def plot_bayesian_histogram(bin_edges, data_dist, color=None, label=None, ax=None, ci=(0.01, 0.99)):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    if ax is None:
        ax = plt.gca()

    if color is None:
        # advance color cycle
        dummy, = ax.plot([], [])
        color = dummy.get_color()

    ci_low, ci_high = data_dist.ppf(ci[0]), data_dist.ppf(ci[1])

    # background boxes
    errorboxes = [
        Rectangle((x1, y1), x2 - x1, y2 - y1)
        for x1, x2, y1, y2
        in zip(bin_edges[:-1], bin_edges[1:], ci_low, ci_high)
    ]

    pc = PatchCollection(errorboxes, facecolor=color, alpha=0.2)
    ax.add_collection(pc)

    # median indicator
    ax.hlines(data_dist.median(), bin_edges[:-1], bin_edges[1:], colors=color, label=label)

    # box edges
    ax.hlines(ci_low, bin_edges[:-1], bin_edges[1:], colors=color, alpha=0.8, linewidth=1)
    ax.hlines(ci_high, bin_edges[:-1], bin_edges[1:], colors=color, alpha=0.8, linewidth=1)
