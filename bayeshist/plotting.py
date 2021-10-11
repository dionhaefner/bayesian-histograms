from typing import Union, Iterable, Optional, Any, Tuple
import numpy as np

from .bayeshist import FrozenDistType


def plot_bayesian_histogram(
    bin_edges: np.ndarray,
    data_dist: FrozenDistType,
    color: Union[str, Iterable[float], None] = None,
    label: Optional[str] = None,
    ax: Any = None,
    ci: Optional[Tuple[float, float]] = (0.01, 0.99)
) -> None:
    """Plot a Bayesian histogram as horizontal lines with credible intervals.

    Parameters:

        bin_edges:
            Coordinates of bin edges

        data_dist:
            n-dimensional Beta distribution (n = number of bins)

        color:
            Color to use (default: use next in current color cycle)

        label:
            Legend label (default: no label)

        ax:
            Matplotlib axis to use (default: current axis)

        ci:
            Credible interval used for shading, use `None` to disable shading.

    Example:

        >>> x = np.random.randn(1000)
        >>> p = 10 ** (-2 + x)
        >>> y = np.random.rand() < p
        >>> bins, beta_dist = bayesian_histogram(x, y)
        >>> plot_bayesian_histogram(bins, beta_dist)

    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    if ax is None:
        ax = plt.gca()

    if color is None:
        # advance color cycle
        dummy, = ax.plot([], [])
        color = dummy.get_color()

    if ci is not None:
        ci_low, ci_high = data_dist.ppf(ci[0]), data_dist.ppf(ci[1])

        # background boxes
        errorboxes = [
            Rectangle((x1, y1), x2 - x1, y2 - y1)
            for x1, x2, y1, y2
            in zip(bin_edges[:-1], bin_edges[1:], ci_low, ci_high)
        ]

        pc = PatchCollection(errorboxes, facecolor=color, alpha=0.2)
        ax.add_collection(pc)

        # box edges
        ax.hlines(ci_low, bin_edges[:-1], bin_edges[1:], colors=color, alpha=0.8, linewidth=1)
        ax.hlines(ci_high, bin_edges[:-1], bin_edges[1:], colors=color, alpha=0.8, linewidth=1)

    # median indicator
    ax.hlines(data_dist.median(), bin_edges[:-1], bin_edges[1:], colors=color, label=label)
