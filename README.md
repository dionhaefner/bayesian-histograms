# Bayesian histograms
Bayesian histograms for estimation of binary event rates, with fully automated bin pruning.

## What is this for?

When you are trying to get a feeling for how the event rate (probability) of a rare event depends on a parameter, while making no assumptions on how the event rate depends on the parameter.

## Installation

```bash
$ pip install bayeshist
```

## Usage example

Assume you have binary samples of a rare event like this:

<p align="center">
<img src="doc/samples.png?raw=true" width="450px">
</p>

Compute and plot a Bayesian histogram:

```python
>>> from bayeshist import bayesian_histogram, plot_bayesian_histogram

# compute Bayesian histogram from samples
>>> bin_edges, beta_dist = bayesian_histogram(X, y, bins=100, pruning_method="bayes")

# beta_dist is a `scipy.stats.Beta` object, so we can get the
# predicted mean event rate for each histogram bin like this:
>>> bin_mean_pred = best_dist.mean()

# plot it up
>>> plot_bayesian_histogram(bin_edges, beta_dist)
```

The result is something like this:

<p align="center">
<img src="doc/bayesian-histogram-comp.png?raw=true" width="450px">
</p>
  
See also [demo.ipynb](demo.ipynb) for a full walkthrough of this example.


Full docstring:

```python

def bayesian_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: Union[int, Iterable] = 100,
    x_range: Optional[Tuple[float, float]] = None,
    prior_params: Optional[Tuple[float, float]] = None,
    pruning_method: Optional[Literal["bayes", "fisher"]] = "bayes",
    pruning_threshold: Optional[float] = None,
    max_bin_size: Optional[float] = None,
) -> Tuple[np.ndarray, FrozenDistType]:
    """Compute Bayesian histogram for data x, binary target y.

    The output is a Beta distribution over the event rate for each bin.

    Parameters:

        x:
            1-dim array of data.

        y:
            1-dim array of binary labels (0 or 1).

        bins:
            int giving the number of equally spaced intial bins,
            or array giving initial bin edges. (default: 100)

        x_range:
            Range spanned by binning. Not used if `bins` is an array.
            (default: [min(x), max(x)])

        prior_params:
            Parameters to use in Beta prior. First value relates to positive,
            second value to negative samples. [0.5, 0.5] represents Jeffrey's prior, [1, 1] a flat
            prior. The default is a weakly informative prior based on the global event rate.
            (default: `[1, num_neg / num_pos]`)

        pruning_method:
            Method to use to decide whether neighboring bins should be merged or not.
            Valid values are "bayes" (Bayes factor), "fisher" (exact Fisher test), or None
            (no pruning). (default: "bayes")

        pruning_threshold:
            Threshold to use in significance test specified by `pruning_method`.
            (default: 2 for "bayes", 0.2 for "fisher")

        max_bin_size:
            Maximum size (in units of x) above which bins will not be merged
            (except empty bins). (default: unlimited size)

    Returns:

        bin_edges: Coordinates of bin edges
        beta_dist: n-dimensional Beta distribution (n = number of bins)

    Example:

        >>> x = np.random.randn(1000)
        >>> p = 10 ** (-2 + x)
        >>> y = np.random.rand() < p
        >>> bins, beta_dist = bayesian_histogram(x, y)
        >>> plt.plot(0.5 * (bins[1:] + bins[:-1]), beta_dist.mean())

    """
    

```
