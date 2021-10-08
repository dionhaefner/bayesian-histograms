# Bayesian histograms
Bayesian histograms for estimation of binary event rates.

## What is this for?

When you are trying to get a feeling for how the event rate (probability) of a rare event depends on a parameter, while making no assumptions on how the event rate depends on the parameter.

## Installation

```bash
$ pip install bayeshist
```

## Usage example

Assume you have binary samples of a rare event like this:

![Samples](doc/samples.png?raw=true)

Compute and plot a Bayesian histogram:

```python
>>> from bayeshist import bayesian_histogram, plot_bayesian_histogram
>>> bin_edges, beta_dist = bayesian_histogram(X, y)
# beta_dist is a `scipy.stats.Beta` object, so we can get the
# predicted mean event rate for each histogram bin like this:
>>> bin_mean_pred = best_dist.mean()
>>> plot_bayesian_histogram(bin_edges, beta_dist)
```

The result is something like this:

![Bayesian histogram](doc/bayesian-histogram-comp.png?raw=true)

See also [demo.ipynb](demo.ipynb) for a full walkthrough of this example.
