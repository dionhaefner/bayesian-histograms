"""bayeshist.py

Bayesian histograms for binary targets.
"""

from typing import Iterable, Literal, Optional, Tuple, Union
from functools import partial

import numpy as np

from scipy.stats import beta as beta_dist, fisher_exact
from scipy.special import betaln

FrozenDistType = type(beta_dist(0, 0))


def _bayes_factor_test(ps1, ns1, ps2, ns2, prior_p, prior_n, threshold=2):
    """Tests whether two binomial datasets come from the same distribution.

    Computes the Bayes Factor of hypotheses:

        H1: Samples are drawn with p_i ~ Beta(alpha_i, beta_i), i={1,2}
        H0: Both samples are drawn with p ~ Beta(alpha_1 + alpha_2, beta_1 + beta_2)

    The Bayes Factor gives the relative increase in data likelihood
    after the split (higher values -> splitting is more favorable).
    """
    # alpha and beta coefficients for distribution of p(y=1)
    alpha_1 = ps1 + prior_p
    beta_1 = ns1 + prior_n

    alpha_2 = ps2 + prior_p
    beta_2 = ns2 + prior_n

    alpha_tot = ps1 + ps2 + prior_p
    beta_tot = ns1 + ns2 + prior_n

    # we could use scipy.state.betabinom here, but betaln is faster
    def betabinom_logp(ps, ns, alpha, beta):
        # this omits choose(n, k), drops out in Bayes factor
        return betaln(ps + alpha, ns + beta) - betaln(alpha, beta)

    bayes_factor = np.exp(
        -betabinom_logp(ps1, ns1, alpha_tot, beta_tot)
        - betabinom_logp(ps2, ns2, alpha_tot, beta_tot)
        + betabinom_logp(ps1, ns1, alpha_1, beta_1)
        + betabinom_logp(ps2, ns2, alpha_2, beta_2)
    )

    return bayes_factor > threshold, bayes_factor


def _fisher_test(ps1, ns1, ps2, ns2, *args, threshold=0.05):
    """Tests whether two binomial datasets come from the same distribution.

    Uses an exact Fisher test. Prior parameters are unused.
    """
    _, pvalue = fisher_exact([[ps1, ps2], [ns1, ns2]])
    return pvalue < threshold, pvalue


def _prune_histogram(bin_edges, pos_samples, neg_samples, test, prior_params, max_bin_size=None, yield_steps=False):
    """Perform histogram pruning.

    This iteratively merges neighboring bins until all neighbor pairs pass
    the given statistical test.
    """
    if max_bin_size is None:
        max_bin_size = float("inf")

    while True:
        new_bins = []
        new_pos_samples = []
        new_neg_samples = []

        num_bins = len(bin_edges) - 1
        splits_reversed = 0

        i = 0

        while True:
            if i == num_bins:
                break

            elif i == num_bins - 1:
                # only 1 bin left, nothing to compare to
                new_bins.append(bin_edges[i])
                new_pos_samples.append(pos_samples[i])
                new_neg_samples.append(neg_samples[i])
                break

            is_significant, test_value = test(
                pos_samples[i],
                neg_samples[i],
                pos_samples[i + 1],
                neg_samples[i + 1],
                *prior_params
            )

            reverse_split = (
                not is_significant
                # ensure that we stay below max_bin_size
                and (bin_edges[i + 1] - bin_edges[i] < max_bin_size)
                # but always merge empty bins
                or (neg_samples[i] == pos_samples[i] == 0)
                or (neg_samples[i + 1] == pos_samples[i + 1] == 0)
            )

            if yield_steps:
                yield dict(
                    i=i - splits_reversed,
                    samples_1=(pos_samples[i], neg_samples[i]),
                    samples_2=(pos_samples[i+1], neg_samples[i+1]),
                    test_value=test_value,
                    is_significant=is_significant,
                    reverse_split=reverse_split,
                    bins=np.concatenate((new_bins, bin_edges[i:])),
                    pos_samples=np.concatenate((new_pos_samples, pos_samples[i:])),
                    neg_samples=np.concatenate((new_neg_samples, neg_samples[i:])),
                )

            if reverse_split:
                splits_reversed += 1
                new_bins.append(bin_edges[i])
                new_pos_samples.append(pos_samples[i] + pos_samples[i + 1])
                new_neg_samples.append(neg_samples[i] + neg_samples[i + 1])
                i += 2
            else:
                # keep everything and proceed with next pair
                new_bins.append(bin_edges[i])
                new_pos_samples.append(pos_samples[i])
                new_neg_samples.append(neg_samples[i])
                i += 1

        new_bins.append(bin_edges[-1])

        assert len(new_bins) == len(bin_edges) - splits_reversed

        bin_edges = new_bins
        pos_samples = new_pos_samples
        neg_samples = new_neg_samples

        if not splits_reversed:
            # no changes made -> we are done
            break

    bin_edges = np.array(bin_edges)
    pos_samples = np.array(pos_samples)
    neg_samples = np.array(neg_samples)

    yield bin_edges, pos_samples, neg_samples


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
    x = np.asarray(x)
    y = np.asarray(y)

    if not np.all(np.isin(np.unique(y), [0, 1])):
        raise ValueError("Binary targets y can only have values 0 and 1")

    if x_range is None:
        x_range = (np.min(x), np.max(x))

    if pruning_method == "bayes":
        if pruning_threshold is None:
            # default bayes factor threshold
            pruning_threshold = 2

        test = partial(_bayes_factor_test, threshold=pruning_threshold)

    elif pruning_method == "fisher":
        if pruning_threshold is None:
            # default p-value threshold
            pruning_threshold = 0.2

        test = partial(_fisher_test, threshold=pruning_threshold)

    elif pruning_method is not None:
        raise ValueError('pruning_method must be "bayes", "fisher", or None.')

    if np.isscalar(bins):
        bin_edges = np.linspace(*x_range, bins + 1)
    else:
        bin_edges = np.asarray(bins)

    neg_samples, _ = np.histogram(x[y == 0], bins=bin_edges)
    pos_samples, _ = np.histogram(x[y == 1], bins=bin_edges)

    if prior_params is None:
        # default prior is weakly informative, using global event rate
        num_pos_samples = np.sum(pos_samples)
        num_neg_samples = np.sum(neg_samples)

        if num_pos_samples > num_neg_samples:
            prior_params = (num_pos_samples / num_neg_samples, 1)
        else:
            prior_params = (1, num_neg_samples / num_pos_samples)

    if pruning_method is not None:
        pruner = _prune_histogram(
            bin_edges, pos_samples, neg_samples, test, prior_params, max_bin_size=max_bin_size
        )
        bin_edges, pos_samples, neg_samples = next(iter(pruner))

    return bin_edges, beta_dist(
        pos_samples + prior_params[0], neg_samples + prior_params[1]
    )
