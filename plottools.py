"""Plot tools: Reusable utilities for working with matplotlib plots.

Contents:
thin_points         Thin data points that are too close to each other
scatter_thin_points Plot a set of thinned data points with weight info
scatter_mark_outliers
                    Mark outliers at the edges of a scatterplot
scatter_outliers_size
                    Make a scatterplot with sizes and outliers

"""


import matplotlib
from matplotlib import pyplot
import numpy as np
from scipy.spatial import KDTree


def thin_points(data, density=None, len_scale=0.01, nmax=None, r=None):
    """Thin a set of data points down to some target density.

    Uses "seedling" thinning algorithm: Pick a point; if it has more
    than a certain number of neighbours within some predefined radius,
    remove it.  Repeat until no points have more than the deisred number
    of neighbours.

    Parameters:
    data        Matrix, size (N,d) - N is num points, d is problem
                dimensionality
    Specify either:
    density     Target density, maximum density in thinned set
    len_scale   Length scale of density averaging, as a fraction of the
                range in the data (maximum range along any dimension).
                Default 0.01.
    or both of:
    nmax        Maximum number of neighbours for a point
    r           Absolute length scale of distance averaging (radius of
                sphere within which neighbours are counted)
    Note that setting nmax=1 will ensure that no two points are closer
    than r

    NB density only implemented in two dimensions (i.e. d==2).

    """
    dim = data.shape[1]
    if density is None and dim != 2:
        raise NotImplementedError("Density not implemented for dimension != 2")
    elif density is None and (nmax is None or r is None):
        raise ValueError("Must specify both nmax and r if not using density")
    elif density is not None:
        # Choose nmax and r such that r is about 1/100 of the range of
        # the data, but make nmax an integer.
        data_range = np.max(np.max(data, axis=0) - np.min(data, axis=0))
        nmax = np.ceil(density * np.pi * (data_range * len_scale)**2)
        r = np.sqrt(nmax / (density * np.pi))
    # Technically the leafsize should be equal to the ratio of volumes of the
    # d-cube to the d-sphere times nmax, but I think this is good enough.
    neightree = KDTree(data, leafsize=max(20, int(nmax) * 2))
    pairs = neightree.query_ball_tree(neightree, r)
    # Remember, a neighbour list will include itself, so exclude that in
    # the counts
    neighcounts = np.array([len(neighlist) - 1 for neighlist in pairs])
    deleted = np.zeros_like(neighcounts, dtype=bool)
    point_weight = np.ones_like(neighcounts, dtype=np.float32)
    while np.any(neighcounts > nmax):
        del_idx = int(np.random.choice(np.arange(len(neighcounts))[
            (neighcounts > nmax) & ~deleted]))
        deleted[del_idx] = True
        # Try just dividing the deleted weight equally among neighbours
        # (could also try only assigning to nearest neighbour, or something
        # in between)
        point_weight[pairs[del_idx]] += (point_weight[del_idx] * 1.0 /
                                         neighcounts[del_idx])
        neighcounts[pairs[del_idx]] -= 1
        neighcounts[del_idx] = 0
        # TODO Remove the deleted point from others' neighbour lists?
        #      Operation cost versus leaving it there?
    return data[~deleted,:], point_weight[~deleted]


def scatter_thin_points(data_thin, weights, method='size', ax=None,
                        **plot_args):
    """Make a scatterplot with a thinned set of points

    Paramters:
    data_thin   Thinned data, e.g. from thin_points()
    weights     Weights of the thinned data, e.g. from thin_points()
                (represents how many real points each thin point
                represents
    method      How to represent the weights; options are 'size',
                'color', and 'alpha'
    ax          Existing axis to plot into, if any
    Any remaining keyword arguments are passed to pyplot.scatter().

    """
    if ax is None:
        ax = pyplot.gca()
    if method == 'size':
        # TODO An automatic determination of the base size based on the
        #      actual plot size (in pixels) might be nice...
        s_scale = plot_args.pop('s', 2)
        # Retain the border (linewidth) to make it easier to see points' sizes
        pl = ax.scatter(data_thin[:,0], data_thin[:,1],
                        s=(weights * s_scale), **plot_args)
    elif method == 'color':
        plot_args.pop('c', None)
        pl = ax.scatter(data_thin[:,0], data_thin[:,1], linewidth=0, c=weights,
                        **plot_args)
    elif method == 'alpha':
        alpha_min = 0.0
        c_base = plot_args.pop('c', 'k')
        c_base = matplotlib.colors.colorConverter.to_rgba(c_base)
        c_array = np.zeros((len(weights), 4))
        c_array[:,:] = c_base
        c_array[:,3] = 1.0 + (alpha_min - 1.0) * np.exp(-1.0 * weights)
        pl = ax.scatter(data_thin[:,0], data_thin[:,1], linewidth=0, c=c_array,
                        **plot_args)
    else:
        raise ValueError("Unknown representation method " + method)
    return pl


def scatter_mark_outliers(xs, ys, xlim, ylim, ax=None, **plot_args):
    """Mark outliers at the edges of a scatterplot

    Paramters:
    xs      List of x-coordinates of points, passed to pyplot.scatter()
    ys      List of y-coordinates of points, passed to pyplot.scatter()
    xlim    Limits of x-axis, as a tuple (xmin, xmax)
    ylim    Limits of y-axis, as a tuple (ymin, ymax)
    ax      Existing axis to plot into, if any

    Any other keyword arguments are passed to pyplot.scatter().

    The outliers are marked at the edge of the graph as large red dots,
    by default size 40, color 'r'.  If a label was present, the outliers
    will be labelled as the same with " (outliers)" appended.

    Returns the collection of scatterplots - (top, bottom, left, right).

    """

    if ax is None:
        ax = pyplot.gca()
    outl_top = xs[(ys > ylim[1]) & (xs >= xlim[0]) & (xs <= xlim[1])]
    outl_bot = xs[(ys < ylim[0]) & (xs >= xlim[0]) & (xs <= xlim[1])]
    # Let out_left and outl_right handle the (literal) corner cases
    outl_left = np.array(ys[(xs < xlim[0])])
    outl_right = np.array(ys[(xs > xlim[1])])
    outl_left[outl_left < ylim[0]] = ylim[0]
    outl_left[outl_left > ylim[1]] = ylim[1]
    outl_right[outl_right < ylim[0]] = ylim[0]
    outl_right[outl_right > ylim[1]] = ylim[1]
    if 'linewidth' not in plot_args:
        plot_args['linewidth'] = 0
    if 'c' not in plot_args:
        plot_args['c'] = 'r'
    if 's' not in plot_args:
        plot_args['s'] = 40
    plot_args['clip_on'] = False
    plot_args['label'] = (plot_args['label'] + ' (outliers)'
                          if 'label' in plot_args else '_')
    op_t = ax.scatter(outl_top, ylim[1] * np.ones_like(outl_top), **plot_args)
    plot_args['label'] = '_'
    op_b = ax.scatter(outl_bot, ylim[0] * np.ones_like(outl_bot), **plot_args)
    op_l = ax.scatter(
        xlim[0] * np.ones_like(outl_left), outl_left, **plot_args)
    op_r = ax.scatter(
        xlim[1] * np.ones_like(outl_right), outl_right, **plot_args)
    # Not typically necessary, but just ensure the outlier markers end
    # up at the edge of the plot
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return op_t, op_b, op_l, op_r


def scatter_outliers_size(data_thin, weights, xlim, ylim, ax=None, c_out='r',
                          s_out=40, **plot_args):
    """Do scatterplot with outliers, encoding the given weights as size.

    NB This doesn't change the size of the outlier markers - it might be
    possible by modifying the original escatter_outliers() function, but
    does it even make sense to encode the weights of outliers?

    Parameters:
    data_thin   Thinned data, e.g. from thin_points()
    weights     Weights of the thinned data, e.g. from thin_points()
                (represents how many real points each thin point
                represents
    xlim        Limits of x-axis, as a tuple (xmin, xmax)
    ylim        Limits of y-axis, as a tuple (ymin, ymax)
    ax          Existing axis to plot into, if any
    c_out       Colour to use to mark outliers, default 'r'
    s_out       Size for outlier markers, default 40 (px)
    Any remaining keyword arguments will only modify the properties of
    the central (non-outlier) scatterplot.

    Returns the original plot as well as the four outlier scatterplots -
    (orig, top, bottom, left, right)

    """
    pl = scatter_thin_points(data_thin, weights, ax=ax, **plot_args)
    outps = scatter_mark_outliers(
        *data_thin.T, xlim=xlim, ylim=ylim, ax=ax, c=c_out, s=s_out)
    return (pl,) + outps

