"""Plot tools: Reusable utilities for working with matplotlib plots.

Contents:
scatter_outliers    Make a scatterplot, marking outliers at the edges
thin_points         Thin data points that are too close to each other

"""


import matplotlib
from matplotlib import pyplot
import numpy as np
from scipy.spatial import KDTree

# Progress!
import tqdm


def escatter_outliers(xs, ys, xlim, ylim, ax=None, c_out='r', s_out=40,
                      **plot_args):
    """Make a scatterplot, marking outliers instead of cutting them off.

    Paramters:
    xs      List of x-coordinates of points, passed to pyplot.scatter()
    ys      List of y-coordinates of points, passed to pyplot.scatter()
    xlim    Limits of x-axis, as a tuple (xmin, xmax)
    ylim    Limits of y-axis, as a tuple (ymin, ymax)
    ax      Existing axis to plot into, if any
    c_out   Colour to use to mark outliers, default 'r'
    s_out   Size for outlier markers, default 40 (px)

    Any other keyword arguments are passed to pyplot.scatter().

    The outliers are marked at the edge of the graph as (typically)
    larger red dots.  If a label was present, the outliers will be
    labelled as the same with " (outliers)" appended.

    """

    if ax is None:
        ax = pyplot.gca()
    plot_args['linewidth'] = 0
    ax.scatter(xs, ys, **plot_args)
    outl_top = xs[(ys > ylim[1]) & (xs >= xlim[0]) & (xs <= xlim[1])]
    outl_bot = xs[(ys < ylim[0]) & (xs >= xlim[0]) & (xs <= xlim[1])]
    # Let out_left and outl_right handle the (literal) corner cases
    outl_left = ys[(xs < xlim[0])]
    outl_right = ys[(xs > xlim[1])]
    outl_left[ys < ylim[0]] = ylim[0]
    outl_left[ys > ylim[1]] = ylim[1]
    outl_right[ys < ylim[0]] = ylim[0]
    outl_right[ys > ylim[1]] = ylim[1]
    plot_args['c'] = c_out
    plot_args['s'] = s_out
    plot_args['alpha'] = 0.8
    plot_args['clip_on'] = False
    plot_args['label'] = (plot_args['label'] + ' (outliers)'
                          if 'label' in plot_args else '')
    escatter(outl_top, ylim[1] * np.ones_like(outl_top), ax=ax, **plot_args)
    plot_args['label'] = '_'
    escatter(outl_bot, ylim[0] * np.ones_like(outl_bot), ax=ax, **plot_args)
    escatter(xlim[0] * np.ones_like(outl_left), outl_left, ax=ax, **plot_args)
    escatter(xlim[1] * np.ones_like(outl_right), outl_right, ax=ax, **plot_args)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # TODO would be useful to return the collection of plots instead
    return None


def thin_points(data, density=None, len_scale=0.01, nmax=None, r=None):
    """Thin a set of data points down to some target density.

    Uses "seedling" thinning algorithm: Go through all points; if any
    have more than a certain number of neighbours within some predefined
    radius, remove them.  Repeat until desired density is achieved.

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
        neighcounts[pairs[del_idx]] -= 1
        neighcounts[del_idx] = 0
        # Try just dividing the deleted weight equally among neighbours
        # (could also try only assigning to nearest neighbour, or something
        # in between)
        point_weight[pairs[del_idx]] += (point_weight[del_idx] * 1.0 /
                                         (len(pairs[del_idx]) - 1))
        point_weight[del_idx] = 0
        print(np.sum(point_weight[~deleted]))
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
        s_scale = plot_args.pop('s', 10)
        pl = ax.scatter(data_thin[:,0], data_thin[:,1], linewidth=0,
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

