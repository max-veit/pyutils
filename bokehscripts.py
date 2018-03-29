#vim: set encoding=utf-8
"""Scripts for making certain plots with Bokeh (bokeh.pydata.org)

Contents:
    plot_acorr      Plot autocorrelation of a timeseries interactively
"""


from __future__ import unicode_literals, print_function, division

import bokeh
import bokeh.plotting as bkp
import bokeh.models as bkm
import numpy as np
from scipy import signal


#TODO make a variant of plottools.scatter_thin_points that works in Bokeh


def plot_acorr(time, data, step_start, sample_dt, corr_guess,
               xlabel="Time t", ylabel="Unnormalized autocorrelation C(t)",
               title=None):
    """Plot the autocorrelation of a timeseries interactively

    Parameters:
        time        Time values of the timeseries
                    Must be sequential and equally spaced
        data        Data of the timeseries
        step_start  Step (array index) at which to start computing
                    averages and correlation
        sample_dt   Time between two successive samples
        corr_guess  Guess for the correlation time of the series

    Optional parameters:
        xlabel      Label for the x axis (default 'Time t')
                    Suggested format: 'Time t / <unit>'
        ylabel      Label for the y axis
                    (default 'Unnormalized autocorrelation C(t)')
                    Suggested format: 'Autocorrelation C(t) / <unit>^2'
        title       Title for the plot
    """
    print("Starting at time {:.2f}".format(time[step_start]))
    data_mean = np.mean(data[step_start:])
    data_zeromean = data[step_start:] - np.mean(data[step_start:])
    n = len(data_zeromean)
    dataacorr = (signal.correlate(data_zeromean, data_zeromean, 'same')
                 / (n - np.abs(np.arange(n) - n//2)))
    window_width = int(corr_guess * 6.0 / sample_dt)
    acorr_int = 0.5 * np.sum(
        dataacorr[n//2 - window_width : n//2 + window_width + 1]) * sample_dt
    corr_time = acorr_int / dataacorr[n//2]
    p = bkp.figure(title=title, x_range=(-20.0 * corr_guess, 20.0*corr_guess),
                   plot_width=800, plot_height=600)
    p.line(time[step_start:] - time[n//2 + step_start],
           dataacorr, line_width=1.0)
    int_region = bkm.BoxAnnotation(
        right=time[window_width], left=-1.0 * time[window_width],
        fill_alpha=0.2, fill_color='orange', name="Integration region")
    p.add_layout(int_region)
    corr_time_region = bkm.BoxAnnotation(
        right=corr_time, left=-1.0*corr_time, fill_alpha=0.3, fill_color='red',
        name="Correlation time")
    p.add_layout(corr_time_region)
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    hovertool = bkm.HoverTool(tooltips=[('Time', '$x'), ('Correlation', '$y')])
    p.add_tools(hovertool)
    bkp.show(p)
    serr_mean = np.sqrt(2.0 * acorr_int / n / sample_dt)
    #TODO try annotating the graph instead of dumping print statements
    print("Correlation time (guess): {:.3g}".format(corr_guess))
    print("Correlation time (integrated): {:3g}".format(corr_time))
    print("Error on mean without correlation: {:3g}".format(np.std(data[step_start:]) / np.sqrt(n)))
    print("Estimated error on mean: {:3g}".format(serr_mean))
    print("Number of effective independent samples: {:2f}".format(n / corr_time * sample_dt))
    print("Estimate with error: {:.4f} ± {:.4f}".format(data_mean, serr_mean))
    return p, data_mean, serr_mean

