Help on function scatterplot in module seaborn.relational:

ssccaatttteerrpplloott(data=None, *, x=None, y=None, hue=None, size=None, style=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, markers=True, style_order=None, legend='auto', ax=None, **kwargs)
    Draw a scatter plot with possibility of several semantic groupings.
    
    The relationship between `x` and `y` can be shown for different subsets
    of the data using the `hue`, `size`, and `style` parameters. These
    parameters control what visual semantics are used to identify the different
    subsets. It is possible to show up to three dimensions independently by
    using all three semantic types, but this style of plot can be hard to
    interpret and is often ineffective. Using redundant semantics (i.e. both
    `hue` and `style` for the same variable) can be helpful for making
    graphics more accessible.
    
    See the :ref:`tutorial <relational_tutorial>` for more information.
    
    The default treatment of the `hue` (and to a lesser extent, `size`)
    semantic, if present, depends on whether the variable is inferred to
    represent "numeric" or "categorical" data. In particular, numeric variables
    are represented with a sequential colormap by default, and the legend
    entries show regular "ticks" with values that may or may not exist in the
    data. This behavior can be controlled through various parameters, as
    described and illustrated below.
    
    Parameters
    ----------
    data : :class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence
        Input data structure. Either a long-form collection of vectors that can be
        assigned to named variables or a wide-form dataset that will be internally
        reshaped.
    x, y : vectors or keys in ``data``
        Variables that specify positions on the x and y axes.
    hue : vector or key in `data`
        Grouping variable that will produce points with different colors.
        Can be either categorical or numeric, although color mapping will
        behave differently in latter case.
    size : vector or key in `data`
        Grouping variable that will produce points with different sizes.
        Can be either categorical or numeric, although size mapping will
        behave differently in latter case.
    style : vector or key in `data`
        Grouping variable that will produce points with different markers.
        Can have a numeric dtype but will always be treated as categorical.
    palette : string, list, dict, or :class:`matplotlib.colors.Colormap`
        Method for choosing the colors to use when mapping the ``hue`` semantic.
        String values are passed to :func:`color_palette`. List or dict values
        imply categorical mapping, while a colormap object implies numeric mapping.
    hue_order : vector of strings
        Specify the order of processing and plotting for categorical levels of the
        ``hue`` semantic.
    hue_norm : tuple or :class:`matplotlib.colors.Normalize`
        Either a pair of values that set the normalization range in data units
        or an object that will map from data units into a [0, 1] interval. Usage
        implies numeric mapping.
    sizes : list, dict, or tuple
        An object that determines how sizes are chosen when `size` is used.
        List or dict arguments should provide a size for each unique data value,
        which forces a categorical interpretation. The argument may also be a
        min, max tuple.
    size_order : list
        Specified order for appearance of the `size` variable levels,
        otherwise they are determined from the data. Not relevant when the
        `size` variable is numeric.
    size_norm : tuple or Normalize object
        Normalization in data units for scaling plot objects when the
        `size` variable is numeric.
    markers : boolean, list, or dictionary
        Object determining how to draw the markers for different levels of the
        `style` variable. Setting to `True` will use default markers, or
        you can pass a list of markers or a dictionary mapping levels of the
        `style` variable to markers. Setting to `False` will draw
        marker-less lines.  Markers are specified as in matplotlib.
    style_order : list
        Specified order for appearance of the `style` variable levels
        otherwise they are determined from the data. Not relevant when the
        `style` variable is numeric.
    legend : "auto", "brief", "full", or False
        How to draw the legend. If "brief", numeric `hue` and `size`
        variables will be represented with a sample of evenly spaced values.
        If "full", every group will get an entry in the legend. If "auto",
        choose between brief or full representation based on number of levels.
        If `False`, no legend data is added and no legend is drawn.
    ax : :class:`matplotlib.axes.Axes`
        Pre-existing axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca`
        internally.
    kwargs : key, value mappings
        Other keyword arguments are passed down to
        :meth:`matplotlib.axes.Axes.scatter`.
    
    Returns
    -------
    :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.
    
    See Also
    --------
    lineplot : Plot data using lines.
    stripplot : Plot a categorical scatter with jitter.
    swarmplot : Plot a categorical scatter with non-overlapping points.
    
    Examples
    --------
    
    .. include:: ../docstrings/scatterplot.rst
