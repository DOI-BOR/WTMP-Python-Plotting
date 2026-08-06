import os
import matplotlib as mpl			# matplotlib is used here only for its named colormap registry, to resolve contour plot colormap names (e.g. 'jet') to actual objects.

import WAT_Constants as WC
import WAT_Functions as WF
import WAT_Reader as WR
import WAT_Plots as WP				# WAT_Plots supplies the style-pattern translators (java-style line/point pattern names -> matplotlib-compatible values) used throughout below.

# Shared constants instance, used here for the default color cycle
# (constants.def_colors).
constants = WC.WAT_Constants()


def getDefaultDefaultLineStyles(i):
    """
    Build a fallback line style for the i-th line on a plot.

    Used as the last-resort default when a line's parameter isn't found
    in the user-supplied defaults file (``defaultLineStyles.xml``) at
    all. Colors cycle through ``constants.def_colors`` by index.

    Parameters
    ----------
    i : int
        Index of this line among all lines on the plot (used to select a
        color from the cycle and to keep multiple lines visually
        distinct).

    Returns
    -------
    dict
        Line style dictionary with ``'linewidth'``, ``'linecolor'``,
        ``'linestylepattern'``, ``'alpha'``, and ``'zorder'`` keys.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getDefaultDefaultLineStyles(0)
    {'linewidth': 2, 'linecolor': '#88CCEE', 'linestylepattern': 'solid', 'alpha': 1.0, 'zorder': 4}
    """

    # Wrap the index around the color list so any number of lines gets a
    # (repeating) color rather than an index error.
    while i >= len(constants.def_colors):
        i = i - len(constants.def_colors)
    # build and return the fixed-shape style dict, using the wrapped index to pick a color
    return {'linewidth': 2, 'linecolor': constants.def_colors[i],
            'linestylepattern': 'solid', 'alpha': 1.0, 'zorder': 4}


def getDefaultDefaultForecastTableHeaders():
    """
    Return the fallback column order for forecast tables.

    Used when no table headers are specified by the user in the report
    input file.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
        Default forecast table column names, in display order.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getDefaultDefaultForecastTableHeaders()
    ['member', 'operationsname', 'metname', 'temptargetname']
    """

    # fixed default column order for forecast tables
    return ['member', 'operationsname', 'metname', 'temptargetname']


def getDefaultDefaultPointStyles(i):
    """
    Build a fallback marker/point style for the i-th series on a plot.

    Used as the last-resort default when a point's parameter isn't found
    in the user-supplied defaults file at all. Colors cycle through
    ``constants.def_colors`` by index (same cycle used for lines, so
    lines and points sharing an index visually match).

    Parameters
    ----------
    i : int
        Index of this series among all series on the plot.

    Returns
    -------
    dict
        Point style dictionary with ``'pointfillcolor'``,
        ``'pointlinecolor'``, ``'symboltype'``, ``'symbolsize'``,
        ``'numptsskip'``, and ``'alpha'`` keys.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getDefaultDefaultPointStyles(0)
    {'pointfillcolor': '#88CCEE', 'pointlinecolor': '#88CCEE', 'symboltype': 1, 'symbolsize': 5, 'numptsskip': 0, 'alpha': 1.0}
    """

    # wrap the index around the color list, same as getDefaultDefaultLineStyles
    while i >= len(constants.def_colors):
        i = i - len(constants.def_colors)
    # build and return the fixed-shape style dict, reusing the same color for fill and outline
    return {'pointfillcolor': constants.def_colors[i], 'pointlinecolor': constants.def_colors[i], 'symboltype': 1,
            'symbolsize': 5, 'numptsskip': 0, 'alpha': 1.0}


def getDefaultDefaultTextStyles():
    """
    Build the fallback style for text/annotation elements on a plot.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Text style dictionary with ``'fontsize'``, ``'fontcolor'``,
        ``'alpha'``, and ``'horizontalalignment'`` keys.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getDefaultDefaultTextStyles()
    {'fontsize': 9, 'fontcolor': 'black', 'alpha': 1.0, 'horizontalalignment': 'left'}
    """

    # fixed default text style dict
    return {'fontsize': 9, 'fontcolor': 'black', 'alpha': 1.0, 'horizontalalignment': 'left'}


def getDefaultLineSettings(defaultLineStyles, LineSettings, param, i, debug=False):
    """
    Fill in missing line/point style settings using layered defaults.

    Resolution order for each style property: (1) whatever is already
    explicitly set in ``LineSettings``; (2) the parameter-specific
    default from ``defaultLineStyles`` (parsed from the user's
    ``defaultLineStyles.xml``), if the parameter is recognized; (3) the
    generic index-based fallback from ``getDefaultDefaultLineStyles``/
    ``getDefaultDefaultPointStyles``. Whether lines, points, or both are
    drawn is first determined via ``getDrawFlags``. Java-style pattern
    names are translated to matplotlib-compatible values at the end of
    each section.

    Parameters
    ----------
    defaultLineStyles : dict
        Parsed contents of the report's ``defaultLineStyles.xml``,
        keyed by lowercase parameter name.
    LineSettings : dict
        Settings dictionary for this specific line/point series; updated
        and returned with any missing keys filled in.
    param : str or None
        The data parameter name (e.g. ``'temperature'``) used to look up
        parameter-specific defaults, or ``None`` if not applicable.
    i : int
        Index of this line/series on the plot, used to select a
        sequential default color/style.
    debug : bool, optional
        Passed through to logging calls (default ``False``).

    Returns
    -------
    dict
        The updated ``LineSettings`` dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> settings = getDefaultLineSettings(defaultLineStyles, {}, 'temperature', 0)
    """

    # decide whether a line, points, or both should be drawn for this series
    LineSettings = getDrawFlags(LineSettings)
    if LineSettings['drawline'].lower() == 'true':
        if param != None:
            if param.lower() in defaultLineStyles.keys():
                # Cycle through the parameter's defined line-style list by
                # index, wrapping around if there are more lines than
                # defined styles.
                while i >= len(defaultLineStyles[param.lower()]['lines']):
                    i = i - len(defaultLineStyles[param.lower()]['lines'])

                # pull the parameter-specific default style for this (wrapped) index
                default_lines = defaultLineStyles[param.lower()]['lines'][i]

                # only fill in keys not already explicitly set on this line
                for key in default_lines.keys():
                    if key not in LineSettings.keys():
                        LineSettings[key] = default_lines[key]

        # Fill anything still missing with the generic index-based
        # fallback style (applies regardless of whether a
        # parameter-specific default was found above).
        default_default_lines = getDefaultDefaultLineStyles(i)
        # fill in any keys still missing after the parameter-specific pass above
        for key in default_default_lines.keys():
            if key not in LineSettings.keys():
                LineSettings[key] = default_default_lines[key]

        # validate/resolve the final line color, falling back to the generic default if invalid
        LineSettings['linecolor'] = WF.confirmColor(LineSettings['linecolor'], default_default_lines['linecolor'], debug=debug)
        # translate any java-style pattern names into matplotlib-compatible values
        LineSettings = WP.translateLineStylePatterns(LineSettings)

    if LineSettings['drawpoints'] == 'true':
        # Same layered-default pattern as above, but for point/marker
        # styling instead of line styling.
        if param in defaultLineStyles.keys():
            # cycle through the parameter's defined style list, wrapping as needed
            while i >= len(defaultLineStyles[param]['lines']):
                i = i - len(defaultLineStyles[param]['lines'])
            default_lines = defaultLineStyles[param]['lines'][i]
            # only fill in keys not already explicitly set on this line
            for key in default_lines.keys():
                if key not in LineSettings.keys():
                    LineSettings[key] = default_lines[key]

        # generic index-based fallback point style
        default_default_points = getDefaultDefaultPointStyles(i)

        # fill in any keys still missing after the parameter-specific pass above
        for key in default_default_points.keys():
            if key not in LineSettings.keys():
                LineSettings[key] = default_default_points[key]

        # validate/resolve the final point fill and outline colors
        LineSettings['pointfillcolor'] = WF.confirmColor(LineSettings['pointfillcolor'], default_default_points['pointfillcolor'], debug=debug)
        LineSettings['pointlinecolor'] = WF.confirmColor(LineSettings['pointlinecolor'], default_default_points['pointlinecolor'], debug=debug)

        try:
            # A skip value of 0 would mean "skip every point" (i.e. draw
            # none); treat that as "skip nothing" (draw every point)
            # instead.
            if int(LineSettings['numptsskip']) == 0:
                LineSettings['numptsskip'] = 1
        except ValueError:
            # numptsskip wasn't a valid integer at all, fall back to a safe default
            WF.print2stdout('Invalid setting for numptsskip.', LineSettings['numptsskip'], debug=debug)
            WF.print2stdout('defaulting to 25', debug=debug)
            LineSettings['numptsskip'] = 25

        # translate any java-style point pattern names into matplotlib-compatible values
        LineSettings = WP.translatePointStylePatterns(LineSettings)

    return LineSettings


def getDefaultGateLineSettings(GateLineSettings, i, debug=False):
    """
    Fill in missing gate-plot line/point style settings with defaults.

    Same layered-default approach as ``getDefaultLineSettings``, but
    simplified for gate operation plots (no parameter-specific lookup —
    gates always use the generic index-based defaults).

    Parameters
    ----------
    GateLineSettings : dict
        Settings dictionary for a single gate's line/point series;
        updated and returned with any missing keys filled in.
    i : int
        Index of this gate among all gates on the plot, used to select a
        sequential default color/style.
    debug : bool, optional
        Passed through to logging calls (default ``False``).

    Returns
    -------
    dict
        The updated ``GateLineSettings`` dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions, though see
        the Notes section below regarding a possible ``KeyError`` in the
        point-color resolution step.

    Notes
    -----
    In the point-styling section, the point fill/outline color
    resolution references ``default_default_lines`` (the line style
    dict) rather than ``default_default_points`` (the point style
    dict). Since the line style dict does not define
    ``'pointlinecolor'``/``'pointfillcolor'`` keys, this could raise a
    ``KeyError`` at runtime if ``GateLineSettings`` did not already have
    those keys set from elsewhere. This matches the source file exactly
    as written and has not been changed, per the "no logic changes"
    scope of this documentation pass.

    Examples
    --------
    >>> settings = getDefaultGateLineSettings({}, 0)
    """

    # decide whether a line, points, or both should be drawn for this gate
    GateLineSettings = getDrawFlags(GateLineSettings)
    if GateLineSettings['drawline'] == 'true':
        # generic index-based fallback line style (no parameter lookup for gates)
        default_default_lines = getDefaultDefaultLineStyles(i)
        # only fill in keys not already explicitly set
        for key in default_default_lines.keys():
            if key not in GateLineSettings.keys():
                GateLineSettings[key] = default_default_lines[key]

        # translate java-style pattern names and validate/resolve the final line color
        GateLineSettings = WP.translateLineStylePatterns(GateLineSettings)
        GateLineSettings['linecolor'] = WF.confirmColor(GateLineSettings['linecolor'], default_default_lines['linecolor'], debug=debug)

    if GateLineSettings['drawpoints'] == 'true':
        # generic index-based fallback point style (no parameter lookup for gates)
        default_default_points = getDefaultDefaultPointStyles(i)
        # only fill in keys not already explicitly set
        for key in default_default_points.keys():
            if key not in GateLineSettings.keys():
                GateLineSettings[key] = default_default_points[key]
        try:
            # Treat a skip value of 0 ("skip every point") as "draw every
            # point" instead, same rationale as getDefaultLineSettings.
            if int(GateLineSettings['numptsskip']) == 0:
                GateLineSettings['numptsskip'] = 1
        except ValueError:
            # numptsskip wasn't a valid integer at all, fall back to a safe default
            WF.print2stdout('Invalid setting for numptsskip.', GateLineSettings['numptsskip'], debug=debug)
            WF.print2stdout('defaulting to 25', debug=debug)
            GateLineSettings['numptsskip'] = 25

        # NOTE: this references default_default_lines (not
        # default_default_points) for the point fallback colors. The
        # 'lines' default dict does not define 'pointlinecolor'/
        # 'pointfillcolor' keys, so this looks like it may raise a
        # KeyError at runtime if GateLineSettings didn't already have
        # these keys set - matches the source file as written; not
        # changed here per the "no logic changes" scope of this pass.
        # validate/resolve the point outline and fill colors
        GateLineSettings['pointlinecolor'] = WF.confirmColor(GateLineSettings['pointlinecolor'], default_default_lines['pointlinecolor'], debug=debug)
        GateLineSettings['pointfillcolor'] = WF.confirmColor(GateLineSettings['pointfillcolor'], default_default_lines['pointfillcolor'], debug=debug)
        # translate java-style point pattern names into matplotlib-compatible values
        GateLineSettings = WP.translatePointStylePatterns(GateLineSettings)

    return GateLineSettings


def getDefaultContourLineSettings(contour_settings):
    """
    Fill in missing contour-line style settings with bare-minimum defaults.

    Unlike the line/point functions above, this uses a single fixed
    default dictionary (no color cycling by index), since contour lines
    are typically drawn in a uniform style.

    Parameters
    ----------
    contour_settings : dict
        Settings dictionary for the contour plot; updated and returned
        with any missing keys filled in.

    Returns
    -------
    dict
        The updated ``contour_settings`` dictionary, with
        ``'linestylepattern'`` translated to its matplotlib equivalent.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> settings = getDefaultContourLineSettings({})
    """

    # fixed default style dict for contour lines, applied wholesale where keys are missing
    default_contour_settings = {'linecolor': 'grey',
                                'linewidth': 1,
                                'linestylepattern': 'solid',
                                'alpha': 1,
                                'contourlinetext': 'false',
                                'fontsize': 10,
                                'text_inline': 'true',
                                'inline_spacing': 10,
                                'legend': 'false'}

    # only fill in keys not already explicitly set on the contour settings
    for key in default_contour_settings.keys():
        if key not in contour_settings:
            contour_settings[key] = default_contour_settings[key]
            # 'text_inline' is used elsewhere as a real Python bool
            # (e.g. passed to matplotlib's clabel), so once the default
            # string value is applied, convert it to an actual bool.
            if key == 'text_inline':
                if contour_settings[key].lower() == 'true':
                    # convert the string default to an actual Python bool
                    contour_settings[key] = True
                else:
                    contour_settings[key] = False

    # translate any java-style pattern names into matplotlib-compatible values
    contour_settings = WP.translateLineStylePatterns(contour_settings)

    return contour_settings


def getDefaultContourSettings(object_settings, debug=False):
    """
    Fill in missing colorbar/colormap settings for a contour plot.

    Ensures ``object_settings['colorbar']`` exists and has a resolved
    matplotlib colormap object plus bin/tick-count defaults. If the user
    specified an invalid colormap name, falls back to the 'jet' colormap
    with a logged warning.

    Parameters
    ----------
    object_settings : dict
        Settings dictionary for the contour object; updated and returned
        with a populated/validated ``'colorbar'`` sub-dictionary.
    debug : bool, optional
        Passed through to logging calls (default ``False``).

    Returns
    -------
    dict
        The updated ``object_settings`` dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions; an invalid
        user-supplied colormap name is caught internally and replaced
        with the 'jet' default.

    Examples
    --------
    >>> settings = getDefaultContourSettings({'colorbar': {'colormap': 'viridis'}})
    """

    # defaultColormap = mpl.cm.get_cmap('jet')
    # resolve the 'jet' colormap once up front, used both as the base default and as a fallback
    defaultColormap = mpl.colormaps['jet']
    default_colorbar_settings = {'colormap': defaultColormap,
                                 'bins': 10,
                                 'numticks': 5}

    if 'colorbar' in object_settings.keys():
        if 'colormap' in object_settings['colorbar'].keys():
            try:
                # usercolormap = mpl.cm.get_cmap(object_settings['colorbar']['colormap'])
                # Resolve the user-supplied colormap name to an actual
                # matplotlib colormap object.
                usercolormap = mpl.colormaps[object_settings['colorbar']['colormap']]
                object_settings['colormap'] = usercolormap
            except ValueError:
                # invalid colormap name given, fall back to the default and log why
                WF.print2stdout('User selected invalid colormap:', object_settings['colorbar']['colormap'], debug=debug)
                WF.print2stdout('Tip: make sure capitalization is correct!', debug=debug)
                WF.print2stdout('Defaulting to Jet.', debug=debug)
                object_settings['colormap'] = defaultColormap
    else:
        # no colorbar settings at all yet, start with an empty dict to fill below
        object_settings['colorbar'] = {}

    # Fill in any colorbar settings the user didn't specify.
    for key in default_colorbar_settings.keys():
        if key not in object_settings['colorbar']:
            object_settings['colorbar'][key] = default_colorbar_settings[key]

    return object_settings


def getDefaultStraightLineSettings(LineSettings, debug):
    """
    Fill in missing style settings for a straight reference line (hline/vline).

    Similar to ``getDefaultLineSettings``, but always uses the generic
    index-0 line defaults with the color forced to black (reference
    lines don't need to cycle through the plot's color palette).

    Parameters
    ----------
    LineSettings : dict
        Settings dictionary for the reference line; updated and returned
        with any missing keys filled in.
    debug : bool
        Passed through to logging calls.

    Returns
    -------
    dict
        The updated ``LineSettings`` dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> settings = getDefaultStraightLineSettings({}, False)
    """

    # decide whether a line, points, or both should be drawn for this reference line
    LineSettings = getDrawFlags(LineSettings)
    # always use index 0 for reference lines, since they don't need to cycle colors
    default_default_lines = getDefaultDefaultLineStyles(0)
    # force the default color to black rather than the first cycle color
    default_default_lines['linecolor'] = 'black' #don't need different colors by default..
    # only fill in keys not already explicitly set
    for key in default_default_lines.keys():
        if key not in LineSettings.keys():
            LineSettings[key] = default_default_lines[key]

    # translate java-style pattern names and validate/resolve the final line color
    LineSettings = WP.translateLineStylePatterns(LineSettings)
    LineSettings['linecolor'] = WF.confirmColor(LineSettings['linecolor'], default_default_lines['linecolor'], debug=debug)

    return LineSettings


def getDefaultTextSettings(TextSettings, debug):
    """
    Fill in missing style settings for a text/annotation element.

    Parameters
    ----------
    TextSettings : dict
        Settings dictionary for the text element; updated and returned
        with any missing keys filled in.
    debug : bool
        Passed through to logging calls.

    Returns
    -------
    dict
        The updated ``TextSettings`` dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> settings = getDefaultTextSettings({}, False)
    """

    # fixed default text style dict
    default_default_text = getDefaultDefaultTextStyles()
    # only fill in keys not already explicitly set
    for key in default_default_text.keys():
        if key not in TextSettings.keys():
            TextSettings[key] = default_default_text[key]

    # validate/resolve the final font color, falling back to the default if invalid
    TextSettings['fontcolor'] = WF.confirmColor(TextSettings['fontcolor'], default_default_text['fontcolor'], debug=debug)

    return TextSettings


#################################################################
# Helper Functions #
#################################################################

def getDrawFlags(LineSettings):
    """
    Infer whether lines and/or points should be drawn for a series.

    If ``'drawline'``/``'drawpoints'`` aren't explicitly set, infers them
    from whether any line-specific or point-specific style keys are
    present in ``LineSettings``. If neither can be inferred (and neither
    is explicitly set), defaults to drawing a line. As a final
    safety net, if both end up ``'false'``, forces ``'drawline'`` to
    ``'true'`` so the series is visible somehow.

    Parameters
    ----------
    LineSettings : dict
        Settings dictionary for a line/point series; updated and
        returned with ``'drawline'`` and ``'drawpoints'`` keys
        (string ``'true'``/``'false'``) set.

    Returns
    -------
    dict
        The updated ``LineSettings`` dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getDrawFlags({'linecolor': 'red'})
    {'linecolor': 'red', 'drawline': 'true', 'drawpoints': 'false'}
    """

    #unless explicitly stated, look for key identifiers to draw lines or not
    # style keys whose presence implies the user intended to draw a line
    LineVars = ['linecolor', 'linestylepattern', 'linewidth']
    # style keys whose presence implies the user intended to draw points
    PointVars = ['pointfillcolor', 'pointlinecolor', 'symboltype', 'symbolsize', 'numptsskip', 'markersize']

    if 'drawline' not in LineSettings.keys():
        # If any line-specific key was set by the user, assume they meant
        # to draw a line.
        for var in LineVars:
            if var in LineSettings.keys():
                LineSettings['drawline'] = 'true'
                break
        if 'drawline' not in LineSettings.keys():
            # no line-specific key was found, default to not drawing a line
            LineSettings['drawline'] = 'false'

    if 'drawpoints' not in LineSettings.keys():
        # Same inference, but for point-specific keys.
        for var in PointVars:
            if var in LineSettings.keys():
                LineSettings['drawpoints'] = 'true'
                break
        if 'drawpoints' not in LineSettings.keys():
            # no point-specific key was found, default to not drawing points
            LineSettings['drawpoints'] = 'false'

    if LineSettings['drawpoints'] == 'false' and LineSettings['drawline'] == 'false':
        # neither could be inferred at all, force a line so the series is visible somehow
        LineSettings['drawline'] = 'true' #gotta do something..

    return LineSettings


def readDefaultLineStylesFile(Report):
    """
    Locate and parse the report's default line styles XML file.

    Parameters
    ----------
    Report : object
        The main Report Generator instance; used for its ``studyDir``
        attribute to locate the expected file path.

    Returns
    -------
    dict
        Parsed default line styles, keyed by lowercase parameter name
        (as produced by ``WR.readDefaultLineStyle``).

    Raises
    ------
    None
        This function does not explicitly raise exceptions; a missing
        file is logged via ``WF.checkExists`` rather than raised.

    Examples
    --------
    >>> defaultLineStyles = readDefaultLineStylesFile(Report)
    """

    # build the expected path to the report's defaultLineStyles.xml file
    defaultLinesFile = os.path.join(Report.studyDir, 'reports', 'defaultLineStyles.xml')
    # Log a warning (via checkExists) if the expected file is missing,
    # rather than failing silently later when it's read.
    WF.checkExists(defaultLinesFile)
    # defaultLinesFile = os.path.join(self.default_dir, 'defaultLineStyles.xml') #TODO: implement with build
    # parse the XML file into the keyed default-style dictionary
    defaultLineStyles = WR.readDefaultLineStyle(defaultLinesFile)
    return defaultLineStyles