import pickle
import datetime as dt
import numpy as np
import matplotlib as mpl				# matplotlib is used here directly for tick locators/formatters and color-cycle utilities beyond what pyplot's high-level API exposes.
from collections import Counter

import WAT_Functions as WF
import WAT_Time as WT
import WAT_Defaults as WD


class Plots(object):
    """
    Low-level matplotlib plotting helpers used by the report generator.

    This class wraps the repetitive parts of building time series plots:
    drawing lines/points/collection envelopes, formatting date and value
    axes and their tick labels, drawing horizontal/vertical reference
    lines, and reconciling data of different time intervals so multiple
    series can share one axis. Higher-level report code (which knows
    about specific plot types like temperature or flow plots) calls into
    these methods to do the actual matplotlib work.

    Attributes
    ----------
    Report : object
        The main Report Generator instance this plotting helper serves.
    """

    def __init__(self, Report):
        """
        Initialize the plotting helper class.

        Parameters
        ----------
        Report : object
            The main Report Generator instance.

        Returns
        -------
        None
            This is a constructor and does not return a value.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots = Plots(Report)
        """

        # keep a reference back to the parent report for shared state (dates, debug flag, constants, etc.)
        self.Report = Report

    def confirmAxis(self, object_settings):
        """
        Ensure the object settings contain an axis list, creating one if missing.

        Parameters
        ----------
        object_settings : dict
            Settings dictionary for the current object.

        Returns
        -------
        dict
            The ``object_settings`` dictionary, guaranteed to contain an
            ``'axs'`` key (a list with at least one, possibly empty,
            axis settings dict).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.confirmAxis({})
        {'axs': [{}]}
        """

        if 'axs' not in object_settings.keys():
            # no axis list defined at all, seed one with a single empty axis
            object_settings['axs'] = [{}] #empty axis object
        return object_settings

    def seperateCollectionLines(self, line_draw_settings):
        """
        Split a forecast collection line into one entry per member.

        Parameters
        ----------
        line_draw_settings : dict
            Draw settings for a collection line; must contain
            ``'members'`` to be split.

        Returns
        -------
        dict
            Dictionary keyed by member, each a full copy of
            ``line_draw_settings`` customized with a per-member label,
            color, and ``'numtimesused'`` index. If ``'members'`` is
            missing, returns ``line_draw_settings`` unchanged.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> collection_draw_settings = plots.seperateCollectionLines(line_draw_settings)
        """

        if 'members' in line_draw_settings:
            collection_draw_settings = {}
            members = line_draw_settings['members']
            # build a customized copy of the settings for each individual member
            for mi, member in enumerate(members):
                # Start each member's settings as a full copy of the
                # shared collection settings, then customize per-member
                # below (numtimesused for color cycling, label, colors).
                collection_draw_settings[member] = {}
                collection_draw_settings[member].update(line_draw_settings)
                collection_draw_settings[member]['numtimesused'] = mi
                if '%%member%%' not in collection_draw_settings[member]['label']:
                    # No member placeholder in the label: just append the
                    # raw member identifier.
                    collection_draw_settings[member]['label'] = f"{collection_draw_settings[member]['label']}: {member}"
                else:
                    # Label contains a %%member%% placeholder: replace it
                    # with the "original" member number (i.e. the
                    # user-facing schedule/member number rather than the
                    # internal collection index).
                    # get the ensemble set for the current member
                    curr_ensemble_set = WF.matchMemberToEnsembleSet(self.Report.ensembleSets, member)
                    collection_draw_settings[member]['label'] = collection_draw_settings[member]['label'].replace('%%member%%', WF.getOriginalMemberNumber(member, curr_ensemble_set, self.Report.DSSFile,
                                                                                                                                                     self.Report.alternativeFpart, self.Report.StartTime, self.Report.EndTime, self.Report.debug))
                # resolve a distinct color for this member, avoiding duplicates across the collection
                collection_draw_settings[member] = WF.fixDuplicateColors(collection_draw_settings[member])
            return collection_draw_settings

        else:
            # no member list at all, nothing to split
            WF.print2stdout('Unable to get members. Cannot seperate collection lines.', debug=self.Report.debug)
            return line_draw_settings

    def setTimeSeriesXlims(self, cur_obj_settings, yearstr, years):
        """
        Resolve '%%year%%' placeholders in x-limits settings for a plot.

        Parameters
        ----------
        cur_obj_settings : dict
            Current plotting object settings dictionary.
        yearstr : str
            The year string to substitute (e.g. ``'2015'`` or the
            report's overall year-range string for all-years plots).
        years : list
            List of years being iterated over; used to check whether
            this is an all-years pass.

        Returns
        -------
        dict
            The updated ``cur_obj_settings`` dictionary.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> cur_obj_settings = plots.setTimeSeriesXlims(cur_obj_settings, '2020', [2020])
        """

        if 'ALLYEARS' not in years:
            # Single-year (or year-block) plot: substitute the actual
            # year string wherever '%%year%%' appears in the settings.
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)
        else:
            # All-years plot: if explicit x-limits reference '%%year%%',
            # resolve them to the report's overall first/last year
            # (rather than a single specific year) before the general
            # substitution pass below.
            if 'xlims' in cur_obj_settings.keys():
                if 'min' in cur_obj_settings['xlims']:
                    # substitute the report's actual first year into the min limit
                    cur_obj_settings['xlims']['min'] = WF.updateFlaggedValues(cur_obj_settings['xlims']['min'],
                                                                              '%%year%%', str(self.Report.years[0]))
                if 'max' in cur_obj_settings['xlims']:
                    # substitute the report's actual last year into the max limit
                    cur_obj_settings['xlims']['max'] = WF.updateFlaggedValues(cur_obj_settings['xlims']['max'],
                                                                              '%%year%%', str(self.Report.years[-1]))
            # run the general substitution pass over everything else
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)

        return cur_obj_settings

    def getRelativeMasterSet(self, linedata, line_settings):
        """
        Sum multiple lines' data together on a common interval and units.

        Parameters
        ----------
        linedata : dict
            Dictionary of line data keyed by flag, each with
            ``'dates'``/``'values'``.
        line_settings : dict
            Per-line settings dictionary; checked for ``'interval'`` and
            ``'type'`` (DSS interval type) per line.

        Returns
        -------
        RelativeMasterSet : array_like
            The summed values across all lines, resampled to a common
            (the coarsest) interval if any lines specified one.
        RelativeLineSettings : dict
            Settings for the resulting summed series, including
            ``'interval'``, ``'type'``, and ``'units'``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        Marked with a ``#TODO: deal with irregular intervals`` comment
        in the original source.

        Examples
        --------
        >>> RelativeMasterSet, RelativeLineSettings = plots.getRelativeMasterSet(linedata, line_settings)
        """

        #add all the data together. then we can use this when plotting it to get %
        #TODO: deal with irregular intervals
        intervals = {}
        biggest_interval = None
        type = 'INST-VAL'
        # Find the coarsest (largest) interval among all lines that
        # specify one, since every line needs to be resampled to a
        # common interval before they can be summed together.
        for line in linedata.keys():
            if 'interval' in line_settings[line].keys():
                # compute this line's actual sampling interval from its dates
                td = WF.getTimeInterval(linedata[line]['dates'])
                if line_settings[line]['interval'].upper() not in intervals.keys():
                    intervals[line_settings[line]['interval'].upper()] = td
                if biggest_interval == None:
                    # first interval seen, seed the running "biggest" tracker
                    biggest_interval = line_settings[line]['interval'].upper()
                    if 'type' in line_settings[line].keys():
                        type = line_settings[line]['type'].upper()
                else:
                    if td > intervals[biggest_interval]:
                        # this line's interval is coarser than the current biggest, replace it
                        biggest_interval = line_settings[line]['interval'].upper()
                        if line_settings[line]['type'] in line.keys():
                            type = line_settings[line]['type'].upper()

        RelativeLineSettings = {'interval': biggest_interval,
                                'type': type}
        RelativeMasterSet = []
        units = []
        for li, line in enumerate(linedata.keys()):
            # Work on deep copies so the original linedata/line_settings
            # dicts aren't mutated by the unit conversion/resampling
            # below.
            curline = pickle.loads(pickle.dumps(linedata[line], -1))
            curline_settings = pickle.loads(pickle.dumps(line_settings[line], -1))
            # standardize every line to metric units before summing them together
            curline['values'], curline_settings['units'] = WF.convertUnitSystem(curline['values'], curline_settings['units'], 'metric', debug=self.Report.debug) #just make everything metric..
            units.append(curline_settings['units'])
            if li == 0:
                # First line seeds the running master sum; resample to
                # the common interval first if one was determined above.
                if biggest_interval != None:
                    _, RelativeMasterSet = WT.changeTimeSeriesInterval(curline['dates'], curline['values'],
                                                                                RelativeLineSettings,
                                                                                self.Report.startYear)
                else:
                    RelativeMasterSet = curline['values']
            else:
                # Subsequent lines are resampled the same way and added
                # into the running total.
                if biggest_interval != None:
                    curline['interval'] = biggest_interval
                    curline['type'] = type
                    _, newvals = WT.changeTimeSeriesInterval(curline['dates'], curline['values'],
                                                              RelativeLineSettings,
                                                              self.Report.startYear)
                    RelativeMasterSet += newvals
                else:
                    RelativeMasterSet += curline['values']

        # Use whichever unit ended up most common across all the lines
        # (all should be metric after the conversion above, so this
        # mostly just picks a representative unit string).
        RelativeLineSettings['units'] = WF.getMostCommon(units)

        return RelativeMasterSet, RelativeLineSettings

    def plot(self, dates, values, curax, line_draw_settings):
        """
        Draw a series as lines, points, or both, per its draw settings.

        Parameters
        ----------
        dates : array_like
            X-axis values (dates or otherwise).
        values : array_like
            Y-axis values to plot.
        curax : matplotlib.axes.Axes
            Axis to plot on.
        line_draw_settings : dict
            Settings configuring how the line is drawn; must contain
            ``'drawline'`` and ``'drawpoints'`` flags.

        Returns
        -------
        None
            Draws directly onto ``curax``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        When both ``'drawline'`` and ``'drawpoints'`` are true, this
        method calls ``plotLinesAndPoints(dates, line_draw_settings,
        curax, line_draw_settings)``, passing ``line_draw_settings`` in
        the ``y`` (values) argument position instead of the actual
        ``values`` array. Since ``plotLinesAndPoints`` expects
        ``(x, y, curaxis, settings)``, this means the ``values`` array
        supplied to ``plot`` is never actually used when both lines and
        points are requested. This matches the source file as written
        and has not been changed here, per the "no logic changes" scope
        of this documentation pass.

        Examples
        --------
        >>> plots.plot(dates, values, curax, line_draw_settings)
        """

        # Dispatch to the appropriate plotting method based on whether
        # lines, points, or both were requested for this series.
        if line_draw_settings['drawline'].lower() == 'true' and line_draw_settings['drawpoints'].lower() == 'true':
            # NOTE: this passes `line_draw_settings` in the `y` (values)
            # argument position instead of `values` - plotLinesAndPoints
            # expects (x, y, curaxis, settings), so the actual `values`
            # array supplied to this method is never used when both
            # lines and points are requested. This matches the source
            # file as written; not changed here per the "no logic
            # changes" scope of this documentation pass.
            self.plotLinesAndPoints(dates, line_draw_settings, curax, line_draw_settings)
        elif line_draw_settings['drawline'].lower() == 'true':
            # only a line is requested
            self.plotLines(dates, values, curax, line_draw_settings)
        elif line_draw_settings['drawpoints'].lower() == 'true':
            # only points are requested
            self.plotPoints(dates, values, curax, line_draw_settings)

    def plotLinesAndPoints(self, x, y, curaxis, settings):
        """
        Draw a line with markers at intervals on the given axis.

        Parameters
        ----------
        x : array_like
            Data for the x axis (dates or otherwise).
        y : array_like
            Data for the y axis.
        curaxis : matplotlib.axes.Axes
            Current axis object to draw on.
        settings : dict
            Settings dictionary for the plot object (line/marker color,
            width, style, size, spacing, etc.).

        Returns
        -------
        None
            Draws directly onto ``curaxis``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.plotLinesAndPoints(x, y, curaxis, settings)
        """

        # draw a single matplotlib line with markers, pulling every visual property from settings
        curaxis.plot(x, y, label=settings['label'], c=settings['linecolor'],
                     lw=settings['linewidth'], ls=settings['linestylepattern'],
                     marker=settings['symboltype'], markerfacecolor=settings['pointfillcolor'],
                     markeredgecolor=settings['pointlinecolor'], markersize=float(settings['symbolsize']),
                     markevery=int(settings['numptsskip']), zorder=float(settings['zorder']),
                     alpha=float(settings['alpha']))

    def plotLines(self, x, y, curaxis, settings):
        """
        Draw a plain line (no markers) on the given axis.

        Parameters
        ----------
        x : array_like
            Data for the x axis (dates or otherwise).
        y : array_like
            Data for the y axis.
        curaxis : matplotlib.axes.Axes
            Current axis object to draw on.
        settings : dict
            Settings dictionary for the plot object (line color, width,
            style, etc.).

        Returns
        -------
        None
            Draws directly onto ``curaxis``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.plotLines(x, y, curaxis, settings)
        """

        # draw a plain matplotlib line with no markers
        curaxis.plot(x, y, label=settings['label'], c=settings['linecolor'],
                     lw=settings['linewidth'], ls=settings['linestylepattern'],
                     zorder=float(settings['zorder']),
                     alpha=float(settings['alpha']))

    def plotPoints(self, x, y, curaxis, settings):
        """
        Draw points (no connecting line) on the given axis.

        Parameters
        ----------
        x : array_like
            Data for the x axis (dates or otherwise).
        y : array_like
            Data for the y axis.
        curaxis : matplotlib.axes.Axes
            Current axis object to draw on.
        settings : dict
            Settings dictionary for the plot object (marker color,
            size, spacing, etc.).

        Returns
        -------
        None
            Draws directly onto ``curaxis``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.plotPoints(x, y, curaxis, settings)
        """

        # numptsskip controls point density: slice every Nth point
        # rather than plotting every single one (keeps dense series from
        # becoming a solid smear of markers).
        curaxis.scatter(x[::int(settings['numptsskip'])], y[::int(settings['numptsskip'])],
                        marker=settings['symboltype'], facecolor=settings['pointfillcolor'],
                        edgecolor=settings['pointlinecolor'], s=float(settings['symbolsize']),
                        label=settings['label'], zorder=float(settings['zorder']),
                        alpha=float(settings['alpha']))

    def plotCollectionEnvelopes(self, dates, values, curax, settings):
        """
        Compute and draw percentile envelope lines for a forecast collection.

        Parameters
        ----------
        dates : array_like
            Dates for the envelope lines' x-axis.
        values : dict
            Collection values keyed by ensemble member.
        curax : matplotlib.axes.Axes
            Current axis to plot on.
        settings : dict
            Settings for the object; must contain ``'envelopes'`` (a
            list of envelope definitions with ``'percent'`` keys) to
            draw anything.

        Returns
        -------
        None
            Draws directly onto ``curax``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.plotCollectionEnvelopes(dates, values, curax, settings)
        """

        if 'envelopes' in settings.keys() and len(values.keys()) > 0:
            # Compute every requested percentile envelope across the
            # ensemble collection, then draw each one as its own line
            # (e.g. the 10th/50th/90th percentile bands).
            collection_evelopes = self.Report.Data.computeCollectionEnvelopes(values, settings['envelopes'])
            for envelope_settings in settings['envelopes']:
                envelope = envelope_settings['percent']
                # merge the plot's general settings underneath this specific envelope's settings
                envelope_settings = WF.replaceDefaults(self, settings, envelope_settings)
                if envelope in collection_evelopes.keys():
                    # draw this percentile's computed series as its own line
                    envelope_vals = collection_evelopes[envelope]
                    self.plotLines(dates, envelope_vals, curax, envelope_settings)

    def formatDateXAxis(self, curax, object_settings, twin=False):
        """
        Format a plot's date x-axis limits (and optional secondary axis).

        Parameters
        ----------
        curax : matplotlib.axes.Axes
            Current plot axis.
        object_settings : dict
            Settings dictionary; checked for ``'xlims'``/``'xlims2'``
            (min/max) depending on ``twin``.
        twin : bool, optional
            If ``True``, configures the top (secondary) axis using
            ``'xlims2'`` (falling back to ``'xlims'`` if not defined)
            instead of the bottom axis (default ``False``).

        Returns
        -------
        bool
            ``useplot``: ``True`` if the resolved x-limits overlap the
            axis's actual data range and the plot should be used,
            ``False`` if the requested window falls entirely outside
            the available data.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> useplot = plots.formatDateXAxis(curax, object_settings)
        """

        useplot = True
        # A twin (secondary/top) axis can have its own xlims2 setting;
        # otherwise it shares the primary axis's xlims.
        if twin:
            if 'xlims2' in object_settings.keys():
                xlims_flag = 'xlims2'
            else:
                WF.print2stdout('Using Same Xlims for top and bottom.', debug=self.Report.debug)
                xlims_flag = 'xlims'
        else:
            xlims_flag = 'xlims'

        if xlims_flag in object_settings.keys():
            xlims = object_settings[xlims_flag]#should be min max flags in here

            if 'min' in xlims.keys():
                xmin = xlims['min']
                if '-' in self.Report.years_str: #multiyear plots use 2008-2019 format
                    if isinstance(xmin, str): #if this gets replaced it will only be a str
                        if self.Report.years_str in xmin: #check for the offender
                            # A multi-year plot's xmin defaulted to the
                            # full year-range string (e.g. "2008-2019")
                            # rather than a single year; replace it with
                            # just the report's actual start year.
                            xmin = xmin.replace(self.Report.years_str, str(self.Report.startYear))

            else:
                # no minimum given, default to the report's overall start time
                xmin = self.Report.StartTime

            if 'max' in xlims.keys():
                xmax = xlims['max']
                if '-' in self.Report.years_str: #multiyear plots use 2008-2019 format
                    if isinstance(xmax, str): #if this gets replaced it will only be a str
                        if self.Report.years_str in xmax: #check for the offender
                            # same year-range-string correction as xmin above, but for the end year
                            xmax = xmax.replace(self.Report.years_str, str(self.Report.endYear))
            else:
                # no maximum given, default to the report's overall end time
                xmax = self.Report.EndTime

            # Read back the axis's CURRENT x-limits (already set by
            # whatever plotted data came before this call) as real
            # datetimes, so the new limits can be clamped to not exceed
            # them.
            current_xlims = curax.get_xlim()
            current_xlims = [n.replace(tzinfo=None) for n in mpl.dates.num2date(current_xlims)]

            if current_xlims[0] < self.Report.StartTime:
                # current axis extends earlier than the report allows, clamp to report start
                starttime = self.Report.StartTime
            else:
                starttime = current_xlims[0]

            if current_xlims[1] > self.Report.EndTime:
                # current axis extends later than the report allows, clamp to report end
                endtime = self.Report.EndTime
            else:
                endtime = current_xlims[1]
            # resolve xmin into an actual datetime, using the clamped bounds as fallback/validation range
            xmin = WT.translateDateFormat(xmin, 'datetime', starttime, starttime, endtime, debug=self.Report.debug)

            try:
                # If xmax parses as a plain number, it likely represents
                # an end-of-year style shorthand; adjust starttime to the
                # beginning of that year if needed.
                xmax = float(xmax)
                tmp_starttime = current_xlims[1] - dt.timedelta(seconds=1)
                if starttime < tmp_starttime: #check for end of year
                    starttime = dt.datetime(tmp_starttime.year, 1, 1, 0, 0)
            except:
                # xmax wasn't numeric, nothing special to do here
                pass
            # resolve xmax into an actual datetime, using the (possibly adjusted) bounds above
            xmax = WT.translateDateFormat(xmax, 'datetime', endtime, starttime, endtime, debug=self.Report.debug)

            # Clamp the resolved xmin/xmax to the axis's existing limits,
            # and flag the plot as unusable if the requested window falls
            # entirely outside the available data range.
            if xmax > current_xlims[1]:
                xmax = current_xlims[1]
            if xmin < current_xlims[0]:
                xmin = current_xlims[0]
            if xmin > current_xlims[1]:
                # requested window starts after all available data, nothing to plot
                useplot = False
            if xmax < current_xlims[0]:
                # requested window ends before all available data, nothing to plot
                useplot = False
            curax.set_xlim(left=xmin, right=xmax)

        else:
            # no xlims settings defined at all for this axis
            WF.print2stdout('No Xlims flag set for {0}'.format(xlims_flag), debug=self.Report.debug)
            WF.print2stdout('Not setting Xlims.', debug=self.Report.debug)

        return useplot

    def formatTickLabels(self, ticks, ticksettings):
        """
        Format tick values into display strings per the given settings.

        Parameters
        ----------
        ticks : array_like
            Existing tick values (numeric or ``datetime.datetime``).
        ticksettings : dict
            Settings dictionary; may contain ``'numdecimals'`` (numeric
            ticks) or ``'datetimeformat'`` (date ticks).

        Returns
        -------
        list of str
            Formatted tick label strings, one per input tick.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.formatTickLabels([1.0, 2.0, 3.0], {})
        ['1', '2', '3']
        """

        newticklabels = []
        allticksinteger = False
        # If every tick happens to be a whole number, default to 0
        # decimal places instead of the usual 2 (unless overridden).
        if np.all([float(tick).is_integer() if isinstance(tick, (int, float)) else False for tick in ticks]):
            allticksinteger=True

        # format each tick individually according to its type
        for tick in ticks:
            if isinstance(tick, (int, float)):

                if 'numdecimals' in ticksettings.keys():
                    # explicit decimal precision requested
                    numdecimals = int(ticksettings['numdecimals'])
                elif allticksinteger:
                    # every tick is a whole number, no decimals needed
                    numdecimals = 0
                else:
                    # default precision
                    numdecimals = 2

                newticklabels.append('{num:,.{digits}f}'.format(num=tick, digits=numdecimals))

            elif isinstance(tick, dt.datetime):
                if 'datetimeformat' in ticksettings.keys():
                    # explicit date format string requested
                    datetimeformat = ticksettings['datetimeformat']
                else:
                    # default date format
                    datetimeformat = '%m/%d/%Y'
                tick_str = tick.strftime(datetimeformat)
                newticklabels.append(tick_str)

            else:
                # Fallback for any other tick type: just stringify it.
                newticklabels.append(str(tick))

        return newticklabels

    def formatTimeSeriesXticks(self, curax, xtick_settings, axis_settings, dateformatflag='dateformat'):
        """
        Configure and apply date x-axis tick placement and labels.

        Parameters
        ----------
        curax : matplotlib.axes.Axes
            Current axis object.
        xtick_settings : dict
            Tick settings dictionary; may contain ``'fontsize'``,
            ``'rotation'``, ``'onmonths'``, ``'ondays'``, ``'spacing'``,
            and/or ``'datetimeformat'``.
        axis_settings : dict
            Axis-level settings dictionary, used as a fallback for
            ``'fontsize'`` and to look up the date-format flag.
        dateformatflag : str, optional
            Key in ``axis_settings`` giving the date format
            (``'datetime'`` or ``'jdate'``) (default ``'dateformat'``).

        Returns
        -------
        None
            Configures ``curax`` directly.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.formatTimeSeriesXticks(curax, xtick_settings, axis_settings)
        """

        # read back the axis's current x-limits as real datetimes
        xmin, xmax = curax.get_xlim()

        xmin = mpl.dates.num2date(xmin).replace(tzinfo=None)
        xmax = mpl.dates.num2date(xmax).replace(tzinfo=None)

        if 'fontsize' in xtick_settings.keys():
            # explicit tick-level font size
            xticksize = float(xtick_settings['fontsize'])
        elif 'fontsize' in axis_settings.keys():
            # fall back to the axis-level font size
            xticksize = float(axis_settings['fontsize'])
        else:
            # final fallback default
            xticksize = 10

        if 'rotation' in xtick_settings.keys():
            rotation = float(xtick_settings['rotation'])
        else:
            rotation = 0

        curax.tick_params(axis='x', labelsize=xticksize, rotation=rotation)

        # Tick placement priority: explicit months > explicit days >
        # explicit spacing interval; only one of these three strategies
        # is applied (whichever is defined first, in that order).
        if 'onmonths' in xtick_settings.keys():
            if isinstance(xtick_settings['onmonths'], dict):
                # A single-item XML list can parse as a dict; normalize
                # it back into a one-element list.
                xtick_settings['onmonths'] = [xtick_settings['onmonths']['month']]
            bymonthday = [1]

            if 'ondays' in xtick_settings.keys():
                if isinstance(xtick_settings['ondays'], dict):
                    # same single-item-dict normalization as above, for days this time
                    xtick_settings['ondays'] = [xtick_settings['ondays']['day']]
                bymonthday = [int(n) for n in xtick_settings['ondays']]

            try:
                # Months given as integers (1-12).
                locator = mpl.dates.MonthLocator([int(n) for n in xtick_settings['onmonths']], bymonthday=bymonthday)
            except ValueError:
                # Months given as names (e.g. 'jan', 'mar'): translate
                # via the shared month-name-to-number lookup instead.
                WF.print2stdout('Invalid month values. Please use integer representation of Months (aka 1, 3, 5, etc...)', debug=self.Report.debug)
                formatted_months = [self.Report.Constants.month2num[n.lower()] for n in xtick_settings['onmonths']]
                locator = mpl.dates.MonthLocator(formatted_months, bymonthday=bymonthday)

            curax.xaxis.set_major_locator(locator)

        elif 'ondays' in xtick_settings.keys():
            if isinstance(xtick_settings['ondays'], dict):
                # normalize a single-item XML dict into a one-element list
                xtick_settings['ondays'] = [xtick_settings['ondays']['day']]
            locator = mpl.dates.DayLocator([int(n) for n in xtick_settings['ondays']])
            curax.xaxis.set_major_locator(locator)

        elif 'spacing' in xtick_settings.keys():
            # A numeric spacing value means "every N days"; build the
            # explicit list of tick dates and set them directly rather
            # than using a matplotlib locator.
            xtickspacing = xtick_settings['spacing']
            try:
                # numeric spacing given, convert it into a pandas-style "ND" interval string
                xtickspacing = float(xtickspacing)
                xtickspacing = f'{xtickspacing}D'
            except ValueError:
                # not numeric, assume it's already a valid interval string
                xtickspacing = xtick_settings['spacing']

            # build the explicit regular tick series across the axis's current range
            newxticks = WT.buildTimeSeries(xmin.replace(tzinfo=None), xmax.replace(tzinfo=None), xtickspacing)

            newxticklabels = self.formatTickLabels(newxticks, xtick_settings)
            curax.set_xticks(newxticks)
            curax.set_xticklabels(newxticklabels)

        if 'datetimeformat' in xtick_settings.keys():
            if axis_settings[dateformatflag].lower() == 'datetime':
                # only apply the custom date format if the axis is actually in datetime mode
                datetimeformat = xtick_settings['datetimeformat']
                fmt = mpl.dates.DateFormatter(datetimeformat)
                curax.xaxis.set_major_formatter(fmt)

        # If the axis is configured to display Julian dates instead of
        # calendar dates, convert whatever ticks matplotlib ended up
        # placing (from the locator/spacing logic above) into jdate
        # labels rather than real dates.
        current_xticks = mpl.dates.num2date(curax.get_xticks())
        if dateformatflag in axis_settings.keys():
            if axis_settings[dateformatflag].lower() == 'jdate':
                if isinstance(current_xticks[0], dt.datetime):
                    # convert the current tick datetimes to jdate values and re-label the same tick positions
                    jdateticklabels = WT.DatetimeToJDate(current_xticks)
                    newxticklabels = self.formatTickLabels(jdateticklabels, xtick_settings)
                    curax.set_xticks(current_xticks)
                    curax.set_xticklabels(newxticklabels)

    def formatYTicks(self, ax, ax_settings, gatedata={}, gate_placement=10, axis='left'):
        """
        Format y-axis limits, ticks, and tick labels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Current axis.
        ax_settings : dict
            Settings for the axis; checked for ``'ylims'``/``'ylims2'``
            and ``'yticks'``/``'yticks2'`` depending on ``axis``.
        gatedata : dict, optional
            Gate operation data; if non-empty, forces the y-range to
            ``0`` to ``gate_placement`` regardless of ``ylims`` (default
            ``{}``).
        gate_placement : float, optional
            Fixed y-max to use when plotting gate data (default ``10``).
        axis : {'left', 'right'}, optional
            Which y-axis this is, determining which settings keys are
            used (default ``'left'``).

        Returns
        -------
        None
            Configures ``ax`` directly.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.formatYTicks(ax, ax_settings)
        """

        # Left/right y-axes use differently-named settings keys.
        if axis == 'left':
            ylimflag = 'ylims'
            yticksflag = 'yticks'
        else:
            ylimflag = 'ylims2'
            yticksflag = 'yticks2'

        # start from the axis's current auto-computed limits
        ymin, ymax = ax.get_ylim()

        if ylimflag in ax_settings.keys():
            if 'min' in ax_settings[ylimflag]:
                # explicit minimum given, override the auto-computed one
                ymin = float(ax_settings[ylimflag]['min'])

            if 'max' in ax_settings[ylimflag]:
                # explicit maximum given, override the auto-computed one
                ymax = float(ax_settings[ylimflag]['max'])

        if len(gatedata.keys()) != 0:
            # Gate plots use a fixed 0-to-gate_placement y-range
            # regardless of any configured ylims, since gate positions
            # are drawn on an arbitrary "stack" axis rather than a real
            # data range.
            ymax = gate_placement
            ymin = 0

        ax.set_ylim(bottom=ymin)
        ax.set_ylim(top=ymax)

        if yticksflag in ax_settings.keys():
            ytick_settings = ax_settings[yticksflag]
            if 'fontsize' in ytick_settings.keys():
                # explicit tick-level font size
                yticksize = float(ytick_settings['fontsize'])
            elif 'fontsize' in ax_settings.keys():
                # fall back to the axis-level font size
                yticksize = float(ax_settings['fontsize'])
            else:
                # final fallback default
                yticksize = 10
            ax.tick_params(axis='y', labelsize=yticksize)

            if 'spacing' in ytick_settings.keys():
                ytickspacing = ytick_settings['spacing']

                if float(ytickspacing).is_integer():
                    # whole-number spacing, keep as an int for cleaner tick values
                    ytickspacing = int(ytickspacing)
                else:
                    ytickspacing = float(ytickspacing)

                # Round the tick range outward to whole numbers before
                # generating evenly-spaced ticks, but only for whichever
                # bound (min/max) wasn't explicitly set by the user.
                if 'ylims' not in ax_settings.keys():
                    # no explicit limits at all, round both bounds outward
                    ymax = int(np.ceil(ymax))
                    ymin = int(np.floor(ymin))
                else:
                    if 'min' not in ax_settings[ylimflag].keys():
                        # only min wasn't explicitly given, round it outward
                        ymin = int(np.floor(ymin))
                    if 'max' not in ax_settings[ylimflag].keys():
                        # only max wasn't explicitly given, round it outward
                        ymax = int(np.ceil(ymax))

                # build the evenly-spaced tick array across the (possibly rounded) range
                newyticks = np.arange(ymin, (ymax+ytickspacing), ytickspacing)
                newyticklabels = self.formatTickLabels(newyticks, ytick_settings)
                ax.set_yticks(newyticks)
                ax.set_yticklabels(newyticklabels)

                # Re-clamp the axis limits to exactly match the generated
                # tick range, so there's no extra padding beyond the last
                # tick.
                ax.set_ylim(bottom=min(newyticks))
                ax.set_ylim(top=max(newyticks))
                return

        # No explicit spacing given: just reformat whatever ticks
        # matplotlib auto-generated, without changing their positions.
        newyticklabels = self.formatTickLabels(ax.get_yticks(), {})
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(newyticklabels)
        ax.set_ylim(bottom=ymin)
        ax.set_ylim(top=ymax)

    def formatXTicks(self, ax, ax_settings, axis='bottom'):
        """
        Format x-axis limits, ticks, and tick labels for a non-date axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Current axis.
        ax_settings : dict
            Settings for the axis; checked for ``'xlims'``/``'xlims2'``
            and ``'xticks'``/``'xticks2'`` depending on ``axis``.
        axis : {'bottom', 'top'}, optional
            Which x-axis this is, determining which settings keys are
            used (default ``'bottom'``).

        Returns
        -------
        None
            Configures ``ax`` directly.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.formatXTicks(ax, ax_settings)
        """

        # Bottom/top x-axes use differently-named settings keys (this is
        # for non-date/numeric x-axes, e.g. profile plots; see
        # formatTimeSeriesXticks for the date-axis equivalent).
        if axis == 'bottom':
            xlimflag = 'xlims'
            xticksflag = 'xticks'
        else:
            xlimflag = 'xlims2'
            xticksflag = 'xticks2'

        # start from the axis's current auto-computed limits
        xmin, xmax = ax.get_xlim()
        if 'xlims' in ax_settings.keys():
            if 'min' in ax_settings[xlimflag]:
                # explicit minimum given, override the auto-computed one
                xmin = float(ax_settings[xlimflag]['min'])
            if 'max' in ax_settings[xlimflag]:
                # explicit maximum given, override the auto-computed one
                xmax = float(ax_settings[xlimflag]['max'])

        if xticksflag in ax_settings.keys():
            xtick_settings = ax_settings[xticksflag]
            if 'fontsize' in xtick_settings.keys():
                # explicit tick-level font size
                xticksize = float(xtick_settings['fontsize'])
            elif 'fontsize' in ax_settings.keys():
                # fall back to the axis-level font size
                xticksize = float(ax_settings['fontsize'])
            else:
                # final fallback default
                xticksize = 10
            ax.tick_params(axis='x', labelsize=xticksize)

            if 'spacing' in xtick_settings.keys():
                xtickspacing = xtick_settings['spacing']
                if float(xtickspacing).is_integer():
                    # whole-number spacing, keep as an int for cleaner tick values
                    xtickspacing = int(xtickspacing)
                else:
                    xtickspacing = float(xtickspacing)
                # build the evenly-spaced tick array across the axis's range
                newxticks = np.arange(xmin, (xmax+xtickspacing), xtickspacing)
                newxticklabels = self.formatTickLabels(newxticks, xtick_settings)
                ax.set_xticks(newxticks)
                ax.set_xticklabels(newxticklabels)

        ax.set_xlim(left=xmin)
        ax.set_xlim(right=xmax)

    def plotHorizontalLines(self, straightlines, ax, object_settings, timestamp_index=0):
        """
        Draw configured horizontal reference lines on a plot.

        Parameters
        ----------
        straightlines : dict
            Dictionary potentially containing an ``'hlines'`` field
            (per-line resolved values/settings).
        ax : matplotlib.axes.Axes
            Current axis.
        object_settings : dict
            Settings dictionary for the entire plot; used for unit
            system and depth/elevation convention resolution.
        timestamp_index : int, optional
            Index into each hline's ``'values'`` array to use, for
            time-series-based hlines (default ``0``).

        Returns
        -------
        None
            Draws directly onto ``ax``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.plotHorizontalLines(straightlines, ax, object_settings)
        """

        if 'hlines' in straightlines.keys():
            hlines = straightlines['hlines']
            # disambiguate labels shared across multiple hlines
            hlines = WF.correctDuplicateLabels(hlines)
            for key in hlines.keys():
                hline_settings = hlines[key]
                value = hline_settings['values'][timestamp_index]
                units = hline_settings['units']
                if 'parameter' in hline_settings.keys():
                    # If the plot's y-axis convention (depth vs.
                    # elevation) doesn't match the hline's own parameter,
                    # convert the value to match so it lands at the
                    # correct vertical position.
                    if object_settings['usedepth'].lower() == 'true':
                        if hline_settings['parameter'].lower() == 'elevation':
                            # plot uses depth, but this hline is defined in elevation, convert it
                            valueconv = self.Report.Profiles.convertElevationsToDepths({'depths': [],
                                                                                       'elevations': [value]},
                                                                                       {},
                                                                                       timestamp_index=timestamp_index)
                            if len(valueconv['hline']['depths']) == 0:
                                # conversion failed, mark the value as unusable
                                WF.print2stdout('Unable to convert horizontal line elevations to depths.', debug=self.Report.debug)
                                value = np.nan
                            else:
                                value = valueconv['hline']['depths'][0]
                    elif object_settings['usedepth'].lower() == 'false':
                        if hline_settings['parameter'].lower() == 'depth':
                            # plot uses elevation, but this hline is defined in depth, convert it
                            valueconv = self.Report.Profiles.convertDepthsToElevations({'depths': [value],
                                                                                        'elevations': []},
                                                                                        {},
                                                                                        timestamp_index=timestamp_index)
                            if len(valueconv['hline']['depths']) == 0:
                                # conversion failed, mark the value as unusable
                                WF.print2stdout('Unable to convert horizontal line depths to elevations.', debug=self.Report.debug)
                                value = np.nan
                            else:
                                value = valueconv['hline']['elevations'][0]

                #currently cant convert these units..
                if units != None:
                    # Prefer a y-axis-specific unit system if defined,
                    # otherwise fall back to the plot's general one.
                    if 'y_unitsystem' in object_settings.keys():
                        unitsystem = object_settings['y_unitsystem']
                    else:
                        unitsystem = object_settings['unitsystem']
                    valueconv, units = WF.convertUnitSystem(value, units, unitsystem, debug=self.Report.debug)
                    value = valueconv

                ### instead, use scalar to be manual
                if 'scalar' in hline_settings.keys():
                    # Manual multiplier for cases where automatic unit
                    # conversion isn't applicable/sufficient.
                    value *= float(hline_settings['scalar'])

                # fill in default styling, then resolve a distinct color to avoid duplicates
                hline_settings = WD.getDefaultStraightLineSettings(hline_settings, self.Report.debug)
                hline_settings = WF.fixDuplicateColors(hline_settings) #used the line, used param, then double up so subtract 1

                if 'label' not in hline_settings.keys():
                    # no label given, don't show one in the legend
                    hline_settings['label'] = None
                if 'zorder' not in hline_settings.keys():
                    # default drawing order if not specified
                    hline_settings['zorder'] = 3

                # draw the horizontal reference line at the resolved value
                ax.axhline(value, label=hline_settings['label'], c=hline_settings['linecolor'],
                           lw=hline_settings['linewidth'], ls=hline_settings['linestylepattern'],
                           zorder=float(hline_settings['zorder']),
                           alpha=float(hline_settings['alpha']))

    def plotVerticalLines(self, straightlines, ax, object_settings, timestamp_index=0, isdate=True):
        """
        Draw configured vertical reference lines on a plot.

        Parameters
        ----------
        straightlines : dict
            Dictionary potentially containing a ``'vlines'`` field
            (per-line resolved values/settings).
        ax : matplotlib.axes.Axes
            Current axis.
        object_settings : dict
            Settings dictionary for the entire plot; used for unit
            system resolution.
        timestamp_index : int, optional
            Index into each vline's ``'values'`` array to use, for
            time-series-based vlines (default ``0``).
        isdate : bool, optional
            If ``True``, the resolved value is treated as a date and
            translated to a datetime before plotting (default ``True``).

        Returns
        -------
        None
            Draws directly onto ``ax``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.plotVerticalLines(straightlines, ax, object_settings)
        """

        if 'vlines' in straightlines.keys():
            vlines = straightlines['vlines']
            # disambiguate labels shared across multiple vlines
            vlines = WF.correctDuplicateLabels(vlines)
            for key in vlines.keys():
                vline_settings = vlines[key]
                value = vline_settings['values'][timestamp_index]
                units = vline_settings['units']

                if isdate:
                    # Vertical lines on a time-series plot use a date
                    # value on the x-axis; resolve it to an actual
                    # datetime before plotting.
                    value = WT.translateDateFormat(value, 'datetime', '',
                                                   self.Report.StartTime, self.Report.EndTime,
                                                   debug=self.Report.debug)

                if 'label' not in vline_settings.keys():
                    # no label given, don't show one in the legend
                    vline_settings['label'] = None
                if 'zorder' not in vline_settings.keys():
                    # default drawing order if not specified
                    vline_settings['zorder'] = 3

                if units != None:
                    # convert the value into the plot's overall unit system
                    valueconv, units = WF.convertUnitSystem(value, units, object_settings['unitsystem'], debug=self.Report.debug)
                    value = valueconv

                # fill in default styling, then resolve a distinct color to avoid duplicates
                vline_settings = WD.getDefaultStraightLineSettings(vline_settings, self.Report.debug)
                vline_settings = WF.fixDuplicateColors(vline_settings) #used the line, used param, then double up so subtract 1

                # draw the vertical reference line at the resolved value
                ax.axvline(value, label=vline_settings['label'], c=vline_settings['linecolor'],
                           lw=vline_settings['linewidth'], ls=vline_settings['linestylepattern'],
                           zorder=float(vline_settings['zorder']),
                           alpha=float(vline_settings['alpha']))

    def fixEmptyYAxis(self, ax, ax2, keepblankax, keepblankax2):
        """
        Hide y-axis ticks on an axis that has no plotted data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Left axis object.
        ax2 : matplotlib.axes.Axes
            Right axis object.
        keepblankax : str
            ``'true'``/``'false'`` string; if ``'true'``, keeps ``ax``'s
            ticks even if it has no data.
        keepblankax2 : str
            ``'true'``/``'false'`` string; if ``'true'``, keeps ``ax2``'s
            ticks even if it has no data.

        Returns
        -------
        None
            Modifies ``ax``/``ax2`` directly.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.fixEmptyYAxis(ax, ax2, 'false', 'false')
        """

        # An axis with no plotted lines/legend entries is considered
        # "empty"; hide its ticks (unless explicitly told to keep them)
        # so an unused axis doesn't clutter the plot with a meaningless
        # tick scale.
        ax_lines, _ = ax.get_legend_handles_labels()
        ax2_lines, _ = ax2.get_legend_handles_labels()
        if len(ax_lines) == 0 and keepblankax.lower() != 'true':
            # left axis has nothing plotted and isn't flagged to be kept, hide its ticks
            ax.set_yticks([])
            ax.set_yticklabels([])
        if len(ax2_lines) == 0 and keepblankax2.lower() != 'true':
            # right axis has nothing plotted and isn't flagged to be kept, hide its ticks
            ax2.set_yticks([])
            ax2.set_yticklabels([])

    def setInitialXlims(self, ax, year):
        """
        Set the initial x-axis date range for a plot before further tweaks.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis object.
        year : int, str, or 'ALLYEARS'
            The plot's current year, a "YYYY-YYYY" range string, or
            ``'ALLYEARS'`` for the full report period.

        Returns
        -------
        None
            Sets ``ax``'s x-limits directly.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.setInitialXlims(ax, 2020)
        """

        if year == 'ALLYEARS':
            # use the report's entire simulation period
            xmin = self.Report.StartTime
            xmax = self.Report.EndTime
        else:
            # A single year or "YYYY-YYYY" range: build the calendar
            # bounds for that period (Jan 1 of the start year through
            # Jan 1 of the year after the end year).
            if isinstance(year, str):
                # multi-year range string, parse both bounds out
                yrsplit = year.split('-')
                tmpmin = dt.datetime(int(yrsplit[0]), 1, 1, 0, 0)
                tmpmax = dt.datetime(int(yrsplit[1])+1, 1, 1, 0, 0)
            else:
                # single year given
                tmpmin = dt.datetime(year, 1, 1, 0, 0)
                tmpmax = dt.datetime(year+1, 1, 1, 0, 0)
            # Clamp to the report's overall data range so a requested
            # year doesn't extend past the actual simulation period.
            if tmpmin < self.Report.StartTime:
                xmin = self.Report.StartTime
            else:
                xmin = tmpmin
            if tmpmax > self.Report.EndTime:
                xmax = self.Report.EndTime
            else:
                xmax = tmpmax

        ax.set_xlim(left=xmin, right=xmax)

    def copyYTicks(self, ax, ax2, units, ax_settings):
        """
        Mirror one axis's y-ticks onto a second axis in another unit system.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Source axis to copy ticks from.
        ax2 : matplotlib.axes.Axes
            Destination axis to duplicate ticks to.
        units : str
            Units of the ticks on ``ax``.
        ax_settings : dict
            Settings for the current axis; checked for
            ``'unitsystem2'`` to determine the target unit system for
            ``ax2``.

        Returns
        -------
        None
            Configures ``ax2`` directly.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> plots.copyYTicks(ax, ax2, 'c', ax_settings)
        """

        # pull the current limits, tick positions, and tick labels from the source axis
        axylims = ax.get_ylim()
        axyticks = ax.get_yticks()
        axyticklabels = ax.get_yticklabels()

        if 'unitsystem2' in ax_settings.keys():
            # Convert the tick VALUES (limits and positions) to the
            # secondary axis's unit system so the two axes stay aligned
            # (same physical positions) while displaying different unit
            # labels.
            axylims, _ = WF.convertUnitSystem(axylims, units, ax_settings['unitsystem2'], debug=self.Report.debug)
            axyticks, _ = WF.convertUnitSystem(axyticks, units, ax_settings['unitsystem2'], debug=self.Report.debug)
            axyticklabels = self.formatTickLabels(axyticks, {})

        # apply the (possibly converted) limits/ticks/labels onto the secondary axis
        ax2.set_ylim(axylims)
        ax2.set_yticks(axyticks)
        ax2.set_yticklabels(axyticklabels)


def translateLineStylePatterns(LineSettings):
    """
    Translate Java-style line style pattern names to matplotlib values.

    Parameters
    ----------
    LineSettings : dict
        Settings dictionary describing how the line is drawn; checked
        for and updates the ``'linestylepattern'`` key.

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
    >>> translateLineStylePatterns({'linestylepattern': 'dash'})
    {'linestylepattern': 'dashed'}
    """

    #java|python
    # Maps the Java/WAT-side line style names to matplotlib linestyle
    # values. 'dash dot-dot' has no built-in matplotlib name, so it uses
    # an explicit on/off dash-pattern tuple instead.
    linestylesdict = {'dash': 'dashed',
                      'dash dot': 'dashdot',
                      'dash dot-dot': (0, (3, 5, 1, 5, 1, 5)), #this one doesnt get a string name?
                      'dot': 'dotted',
                      'solid': 'solid'}

    if 'linestylepattern' in LineSettings.keys():
        if LineSettings['linestylepattern'].lower() in linestylesdict.values(): #existing python values
            # Already a valid matplotlib name (e.g. someone typed
            # 'dashed' directly); just normalize casing.
            LineSettings['linestylepattern'] = LineSettings['linestylepattern'].lower() #use python but lower it
        else:
            try:
                # translate the java-style name to its matplotlib equivalent
                LineSettings['linestylepattern'] = linestylesdict[LineSettings['linestylepattern'].lower()]
            except KeyError:
                # unrecognized pattern name, fall back to solid
                WF.print2stdout('Invalid lineStylePattern:', LineSettings['linestylepattern'])
                WF.print2stdout('Defaulting to Solid.')
                LineSettings['linestylepattern'] = 'solid'
    else:
        # no pattern specified at all, default to solid
        WF.print2stdout('lineStylePattern undefined for line. Using solid')
        LineSettings['linestylepattern'] = 'solid'

    return LineSettings


def translatePointStylePatterns(LineSettings):
    """
    Translate Java-style point/marker symbol codes to matplotlib values.

    Parameters
    ----------
    LineSettings : dict
        Settings dictionary describing how the line/points are drawn;
        checked for and updates the ``'symboltype'`` key.

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
    >>> translatePointStylePatterns({'symboltype': 2})
    {'symboltype': 'o'}
    """

    #java|python
    #https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    # Maps the Java/WAT-side numeric symbol codes to matplotlib marker
    # characters.
    pointstylesdict = {1: 's', #square
                       2: 'o', #circle
                       3: '^', #triangle up
                       4: 'v', #triangle down
                       5: 'D', #diamond
                       6: '*' #star
                       }

    if 'symboltype' in LineSettings.keys():
        if LineSettings['symboltype'] in pointstylesdict.values(): #existing python values
            # Already a valid matplotlib marker character; keep as-is
            # (must preserve case, e.g. 'D' vs 'd' are different markers).
            LineSettings['symboltype'] = LineSettings['symboltype'] #needs to be case sensitive..
        else:
            try:
                # translate the java-style numeric code to its matplotlib marker character
                LineSettings['symboltype'] = pointstylesdict[int(LineSettings['symboltype'])]
            except:
                # unrecognized/invalid symbol code, fall back to a square marker
                WF.print2stdout('Invalid Symboltype:', LineSettings['symboltype'])
                WF.print2stdout('Defaulting to Square.')
                LineSettings['symboltype'] = 's'

    else:
        # no symbol type specified at all, default to a square marker
        WF.print2stdout('Symbol not defined. Defaulting to Square.')
        LineSettings['symboltype'] = 's'

    return LineSettings