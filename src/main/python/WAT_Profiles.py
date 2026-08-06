import numpy as np
from functools import reduce
from scipy import interpolate

import WAT_Functions as WF
import WAT_Reader as WR
import WAT_Time as WT


class Profiles(object):
    """
    Helpers for building, converting, and validating vertical profile data.

    Vertical profiles (e.g. temperature vs. depth/elevation at one or
    more timestamps) need several kinds of processing that time series
    data doesn't: figuring out WHICH timestamps to sample, converting
    between depth and elevation conventions (which requires knowing the
    water surface elevation), unit conversion for 2-D profile arrays,
    filtering/trimming profiles to a valid range, and running data
    validity checks with warnings. This class centralizes all of that
    profile-specific logic used by the report generator's profile plots
    and tables.

    Attributes
    ----------
    Report : object
        The main Report Generator instance this profile helper serves.
    """

    def __init__(self, Report):
        """
        Initialize the profile helper class.

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
        >>> profiles = Profiles(Report)
        """
        # keep a reference back to the parent report for shared state
        self.Report = Report

    def getProfileDates(self, Line_info, StartTime, EndTime):
        """
        Get the available timestamps from an observed text profile.

        Parameters
        ----------
        Line_info : dict
            Line settings dictionary; must include a ``'filename'`` key
            pointing at the observed profile text file.
        StartTime : datetime.datetime
            Start of the window to read dates within.
        EndTime : datetime.datetime
            End of the window to read dates within.

        Returns
        -------
        list
            List of available timestamps, or an empty list if
            ``Line_info`` doesn't describe an observed file source.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        Marked with a ``#TODO: set up for not observed data??`` comment
        in the original source.

        Examples
        --------
        >>> profiles.getProfileDates(Line_info, StartTime, EndTime)
        """

        if 'filename' in Line_info.keys(): #Get data from Observed
            # observed text profile: read the available dates directly from the file
            times = WR.getTextProfileDates(Line_info['filename'], StartTime, EndTime) #TODO: set up for not observed data??
            return times

        # not an observed-file source, nothing to return
        WF.print2stdout('Illegal Dates selection.', debug=self.Report.debug)
        return []

    def getProfileInterpResolution(self, object_settings, default=30):
        """
        Determine the interpolation resolution to use for profile stats.

        Parameters
        ----------
        object_settings : dict
            Currently selected object settings dictionary; checked for
            ``'resolution'`` and ``'interpolationsource'`` keys.
        default : int, optional
            Fallback resolution to use if not defined in user settings
            (default ``30``).

        Returns
        -------
        int or str
            The number of interpolation steps to use, or the name of a
            line flag (e.g. ``'Observed'``) to use as the resolution
            reference instead of a fixed count.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> profiles.getProfileInterpResolution(object_settings)
        30
        """

        # collect the flags of every line defined for this object
        keys = [n['flag'] for n in object_settings[object_settings['datakey']]]
        if 'resolution' in object_settings.keys():
            # Explicit numeric resolution (interpolation step count) was
            # given; use it directly.
            resolution = object_settings['resolution']
            return int(resolution)
        elif 'interpolationsource' in object_settings.keys():
            # A named data source (e.g. 'Observed') was given as the
            # resolution reference instead of a fixed number; use it if
            # it's actually one of this object's line flags.
            resolution = object_settings['interpolationsource']
            if resolution in keys:
                return resolution
            else:
                # requested source isn't a valid line flag, fall back to the numeric default
                WF.print2stdout(f'InterpolationSource {resolution} not found in keys. Setting to default value '
                                f'resolution: {default}', debug=self.Report.debug)
                resolution = default
                return int(resolution)
        else:
            # No resolution setting at all: prefer the 'Observed' line as
            # the natural reference resolution if one exists, otherwise
            # fall back to the numeric default.
            if 'Observed' in keys:
                return 'Observed'
            else:
                WF.print2stdout(f'Resolution not defined. Setting to default value resolution: {default}', debug=self.Report.debug)
                resolution = default
                return int(resolution)

    def getProfileTimestamps(self, object_settings, StartTime, EndTime):
        """
        Determine the timestamps to sample profile data at.

        Reads (or builds) the list of profile timestamps based on the
        object's ``datessource_flag`` setting: a single line reference
        (str), an explicit date/date-block (dict), an explicit list of
        dates (list), or (as a fallback) evenly-spaced regular
        timesteps if none of the above yield any timestamps.

        Parameters
        ----------
        object_settings : dict
            Currently selected object settings dictionary; must contain
            ``'datessource_flag'`` and (depending on its type)
            ``'datessource'``/``'datakey'``.
        StartTime : datetime.datetime
            Start of the report time window.
        EndTime : datetime.datetime
            End of the report time window.

        Returns
        -------
        numpy.ndarray
            Array of timestamp values to be plotted/tabulated.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> profiles.getProfileTimestamps(object_settings, StartTime, EndTime)
        """

        if isinstance(object_settings['datessource_flag'], str):
            # A single line flag was given as the date source: pull
            # dates straight from that line's own observed data file.
            timestamps = []
            for line in object_settings[object_settings['datakey']]:
                if line['flag'] == object_settings['datessource_flag']:
                    timestamps = self.getProfileDates(line, StartTime, EndTime)
        elif isinstance(object_settings['datessource_flag'], dict): #single date instance..
            # A single explicit date (or date-block dict) was given
            # instead of a line reference.
            timestamps = []
            if 'dates' in object_settings['datessource'].keys():
                datekey = 'dates'
            elif 'date' in object_settings['datessource'].keys():
                datekey = 'date'
            tstamp_dates = object_settings['datessource'][datekey]
            # resolve each configured date string into an actual datetime
            for d in tstamp_dates:
                dfrmt = WT.translateDateFormat(d, 'datetime', None, StartTime, EndTime, None, debug=self.Report.debug)
                if dfrmt != None:
                    timestamps.append(dfrmt)
                else:
                    WF.print2stdout('Invalid Timestamp', d, debug=self.Report.debug)

        elif isinstance(object_settings['datessource_flag'], list): #single date instance..
            # An explicit list of dates was given directly.
            timestamps = []
            tstamp_dates = object_settings['datessource']
            # resolve each configured date string into an actual datetime
            for d in tstamp_dates:
                dfrmt = WT.translateDateFormat(d, 'datetime', None, StartTime, EndTime, None, debug=self.Report.debug)
                if dfrmt != None:
                    timestamps.append(dfrmt)
                else:
                    WF.print2stdout('Invalid Timestamp', d, debug=self.Report.debug)

        if len(timestamps) == 0:
            #if something fails, or not implemented, or theres just no dates in the window, make some up
            timestamps = WT.makeRegularTimesteps(StartTime, EndTime, self.Report.debug, days=15)

        return np.asarray(timestamps)

    def getProfileTimestampYearMonthIndex(self, object_settings, years):
        """
        Group profile timestamp indices by year and month.

        Parameters
        ----------
        object_settings : dict
            Settings dictionary for the current object; must contain a
            ``'timestamps'`` array.
        years : list of int
            Years to build the grouping for.

        Returns
        -------
        list
            Nested list indexed as ``[year_index][month_index]``, each
            entry a list of indices into ``object_settings['timestamps']``
            falling in that year/month.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> profiles.getProfileTimestampYearMonthIndex(object_settings, [2020])
        """

        # Build a nested [year][month] structure of timestamp indices, so
        # callers can quickly look up "which timestamps fall in March
        # 2015" for grouping/statistics purposes.
        timestamp_indexes = []
        for year in years:
            year_idx = []
            # scan every month for this year, collecting matching timestamp indices
            for mon in range(1,13):
                mon_idx = []
                for ti, timestamp in enumerate(object_settings['timestamps']):
                    if timestamp.year == year and timestamp.month == mon:
                        mon_idx.append(ti)
                year_idx.append(mon_idx)
            timestamp_indexes.append(year_idx)
        return timestamp_indexes

    def convertDepthsToElevations(self, data, wse_data, timestamp_index=None):
        """
        Convert observed-profile depths into elevations using WSE data.

        Parameters
        ----------
        data : dict
            Dictionary of line data keyed by flag; each entry missing
            ``'elevations'`` (i.e. an empty list) has its depths
            converted in place.
        wse_data : dict
            Water-surface-elevation time series data, keyed by
            ``'<flag>_wse'``, used as the reference elevation for the
            depth-to-elevation conversion.
        timestamp_index : int, optional
            If given, only convert the single specified timestamp index
            rather than every timestamp in the line's data.

        Returns
        -------
        dict
            The ``data`` dictionary, with elevations filled in for any
            line that was missing them.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data = profiles.convertDepthsToElevations(data, wse_data)
        """

        for ld in data.keys():
            # found_elevs = False
            if data[ld]['elevations'] == []:
                # This line has only depths, no elevations: convert its
                # depths to elevations using the matching WSE series.
                noelev_flag = ld
                wse_data_key = ld + '_wse'

                converted_elevations = []

                # Find a fallback line to pull a water-surface elevation
                # from if this line has no dedicated WSE series of its
                # own; prefer 'Computed' if present since model output
                # is generally the most reliable WSE source.
                otherkey = None
                if 'Computed' in data.keys():
                    if len(data['Computed']['elevations']) > 0:
                        otherkey = 'Computed'
                else:
                    # no 'Computed' line, search for any other line that has elevations
                    for key in data.keys():
                        if len(data[key]['elevations']) > 0:
                            otherkey = key
                            break

                use_index = False
                if timestamp_index != None:
                    # Caller wants just one specific timestamp
                    # (e.g. for a single hline value) rather than the
                    # whole profile's time range.
                    use_index = True
                    timesteps = [timestamp_index]  # use the index directly
                else:
                    timesteps = data[ld]['times']

                # convert each timestep's depths to elevations one at a time
                for tsi, ts in enumerate(timesteps):
                    if wse_data_key in wse_data.keys():
                        if use_index:
                            try:
                                # single explicit timestamp index requested
                                wse_at_timestep = wse_data[wse_data_key]['elevations'][ts]
                                e = self.convertObsDepths2Elevations(data[noelev_flag]['depths'][tsi],
                                                                     wse_at_timestep)
                            except IndexError:
                                # requested index out of range, fall back to an all-NaN result
                                e = np.full_like(data[noelev_flag]['depths'], fill_value=np.nan)
                        else:
                            # find the closest matching WSE reading for this profile timestamp
                            wse_at_timestep = self.matchProfileTimestamps(ts, wse_data[wse_data_key], onflag='elevations')['elevations']
                            e = self.convertObsDepths2Elevations(data[noelev_flag]['depths'][tsi],
                                                                 wse_at_timestep)
                    else:
                        # no dedicated WSE series exists for this line at all
                        e = np.full_like(data[noelev_flag]['depths'][tsi], fill_value=np.nan)
                    if np.all(np.isnan(e)): #if theyre all nan
                        # This line's own WSE didn't produce a usable
                        # elevation conversion; fall back to the max WSE
                        # from the other line's elevation data instead.
                        if otherkey != None:
                            try:
                                selected_elevation_data = data[otherkey]['elevations'][tsi]
                                maxelev = WF.getMaxWSEFromElev(selected_elevation_data)
                                e = self.convertObsDepths2Elevations(data[noelev_flag]['depths'][tsi],
                                                                     maxelev)
                            except IndexError:
                                # fallback line doesn't have data at this timestep either
                                e = np.full_like(data[noelev_flag]['depths'][tsi], fill_value=np.nan)
                        else:
                            # no fallback line available at all
                            e = np.full_like(data[noelev_flag]['depths'][tsi], fill_value=np.nan)

                    converted_elevations.append(e)
                data[noelev_flag]['elevations'] = converted_elevations
        return data

    def convertElevationsToDepths(self, data, wse_data, timestamp_index=None):
        """
        Convert observed-profile elevations into depths using WSE data.

        Mirror image of ``convertDepthsToElevations``.

        Parameters
        ----------
        data : dict
            Dictionary of line data keyed by flag; each entry missing
            ``'depths'`` (i.e. an empty list) has its elevations
            converted in place.
        wse_data : dict
            Water-surface-elevation time series data, keyed by
            ``'<flag>_wse'``, used as the reference elevation for the
            elevation-to-depth conversion.
        timestamp_index : int, optional
            If given, only convert the single specified timestamp index
            rather than every timestamp in the line's data.

        Returns
        -------
        dict
            The ``data`` dictionary, with depths filled in for any line
            that was missing them.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        In the fallback search for ``otherkey`` (used when no
        ``'Computed'`` line is present), the condition
        ``np.isnan(data[key]['depths']) > 0`` is unlikely to behave as a
        meaningful truthiness check on a possibly empty list/array. The
        equivalent branch in ``convertDepthsToElevations`` instead uses
        ``len(data[key]['elevations']) > 0``, which appears to be the
        intended check. This matches the source file as written and has
        not been changed here, per the "no logic changes" scope of this
        documentation pass.

        Examples
        --------
        >>> data = profiles.convertElevationsToDepths(data, wse_data)
        """

        for ld in data.keys():
            if data[ld]['depths'] == []:
                # Mirror image of convertDepthsToElevations above: this
                # line has only elevations, no depths, so convert
                # elevations to depths using the matching WSE series.
                nodepth_flag = ld
                wse_data_key = ld + '_wse'

                converted_depths = []

                otherkey = None
                if 'Computed' in data.keys():
                    if len(data['Computed']['depths']) > 0:
                        otherkey = 'Computed'
                else:
                    # NOTE: `np.isnan(data[key]['depths'])` on a possibly
                    # empty list/array is unlikely to behave as a
                    # meaningful truthiness check here; the equivalent
                    # branch above (in convertDepthsToElevations) instead
                    # uses `len(data[key]['elevations']) > 0`, which looks
                    # like the intended check. This matches the source
                    # file as written; not changed here per the "no logic
                    # changes" scope of this documentation pass.
                    for key in data.keys():
                        if not np.isnan(data[key]['depths']) > 0:
                            otherkey = key
                            break

                use_index = False
                if timestamp_index != None:
                    use_index = True
                    timesteps = [timestamp_index] #use the index directly
                else:
                    timesteps = data[ld]['times']

                # convert each timestep's elevations to depths one at a time
                for tsi, ts in enumerate(timesteps):
                    if wse_data_key in wse_data.keys():
                        if use_index:
                            try:
                                # single explicit timestamp index requested
                                wse_at_timestep = wse_data[wse_data_key]['elevations'][ts]
                                d = self.convertObsElevations2Depths(data[nodepth_flag]['elevations'][tsi],
                                                                     wse_at_timestep)
                            except IndexError:
                                # requested index out of range, fall back to an all-NaN result
                                d = np.full_like(data[nodepth_flag]['elevations'], fill_value=np.nan)
                        else:
                            # find the closest matching WSE reading for this profile timestamp
                            wse_at_timestep = self.matchProfileTimestamps(ts, wse_data[wse_data_key], onflag='elevations')['elevations']
                            d = self.convertObsElevations2Depths(data[nodepth_flag]['elevations'][tsi],
                                                                 wse_at_timestep)
                    else:
                        # no dedicated WSE series exists for this line at all
                        d = np.full_like(data[nodepth_flag]['elevations'], fill_value=np.nan)
                    if np.all(np.isnan(d)): #if theyre all nan
                        # fall back to the other line's WSE if this line's own conversion failed
                        if otherkey != None:
                            try:
                                selected_elevation_data = data[otherkey]['elevations'][tsi]
                                maxelev = WF.getMaxWSEFromElev(selected_elevation_data)
                                d = self.convertObsElevations2Depths(data[nodepth_flag]['elevations'][tsi],
                                                                     maxelev)
                            except IndexError:
                                # fallback line doesn't have data at this timestep either
                                d = np.full_like(data[nodepth_flag]['elevations'], fill_value=np.nan)
                        else:
                            # no fallback line available at all
                            d = np.full_like(data[nodepth_flag]['elevations'], fill_value=np.nan)

                    converted_depths.append(d)
                data[nodepth_flag]['depths'] = converted_depths
        return data

    def convertObsDepths2Elevations(self, input_depths, max_wse):
        """
        Convert a single timestep's observed depths to elevations.

        Parameters
        ----------
        input_depths : array_like
            Depths for observed data at a single timestep.
        max_wse : float or array_like
            Reference water surface elevation (or array of elevations)
            at the timestep, used as the surface to measure depths down
            from.

        Returns
        -------
        numpy.ndarray
            Observed elevations, or an all-NaN array if ``max_wse`` is
            missing/invalid.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> profiles.convertObsDepths2Elevations([1.0, 2.0], 100.0)
        array([99., 98.])
        """

        obs_elev = []
        # Bail out with an all-NaN result if the WSE reference itself is
        # missing/invalid; a depth measurement is meaningless without a
        # known water surface elevation to measure down from.
        if isinstance(max_wse, (list, np.ndarray)):
            if len(max_wse) == 0:
                # no WSE values at all
                return np.full(len(input_depths), np.nan)
            if np.all(np.isnan(max_wse)):
                # every WSE value is NaN
                return np.full(len(input_depths), np.nan)
        elif np.isnan(max_wse):
            # single scalar WSE value, but it's NaN
            return np.full(len(input_depths), np.nan)
        # elevation = water surface elevation - depth below surface
        for depth in input_depths:
            obs_elev.append(max_wse - depth)
        return np.asarray(obs_elev)

    def convertObsElevations2Depths(self, input_elevs, max_wse):
        """
        Convert observed elevations to depths below the water surface.

        Parameters
        ----------
        input_elevs : array_like
            Nested elevation arrays (one sub-array per timestep) for
            observed data.
        max_wse : array_like
            Reference water surface elevation(s) used as the surface to
            measure depths down from.

        Returns
        -------
        list
            List of depth arrays (one per timestep), or a single all-NaN
            array if ``max_wse`` is empty.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> profiles.convertObsElevations2Depths([[99.0, 98.0]], 100.0)
        [array([1., 2.])]
        """

        out_depth = []
        if len(max_wse) == 0:
            # no WSE values at all, can't compute any depth
            out_depth.append(np.full(len(input_elevs), np.nan)) #make nan boys
        else:
            # depth below surface = water surface elevation - elevation
            for i, e in enumerate(input_elevs):
                d = []
                for elev in e:
                    d.append(max_wse - elev)
                out_depth.append(np.asarray(d))
        return out_depth

    def convertProfileDataUnits(self, object_settings, data, line_settings):
        """
        Convert profile value and y-axis units per the object's unit systems.

        Parameters
        ----------
        object_settings : dict
            Settings dictionary for the current object; checked for
            ``'unitsystem'`` (x/value axis) and ``'y_unitsystem'``
            (depth/elevation axis).
        data : dict
            Profile data keyed by line flag, with ``'values'``,
            ``'depths'``, and ``'elevations'`` arrays to convert.
        line_settings : dict
            Per-line settings dictionary; each entry's ``'units'`` and
            ``'y_units'`` are updated to the converted unit strings.

        Returns
        -------
        data : dict
            The ``data`` dictionary with converted values.
        line_settings : dict
            The ``line_settings`` dictionary with updated unit strings.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, line_settings = profiles.convertProfileDataUnits(object_settings, data, line_settings)
        """

        # X-axis (value) unit conversion, e.g. temperature C -> F.
        if 'unitsystem' not in object_settings.keys():
            WF.print2stdout('Unit system (unitsystem) not defined.', debug=self.Report.debug)
            # return data, line_settings
        else:
            # convert every line's value profiles to the requested unit system
            for flag in data.keys():
                if line_settings[flag]['units'] == None:
                    # no units known for this line, nothing to convert
                    continue
                else:
                    profiles = data[flag]['values']
                    profileunits = line_settings[flag]['units']
                    # convert each timestep's profile individually
                    for pi, profile in enumerate(profiles):
                        profile, newunits = WF.convertUnitSystem(profile, profileunits, object_settings['unitsystem'])
                        profiles[pi] = profile
                    line_settings[flag]['units'] = newunits
        # Y-axis (depth/elevation) unit conversion, e.g. meters -> feet;
        # applied to both depths and elevations arrays since a profile
        # may carry either or both.
        if 'y_unitsystem' not in object_settings.keys():
            WF.print2stdout('Y Unit system (y_unitsystem) not defined.', debug=self.Report.debug)
        else:
            # convert every line's depth/elevation profiles to the requested unit system
            for flag in data.keys():
                if line_settings[flag]['y_units'] == None:
                    # no y-axis units known for this line, nothing to convert
                    continue
                else:
                    yunits = line_settings[flag]['y_units']
                    # convert each timestep's depth profile individually
                    for pi, profile in enumerate(data[flag]['depths']):
                        profile, newunits = WF.convertUnitSystem(profile, yunits, object_settings['y_unitsystem'])
                        data[flag]['depths'][pi] = profile
                    # convert each timestep's elevation profile individually
                    for pi, profile in enumerate(data[flag]['elevations']):
                        profile, newunits = WF.convertUnitSystem(profile, yunits, object_settings['y_unitsystem'])
                        data[flag]['elevations'][pi] = profile
                    line_settings[flag]['y_units'] = newunits
        return data, line_settings

    def filterProfileData(self, data, line_settings, object_settings):
        """
        Filter profile data points by configured x/y limits and omit values.

        Parameters
        ----------
        data : dict
            Profile data keyed by line flag, with ``'values'``,
            ``'depths'``/``'elevations'`` arrays to filter in place.
        line_settings : dict
            Per-line settings dictionary; checked for ``'xlims'``,
            ``'ylims'``, ``'filterbylimits'``, and
            ``'omitvalue'``/``'omitvalues'`` overrides.
        object_settings : dict
            Object-level settings dictionary providing defaults for the
            same limit/filter keys, plus ``'usedepth'`` to determine
            which y-axis convention (depth or elevation) is active.

        Returns
        -------
        data : dict
            The filtered ``data`` dictionary.
        object_settings : dict
            The (unmodified) ``object_settings`` dictionary, returned
            for convenience.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, object_settings = profiles.filterProfileData(data, line_settings, object_settings)
        """

        xmax = None
        xmin = None
        ymax = None
        ymin = None

        # Determine which y-axis convention this object uses (depth or
        # elevation) so filtering is applied to the right array; the
        # other convention's array (if present) gets filtered in lockstep
        # below so the two stay aligned.
        if 'usedepth' in object_settings.keys():
            if object_settings['usedepth'].lower() == 'true':
                yflag = 'depths'
                other_yflag = 'elevations'
            else:
                yflag = 'elevations'
                other_yflag = 'depths'
        else:
            # can't filter y-values without knowing which convention is active
            WF.print2stdout('UseDepth flag not set. Cannot filter properly.', debug=self.Report.debug)
            return data, object_settings

        # Object-level default x/y limits (may be overridden per-line
        # below).
        if 'xlims' in object_settings.keys():
            if 'max' in object_settings['xlims'].keys():
                xmax = float(object_settings['xlims']['max'])
            if 'min' in object_settings['xlims'].keys():
                xmin = float(object_settings['xlims']['min'])

        if 'ylims' in object_settings.keys():
            if 'max' in object_settings['ylims'].keys():
                ymax = float(object_settings['ylims']['max'])
            if 'min' in object_settings['ylims'].keys():
                ymin = float(object_settings['ylims']['min'])

        # Find Index of ALL acceptable values.
        for lineflag in data.keys():
            cur_data = data[lineflag]
            cur_line_settings = line_settings[lineflag]

            # Per-line limits override the object-level defaults if set.
            current_xmax = xmax
            current_xmin = xmin
            current_ymax = ymax
            current_ymin = ymin
            if 'xlims' in cur_line_settings.keys():
                if 'max' in cur_line_settings['xlims'].keys():
                    current_xmax = float(cur_line_settings['xlims']['max'])
                if 'min' in cur_line_settings['xlims'].keys():
                    current_xmin = float(cur_line_settings['xlims']['min'])
            if 'ylims' in cur_line_settings.keys():
                if 'max' in cur_line_settings['ylims'].keys():
                    current_ymax = float(cur_line_settings['ylims']['max'])
                if 'min' in cur_line_settings['ylims'].keys():
                    current_ymin = float(cur_line_settings['ylims']['min'])

            # filterbylimits gates whether the x/y limits above actually
            # trim the DATA (vs. just controlling the plotted view range
            # elsewhere); per-line setting wins over the object-level one.
            filtbylims = False
            if 'filterbylimits' in cur_line_settings.keys():
                if cur_line_settings['filterbylimits'].lower() == 'true':
                    filtbylims = True
            else:
                if 'filterbylimits' in object_settings.keys():
                    if object_settings['filterbylimits'].lower() == 'true':
                        filtbylims = True

            if 'omitvalue' in cur_line_settings.keys():
                # single sentinel value to omit
                omitvalues = [float(cur_line_settings['omitvalue'])]
            elif 'omitvalues' in cur_line_settings.keys():
                # multiple sentinel values to omit
                omitvalues = [float(n) for n in cur_line_settings['omitvalues']]
            else:
                # no sentinel values configured for this line
                omitvalues = None

            # apply the filters to every timestep's profile for this line
            for pi, profile in enumerate(cur_data['values']):
                ydata = cur_data[yflag][pi]

                # Build index masks for each active filter criterion,
                # defaulting to "keep everything" (full index range) for
                # any criterion that isn't active for this line.
                if current_xmax != None and filtbylims:
                    xmax_filt = np.where(profile <= current_xmax)
                else:
                    xmax_filt = np.arange(len(profile))

                if current_xmin != None and filtbylims:
                    xmin_filt = np.where(profile >= current_xmin)
                else:
                    xmin_filt = np.arange(len(profile))

                if current_ymax != None and filtbylims:
                    ymax_filt = np.where(ydata <= current_ymax)
                else:
                    ymax_filt = np.arange(len(ydata))

                if current_ymin != None and filtbylims:
                    ymin_filt = np.where(ydata >= current_ymin)
                else:
                    ymin_filt = np.arange(len(ydata))

                if omitvalues != None:
                    # build a running mask excluding every sentinel value in turn
                    omitvals_filt = []
                    for omitval in omitvalues:
                        omitval_filt = np.where(profile != omitval)
                        omitvals_filt = np.append(omitvals_filt, omitval_filt)
                else:
                    omitvals_filt = np.arange(len(profile))

                # Intersect every mask together so only indices passing
                # ALL active filters survive.
                master_filter = reduce(np.intersect1d, (xmax_filt, xmin_filt, ymax_filt, ymin_filt, omitvals_filt)).astype(int)

                # apply the combined mask to the values array
                data[lineflag]['values'][pi] = profile[master_filter]
                try:
                    if len(cur_data[other_yflag][pi]) == len(cur_data[yflag][pi]):
                        #if there isnt enough data, dont filter it the same. They need to be the same.
                        # apply the same mask to the "other" y-axis array so the two stay aligned
                        data[lineflag][other_yflag][pi] = cur_data[other_yflag][pi][master_filter]
                except IndexError:
                    if len(cur_data[other_yflag]) != len(cur_data[yflag]):
                        # entirely missing "other" y-axis data for this timestep
                        WF.print2stdout(f'Cannot filter {other_yflag} due to no values', debug=self.Report.debug)
                    else:
                        # mismatched lengths between the two y-axis arrays for this timestep
                        WF.print2stdout(f'Cannot filter {other_yflag} due to different number of values compared to '
                                        f'{yflag}. {len(cur_data[other_yflag][pi])}: {len(cur_data[yflag][pi])}',
                                        debug=self.Report.debug)
                # apply the combined mask to the primary y-axis array as well
                data[lineflag][yflag][pi] = ydata[master_filter]

        return data, object_settings

    def stackProfileIndicies(self, exist_data, new_data):
        """
        Merge a new data array into an existing one for stacked contours.

        Used for contour plots of several reaches split into different
        groups, stacking them together so they function as a single
        reach.

        Parameters
        ----------
        exist_data : dict
            Existing accumulated data, keyed by run flag then item flag.
        new_data : dict
            New data to merge into ``exist_data``, in the same nested
            structure.

        Returns
        -------
        dict
            The updated ``exist_data`` dictionary.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> exist_data = profiles.stackProfileIndicies(exist_data, new_data)
        """

        for runflag in new_data.keys():
            if runflag not in exist_data.keys():
                # first time seeing this run at all, initialize an empty entry for it
                exist_data[runflag] = {}
            for itemflag in new_data[runflag]:
                if itemflag not in exist_data[runflag].keys():
                    # First time seeing this item for this run: just copy
                    # it over directly.
                    exist_data[runflag][itemflag] = new_data[runflag][itemflag]
                else:
                    # Already have data for this item: append/concatenate
                    # rather than overwrite, using the appropriate method
                    # for the container type.
                    if isinstance(new_data[runflag][itemflag], list):
                        exist_data[runflag][itemflag] += new_data[runflag][itemflag]
                    elif isinstance(new_data[runflag][itemflag], np.ndarray):
                        exist_data[runflag][itemflag] = np.append(exist_data[runflag][itemflag], new_data[runflag][itemflag])
        return exist_data

    def normalize2DElevations(self, vals, elevations):
        """
        Interpolate W2 reservoir profile data onto a common elevation grid.

        Parameters
        ----------
        vals : array_like
            List of value arrays, one per timestamp, each aligned with
            the corresponding row in ``elevations``.
        elevations : numpy.ndarray
            2-D array of elevation values (timestamp x elevation-index),
            which may differ from row to row.

        Returns
        -------
        newvals : numpy.ndarray
            Values interpolated onto the shared elevation grid.
        new_elevations : numpy.ndarray
            The common elevation grid used for interpolation, spanning
            the overall min/max elevation across all timestamps.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> newvals, new_elevations = profiles.normalize2DElevations(vals, elevations)
        """

        newvals = []
        # W2 reservoir profiles can have a different elevation grid at
        # each timestamp; build one common elevation grid spanning the
        # overall min/max elevation seen across all timestamps, then
        # interpolate every timestamp's values onto that shared grid so
        # they can be plotted/compared consistently.
        top_elev = np.nanmax([np.nanmax(n) for n in elevations if ~np.all(np.isnan(n))])
        bottom_elev = np.nanmin([np.nanmin(n) for n in elevations if ~np.all(np.isnan(n))])
        # build the shared elevation grid, same number of points as the original data's width
        new_elevations = np.linspace(bottom_elev, top_elev, elevations.shape[1])
        # interpolate every timestamp's values onto the shared grid one at a time
        for vi, v in enumerate(vals):
            valelev_interp = interpolate.interp1d(elevations[vi], v, bounds_error=False, fill_value ="extrapolate")
            newvals.append(valelev_interp(new_elevations))
        return np.asarray(newvals), np.asarray(new_elevations)

    def matchProfileTimestamps(self, input_timestamps, timeseries_dict, onflag='values'):
        """
        Pull time series values that align with a set of profile dates.

        Parameters
        ----------
        input_timestamps : list
            Profile timestep dates to match against.
        timeseries_dict : dict
            Time series data dictionary, must contain a ``'dates'`` key
            plus the array named by ``onflag``.
        onflag : str, optional
            The key in ``timeseries_dict`` holding the values to select
            (default ``'values'``).

        Returns
        -------
        dict
            New dictionary with ``onflag`` and ``'dates'`` trimmed to
            the matched timestamps, plus any other keys from
            ``timeseries_dict`` copied through unchanged.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> output = profiles.matchProfileTimestamps(input_timestamps, timeseries_dict)
        """

        output = {}
        # Find the closest matching time series index for each requested
        # profile timestamp, then pull out just those values/dates.
        timestamp_idx = WR.getClosestTime(input_timestamps, timeseries_dict['dates'])
        output[onflag] = timeseries_dict[onflag][timestamp_idx]
        output['dates'] = timeseries_dict['dates'][timestamp_idx]
        # Carry over any other metadata keys unchanged (e.g. units).
        for key in timeseries_dict.keys():
            if key not in output.keys():
                output[key] = timeseries_dict[key]
        return output

    def checkProfileValidity(self, data, object_settings, combineyears=False, includeallyears=False):
        """
        Run data-quality checks on profile data and collect warnings.

        Only runs when debug logging is enabled (this is a diagnostic
        aid, not part of normal report generation). For every profile
        (each dataset x timestep), checks for: non-monotonic depth/
        elevation values, negative values, duplicate values, too few
        points, and "clustering" (an unusually large share of points
        bunched up near the surface/top of the profile, which often
        indicates a data quality issue). Detected issues are logged and
        collected into a per-dataset, per-year warnings dictionary.

        Parameters
        ----------
        data : dict
            Profile data keyed by dataset flag, each with ``'times'``
            and either/both ``'depths'``/``'elevations'`` (whichever is
            populated is used for the checks).
        object_settings : dict
            Object settings dictionary; a ``'warnings'`` sub-dictionary
            is created/updated in place.
        combineyears : bool, optional
            If ``True`` (or if ``object_settings['splitbyyear']`` is
            ``'false'``), warnings across all years are combined into a
            single ``'ALLYEARS'`` entry instead of being kept per-year.
        includeallyears : bool, optional
            If ``True`` (or if ``object_settings['includeallyears']`` is
            ``'true'``), an additional ``'ALLYEARS'`` combined entry is
            added alongside the per-year entries.

        Returns
        -------
        dict
            ``object_settings['warnings']``: a dict keyed by dataset
            flag, each mapping year (or ``'ALLYEARS'``) to a list of
            unique warning strings.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> warnings = profiles.checkProfileValidity(data, object_settings)
        """

        if not self.Report.debug:
            # this is purely a diagnostic aid, skip entirely outside debug mode
            return {}
        if 'warnings' not in object_settings.keys():
            object_settings['warnings'] = {}
        # thresholds controlling the clustering/insufficiency checks below
        range_percent_threshold = 1 #percent of the range to use for clustering detection
        percent_vals_under_threshold = 25 #percent of values in threshold for clustering detection
        minimum_number_values = 5 #min amount of points

        if 'splitbyyear' in object_settings.keys():
            if object_settings['splitbyyear'].lower() == 'false':
                # not splitting by year, warnings should be combined into ALLYEARS
                combineyears = True
        if 'includeallyears' in object_settings.keys():
            if object_settings['includeallyears'].lower() == 'true':
                # also want a combined ALLYEARS entry alongside per-year ones
                includeallyears = True

        # run the validity checks dataset by dataset
        for di, d in enumerate(data.keys()):
            WF.print2stdout(f'Assessing Dataset: {d}', debug=self.Report.debug)
            if d not in object_settings['warnings'].keys():
                object_settings['warnings'][d] = {}
            # Prefer depths over elevations for the checks below (easier
            # to detect negative/invalid values on a depth scale, which
            # should always be non-negative).
            usedepth = False
            yflag = None
            if 'depths' in data[d].keys():
                if len(data[d]['depths']) > 0: #try and use depths if possible, easier to detect negative
                    usedepth = True
                    yflag = 'depths'
            if not usedepth:
                if 'elevations' in data[d].keys():
                    if len(data[d]['elevations']) > 0:
                        usedepth = False
                        yflag = 'elevations'

            if yflag == None:
                # neither depths nor elevations available, nothing to check
                WF.print2stdout('No values for dataset.', debug=self.Report.debug)
                continue

            yvalues = data[d][yflag]

            # check each timestep's profile individually
            for yvsi, yvalset in enumerate(yvalues):
                if len(yvalset) > 0:
                    yearflag = data[d]['times'][yvsi].year
                    if yearflag not in object_settings['warnings'][d].keys():
                        object_settings['warnings'][d][yearflag] = []
                    datarange = max(yvalset) - min(yvalset)
                    number_of_points = len(yvalset)
                    monotonic = []
                    has_duplicates = False
                    has_negative = False
                    enough_points = True

                    # Determine whether this profile is expected to be
                    # increasing or decreasing (based on first vs. last
                    # value), so monotonicity can be checked in the right
                    # direction.
                    if yvalset[0] > yvalset[-1]:
                        increasing = False
                    else:
                        increasing = True

                    if len(yvalset) > len(list(set(yvalset))):
                        # fewer unique values than total values means duplicates exist
                        has_duplicates = True

                    if len(yvalset) < minimum_number_values:
                        # too few points to trust the profile
                        enough_points = False

                    # Walk the profile point-by-point checking for
                    # negative values and whether each step continues in
                    # the expected (increasing/decreasing) direction.
                    for yvi, yval in enumerate(yvalset):
                        if yvi == 0:
                            # nothing to compare the very first value against
                            continue
                        else:
                            if yval < 0:
                                has_negative = True
                            if increasing:
                                if yval >= yvalset[yvi-1]:
                                    monotonic.append(True)
                                else:
                                    monotonic.append(False)
                            else:
                                if yval <= yvalset[yvi-1]:
                                    monotonic.append(True)
                                else:
                                    monotonic.append(False)

                    # Clustering check: flag profiles where an unusually
                    # large share of points fall within a narrow band
                    # near the top of the profile (near-surface for
                    # depths, near-max for elevations), which can
                    # indicate the profile didn't actually sample much of
                    # the water column.
                    if usedepth: #clustering when close to 0
                        top_yval = min(yvalset)
                        #add the percent threshold for depth, 0 and going downwards by increasing values
                        threshold_datarange = top_yval + (datarange * (range_percent_threshold / 100))
                        threshold_number_vals = len(np.where(yvalset < threshold_datarange)[0])
                    else: #clustering when close to the max
                        top_yval = max(yvalset)
                        #subtract the percent threshold for depth, 0 and going downwards by decreasing values
                        threshold_datarange = top_yval - (datarange * (range_percent_threshold / 100))
                        threshold_number_vals = len(np.where(yvalset > threshold_datarange)[0])

                    WF.print2stdout(f"\nProfile Date: {data[d]['times'][yvsi]}", debug=self.Report.debug)
                    WF.print2stdout(f'Number under {range_percent_threshold}% ({round(threshold_datarange, 2)}): {threshold_number_vals}/{number_of_points} '
                                    f'({round((threshold_number_vals / number_of_points) * 100, 2)}%)', debug=self.Report.debug)

                    # Record each detected issue as a warning string;
                    # note the clustering check is skipped entirely when
                    # there weren't enough points to trust it.
                    if not np.all(monotonic):
                        WF.print2stdout(f'Profile non-monotonic.', debug=self.Report.debug)
                        object_settings['warnings'][d][yearflag].append('non-monotonic values')
                    if has_negative:
                        WF.print2stdout(f'Profile contains negative {yflag}', debug=self.Report.debug)
                        object_settings['warnings'][d][yearflag].append('negative values')
                    if has_duplicates:
                        WF.print2stdout(f'Profile contains duplicate {yflag}', debug=self.Report.debug)
                        object_settings['warnings'][d][yearflag].append('duplicate values')
                    if not enough_points:
                        WF.print2stdout(f'Profile contains insufficient {yflag} points', debug=self.Report.debug)
                        object_settings['warnings'][d][yearflag].append('insufficient values')
                    else:
                        # only check for clustering if there were enough points to trust the check
                        if (threshold_number_vals / number_of_points) * 100 > percent_vals_under_threshold:
                            WF.print2stdout(f'Profile may contain top clustering.', debug=self.Report.debug)
                            object_settings['warnings'][d][yearflag].append('clustering')
                else:
                    # no values at all for this timestep, nothing to check
                    WF.print2stdout(f'No values for {d} for {data[d]["times"][yvsi]}', debug=self.Report.debug)
                    continue
            if combineyears or includeallyears:
                # Flatten every year's warnings into one combined list,
                # either replacing the per-year breakdown entirely
                # (combineyears) or adding it as an extra 'ALLYEARS'
                # entry alongside the per-year ones (includeallyears).
                allwarnings = []
                for yearflag in object_settings['warnings'][d].keys():
                    for warningflag in object_settings['warnings'][d][yearflag]:
                        allwarnings.append(warningflag)
                if combineyears: #only output all years
                    object_settings['warnings'][d] = {'ALLYEARS': allwarnings}
                elif includeallyears:
                    object_settings['warnings'][d]['ALLYEARS'] = allwarnings
            # De-duplicate each year's warning list (the same issue type
            # may have been appended multiple times across timesteps).
            for yearkey in object_settings['warnings'][d].keys():
                object_settings['warnings'][d][yearkey] = list(set(object_settings['warnings'][d][yearkey]))

        return object_settings['warnings']

    def writeWarnings(self, warnings, year):
        """
        Write formatted profile-validity warnings into the report as text boxes.

        Parameters
        ----------
        warnings : dict
            Warnings dictionary as produced by ``checkProfileValidity``,
            keyed by dataset flag then year.
        year : int or str
            The year (or ``'ALLYEARS'``) to write warnings for.

        Returns
        -------
        None
            Adds a text box to the report for each dataset with
            warnings for ``year``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> profiles.writeWarnings(warnings, 2020)
        """
        # write one text box per dataset that has any warnings for this year
        for key in warnings.keys():
            if len(warnings[key][year]) > 0:
                message = self.formatProfileWarningMessages(warnings[key][year], key)
                self.Report.makeTextBox({'text': message})

    def formatProfileWarningMessages(self, warnings, key):
        """
        Format a list of profile issues into a readable warning sentence.

        Parameters
        ----------
        warnings : list of str
            List of issue descriptions (e.g. ``'negative values'``).
        key : str
            Name of the dataset the warnings apply to.

        Returns
        -------
        str
            A grammatically-formatted warning message, e.g. "Some
            profiles in X may be invalid due to A, B, and C."

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> profiles.formatProfileWarningMessages(['negative values', 'duplicate values'], 'Observed')
        'Some profiles in Observed may be invalid due to negative values and duplicate values'
        """

        # Build a grammatically-correct comma-separated list of issues
        # (e.g. "X, Y, and Z." for 3+, "X and Y" for exactly 2, "X." for
        # just 1).
        message = f'Some profiles in {key} may be invalid due to'
        if len(warnings) > 2:
            # 3+ issues, use comma-separated list with "and" before the last one
            for wi, warn in enumerate(warnings):
                if wi == len(warnings) - 1:
                    message += f' and {warn}.'
                else:
                    message += f' {warn},'
        elif len(warnings) == 2:
            # exactly 2 issues, join with "and" and no commas
            message += f' {warnings[0]} and {warnings[1]}'
        else:
            # exactly 1 issue
            message += f' {warnings[0]}.'
        return message

    def confirmValidDepths(self, data):
        """
        Determine whether profile tables should use depths or elevations.

        Parameters
        ----------
        data : dict
            Profile data keyed by dataset flag, each with ``'depths'``
            and ``'elevations'`` arrays.

        Returns
        -------
        str
            ``'True'`` if depths can be used for every profile in every
            dataset, ``'False'`` if any profile is missing depth data
            (in which case elevations should be used instead, for
            consistency).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> profiles.confirmValidDepths(data)
        'True'
        """

        # Only use depths for the whole table if EVERY profile in every
        # dataset actually has depth data; if any profile is missing
        # depths (but has elevations), fall back to elevations for
        # everything so the table stays consistent.
        usedepths = 'True' #innocent until proven guilty
        for key in data.keys():
            numDepthProfiles = len(data[key]['depths'])
            numElevProfiles = len(data[key]['elevations'])
            if numDepthProfiles == 0 and numElevProfiles > 0:
                # this dataset has elevations but no depths at all
                WF.print2stdout(f'Not using depths for {key}.', debug=self.Report.debug)
                usedepths = 'False'
            # also check individual timesteps within the dataset for missing depths
            for dpi, depthprofile in enumerate(data[key]['depths']):
                if len(depthprofile) == 0 and len(data[key]['elevations'][dpi]) > 0:
                    WF.print2stdout(f'Not using depths for {key}.', debug=self.Report.debug)
                    usedepths = 'False'
        return usedepths

    def snapTo0Depth(self, data, line_settings):
        """
        Add a synthetic surface (depth 0) point to profiles that lack one.

        Parameters
        ----------
        data : dict
            Profile data keyed by line flag, with ``'depths'`` and
            ``'elevations'`` arrays; updated in place for lines with
            ``'snapto0depth'`` set to ``'true'``.
        line_settings : dict
            Per-line settings dictionary; checked for the
            ``'snapto0depth'`` flag.

        Returns
        -------
        dict
            The updated ``data`` dictionary.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data = profiles.snapTo0Depth(data, line_settings)
        """

        # Some profiles don't include a measurement exactly at the water
        # surface (depth 0); when 'snapto0depth' is requested, add a
        # synthetic surface point using the shallowest available
        # measurement's elevation as the implied water surface, so
        # profile plots visually extend up to the surface.
        for key in data.keys():
            if 'snapto0depth' in line_settings[key].keys():
                if line_settings[key]['snapto0depth'].lower() == 'true':
                    # process each timestep's depth profile individually
                    for dsi, depthset in enumerate(data[key]['depths']):
                        if len(depthset) == 0:
                            # nothing to snap for an empty profile
                            continue
                        if 0.0 not in depthset:
                            # find the shallowest existing measurement to use as the reference point
                            distance_from_wse = min(depthset)
                            min_depth_i = np.where(depthset == distance_from_wse)
                            # infer the implied water surface elevation from that shallowest point
                            max_elevation = max(data[key]['elevations'][dsi])
                            wse_elevation = max_elevation + distance_from_wse
                            # overwrite the shallowest point with the synthetic surface point
                            data[key]['elevations'][dsi][min_depth_i] = wse_elevation
                            data[key]['depths'][dsi][min_depth_i] = 0.0

        return data