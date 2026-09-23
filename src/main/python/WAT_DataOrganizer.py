# Standard library imports for filesystem paths, array math, error
# reporting, object caching (pickle), and grouping/counting helpers.
import os
import numpy as np
import pandas as pd
import traceback
import pickle
import itertools
from collections import Counter

import WAT_Functions as WF				# WAT_Functions holds general-purpose helpers (logging, name sanitizing, data matching, etc.) used throughout this module.
import WAT_Reader as WDR				# WAT_Reader (aliased twice, as WDR and WR, by the original code) provides low-level readers for DSS, text-profile, and formatted-table files.
import WAT_Time as WT
import WAT_Reader as WR
import WAT_ResSim_Results as WRSS		# WAT_ResSim_Results is used here specifically to read data out of externally supplied HEC-ResSim H5 result files.
import WAT_Constants as WC

# Single shared WAT_Constants instance (units, conversions, etc.) used for
# unit lookups when a data source doesn't otherwise supply its own units.
constants = WC.WAT_Constants()


class DataOrganizer(object):
    """
    Reads report input data from disk/DSS/H5 sources and organizes it.

    This class is the central data-access layer for the report generator.
    It reads time series, vertical profiles, contours, tables, and gate
    operation data from whatever source each line/object is configured to
    use (DSS records, CE-QUAL-W2 output files, ResSim H5 files, text
    profiles, formatted tables, etc.), and caches everything it reads in
    an in-memory dictionary (``self.Memory``) keyed by a deterministic
    "memory key" built from each data source's settings. This avoids
    re-reading the same source file/record multiple times when several
    plots or tables reference it.

    Attributes
    ----------
    Report : object
        The main Report Generator instance this data organizer serves.
    Memory : dict
        In-memory cache of everything read from disk/DSS/H5, keyed by the
        string returned from ``buildMemoryKey()``.
    """

    def __init__(self, Report):
        """
        Initialize the data organizer and its memory cache.

        Parameters
        ----------
        Report : object
            The main Report Generator instance (``self`` from the main
            report generator script).

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
        >>> organizer = DataOrganizer(Report)
        """

        # keep a reference back to the parent report for shared state (dates, debug flag, etc.)
        self.Report = Report #self from main
        # Set up the empty data cache used by every "get*" method below.
        self.intializeMemory()

    def intializeMemory(self):
        """
        Initialize the empty in-memory data cache.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets the instance attribute ``self.Memory`` to an empty dict.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> organizer.intializeMemory()
        >>> organizer.Memory
        {}
        """

        # self.Memory caches everything read from disk/DSS/H5, keyed by
        # the string returned from buildMemoryKey(), so repeated requests
        # for the same source don't re-hit the filesystem.
        self.Memory = {}

    def buildMemoryKey(self, data_info):
        """
        Build a unique, filesystem-safe key identifying a data source.

        The returned string is used both as the ``self.Memory`` cache key
        (so repeated requests for the same source are served from memory)
        and as the base file name for the CSV data-log written by
        ``writeDataFiles``. Which fields of ``data_info`` are used to
        build the name depends on which data-source type is present
        (DSS record, CE-QUAL-W2 file, ResSim H5 file, coordinate-based
        lookup, observed profile text file, or W2/ResSim identifiers).

        Parameters
        ----------
        data_info : dict
            Settings dictionary describing a single line/data source
            (as parsed from the report XML).

        Returns
        -------
        str
            A unique name (truncated to 150 characters) for this data
            source, or ``'NULL'`` if the source type could not be
            determined.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> key = organizer.buildMemoryKey({'dss_path': '/A/B/C//1HOUR/F/',
        ...                                 'dss_filename': 'input.dss'})
        """

        # very_special_flags = f'{self.Report.SimulationName.replace(" ", "").replace(":", "")}_{self.Report.baseSimulationName.replace(" ", "").replace(":", "")}'
        # A sanitized version of the simulation name is appended to every
        # generated key so that the same data source used across
        # different simulations/comparisons doesn't collide in memory.
        very_special_flags = WF.sanitizeText(self.Report.SimulationName)

        if 'dss_path' in data_info.keys(): #Get data from DSS record
            if 'dss_filename' in data_info.keys():
                # Combine the DSS filename with selected path parts
                # (A, B, C, F path pieces) so the key uniquely identifies
                # this specific DSS record.
                # sanitize the base DSS filename (drop the .dss extension) for use in the key
                outname = WF.sanitizeText(os.path.basename(data_info['dss_filename'])[:-4])
                # split the DSS path into its slash-delimited A/B/C/.../F segments
                dssnamesplit = data_info['dss_path'].split('/')
                # pick out just the path segments that meaningfully identify this record
                dssname_pick = WF.sanitizeText(f"{dssnamesplit[1]}_{dssnamesplit[2]}_{dssnamesplit[3]}_{dssnamesplit[5]}_{dssnamesplit[-2]}")
                # combine simulation flag, filename, and path pieces into the final key
                outname = very_special_flags + '_' + outname + '_' + dssname_pick

        elif 'w2_file' in data_info.keys():
            if 'structurenumbers' in data_info.keys():
                # Structure (gate/outlet) time series: normalize the
                # structure number(s) into a list regardless of whether a
                # single dict, single string, or list/array was supplied.
                outname = WF.sanitizeText(os.path.basename(data_info['w2_file']).split('.')[0])
                if isinstance(data_info['structurenumbers'], dict):
                    # single structure given as a dict, wrap it in a one-element list
                    structure_nums = [data_info['structurenumbers']['structurenumber']]
                elif isinstance(data_info['structurenumbers'], str):
                    # single structure given as a bare string, wrap it in a one-element list
                    structure_nums = [data_info['structurenumbers']]
                elif isinstance(data_info['structurenumbers'], (list, np.ndarray)):
                    # already a list/array of structures, use as-is
                    structure_nums = data_info['structurenumbers']
                else:
                    # unrecognized type, fall back to an empty placeholder
                    structure_nums = ''
                # append the joined structure numbers and simulation flag to the base filename
                outname += '_Struct_' + '_'.join(structure_nums) + f'_{very_special_flags}'
            else:
                # Plain W2 output file column: include the column name (if
                # given) so different columns from the same file get
                # distinct keys.
                if 'column' in data_info.keys():
                    # fold the (whitespace-stripped) column name into the simulation flag portion
                    very_special_flags += f'_Colf{data_info["column"].replace(" ", "")}'
                # combine the base filename with the simulation/column flag
                outname = f"{os.path.basename(data_info['w2_file']).split('.')[0]}_{very_special_flags}"

        elif 'h5file' in data_info.keys():
            # External ResSim H5 file: key on the file name plus either
            # (easting, northing) coordinates or a ResSim result name,
            # whichever locates the requested value in that file.
            # sanitize the H5 file's base name for use in the key
            h5name = WF.sanitizeText(os.path.basename(data_info['h5file']).split('.h5')[0] + 'h5')
            if 'easting' in data_info.keys() and 'northing' in data_info.keys():
                # keyed by coordinates
                outname = 'externalh5_{0}_{1}_{2}_{3}_{4}'.format(h5name, WF.sanitizeText(data_info['parameter']), data_info['easting'], data_info['northing'], very_special_flags)
            elif 'ressimname' in data_info.keys():
                # keyed by named ResSim result instead of coordinates
                outname = 'externalh5_{0}_{1}_{2}_{3}'.format(h5name, WF.sanitizeText(data_info['parameter']), WF.sanitizeText(data_info['ressimresname']), very_special_flags)

        elif 'easting' in data_info.keys() and 'northing' in data_info.keys():
            # Coordinate-based lookup against the current model's own
            # results (not an external H5 file).
            # build the key directly from parameter name, coordinates, and simulation flag
            outname = '{0}_{1}_{2}_{3}'.format(WF.sanitizeText(data_info['parameter']), data_info['easting'], data_info['northing'], very_special_flags)

        elif 'filename' in data_info.keys(): #Get data from Observed Profile
            # observed text-file profile, keyed by its base filename
            outname = WF.sanitizeText(os.path.basename(data_info['filename']).split('.')[0].replace(' ', '_')) + f'_{very_special_flags}'

        elif 'w2_segment' in data_info.keys():
            # W2 vertical profile at a given model segment.
            # combine the current model's output filename, the segment number, and the simulation flag
            outname = 'W2_{0}_{1}_profile'.format(self.Report.ModelAlt.output_file_name.split('.')[0], data_info['w2_segment']) + f'_{very_special_flags}'

        elif 'ressimresname' in data_info.keys():
            # ResSim result-name based time series/profile, keyed on the
            # source H5 file plus parameter and result name. If a target
            # elevation/value is specified, append a short "trgt" suffix
            # so target-based series don't collide with plain ones.
            # build the base key from the model's own H5 filename, parameter, result name, and sim flag
            outname = '{0}_{1}_{2}_{3}'.format(WF.sanitizeText(os.path.basename(self.Report.ModelAlt.h5fname).split('.')[0] +'_h5'),
                                               WF.sanitizeText(data_info['parameter']), WF.sanitizeText(data_info['ressimresname']), very_special_flags)
            if 'target' in data_info.keys():
                # append target parameter/value info so this doesn't collide with a non-target key
                outname += '_trgt'
                if 'parameter' in data_info['target'].keys():
                    # tack on the first 4 characters of the target parameter name
                    outname += data_info['target']['parameter'][:4]
                if 'value' in data_info['target'].keys():
                    # tack on the target value itself
                    outname += data_info['target']['value']
        else:
            # None of the recognized data-source keys were present.
            outname = 'NULL'

        # Truncate to keep the key usable as a file name on all platforms.
        return outname[:150]

    #################################################################
    #TimeSeries Functions
    #################################################################

    def updateTimeSeriesDataDictionary(self, data, line_settings, line):
        """
        Fetch one line's time series and add it to the data dictionaries.

        Calls ``getTimeSeries`` to read/retrieve the data for ``line``,
        then (if valid data was returned) stores the values/dates under a
        unique flag in ``data`` and records the associated metadata and
        settings under the same flag in ``line_settings``. If the line's
        flag is already used, a numeric suffix is appended to make it
        unique.

        Parameters
        ----------
        data : dict
            Dictionary accumulating ``{flag: {'values':..., 'dates':...}}``
            entries for all lines processed so far.
        line_settings : dict
            Dictionary accumulating settings/metadata for each flag.
        line : dict
            Settings dictionary for the single line being processed.

        Returns
        -------
        data : dict
            Updated data dictionary (with this line added, if valid).
        line_settings : dict
            Updated line settings dictionary.
        datacheck : bool
            ``True`` if this line produced usable data and was added,
            ``False`` otherwise.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, line_settings, ok = organizer.updateTimeSeriesDataDictionary({}, {}, line)
        """

        # read the actual time series values/dates/metadata for this line
        dates, values, metadata = self.getTimeSeries(line, makecopy=False) #TODO: update
        # default to "no usable data" until proven otherwise below
        datacheck = False
        if WF.checkData(values):
            # data came back usable, proceed to register it under a flag
            datacheck = True
            flag = line['flag']
            # If this flag is already in use (e.g. plotting the same
            # variable for multiple simulations), keep appending an
            # incrementing suffix until we find a free flag name.
            if flag in line_settings.keys() or flag in data.keys():
                count = 1
                newflag = flag + '_{0}'.format(count)
                # keep incrementing the suffix until we land on an unused flag name
                while newflag in data.keys():
                    count += 1
                    newflag = flag + '_{0}'.format(count)
                WF.print2stdout(f'The current flag was {flag}', debug=self.Report.debug)
                flag = newflag
                WF.print2stdout(f'The new flag is {newflag}', debug=self.Report.debug)
            # compute the memory key so we can log which source this data came from
            datamem_key = self.buildMemoryKey(line)
            # If the source didn't report units but the line settings
            # specify them explicitly, use the user-specified units.
            if 'units' in line.keys() and metadata['units'] == None:
                metadata['units'] = line['units']

            # record which memory/log file this data corresponds to
            line_settings[flag] = {'logoutputfilename': datamem_key}

            # store the actual values/dates under this flag
            data[flag] = {'values': values,
                          'dates': dates}

            #add flags and settings to linesettings..
            # merge in the metadata read from source, then the original line settings on top
            line_settings[flag].update(metadata)
            line_settings[flag].update(line)

        return data, line_settings, datacheck

    def getProfileWSE(self, settings, onflag='lines'):
        """
        Extract water-surface-elevation (WSE) time series for lines.

        For every entry in ``settings[onflag]`` that defines a nested
        ``'wse'`` data source, reads that source as a time series and
        stores it keyed as ``<line flag>_wse`` so it can be matched back
        up with its parent line later (e.g. to filter a profile/table to
        a target elevation).

        Parameters
        ----------
        settings : dict
            Settings dictionary containing the list of line/data objects.
        onflag : str, optional
            The key in ``settings`` holding the list of data objects to
            scan for a ``'wse'`` sub-entry (default ``'lines'``, but some
            callers use a different key such as ``'datapaths'``).

        Returns
        -------
        dict
            Dictionary of ``{'<flag>_wse': {'elevations':..., 'dates':...,
            'logoutputfilename':...}}`` entries, one per line that
            defined a ``'wse'`` source.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> wse_data = organizer.getProfileWSE(settings)
        """

        # dict that will collect one entry per line that defines a nested WSE source
        wse_data = {}
        # scan every data object for a nested 'wse' data-source definition
        for dataobject in settings[onflag]:
            if 'wse' in dataobject.keys():
                # Read the WSE time series the same way any other time
                # series is read, then tag it with a '_wse' suffixed key.
                dates, values, metadata = self.getTimeSeries(dataobject['wse'], makecopy=False)
                # compute the memory key for this WSE source, for logging/tracing
                datamem_key = self.buildMemoryKey(dataobject['wse'])
                # build the '_wse' suffixed key that ties this back to its parent line
                new_key = dataobject['flag'] + '_wse'
                wse_data[new_key] = {'elevations': values,
                                      'dates': dates,
                                      'logoutputfilename': datamem_key}
                # fold in any additional metadata returned from the read
                wse_data[new_key].update(metadata)

        return wse_data

    def getMembers(self, object_settings, data_settings):
        """
        Determine which forecast ensemble members should be used.

        Priority order: (1) if we're currently iterating over a single
        forecast member, use just that member; (2) if the object's
        settings explicitly list members, use those (reformatted to the
        standard member-name convention); (3) if this is a forecast
        report with no explicit member list, use every member defined for
        the report; (4) otherwise, use the intersection of members
        present across all supplied data sets (so only members common to
        every data set are used).

        Parameters
        ----------
        object_settings : dict
            Settings dictionary for the current plot/table/profile object.
        data_settings : dict
            Dictionary of per-data-set settings, each expected to contain
            a ``'members'`` list when member intersection is needed.

        Returns
        -------
        list or numpy.ndarray
            The list (or array) of member identifiers to use.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> members = organizer.getMembers(object_settings, data_settings)
        """

        if self.Report.memberiteration: #if its a forecast iteration, grab the current iteration
            # only the single member currently being iterated over is needed
            members = [self.Report.member]
        elif 'members' in object_settings.keys(): #if user defined, use the user defined ones
            # reformat each user-supplied member name to the standard convention
            members = [WF.formatMembers(n) for n in object_settings['members']]
        elif self.Report.reportType == 'forecast': #use the forecasts defined
            # forecast report with no explicit list, use every defined member
            members = self.Report.allMembers
        else: #otherwise, we just use everything that we have (if multi datasets, get the overlapping
            # Intersect the member lists of each data set in turn so only
            # members present in ALL data sets are kept.
            members = []
            # step through each data set, narrowing the running member list to the intersection
            for i, ds in enumerate(data_settings.keys()):
                if i == 0:
                    # seed the running intersection with the first dataset's members
                    members = data_settings[ds]['members']
                else:
                    # narrow down to only members present in both the running set and this dataset
                    members = np.intersect1d(data_settings[ds]['members'], members)
        return members

    def filterTimeSeries(self, data, line_settings):
        """
        Apply target-elevation, generic value filters, and x-limits.

        For every line/flag in ``data`` this applies, in order:

        1. **Target elevation filtering** (if ``'target_elevation'`` is
           set): zeroes out flow/structure values at timesteps where the
           associated elevation time series does not equal the target
           elevation. Handles both forecast (dict-of-members) and
           single-simulation (array) value shapes, as well as CE-QUAL-W2
           structure-flow results (dict keyed by structure number).
        2. **Generic value filters** (if ``'filters'`` is set): sets
           values to ``NaN`` wherever a comparison (``under``, ``over``,
           or ``equals`` a given value) evaluates ``True``, either against
           the line's own values or against a separately defined filter
           time series.
        3. **X-axis limit trimming** (if ``'xlims'`` is set): delegates to
           ``WF.applyXLimits`` to trim the dates/values to a plotting
           window.

        Parameters
        ----------
        data : dict
            Dictionary of ``{flag: {'values':..., 'dates':...}}`` entries
            to filter in place.
        line_settings : dict
            Per-flag settings dictionary; each entry is checked for
            ``'target_elevation'``, ``'filters'``, and ``'xlims'`` keys.

        Returns
        -------
        dict
            The same ``data`` dictionary, with filters applied in place.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data = organizer.filterTimeSeries(data, line_settings)
        """

        # walk every line/flag currently in the data dictionary
        for d in data.keys():
            if 'target_elevation' in line_settings[d].keys():
                # parse out the target elevation value once for this line
                target_elevation = float(line_settings[d]['target_elevation'])
                values = data[d]['values']
                if isinstance(values, dict): #w2 results kept in dict with elev
                    if self.Report.reportType == 'forecast': #has to be from DSS
                        if 'elevation' in line_settings[d].keys():
                            # Forecast collection: values are keyed by
                            # ensemble member. For each member, pull its
                            # elevation series and zero out flow values at
                            # any timestep where elevation != target.
                            # loop over every ensemble member present in this line's values
                            for member in values.keys():
                                # pull out this member's own value array to operate on
                                member_values = values[member]
                                if 'flag' not in line_settings[d]['elevation'].keys():
                                    # configure the elevation sub-settings to read just this member
                                    line_settings[d]['elevation']['flag'] = line_settings[d]['flag'] + '_elev'
                                    # restrict the elevation read to just the current member
                                    line_settings[d]['elevation']['members'] = [member]
                                    # read this member's elevation time series
                                    elev_times, elev_values, elev_metadata = self.getTimeSeries(line_settings[d]['elevation'])
                                    # find every timestep where elevation doesn't match the target
                                    targelev_failed = np.where(elev_values[member] != target_elevation)
                                    if len(elev_times) == len(data[d]['dates']):
                                        # times line up directly, zero out failing timesteps in place
                                        member_values[targelev_failed] = 0.
                                        # write the updated member values back into the main data dict
                                        data[d]['values'][member] = member_values
                                    else:
                                        # Elevation and flow series aren't
                                        # on the same timestamps; match
                                        # them up before comparing.
                                        WF.print2stdout(f'Values and Elevations in {d} member {member} different. Equalizing.',
                                                        debug=self.Report.debug)
                                        # align the flow and elevation series onto a shared set of timestamps
                                        mainvalues, elev_data = WF.matchData(
                                                                        {'dates': data[d]['dates'], 'values': member_values},
                                                                        {'dates': elev_times, 'values': elev_values[member]})
                                        # re-run the target-elevation comparison on the now-aligned data
                                        targelev_failed = np.where(elev_data['values'] != target_elevation)
                                        mainvalues['values'][targelev_failed] = 0.
                                        # store both the zeroed values and the newly aligned dates back
                                        data[d]['values'][member] = mainvalues['values']
                                        data[d]['dates'] = mainvalues['dates']
                    else:
                        # Single-simulation W2 structure results: values
                        # is a dict keyed by structure number, each with
                        # its own 'elevcl' (elevation of centerline) and
                        # 'q(m3/s)' (flow) arrays.
                        # loop over every structure number present in this line's values
                        for sn in values.keys():
                            # find timesteps where this structure's own elevation doesn't match target
                            targelev_failed = np.where(values[sn]['elevcl'] != target_elevation)
                            # zero out the flow values at those failing timesteps
                            data[d]['values'][sn]['q(m3/s)'][targelev_failed] = 0.
                elif isinstance(values, (list, np.ndarray)):
                    # Plain single time series (not per-member/structure):
                    # same target-elevation zeroing logic as above.
                    if 'elevation' in line_settings[d].keys():
                        if 'flag' not in line_settings[d]['elevation'].keys():
                            # configure the elevation sub-settings so it can be read as its own series
                            line_settings[d]['elevation']['flag'] = line_settings[d]['flag']+'_elev'
                            # read the elevation time series associated with this line
                            elev_times, elev_values, elev_metadata = self.getTimeSeries(line_settings[d]['elevation'])
                            # find every timestep where elevation doesn't match the target
                            targelev_failed = np.where(elev_values != target_elevation)
                            if len(elev_times) == len(data[d]['values']):
                                # times line up directly, zero out failing timesteps in place
                                data[d]['values'][targelev_failed] = 0.
                            else:
                                # times don't align, match them up in time before comparing
                                WF.print2stdout(f'Values and Elevations in {d} different. Equalizing.', debug=self.Report.debug)
                                # align the main value series and elevation series onto shared timestamps
                                mainvalues, elev_data = WF.matchData({'dates': data[d]['dates'], 'values': data[d]['values']},
                                                                     {'dates': elev_times, 'values': elev_values})
                                # re-run the target-elevation comparison on the now-aligned data
                                targelev_failed = np.where(elev_data['values'] != target_elevation)
                                mainvalues['values'][targelev_failed] = 0.
                                # store both the zeroed values and the newly aligned dates back
                                data[d]['values'] = mainvalues['values']
                                data[d]['dates'] = mainvalues['dates']

            if 'filters' in line_settings[d].keys():
                # Generic value filters: each filter can compare either
                # the line's own values, or a separately defined filter
                # time series, against a threshold, and NaN-out values
                # where the comparison holds.
                # apply every filter defined for this line, one at a time
                for filter in line_settings[d]['filters']:
                    if 'value' not in filter.keys():
                        # no threshold value defined, this filter can't be applied
                        WF.print2stdout('Value not defined in filter. Not using filter.', debug=self.Report.debug)
                    else:
                        # threshold value this filter compares against
                        value = float(filter['value'])
                    # flag tracking whether this filter reads its own separate data source
                    use_filter_ts = False
                    # If the filter itself defines a data source, read that
                    # as a separate series to filter on; otherwise filter
                    # based on the line's own values.
                    if np.any([n.lower() in ['w2_file', 'dss_path', 'easting', 'h5file', 'ressimresname'] for n in filter.keys()]):
                        # filter defines its own source, read it as a separate time series
                        use_filter_ts = True
                        filter_times, filter_values, filter_metadata = self.getTimeSeries(filter)
                    if use_filter_ts:
                        # compare against the separately-read filter series
                        data_to_filter = filter_values
                    else:
                        # compare against the line's own values
                        data_to_filter = data[d]['values']

                    # build a boolean mask based on the requested comparison type
                    if 'when' in filter.keys():
                        if filter['when'].lower() == 'under':
                            # mask every point below the threshold
                            filtermask = np.where(data_to_filter < value)
                        elif filter['when'].lower() == 'over':
                            # mask every point above the threshold
                            filtermask = np.where(data_to_filter > value)
                        elif filter['when'].lower() == 'equals':
                            # mask every point exactly equal to the threshold
                            filtermask = np.where(data_to_filter == value)
                    else:
                        # no comparison type specified, default to exact equality
                        WF.print2stdout('When condition not set in filter. Assuming equals.', debug=self.Report.debug)
                        filtermask = np.where(data_to_filter == value)
                    try:
                        # apply the mask, NaNing out values that match the filter condition
                        data[d]['values'][filtermask] = np.nan
                    except IndexError:
                        # Happens if the filter series and main data
                        # series don't line up index-for-index (different
                        # length/interval); skip filtering rather than
                        # crash.
                        WF.print2stdout('Filter and Data Index not equal. Not Filtering data.', debug=self.Report.debug)
                        WF.print2stdout('Confirm that Data and Filter are on the same timeseries interval', debug=self.Report.debug)

            if 'xlims' in line_settings[d].keys():
                # Trim the series to a user-specified x-axis (date) window.
                xlims = line_settings[d]['xlims']
                # delegate to the shared x-limit trimming helper
                data[d]['dates'], data[d]['values'] = WF.applyXLimits(self.Report, data[d]['dates'], data[d]['values'], xlims)

        return data

    def getTimeSeriesDataDictionary(self, settings):
        """
        Build the data dictionary for all time series lines in an object.

        Iterates every entry in ``settings['lines']`` and reads its time
        series via ``updateTimeSeriesDataDictionary``. Two special cases
        are handled:

        - A line flagged ``'computed'`` is expanded into one line per
          accepted simulation ID (``self.Report.accepted_IDs``), so a
          single "computed" line definition produces one series per
          simulation/alternative being reported on.
        - Any other line has model-specific placeholder values resolved
          (via ``WF.replaceflaggedValues``) or, for comparison plots, has
          its settings reconfigured for the report's base simulation ID.

        Lines missing a ``'flag'`` key, or that don't match the current
        model type, are skipped.

        Parameters
        ----------
        settings : dict
            Currently selected object's settings dictionary; must contain
            a ``'lines'`` key to produce any output.

        Returns
        -------
        data : dict
            Dictionary of time series data keyed by (unique) line flag.
        line_settings : dict
            Dictionary of settings/metadata for each flag in ``data``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, line_settings = organizer.getTimeSeriesDataDictionary(settings)
        """

        # accumulator dicts built up as each line is processed
        data = {}
        line_settings = {}
        if 'lines' in settings.keys():
            # process every line defined for this object
            for line in settings['lines']:
                # tracks how many times this flag has produced valid output, used for disambiguation
                numtimesused = 0

                if self.Report.memberiteration:
                    # if 'members' not in line.keys():
                    # During a per-member forecast iteration, force this
                    # line to only use the current member being processed.
                    line['members'] = [self.Report.member]
                    line['allmembers'] = self.Report.allMembers

                if 'flag' not in line.keys():
                    # can't process a line with no identifying flag, skip it entirely
                    WF.print2stdout('Flag not set for line (Computed/Observed/etc)', debug=self.Report.debug)
                    WF.print2stdout('Not plotting Line:', line, debug=self.Report.debug)
                    continue

                elif line['flag'].lower() == 'computed':
                    # 'computed' lines represent model output and get
                    # expanded into one line per accepted simulation ID
                    # (e.g. one per alternative/scenario being compared).
                    # loop over every accepted simulation ID for this report
                    for ID in self.Report.accepted_IDs:
                        # deep-copy the line settings so each ID gets its own independent config
                        curline = pickle.loads(pickle.dumps(line, -1))
                        # reconfigure the copied line's settings for this specific simulation ID
                        curline = self.Report.configureSettingsForID(ID, curline)
                        if not self.Report.checkModelType(curline):
                            # this ID's model type doesn't match what this line needs, skip it
                            continue
                        curline['numtimesused'] = numtimesused
                        # read and register this ID's version of the line
                        data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, curline)
                        if success:
                            # bump the usage counter so the next ID (if any) disambiguates correctly
                            numtimesused += 1
                else:
                    # Non-computed lines (e.g. Observed data, straight
                    # reference lines): resolve model-specific
                    # placeholders, or reconfigure to the base simulation
                    # ID when plotting a comparison against a different
                    # currently-loaded ID.
                    if self.Report.reportType == 'forecast':
                        # forecast reports resolve model-specific placeholders directly
                        line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')

                    else:
                        if self.Report.currentlyloadedID != self.Report.base_id: #for comparison plotting mostly
                            # currently on a comparison ID, reconfigure this line to use the base ID's settings instead
                            line = self.Report.configureSettingsForID(self.Report.base_id, line)
                        else:
                            # already on the base ID, just resolve model-specific placeholders
                            line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')
                    line['numtimesused'] = numtimesused
                    if not self.Report.checkModelType(line):
                        # model type mismatch, skip this line entirely
                        continue
                    # read and register this non-computed line
                    data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, line)
                    if success:
                        numtimesused += 1

        return data, line_settings

    def checkForIdenticalMembers(self, data, metadata):
        """
        Group forecast members that have byte-for-byte identical data.

        For a collection (forecast ensemble) time series, this compares
        every member's values (ignoring NaNs) against every other
        member's, and records groups of members whose data is identical.
        This is used to avoid plotting/tabulating visually redundant
        duplicate ensemble members (e.g. when several forecast members
        happen to produce the same trace).

        Parameters
        ----------
        data : dict
            Dictionary keyed by member name, each holding that member's
            value array.
        metadata : dict
            Metadata dictionary for the collection; must contain a
            ``'members'`` list. Updated in place with an
            ``'identicalmembers'`` key.

        Returns
        -------
        dict
            The ``metadata`` dictionary, with ``metadata['identicalmembers']``
            set to a list of member-name groups whose data matched exactly.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> metadata = organizer.checkForIdenticalMembers(data, metadata)
        """

        # start with an empty list of duplicate-member groups
        metadata['identicalmembers'] = []
        # compare every member's data against every other member's, looking for exact matches
        for member in metadata['members']:
            # will collect every member found to be identical to this one
            membergroup = []
            if not any([member in n for n in metadata['identicalmembers']]): #check to see if we've already found a match
                # strip NaNs before comparing so missing-data patterns don't cause false mismatches
                main_member_data = WF.ignoreNans(data[member])
                # compare this member's data against every other member's data
                for othermember in metadata['members']: #check all other members in the list
                    if othermember != member:
                        if np.all(main_member_data == WF.ignoreNans(data[othermember])):
                            # exact match found, group these two (or more) members together
                            if member not in membergroup:
                                membergroup.append(member)
                            membergroup.append(othermember)
            if len(membergroup) > 0:
                # only record groups that actually found at least one duplicate
                metadata['identicalmembers'].append(membergroup)
        return metadata

    def checkForDuplicateObject(self, settings, member):
        """
        Check whether every line for a member is a duplicate of another.

        Used together with ``checkForIdenticalMembers`` to decide whether
        an entire plot/object for a given forecast member is redundant
        (i.e. every one of its lines has an identical twin among the
        other members), in which case the object can be skipped for that
        member.

        Parameters
        ----------
        settings : dict
            Dictionary of line settings (one entry per line in the
            object), each optionally containing an ``'identicalmembers'``
            list (as produced by ``checkForIdenticalMembers``).
        member : str
            The member identifier being checked.

        Returns
        -------
        bool
            ``True`` if every line's data for ``member`` is duplicated by
            another member (or if ``settings`` is empty -> handled as a
            special ``False`` case below); ``False`` if at least one line
            is not a duplicate for this member.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> organizer.checkForDuplicateObject(line_settings, 'member_5')
        False
        """

        # assume fully duplicate until a non-duplicate line proves otherwise
        fully_duplicate = True
        if len(settings) == 0:
            # no lines at all, nothing to consider duplicate
            return False
        # a single non-duplicate line is enough to disqualify the whole object
        for name, setting in settings.items():
            if 'identicalmembers' in setting:
                if not any([member in n for n in setting['identicalmembers']]):
                    # this line's data for this member isn't a duplicate, so the object isn't fully redundant
                    fully_duplicate = False
                    break
        return fully_duplicate

    def checkForLowestMember(self, line_settings, member, linekey=None):
        """
        Check whether ``member`` is the lowest-numbered in its duplicate
        group.

        When several members share identical data (see
        ``checkForIdenticalMembers``), only one representative member
        needs to be plotted/labeled; this designates the numerically
        lowest member in the group as that representative.

        Parameters
        ----------
        line_settings : dict
            Per-line settings dictionary containing ``'identicalmembers'``
            groupings.
        member : str or int
            The member identifier being checked.
        linekey : str, optional
            Which line's settings to use for the lookup; if not given,
            the first line's settings are used (valid because all lines
            share the same duplicate-member grouping by this point).

        Returns
        -------
        bool
            ``True`` if ``member`` is the lowest value among its
            duplicate-member group, ``False`` otherwise.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> organizer.checkForLowestMember(line_settings, 'member_1')
        True
        """
        # find the other members sharing an identical-data group with this one
        other_members = self.getOtherMembers(line_settings, member, linekey)
        # this member is the representative if it's strictly lower than all its duplicates
        if member < min(other_members):
            return True
        return False

    def getOtherMembers(self, line_settings, member, linekey=None):
        """
        Return the other members that share identical data with ``member``.

        Parameters
        ----------
        line_settings : dict
            Per-line settings dictionary containing ``'identicalmembers'``
            groupings.
        member : str or int
            The member identifier to look up.
        linekey : str, optional
            Which line's settings to use for the lookup; defaults to the
            first available line.

        Returns
        -------
        list
            The other member identifiers in ``member``'s duplicate group
            (excluding ``member`` itself); empty if no duplicate group
            contains ``member``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> organizer.getOtherMembers(line_settings, 'member_1')
        ['member_2', 'member_3']
        """
        if linekey == None:
            # all lines share the same duplicate-member grouping by this point, so any line will do
            linekey = list(line_settings.keys())[0]  # if we are here, then all the lines have the duplicate member group
        # pull the duplicate-member groupings recorded for this line
        line_members = line_settings[linekey]['identicalmembers']
        # will hold the other members found in member's duplicate group, if any
        other_members = []
        # Find the duplicate-member group (if any) that contains `member`,
        # and return every other member in that same group.
        for membergroup in line_members:
            if member in membergroup:
                # found the group containing this member, grab everyone else in it
                other_members = [n for n in membergroup if n != member]
        return other_members

    def getStraightLineValue(self, settings):
        """
        Resolve horizontal/vertical reference-line values for a plot.

        Handles two ways a straight line (``hlines``/``vlines``) can be
        defined in the settings:

        - A fixed constant ``'value'``: broadcast to a flat array the
          same length as the plot's timestamps (or a single-element list
          if no timestamps are given).
        - A data-source-based line (no ``'value'`` key): reads it as a
          time series via ``getTimeSeriesDataDictionary`` and, if the
          object has profile timestamps, picks out the value closest to
          each timestamp (``WR.getClosestTime``).

        Parameters
        ----------
        settings : dict
            Object settings dictionary, optionally containing ``'hlines'``
            and/or ``'vlines'`` lists, and optionally ``'timestamps'``.

        Returns
        -------
        dict
            Dictionary of the form ``{'hlines': {...}, 'vlines': {...}}``
            (only including keys that were present in ``settings``), each
            mapping a line key to its resolved ``'values'`` and settings.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> straightlines = organizer.getStraightLineValue(settings)
        """

        # dict that will hold the resolved hlines/vlines data
        straightlines = {}
        types_of_straightlines = ['hlines', 'vlines']
        # process horizontal and vertical reference lines using the same logic
        for tosl in types_of_straightlines:
            if tosl in settings.keys():
                straightlines[tosl] = {}
                # process every individual reference line of this type (h or v)
                for line in settings[tosl]:
                    if 'value' in line.keys(): #if defined single value for all plots
                        # Fixed constant: repeat the value once per
                        # timestamp so it plots as a flat reference line.
                        value = float(line['value'])
                        const_key = f'constant_{value}'
                        if 'timestamps' in settings.keys():
                            # repeat the constant value once per timestamp
                            values = [value] * len(settings['timestamps'])
                        else:
                            # no timestamps defined, just a single-element placeholder
                            values = [value]
                        straightlines[tosl][const_key] = {'values': values,
                                                          'numtimesused': 0}
                        # carry over any additional settings from the line definition
                        for key in line.keys():
                            if key not in straightlines[tosl][const_key].keys():
                                straightlines[tosl][const_key][key] = line[key]
                        if 'units' not in straightlines[tosl][const_key].keys():
                            # no units defined for this constant line, default to None
                            straightlines[tosl][const_key]['units'] = None
                    else:
                        # Data-driven line: read it like any other time
                        # series, then (for profile-style plots with
                        # discrete timestamps) sample the closest value to
                        # each timestamp.
                        timeserieslines, timeserieslinesettings = self.getTimeSeriesDataDictionary({'lines': [line]})
                        # process each resulting time series (usually just one)
                        for timeserieslinekey in timeserieslines.keys():
                            values = timeserieslines[timeserieslinekey]['values']
                            dates = timeserieslines[timeserieslinekey]['dates']
                            if 'timestamps' in settings.keys():
                                # find, for each profile timestamp, the closest available data index
                                idx = WR.getClosestTime(settings['timestamps'], dates)
                                # will hold the sampled value for each profile timestamp
                                v_idx = []
                                # pick the value at (or NaN if beyond) each matched index
                                for id in idx:
                                    if id > len(values):
                                        # index out of range, no matching value available
                                        v_idx.append(np.nan)
                                    else:
                                        v_idx.append(values[id])
                                straightlines[tosl][timeserieslinekey] = {'values': v_idx}
                            # carry over any additional settings from the time series read
                            for key in timeserieslinesettings[timeserieslinekey].keys():
                                if key not in straightlines[tosl][timeserieslinekey].keys():
                                    straightlines[tosl][timeserieslinekey][key] = timeserieslinesettings[timeserieslinekey][key]

        return straightlines

    def getTimeSeries(self, Line_info, makecopy=True):
        """
        Read a single time series from whichever source it is defined by.

        This is the central time series reader for the whole module. It
        inspects ``Line_info`` to determine the data-source type (DSS
        record, DSS collection/ensemble path, CE-QUAL-W2 output file,
        external ResSim H5 file by coordinates, current model results by
        coordinates, or current model ResSim result name), reads from
        that source (or from ``self.Memory`` if it was already read),
        then applies any requested value-omission, interval-resampling,
        and time-window trimming before returning.

        Memory caching notes
        ---------------------
        For plain (non-collection) sources, if the computed memory key is
        already cached, the cached data is returned/copied directly. For
        DSS *collection* (ensemble) records, the cache additionally
        tracks which members have been read, so if a later call asks for
        members not yet cached, only those missing members are re-read
        from the DSS file and merged into the cached collection
        (``metadata['partialmemory']``).

        Parameters
        ----------
        Line_info : dict
            Settings dictionary describing the data source for a single
            line (e.g. containing ``'dss_path'``/``'dss_filename'``,
            ``'w2_file'``, ``'h5file'``, ``'easting'``/``'northing'``, or
            ``'ressimresname'`` keys), plus optional
            ``'omitvalue'``/``'omitvalues'``, ``'interval'``, and
            ``'window'`` post-processing settings.
        makecopy : bool, optional
            If ``True`` (default), data retrieved from ``self.Memory`` is
            deep-copied (via pickle round-trip) before being returned, so
            the caller can safely mutate it without corrupting the cache.
            If ``False``, the cached object is returned directly (faster,
            but callers must not mutate it).

        Returns
        -------
        times : numpy.ndarray or list
            Timestamps for the series (empty array if no data found).
        values : numpy.ndarray, list, or dict
            The data values. For DSS collections this is a dict keyed by
            ensemble member; otherwise typically an array.
        metadata : dict
            Metadata about the read, including at least ``'collection'``,
            ``'frommemory'``, ``'partialmemory'``, and ``'units'``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        Two items are marked TODO in the original source: moving the
        memory-lookup logic outside of this function so it's only called
        once, and validating that the cached interval matches the
        requested interval when reading from memory.

        Examples
        --------
        >>> times, values, metadata = organizer.getTimeSeries(line_info)
        """

        # base metadata dict shared/extended by every source-type branch below
        metadata = {'collection': False,
                    'frommemory': False,
                    'partialmemory': False
                    }

        if 'dss_path' in Line_info.keys(): #Get data from DSS record
            if 'dss_filename' not in Line_info.keys():
                # can't read a DSS record without knowing which file it's in
                WF.print2stdout('DSS_Filename not set for Line: {0}'.format(Line_info), debug=self.Report.debug)
                return np.array([]), np.array([]), metadata
            else:
                # compute the memory key for this specific DSS record/path
                datamem_key = self.buildMemoryKey(Line_info)

                # A DSS B-path segment starting with '*|' marks this as a
                # collection (ensemble/forecast) path rather than a single
                # record; the '*' is a DSS wildcard across member names.
                if Line_info['dss_path'].split('/')[6].startswith('*|'):

                    # mark this as a multi-member collection read
                    metadata['collection'] = True
                    metadata['allmembers'] = False
                    if 'members' in Line_info.keys():
                        # caller explicitly specified which members are wanted
                        wanted_members = Line_info['members']
                    elif self.Report.reportType == 'forecast': #uses the ones defined in the forecast
                        # forecast report, use every member the report itself tracks
                        wanted_members = self.Report.allMembers
                    else: #otherwise, grab ALL in the dssfile
                        # no explicit members requested, grab everything found in the file
                        wanted_members = 'all'
                        metadata['allmembers'] = True
                    #grab all the members on the first go, we will be using the data
                    #then we can keep track of duplicates. Save to memory, but only return the wanted ones
                    # track the original member list separately, since wanted_members can change below
                    metadata['members'] = self.Report.allMembers #keep track of the original series, as this can change

                if datamem_key in self.Memory.keys():
                    WF.print2stdout('Reading {0} from memory'.format(datamem_key), debug=self.Report.debug) #noisy
                    if makecopy:
                        # deep-copy so the caller can safely mutate without corrupting the cache
                        datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                    else:
                        # caller accepts a direct reference to the cached object (faster, no mutation allowed)
                        datamementry = self.Memory[datamem_key]
                    # unpack the cached times/values/metadata
                    times = datamementry['times']
                    values = datamementry['values']
                    metadata = datamementry['metadata']

                    metadata['frommemory'] = True #did we get data from memory
                    members_to_grab = [] #reset, but we still know our members
                    if metadata['collection']:
                        # For collections, verify every wanted member is
                        # actually present in the cached data; if not,
                        # we'll need to fetch just the missing ones below.
                        if not metadata['allmembers']: #check if we've ever grabbed them all. if we did, no need to go back
                            # check each wanted member individually against what's cached
                            for member in wanted_members:
                                # if member not in metadata['members']:
                                if member not in datamementry['metadata']['members']:
                                    # this member isn't cached yet, mark it for a fresh read
                                    members_to_grab.append(member)
                                    metadata['frommemory'] = False
                                    metadata['partialmemory'] = True
                                else:
                                    # member already present in the cache
                                    metadata['partialmemory'] = True

                        if len(members_to_grab) > 0:
                            WF.print2stdout(f'Not all members in memory. Getting remaining: {members_to_grab}', debug=self.Report.debug)

                if not metadata['frommemory']:
                    if metadata['collection']:
                        if metadata['partialmemory']: #if we've only grabbed some of them...
                            # Only fetch the members that were missing from
                            # the cache, then merge them into the existing
                            # cached collection values.
                            # read just the missing members from the DSS file
                            coll_times, coll_values, units, coll_members = WDR.readCollectionsDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                                                                              members_to_grab, self.Report.StartTime,
                                                                                                              self.Report.EndTime, self.Report.debug)

                            # merge the newly-read members into the existing cached value dict
                            values.update(coll_values)
                            # union the newly-read members into the tracked member list
                            members = list(set(metadata['members'] + coll_members))
                        else:
                            # Nothing cached yet: read the whole collection.
                            times, values, units, members = WDR.readCollectionsDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                                                          metadata['members'], self.Report.StartTime,
                                                                                          self.Report.EndTime, self.Report.debug)
                        # record the (possibly expanded) member list back into metadata
                        metadata['members'] = members #todo: look at this closer

                        #grab the wanted guys
                        # values = [{key: values[key]} for key in wanted_members] #this is done below, but this looks cool
                    else:
                        # Single (non-collection) DSS record.
                        times, values, units = WDR.readDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                               self.Report.StartTime, self.Report.EndTime,
                                                               self.Report.debug)
                    # record the units returned from the DSS read
                    metadata['units'] = units
                    # Cache a deep copy of everything read so subsequent
                    # requests for this same key can be served from memory.
                    self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                                 'values': pickle.loads(pickle.dumps(values, -1)),
                                                 'metadata': pickle.loads(pickle.dumps(metadata, -1))}

                if np.any(values == None):
                    # no usable values came back, bail out with empty results
                    return np.array([]), np.array([]), metadata
                elif len(values) == 0:
                    # values array is empty, also nothing usable to return
                    return np.array([]), np.array([]), metadata

        elif 'w2_file' in Line_info.keys():
            # CE-QUAL-W2 output file column (or structure/outlet time
            # series). Only valid if the current model program is W2.
            if self.Report.program.lower() != 'cequalw2':
                # this line type only applies to a W2 model, bail out otherwise
                return np.array([]), np.array([]), None
            # compute the memory key for this W2 output file/column
            datamem_key = self.buildMemoryKey(Line_info)
            if datamem_key in self.Memory.keys():
                # already cached, pull it straight from memory
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                metadata = datamementry['metadata']
                metadata['frommemory'] = True

            if not metadata['frommemory']:
                if 'structurenumbers' in Line_info.keys():
                    # Ryan Miles: yeah looks like it's str_brX.npt, and X is 1-# of branches (which is defined in the control file)
                    # read a structured (gate/outlet) time series for the requested structures
                    times, values = self.Report.ModelAlt.readStructuredTimeSeries(Line_info['w2_file'], Line_info['structurenumbers'])
                else:
                    # read a plain column from the W2 output file
                    times, values = self.Report.ModelAlt.readTimeSeries(Line_info['w2_file'], **Line_info)

                # Determine units: explicit line units win, otherwise
                # look up the metric unit for the named parameter, else
                # leave undefined.
                if 'units' in Line_info.keys():
                    # explicit units given, use them directly
                    metadata['units'] = Line_info['units']
                elif 'parameter' in Line_info.keys():
                    # look up the metric unit associated with the named parameter
                    plotunits = constants.units[Line_info['parameter'].lower()]
                    metadata['units'] = plotunits['metric']
                else:
                    # no way to determine units, leave undefined
                    metadata['units'] = None

                # cache a deep copy for future requests of this same source
                self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                            'values': pickle.loads(pickle.dumps(values, -1)),
                                            'metadata': pickle.loads(pickle.dumps(metadata, -1))}

        elif 'h5file' in Line_info.keys() and 'easting' in Line_info.keys() and 'northing' in Line_info.keys():
            # External ResSim H5 results file, looked up by
            # (easting, northing) coordinates rather than a named result.
            # compute the memory key for this external H5 coordinate lookup
            datamem_key = self.buildMemoryKey(Line_info)
            if 'subdomain' in Line_info.keys():
                # a specific subdomain was requested for this coordinate lookup
                subdomain = Line_info['subdomain']
            else:
                # no subdomain restriction specified
                subdomain = None
            if datamem_key in self.Memory.keys():
                # already cached, pull it straight from memory
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                metadata = datamementry['metadata']
                metadata['frommemory'] = True

            if not metadata['frommemory']:
                filename = Line_info['h5file']
                if not os.path.exists(filename):
                    # can't proceed without the source file
                    WF.print2stdout('ERROR: H5 file does not exist:', filename, debug=self.Report.debug)
                    return [], [], None
                # Open the external H5 result set fresh for this read (it
                # is a lightweight wrapper, not itself cached across
                # calls) and pull the requested coordinate's time series.
                externalResSim = WRSS.ResSim_Results('', '', '', '', self.Report, external=True)
                # open the H5 file and load its time/subdomain metadata before reading
                externalResSim.openH5File(filename)
                externalResSim.load_time() #load time vars from h5
                externalResSim.loadSubdomains()
                # read the actual coordinate-based time series from the H5 file
                times, values, units = externalResSim.readTimeSeries(Line_info['parameter'],
                                                              float(Line_info['easting']),
                                                              float(Line_info['northing']),
                                                              subdomain=subdomain)
                if 'units' in Line_info.keys():
                    # explicit units override whatever the H5 file reports
                    metadata['units'] = Line_info['units']
                else:
                    # use the units reported by the H5 read itself
                    metadata['units'] = units

                # cache a deep copy for future requests of this same source
                self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                            'values': pickle.loads(pickle.dumps(values, -1)),
                                            'metadata': pickle.loads(pickle.dumps(metadata, -1))}

        elif 'easting' in Line_info.keys() and 'northing' in Line_info.keys():
            # Coordinate-based lookup against the CURRENT model's own
            # results (as opposed to an external H5 file above).
            # compute the memory key for this current-model coordinate lookup
            datamem_key = self.buildMemoryKey(Line_info)
            if 'subdomain' in Line_info.keys():
                # a specific subdomain was requested for this coordinate lookup
                subdomain = Line_info['subdomain']
            else:
                # no subdomain restriction specified
                subdomain = None
            if datamem_key in self.Memory.keys():
                # already cached, pull it straight from memory
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                metadata = datamementry['metadata']
                metadata['frommemory'] = True

            if not metadata['frommemory']:
                # read the coordinate-based time series directly from the current model's results
                times, values, units = self.Report.ModelAlt.readTimeSeries(Line_info['parameter'],
                                                                    float(Line_info['easting']),
                                                                    float(Line_info['northing']),
                                                                    subdomain=subdomain)
                if 'units' in Line_info.keys():
                    # explicit units override whatever the model read reports
                    metadata['units'] = Line_info['units']
                else:
                    # use the units reported by the model read itself
                    metadata['units'] = units

                # cache a deep copy for future requests of this same source
                self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                             'values': pickle.loads(pickle.dumps(values, -1)),
                                             'metadata': pickle.loads(pickle.dumps(metadata, -1))}

        elif "ressimresname" in Line_info.keys():
            # Named ResSim result (reservoir/reach), used for the current
            # model's own output. Supports two special sub-modes:
            # 'target' (value of another parameter at a target condition,
            # e.g. elevation at the time flow first hits a target) and
            # 'fwa' (flow-weighted-average reservoir output).
            # compute the memory key for this named ResSim result
            datamem_key = self.buildMemoryKey(Line_info)
            if datamem_key in self.Memory.keys():
                # already cached, pull it straight from memory
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                metadata = datamementry['metadata']
                metadata['frommemory'] = True

            if not metadata['frommemory']:
                # start with empty placeholders in case the model type check below fails
                times = []
                values = []
                if self.Report.program.lower() != 'ressim':
                    # this line type only applies to ResSim models
                    WF.print2stdout('Incorrect model type for line using ResSimResName', debug=self.Report.debug)
                    return [], [], metadata

                if 'target' in Line_info.keys():
                    # Default to elevation if no parameter is specified for
                    # the target-value timeseries.
                    if 'parameter' not in Line_info.keys():
                        # no parameter given, fall back to elevation as a sensible default
                        WF.print2stdout('No parameter for profile target timeseries.', debug=self.Report.debug)
                        WF.print2stdout('Assuming output is elevation.', debug=self.Report.debug)
                        metadata['parameter'] = 'elevation'
                    else:
                        # use the explicitly given parameter
                        metadata['parameter'] = Line_info['parameter']
                    # read the target-condition time series for the requested parameter
                    times, values, units = self.Report.ModelAlt.getProfileTargetTimeseries(Line_info['ressimresname'],
                                                                                    Line_info['parameter'],
                                                                                    metadata['target'])
                    metadata['type'] = 'target'
                    metadata['units'] = units

                elif 'fwa' in Line_info.keys():
                    if Line_info['fwa'].lower() == 'true': #not sure what to do otherwise..
                        # Default to temperature if no parameter is
                        # specified for the FWA reservoir timeseries.
                        if 'parameter' not in Line_info.keys():
                            # no parameter given, fall back to temperature as a sensible default
                            WF.print2stdout('No parameter for FWA reservoir timeseries.', debug=self.Report.debug)
                            WF.print2stdout('Assuming output is temperature.', debug=self.Report.debug)
                            metadata['parameter'] = 'temperature'
                        else:
                            # use the explicitly given parameter
                            metadata['parameter'] = Line_info['parameter']
                        # read the flow-weighted-average reservoir output time series
                        times, values, units = self.Report.ModelAlt.getFWAReservoirOutputTimeseries(Line_info['ressimresname'],
                                                                                             metadata['parameter'])
                        metadata['type'] = 'fwa'
                if 'units' in Line_info.keys():
                    # explicit units override whatever mode above determined
                    metadata['units'] = Line_info['units']
                else:
                    # no explicit override, leave whatever units mode above resolved (or None)
                    metadata['units'] = None

        else:
            # None of the recognized data-source key combinations matched.
            WF.print2stdout('No Data Defined for line', debug=self.Report.debug)
            return np.array([]), np.array([]), metadata

        if metadata['collection']:
            # Tag identical-data members and (unless every member was
            # requested) filter down to just the members the caller asked
            # for.
            # find duplicate-data member groupings before filtering the member list down
            metadata = self.checkForIdenticalMembers(values, metadata)
            if wanted_members != 'all':
                # restrict the values dict down to only the members the caller actually wants
                values = WF.filterByMember(values, wanted_members)
            #we should check for duplicates here

        # Optional post-processing shared by every source type: replace
        # sentinel "omit" value(s) with NaN so they don't get plotted or
        # skew computed statistics.
        if 'omitvalue' in Line_info.keys():
            # single sentinel value to NaN-out
            omitval = float(Line_info['omitvalue'])
            values = WF.NaNOmittedValues(values, omitval, debug=self.Report.debug)
        elif 'omitvalues' in Line_info.keys():
            # multiple sentinel values to NaN-out, one at a time
            omitvals = [float(n) for n in Line_info['omitvalues']]
            # apply the same NaN-out logic once per sentinel value in the list
            for omitval in omitvals:
                values = WF.NaNOmittedValues(values, omitval, debug=self.Report.debug)

        # Resample to a different reporting interval, if requested.
        if 'interval' in Line_info.keys():
            times, values = WT.changeTimeSeriesInterval(times, values, Line_info, self.Report.startYear)
            metadata['interval_mod'] = True

        # if a window is provided, trim to it
        if 'window' in Line_info.keys():
            times, values = WT.trimWindow(times, values, Line_info['window'])
            metadata['window_mod'] = True

        return times, values, metadata

    def computeCollectionEnvelopes(self, values, envelopes):
        """
        Compute percentile envelope curves across a forecast collection.

        For each requested percentile (e.g. 10th, 50th, 90th), computes
        the value of that percentile across all members at every
        timestep, ignoring NaNs. Percentiles outside the valid 0-100
        range (or otherwise invalid) are dropped with a log message.

        Parameters
        ----------
        values : dict
            Dictionary keyed by ensemble member, each mapping to that
            member's value array (all arrays assumed to be the same
            length/aligned in time).
        envelopes : list of dict
            List of envelope definitions, each expected to contain a
            ``'percent'`` key (e.g. ``'10'``, ``'50'``, ``'90'``).

        Returns
        -------
        dict
            Dictionary keyed by the percentile tag string, each mapping
            to a list of the computed quantile values (one per timestep).
            Invalid/out-of-range percentiles are omitted from the result.

        Raises
        ------
        None
            This function does not explicitly raise exceptions; invalid
            percentiles are handled internally and logged rather than
            propagated.

        Examples
        --------
        >>> envelopes = organizer.computeCollectionEnvelopes(values, [{'percent': '10'}, {'percent': '90'}])
        """

        # accumulator for the computed quantile series, keyed by percentile tag
        collected_envelopes = {}
        # Initialize an empty result list for each requested percentile.
        for envelope in envelopes:
            if 'percent' in envelope.keys():
                envelope_tag = envelope['percent']
                collected_envelopes[envelope_tag] = []
        # step through every timestep, computing each requested quantile across all members
        for vi in range(len(values[list(values.keys())[0]])): #for each value in a single set of values
            # compute every requested percentile at this single timestep
            for envelope in collected_envelopes.keys():
                try:
                    # convert the percentile tag to a float and rescale to the 0-1 range
                    quantile = float(envelope)
                    quantile = quantile / 100 #envlopes come in as 0-100, but we need 0 - 1
                    # validate the quantile is actually within the legal 0-1 range
                    assert(0. <= quantile <= 1.)
                    # Gather this timestep's value across every member,
                    # then compute the requested quantile (ignoring NaNs
                    # unless every member is NaN at this timestep).
                    # collect this timestep's value from every ensemble member
                    quantilevals = []
                    for key, vs in values.items():
                        quantilevals.append(vs[vi])
                    if np.all(np.isnan(quantilevals)):
                        # every member is NaN at this timestep, propagate NaN rather than error
                        collected_envelopes[envelope].append(np.nan)
                    else:
                        # compute the requested quantile across the gathered member values
                        collected_envelopes[envelope].append(np.nanquantile(quantilevals, quantile))
                except AssertionError:
                    # Percentile was outside the valid 0-1 range after
                    # conversion; log why and drop this envelope entirely.
                    if quantile < 0.:
                        WF.print2stdout(f'Envelope {envelope} under 0. Envelopes must be between 0 and 1.', debug=self.Report.debug)
                    elif quantile > 1.:
                        WF.print2stdout(f'Envelope {envelope} over 1. Envelopes must be between 0 and 1.', debug=self.Report.debug)
                    else:
                        WF.print2stdout(f'Unknown Collection plot envelope {envelope}. Skipping.', debug=self.Report.debug)
                    # drop this invalid envelope from the result set entirely
                    collected_envelopes.pop(envelope)
                except:
                    # Any other failure (e.g. non-numeric percentile tag).
                    WF.print2stdout(f'Unknown Collection plot envelope {envelope}. Skipping.', debug=self.Report.debug)
                    # drop this failed envelope from the result set entirely
                    collected_envelopes.pop(envelope)

        return collected_envelopes

    #################################################################
    #Profile Functions
    #################################################################

    def getProfileDataDictionary(self, settings):
        """
        Build the data dictionary for all profile lines in an object.

        Mirrors ``getTimeSeriesDataDictionary`` but for vertical profile
        data: a ``'computed'`` line is expanded into one profile per
        accepted simulation ID, while other lines have model-specific
        placeholders resolved (or are reconfigured to the base simulation
        ID for comparison plots). Each line is read via
        ``updateProfileDataDictionary``.

        Parameters
        ----------
        settings : dict
            Object settings dictionary; must contain ``'timestamps'`` and
            a list of lines under the key named by ``settings['datakey']``.

        Returns
        -------
        data : dict
            Dictionary of profile data keyed by (unique) line flag.
        line_settings : dict
            Dictionary of settings/metadata for each flag in ``data``.
        missing : list of str
            Flags for lines that were expected but produced no data.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, line_settings, missing = organizer.getProfileDataDictionary(settings)
        """

        # accumulator dicts, plus a list tracking flags that failed to produce data
        data = {}
        line_settings = {}
        missing = []
        timestamps = settings['timestamps']
        # process every line defined under this object's data key
        for line in settings[settings['datakey']]:
            numtimesused = 0
            if 'flag' not in line.keys():
                # can't process a line with no identifying flag, skip it entirely
                WF.print2stdout('Flag not set for line (Computed/Observed/etc)', debug=self.Report.debug)
                WF.print2stdout('Not plotting Line:', line, debug=self.Report.debug)
                continue
            elif line['flag'].lower() == 'computed':
                # for ID in self.SimulationVariables.keys():
                # Expand a 'computed' profile line into one profile per
                # accepted simulation ID.
                # loop over every accepted simulation ID for this report
                for ID in self.Report.accepted_IDs:
                    # deep-copy the line settings so each ID gets its own independent config
                    curline = pickle.loads(pickle.dumps(line, -1))
                    curline = self.Report.configureSettingsForID(ID, curline)
                    curline['numtimesused'] = numtimesused
                    curline['ID'] = ID
                    if not self.Report.checkModelType(curline):
                        # this ID's model type doesn't match, skip it
                        continue
                    # read and register this ID's version of the profile
                    data, line_settings, success = self.updateProfileDataDictionary(data, line_settings, curline, timestamps)
                    if success:
                        numtimesused += 1
                    else:
                        # track the flag as missing since no data came back for it
                        missing.append(curline['flag'])
            else:
                # Non-computed line: resolve to the base simulation ID or
                # replace model-specific placeholder values as needed.
                if self.Report.currentlyloadedID != self.Report.base_id:
                    # currently on a comparison ID, reconfigure to the base ID's settings
                    line = self.Report.configureSettingsForID(self.Report.base_id, line)
                else:
                    # already on the base ID, resolve model-specific placeholders directly
                    line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')
                line['numtimesused'] = numtimesused
                if not self.Report.checkModelType(line):
                    # model type mismatch, skip this line entirely
                    continue
                # read and register this non-computed profile line
                data, line_settings, success = self.updateProfileDataDictionary(data, line_settings, line, timestamps)
                if success:
                    numtimesused += 1
                else:
                    # track the flag as missing since no data came back for it
                    missing.append(line['flag'])

        return data, line_settings, missing

    def updateProfileDataDictionary(self, data, line_settings, profile, timestamps):
        """
        Fetch one profile line's data and add it to the data dictionaries.

        Calls ``getProfileValues`` to read/retrieve the profile for
        ``profile`` at the given ``timestamps``, then (if any values were
        returned) stores the values/elevations/depths/times under a
        unique key in ``data`` and records the associated metadata and
        settings under the same key in ``line_settings``.

        Parameters
        ----------
        data : dict
            Dictionary accumulating profile entries processed so far.
        line_settings : dict
            Dictionary accumulating settings/metadata for each profile.
        profile : dict
            Settings dictionary for the single profile line being
            processed; must include ``'flag'`` and ``'numtimesused'``.
        timestamps : list or str
            Timestamps to extract profile data at (or ``'all'``).

        Returns
        -------
        data : dict
            Updated data dictionary (with this profile added, if valid).
        line_settings : dict
            Updated line settings dictionary.
        datacheck : bool
            ``True`` if this profile produced usable data, ``False``
            otherwise.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, line_settings, ok = organizer.updateProfileDataDictionary({}, {}, profile, timestamps)
        """

        # read the actual profile values/elevations/depths/times for this line
        vals, elevations, depths, times, metadata = self.getProfileValues(profile, timestamps) #Test this speed for grabbing all profiles and then choosing
        # default to "no usable data" until proven otherwise below
        datacheck = False
        if len(vals) > 0:
            # data came back usable, proceed to register it under a key
            datacheck = True
            datamem_key = self.buildMemoryKey(profile)

            # If this flag is already used, disambiguate with the
            # numtimesused counter (set by the caller) rather than an
            # incrementing suffix, since profiles are re-fetched per
            # timestamp set rather than accumulated line-by-line.
            if profile['flag'] in line_settings.keys() or profile['flag'] in data.keys():
                # flag already used, append the usage counter to disambiguate
                datakey = '{0}_{1}'.format(profile['flag'], profile['numtimesused'])
            else:
                # flag is free, use it as-is
                datakey = profile['flag']

            # record which memory/log file this profile corresponds to
            line_settings[datakey] = {'logoutputfilename': datamem_key,
                                      }

            # merge in the metadata read from source, then the original profile settings on top
            line_settings[datakey].update(metadata)
            line_settings[datakey].update(profile)

            # store the actual profile arrays under this key
            data[datakey] = {'values': vals,
                             'elevations': elevations,
                             'depths': depths,
                             'times': times
                             }

            # fill in any remaining profile keys not already captured above
            for key in profile.keys():
                if key not in line_settings[datakey].keys():
                    line_settings[datakey][key] = profile[key]

        return data, line_settings, datacheck

    def getProfileValues(self, Profile_info, timesteps):
        """
        Read a vertical profile's data from whichever source defines it.

        Similar in structure to ``getTimeSeries``, but for profile data
        (values plotted against depth or elevation at one or more
        timesteps rather than plotted against time). Supports observed
        text-file profiles, external ResSim H5 files, CE-QUAL-W2 model
        segments, and the current model's own ResSim result names.
        Results are cached in ``self.Memory`` keyed by the profile's
        memory key; a cached entry is only reused if it was read for the
        same timestep selection (either the same explicit list of
        timesteps, or previously "all" and "all" is requested again).

        Parameters
        ----------
        Profile_info : dict
            Settings dictionary describing the profile's data source
            (``'filename'``, ``'h5file'``+``'ressimresname'``,
            ``'w2_segment'``, or ``'ressimresname'``), plus ``'flag'``
            and optionally ``'units'``/``'y_convention'``.
        timesteps : list or str
            Specific timesteps to extract, or the string ``'all'`` to
            request every timestep in the source.

        Returns
        -------
        values : list or numpy.ndarray
            Profile values (e.g. temperature) at each depth/elevation.
        elevations : list or numpy.ndarray
            Elevation for each value, if available (may be empty if the
            source only supplies depths).
        depths : list or numpy.ndarray
            Depth for each value, if available (may be empty if the
            source only supplies elevations).
        times : list or numpy.ndarray
            The timestep(s) the profile data corresponds to.
        metadata : dict
            Metadata about the read (units, source, whether it came from
            memory, etc.).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> values, elevations, depths, times, metadata = organizer.getProfileValues(profile_info, 'all')
        """

        # base metadata dict describing this profile read
        metadata = {'frommemory': False,
                    'subset': False,
                    'flag': Profile_info['flag'],
                    'units': None,
                    'y_units': None,
                    'isprofile': True
                    }

        if 'units' in Profile_info.keys():
            # explicit units given in the profile settings, record them up front
            metadata['units'] = Profile_info['units']

        # compute the memory key for this profile and set up empty defaults
        datamemkey = self.buildMemoryKey(Profile_info)
        values, elevations, depths, times = [], [], [], []

        # A string timesteps value (e.g. 'all') means "every timestep in
        # the source", i.e. not a subset; an explicit list means we only
        # want a subset of the full profile.
        if isinstance(timesteps, str):
            # 'all' (or similar) requested, not a subset
            metadata['subset'] = False
        else:
            # explicit list of timesteps requested, this is a subset
            metadata['subset'] = True

        if datamemkey in self.Memory.keys():
            # pull a deep copy of the cached entry so we can inspect it safely
            dm = pickle.loads(pickle.dumps(self.Memory[datamemkey], -1))
            subset = dm['metadata']['subset'] #if when grabbed, all timesteps were grabbed, or specific ones
            metadata['frommemory'] = True
            WF.print2stdout(f'retrieving {datamemkey} profile from datamemory', debug=self.Report.debug)
            if isinstance(timesteps, str): #if looking for all
                if not subset: #the last time data was grabbed, it was not a subset, aka all
                    # cached data covers everything, so it's safe to reuse for an 'all' request
                    metadata['units'] = dm['units']
                    values, elevations, depths, times = dm['values'], dm['elevations'], dm['depths'], dm['times']
                else:
                    # Cached data was only a subset, but we now need
                    # everything, so re-read from source instead.
                    WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)
                    metadata['frommemory'] = False
            elif np.array_equal(timesteps, dm['times']):
                # requested timesteps exactly match what's cached, safe to reuse
                metadata['units'] = dm['metadata']['units']
                values, elevations, depths, times = dm['values'], dm['elevations'], dm['depths'], dm['times']
            else:
                # Requested timesteps don't match what's cached; re-read.
                WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)
                metadata['frommemory'] = False

        #read from source..
        if not metadata['frommemory']:
            if 'filename' in Profile_info.keys(): #Get data from Observed
                # Observed text-file profile. The vertical axis convention
                # (depth vs. elevation) determines which of
                # elevations/depths gets populated from the raw y-values.
                filename = Profile_info['filename']
                metadata['source'] = filename
                # read the raw observed profile values and their y-axis (depth or elevation) values
                values, yvals, times = WDR.readTextProfile(filename, timesteps, self.Report.StartTime, self.Report.EndTime)
                if 'y_convention' in Profile_info.keys():
                    metadata['y_convention'] = Profile_info['y_convention']
                    if Profile_info['y_convention'].lower() == 'depth':
                        # y-values represent depth, so leave elevations empty
                         values, elevations, depths, times = values, [], yvals, times
                    elif Profile_info['y_convention'].lower() == 'elevation':
                        # y-values represent elevation, so leave depths empty
                        values, elevations, depths, times = values, yvals, [], times
                    else:
                        # unrecognized convention string, fall back to assuming depth
                        WF.print2stdout('Unknown value for flag y_convention: {0}'.format(Profile_info['y_convention']), debug=self.Report.debug)
                        WF.print2stdout('Please use "depth" or "elevation"', debug=self.Report.debug)
                        WF.print2stdout('Assuming depths...', debug=self.Report.debug)
                        values, elevations, depths, times = values, [], yvals, times
                else:
                    # no convention specified at all, default to depth assumption
                    WF.print2stdout('No value for flag y_convention', debug=self.Report.debug)
                    WF.print2stdout('Assuming depths...', debug=self.Report.debug)
                    values, elevations, depths, times = values, [], yvals, times

            elif 'h5file' in Profile_info.keys() and 'ressimresname' in Profile_info.keys():
                # External ResSim H5 file profile for a named result.
                filename = Profile_info['h5file']
                metadata['source'] = filename
                if not os.path.exists(filename):
                    # log the missing file, though the code proceeds to attempt opening it below regardless
                    WF.print2stdout('ERROR: H5 file does not exist:', filename, debug=self.Report.debug)
                externalResSim = WRSS.ResSim_Results('', '', '', '', self.Report, external=True)
                # open the H5 file and load its time/subdomain metadata before reading
                externalResSim.openH5File(filename)
                externalResSim.load_time() #load time vars from h5
                externalResSim.loadSubdomains()
                # read the actual profile data for the requested result/parameter
                values, elevations, depths, times, units = externalResSim.readProfileData(Profile_info['ressimresname'],
                                                                                          Profile_info['parameter'], timesteps)
                metadata['units'] = units

            elif 'w2_segment' in Profile_info.keys():
                # CE-QUAL-W2 model segment profile (only valid if the
                # current model program is W2).
                if self.Report.program.lower() == 'cequalw2':
                    if 'w2_file' in Profile_info.keys():
                        # explicit results file given, use it
                        resultsfile = Profile_info['w2_file']
                    else:
                        # no explicit file, let the model figure out the default
                        resultsfile = None
                    metadata['source'] = resultsfile
                    # read the profile data for the requested segment from the W2 model
                    values, elevations, depths, times = self.Report.ModelAlt.readProfileData(Profile_info['w2_segment'],
                                                                                             timesteps,
                                                                                             resultsfile=resultsfile)
                    metadata['units'] = 'c' #W2 outputs in metric
                    # W2 stores times as Julian dates relative to the
                    # model start year; convert to real datetimes.
                    times = WT.JDateToDatetime(times, self.Report.startYear)

            elif 'ressimresname' in Profile_info.keys():
                # Current model's own ResSim result-name profile (only
                # valid if the current model program is ResSim).
                if self.Report.program.lower() == 'ressim':
                    metadata['source'] = Profile_info['ressimresname']
                    # read the profile data for the requested ResSim result
                    values, elevations, depths, times, units = self.Report.ModelAlt.readProfileData(Profile_info['ressimresname'],
                                                                                                   Profile_info['parameter'], timesteps,
                                                                                                   )
                    metadata['units'] = units

        # Cache whatever was read (or already-cached-and-reused values)
        # under this profile's memory key.
        self.Memory[datamemkey] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                   'values': pickle.loads(pickle.dumps(values, -1)),
                                   'elevations': pickle.loads(pickle.dumps(elevations, -1)),
                                   'depths': pickle.loads(pickle.dumps(depths, -1)),
                                   'metadata': pickle.loads(pickle.dumps(metadata, -1))}

        if len(values) == 0:
            # log that nothing could be found for this profile, for troubleshooting
            WF.print2stdout('No Data Defined for Profile', debug=self.Report.debug)
            WF.print2stdout('Profile:', Profile_info, debug=self.Report.debug)

        return values, elevations, depths, times, metadata
        # return [], [], [], [], metadata

    def getReservoirContourDataDictionary(self, settings):
        """
        Build the data dictionary for a reservoir contour plot.

        For every data path and every accepted simulation ID, reads the
        full profile (``getProfileValues`` with ``timesteps='all'``)
        along with its top-water elevation series
        (``getProfileTopWater``), optionally resampling both to a
        different reporting interval, and stores the result under a
        unique flag. Temporarily switches the active simulation ID for
        each iteration, then resets back to the report's base ID when
        done.

        Parameters
        ----------
        settings : dict
            Object settings dictionary containing a ``'datapaths'`` list.

        Returns
        -------
        data : dict
            Dictionary of contour data (values, dates, elevations,
            topwater) keyed by (unique) flag.
        data_settings : dict
            Dictionary of settings/metadata for each flag in ``data``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, data_settings = organizer.getReservoirContourDataDictionary(settings)
        """

        # accumulator dicts built up across every datapath/simulation ID combination
        data = {}
        data_settings = {}
        # process every reservoir datapath configured for this contour plot
        for datapath in settings['datapaths']:
            # loop over every accepted simulation ID for this report
            for ID in self.Report.accepted_IDs:
                # Reconfigure a fresh copy of this datapath's settings for
                # the current simulation ID being iterated over.
                curreach = pickle.loads(pickle.dumps(datapath, -1))
                curreach = self.Report.configureSettingsForID(ID, curreach)
                if not self.Report.checkModelType(curreach):
                    # this ID's model type doesn't match, skip it
                    continue
                # read the full profile (all timesteps) and its top-water elevation series
                values, elevations, depths, dates, metadata = self.getProfileValues(curreach, 'all')
                topwater = self.getProfileTopWater(curreach, 'all')
                if 'interval' in curreach.keys():
                    # resample both the profile values and top-water series to the requested interval
                    dates_change, values = WT.changeTimeSeriesInterval(dates, values, curreach,
                                                                       self.Report.startYear)
                    dates_change, topwater = WT.changeTimeSeriesInterval(dates, topwater, curreach,
                                                                         self.Report.startYear)
                    # use the (identical) resampled date series going forward
                    dates = dates_change
                if WF.checkData(values):
                    # Determine the flag to store this ID's data under,
                    # disambiguating with a numeric suffix if needed.
                    if 'flag' in datapath.keys():
                        # explicit flag given, use it
                        flag = datapath['flag']
                    elif 'label' in datapath.keys():
                        # fall back to the label if no flag was given
                        flag = datapath['label']
                    else:
                        # no flag or label at all, auto-generate one from the simulation ID
                        flag = 'reservoir_{0}'.format(ID)
                    if flag in data.keys():
                        # flag already used by another ID, disambiguate with an incrementing suffix
                        count = 1
                        newflag = flag + '_{0}'.format(count)
                        while newflag in data.keys():
                            count += 1
                            newflag = flag + '_{0}'.format(count)
                        WF.print2stdout('The current flag is {0}'.format(flag), debug=self.Report.debug)
                        flag = newflag
                        WF.print2stdout('The new flag is {0}'.format(newflag), debug=self.Report.debug)
                    # compute the memory key for logging/tracing purposes
                    datamem_key = self.buildMemoryKey(datapath)

                    # record which memory/log file and simulation ID this entry corresponds to
                    data_settings[flag] = {'logoutputfilename': datamem_key,
                                           'ID': ID,
                                           }

                    # store the actual contour arrays under this flag
                    data[flag] = {'values': values,
                                  'dates': dates,
                                  'elevations': elevations,
                                  'topwater': topwater,
                                  'ID': ID}

                    # merge in the metadata read from source, then the original datapath settings on top
                    data_settings[flag].update(metadata)
                    data_settings[flag].update(curreach)

                    # fill in any remaining datapath keys not already captured above
                    for key in datapath.keys():
                        if key not in data_settings[flag].keys():
                            data_settings[flag][key] = datapath[key]
        #reset
        # Restore the report's active simulation ID now that we're done
        # looping over every accepted ID.
        self.Report.loadCurrentID(self.Report.base_id)
        self.Report.loadCurrentModelAltID(self.Report.base_id)
        return data, data_settings

    def getProfileTopWater(self, profile, timesteps):
        """
        Retrieve the top-water (surface) elevation for a profile source.

        Used for reservoir contour plots to know where the water surface
        sits at each timestep, independent of the profile's own value
        data. Checks memory first, then falls back to reading from the
        profile's underlying source (observed text file, external H5,
        W2 segment, or current-model ResSim result).

        Parameters
        ----------
        profile : dict
            Settings dictionary describing the profile's data source.
        timesteps : list or str
            Specific timesteps to extract, or ``'all'``.

        Returns
        -------
        list or numpy.ndarray
            Top-water elevation values, one per requested timestep
            (empty list if unavailable for this source type, e.g. an
            observed profile recorded in depth rather than elevation).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> topwater = organizer.getProfileTopWater(profile, 'all')
        """

        # compute the memory key for this profile's top-water lookup
        datamemkey = self.buildMemoryKey(profile)
        if datamemkey in self.Memory.keys():
            # pull a deep copy of the cached entry so we can inspect it safely
            dm = pickle.loads(pickle.dumps(self.Memory[datamemkey], -1))
            WF.print2stdout('retrieving profile topwater from datamem', debug=self.Report.debug)
            if isinstance(timesteps, str): #if looking for all
                if dm['subset'] == 'false': #the last time data was grabbed, it was not a subset, aka all
                    # cached data covers everything, safe to reuse
                    return dm['topwater']
                else:
                    # cached data was only a subset, need to re-extract
                    WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)
            elif np.array_equal(timesteps, dm['times']):
                # requested timesteps exactly match what's cached
                return dm['topwater']
            else:
                # cached timesteps don't match what's requested, need to re-extract
                WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)

        if 'filename' in profile.keys(): #Get data from Observed
            # Observed text profile: top-water is only derivable if the
            # profile's y-axis is already in elevation (the first y-value
            # per timestep is the shallowest point, i.e. the surface).
            filename = profile['filename']
            # read the raw profile values and y-axis values from the observed text file
            values, yvals, times = WDR.readTextProfile(filename, timesteps, self.Report.StartTime, self.Report.EndTime)
            if 'y_convention' in profile.keys():
                if profile['y_convention'].lower() == 'elevation':
                    # first y-value per timestep is the shallowest, i.e. surface elevation
                    return [yval[0] for yval in yvals]
                elif profile['y_convention'].lower() == 'depth':
                    # can't derive top-water elevation directly from a depth-only profile
                    WF.print2stdout('Unable to get topwater from depth.', debug=self.Report.debug)
                    return []
                else:
                    # unrecognized convention, fall back to assuming elevation
                    WF.print2stdout('Unknown value for flag y_convention: {0}'.format(profile['y_convention']), debug=self.Report.debug)
                    WF.print2stdout('Please use "elevation"', debug=self.Report.debug)
                    WF.print2stdout('Assuming elevations...', debug=self.Report.debug)
                    return [yval[0] for yval in yvals]
            else:
                # no convention specified, default to elevation assumption
                WF.print2stdout('No value for flag y_convention', debug=self.Report.debug)
                WF.print2stdout('Assuming elevation...', debug=self.Report.debug)
                return [yval[0] for yval in yvals]

        elif 'h5file' in profile.keys() and 'ressimresname' in profile.keys():
            # external ResSim H5 profile: verify the file exists before opening it
            filename = profile['h5file']
            if not os.path.exists(filename):
                # can't proceed without the source file
                WF.print2stdout('ERROR: H5 file does not exist:', filename, debug=self.Report.debug)
                return []
            externalResSim = WRSS.ResSim_Results('', '', '', '', self.Report, external=True)
            # open the H5 file and load its time/subdomain metadata before reading
            externalResSim.openH5File(filename)
            externalResSim.load_time() #load time vars from h5
            externalResSim.loadSubdomains()
            # read the top-water elevation series for the requested result
            topwater = externalResSim.readProfileTopwater(profile['ressimresname'], timesteps)
            return topwater

        elif 'w2_segment' in profile.keys():
            # W2 profile top-water, only valid for a W2 model
            if self.Report.program.lower() != 'cequalw2':
                # model type mismatch, nothing to return
                return []
            # read the top-water elevation series for the requested W2 segment
            topwater = self.Report.ModelAlt.readProfileTopwater(profile['w2_segment'], timesteps)
            return topwater

        elif 'ressimresname' in profile.keys():
            # current-model ResSim profile top-water, only valid for a ResSim model
            if self.Report.program.lower() != 'ressim':
                # model type mismatch, nothing to return
                return []
            # read the top-water elevation series for the requested ResSim result
            topwater = self.Report.ModelAlt.readProfileTopwater(profile['ressimresname'], timesteps)
            return topwater

        # no recognized source type matched at all
        WF.print2stdout('No Data Defined for line', debug=self.Report.debug)
        WF.print2stdout('Profile:', profile, debug=self.Report.debug)
        return []

    def commitProfileDataToMemory(self, data, line_settings, object_settings):
        """
        Write already-processed profile data back into the memory cache.

        Used after profile data has been filtered/post-processed (so the
        cache reflects the final, report-ready values) rather than the
        raw values originally read by ``getProfileValues``. Only writes
        when the cache entry is missing or stale (i.e. was cached for a
        different set of timestamps than the object currently needs).

        Parameters
        ----------
        data : dict
            Dictionary of processed profile data keyed by line/flag.
        line_settings : dict
            Per-line settings dictionary; each entry must contain a
            ``'metadata'`` dict and a ``'logoutputfilename'`` memory key.
        object_settings : dict
            Settings for the current object; must contain
            ``'timestamps'`` giving the timestamps the data corresponds
            to.

        Returns
        -------
        None
            This function does not return a value; it updates
            ``self.Memory`` in place.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> organizer.commitProfileDataToMemory(data, line_settings, object_settings)
        """

        # process every processed profile line and decide whether it needs to be (re)cached
        for line in data.keys():
            # assume no write is needed until proven otherwise below
            write = False
            # Deep-copy everything so the cached copy can't be mutated by
            # later in-place edits to the caller's `data` dict.
            values = pickle.loads(pickle.dumps(data[line]['values'], -1))
            depths = pickle.loads(pickle.dumps(data[line]['depths'], -1))
            elevations = pickle.loads(pickle.dumps(data[line]['elevations'], -1))
            metadata = pickle.loads(pickle.dumps(line_settings[line]['metadata'], -1))
            # mark this cached entry as profile data for downstream consumers (e.g. writeDataFiles)
            metadata['isprofile'] = True
            datamem_key = line_settings[line]['logoutputfilename']
            if datamem_key not in self.Memory.keys():
                # nothing cached yet for this key, always write
                write = True
            else:
                # Only overwrite the cache if the timestamps it was
                # cached for don't match what's needed now.
                if not np.array_equal(object_settings['timestamps'], self.Memory[datamem_key]['times']):
                    # cached timestamps are stale, need to overwrite
                    write = True

            if write:
                # commit the processed (filtered/final) profile data into the cache
                self.Memory[datamem_key] = {'times': object_settings['timestamps'],
                                            'values': values,
                                            'elevations': elevations,
                                            'depths': depths,
                                            'metadata': metadata
                                            }

    #################################################################
    #Table Functions
    #################################################################

    def getTableDataDictionary(self, object_settings, type='timeseries'):
        """
        Build the data dictionary for a table's data paths.

        Supports two table data types:

        - ``'timeseries'``: each data path is read as a time series
          (reusing the same ``updateTimeSeriesDataDictionary`` logic as
          plots), with ``'computed'`` paths again expanded to one entry
          per accepted simulation ID.
        - ``'formatted'``: each data path points to an existing formatted
          text table file, which is read in as-is via
          ``WR.readFormattedTable_Pandas`` rather than being interpreted
          as a time series.

        Data paths missing a ``'flag'`` are assigned an auto-generated
        temporary flag (``FlagNNNNNN``) with a warning, since tables may
        not group/label correctly without a real flag.

        Parameters
        ----------
        object_settings : dict
            Currently selected object's settings dictionary; must contain
            a ``'datapaths'`` list.
        type : str, optional
            Either ``'timeseries'`` or ``'formatted'`` (default
            ``'timeseries'``), case-insensitive.

        Returns
        -------
        data : dict
            Dictionary of table data keyed by flag (time series dicts for
            ``'timeseries'`` type, pandas DataFrames for ``'formatted'``
            type).
        line_settings : dict
            Dictionary of settings/metadata for each flag in ``data``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, line_settings = organizer.getTableDataDictionary(object_settings)
        """

        # accumulator dicts, plus a counter used to name any auto-generated placeholder flags
        data = {}
        line_settings = {}
        temp_flag_number = 1
        # process every data path configured for this table
        for dp in object_settings['datapaths']:
            numtimesused = 0
            if 'flag' not in dp.keys():
                # Auto-generate a placeholder flag so processing can
                # continue, though grouping/labeling may not be ideal.
                WF.print2stdout('Flag not set for line (Computed/Observed/etc)', debug=self.Report.debug)
                # build a zero-padded temporary flag name and increment the counter
                temp_flag = f'Flag{str(temp_flag_number).zfill(6)}'
                WF.print2stdout(f'Using Temporary flag {temp_flag}, but table may not work as intended', debug=self.Report.debug)
                dp['flag'] = temp_flag
                temp_flag_number += 1
            if type.lower() == 'timeseries':
                if dp['flag'].lower() == 'computed':
                    # Expand into one entry per accepted simulation ID,
                    # same pattern as getTimeSeriesDataDictionary.
                    # loop over every accepted simulation ID for this report
                    for ID in self.Report.accepted_IDs:
                        # deep-copy this data path's settings so each ID gets its own independent config
                        cur_dp = pickle.loads(pickle.dumps(dp, -1))
                        cur_dp = self.Report.configureSettingsForID(ID, cur_dp)
                        cur_dp['numtimesused'] = numtimesused
                        cur_dp['ID'] = ID
                        if not self.Report.checkModelType(cur_dp):
                            # this ID's model type doesn't match, skip it
                            continue
                        # read and register this ID's version of the data path
                        data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, cur_dp)
                        if success:
                            numtimesused += 1
                            dp_used = True
                else:
                    if self.Report.currentlyloadedID != self.Report.base_id:
                        # currently on a comparison ID, reconfigure to the base ID's settings
                        dp = self.Report.configureSettingsForID(self.Report.base_id, dp)
                    else:
                        # already on the base ID, resolve model-specific placeholders directly
                        dp = WF.replaceflaggedValues(self.Report, dp, 'modelspecific')
                    dp['numtimesused'] = numtimesused
                    if not self.Report.checkModelType(dp):
                        # model type mismatch, skip this data path entirely
                        continue
                    # read and register this non-computed data path
                    data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, dp)
                    if success:
                        numtimesused += 1

            elif type.lower() == 'formatted':
                # Formatted table: read the file as-is (no time series
                # extraction) and just record its memory key/settings.
                dp = self.Report.configureSettingsForID(self.Report.base_id, dp)
                if 'filename' in dp.keys():
                    # read the pre-formatted table file directly into a dataframe
                    data[dp['flag']] = WR.readFormattedTable_Pandas(dp['filename'])
                line_settings[dp['flag']] = {}
                # compute the memory key for logging/tracing purposes
                datamem_key = self.buildMemoryKey(dp)
                line_settings[dp['flag']]['logoutputfilename'] = datamem_key
                # merge in all of this data path's original settings
                line_settings[dp['flag']].update(dp)

        # if self.Report.memberiteration:
        #     line_settings = self.checkForIdenticalMembers(data, line_settings)

        return data, line_settings

    def filterFormattedTable(self, data, object_settings, primarykey=None):
        """
        Filter a formatted table's columns/rows based on user settings.

        Supports three independent filtering behaviors, applied in order:

        1. ``'headers'``: keep only the listed columns, dropping all
           others.
        2. ``'filters'``: keep only rows whose primary-key value is in
           the given list (optionally reformatted to the collection
           member naming convention first, if
           ``'formatprimaryascollection'`` is set).
        3. Forecast reports with ``'formatprimaryascollection'`` set:
           additionally drop any row whose primary-key value isn't one of
           the report's known forecast members.

        Parameters
        ----------
        data : pandas.DataFrame
            The table to filter.
        object_settings : dict
            Settings dictionary for the table object, checked for
            ``'headers'``, ``'filters'``, and
            ``'formatprimaryascollection'`` keys.
        primarykey : str, optional
            The table's primary key column; if not given, it is
            determined via ``getPrimaryTableKey``.

        Returns
        -------
        pandas.DataFrame
            The filtered table (modified/returned in place).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> filtered = organizer.filterFormattedTable(table, object_settings)
        """

        if data.empty:
            # nothing to filter on an empty table
            return data

        if primarykey == None:
            # no primary key given, determine one automatically
            primarykey = self.getPrimaryTableKey(data, object_settings)

        if 'headers' in object_settings.keys():
            # Drop every column not in the user-selected header list.
            selected_headers = object_settings['headers']
            columns = data.columns
            # remove every column that isn't in the requested header list
            for column in columns:
                if column not in selected_headers:
                    data.drop(column, axis=1, inplace=True)

        if 'filters' in object_settings.keys():
            # Drop every row whose primary-key value isn't in the
            # user-selected list (converting the selection list to the
            # collection member format first, if requested).
            selected_rows = object_settings['filters']
            if 'formatprimaryascollection' in object_settings.keys():
                if object_settings['formatprimaryascollection'].lower() == 'true':
                    # reformat each selected row value to match the collection member naming convention
                    selected_rows = [WF.formatMembers(n) for n in selected_rows] #match the table if theyve been converted

            # remove every row whose primary-key value isn't in the selected list
            for index, row in data.iterrows():
                if row[primarykey] not in selected_rows:
                    data.drop(index=index, inplace=True)

        if self.Report.reportType == 'forecast':
            # For forecast reports, additionally restrict rows to only
            # known forecast members when the primary key represents
            # ensemble members.
            if 'formatprimaryascollection' in object_settings.keys():
                if object_settings['formatprimaryascollection'].lower() == 'true':
                    # remove any row whose primary-key value isn't a recognized forecast member
                    for index, row in data.iterrows():
                        if row[primarykey] not in self.Report.allMembers:
                            data.drop(index=index, inplace=True)

        return data

    def mergeFormattedTables(self, data, data_settings, object_settings):
        """
        Merge multiple formatted tables into a single table on a common key.

        If only one table is present, it is returned unchanged. Otherwise,
        the merge column is either explicitly given
        (``object_settings['merge_on']``) or auto-detected as a column
        name common to every table being merged. Tables are merged
        pairwise (via ``pandas.merge``) in the order they appear in
        ``data``.

        Parameters
        ----------
        data : dict
            Dictionary of ``{flag: pandas.DataFrame}`` tables to merge.
        data_settings : dict
            Dictionary of settings/metadata per table flag; merged
            together into a single settings dict as tables are combined.
        object_settings : dict
            Settings for the current table object; may specify
            ``'merge_on'``.

        Returns
        -------
        main_table : pandas.DataFrame
            The merged table (or the sole table, if only one was given).
        main_data_settings : dict
            The merged settings dictionary.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> merged_table, merged_settings = organizer.mergeFormattedTables(data, data_settings, object_settings)
        """

        # collect the list of table flags in insertion order
        table_keys = list(data.keys())

        if len(table_keys) < 2: #todo: can this be zero?
            # only one table, nothing to merge
            main_table = data[table_keys[0]]
            main_data_settings = data_settings[table_keys[0]]

        else:
            if 'merge_on' in object_settings.keys():
                # user explicitly specified which column to merge tables on
                merge_on = object_settings['merge_on']
            else:
                # Auto-detect a column present in every table by counting
                # how many tables each column name appears in.
                # flatten every table's column list into a single list for counting
                common_keys = [list(data[n].columns) for n in table_keys]
                common_keys = list(itertools.chain(*common_keys))
                common_keys_count = Counter(common_keys)
                merge_on = None
                # find the first column name that appears in every single table
                for ckc in common_keys_count.keys(): #TODO: come back and improve this logic
                    if common_keys_count[ckc] == len(table_keys):
                        # this column appears in every table, use it as the merge key
                        merge_on = ckc
            if merge_on != None:
                # Merge every table into the first one, in order,
                # accumulating settings along the way.
                main_table = data[table_keys[0]]
                main_data_settings = data_settings[table_keys[0]]
                # merge each remaining table into the running main table in turn
                for tk in table_keys[1:]:
                    main_table = pd.merge(main_table, data[tk], on=merge_on)
                    main_data_settings.update(data_settings[tk])
                #Todo: does the flag matter?

            else:
                # No common column found; fall back to just using the
                # first table unmerged.
                WF.print2stdout('Unable to find common key in tables.', debug=self.Report.debug)
                main_table = data[table_keys[0]]
                main_data_settings = data_settings[table_keys[0]]

        return main_table, main_data_settings

    def getPrimaryTableKey(self, data, object_settings):
        """
        Determine the primary (row-identifying) key column for a table.

        Priority order: an explicit ``'primarykey'`` setting, then an
        explicit ``'merge_on'`` setting, then (as a last resort) the
        first column of the table/first table in a dict of tables, with
        a warning logged recommending the user specify ``'primarykey'``
        explicitly.

        Parameters
        ----------
        data : dict or pandas.DataFrame
            Either a single formatted table, or a dict of
            ``{flag: pandas.DataFrame}`` tables.
        object_settings : dict
            Settings dictionary for the table object.

        Returns
        -------
        str
            The name of the column to treat as the primary key.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> primarykey = organizer.getPrimaryTableKey(data, object_settings)
        """

        if 'primarykey' in object_settings:
            # explicit primary key given, use it directly
            primarykey = object_settings['primarykey']
        elif 'merge_on' in object_settings.keys():
            # fall back to the merge column, if one was defined
            primarykey = object_settings['merge_on']
        else:
            # Fall back to the first column of the first table found.
            if isinstance(data, dict):
                # dict of tables, grab the columns from the first one
                firstkey = list(data.keys())[0]
                primarykey = list(data[firstkey].columns)
                if len(primarykey) == 0:
                    # no columns found at all, fall back to using the table's flag name itself
                    primarykey = firstkey
                else:
                    # take the very first column as the primary key
                    primarykey = primarykey[0]
            elif isinstance(data, pd.DataFrame):
                # single dataframe, just take its first column
                primarykey = list(data.columns)[0]
            WF.print2stdout('Unable to establish table primary key based on input.', debug=self.Report.debug)
            WF.print2stdout('To fix, specify a "primarykey" flag in the input file.', debug=self.Report.debug)
            WF.print2stdout(f'Using first column, {primarykey}.', debug=self.Report.debug)
        return primarykey

    #################################################################
    #Contour Functions
    #################################################################

    def getContours(self, settings):
        """
        Read longitudinal contour data (values vs. distance vs. time).

        Contours represent a 2-D grid of values (e.g. temperature) along
        a river reach (distance) over time, sourced from either a ResSim
        subdomain or a W2 model segment. Checks memory first, and
        re-reads if the cached entry used a different reporting interval
        than requested.

        Parameters
        ----------
        settings : dict
            Object settings describing the contour's data source
            (``'ressimresname'``+``'parameter'`` or
            ``'w2_file'``+``'parameter'``), plus an optional
            ``'interval'`` for resampling.

        Returns
        -------
        times : list or numpy.ndarray
            Timestamps for the contour grid.
        values : numpy.ndarray
            2-D array of contour values (distance x time, or similar).
        distance : list or numpy.ndarray
            Distance/station values along the reach.
        metadata : dict
            Metadata about the read (units, source, whether it came from
            memory, etc.).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> times, values, distance, metadata = organizer.getContours(settings)
        """

        # base metadata dict describing this contour read
        metadata = {'iscontour': True,
                    'units': None,
                    'frommemory': False,
                    'interval': None}

        # compute the memory key for this contour source and set up empty defaults
        datamem_key = self.buildMemoryKey(settings)
        values, distance, times = [], [], []

        if datamem_key in self.Memory.keys():
            WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
            # pull a deep copy of the cached entry so we can inspect/modify it safely
            datamem_entry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
            times = datamem_entry['dates']
            values = datamem_entry['values']
            metadata = datamem_entry['metadata']
            distance = datamem_entry['distance']
            if 'interval' in settings.keys():
                if settings['interval'].lower() != metadata['interval']:
                    # cached interval doesn't match what's requested now, force a re-read
                    WF.print2stdout('incorrect interval in memory. Re-extracting..', debug=self.Report.debug)
                    metadata['frommemory'] = False

        if not metadata['frommemory']:
            if 'ressimresname' in settings.keys(): #Ressim subdomain
                # Confirm the requested subdomain actually exists in this
                # model before attempting to read it.
                checkdomain = self.Report.ModelAlt.checkSubdomain(settings['ressimresname'])
                if not checkdomain:
                    # subdomain doesn't exist, nothing to return
                    return [], [], [], metadata
                # read the actual contour grid for this subdomain/parameter
                times, values, distance = self.Report.ModelAlt.readSubdomain(settings['parameter'],
                                                                             settings['ressimresname'])
        elif 'w2_file' in settings.keys():
            # read the contour grid for a W2 model segment
            times, values, distance = self.Report.ModelAlt.readSegment(settings['w2_file'],
                                                                       settings['parameter'])

        if 'interval' in settings.keys():
            # resample the contour grid's time axis to the requested interval
            times, values = WT.changeTimeSeriesInterval(times, values, settings, self.Report.startYear)

        # cache a deep copy of everything read for future requests
        self.Memory[datamem_key] = {'dates': pickle.loads(pickle.dumps(times, -1)),
                                    'values': pickle.loads(pickle.dumps(values, -1)),
                                    'distance': pickle.loads(pickle.dumps(distance, -1)),
                                    'metadata': pickle.loads(pickle.dumps(metadata, -1))}

        return times, values, distance, metadata

    def getContourDataDictionary(self, settings):
        """
        Build the data dictionary for all reaches in a contour plot.

        For every reach and every accepted simulation ID, reads the
        contour data (via ``getContours``) and stores it under a unique
        flag, optionally scaling the distance axis by a user-supplied
        ``'y_scalar'``. Resets the active simulation ID back to the
        report's base ID once done.

        Parameters
        ----------
        settings : dict
            Object settings dictionary containing a ``'reaches'`` list
            and optionally ``'y_scalar'``.

        Returns
        -------
        data : dict
            Dictionary of contour data (values, dates) keyed by
            (unique) flag.
        data_settings : dict
            Dictionary of settings/metadata (including ``'distance'``)
            for each flag in ``data``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, data_settings = organizer.getContourDataDictionary(settings)
        """

        # accumulator dicts built up across every reach/simulation ID combination
        data = {}
        data_settings = {}
        # missing = []
        # process every reach configured for this contour plot
        for reach in settings['reaches']:
            # loop over every accepted simulation ID for this report
            for ID in self.Report.accepted_IDs:
                # deep-copy this reach's settings so each ID gets its own independent config
                curreach = pickle.loads(pickle.dumps(reach, -1))
                curreach = self.Report.configureSettingsForID(ID, curreach)
                if not self.Report.checkModelType(curreach):
                    # this ID's model type doesn't match, skip it
                    continue
                # read the contour grid for this reach/simulation combination
                dates, values, distance, metadata = self.getContours(curreach)
                if WF.checkData(values):
                    # Determine the flag to store this ID's data under,
                    # disambiguating with a numeric suffix if needed.
                    if 'flag' in reach.keys():
                        # explicit flag given, use it
                        flag = reach['flag']
                    elif 'label' in reach.keys():
                        # fall back to the label if no flag was given
                        flag = reach['label']
                    else:
                        # no flag or label at all, auto-generate one from the simulation ID
                        flag = 'reach_{0}'.format(ID)
                    if flag in data.keys():
                        # flag already used by another ID, disambiguate with an incrementing suffix
                        count = 1
                        newflag = flag + '_{0}'.format(count)
                        while newflag in data.keys():
                            count += 1
                            newflag = flag + '_{0}'.format(count)
                        WF.print2stdout('The current flag is {0}'.format(flag), debug=self.Report.debug)
                        flag = newflag
                        WF.print2stdout('The new flag is {0}'.format(newflag), debug=self.Report.debug)
                    # compute the memory key for logging/tracing purposes
                    datamem_key = self.buildMemoryKey(reach)

                    if 'y_scalar' in settings.keys():
                        # Rescale the distance axis (e.g. m to km).
                        y_scalar = float(settings['y_scalar'])
                        distance *= y_scalar

                    # record the distance axis, simulation ID, and memory key for this flag
                    data_settings[flag] = {'distance': distance,
                                           'ID': ID,
                                           'logoutputfilename': datamem_key}

                    # store the actual contour arrays under this flag
                    data[flag] = {'values': values,
                                  'dates': dates,
                                  'ID': ID}

                    # merge in the metadata read from source, then the original reach settings on top
                    data_settings[flag].update(metadata)
                    data_settings[flag].update(reach)
                # else:
                    # missing.append(curreach['flag'])

        #reset
        # restore the active simulation ID once we're done looping over every accepted ID
        self.Report.loadCurrentID(self.Report.base_id)
        self.Report.loadCurrentModelAltID(self.Report.base_id)
        return data, data_settings

    #################################################################
    #Gate Functions
    #################################################################

    def getGateDataDictionary(self, settings, makecopy=True):
        """
        Build the data dictionary for gate operation plots.

        Gate operation settings are grouped ("gateops"), each containing
        one or more individual gates. For every gate-operation group,
        every gate's time series is read (via ``getTimeSeries``), with
        zero values converted to NaN (so closed/inactive gates don't draw
        as a flat line at zero). Gates are grouped under their parent
        gate-operation's flag/label so they can be plotted together.

        Parameters
        ----------
        settings : dict
            Object settings dictionary, expected to contain a
            ``'gateops'`` list, each with a ``'gates'`` list of
            individual gate data-source settings.
        makecopy : bool, optional
            Passed through to ``getTimeSeries``; set ``False`` to skip
            deep-copying cached data for speed (default ``True``).

        Returns
        -------
        data : dict
            Nested dictionary of the form
            ``{gateop_key: {'gates': {gate_flag: {'values':..., 'dates':...}}}}``.
        line_data : dict
            Matching nested dictionary of settings/metadata for each gate,
            including a ``'gategroup'`` tag identifying which
            gate-operation it belongs to.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, line_data = organizer.getGateDataDictionary(settings)
        """

        # accumulator dicts, nested by gate-operation group then by individual gate
        data = {}
        line_data = {}
        if 'gateops' in settings.keys():
            # process every gate-operation group defined for this object
            for gi, gateop in enumerate(settings['gateops']):

                # Determine the key to group this gate-operation's gates
                # under: explicit flag, explicit label, or an
                # auto-numbered fallback.
                if 'flag' in gateop.keys():
                    if gateop['flag'] not in data.keys():
                        # first time seeing this flag, initialize its entry
                        data[gateop['flag']] = {}
                        line_data[gateop['flag']] = {}
                        gateopkey = gateop['flag']
                elif 'label' in gateop.keys():
                    if gateop['label'] not in data.keys():
                        # first time seeing this label, initialize its entry
                        data[gateop['label']] = {}
                        line_data[gateop['flag']] = {}
                        gateopkey = gateop['label']
                else:
                    if 'GATEOP_{0}'.format(gi) not in data.keys():
                        # no flag or label given, auto-generate one from the group index
                        gateopkey = 'GATEOP_{0}'.format(gi)
                        data[gateopkey] = {}
                        line_data[gateopkey] = {}

                # set up the nested 'gates' sub-dict for this group
                data[gateopkey]['gates'] = {}
                line_data[gateopkey]['gates'] = {}
                # process every individual gate within this gate-operation group
                for gate in gateop['gates']:
                    # read this gate's time series (units/metadata discarded here)
                    dates, values, _ = self.getTimeSeries(gate, makecopy=makecopy)
                    if 'flag' in gate.keys():
                        # explicit flag given for this gate, use it
                        flag = gate['flag']
                    else:
                        # no flag given, fall back to a generic placeholder
                        flag = 'gate'
                    # Disambiguate duplicate gate flags within this group.
                    if flag in data[gateopkey]['gates'].keys():
                        # flag already used within this group, disambiguate with an incrementing suffix
                        count = 1
                        newflag = flag + '_{0}'.format(count)
                        while newflag in data[gateopkey]['gates'].keys():
                            count += 1
                            newflag = flag + '_{0}'.format(count)
                        flag = newflag
                    # compute the memory key for logging/tracing purposes
                    datamem_key = self.buildMemoryKey(gate)
                    # Treat a gate value of exactly 0 as "not operating"
                    # (NaN) rather than plotting a flat zero line.
                    value_msk = np.where(values==0)
                    values[value_msk] = np.nan
                    if 'flag' in gateop.keys():
                        # tag this gate with its parent gate-operation's flag
                        gategroup = gateop['flag']
                    else:
                        # no flag on the parent group, fall back to an auto-generated group tag
                        gategroup = 'gategroup_{0}'.format(gi)
                    # store the actual gate values/dates under this flag
                    data[gateopkey]['gates'][flag] = {'values': values,
                                                      'dates': dates}

                    # record the memory key and group tag for this gate
                    line_data[gateopkey]['gates'][flag] = {'logoutputfilename': datamem_key,
                                                          'gategroup': gategroup}

                    # fill in any remaining gate settings not already captured above
                    for key in gate.keys():
                        if key not in line_data[gateopkey]['gates'][flag].keys():
                            line_data[gateopkey]['gates'][flag][key] = gate[key]

                # Copy any gate-operation-level (group) settings down onto
                # both the data and line_data dicts for this group.
                for key in gateop.keys():
                    if key not in data[gateopkey].keys():
                        data[gateopkey][key] = gateop[key]
                    if key not in line_data[gateopkey].keys():
                        line_data[gateopkey][key] = gateop[key]

        return data, line_data

    #################################################################
    #Logging Functions
    #################################################################

    def writeDataFiles(self):
        """
        Write every cached data entry out to a CSV file for review.

        Iterates ``self.Memory`` and writes one CSV per cached entry
        (named after its sanitized memory key) into
        ``self.Report.CSVPath``, so the exact data used to build each
        plot/table can be inspected after the report runs. Handles three
        shapes of cached data: profiles (values vs. elevation/depth),
        contours (skipped -- too slow to write per-run), and general time
        series (1-D, 2-D/multi-series, or per-member dict).

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value; it writes CSV files
            to disk as a side effect.

        Raises
        ------
        None
            This function does not propagate exceptions; any error while
            writing a given entry is caught, logged, and skipped so one
            failure doesn't halt the rest of the export.

        Examples
        --------
        >>> organizer.writeDataFiles()
        """

        # write one CSV per cached memory entry
        for key in self.Memory.keys():
            # sanitize the memory key so it's safe to use as a filename
            cleankey = WF.cleanFileName(key)
            csv_name = os.path.join(self.Report.CSVPath, '{0}.csv'.format(cleankey))
            try:
                metadata = self.Memory[key]['metadata']
                if 'isprofile' in metadata:
                    # if self.Memory[key]['isprofile'] == True:
                    if metadata['isprofile'] == True:
                        # Profile data: build a table of Dates/Values/
                        # Elevations/Depths, aligning ragged per-timestep
                        # arrays into flat columns via matcharrays/getListItems.
                        alltimes = self.Memory[key]['times']
                        allvalues = self.Memory[key]['values']
                        # expand the times array to match the (possibly ragged) shape of the values array
                        alltimes = WF.matcharrays(alltimes, allvalues)
                        allelevs = self.Memory[key]['elevations']
                        alldepths = self.Memory[key]['depths']
                        if len(allelevs) == 0: #elevations may not always fall out
                            # no elevations available, derive matching-shaped placeholders from depths instead
                            allelevs = WF.matcharrays(allelevs, alldepths)
                        units = metadata['units']
                        # flatten each ragged array into a single flat list of items for the CSV columns
                        values = WF.getListItems(allvalues)
                        times = WF.getListItems(alltimes)
                        elevs = WF.getListItems(allelevs)
                        depths = WF.getListItems(alldepths)
                        if isinstance(values, (list, np.ndarray)):
                            # single flat series, build a simple 4-column dataframe
                            df = pd.DataFrame({'Dates': times, 'Values ({0})'.format(units): values, 'Elevations': elevs,
                                               'Depths': depths})
                        elif isinstance(values, dict):
                            # per-member/key profile data, build one set of columns per key
                            colvals = {}
                            colvals['Dates'] = times
                            # build the value/elevation/depth columns for each member key
                            for key in values:
                                colvals[key] = values[key]
                                colvals[key] = elevs[key]
                                colvals[key] = depths[key]
                            df = pd.DataFrame(colvals)
                elif 'iscontour' in metadata.keys():
                    continue #were not doing this for now, takes ~ 5 seconds per 3yr reach..

                    # if self.Data.Memory[key]['iscontour'] == True:
                    #     alltimes = self.Data.Memory[key]['dates']
                    #     allvalues = self.Data.Memory[key]['values'].T #this gets transposed a few times.. we want distance/date
                    #     alldistance = self.Data.Memory[key]['distance']
                    #     times = WF.matcharrays(alltimes, allvalues)
                    #     distances = WF.matcharrays(alldistance, allvalues)
                    #     values = WF.getListItems(allvalues)
                    #     units = self.Data.Memory[key]['units']
                    #     newstime = time.time()
                    #     df = pd.DataFrame({'Dates': times, 'Values ({0})'.format(units): values, 'Distances': distances,
                    #                        })
                else:
                    # General time series: build a Dates/Values table,
                    # handling three possible value shapes below.
                    allvalues = self.Memory[key]['values']
                    alltimes = self.Memory[key]['times']
                    # metadata = self.Memory[key]['metadata']
                    units = metadata['units']
                    # flatten the times array into a plain list for the CSV
                    times = WF.getListItems(alltimes)
                    if isinstance(allvalues, (list, np.ndarray)):
                        # Determine if this is a single 1-D series or a
                        # collection of multiple series (2-D/nested).
                        multidimensional = False
                        if isinstance(allvalues, list):
                            # check the first element to see if it's itself a list/array (i.e. nested)
                            if len(allvalues) > 0:
                                if isinstance(allvalues[0], (list, np.ndarray)):
                                    multidimensional = True
                        else:
                            # numpy array case: check the shape directly for a second dimension
                            if len(allvalues.shape) == 2:
                                multidimensional = True
                        if not multidimensional:
                            # single flat series, build a simple 2-column dataframe
                            values = WF.getListItems(allvalues)
                            df = pd.DataFrame({'Dates': times, 'Values ({0})'.format(units): values})
                        else:
                            # Multiple series: one "Values N" column per
                            # sub-array.
                            df_dict = {'Dates': times}
                            # build one "Values N" column per nested sub-array
                            for vi, v in enumerate(allvalues):
                                values = WF.getListItems(v)
                                df_dict[f'Values {vi} ({units})'] = values
                            df = pd.DataFrame(df_dict)
                    elif isinstance(allvalues, dict):
                        # Forecast collection: one column per ensemble
                        # member, keyed by member name.
                        colvals = {'Dates': times}
                        # values = WF.getListItems(allvalues)
                        # build one column per ensemble member, with units appended to the header if known
                        for key, values in allvalues.items():
                            if units != None:
                                colvals[f'{key} ({units})'] = values
                            else:
                                colvals[f'{key}'] = values
                        df = pd.DataFrame(colvals)

                # write the assembled dataframe out to its CSV file
                df.to_csv(csv_name, index=False)

            except:
                # Never let a single failed CSV export crash report
                # generation; log the error and continue with the rest.
                WF.print2stdout(f'ERROR WRITING CSV FILE {csv_name}')
                WF.print2stdout(traceback.format_exc(), debug=self.Report.debug)

    def scaleValuesByTable(self, data, line_settings):
        """
        Scale a line's values using a lookup-table of scalars.

        For lines whose settings define both a ``'scalartable'`` (a file
        mapping specific "scale from" values to multiplier scalars) and a
        ``'scalefrom'`` data source, this reads the scale-from time
        series, matches it up in time with the line's own data if
        needed, then for every timestep where the scale-from value
        exactly matches one of the table's target values, multiplies the
        line's value at that timestep by the corresponding scalar. This
        matches on *specific* values only, not ranges.

        Parameters
        ----------
        data : dict
            Dictionary of ``{flag: {'values':..., 'dates':...}}`` line
            data to scale (modified in place).
        line_settings : dict
            Per-flag settings dictionary; entries with both
            ``'scalartable'`` and ``'scalefrom'`` keys are processed.

        Returns
        -------
        dict
            The ``data`` dictionary, with matching lines' values scaled
            in place, and the resolved scale-from series attached under
            ``data[flag][flag + '_scalefrom']`` for reference.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data = organizer.scaleValuesByTable(data, line_settings)
        """

        # process every line, scaling only those that define both a scalar table and a scale-from source
        for d in data.keys():
            if 'scalartable' in line_settings[d].keys() and 'scalefrom' in line_settings[d].keys():
                WF.print2stdout(f'Scalar table found for {d}', debug=self.Report.debug)
                # load the target-value -> scalar lookup table from disk
                tablevalues = WDR.readScalarTable(line_settings[d]['scalartable'])
                # build the key used to attach the resolved scale-from series back onto this line
                scalefromflag = line_settings[d]['flag']+'_scalefrom'
                # read the scale-from time series that drives the scaling decision
                scalefrom_dates, scalefrom_values, scalefrom_metadata = self.getTimeSeries(line_settings[d]['scalefrom'])
                if len(scalefrom_values) != len(data[d]['values']):
                    # Scale-from series and base series aren't aligned;
                    # match them up in time before comparing.
                    WF.print2stdout(f'Values and Scaledby in different time intervals. Equalizing..', debug=self.Report.debug)
                    # align the base line data and scale-from series onto shared timestamps
                    base_data, scaledFrom_data = WF.matchData({'dates': data[d]['dates'], 'values': data[d]['values']},
                                                         {'dates': scalefrom_dates, 'values': scalefrom_values})
                else:
                    # already aligned, no matching needed
                    base_data = data[d]
                    scaledFrom_data = {'values': scalefrom_values,
                                       'dates': scalefrom_dates,
                                       'metadata': scalefrom_metadata}

                #MAKE SURE NPARRAY
                # For every target value defined in the scalar table,
                # find every timestep where the scale-from series exactly
                # equals that target and apply the corresponding scalar.
                for target in tablevalues.keys():
                    if target in scaledFrom_data['values']:
                        # pull the scalar and find every timestep matching this target value exactly
                        scalar = tablevalues[target]
                        target_i = [i for i, n in enumerate(scaledFrom_data['values']) if n == target]
                        # apply the scalar to just those matching timesteps
                        base_data['values'][target_i] *= scalar

                # write the (possibly scaled) values back, and attach the scale-from series for reference
                data[d]['values'] = base_data['values']
                data[d][scalefromflag] = scaledFrom_data

        return data