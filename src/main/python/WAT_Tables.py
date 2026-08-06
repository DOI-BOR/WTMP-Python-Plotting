import datetime as dt
import pickle
import numpy as np
from matplotlib.colors import to_hex		# to_hex normalizes user-supplied color strings into #RRGGBB hex form for consistent output in the generated table XML.
from scipy import interpolate

import WAT_Functions as WF
import WAT_Time as WT


class Tables(object):
    """
    Helpers for building the report's statistics/data tables.

    Table objects in the report are built as a "table_constructor"
    dictionary of columns, each with a header, a list of "|"-delimited
    ``rowname|value`` row strings, and a parallel list of per-row
    threshold colors. This class provides the machinery to build those
    row/column templates (still containing unresolved ``%%stat%%``-style
    placeholders) from the report's settings, compute the actual
    statistic values from time series/profile data, apply threshold-
    based coloring/formatting, and finally write the finished table out
    to the report XML.

    Attributes
    ----------
    Report : object
        The main Report Generator instance this table helper serves.
    forecastTableHeaders : dict
        Mapping of recognized forecast-table column flags to their
        display labels; only set for forecast reports.
    """

    def __init__(self, Report):
        """
        Set up the table helper and (for forecast reports) its header definitions.

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
        >>> tables = Tables(Report)
        """
        # keep a reference back to the parent report for shared state
        self.Report = Report
        if self.Report.reportType == 'forecast':
            # forecast reports need the column display-name lookup for forecast tables
            self.defineForecastTableHeaders()

    def buildHeadersByTimestamps(self, timestamps, years):
        """
        Group profile timestamps into per-year header lists.

        Converts every timestamp to a comparable year value (handling
        both real ``datetime`` objects and raw Julian-date floats), then
        buckets them by the requested year(s).

        Parameters
        ----------
        timestamps : list
            List of available profile timestamps (datetimes or
            Julian-date floats).
        years : list
            Years (or ``'ALLYEARS'``-style entries) to filter/group by;
            each entry produces its own header list.

        Returns
        -------
        headers : list of list
            One list of matched timestamps (as datetimes or strings) per
            requested year.
        headers_i : list of list
            Parallel list of the original index into ``timestamps`` for
            each matched entry in ``headers``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> headers, headers_i = tables.buildHeadersByTimestamps(timestamps, [2020, 2021])
        """

        headers = []
        headers_i = []

        # bucket every timestamp into its matching requested year
        for year in years:
            h = []
            hi = []
            for ti, timestamp in enumerate(timestamps):
                if isinstance(timestamp, dt.datetime):
                    if year == timestamp.year:
                        h.append(timestamp)
                        hi.append(ti)

                elif isinstance(timestamp, float):
                    # Julian-date timestamps need conversion before their
                    # year can be compared.
                    ts_dt = WT.JDateToDatetime(timestamp, self.Report.startYear)
                    if year == ts_dt.year:
                        h.append(str(timestamp))
                        hi.append(ti)
            headers.append(h)
            headers_i.append(hi)

        return headers, headers_i

    def buildErrorStatsTable(self, object_settings, data_settings):
        """
        Build the header/row templates for an error statistics table.

        For non-comparison reports, the configured headers/rows are used
        as-is. For comparison reports, every configured header
        (typically containing a ``%%<data flag>%%`` placeholder like
        ``%%Computed%%``) gets expanded into one real header per data
        source found in ``data_settings`` (excluding "Observed"), with
        each row's corresponding value column duplicated to match.

        Parameters
        ----------
        object_settings : dict
            Object settings dictionary; must contain ``'rows'`` and
            (for comparison reports) ``'headers'``.
        data_settings : dict
            Per-data-source settings dictionary, each entry expected to
            have a ``'flag'`` and optionally ``'label'``/``'ID'``.

        Returns
        -------
        headers : list of str
            Resolved column headers.
        rows : list of str
            ``"rowname|value1|value2|..."`` row template strings, with
            one value slot per resolved header.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> headers, rows = tables.buildErrorStatsTable(object_settings, data_settings)
        """

        headers = []
        rows = []
        # seed each row with just its row-name (before the first '|'), value columns are appended below
        for ri, row in enumerate(object_settings['rows']):
            rows.append(row.split('|')[0])

        if self.Report.iscomp: #comp run
            # ===================== Comparison run: expand each configured header once per data source =====================
            for i, header in enumerate(object_settings['headers']):
                curheader = pickle.loads(pickle.dumps(header, -1))
                for datakey in data_settings.keys():
                    ds = data_settings[datakey]
                    dk_flag = ds['flag']
                    dk_keys = ds.keys()
                    isused = False
                    if '%%{0}%%'.format(dk_flag) in curheader:
                        #like %%Computed%%%%SimulationName%%, %%Observed%%, etc..
                        # This header explicitly names this data source's
                        # flag: resolve it to a real display name.
                        if 'label' in dk_keys: #if theres a label, just use that, easy
                            # curheader = curheader.replace('%%{0}%%'.format(dk_flag), ds['label'])
                            curheader = ds['label']
                        elif 'ID' in dk_keys: #otherwise we will go find the settings and search for flags that are model spec
                            # switch to this simulation ID temporarily to resolve its name
                            ID = ds['ID']
                            # curheader = self.Report.configureSettingsForID(ID, curheader)
                            self.Report.loadCurrentID(ID)
                            curheader = self.Report.SimulationName
                        else: #if there are none, just remove the flag and we will add the flag based off of the flag :thumbsup:
                            #example "%%Computed%% Computed"
                            #this is less than ideal for comparison plots and I hope doesnt really happen, but I have to catch it
                            curheader = curheader.replace('%%{0}%%'.format(dk_flag), '')
                        headers.append(curheader)
                        isused = True

                    else:
                        #if the headers dont call out a flag, we need to build these smarter..
                        # No explicit flag referenced in the configured
                        # header: build one automatically for every
                        # non-Observed data source.
                        if dk_flag.lower() != 'observed': #ignore the observed data for error stat plots
                            if 'label' in dk_keys: #if theres a label, just use that, easy
                                curheader = ds['label']
                            elif 'ID' in dk_keys: #otherwise we will go find the settings and search for flags that are model spec
                                ID = ds['ID']
                                self.Report.loadCurrentID(ID)
                                curheader = self.Report.SimulationName
                            else: #if nothing else...
                                curheader = datakey
                            headers.append(curheader)
                            isused = True

                    if isused:
                        # For every resolved header, duplicate this row's
                        # matching value-column template into a new
                        # column tagged with this specific data source's
                        # key, so it can be resolved later per data
                        # source.
                        for ri, row in enumerate(object_settings['rows']):
                            srow = row.split('|')[1:][i]
                            rows[ri] += '|{0}'.format(srow.replace(dk_flag, datakey))

        else:
            # non-comparison report, use the configured headers/rows directly
            headers = object_settings['headers']
            rows = object_settings['rows']

        return headers, rows

    def buildMonthlyStatsTable(self, object_settings, data_settings):
        """
        Build the header/row templates for a monthly statistics table.

        Same general expansion pattern as ``buildErrorStatsTable`` for
        comparison reports, but iterates by ROW (since monthly tables
        put months/stats in rows) rather than by header.

        Parameters
        ----------
        object_settings : dict
            Object settings dictionary; must contain ``'rows'`` and
            optionally ``'headers'``.
        data_settings : dict
            Per-data-source settings dictionary, each entry expected to
            have a ``'flag'`` and optionally ``'label'``/``'ID'``.

        Returns
        -------
        headers : list of str
            Resolved column headers.
        rows : list of str
            ``"rowname|value1|value2|..."`` row template strings.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> headers, rows = tables.buildMonthlyStatsTable(object_settings, data_settings)
        """

        headers = []
        rows = []
        # seed each row with just its row-name, value columns are appended below
        for ri, row in enumerate(object_settings['rows']):
            rows.append(row.split('|')[0])
        if 'headers' in object_settings.keys():
            assigned_headers = object_settings['headers']
        else:
            assigned_headers = []
        #unlike error tables, we need to build by row here..

        if self.Report.iscomp: #comp run
            # ===================== Comparison run: expand each row's value columns per data source =====================
            for ri, row in enumerate(object_settings['rows']):
                srows = row.split('|')[1:]
                for sri, srow in enumerate(srows):
                    if len(assigned_headers) >= sri+1:
                        assigned_header = assigned_headers[sri]
                    else:
                        assigned_header = None
                    for datakey in data_settings:
                        ds = data_settings[datakey]
                        dk_flag = ds['flag']
                        dk_keys = ds.keys()
                        isused = False
                        if dk_flag in srow.split('.'):
                            # This value cell references the current data
                            # source's flag: resolve the header for it,
                            # preferring an explicit assigned header if
                            # one names this flag.
                            if assigned_header != None:
                                if f'%%{dk_flag}%%' in assigned_header: #check assigned headers first
                                    curheader = assigned_header.replace(f'%%{dk_flag}%%', '')
                                    if 'ID' in dk_keys:
                                        ID = ds['ID']
                                        # curheader = self.Report.configureSettingsForID(ID, curheader)
                                        self.Report.loadCurrentID(ID)
                                        curheader = self.Report.SimulationName
                                    if curheader not in headers:
                                        headers.append(curheader)
                                    isused = True
                            if not isused:
                                if 'label' in dk_keys: #if theres a label, just use that, easy
                                    curheader = ds['label']
                                elif 'ID' in dk_keys: #otherwise we will go find the settings and search for flags that are model spec
                                    ID = ds['ID']
                                    self.Report.loadCurrentID(ID)
                                    curheader = self.Report.SimulationName
                                else: #if nothing else...
                                    curheader = datakey
                                if curheader not in headers:
                                    headers.append(curheader)
                                isused = True
                        if isused:
                            rows[ri] += '|{0}'.format(srow.replace(dk_flag, datakey))

        else:
            # non-comparison report, use the configured headers/rows directly
            headers = object_settings['headers']
            rows = object_settings['rows']

        return headers, rows

    def buildSingleStatTable(self, object_settings, data):
        """
        Build the header/row templates for a single-statistic (year x data-source) table.

        Determines how many "data flags" the requested statistic needs
        (mean/count need only one series; comparison stats like RMSE
        need a computed + observed pair), resolves the column headers
        (data source labels for comparison reports, month names
        otherwise), and builds one row per year with
        ``%%stat.<key>.MONTH=<month>%%``-style placeholder cells to be
        resolved later.

        Parameters
        ----------
        object_settings : dict
            Object settings dictionary; must contain ``'statistic'`` and
            ``'years'``, and optionally ``'headers'``/
            ``'missingmarker'``.
        data : dict
            Per-data-source settings dictionary, each entry expected to
            have a ``'flag'`` and optionally ``'label'``/``'ID'``.

        Returns
        -------
        headers : list of str
            Resolved column headers (data source names, or month names).
        rows : list of str
            ``"year|value1|value2|..."`` row template strings with one
            value slot per month (and, if two flags are needed, one per
            computed/observed pairing).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> headers, rows = tables.buildSingleStatTable(object_settings, data)
        """

        headers = []
        rows = []
        months = [n for n in self.Report.Constants.mo_str_3]
        stat = object_settings['statistic']
        datakeys = list(data.keys())
        if len(datakeys) == 0:
            hasdata = False
        else:
            hasdata = True

        # Some statistics (mean, count) only need a single series;
        # comparison-style statistics (RMSE, bias, etc.) need a
        # computed/observed pair.
        if stat in ['mean', 'count']:
            numflagsneeded = 1
        else:
            numflagsneeded = 2

        if len(datakeys) < numflagsneeded:
            # not enough data sources to compute this statistic at all
            hasdata = False

        if self.Report.iscomp:
            # ===================== Comparison run: resolve column headers per data source =====================
            if 'headers' in object_settings.keys():
                hdrs = object_settings['headers']
                for curheader in hdrs:
                    isused = False
                    for datakey in datakeys:
                        if '%%{0}%%'.format(data[datakey]['flag']) in curheader:
                            if 'ID' in data[datakey].keys():
                                ID = data[datakey]['ID']
                                # tmpheader = self.Report.configureSettingsForID(ID, curheader)
                                self.Report.loadCurrentID(ID)
                                tmpheader = self.Report.SimulationName
                            else:
                                tmpheader = pickle.loads(pickle.dumps(curheader, -1))
                            tmpheader = tmpheader.replace('%%{0}%%'.format(data[datakey]['flag']), '')
                            headers.append(tmpheader)
                            isused = True
                    if not isused:
                        if '%%' in curheader: #check for unused flags.. if theyre there, move on
                            continue
                        else:
                            headers.append(curheader)

            else:
                # No explicit headers configured: auto-build one per
                # data source (excluding "Observed" when a computed/
                # observed pairing is needed, since Observed is the
                # comparison reference rather than its own column).
                for datakey in datakeys:
                    if 'label' in data[datakey].keys():
                        if numflagsneeded == 2:
                            if data[datakey]['flag'].lower() != 'observed':
                                headers.append(data[datakey]['label'])
                        else:
                            headers.append(data[datakey]['label'])
                    elif 'ID' in data[datakey].keys():
                        if numflagsneeded == 2:
                            if data[datakey]['flag'].lower() != 'observed':
                                ID = data[datakey]['ID']
                                self.Report.loadCurrentID(ID)
                                headers.append(self.Report.SimulationName)
                    else:
                        if numflagsneeded == 2:
                            if data[datakey]['flag'].lower() != 'observed':
                                headers.append(datakey)
                            else:
                               continue
                        else:
                            headers.append(datakey)
        else:
            # non-comparison report, use plain month names as headers
            headers = months

        # Split data sources into "computed" and "observed" groups
        # (inferring one group from the other if only one is explicitly
        # flagged), used below to pair them up for two-flag statistics.
        computed_keys = []
        observed_keys = []
        if hasdata:
            for datakey in datakeys:
                if data[datakey]['flag'].lower() == 'computed': #get the easy ones
                    computed_keys.append(datakey)
                elif data[datakey]['flag'].lower() == 'observed':
                    observed_keys.append(datakey)
            if len(computed_keys) == 0 and len(observed_keys) > 0: #if none are computed and some are observed, assume the rest computed
                for datakey in datakeys:
                    if data[datakey]['flag'] not in observed_keys:
                        computed_keys.append(datakey)
            if len(observed_keys) == 0 and len(computed_keys) > 0: #if some are computed and none are observed, assume the rest computed
                for datakey in datakeys:
                    if data[datakey]['flag'] not in computed_keys:
                        observed_keys.append(datakey)

        if 'missingmarker' in object_settings.keys():
            missingmarker = object_settings['missingmarker']
        else:
            missingmarker = '-'

        # ===================== One row per year, one value-cell per month =====================
        for year in object_settings['years']:
            row = f'{year}'
            for month in months:
                if not hasdata:
                    # no usable data sources at all, mark every cell as missing
                    row += f'|{missingmarker}'
                else:
                    if numflagsneeded == 1:
                        # single-series statistic, one cell per computed data source
                        for datakey in computed_keys:
                            row += f'|%%{stat}.{datakey}.MONTH={month.upper()}%%'
                    else:
                        if len(computed_keys) == 0:
                            row += f'|{missingmarker}'
                        else:
                            # two-series statistic, one cell per computed/observed pairing
                            for cflag in computed_keys:
                                if len(observed_keys) == 0:
                                    row += f'|{missingmarker}'
                                else:
                                    for oflag in observed_keys:
                                        row += f'|%%{stat}.{cflag}.MONTH={month.upper()}.{data[oflag]["flag"]}.MONTH={month.upper()}%%'
            rows.append(row)

        return headers, rows

    def buildFormattedTable(self, data):
        """
        Convert a pandas DataFrame into header/row strings for the table writer.

        Parameters
        ----------
        data : pandas.DataFrame
            The table to convert.

        Returns
        -------
        headers : pandas.Index
            The DataFrame's column names.
        rows : list of str
            One ``"|"``-delimited string per row, values in column order.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> headers, rows = tables.buildFormattedTable(data)
        """

        headers = data.columns #ez
        rows = []
        # build one "|"-delimited row string per dataframe row
        for i, row in data.iterrows():
            built_row = ''
            for rowval in row.values:
                if built_row == '':
                    # first value in the row, no leading separator needed
                    built_row = str(rowval)
                else:
                    built_row += f'|{rowval}'
            rows.append(built_row)
        return headers, rows

    def buildProfileStatsTable(self, object_settings, timestamp, data):
        """
        Build the header/row templates for a profile statistics table at one timestamp.

        Same header-expansion pattern as ``buildErrorStatsTable``, but
        for a single fixed timestamp per call (used once per profile
        timestamp header by the calling report method).

        Parameters
        ----------
        object_settings : dict
            Object settings dictionary; must contain ``'rows'`` and
            (for comparison reports) ``'headers'``.
        timestamp : str
            The timestamp this table is being built for; used directly
            as the single header for non-comparison reports.
        data : dict
            Per-data-source settings dictionary, each entry expected to
            have a ``'flag'`` and optionally ``'ID'``.

        Returns
        -------
        headers : list
            Resolved column headers.
        rows : list of str
            ``"rowname|value1|value2|..."`` row template strings.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> headers, rows = tables.buildProfileStatsTable(object_settings, timestamp, data)
        """

        headers = []
        rows = []
        # seed each row with just its row-name, value columns are appended below
        for ri, row in enumerate(object_settings['rows']):
            rows.append(row.split('|')[0])

        if self.Report.iscomp: #comp run
            # expand each configured header once per matching data source, same pattern as buildErrorStatsTable
            for i, header in enumerate(object_settings['headers']):
                curheader = pickle.loads(pickle.dumps(header, -1))
                for datakey in data.keys():
                    if '%%{0}%%'.format(data[datakey]['flag']) in curheader: #found data specific flag
                        if 'ID' in data[datakey].keys():
                            ID = data[datakey]['ID']
                            tmpheader = self.Report.configureSettingsForID(ID, curheader)
                        else:
                            tmpheader = pickle.loads(pickle.dumps(curheader, -1))
                        tmpheader = tmpheader.replace('%%{0}%%'.format(data[datakey]['flag']), '')
                        headers.append(tmpheader)
                        for ri, row in enumerate(object_settings['rows']):
                            srow = row.split('|')[1:][i]
                            rows[ri] += '|{0}'.format(srow.replace(data[datakey]['flag'], datakey))


        else: #single run
            # non-comparison report, the timestamp itself is the only header needed
            headers = [timestamp]
            rows = object_settings['rows']

        return headers, rows

    def filterTableData(self, data, object_settings):
        """
        NaN-out table time series values outside configured x/y limits or matching omit values.

        Parameters
        ----------
        data : dict
            Dictionary of data keyed by flag, each with ``'values'``
            (array or per-member dict) and ``'dates'``.
        object_settings : dict
            Settings for the current object; checked for ``'xlims'``,
            ``'ylims'``, and (per-line, falling back to object-level)
            ``'filterbylimits'``/``'omitvalue'``/``'omitvalues'``.

        Returns
        -------
        dict
            The ``data`` dictionary, with out-of-range/omitted values
            set to NaN in place.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        Marked with a ``#TODO: make this iterative so its less ugly for
        dicts`` comment in the original source.

        Examples
        --------
        >>> data = tables.filterTableData(data, object_settings)
        """

        xmax = None
        xmin = None
        ymax = None
        ymin = None

        if 'xlims' in object_settings.keys():
            if 'max' in object_settings['xlims'].keys():
                # xmax = float(object_settings['xlims']['max'])
                if object_settings['xlims']['max'] is None:
                    # no explicit value, fall back to the report's overall end time
                    xmax = self.Report.EndTime
                else:
                    xmax = WF.updateFlaggedValues(object_settings['xlims']['max'],'%%year%%', str(max(self.Report.years)))
                    xmax = WT.translateDateFormat(xmax, 'datetime', self.Report.EndTime, self.Report.StartTime,
                                                  self.Report.EndTime, debug=self.Report.debug)

            if 'min' in object_settings['xlims'].keys():
                # xmin = float(object_settings['xlims']['min'])
                if object_settings['xlims']['min'] is None:
                    # no explicit value, fall back to the report's overall start time
                    xmin = self.Report.StartTime
                else:
                    xmin = WF.updateFlaggedValues(object_settings['xlims']['min'],'%%year%%', str(min(self.Report.years)))
                    xmin = WT.translateDateFormat(xmin, 'datetime', self.Report.StartTime, self.Report.StartTime,
                                              self.Report.EndTime, debug=self.Report.debug)

        if 'ylims' in object_settings.keys():
            if 'max' in object_settings['ylims'].keys():
                ymax = float(object_settings['ylims']['max'])
            if 'min' in object_settings['ylims'].keys():
                ymin = float(object_settings['ylims']['min'])

        # Find Index of ALL acceptable values.
        #TODO: make this iterative so its less ugly for dicts
        for lineflag in data.keys():
            line = data[lineflag]
            values = line['values']
            dates = line['dates']

            # Per-line filterbylimits setting overrides the object-level
            # one; defaults to True (filter) if neither is set.
            filtbylims = True
            if 'filterbylimits' in line.keys():
                if line['filterbylimits'].lower() == 'false':
                    filtbylims = False
            else:
                if 'filterbylimits' in object_settings.keys():
                    if object_settings['filterbylimits'].lower() == 'false':
                        filtbylims = False

            if 'omitvalue' in line.keys():
                # single sentinel value to omit
                omitvalues = [float(line['omitvalue'])]
            elif 'omitvalues' in line.keys():
                # multiple sentinel values to omit
                omitvalues = [float(n) for n in line['omitvalues']]
            else:
                omitvalues = None

            ### FALSE WHEN OUT OF BOUNDS, TRUE WHEN KEEP

            # Build boolean masks for each active filter criterion
            # (x-date range, y-value range, omit values), defaulting to
            # "keep everything" (all True) for inactive criteria.
            if xmax != None and filtbylims:

                xmax_filt = (dates <= xmax)
            else:
                xmax_filt = np.full(dates.shape, True)

            if xmin != None and filtbylims:
                xmin_filt = (dates >= xmin)
            else:
                xmin_filt = np.full(dates.shape, True)

            if ymax != None and filtbylims:
                # Values can be a plain array or a per-member dict
                # (forecast collections); build the mask per-member when
                # needed.
                if isinstance(values, dict):
                    ymax_filt = {}
                    for key, vs in values.items():
                        ymax_filt[key] = (vs <= ymax)
                else:
                    ymax_filt = (values <= ymax)

            else:
                if isinstance(values, dict):
                    ymax_filt = {}
                    for key, vs in values.items():
                        ymax_filt[key] = np.full(vs.shape, True)
                else:
                    ymax_filt = np.full(values.shape, True)

            if ymin != None and filtbylims:
                if isinstance(values, dict):
                    ymin_filt = {}
                    for key, vs in values.items():
                        ymin_filt[key] = (vs >= ymin)
                else:
                    ymin_filt = (values >= ymin)

            else:
                if isinstance(values, dict):
                    ymin_filt = {}
                    for key, vs in values.items():
                        ymin_filt[key] = np.full(vs.shape, True)
                else:
                    ymin_filt = np.full(values.shape, True)

            if omitvalues != None:
                if isinstance(values, dict):
                    omit_filt = {}
                else:
                    omitvals_filt = []
                if isinstance(values, dict):
                    # apply the omit-value mask per member for collection data
                    for key, vs in values.items():
                        omit_filt[key] = []
                        for omitval in omitvalues:
                            omitval_filt = (vs != omitval)
                            omit_filt[key] = np.append(omitvals_filt, omitval_filt)
                else:
                    # apply the omit-value mask to the flat array
                    for omitval in omitvalues:
                        omitval_filt = (values != omitval)
                        omitvals_filt = np.append(omitvals_filt, omitval_filt)
            else:
                if isinstance(values, dict):
                    omitvals_filt = {}
                    for key, vs in values.items():
                        omitvals_filt[key] = np.full(vs.shape, True)
                else:
                    omitvals_filt = np.full(values.shape, True)

            # Combine every mask together and apply: any timestep/member
            # failing ANY active filter gets NaN'd out.
            if isinstance(values, dict):
                new_values = {}
                for key, vs in values.items():
                    master_filter = xmax_filt & xmin_filt & ymax_filt[key] & ymin_filt[key] & omitvals_filt[key]
                    vs[~master_filter] = np.nan
                    new_values[key] = vs
                data[lineflag]['values'] = new_values
            else:
                master_filter = xmax_filt & xmin_filt & ymax_filt & ymin_filt & omitvals_filt
                values[~master_filter] = np.nan
                data[lineflag]['values'] = values

        return data

    def correctTableUnits(self, data, data_settings, object_settings):
        """
        Convert every table data source's values to the configured unit system.

        Parameters
        ----------
        data : dict
            Dictionary of data keyed by flag, each with ``'values'``.
        data_settings : dict
            Per-data-source settings dictionary, each with ``'units'``
            and optionally ``'parameter'``.
        object_settings : dict
            Settings for the current plot/table object; checked for
            ``'unitsystem'``.

        Returns
        -------
        data : dict
            The ``data`` dictionary with converted values.
        data_settings : dict
            The ``data_settings`` dictionary with updated ``'units'``
            strings.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, data_settings = tables.correctTableUnits(data, data_settings, object_settings)
        """

        # convert every data source's values individually
        for datapath in data.keys():
            values = data[datapath]['values']
            units = data_settings[datapath]['units']
            if 'parameter' in data_settings[datapath].keys():
                # resolve missing units from the parameter's default, if needed
                units = WF.configureUnits(object_settings, data_settings[datapath]['parameter'], units)
            if 'unitsystem' in object_settings.keys():
                data[datapath]['values'], data_settings[datapath]['units'] = WF.convertUnitSystem(values, units, object_settings['unitsystem'], debug=self.Report.debug)

        return data, data_settings

    def getStatsLineData(self, row, data_dict, year='ALLYEARS', data_key=None):
        """
        Resolve a table row's ``%%flag%%``/``%%flag.MONTH=X%%`` placeholders into actual data.

        Parses the row's flag references (data source flags and,
        optionally, a ``MONTH=`` filter), pulls the matching data from
        ``data_dict``, and (if a specific year was requested) trims it
        further to that year.

        Parameters
        ----------
        row : str
            Row template string containing ``%%...%%`` flag references
            (e.g. ``'%%Computed.MONTH=JAN%%'``).
        data_dict : dict
            Dictionary of available data keyed by flag, each with
            ``'values'`` and ``'dates'``.
        year : int, str, or 'ALLYEARS', optional
            Year to filter to, or ``'ALLYEARS'`` to use the report's
            full year range (default ``'ALLYEARS'``).
        data_key : str or int, optional
            If given, index into ``curvalues[data_key]`` (e.g. a
            specific forecast member) instead of using the whole
            ``curvalues`` array directly.

        Returns
        -------
        data : dict
            Dictionary of resolved data (keyed by matched flag), each
            with ``'values'`` and ``'dates'`` trimmed to the requested
            year/month.
        sr_month : str or int
            The month filter found in the row (empty string if none).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data, sr_month = tables.getStatsLineData('%%rmse.Computed.Observed%%', data_dict)
        """

        data = {}

        rrow = row.replace('%%', '')
        s_row = rrow.split('.')
        sr_month = ''
        curflag = None
        # Walk the dot-separated pieces of the row, picking out any
        # recognized data flag and any 'MONTH=X' filter piece.
        for sr in s_row:
            if sr in data_dict.keys():
                curflag = sr
                curvalues = data_dict[sr]['values']
                curdates = np.array(data_dict[sr]['dates'])
                if data_key != None:
                    # specific member/index requested, pull just that slice
                    data[curflag] = {'values': curvalues[data_key], 'dates': curdates}
                else:
                    if isinstance(curvalues, dict):
                        # curvalues is a per-member dict but no specific
                        # data_key was given to select which member:
                        # can't resolve a single series, so bail out.
                        WF.print2stdout('Unable to get data for row. Expected key but none found.', debug=self.Report.debug)
                        return data, sr_month
                    else:
                        data[curflag] = {'values': np.asarray(curvalues), 'dates': curdates}
            else:
                if '=' in sr:
                    sr_spl = sr.split('=')
                    if sr_spl[0].lower() == 'month':
                        # Parse the month filter, accepting either an
                        # integer or a 3-letter month code.
                        sr_month = sr_spl[1]
                        try:
                            sr_month = int(sr_month)
                        except ValueError:
                            try:
                                sr_month = self.Report.Constants.month2num[sr_month.lower()]
                            except KeyError:
                                # not a valid integer or recognized month code at all
                                WF.print2stdout('Invalid Entry for {0}'.format(sr), debug=self.Report.debug)
                                WF.print2stdout('Try using interger values or 3 letter monthly code.', debug=self.Report.debug)
                                WF.print2stdout('Ex: MONTH=1 or MONTH=JAN', debug=self.Report.debug)
                                continue
                        if curflag == None:
                            # month filter found before any data flag was matched, can't proceed
                            WF.print2stdout('Invalid Table row for {0}'.format(row), debug=self.Report.debug)
                            WF.print2stdout('Data Key not contained within {0}'.format(data_dict.keys()), debug=self.Report.debug)
                            WF.print2stdout('Please check Datapaths in the XML file, or modify the rows to have the correct flags'
                                  ' for the data present', debug=self.Report.debug)
                            return data, ''

                        # Trim the current flag's data down to just the
                        # requested month, across every requested year
                        # (either the single specified year, or every
                        # year in the report for ALLYEARS).
                        newvals = np.array([])
                        newdates = np.array([])
                        if year != 'ALLYEARS':
                            year_loops = [year]
                        else:
                            year_loops = self.Report.years
                        if len(curdates) > 0:
                            for yearloop in year_loops:
                                s_idx, e_idx = WF.getYearlyFilterIdx(curdates, yearloop)
                                if None not in [s_idx, e_idx]:
                                    # yearvals = curvalues[s_idx:e_idx+1]
                                    # yeardates = curdates[s_idx:e_idx+1]
                                    yearvals = data[curflag]['values'][s_idx:e_idx+1]
                                    yeardates = data[curflag]['dates'][s_idx:e_idx+1]
                                else:
                                    # nothing falls in this year at all
                                    yearvals = []
                                    yeardates = []

                                if len(yeardates) > 0:
                                    s_idx, e_idx = WF.getMonthlyFilterIdx(yeardates, sr_month)

                                    newvals = np.append(newvals, yearvals[s_idx:e_idx+1])
                                    newdates = np.append(newdates, yeardates[s_idx:e_idx+1])

                        data[curflag]['values'] = newvals
                        data[curflag]['dates'] = newdates

        if year != 'ALLYEARS':
            # No month filter was present in the row (or additionally,
            # after any month filtering above): trim every resolved
            # flag's data down to the single requested year.
            for flag in data.keys():
                if len(data[flag]['dates']) == 0:
                    continue
                s_idx, e_idx = WF.getYearlyFilterIdx(data[flag]['dates'], year)
                if None not in [s_idx, e_idx]:
                    data[flag]['values'] = data[flag]['values'][s_idx:e_idx+1]
                    data[flag]['dates'] = data[flag]['dates'][s_idx:e_idx+1]
                else:
                    # nothing falls in this year for this flag at all
                    data[flag]['values'] = []
                    data[flag]['dates'] = []

        return data, sr_month

    def getStatsLine(self, row, data):
        """
        Compute the actual statistic value for a resolved row's data.

        Parameters
        ----------
        row : str
            The row template string; its leading ``%%<stat>`` prefix
            (e.g. ``'%%rmse'``) determines which statistic to compute.
        data : dict
            Dictionary of resolved data (as returned by
            ``getStatsLineData``), keyed by data flag, each with
            ``'values'``/``'dates'``.

        Returns
        -------
        out_stat : float
            The computed statistic value (NaN if it couldn't be
            computed).
        stat : str
            The name of the statistic computed (empty string if the row
            didn't match a recognized statistic prefix).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> out_stat, stat = tables.getStatsLine('%%rmse.Computed.Observed%%', data)
        """

        # stat_flag_Req = {'%%meanbias': 2,
        #                  '%%mae': 2,
        #                  '%%rmse': 2,
        #                  '%%nse': 2,
        #                  '%%count': 2, #can also be 1
        #                  '%%mean': 1}

        flags = list(data.keys())

        if len(flags) > 0:
            # Prefer 'Computed' as the first data source and 'Observed'
            # as the second, when both are present; otherwise just use
            # whichever flags were resolved, in order.
            if 'Computed' in flags:
                flag1 = 'Computed'
                if len(flags) >= 2:
                    if 'Observed' in flags:
                        flag2 = 'Observed'
                    else:
                        flag2 = [n for n in flags if n != flag1][0] #not computed

            else:
                flag1 = flags[0]
                if len(flags) >= 2:
                    flag2 = flags[1]
        else:
            # no resolved data flags at all
            WF.print2stdout(f'Insufficient data for row {row}.', debug=self.Report.debug)
            WF.print2stdout(f'Flags: {flags}', debug=self.Report.debug)
            return np.nan, ''

        out_stat = np.nan

        # bail out if any resolved flag's data is entirely empty
        for key in data.keys():
            if len(data[key]) == 0:
                WF.print2stdout(f'No data in dataset {key}.', debug=self.Report.debug)
                return np.nan, ''

        # Dispatch to the matching WAT_Functions statistic calculator
        # based on the row's leading flag prefix. Two-series statistics
        # (meanbias/mae/rmse/nse) require flag1+flag2 and are skipped
        # (left as NaN) if only one data source was resolved.
        if row.lower().startswith('%%meanbias'):
            if len(flags) > 1:
                out_stat = WF.calcMeanBias(data[flag1], data[flag2])
            stat = 'meanbias'
        elif row.lower().startswith('%%mae'):
            if len(flags) > 1:
                out_stat = WF.calcMAE(data[flag1], data[flag2])
            stat = 'mae'
        elif row.lower().startswith('%%rmse'):
            if len(flags) > 1:
                out_stat = WF.calcRMSE(data[flag1], data[flag2])
            stat = 'rmse'
        elif row.lower().startswith('%%nse'):
            if len(flags) > 1:
                out_stat = WF.calcNSE(data[flag1], data[flag2])
            stat = 'nse'
        elif row.lower().startswith('%%count'):
            # Count supports both one- and two-series forms.
            if len(flags) == 1:
                out_stat = WF.getCount(data[flag1])
            elif len(flags) > 1:
                out_stat = WF.getMultiDatasetCount(data[flag1], data[flag2])
            stat = 'count'
        elif row.lower().startswith('%%mean'):
            if len(flags) == 1:
                out_stat = WF.calcMean(data[flag1])
            stat = 'mean'
        elif row.lower().startswith('%%maximum'):
            if len(flags) == 1:
                out_stat = WF.calcMax(data[flag1])
            stat = 'maximum'
        elif row.lower().startswith('%%minimum'):
            if len(flags) == 1:
                out_stat = WF.calcMin(data[flag1])
            stat = 'minimum'
        else:
            # Not a recognized statistic prefix: return the row
            # unchanged (it's likely already a literal value/label
            # rather than a stat placeholder).
            if '%%' in row:
                WF.print2stdout('Unable to convert flag in row', row, debug=self.Report.debug)
            return row, ''

        return out_stat, stat

    def matchThresholdToStat(self, stat, object_settings):
        """
        Collect every threshold rule that applies to a given statistic.

        Supports both the deprecated ``'tablecolors'`` setting and the
        current ``'thresholds'`` setting; a threshold with no
        ``'statistic'`` key is treated as generic and applies to every
        statistic.

        Parameters
        ----------
        stat : str
            Name of the statistic being formatted (e.g. ``'rmse'``).
        object_settings : dict
            Object settings dictionary, checked for ``'tablecolors'``
            (deprecated) or ``'thresholds'``.

        Returns
        -------
        list of dict
            List of formatted threshold rule dicts applicable to
            ``stat``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> thresholds = tables.matchThresholdToStat('rmse', object_settings)
        """

        thresholds = []
        if 'tablecolors' in object_settings.keys() or 'thresholds' in object_settings.keys():
            if 'tablecolors' in object_settings.keys():
                # deprecated setting name still supported, but warn about the migration path
                WF.print2stdout('The flag "tablecolors" is deprecated as of 5.4.26. Please use "thresholds" instead, '
                                'including the specified "statistic" flag within the <threshold> object.')
                modflag = 'tablecolors'
            else:
                modflag = 'thresholds'
            # collect every threshold rule matching this statistic (or generic ones with no statistic filter)
            for threshold in object_settings[modflag]:
                if 'statistic' in threshold.keys():
                    if stat.lower() == threshold['statistic'].lower():
                        if modflag == 'tablecolors':
                            thresholds += self.formatThreshold_deprec(threshold)
                        else:
                            thf = self.formatThreshold(threshold)
                            if len(thf.keys()) > 0:
                                thresholds.append(thf)
                else: #no stat specified, generic and applies to all
                    if modflag == 'tablecolors':
                        thresholds += self.formatThreshold_deprec(threshold)
                    else:
                        thf = self.formatThreshold(threshold)
                        if len(thf.keys()) > 0:
                            thresholds.append(thf)

        return thresholds

    def matchNumberFormatByStat(self, stat, settings):
        """
        Determine the number-formatting rule(s) to use for a given statistic.

        Parameters
        ----------
        stat : str
            Name of the statistic being formatted.
        settings : dict
            Settings dictionary, checked for a ``'numberformats'`` list.

        Returns
        -------
        list of dict
            The statistic-specific formatting rules if any were found
            (with a default of 0 decimal places auto-added for
            ``'count'`` if not otherwise specified); otherwise the
            generic (non-statistic-specific) formatting rules.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> numberFormat = tables.matchNumberFormatByStat('rmse', settings)
        """

        numberFormats_default = []
        numberFormats_statspec = []

        if 'numberformats' in settings.keys():
            # separate the statistic-specific formatting rules from the generic ones
            for numberformat in settings['numberformats']:
                if 'stats' in numberformat:
                    if stat.lower() in [n.lower() for n in numberformat['stats']]:
                        numberFormats_statspec.append(numberformat)
                else:
                    numberFormats_default.append(numberformat)
        if isinstance(stat, str):
            if stat.lower() == 'count':
                # Counts are always whole numbers; default to 0 decimal
                # places unless the user configured something else.
                if len(numberFormats_statspec) == 0:
                    numberFormats_statspec.append({'decimalplaces': 0})

        if len(numberFormats_statspec) > 0:
            return numberFormats_statspec
        else:
            return numberFormats_default

    def formatThreshold_deprec(self, object_settings):
        """
        Parse the deprecated (pre-5.4.26) ``'thresholds'`` settings format.

        Deprecated as of 5.4.26; use ``formatThreshold`` (with the
        current ``'thresholds'`` XML structure) instead.

        Parameters
        ----------
        object_settings : dict
            Settings dictionary containing a nested ``'thresholds'``
            list (old-format).

        Returns
        -------
        list of dict
            List of formatted threshold rule dicts, each with
            ``'value'``, ``'color'``, ``'colorwhen'``, ``'when'``, and
            optionally ``'replacement'``. Thresholds without a
            ``'value'`` are dropped.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> thresholds = tables.formatThreshold_deprec(object_settings)
        """

        default_color = '#a6a6a6' #default, grey
        default_when = 'under' #default
        accepted_threshold_conditions = ['under', 'over']
        thresholds = []

        if 'thresholds' in object_settings.keys():
            # parse and fill in defaults for every configured threshold rule
            for threshold in object_settings['thresholds']:
                threshold_settings = {}

                if 'value' in threshold.keys():
                    threshold_settings['value'] = float(threshold['value'])
                else:
                    continue #dont record this threshold

                if 'color' in threshold.keys():
                    threshold_settings['color'] = self.formatThresholdColor(threshold['color'], default=default_color)
                else:
                    threshold_settings['color'] = default_color

                if 'colorwhen' in threshold.keys():
                    if any([n.lower() == threshold['colorwhen'].lower() for n in accepted_threshold_conditions]):
                        threshold_settings['colorwhen'] = threshold['colorwhen'].lower()
                    else:
                        # invalid comparison keyword given, fall back to the default
                        WF.print2stdout(f"Invalid threshold setting {threshold['colorwhen']}", debug=self.Report.debug)
                        WF.print2stdout(f'Please select value in {accepted_threshold_conditions}', debug=self.Report.debug)
                        WF.print2stdout(f'Setting to default, {default_when}', debug=self.Report.debug)
                        threshold_settings['colorwhen'] = default_when
                else:
                    threshold_settings['colorwhen'] = default_when

                if 'when' in threshold.keys():
                    if any([n.lower() == threshold['when'].lower() for n in accepted_threshold_conditions]):
                        threshold_settings['when'] = threshold['when'].lower()
                    else:
                        # invalid comparison keyword given, fall back to the default
                        WF.print2stdout(f"Invalid threshold setting {threshold['colorwhen']}", debug=self.Report.debug)
                        WF.print2stdout(f'Please select value in {accepted_threshold_conditions}', debug=self.Report.debug)
                        WF.print2stdout(f'Setting to default, {default_when}', debug=self.Report.debug)
                        threshold_settings['when'] = default_when
                else:
                    threshold_settings['when'] = default_when

                if 'replacement' in threshold.keys():
                    threshold_settings['replacement'] = str(threshold['replacement'])

                thresholds.append(threshold_settings)

        return thresholds

    def getThresholdsfromSettings(self, object_settings):
        """
        Parse and format every threshold rule in an object's settings.

        Parameters
        ----------
        object_settings : dict
            Settings dictionary, checked for a ``'thresholds'`` list
            (current format).

        Returns
        -------
        list of dict
            List of formatted threshold rule dicts.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> thresholds = tables.getThresholdsfromSettings(object_settings)
        """

        thresholds = []
        if 'thresholds' in object_settings.keys():
            # format and collect every configured threshold rule
            for threshold in object_settings['thresholds']:
                thf = self.formatThreshold(threshold)
                if len(thf.keys()) > 0:
                    thresholds.append(thf)
        return thresholds

    def formatThreshold(self, threshold):
        """
        Parse and fill in defaults for a single threshold rule.

        Parameters
        ----------
        threshold : dict
            Settings dictionary for one threshold rule; must contain
            ``'value'`` to be recorded (otherwise dropped).

        Returns
        -------
        dict
            Formatted threshold dict with ``'value'``, ``'color'``,
            ``'colorwhen'``, ``'when'``, and optionally
            ``'replacement'``; an empty dict if no ``'value'`` was
            given.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.formatThreshold({'value': '2.0', 'colorwhen': 'over'})
        """

        default_color = '#a6a6a6' #default, grey
        default_when = 'under' #default
        accepted_threshold_conditions = ['under', 'over']

        threshold_settings = {}

        if 'value' in threshold.keys():
            threshold_settings['value'] = float(threshold['value'])
        else:
            # no threshold value given at all, this rule can't be used
            return {} #dont record this threshold

        if 'color' in threshold.keys():
            threshold_settings['color'] = self.formatThresholdColor(threshold['color'], default=default_color)
        else:
            threshold_settings['color'] = default_color

        if 'colorwhen' in threshold.keys():
            if any([n.lower() == threshold['colorwhen'].lower() for n in accepted_threshold_conditions]):
                threshold_settings['colorwhen'] = threshold['colorwhen'].lower()
            else:
                # invalid comparison keyword given, fall back to the default
                WF.print2stdout(f"Invalid threshold setting {threshold['colorwhen']}", debug=self.Report.debug)
                WF.print2stdout(f'Please select value in {accepted_threshold_conditions}', debug=self.Report.debug)
                WF.print2stdout(f'Setting to default, {default_when}', debug=self.Report.debug)
                threshold_settings['colorwhen'] = default_when
        else:
            threshold_settings['colorwhen'] = default_when

        if 'when' in threshold.keys():
            if any([n.lower() == threshold['when'].lower() for n in accepted_threshold_conditions]):
                threshold_settings['when'] = threshold['when'].lower()
            else:
                # invalid comparison keyword given, fall back to the default
                WF.print2stdout(f"Invalid threshold setting {threshold['colorwhen']}", debug=self.Report.debug)
                WF.print2stdout(f'Please select value in {accepted_threshold_conditions}', debug=self.Report.debug)
                WF.print2stdout(f'Setting to default, {default_when}', debug=self.Report.debug)
                threshold_settings['when'] = default_when
        else:
            threshold_settings['when'] = default_when

        if 'replacement' in threshold.keys():
            threshold_settings['replacement'] = str(threshold['replacement'])

        return threshold_settings

    def formatThresholdColor(self, in_color, default='#a6a6a6'):
        """
        Normalize a threshold color string to hex, falling back to a default.

        Parameters
        ----------
        in_color : str
            Color string to validate/convert (a hex string or any
            matplotlib-recognized color name).
        default : str, optional
            Hex color to fall back to if ``in_color`` is invalid
            (default ``'#a6a6a6'``, grey).

        Returns
        -------
        str
            The resolved hex color string.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.formatThresholdColor('lightblue')
        '#add8e6'
        """

        threshold_color = default
        if in_color.startswith('#'):
            # already a hex string, use it directly
            threshold_color = in_color
        else:
            try:
                # try converting a named color (or other matplotlib-recognized format) to hex
                threshold_color = to_hex(in_color)
            except ValueError:
                # invalid color string, fall back to the default
                WF.print2stdout(f'Invalid color of {in_color}', debug=self.Report.debug)

        return threshold_color

    def getTableDates(self, year, object_settings, month='None'):
        """
        Resolve the display start/end date string for a table's data-provenance log entry.

        Parameters
        ----------
        year : int or str
            Selected year, or ``'ALLYEARS'``.
        object_settings : dict
            Object settings dictionary; checked for ``'xlims'``
            (explicit min/max override dates).
        month : str, optional
            Selected month name/number (for monthly tables), or
            ``'None'`` if not applicable (default ``'None'``).

        Returns
        -------
        start_date : str
            Formatted start date string (``'%d %b %Y'``).
        end_date : str
            Formatted end date string (``'%d %b %Y'``).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> start_date, end_date = tables.getTableDates(2020, object_settings)
        """

        xmin = 'NONE'
        xmax = 'NONE'
        if 'xlims' in object_settings.keys():
            if 'min' in object_settings['xlims'].keys():
                if object_settings['xlims']['min'] is not None:
                    # resolve the explicit configured minimum date
                    xmin = WF.updateFlaggedValues(object_settings['xlims']['min'],'%%year%%', str(min(self.Report.years)))
                    xmin = WT.translateDateFormat(xmin, 'datetime', self.Report.StartTime,
                                                  self.Report.StartTime, self.Report.EndTime,
                                                  debug=self.Report.debug)
                    xmin = xmin.strftime('%d %b %Y')
            if 'max' in object_settings['xlims'].keys():
                if object_settings['xlims']['max'] is not None:
                    # resolve the explicit configured maximum date
                    xmax = WF.updateFlaggedValues(object_settings['xlims']['max'],'%%year%%', str(max(self.Report.years)))
                    xmax = WT.translateDateFormat(xmax, 'datetime', self.Report.EndTime,
                                                  self.Report.StartTime, self.Report.EndTime,
                                                  debug=self.Report.debug)
                    xmax = xmax.strftime('%d %b %Y')

        # Priority for the start date: explicit xlims min > the report's
        # actual start time (if this is the report's start year) >
        # January 1st of the requested year (or the report's overall
        # start year, for ALLYEARS).
        if xmin != 'NONE':
            start_date = xmin
        elif year == self.Report.startYear:
            start_date = self.Report.StartTime.strftime('%d %b %Y')
        else:
            if str(year).lower() == 'allyears':
                start_date = '01 Jan {0}'.format(self.Report.startYear)
            else:
                start_date = '01 Jan {0}'.format(year)

        # Mirror image priority for the end date.
        if xmax != 'NONE':
            end_date = xmax
        elif year == self.Report.endYear:
            end_date = self.Report.EndTime.strftime('%d %b %Y')
        else:
            if str(year).lower() == 'allyears':
                end_date = '31 Dec {0}'.format(self.Report.endYear)
            else:
                end_date = '31 Dec {0}'.format(year)

        if month != 'None':
            # A specific month was requested (monthly table): narrow the
            # start/end dates down to just that month, handling the
            # December-rollover edge case for computing the last day of
            # the month.
            try:
                month = int(month)
            except ValueError:
                month = self.Report.Constants.month2num[month.lower()]

            try:
                # try the direct month substitution first (works for months with 31+ days in the start date's month)
                start_date = dt.datetime.strptime(start_date, '%d %b %Y').replace(month=month).strftime('%d %b %Y')
            except ValueError:
                # direct substitution failed (e.g. day 31 doesn't exist in the target month), fall back to
                # computing the last day of the target month by rolling to the 1st of the next month and subtracting a day
                start_date = dt.datetime.strptime(start_date, '%d %b %Y')
                start_date = start_date.replace(day=1)
                start_date = start_date.replace(month=month+1)
                start_date -= dt.timedelta(days=1)
                start_date = start_date.strftime('%d %b %Y')
            try:
                # same direct substitution attempt for the end date
                end_date = dt.datetime.strptime(end_date, '%d %b %Y').replace(month=month).strftime('%d %b %Y')
            except ValueError:
                # same rollover fallback for the end date
                end_date = dt.datetime.strptime(end_date, '%d %b %Y')
                end_date = end_date.replace(day=1)
                end_date = end_date.replace(month=month+1)
                end_date -= dt.timedelta(days=1)
                end_date = end_date.strftime('%d %b %Y')

        return start_date, end_date

    def convertHeaderFormats(self, headers, object_settings):
        """
        Convert profile-line-table headers to the configured date display format.

        Parameters
        ----------
        headers : list of list
            Nested list of timestamp headers (one list per year), as
            produced by ``buildHeadersByTimestamps``.
        object_settings : dict
            Object settings dictionary; checked for ``'dateformat'``
            (``'datetime'`` (default) or ``'jdate'``).

        Returns
        -------
        list of list
            The same nested structure, with each header converted to a
            formatted string (``'%d%b%Y'`` for datetime, or the raw
            Julian-date string for jdate).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> new_headers = tables.convertHeaderFormats(headers, object_settings)
        """

        if 'dateformat' not in object_settings.keys():
            # default to calendar-date display if not specified
            object_settings['dateformat'] = 'datetime'

        new_headers = []
        # convert every year's header list independently
        for headeryear in headers:
            nh = []
            for header in headeryear:
                if object_settings['dateformat'].lower() == 'datetime':
                    header = WT.translateDateFormat(header, 'datetime', '',
                                                    self.Report.StartTime, self.Report.EndTime,
                                                    debug=self.Report.debug)
                    header = header.strftime('%d%b%Y')
                elif object_settings['dateformat'].lower() == 'jdate':
                    header = WT.translateDateFormat(header, 'jdate', '',
                                                    self.Report.StartTime, self.Report.EndTime,
                                                    debug=self.Report.debug)
                    header = str(header)
                nh.append(header)
            new_headers.append(nh)

        return new_headers

    def formatPrimaryKey(self, data, object_settings):
        """
        Reformat a formatted table's primary key column as collection member numbers.

        Parameters
        ----------
        data : dict
            Dictionary of ``{flag: pandas.DataFrame}`` formatted tables.
        object_settings : dict
            Object settings dictionary; checked for
            ``'formatprimaryascollection'`` and ``'primarykey'``.

        Returns
        -------
        dict
            The ``data`` dictionary, with the primary key column
            reformatted in place (only if
            ``'formatprimaryascollection'`` is ``'true'``).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> data = tables.formatPrimaryKey(data, object_settings)
        """

        if 'formatprimaryascollection' in object_settings.keys():
            if object_settings['formatprimaryascollection'].lower() == 'true':
                primarykey = object_settings['primarykey']
                # reformat the primary key value for every row in every table
                for datakey in data.keys():
                    df = data[datakey]
                    for i, row in df.iterrows():
                        df.loc[i, primarykey] = WF.formatMembers(row[primarykey])
                    data[datakey] = df
        return data

    def formatStatsProfileLineData(self, row, data_dict, interpolation, usedepth, index):
        """
        Interpolate every profile line referenced in a row onto a common depth/elevation grid.

        Finds the overlapping depth/elevation range shared by every
        referenced profile at this timestamp, then interpolates each
        profile's values onto a common set of y-values (either a fixed
        number of evenly-spaced points, or the exact y-values of one
        designated reference line) so statistics can be computed
        consistently across profiles that may not share the same raw
        sampling points.

        Parameters
        ----------
        row : str
            Row template string containing ``%%flag%%``-style
            references to lines in ``data_dict``.
        data_dict : dict
            Dictionary of available profile line data, keyed by flag,
            each with ``'values'``, ``'depths'``, and ``'elevations'``
            (lists indexed by timestamp).
        interpolation : int or str
            Either a fixed number of interpolation points, or the flag
            name of a reference line whose own depth/elevation values
            should be used as the interpolation target grid.
        usedepth : str
            ``'true'``/``'false'`` string selecting whether to use
            depths or elevations as the y-axis.
        index : int
            Timestamp index to extract profile data at.

        Returns
        -------
        dict
            Dictionary keyed by flag, each with ``'values'`` and the
            selected y-axis key (``'depths'`` or ``'elevations'``)
            interpolated onto the common grid.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> out_data = tables.formatStatsProfileLineData(row, data_dict, 30, 'true', 0)
        """

        rrow = row.replace('%%', '')
        s_row = rrow.split('.')
        flags = []
        out_data = {}
        # extract just the recognized data-flag references from the row
        for sr in s_row:
            if sr in data_dict.keys():
                flags.append(sr)

        # If `interpolation` names one of the referenced lines, use that
        # line's own y-values as the interpolation target grid instead
        # of building an evenly-spaced grid.
        useflagforinterp = False
        if isinstance(interpolation, str):
            # if interpolation in [n.lower() for n in flags]:
            if interpolation in flags:
                useflagforinterp = True
            else:
                # requested reference flag doesn't actually exist, fall back to a fixed resolution
                WF.print2stdout(f'Flag for output {interpolation} not found in data flags {flags}. Defaulting to '
                                f'interpolating both at 30 pt resolution', debug=self.Report.debug)
                interpolation = 30

        if usedepth.lower() == 'true':
            y_flag = 'depths'
        else:
            y_flag = 'elevations'

        if not useflagforinterp:
            # Determine the overlapping y-range shared by every
            # referenced line at this timestamp (the tightest common
            # top/bottom bound), so the interpolation grid doesn't
            # extrapolate any single line beyond its own real data.
            top = None
            bottom = None

            for flag in flags:
                #get elevs
                if usedepth.lower() == 'true':
                    depths = data_dict[flag]['depths'][index]
                    if len(depths) > 0:
                        top_depth = np.min(depths)
                        bottom_depth = np.max(depths)
                        #find limits comparing flags so we can be sure to interpolate over the same data
                        if top == None:
                            top = top_depth
                        else:
                            if top_depth > top:
                                top = top_depth

                        if bottom == None:
                            bottom = bottom_depth
                        else:
                            if bottom_depth < bottom:
                                bottom = bottom_depth

                else:
                    elevs = data_dict[flag]['elevations'][index]
                    if len(elevs) > 0:
                        top_elev = np.max(elevs)
                        bottom_elev = np.min(elevs)
                        #find limits comparing flags so we can be sure to interpolate over the same data
                        if top == None:
                            top = top_elev
                        else:
                            if top_elev < top:
                                top = top_elev

                        if bottom == None:
                            bottom = bottom_elev
                        else:
                            if bottom_elev > bottom:
                                bottom = bottom_elev

            if bottom == None and top == None:
                # no data at all for any referenced line at this timestamp
                output_interp_yvalues = []
            elif bottom == top:
                # No overlapping range at all (or zero-width range):
                # nothing usable to interpolate onto.
                output_interp_yvalues = []
            else:
                if usedepth.lower() == 'true':
                    #build elev profiles
                    output_interp_yvalues = np.arange(top, bottom, (bottom-top) / float(interpolation))
                else:
                    output_interp_yvalues = np.arange(bottom, top, (top-bottom) / float(interpolation))


        # ===================== Interpolate every referenced line onto the target grid =====================
        for flag in flags:
            out_data[flag] = {}
            #interpolate over all values and then get interp values

            if len(data_dict[flag]['values'][index]) < 2:
                # Not enough points to interpolate from at all: fill
                # with NaN placeholders sized to match the target grid.
                WF.print2stdout('Insufficient data points with current bounds for {0}'.format(flag), debug=self.Report.debug)
                if not useflagforinterp:
                    out_data[flag]['values'] = np.full(len(output_interp_yvalues), np.nan)
                    out_data[flag][y_flag] = np.full(len(output_interp_yvalues), np.nan)

                else:
                    out_data[flag]['values'] = np.full_like(data_dict[interpolation]['values'][index], np.nan)
                    out_data[flag][y_flag] = np.full_like(data_dict[interpolation][y_flag][index], np.nan)
                continue

            if not useflagforinterp:
                if len(output_interp_yvalues) == 0:
                    # no usable interpolation grid was built, nothing to compute
                    WF.print2stdout(f'Insufficient {y_flag} points for row {flag} in {row}', debug=self.Report.debug)
                    out_data[flag]['values'] = []
                    out_data[flag]['depths'] = []
                    out_data[flag]['elevations'] = []
                    continue

            else:
                if len(data_dict[interpolation][y_flag][index]) == 0:
                    # the reference line itself has no data at this timestamp
                    WF.print2stdout(f'Insufficient {y_flag} points for row {interpolation} in {row}', debug=self.Report.debug)
                    out_data[flag]['values'] = []
                    out_data[flag]['depths'] = []
                    out_data[flag]['elevations'] = []
                    continue

            if not np.all(data_dict[flag][y_flag][index][:-1] != data_dict[flag][y_flag][index][1:]): #check for duplicate yvals
                # scipy's interpolator can't handle duplicate x-values;
                # drop any duplicated y-axis points before interpolating.
                WF.print2stdout(f'Found duplicate values in {y_flag} for {flag} at index {index}', debug=self.Report.debug)
                duplicatemask = data_dict[flag][y_flag][index][:-1] != data_dict[flag][y_flag][index][1:]
                duplicatemask = np.insert(duplicatemask, 0, True)
                data_dict[flag][y_flag][index] = data_dict[flag][y_flag][index][duplicatemask]
                data_dict[flag]['values'][index] = data_dict[flag]['values'][index][duplicatemask]

            if useflagforinterp:
                if flag == interpolation:
                    # This IS the reference line: no interpolation
                    # needed, use its own values directly.
                    out_data[flag][y_flag] = data_dict[flag][y_flag][index]
                    out_data[flag]['values'] = data_dict[flag]['values'][index]
                else:
                    # Interpolate this line onto the reference line's
                    # exact y-values; anything outside this line's own
                    # range becomes NaN rather than extrapolated.
                    f_interp = interpolate.interp1d(data_dict[flag][y_flag][index], data_dict[flag]['values'][index],
                                                    bounds_error=False, fill_value=np.nan)
                    out_data[flag][y_flag] = data_dict[interpolation][y_flag][index]
                    out_data[flag]['values'] = f_interp(data_dict[interpolation][y_flag][index])
            else:
                # Interpolate this line onto the evenly-spaced common
                # grid computed above; extrapolate at the edges since
                # the grid was already clamped to the overlapping range.
                f_interp = interpolate.interp1d(data_dict[flag][y_flag][index], data_dict[flag]['values'][index], fill_value='extrapolate')
                out_data[flag]['values'] = f_interp(output_interp_yvalues)
                out_data[flag][y_flag] = output_interp_yvalues

        return out_data

    def replaceComparisonSettings(self, object_settings, iscomp):
        """
        Substitute comparison-specific settings over the normal ones when applicable.

        Parameters
        ----------
        object_settings : dict
            Settings dictionary; must contain ``'replaced_defaults'``
            (a list of keys already overridden by the user via
            ``WF.replaceDefaults``).
        iscomp : bool
            Whether this is a comparison report run.

        Returns
        -------
        dict
            The ``object_settings`` dictionary, with each mapped
            comparison-specific setting (e.g. ``'comparisonheaders'``)
            copied over its normal counterpart (e.g. ``'headers'``),
            unless the user already explicitly set the normal one.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> object_settings = tables.replaceComparisonSettings(object_settings, True)
        """

        replace_flags = {'comparisonheaders': 'headers'}
        replaced_defaults = object_settings['replaced_defaults']
        if iscomp:
            # substitute each comparison-specific setting over its normal counterpart
            for comparisonflag in replace_flags.keys():
                normalflag = replace_flags[comparisonflag]
                if comparisonflag in object_settings.keys():
                    if normalflag in replaced_defaults:
                        # User already explicitly set the normal
                        # setting; don't clobber it with the comparison
                        # variant.
                        continue
                    object_settings[normalflag] = object_settings[comparisonflag]

        return object_settings

    def replaceIllegalJasperCharacters(self, tablelist):
        """
        Replace characters Jasper can't render directly with HTML entities.

        Parameters
        ----------
        tablelist : list of str
            List of table cell values to sanitize.

        Returns
        -------
        list of str
            The list with ``<``/``>`` replaced by their HTML entity
            equivalents.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.replaceIllegalJasperCharacters(['a < b'])
        ['a &#60; b']
        """

        illegal_chars = {"<": "&#60;",
                         ">": "&#62;"}
        newtablelist = []
        # sanitize each cell value individually
        for tl in tablelist:
            replaced = False
            for key, char in illegal_chars.items():
                if key in tl:
                    newtablelist.append(tl.replace(key, char))
                    replaced = True
            if not replaced:
                # nothing needed replacing, keep the value as-is
                newtablelist.append(tl)
        return newtablelist

    def replaceIllegalJasperCharactersHeadings(self, headers):
        """
        Sanitize a list of table headers for Jasper compatibility.

        Parameters
        ----------
        headers : list of str
            Table header strings.

        Returns
        -------
        list of str
            Sanitized headers (see ``replaceIllegalJasperCharacters``).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.replaceIllegalJasperCharactersHeadings(['Flow < 100'])
        """

        return self.replaceIllegalJasperCharacters(headers)

    def replaceIllegalJasperCharactersRows(self, rows):
        """
        Sanitize a list of "|"-delimited table rows for Jasper compatibility.

        Parameters
        ----------
        rows : list of str
            ``"|"``-delimited row strings.

        Returns
        -------
        list of str
            Sanitized rows, with each ``"|"``-delimited cell individually
            sanitized then rejoined.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.replaceIllegalJasperCharactersRows(['Jan|<5'])
        """

        new_rows = []
        # sanitize each row's individual cells, then rejoin with the original delimiter
        for row in rows:
            new_rows.append('|'.join(self.replaceIllegalJasperCharacters(row.split('|'))))
        return new_rows

    def configureHeadingsGroups(self, headings):
        """
        Group heading indices by their first-column value.

        Parameters
        ----------
        headings : list
            List of heading tuples/lists, each with the grouping key as
            its first element.

        Returns
        -------
        list of list
            One list of indices into ``headings`` per unique first-
            column value found, in order of first appearance.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.configureHeadingsGroups(headings)
        """

        headings_groups = []
        # collect each unique first-column value, preserving first-appearance order
        for h in headings:
            if h[0] not in headings_groups:
                headings_groups.append(h[0])
        headings_i = [[] for n in headings_groups]
        # for every group, collect the indices of every heading sharing that group value
        for hgi, hgroup in enumerate(headings_groups):
            for hi, h in enumerate(headings):
                if str(h[0]) == str(hgroup):
                    headings_i[hgi].append(hi)
        return headings_i

    def configureRowsForCollection(self, rows, object_settings):
        """
        Expand ``%%member%%``-templated rows into one row per forecast member.

        Parameters
        ----------
        rows : list of str
            Row template strings, potentially containing a
            ``'%%member%%'`` placeholder.
        object_settings : dict
            Settings dictionary; checked for an explicit ``'members'``
            list, otherwise every report member is used.

        Returns
        -------
        list of str
            The expanded row list; rows without a member placeholder
            are passed through unchanged, one copy each.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> formatted_rows = tables.configureRowsForCollection(rows, object_settings)
        """

        formatted_rows = []
        #figure out members first
        if 'members' in object_settings.keys(): #if a subset
            members = object_settings['members']
        else: #otherwise, get them all
            members = self.Report.allMembers
        for row in rows:
            if '%%member%%' in row:
                # Expand this row once per member, substituting each
                # member number into the placeholder.
                for member in members:
                    srow = row.split('|')
                    frow = []
                    for sr in srow:
                        if '%%member%%' in sr:
                            sr = sr.replace('%%member%%', f'%%member.{member}%%')
                        frow.append(sr)
                    formatted_rows.append('|'.join(frow))
            else:
                # no member placeholder at all, pass through unchanged
                formatted_rows.append(row)
        return formatted_rows

    def writeTable(self, table_constructor):
        """
        Write a table_constructor dictionary's columns out to the report XML.

        For comparison reports, wraps groups of columns sharing the same
        ``'datecolumn'`` value in date-column start/end markers.

        Parameters
        ----------
        table_constructor : dict
            Dictionary of columns keyed by index, each with
            ``'header'``, ``'rows'``, ``'thresholdcolors'``, and (for
            comparison reports) ``'datecolumn'``.

        Returns
        -------
        None
            Writes the table's columns (and, for comparison reports,
            date-column groupings) to the report XML.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.writeTable(table_constructor)
        """

        lastdatecol = ''
        # write every column in index order, wrapping date-column boundaries for comparison reports
        for i in range(max(table_constructor.keys())+1):
            if i not in table_constructor.keys():
                # gap in the index sequence, skip it
                continue
            current_col = table_constructor[i]
            if self.Report.iscomp:
                if current_col['datecolumn'] != lastdatecol: #if the date column is different
                    if lastdatecol != '': #but not the first one
                        self.Report.XML.writeDateColumnEnd() #write date column end
                    self.Report.XML.writeDateColumn(current_col['datecolumn']) #write date column
                    lastdatecol = current_col['datecolumn'] #set last date column
            self.Report.XML.writeTableColumn(current_col['header'], current_col['rows'], thresholdcolors=current_col['thresholdcolors'])
            # if self.Report.iscomp:
            #     if i == (len(table_constructor.keys())-1):
            #         self.Report.XML.writeDateColumnEnd()
        if self.Report.iscomp:
            # close out the final date column group
            self.Report.XML.writeDateColumnEnd() #write date column end ifccomp
        self.Report.XML.writeTableEnd()

    def writeMissingTableItemsWarning(self, description):
        """
        Write a text-box note that some columns of a table were dropped for lacking data.

        Parameters
        ----------
        description : str
            Description/title of the table this warning applies to.

        Returns
        -------
        None
            Adds a text box to the report.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.writeMissingTableItemsWarning('Monthly Averages')
        """

        self.Report.makeTextBox({'text': f'Some items in Table "{description}" not generated due to insufficient data.'})

    def writeMissingTableWarning(self, description):
        """
        Write a text-box note that an entire table was skipped for lacking data.

        Parameters
        ----------
        description : str
            Description/title of the table this warning applies to.

        Returns
        -------
        None
            Adds a text box to the report.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.writeMissingTableWarning('Monthly Averages')
        """

        self.Report.makeTextBox({'text': f'\nTable "{description}" not generated due to insufficient data.'})

    def defineForecastTableHeaders(self):
        """
        Define the display names for forecast table column flags.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets ``self.forecastTableHeaders``, a dict mapping each
            recognized XML column name to its display label.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.defineForecastTableHeaders()
        >>> tables.forecastTableHeaders['metname']
        'Met Set'
        """

        #{name from XML | display name}
        # fixed lookup table mapping internal column flags to their display labels
        self.forecastTableHeaders = {'name': 'Name',
                                     'operationsname': 'Operations',
                                     'metname': 'Met Set',
                                     'temptargetname': 'Temp Target',
                                     'member': 'Member Number'
                                     }

    def confirmForecastTableHeaders(self, columns):
        """
        Validate user-configured forecast table columns against the recognized set.

        Parameters
        ----------
        columns : list of str
            User-configured column names.

        Returns
        -------
        list of str
            Only the columns that matched a recognized header name;
            unrecognized columns are dropped (with a logged warning).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.confirmForecastTableHeaders(['member', 'badcolumn'])
        ['member']
        """

        rejected_columns = []
        approved_columns = []
        # sort each requested column into approved or rejected based on the recognized header set
        for column in columns:
            if column.lower() not in self.forecastTableHeaders.keys():
                rejected_columns.append(column)
            else:
                approved_columns.append(column)
        if len(rejected_columns) > 0:
            WF.print2stdout(f'Invalid column(s) selected: {rejected_columns}')
            WF.print2stdout(f'Approved column(s): {self.forecastTableHeaders.keys()}')

        return approved_columns

    def formatForecastTableHeaders(self, headers):
        """
        Convert forecast table column flags into their display labels.

        Parameters
        ----------
        headers : list of str
            Column flag names (e.g. ``'metname'``).

        Returns
        -------
        list of str
            The matching display labels (e.g. ``'Met Set'``), in the
            same order.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.formatForecastTableHeaders(['member', 'metname'])
        ['Member Number', 'Met Set']
        """

        formatted_headers = []
        # translate each internal flag name to its display label, in order
        for header in headers:
            formatted_headers.append(self.forecastTableHeaders[header.lower()])
        return formatted_headers

    # def checkForMissingData(self, row_val, missing):
        # row_split = [n.replace('%', '') for n in row_val.split('.')]
        # for m in missing:
        #     if m in row_split:
        #         return True
        # return False
    def checkForMissingData(self, row_val, row_data):
        """
        Check whether a row's referenced data flags have any usable data.

        Parameters
        ----------
        row_val : str
            Row value string containing ``%%flag%%``-style references.
        row_data : dict
            Dictionary of resolved data keyed by flag, each with
            ``'values'``.

        Returns
        -------
        bool
            ``True`` if any flag referenced in ``row_val`` has invalid/
            missing data (per ``WF.checkData``), ``False`` otherwise.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.checkForMissingData('%%rmse.Computed.Observed%%', row_data)
        False
        """
        # strip the %% markers so the flags can be matched directly against row_data's keys
        row_split = [n.replace('%', '') for n in row_val.split('.')]
        for key in row_data:
            if key in row_split:
                check = WF.checkData(row_data[key]['values'])
                if not check:
                    # at least one referenced flag has unusable data
                    return True
        return False

    def getStat(self, row_val):
        """
        Extract the statistic name from a row value's leading flag.

        Parameters
        ----------
        row_val : str
            Row value string (e.g. ``'%%rmse.Computed.Observed%%'``).

        Returns
        -------
        str
            The lowercased statistic name (e.g. ``'rmse'``), taken from
            the first dot-separated, ``%``-stripped segment.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> tables.getStat('%%rmse.Computed.Observed%%')
        'rmse'
        """
        # take the first dot-separated segment and strip the %% markers
        srv = row_val.split('.')[0].lower().replace('%', '')
        return srv