import os
import pandas as pd

import WAT_Functions as WF


class WAT_Logger(object):
    """
    Accumulates metadata about report data/simulations and writes a CSV log.

    Every plot, table, and profile the report generator produces can
    call ``addLogEntry`` (and simulations can call ``addSimLogEntry``) to
    record what data was used, where it came from, and when it was
    computed. All of this is kept in a single dictionary of parallel
    lists (``self.Log``) and written out as one row-aligned CSV
    (``Log.csv``) at the end of report generation via ``writeLogFile``,
    giving a full audit trail of everything that went into the report.

    Attributes
    ----------
    Report : object
        The main Report Generator instance this logger serves.
    Log : dict
        Dictionary of parallel column lists accumulating all logged
        entries; see ``buildLogFile`` for the full list of columns.
    """

    def __init__(self, Report):
        """
        Initialize the logger and build the empty log dictionary.

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
        >>> logger = WAT_Logger(Report)
        """

        # keep a reference back to the parent report for shared state
        self.Report = Report
        # initialize the empty log dictionary with all expected columns
        self.buildLogFile()

    def buildLogFile(self):
        """
        Build the log dictionary of parallel column lists.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets the instance attribute ``self.Log``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> logger.buildLogFile()
        >>> logger.Log['type']
        []
        """

        # Every log "column" starts as an empty list; entries are
        # appended in parallel across all keys as data/simulation
        # metadata gets logged (see addLogEntry/addSimLogEntry).
        self.Log = {'type': [], 'name': [], 'description': [], 'value': [], 'units': [], 'observed_data_path': [],
                    'start_time': [], 'end_time': [], 'compute_time': [], 'program': [], 'alternative_name': [],
                    'fpart': [], 'program_directory': [], 'region': [], 'value_start_date': [], 'value_end_date': [],
                    'function': [], 'logoutputfilename': []}

    def equalizeLog(self):
        """
        Pad every log column out to the same length with empty strings.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Updates ``self.Log`` in place.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> logger.equalizeLog()
        """

        # Find the length of the longest column; some columns (e.g.
        # simulation-level fields vs. per-plot fields) may be appended to
        # at different rates, so this pads the shorter ones out with
        # empty strings to keep every column the same length before
        # building a DataFrame from them.
        longest_array_len = 0
        # first pass: determine the length of the longest column
        for key in self.Log.keys():
            if len(self.Log[key]) > longest_array_len:
                longest_array_len = len(self.Log[key])
        # second pass: pad every shorter column out to match that length
        for key in self.Log.keys():
            if len(self.Log[key]) < longest_array_len:
                num_entries = longest_array_len - len(self.Log[key])
                # append empty-string placeholders one at a time until this column catches up
                for i in range(num_entries):
                    self.Log[key].append('')

    def writeLogFile(self, images_path):
        """
        Write the accumulated log data out to a CSV file.

        Parameters
        ----------
        images_path : str
            Directory path to write ``Log.csv`` into (the report's
            images/output directory).

        Returns
        -------
        None
            Writes ``Log.csv`` to disk.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> logger.writeLogFile('/path/to/report/images')
        """

        # Assemble the log columns into a DataFrame in a specific,
        # readable column order (rather than the dict's insertion order),
        # renaming 'logoutputfilename' to the more descriptive
        # 'CSVOutputFilename' for the final CSV header.
        df = pd.DataFrame({'observed_data_path': self.Log['observed_data_path'],
                           'start_time': self.Log['start_time'],
                           'end_time': self.Log['end_time'],
                           'compute_time': self.Log['compute_time'],
                           'program': self.Log['program'],
                           'region': self.Log['region'],
                           'alternative_name': self.Log['alternative_name'],
                           'fpart': self.Log['fpart'],
                           'program_directory': self.Log['program_directory'],
                           'type': self.Log['type'],
                           'name': self.Log['name'],
                           'description': self.Log['description'],
                           'function': self.Log['function'],
                           'value': self.Log['value'],
                           'units': self.Log['units'],
                           'value_start_date': self.Log['value_start_date'],
                           'value_end_date': self.Log['value_end_date'],
                           'CSVOutputFilename': self.Log['logoutputfilename']})

        # write the assembled dataframe out to Log.csv in the given directory
        df.to_csv(os.path.join(images_path, 'Log.csv'), index=False)

    def addLogEntry(self, keysvalues, isdata=False):
        """
        Add one entry to the log, appending a value to each named column.

        Parameters
        ----------
        keysvalues : dict
            Dictionary of ``{column_name: value}`` pairs to append.
        isdata : bool, optional
            If ``True``, ensures every standard "data" column gets an
            entry (defaulting to ``''`` if not supplied in
            ``keysvalues``) so the row stays aligned with data-related
            log entries (default ``False``).

        Returns
        -------
        None
            Updates ``self.Log`` in place.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> logger.addLogEntry({'type': 'plot', 'name': 'Shasta Temp'}, isdata=True)
        """

        # Append each supplied value to its matching column.
        for key in keysvalues.keys():
            self.Log[key].append(keysvalues[key])
        if isdata:
            # For data-related log entries (plots/tables/profiles), make
            # sure every "data" column gets an entry even if the caller
            # didn't supply one, so equalizeLog() doesn't need to pad
            # these specific columns later and the row stays aligned.
            allkeys = ['type', 'name', 'function', 'description', 'value', 'units',
                       'value_start_date', 'value_end_date', 'logoutputfilename']
            # fill in an empty placeholder for any expected data column not already supplied
            for key in allkeys:
                if key not in keysvalues.keys():
                    self.Log[key].append('')

    def addSimLogEntry(self, accepted_IDs, SimulationVariables, observedDir):
        """
        Add one log row per accepted simulation ID with its metadata.

        Parameters
        ----------
        accepted_IDs : list
            Simulation IDs to log entries for.
        SimulationVariables : dict
            Dictionary of settings keyed by simulation ID, each expected
            to contain ``'StartTimeStr'``, ``'EndTimeStr'``,
            ``'LastComputed'``, ``'program'``, ``'modelAltName'``,
            ``'alternativeFpart'``, and ``'alternativeDirectory'``.
        observedDir : str
            Path to the observed-data directory to record for each row.

        Returns
        -------
        None
            Updates ``self.Log`` in place.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> logger.addSimLogEntry(['base', 'alt_1'], SimulationVariables, '/path/to/observed')
        """

        # One log row per accepted simulation ID, capturing the
        # simulation-level metadata (time range, program, alternative
        # name, etc.) rather than any specific plot/data value.
        for ID in accepted_IDs:
            # log the ID and its full settings dict for traceability/debugging
            WF.print2stdout('ID:', ID)
            WF.print2stdout('Simvars:', SimulationVariables[ID])
            # append this simulation's metadata to each corresponding log column
            self.Log['observed_data_path'].append(observedDir)
            self.Log['start_time'].append(SimulationVariables[ID]['StartTimeStr'])
            self.Log['end_time'].append(SimulationVariables[ID]['EndTimeStr'])
            self.Log['compute_time'].append(SimulationVariables[ID]['LastComputed'])
            self.Log['program'].append(SimulationVariables[ID]['program'])
            self.Log['alternative_name'].append(SimulationVariables[ID]['modelAltName'])
            self.Log['fpart'].append(SimulationVariables[ID]['alternativeFpart'])
            self.Log['program_directory'].append(SimulationVariables[ID]['alternativeDirectory'])