import os
import sys

import numpy as np
import datetime as dt
import pandas as pd
from scipy.interpolate import interp1d
from collections import Counter
import linecache

import WAT_Functions as WF
import WAT_Time as WT


class W2_Results(object):
    '''
    Represents the results of a CE-QUAL-W2 (W2) model run, providing
    methods to locate, parse, and read W2 control files and output files
    (profile temperatures, structured time series, and general time
    series) for use in Reclamation's WTMP reporting/plotting tools.

    Attributes
    ----------
    W2_path : str
        Path to the W2 model installation/run.
    alt_name : str
        Name of the run alternative, used for display/pathing purposes.
    run_path : str
        Directory containing the specific W2 alternative run.
    starttime : datetime.datetime
        Start time of the simulation.
    endtime : datetime.datetime
        End time of the simulation.
    Report : object
        Instance of the main report script, used for debug flags and
        shared reporting utilities.
    control_file : str
        Full path to the discovered W2 control file (CSV or NPT format).
    control_file_type : str
        Either 'csv' or 'npt', indicating the format of the control file.
    '''

    def __init__(self, W2_path, alt_name, alt_Dir, starttime, endtime, Report):
        '''
        Initialize a W2_Results instance by locating the control file for
        the given run, parsing it, and building the internal time series
        needed to interpolate irregular W2 output onto a regular time
        grid.

        Parameters
        ----------
        W2_path : str
            Path to the W2 run.
        alt_name : str
            Name of run alternative for pathing, e.g. 'Shasta from DSS 14'.
        alt_Dir : str
            Directory of the alternative.
        starttime : datetime.datetime
            Start time of simulation.
        endtime : datetime.datetime
            End time of simulation.
        Report : object
            Instance from the main report script, used for debug flags.

        Returns
        -------
        None
            This is a constructor and does not return a value.

        Raises
        ------
        SystemExit
            Raised (via `sys.exit(1)`) if neither a CSV nor NPT control
            file can be found in `alt_Dir`.

        Examples
        --------
        >>> results = W2_Results('/path/to/w2', 'Alt1', '/path/to/alt1',
        ...                       starttime, endtime, Report)
        '''

        # store basic run metadata as instance attributes
        self.W2_path = W2_path
        # name of the alternative, used later for labeling/pathing
        self.alt_name = alt_name #confirm this terminology
        # directory containing this specific alternative's run files
        self.run_path = alt_Dir
        # simulation start/end times, passed in from the calling report
        self.starttime = starttime
        self.endtime = endtime
        # keep a reference to the parent Report object for debug flags, etc.
        self.Report = Report
        # self.interval_min = interval_min #output time series

        # W2 control files can come in either CSV or legacy NPT format, so check both
        # build the expected path for each possible control file naming convention
        control_file_csv = os.path.join(self.run_path, 'w2_con.csv') #this should always be the same, UNTIL IT WASNT
        control_file_npt = os.path.join(self.run_path, 'w2_con.npt') #this should always be the same, UNTIL IT WASNT
        # prefer CSV format if present
        if os.path.exists(control_file_csv):
            self.control_file = control_file_csv
            self.control_file_type = 'csv'
        # otherwise fall back to the legacy NPT format
        elif os.path.exists(control_file_npt):
            self.control_file = control_file_npt
            self.control_file_type = 'npt'
        else:
            # neither format found, so we can't proceed with this run
            WF.print2stderr('Unknown or missing W2 control file.')
            sys.exit(1)

        # parse the control file into usable sections
        self.readControlFile()
        # dates are output irregular, so we need to build a regular time series to interpolate to
        self.buildTimes()
        # NPT and CSV formats store the output file name under different cards
        if self.control_file_type == 'npt': #turn off, make user input full W2 file instead of trying to build it?
            self.getOutputFileName_NPT() #get the W2 sanctioned output file name convention
        elif self.control_file_type == 'csv':
            self.getOutputFileName_CSV()

    def buildTimes(self):
        '''
        Build the two regular time series used to snap irregular W2
        output onto, since W2 supports two different output intervals:
        one for general time series results (TSR) and one for QWO/TWO
        withdrawal output files (WDO).

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value. Instead, it sets the
            following attributes on the instance:
                self.wdo_jd_dates : numpy.ndarray
                self.wdo_dt_dates : numpy.ndarray
                self.jd_dates : numpy.ndarray
                self.dt_dates : numpy.ndarray

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.buildTimes()
        >>> results.jd_dates[:3]
        array([1.0, 1.04166667, 1.08333333])
        '''

        # pull the interval settings (tsr_interval, wdo_interval) from the control file
        self.getInterval()
        # pull the simulation start/end times (as both jdate and datetime)
        self.getW2StartTime()
        # build the WDO (withdrawal, QWO/TWO) time series using its own interval
        self.wdo_jd_dates, self.wdo_dt_dates = self.buildTimesbyInterval(self.tmstrtJDate,
                                                                         self.tmendJDate,
                                                                         self.wdo_interval)

        #Above case is ONLY for QWO files
        # build the general TSR time series using its own (usually different) interval
        self.jd_dates, self.dt_dates = self.buildTimesbyInterval(self.tmstrtJDate,
                                                                 self.tmendJDate,
                                                                 self.tsr_interval)

    def readControlFile(self):
        '''
        Read and format the lines of the W2 control file into logical
        "sections" (usually grouped by a header line and the values that
        follow it), delegating to the appropriate parser based on
        control file type.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value. Instead, it sets the
            following attributes on the instance:
                self.cf_lines : numpy.ndarray
                    Raw lines from the control file.
                self.line_sections : dict or list
                    Parsed sections of the control file.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.readControlFile()
        >>> results.line_sections is not None
        True
        '''

        # read the raw file lines first
        self.cf_lines = self.getControlFileLines(self.control_file)
        # then dispatch to the correct format-specific parser
        if self.control_file_type == 'npt':
            # NPT files use fixed-width columns and are parsed differently than CSV
            self.line_sections = self.formatNPTCFLines(self.cf_lines)
        else: #csv
            # CSV files are comma (or whitespace) delimited
            self.line_sections = self.formatCSVCFLines(self.cf_lines)

    def getInterval(self):
        '''
        Get the output time series intervals from the W2 control file.
        There are two intervals defined: one for the general time series
        results (TSR) and one for the irregular withdrawal output files
        (WDO, i.e. QWO/TWO files).

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value. Instead, it sets the
            following attributes on the instance:
                self.tsr_interval : float
                self.wdo_interval : float

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though a
            `KeyError` or `IndexError` may occur if the expected control
            variables are missing from the control file.

        Examples
        --------
        >>> results.getInterval()
        >>> results.tsr_interval
        60.0
        '''

        # pull the TSR and WDO frequency values, format depends on control file type
        if self.control_file_type == 'npt':
            # NPT stores frequency values as the first element of a list under each card
            tsr_interval = float(self.getNPTControlVariable(self.line_sections, 'TSR FREQ')[0])
            wdo_interval = float(self.getNPTControlVariable(self.line_sections, 'WITH FRE')[0])
        else:
            # CSV stores the interval value at a fixed column position (index 5) in each row
            tsr = self.getCSVControlVariable(self.line_sections, 'TSR')
            tsr_interval = float(tsr[5])
            wdo = self.getCSVControlVariable(self.line_sections, 'WDO')
            wdo_interval = float(wdo[5])
        # store both intervals for later use in buildTimes()
        self.tsr_interval = tsr_interval
        self.wdo_interval = wdo_interval

    def getW2StartTime(self):
        '''
        Get the simulation time window (start and end times) from the W2
        control file, converting the jdate values into datetime objects.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value. Instead, it sets the
            following attributes on the instance:
                self.tmstrt : float
                    Start time as a jdate.
                self.tmend : float
                    End time as a jdate.
                self.tmyear : int
                    Reference year for the jdate conversion.
                self.tmstrtJDate : datetime.datetime
                    Start time as a datetime.
                self.tmendJDate : datetime.datetime
                    End time as a datetime.

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though a
            `KeyError` may occur if expected control variables are
            missing.

        Examples
        --------
        >>> results.getW2StartTime()
        >>> results.tmstrtJDate
        datetime.datetime(2015, 1, 1, 0, 0)
        '''

        # NPT format keeps start/end/year grouped under one 'TIME CON' section
        if self.control_file_type == 'npt':
            # pull the whole TIME CON dict, then index into it for each specific field
            timecon = self.getNPTControlVariable(self.line_sections, 'TIME CON')
            self.tmstrt = float(timecon['TMSTRT'])
            self.tmend = float(timecon['TMEND'])
            self.tmyear = int(timecon['YEAR'])
            # convert both jdate values to real datetimes using the reference year
            startend = WT.JDateToDatetime([self.tmstrt, self.tmend], self.tmyear)
            self.tmstrtJDate = startend[0]
            self.tmendJDate = startend[1]
        else:
            # CSV format stores each variable under its own separate card name
            self.tmstrt = float(self.getCSVControlVariable(self.line_sections, 'TMSTRT'))
            self.tmend = float(self.getCSVControlVariable(self.line_sections, 'TMEND'))
            self.tmyear = int(self.getCSVControlVariable(self.line_sections, 'YEAR'))
            # convert both jdate values to real datetimes using the reference year
            startend = WT.JDateToDatetime([self.tmstrt, self.tmend], self.tmyear)
            self.tmstrtJDate = startend[0]
            self.tmendJDate = startend[1]

    def get_tempprofile_layers(self):
        '''
        Get the temperature profile output layers and their corresponding
        segments from the control file.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value. Instead, it sets the
            following attributes on the instance:
                self.layers : numpy.ndarray or float
                    Layer numbers for profile output.
                self.segments : numpy.ndarray or int
                    Segment numbers corresponding to each layer.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.get_tempprofile_layers()
        >>> results.layers
        array([1., 2., 3.])
        '''

        # NPT format returns multiple layer/segment values as lists under a single card
        if self.control_file_type == 'npt':
            # filter out any empty strings before converting to numeric arrays
            self.layers = np.asarray([float(n) for n in self.getNPTControlVariable(self.line_sections, 'TSR LAYE') if n != ''])
            self.segments = np.asarray([int(n) for n in self.getNPTControlVariable(self.line_sections, 'TSR SEG') if n != ''])

        else:
            # CSV format returns a single layer/segment value rather than a list
            self.layers = float(self.getCSVControlVariable(self.line_sections, 'TSR LAYE'))
            self.segments = int(self.getCSVControlVariable(self.line_sections, 'TSR SEG'))

    def getOutputFileName_NPT(self):
        '''
        Get the name of the output file(s) as defined in an NPT-format
        control file.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value. Instead, it sets the
            following attribute on the instance:
                self.output_file_name : str

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.getOutputFileName_NPT()
        >>> results.output_file_name
        'spr.opt'
        '''

        # self.output_file_name = self.getControlVariable(self.line_sections, 'TSR FILE')[0]
        # grab the raw TSR FILE card contents, which may be split across multiple fields
        output_file_name = self.getNPTControlVariable(self.line_sections, 'TSR FILE')
        # the file name is sometimes split across multiple lines/fields, so rejoin them
        output_file_name = ''.join(output_file_name) #sometimes this goes multi line, but it shouldnt
        # if len(output_file_name) > 0:
        # store the final joined file name
        self.output_file_name = output_file_name

    def getOutputFileName_CSV(self):
        '''
        Get the name of the output file(s) as defined in a CSV-format
        control file.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value. Instead, it sets the
            following attribute on the instance (only if a value is
            found):
                self.output_file_name : str

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.getOutputFileName_CSV()
        >>> results.output_file_name
        'spr.csv'
        '''

        # grab the whole TSR row from the CSV control file sections
        output_file_name = self.getCSVControlVariable(self.line_sections, 'TSR')
        # 4th field (index 3) in the TSR row holds the file name
        if len(output_file_name) > 0:
            self.output_file_name = output_file_name[3]

    def getControlFileLines(self, control_file):
        '''
        Read all lines from the given control file.

        Parameters
        ----------
        control_file : str
            Full path to the control file.

        Returns
        -------
        numpy.ndarray
            Array of all lines in the control file.

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though
            `FileNotFoundError` may occur if `control_file` is invalid.

        Examples
        --------
        >>> lines = results.getControlFileLines('/path/to/w2_con.csv')
        '''

        # open the file, read every line into a list, then close explicitly
        file_read = open(control_file, 'r')
        file_lines = file_read.readlines()
        file_read.close()
        # convert to a numpy array for easier downstream slicing/indexing
        return np.asarray(file_lines)

    def formatNPTCFLines(self, cf_lines):
        '''
        Separate NPT-format control file lines into sections, based on
        blank-line separators in the file. Each field within a line is
        assumed to occupy an 8-character-wide column.

        Control files are generally formatted like:

        NPT FORMAT::

            NWB	 NBR	 IMX	 KMX	 NPROC	 CLOSEC
            1	  4	      83	 135	     1	     ON

        OR::

            WD1
            OFF
            52
            644.35
            2
            135

        The file is then split into sections for easier parsing, with
        sections delimited by blank lines in the file.

        Parameters
        ----------
        cf_lines : numpy.ndarray
            Control file lines from `self.getControlFileLines()`.

        Returns
        -------
        dict
            Dictionary of parsed sections, keyed by the section's main
            header/flag. Depending on the structure of a given section,
            the value may be:
                1. A dict of lists, if subitems share the same headers
                   (minus the first).
                2. A dict of dicts, if subitems have differing headers
                   (minus the first).
                3. A list, if there are no subitems and all headers are
                   the same.
                4. A dict, if there are no subitems and headers differ.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> sections = results.formatNPTCFLines(results.cf_lines)
        >>> 'TIME CON' in sections
        True
        '''

        # dictionary that will hold all parsed sections, keyed by section header
        sections = {}
        # flag tracking whether we've already read the header row of the current section
        got_headers = False
        # placeholder structure (dict or list) used as a template for the current section's contents
        sections_contents_template = []
        # actual contents accumulated for the current section
        sections_contents = []
        # the primary header/flag name for the current section
        main_flag = None
        # secondary header names (columns after the first) for the current section
        other_headers = []
        # skip the first ten lines, they're header/comment garbage not needed for parsing
        for line in cf_lines[10:]: #skip the first ten lines, theyre garb.
            # line = line.strip()
            # split the line into fixed 8-character-wide fields
            line = [line[i:i+8] for i in range(0, len(line), 8)] #npt files are spaced 8 chars wide
            # drop a trailing newline-only field if present
            if line[-1] == '\n':
                line = line[:-1]
            # strip whitespace from every field
            line = [n.strip() for n in line]

            if (len(line) == 1 and line[0] == '') or (len(line) == 0):
                #store contents and reset
                # blank line marks the end of a section, so save what we've collected
                if main_flag != None:
                    sections[main_flag] = sections_contents
                else:
                    # nothing was ever collected for this "section" (e.g. leading blank line), skip
                    continue
                    # print('Main flag none. Skip.')
                # reset all section-tracking variables for the next section
                got_headers = False
                sections_contents_template = []
                sections_contents = []
                main_flag = None
                other_headers = []

            else:

                if not got_headers:
                    #this is our header
                    # first non-blank line of a section defines the headers
                    main_flag = line[0]
                    other_headers = line[1:]
                    if len(line) > 1: #if there are more headers
                        # count how many times each secondary header repeats
                        other_headers = Counter(line[1:])
                        if max(other_headers.values()) == 1: #every other header is unique
                            # headers are all unique, so this section will be a dict keyed by subitem
                            # sections_contents_template = {n: {} for n in other_headers}
                            sections_contents_template = {}
                        else:
                            # headers repeat, so this section will just be a flat list
                            sections_contents_template = []
                    else:
                        # no secondary headers at all, so this is a simple list-only section
                        sections[main_flag] = []
                    got_headers = True
                else:
                    if line[0] != '':
                        #we have a sub item, and not just a big list, or some features
                        # this row starts a new named subitem within the section
                        subitem = line[0]
                        subitem_contents = line[1:]
                        sections_contents = sections_contents_template
                        if isinstance(sections_contents, dict):
                            # build a dict of header->value for this subitem, filling gaps with empty strings
                            subsections_contents = {n: {} for n in other_headers}
                            for i, key in enumerate(other_headers):
                                try:
                                    subsections_contents[key] = subitem_contents[i]
                                except IndexError:
                                    subsections_contents[key] = ''
                            sections_contents[subitem] = subsections_contents
                        elif isinstance(sections_contents, list):
                            # just append the values onto the running list
                            sections_contents += subitem_contents
                    else:
                        # continuation row with no subitem name, append values to current structure
                        subitem_contents = line[1:]
                        sections_contents = sections_contents_template
                        if isinstance(sections_contents, dict):
                            # fill in values for each header, defaulting to empty string if missing
                            for i, key in enumerate(other_headers):
                                try:
                                    sections_contents[key] = subitem_contents[i]
                                except IndexError:
                                    sections_contents[key] = ''
                        elif isinstance(sections_contents, list):
                            # just append the values onto the running list
                            sections_contents += subitem_contents

        # return the fully assembled dictionary of sections
        return sections

    def formatCSVCFLines(self, cf_lines):
        '''
        Separate CSV-format control file lines into sections, based on
        blank-line separators in the file.

        Control files are generally formatted like:

        CSV FORMAT::

            NWB	 NBR	 IMX	 KMX	 NPROC	 CLOSEC
            1	  4	      83	 135	     1	     ON

        OR::

            WD1
            OFF
            52
            644.35
            2
            135

        The file is then split into sections for easier parsing, with
        sections delimited by blank lines in the file.

        Parameters
        ----------
        cf_lines : numpy.ndarray
            Control file lines from `self.getControlFileLines()`.

        Returns
        -------
        list
            List of parsed sections, each typically represented as a
            `[header, body]` pair.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> sections = results.formatCSVCFLines(results.cf_lines)
        '''

        # list that will hold all fully parsed sections
        sections = []
        # working buffer for the section currently being accumulated
        small_section = []
        for line in cf_lines:
            # split on commas for CSV, or on triple-space for whitespace-delimited variants
            if self.control_file_type == 'csv':
                line = line.strip().split(',')
            else:
                line = line.strip().split('   ')

            # treat a line whose first field is empty as an effectively empty line
            if len(line) > 0:
                if line[0] == '':
                    line = []

            # drop any empty fields left over from splitting
            line = [n.strip() for n in line if n != '']


                # line = ''.join(list(filter((',').__ne__, list(line))))
            if len(line) == 0 and len(small_section) == 0:
                # nothing collected yet and this is a blank line, skip it
                continue
            if len(line) == 0 and len(small_section) != 0:
                #check section here
                # blank line after content means the current section is complete
                if len(small_section) > 2:
                    # more than 2 rows collected, so treat row 0 as header and combine the rest into a body list
                    header = small_section[0]
                    body = []
                    for n in small_section[1:]:
                        if len(n) > 1:
                            # multi-value row, append as-is
                            body.append(n)
                        else:
                            # single-value row, unwrap it from its list
                            body.append(n[0])
                    small_section = [header, body]
                if len(small_section) > 1:
                    # pad the body so it's the same length as the header row
                    if len(small_section[0]) > len(small_section[1]):
                        small_section[1] += [''] * (len(small_section[0]) - len(small_section[1]))
                # store the completed section and reset the buffer
                sections.append(small_section)
                small_section = []
            else:
                # still within a section, keep accumulating lines
                small_section.append(line)

        # return the full list of parsed sections
        return sections

    def getCSVControlVariable(self, lines_sections, variable):
        '''
        Parse the split CSV control file sections from
        `self.formatCSVCFLines()` for a given control card. Cards usually
        preface headers in the control file. For example::

            DLT MAX   DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX
                      3600.00

            DLT FRN     DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF
                       0.900

        If the user wanted the DLT Max value, they would search for
        'DLT MAX'. If there is only one instance of the flag, a single
        array/value is returned; otherwise a list of results is returned
        and the caller can narrow it down.

        Parameters
        ----------
        lines_sections : list
            Formatted control file sections from `self.formatCSVCFLines()`.
        variable : str
            Control card/variable name to search for.

        Returns
        -------
        numpy.ndarray or str
            The matched value(s) for the requested control variable.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> value = results.getCSVControlVariable(results.line_sections, 'TMSTRT')
        '''

        # find every section whose header line contains the requested variable name
        variable_lines_idx = [i for i, line in enumerate(lines_sections) if variable in line[0]]
        # collect the output value(s) for each matching section
        outputs = []
        for var_line_idx in variable_lines_idx:
            # for line in lines_sections[var_line_idx].split('\n')[1:]: #skip header
            line = lines_sections[var_line_idx]
            if len(line[0]) != len(line[1]): #for cases of vert stack in csv
                # header/body lengths mismatch, values are just stacked vertically, take body directly
                cur_otpt = line[1]
            else:
                # find the position in the header row matching the variable, then grab the value at that position
                idx = np.where(np.asarray(line[0]) == variable)
                cur_otpt = np.asarray(line[1])[idx]
                if len(cur_otpt) > 1:
                    # multiple matches found, take the first non-empty one
                    for item in cur_otpt:
                        if item != '':
                            cur_otpt = np.asarray(line[1])[idx][0]
            outputs.append(cur_otpt)

        # if more than one matching section was found, just take the first result
        if len(outputs) > 1:
            return outputs[0][0]
        # otherwise return the single match found
        return outputs[0]

    def getNPTControlVariable(self, lines_sections, variable):
        '''
        Parse the split NPT control file sections from
        `self.formatNPTCFLines()` for a given control card. Cards usually
        preface headers in the control file. For example::

            DLT MAX   DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX
                      3600.00

            DLT FRN     DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF
                       0.900

        If the user wanted the DLT Max value, they would search for
        'DLT MAX'. If there is only one instance of the flag, a single
        array/value is returned; otherwise a list of results is returned
        and the caller can narrow it down.

        Parameters
        ----------
        lines_sections : dict
            Formatted control file sections from `self.formatNPTCFLines()`.
        variable : str
            Control card/variable name to search for.

        Returns
        -------
        list, dict, or None
            The parsed section contents for the requested variable, or
            `None` if the variable is not present in `lines_sections`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> value = results.getNPTControlVariable(results.line_sections, 'TSR FREQ')
        '''

        # dict lookup is straightforward since formatNPTCFLines() already keys by section name
        if variable in lines_sections.keys():
            # variable found, return its parsed contents directly
            return lines_sections[variable]
        else:
            # variable not present in this control file
            return None

    def buildTimesbyInterval(self, start_day, end_day, interval):
        '''
        Create a regular time series between two jdates at a given
        interval. W2 output time series are irregular, so a regular time
        series is built here for later interpolation.

        Parameters
        ----------
        start_day : float
            Start day of the simulation, as a jdate.
        end_day : float
            End day of the simulation, as a jdate.
        interval : float
            Desired output time series interval, in minutes (e.g. 60 for
            hourly, 15 for 15-minute/4-per-hour).

        Returns
        -------
        jd_dates : numpy.ndarray
            Regular time series expressed as jdates (days past Jan 1,
            starting at 1).
        dt_dates : numpy.ndarray
            Regular time series expressed as datetime objects.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> jd_dates, dt_dates = results.buildTimesbyInterval(1.0, 365.0, 60)
        '''

        # turns out jdates can be passed into timedelta (decimals) and it works correctly. Just subtract 1 becuase jdates
        # start at 1
        # convert the interval (minutes) to whole seconds, then build the regular datetime series
        dt_dates = pd.date_range(start_day,end_day,freq=dt.timedelta(seconds=np.floor(dt.timedelta(interval).total_seconds()))).to_pydatetime()
        # convert the same regular series to jdate format for convenience elsewhere
        jd_dates = np.asarray(WT.DatetimeToJDate(dt_dates))

        # return both representations of the regular time series
        return jd_dates, dt_dates

    def readProfileData(self, seg, timesteps, resultsfile=None):
        '''
        Get temperature profile values (water temperatures, elevations,
        and depths) from the output files for a given segment, dispatched
        to the appropriate format-specific reader.

        Results are organized into arrays filled with NaN by default,
        populated where valid data exists. Water temperatures are
        organized into 2-D arrays of dates by layers, so a single date
        can be indexed to retrieve all temperature layers for that
        timestep. Because W2 model output comes out on an irregular time
        series, values are interpolated to align with the desired output
        times. Water surface elevations (used for depth/elevation
        calculations elsewhere) are the same across all valid layers at a
        given timestep.

        Parameters
        ----------
        seg : int
            Segment number to read profile data for.
        timesteps : list or numpy.ndarray
            Desired output timesteps.
        resultsfile : str, optional
            Explicit results file to read from (CSV format only). If not
            given, the file is determined automatically. Default is None.

        Returns
        -------
        values : numpy.ndarray or list
            Array of water temperatures.
        elevations : numpy.ndarray or list
            Array of water surface elevations, in feet.
        depths : numpy.ndarray or list
            Array of depths, in feet.
        dates : numpy.ndarray or list
            Corresponding dates for the returned values.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> values, elevations, depths, dates = results.readProfileData(52, timesteps)
        '''

        # dispatch to the correct reader depending on control file format
        if self.control_file_type == 'npt':
            # NPT format stores each layer's output in a separate file
            values, elevations, depths, dates = self.readProfileData_NPT(seg, timesteps)
        elif self.control_file_type == 'csv':
            # CSV format stores all layers together in one results file
            values, elevations, depths, dates = self.readProfileData_CSV(seg, timesteps, resultsfile)

        # return whichever result set was produced above
        return values, elevations, depths, dates

    def readProfileData_CSV(self, seg, timesteps, resultsfile=None):
        '''
        Read temperature profile data for a given segment from a CSV
        format W2 output file. Output for CSV W2 runs is not always in
        the default spr.csv file. Headers look like::

            Constituent	Julian_day	Depth	Elevation	Seg_111	Elevation	Seg_113

        Parameters
        ----------
        seg : int
            Segment number to read profile data for.
        timesteps : list, numpy.ndarray, or str
            Desired output timesteps. If not a list/array (e.g. a string
            like 'all'), all unique available dates are returned instead.
        resultsfile : str, optional
            Explicit results file to read from. If not given, the file is
            determined via `self.getResultsFile_CSV()`. Default is None.

        Returns
        -------
        select_values : list
            Water temperature values for each requested timestep.
        select_elevations : list
            Water surface elevations (feet) for each requested timestep.
        select_depths : list
            Depths (feet) for each requested timestep.
        select_times : list
            Corresponding requested timestamps, or sorted unique available
            dates if `timesteps` was not a list/array.

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though
            empty lists are returned if the results file or requested
            segment cannot be found.

        Examples
        --------
        >>> values, elevations, depths, times = results.readProfileData_CSV(111, timesteps)
        '''

        # determine which results file to read from if not explicitly given
        if resultsfile == None:
            resultsfile = self.getResultsFile_CSV()
        # build the full path and confirm the file actually exists
        outputfile = os.path.join(self.run_path, resultsfile)
        if not os.path.exists(outputfile):
            WF.print2stdout(f'Results file {outputfile} does not exist.', debug=self.Report.debug)
            return [], [], [], []
        # load the whole CSV into a dataframe for easy column access
        output = pd.read_csv(outputfile)
        segment_header = f'Seg_{seg}'
        # segment_index = np.where(output.columns.startswith(segment_header))[0] #should only ever be 1
        # find the column corresponding to the requested segment (should only ever be one match)
        segment_index = [i for i, col in enumerate(output.columns) if col.startswith(segment_header)] #should only ever be 1
        if len(segment_index) == 0:
            # requested segment isn't present in this output file at all
            WF.print2stdout(f'ERROR: segment {seg} not found in output file {outputfile}', debug=self.Report.debug)
            return [], [], [], []
        # elevation header sits immediately before the value column in the CSV layout
        segment_elevation_header = output.columns[segment_index[0]-1]
        segment_value_header = output.columns[segment_index[0]] #stupid segment header can have a space at the end...
        # pull out the shared jdate/depth columns and this segment's value/elevation columns
        all_jdates = output['Julian_day'].values #csv depths are always the same for all segments output
        all_dtdates = WT.JDateToDatetime(all_jdates, self.starttime.year) #csv depths are always the same for all segments output
        all_depths = output['Depth'].values #csv jdates are always the same for all segments output
        all_values = output[segment_value_header].values
        all_elevations = output[segment_elevation_header].values
        # get the distinct timestamps present in the file (there may be many depth rows per timestamp)
        unique_dates = np.asarray(list(set(all_dtdates)))

        if len(unique_dates) == 0:
            # no data at all was found in this output file
            WF.print2stdout('No values found in output.', debug=self.Report.debug)
            return [], [], [], []

        if isinstance(timesteps, (list, np.ndarray)):
            # accumulate results for each requested timestep
            select_values = []
            select_elevations = []
            select_depths = []
            select_times = []
            # for each requested timestep, find the closest matching data and slice out its profile
            for t, time in enumerate(timesteps):
                timestep = WT.getIdxForTimestamp(unique_dates, time)
                if timestep > -1:#timestep in model
                    # pull all rows (depths) matching this timestamp
                    indicies = np.where(all_dtdates == unique_dates[timestep])
                    values = all_values[indicies]
                    elevations = all_elevations[indicies]
                    depths = all_depths[indicies]
                    # WSE = elevations[timestep] #Meters #get WSE
                    if not WF.checkData(elevations): #if elevations is bad, skip usually first timestep...
                        # bad/missing elevation data, append empty placeholders instead
                        select_elevations.append(np.array([]))
                        select_depths.append(np.array([]))
                        select_values.append(np.array([]))
                        select_times.append(time)
                        continue
                    # convert meters to feet (3.28084 conversion factor)
                    select_elevations.append(elevations[:] * 3.28084)
                    select_depths.append(depths[:] * 3.28084)
                    select_values.append(values[:])
                    select_times.append(time)

                else: #if timestep NOT in model, add empties
                    # no matching timestep found, append empty placeholders
                    select_values.append(np.array([])) #find WTs
                    select_elevations.append(np.array([]))
                    select_depths.append(np.array([]))
                    select_times.append(time)

            # trim all arrays down to a common length in case of mismatches
            select_values, select_elevations, select_depths = self.matchProfileLengths(select_values, select_elevations, select_depths)
            return select_values, select_elevations, select_depths, select_times,
        else:
            # not a list of timesteps, just return the full set of available dates instead
            return [], [], [], sorted(unique_dates)

    def readProfileData_NPT(self, seg, timesteps):
        '''
        Read temperature profile data for a given segment from NPT-format
        output files (one file per layer).

        Parameters
        ----------
        seg : int
            Segment number to read profile data for.
        timesteps : list or numpy.ndarray
            Desired output timesteps.

        Returns
        -------
        select_wt or wt : list or numpy.ndarray
            Water temperature values for each requested timestep (or all
            timesteps if `timesteps` is not a list/array).
        elevations : list or numpy.ndarray
            Water surface elevations (feet) for each requested timestep.
        depths : list
            Depths (feet) for each requested timestep. Empty list if
            `timesteps` is not a list/array.
        times : numpy.ndarray
            Corresponding requested timestamps, or `self.dt_dates` if
            `timesteps` is not a list/array.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> wt, elevations, depths, times = results.readProfileData_NPT(52, timesteps)
        '''

        # populate self.layers / self.segments (output at fixed depths, e.g. every 2m)
        self.get_tempprofile_layers() #get the output layers. out at 2m depths

        # accumulators for temperature and elevation series, one entry appended per layer
        wt = []
        WS_Elev = []
        # will hold the full/reference set of jday values from the first valid layer file
        profile_jdates = []

        # loop over every output layer, reading its per-segment output file
        for i in range(1,len(self.layers)+1):
            # WF.print2stdout('{0} of {1}'.format(i, len(self.layers)+1))
            # build the expected NPT output filename convention for this layer/segment
            ofn = '{0}_{1}_seg{2}.{3}'.format(self.output_file_name.split('.')[0],
                                              i,
                                              seg,
                                              self.output_file_name.split('.')[1])
            # skip this layer entirely if it doesn't belong to the requested segment
            if self.segments[i-1] != int(seg):
                continue
            ofn_path = os.path.join(self.run_path, ofn)
            if not os.path.exists(ofn_path):
                # file missing for this layer, log and move to the next layer
                WF.print2stdout('File {0} not found'.format(ofn_path))
                continue
            headerline=0
            # scan the file to find the header row (starts with 'jday')
            with open(ofn_path) as ofnf:
                for li, line in enumerate(ofnf):
                    if line.lower().startswith('jday'):
                        headerline=li
                        break

            # read the actual data starting from the detected header row
            op_file = pd.read_csv(ofn_path, header=headerline, skip_blank_lines=False)
            # normalize column names to lowercase for consistent access
            op_file.columns = op_file.columns.str.lower()

            # Get the full set of result JDATE values from the uppermost layer file with data in it
            if len(profile_jdates) == 0:
                profile_jdates = op_file['jday']

            if len(op_file['jday']) > 1:
                # if the new dataset doesn't have the full set of JDAY values,
                #    append new empty rows with the missing dates to the dataframe,
                #    then sort to put them in chronlogical order again
                file_jdates = op_file['jday']
                if len(file_jdates) < len(profile_jdates):
                    # this layer's file is missing some timestamps present in the reference layer
                    new_dates = []
                    search_index = 0
                    # Find the missing JDATE values
                    for jd in profile_jdates:
                        try:
                            jdf = file_jdates[search_index]
                        except KeyError:
                            # JDATE values in the file don't go to the end of the run
                            new_dates.append(jd)
                            continue
                        if(jd < jdf):
                                # reference date isn't present in this file, mark it as missing
                                new_dates.append(jd)
                        else:
                            # dates match up so far, advance to the next comparison point
                            search_index += 1

                    # build, concatonate, and sort-in rows for the missing JDATE values
                    new_values = np.full((len(new_dates),len(op_file.columns)), np.nan)
                    for i, jd in enumerate(new_dates):
                        # fill in just the jday column for each new placeholder row; rest stays NaN
                        new_values[i,0] = jd
                    # append the placeholder rows and re-sort chronologically
                    op_file = pd.concat([op_file, pd.DataFrame(columns=op_file.columns, data=new_values, index=range(1,len(new_dates)+1))])
                    op_file = op_file.sort_values('jday', ignore_index=True)

                # pull out the temperature and elevation columns for this layer
                wt_vals = op_file['t2(c)']
                elev_vals = op_file['elws(m)']
                wt.append(wt_vals.values)
                WS_Elev.append(elev_vals.values)

        # pad each layer's series to the full expected length, then transpose to date x layer
        max_len = len(self.jd_dates)
        # pad shorter layer arrays with NaN so all layers align to the same length before stacking
        wt = np.asarray([np.pad(array, (0, max_len - len(array)), mode='constant', constant_values=np.nan) for array in wt]).T
        WS_Elev = np.asarray([np.pad(array, (0, max_len - len(array)), mode='constant', constant_values=np.nan) for array in WS_Elev]).T

        # get just the layers that belong to the requested segment
        segment_layers = np.asarray([n for i, n in enumerate(self.layers) if self.segments[i] == int(seg)])

        if isinstance(timesteps, (list, np.ndarray)):
            # accumulators for the requested-timestep results
            select_wt = []
            elevations = []
            depths = []
            times = []
            # for each requested timestep, find the matching data and compute elevation/depth per layer
            for t, time in enumerate(timesteps):
                e = []
                timestep = WT.getIdxForTimestamp(self.dt_dates, time)
                if timestep > -1:#timestep in model
                    try:
                        WSE = WS_Elev[timestep] #Meters #get WSE
                    except IndexError:
                        # index somehow out of range, treat as no WSE available
                        WSE = []
                    if not WF.checkData(WSE): #if WSE is bad, skip usually first timestep...
                        # bad/missing WSE data, append empty placeholders instead
                        elevations.append(np.array([]))
                        depths.append(np.array([]))
                        select_wt.append(np.array([]))
                        times.append(time)
                        continue
                    # find the first valid (non-NaN) WSE value across the layers
                    WSE = WSE[np.where(~np.isnan(WSE))][0] #otherwise find valid
                    # WSE_array = np.full((self.layers.shape), WSE)
                    # broadcast the WSE value across all layers, then compute elevation-above-layer as feet
                    WSE_array = np.full((segment_layers.shape), WSE)
                    e = (WSE_array - segment_layers) * 3.28084
                    # trim elevation array to match however much temperature data is available
                    e = e[:len(wt[timestep])]
                    select_wt.append(wt[timestep][:]) #find WTs

                else: #if timestep NOT in model, add empties
                    # requested time has no matching model timestep, append empty placeholders
                    select_wt.append(np.array([])) #find WTs
                    elevations.append(np.array([]))
                    depths.append(np.array([]))
                    times.append(time)
                # append the computed elevation and depth arrays for this timestep
                elevations.append(np.asarray(e)) #then append for timestep
                depths.append((segment_layers * 3.28084)[:len(e)]) #append dpeths
                times.append(time) #get time
            # trim all arrays down to a common length in case of mismatches
            select_wt, elevations, depths = self.matchProfileLengths(select_wt, elevations, depths)

            return select_wt, elevations, depths, np.asarray(times)
        else:
            # not a list of timesteps, so return the entire dataset for all times
            elevations = ((WS_Elev - segment_layers) * 3.28084)[:len(wt)]
            return wt, elevations, [], self.dt_dates

    def readProfileTopwater(self, seg, timesteps):
        '''
        Get the water surface elevation (WSE) for each requested timestep,
        for use in filtering/masking profile contour plots.

        Parameters
        ----------
        seg : int
            Segment number to compute WSE for.
        timesteps : list, numpy.ndarray, or str
            Desired output timesteps, or 'all' to return the full WSE
            time series.

        Returns
        -------
        list or numpy.ndarray
            List of WSE values corresponding to each requested timestep
            (NaN where unavailable), or the full WSE array (converted to
            feet) if `timesteps` is not a list/array.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> wse = results.readProfileTopwater(52, timesteps)
        '''

        # populate self.layers / self.segments (output at fixed depths, e.g. every 2m)
        self.get_tempprofile_layers() #get the output layers. out at 2m depths

        # initialize the full WSE array (layers x regular timesteps) as NaN
        WS_Elev = np.full((len(self.layers), len(self.jd_dates)), np.nan)

        # read each layer's output file and interpolate WSE onto the regular time grid
        for i in range(1,len(self.layers)+1):
            # build the expected filename for this layer/segment combination
            ofn = '{0}_{1}_seg{2}.{3}'.format(self.output_file_name.split('.')[0],
                                              i,
                                              seg,
                                              self.output_file_name.split('.')[1])
            ofn_path = os.path.join(self.run_path, ofn)
            if not os.path.exists(ofn_path):
                # missing file for this layer, log and skip
                WF.print2stdout('File {0} not found'.format(ofn_path))
                continue
            headerline=0
            # scan the file to find the header row (starts with 'jday')
            with open(ofn_path) as ofnf:
                for li, line in enumerate(ofnf):
                    if line.lower().startswith('jday'):
                        headerline=li
                        break

            # read the layer's irregular output data starting at the header row
            op_file = pd.read_csv(ofn_path, header=headerline, skip_blank_lines=False)
            op_file.columns = op_file.columns.str.lower()
            if len(op_file['jday']) > 1:
                # build an interpolator over this layer's irregular output, then sample it at our regular jdates
                Elev_interp = interp1d(op_file['jday'], op_file['elws(m)'])
                # only interpolate within the range actually covered by this layer's data
                jdate_minmask = np.where(min(op_file['jday']) <= self.jd_dates)
                jdate_maxmask = np.where(max(op_file['jday']) >= self.jd_dates)
                jdate_msk = np.intersect1d(jdate_maxmask, jdate_minmask)
                # start with an all-NaN array, then fill in only the valid interpolated range
                wsElev_ts_Vals = np.full(len(self.jd_dates), np.nan)
                wsElev_ts_Vals[jdate_msk] = Elev_interp(self.jd_dates[jdate_msk])
                WS_Elev[i-1] = wsElev_ts_Vals

        # transpose so rows are timesteps and columns are layers
        WS_Elev = np.asarray(WS_Elev).T

        if isinstance(timesteps, (list, np.ndarray)):

            # accumulate one WSE value per requested timestep
            WSE_out = []
            # for each requested timestep, find the nearest match and pull the first valid WSE
            for t, time in enumerate(timesteps):
                timestep = WT.getIdxForTimestamp(self.dt_dates, time)
                if timestep > -1:#timestep in model
                    WSE = WS_Elev[timestep] #Meters #get WSE
                    if not WF.checkData(WSE): #if WSE is bad, skip usually first timestep...
                        # bad/missing data for this timestep, record NaN
                        WSE_out.append(np.nan)
                        continue
                    # find the first valid (non-NaN) WSE across layers
                    WSE = WSE[np.where(~np.isnan(WSE))][0] #otherwise find valid
                    WSE_out.append(WSE)

                else: #if timestep NOT in model, add empties
                    # no matching model timestep, record NaN
                    WSE_out.append(np.nan)

            return WSE_out
        else:
            # not a list of timesteps, return the full WSE series (feet) for the top layer
            return WS_Elev[:,0] * 3.28084

    def readStructuredTimeSeries(self, output_file_name, structure_nums, skiprows=2):
        """
        Read a structured (multi-structure) W2 output file. Output files
        usually have a header with repeated headers for each structure::

             Branch:           1  # of structures:          23  outlet temperatures
            JDAY      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL

        Individual structures are distinguished by the number of times a
        given header repeats.

        Parameters
        ----------
        output_file_name : str
            Name of output file within `self.run_path`.
        structure_nums : list
            Number values of structures to output. Negative values are
            treated as reverse indices from the last structure.
        skiprows : int, optional
            Number of header rows to skip before the header line.
            Default is 2.

        Returns
        -------
        dates : numpy.ndarray
            Array of jdate values.
        values : dict
            Dictionary keyed by structure number, each containing a dict
            of parameter name to numpy array of values.

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though
            empty lists are returned if the output file cannot be found.

        Examples
        --------
        >>> dates, values = results.readStructuredTimeSeries('spillway.opt', [1, 2, 3])
        """

        # confirm the requested output file actually exists before doing anything else
        ofn_path = os.path.join(self.run_path, output_file_name)
        if not os.path.exists(ofn_path):
            WF.print2stdout(f'File {ofn_path} not found!')
            return [], []

        # normalize requested structure numbers to ints
        structure_nums = [int(n) for n in structure_nums]
        values = {}

        # read the header row to determine which parameter columns are present
        with open(ofn_path, 'r') as o:
            for i, line in enumerate(o):
                if i == skiprows-1:
                    # header row found: strip commas, lowercase, split into fields, and skip the jday column
                    headers = line.strip().lower().replace(',','').split()[1:] #skipjdate..
                    break
        # count how many times each header repeats (this tells us how many structures exist)
        header_count = Counter(headers)
        # reduce to just the unique parameter names
        headers = list(set(headers))

        # read the full data file using the detected header row
        stsf = pd.read_csv(ofn_path, header=skiprows-1, delim_whitespace=True)
        # normalize column names: lowercase and strip stray commas
        stsf.columns = stsf.columns.str.lower()
        stsf.columns = [n.replace(',','') for n in stsf.columns]
        # for each requested structure, pull out its columns for every header/parameter
        for structure_num in structure_nums:
            if structure_num < 0:
                # negative index means "count from the end", using the smallest header repeat count as the total
                structure_num = min(header_count.values()) + structure_num+1 #reverse index the fun way, use min len incase doesnt match for some reason
            if structure_num not in values.keys():
                # first time seeing this structure number, initialize its dict
                values[structure_num] = {}
            for header in headers:
                # repeated header columns are suffixed with '.N' for structures after the first
                if structure_num == 1:
                    hname = header
                else:
                    hname = header+'.{0}'.format(structure_num-1)
                # parse the column values to floats, stripping any stray commas first
                vals = np.asarray([float(str(n).replace(',','')) for n in stsf[hname].tolist()])
                values[structure_num][header.lower()] = vals

        # extract and clean up the jday column separately
        dates = stsf['jday'].tolist()
        dates = np.asarray([float(str(n).replace(',', '')) for n in dates])

        return dates, values

    def filterByParameter(self, values, line_info):
        '''
        W2 results files contain multiple parameters in a single file, so
        this filters down to the single parameter requested by the given
        line settings.

        Parameters
        ----------
        values : dict
            Dictionary of lists/dicts of values, keyed by structure or
            member.
        line_info : dict
            Line settings dictionary. Must contain a 'parameter' key (one
            of 'flow', 'temperature', or 'waterlevel') if filtering is to
            be performed.

        Returns
        -------
        new_values : dict
            Filtered values dictionary, containing only the requested
            parameter's values for each key in `values`. If no
            'parameter' key is present in `line_info`, the original
            `values` is returned unchanged.
        parameter : str
            The parameter name that was filtered to, or an empty string
            if no parameter was specified.

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though a
            `KeyError` may occur if `line_info['parameter']` is not one of
            the recognized parameter names.

        Examples
        --------
        >>> new_values, parameter = results.filterByParameter(values, {'parameter': 'temperature'})
        '''

        # map user-facing parameter names to the internal header names used in W2 output
        headerparam = {'flow': 'q(m3/s)',
                       'temperature': 't(c)',
                       'waterlevel': 'elevcl'}

        if 'parameter' not in line_info.keys():
            # no parameter specified in the line settings, so nothing to filter
            WF.print2stdout('Parameter not specified.', debug=self.Report.debug)
            WF.print2stdout('Line Info:', line_info, debug=self.Report.debug)
            return values, ''
        # dict that will hold just the filtered-down values
        new_values = {}
        # look up the internal header name matching the requested parameter
        target_header = headerparam[line_info['parameter']]
        # pull just the requested parameter's values out of each structure's dict
        for key in values.keys():
            new_values[key] = values[key][target_header]
        return new_values, line_info['parameter']

    def matchProfileLengths(self, select_val, elevations, depths):
        '''
        Trim value, elevation, and depth arrays so they all share a
        common length. This is necessary because values are sometimes not
        output at every elevation.

        Parameters
        ----------
        select_val : list
            Selected values.
        elevations : list
            Selected elevations.
        depths : list
            Selected depths.

        Returns
        -------
        select_val : list
            Values trimmed to the shortest common length.
        elevations : list
            Elevations trimmed to the shortest common length.
        depths : list
            Depths trimmed to the shortest common length.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> values, elevations, depths = results.matchProfileLengths(values, elevations, depths)
        '''

        # get the length of each of the three lists
        len_val = len(select_val)
        len_elev = len(elevations)
        len_depth = len(depths)
        # find the shortest of the three lengths
        min_len = min((len_val, len_elev, len_depth))
        # trim all three lists down to that shortest common length
        return select_val[:min_len], elevations[:min_len], depths[:min_len]

    def readTimeSeries(self, output_file_name, column=1, skiprows=3, **kwargs):
        '''
        Get an output time series from W2 at a specified location/column.
        Like the temperature profiles, output frequency is variable, so
        results are later interpolated onto a regular time series.

        Parameters
        ----------
        output_file_name : str
            Name of the output file within `self.run_path`.
        column : int or str, optional
            Column index (0-based) or column header name to read values
            from. Default is 1.
        skiprows : int, optional
            Number of header rows to skip before data begins. Default is
            3.
        **kwargs
            Additional keyword arguments (unused, accepted for interface
            compatibility).

        Returns
        -------
        dt_dates : numpy.ndarray or list
            Datetime values corresponding to each returned value.
        values : numpy.ndarray or list
            Array of parsed values from the requested column.

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though
            empty lists are returned if the output file or requested
            header cannot be found.

        Examples
        --------
        >>> dates, values = results.readTimeSeries('spr.opt', column='t(c)')
        '''

        # column/skiprows may arrive as strings, so coerce to int where possible
        try:
            column = int(column)
        except:
            # column is not numeric (likely a header name string), leave as-is
            pass

        try:
            skiprows = int(skiprows)
        except:
            # skiprows couldn't be converted, leave as-is
            pass


        # build the full path to the requested output file
        ofn_path = os.path.join(self.run_path, output_file_name)

        if not os.path.exists(ofn_path):
            # file doesn't exist, nothing to read
            WF.print2stdout('Data File not found!', ofn_path)
            return [], []

        # QWO/TWO files use the WDO interval/time series, everything else uses the TSR series
        if output_file_name.lower().startswith(('qwo', 'two')):
            jd_dates = self.wdo_jd_dates
            dt_dates = self.wdo_dt_dates
        else:
            jd_dates = self.jd_dates
            dt_dates = self.dt_dates

        if isinstance(column, str):
            # column given by name, so look up its position in the header row
            header = linecache.getline(ofn_path, int(skiprows)).strip().replace(' ','').lower().split(',') #1 indexed, for some reason
            cidx = np.where(np.asarray(header) == column.replace(' ','').lower())[0]
            if len(cidx) > 0:
                # found the header name, resolve it to a numeric column index
                column = cidx[0]
            else:
                # requested header name doesn't exist in the file
                WF.print2stdout(f'Header {column} not found in file', debug=self.Report.debug)
                return [], []

        # accumulators for parsed dates and values
        dates = []
        values = []
        # manually parse the file line by line, since delimiters can vary (comma or whitespace)
        with open(ofn_path, 'r') as o:
            for i, line in enumerate(o):
                if i >= int(skiprows):
                    # split on commas first
                    sline = line.split(',')
                    if len(sline) == 1: #not csv TODO: figure out this but better..
                        # comma split didn't work, fall back to whitespace splitting
                        sline = line.split()
                    # first field is always the jdate
                    dates.append(float(sline[0].strip()))
                    # if isinstance(column, int):
                    # pull the requested column's value
                    values.append(float(sline[column].strip()))
                    # elif isinstance(column, str):
                        # header = linecache.getline(ofn_path, int(skiprows)).strip().lower().split() #1 indexed, for some reason
                        # cidx = np.where(np.asarray(header) == column.lower())[0]
                        # values.append(float(sline[cidx].strip()))


        # trim the regular time series to match however many values were actually parsed
        if len(dt_dates) > len(values):
            # more regular timesteps than parsed values, trim the timesteps down
            dt_dates = dt_dates[:len(values)] #if the interval is off and its shifted, you better believe theres a missing value here. for fun.

        if len(dt_dates) < len(values): #in the event data file has full year of output and the time window changes
            # more parsed values than regular timesteps, trim the values down instead
            values = values[:len(dt_dates)]

        # return the aligned dates and values
        return dt_dates, np.asarray(values)

    def readSegment(self, filename, parameter):
        '''
        Read segment output values for a given parameter. This is a
        temporary/experimental method pending a full approach for W2
        contour plots.

        Parameters
        ----------
        filename : str
            Name of the output file within `self.run_path`.
        parameter : str
            Parameter to get data for (must be a recognized parameter
            name; see `self.getParameterFileStr()`).

        Returns
        -------
        None
            This function currently does not return usable output; the
            final `return` statement is commented out in the source and
            the parsing logic below it is incomplete/unreachable.

        Raises
        ------
        None
            This function does not explicitly raise exceptions in its
            current (incomplete) form.

        Notes
        -----
        This method appears to be a work in progress: several branches of
        the parsing logic are commented out, and the loop as written does
        not currently populate or return `segments`, `dates`, or
        `output_values` in a usable way.

        Examples
        --------
        >>> results.readSegment('contour.opt', 'temperature')
        '''

        # resolve the user-facing parameter name to its internal file-string representation
        read_param = self.getParameterFileStr(parameter)
        if read_param == None:
            # unrecognized parameter, nothing to read
            return [], [], []
        # build the full path to the segment output file
        ofn_path = os.path.join(self.run_path, filename)

        # placeholders for values/segments/dates that this (incomplete) method would populate
        output_values = np.array([])
        segments = []
        dates = []
        # flags used to track parsing state while scanning the file
        checkForVar = False
        record_vals = False
        gotvalues = True
        # scan the file looking for the "Model run at" marker, then parse subsequent parameter blocks
        with open(ofn_path, 'r') as otf:
            for line in otf:
                if checkForVar and line.strip() != '':
                    if parameter in line.lower():
                        # found a block matching the requested parameter
                        recordVals = True
                        # parse out the date/time fields embedded in the parameter header line
                        sline = line.lower().split(parameter).split()
                        month = sline[0]
                        day = sline[1]
                        year = sline[2]
                        time = sline[8]
                        hours = time.split('.')[0]
                        minutes = time.split('.')[1]
                        date = '{0} {1}, {2} {3}:{4}'.format(month, day, year, hours, minutes)
                        dates.append(dt.datetime.strptime(sline[0].strip(), '%B %d, %Y %H:%M'))
                # elif record_vals == True:
                #     if line.startswith(' Layer'):
                #         sline = line.split()
                #         for segnum in sline[2:]:
                #             if segnum not in output_values:

                elif line.startswith(' Model run at'):
                    # marker line found, start checking subsequent lines for the target parameter
                    checkForVar = True

        # otf.split('\n')

    def getParameterFileStr(self, parameter):
        '''
        Get the internal, properly-capitalized parameter name used in W2
        segment output files, based on a user-facing parameter name.

        Parameters
        ----------
        parameter : str
            Desired parameter name (case-insensitive).

        Returns
        -------
        str or None
            The internal formatted parameter name if recognized, or
            `None` if `parameter` is not a valid/known parameter.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.getParameterFileStr('temperature')
        'Temperature'
        '''

        #input:output
        # lookup table mapping lowercase user input to the properly formatted internal name
        fileparams = {'temperature': 'Temperature',
                      'density': 'Density',
                      'vertical eddy viscosity': 'Vertical eddy viscosity',
                      'velocity shear stress': 'Velocity shear stress',
                      'internal shear': 'Internal shear',
                      'bottom shear': 'Bottom shear',
                      'longitudinal momentum': 'Longitudinal momentum',
                      'horizontal density gradient': 'Horizontal density gradient',
                      'vertical momentum': 'Vertical momentum',
                      'horizontal pressure gradient': 'Horizontal pressure gradient',
                      'gravity term channel slope': 'Gravity term channel slope',
                      'horizontal velocity': 'Horizontal velocity',
                      'vertical velocity': 'Vertical velocity'}

        # look up the parameter (case-insensitive); if missing, warn and return None
        if parameter.lower() not in fileparams:
            WF.print2stdout('Parameter {0} not in acceptable parameters.'.format(parameter), debug=self.Report.debug)
            return None
        else:
            return fileparams[parameter.lower()]

    def getResultsFile_CSV(self):
        '''
        Find the name of the output results file for CSV-format profile
        runs by calculating its expected position within the control
        file, based on a fixed offset formula.

        Per Reclamation guidance (Ryan Miles, email dated 10/11/2022 at
        10:35), this offset is fixed and can be calculated directly
        rather than searched for.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The name of the results file, as found in the control file at
            the calculated offset row.

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though an
            `IndexError` may occur if `self.cf_lines` does not contain
            enough rows for the expected offsets.

        Notes
        -----
        The offset calculation accounts for the number of structures and
        several water quality constituent counts (general constituents,
        suspended solids, algae, epiphyton, BOD groups, macrophyte
        groups, and zooplankton groups) defined in the control file, since
        each of these adds additional rows to the control file before the
        results file name appears.

        Examples
        --------
        >>> results.getResultsFile_CSV()
        'spr.csv'
        '''

        # fixed base row number where the offset calculation starts counting from
        rootrow = 768

        # number of structures determines how many extra rows precede the results file entry
        structureoffsetline = [int(i) for i in self.cf_lines[135].strip().split(',') if i]
        # each structure adds 6 rows, with a minimum floor of 5 structures assumed
        structureoffset = (max(5, max(structureoffsetline))) * 6

        # water quality constituent counts also add to the offset
        constituentsoffsetline = [int(i) for i in self.cf_lines[21].strip().split(',') if i]
        #Offset = NGC + NSS + NAL + (NBOD * 3) + 32 + NZP + 4
        #NGC, NSS, NAL, NEP, NBOD, NMC, NZP
        # unpack each constituent count from its fixed position in the control file
        NGC = constituentsoffsetline[0]
        NSS = constituentsoffsetline[1]
        NAL = constituentsoffsetline[2]
        NEP = constituentsoffsetline[3]
        NBOD = constituentsoffsetline[4]
        NMC = constituentsoffsetline[5]
        NZP = constituentsoffsetline[6]

        # combine general constituent, suspended solids, algae, BOD, and zooplankton counts into one offset
        constituentsoffset = NGC + NSS + NAL + (NBOD * 3) + 32 + NZP + 4

        #Offset = Max(5, NEP) * 3
        # epiphyton offset, with a minimum floor of 5 assumed groups
        epiphytonoffset = (max(5, NEP)) * 3

        #Offset = Max(5, NAL) + Max(5, NZP)
        # zooplankton offset, combining algae and zooplankton minimum floors
        zooplanktonoffset = max(5, NAL) + max(5, NZP)

        #Offset = Max(5, NMC) * 3
        # macrophyte offset, with a minimum floor of 5 assumed groups
        macrophyteoffset = max(5, NMC) * 3

        # sum all offsets together to find the row containing the results file name
        totaloffset = structureoffset + constituentsoffset + epiphytonoffset + zooplanktonoffset + macrophyteoffset
        # add the total offset to the root row to get the actual target row index
        inputfile_idx = rootrow + totaloffset
        # extract just the file name (first whitespace-delimited token) from that row
        results_file = self.cf_lines[inputfile_idx-1].strip().split()[0] #excel is 1 idx..
        return results_file