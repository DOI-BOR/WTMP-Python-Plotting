import os, sys
import numpy as np
import datetime as dt
import pandas as pd
from scipy.interpolate import interp1d			# interp1d is used to interpolate scalar-table target values.
from pydsstools.heclib.dss import HecDss		# HecDss is the pydsstools wrapper for reading DSS records/collections.
from collections import Counter
import xml.etree.ElementTree as ET
import pendulum									# pendulum provides robust date parsing for formats not covered by WAT_Time's own parsing.
import traceback

import WAT_Functions as WF


def definedVarCheck(Block, flags):
    """
    Confirm that a set of required flags exist as child tags of an XML block.

    Parameters
    ----------
    Block : xml.etree.ElementTree.Element
        XML element to check the children of (e.g. a header block).
    flags : list of str
        Tag names that must all be present as direct children of
        ``Block`` for it to be considered valid.

    Returns
    -------
    bool
        ``True`` if every flag in ``flags`` is present as a child tag,
        ``False`` if any are missing.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> definedVarCheck(Block, ['Name', 'Value'])
    True
    """

    # Collect every direct child tag name once, then check membership
    # for each required flag rather than re-scanning the block per flag.
    tags = [n.tag for n in list(Block)]
    for flag in flags:
        if flag not in tags:
            # a required tag is missing, this block doesn't qualify
            return False
    return True

# def readSimulationFile(simulationfile):
#     '''
#     Read the right csv file, and determine what region you are working with.
#     Simulation CSV files are named after the simulation, and consist of program, model alter name, and then region(s)
#     :param simulation_name: name of simulation to find file
#     :param studyfolder: full path to study folder
#     :returns: dictionary containing information from file
#     '''
#
#     WF.print2stdout('Attempting to read {0}'.format(simulationfile))
#     if not os.path.exists(simulationfile):
#         WF.print2stderr(f'Could not find CSV file: {simulationfile}')
#         WF.print2stderr(f'Please create {simulationfile} in the Reports Directory and run report again.')
#         sys.exit(1)
#     sim_info = {}
#     with open(simulationfile, 'r') as sf:
#         for i, line in enumerate(sf):
#             if len(line.strip()) > 0:
#                 sline = line.strip().split(',')
#                 sim_info[i] = {'deffile': sline[-1].strip()} #comparison reports always put xml last
#                 sline = sline[:-1]
#                 sim_info[i]['programs'] = []
#                 sim_info[i]['modelaltnames'] = []
#                 for si, s in enumerate(sline):
#                     if len(s.strip()) > 1:
#                         if si % 2 == 0: #even
#                             sim_info[i]['programs'].append(s.strip())
#                         else: #odd
#                             sim_info[i]['modelaltnames'].append(s.strip())
#     return sim_info

def readGraphicsDefaults(GD_file):
    """
    Parse the graphics defaults XML file into a settings dictionary.

    Parameters
    ----------
    GD_file : str
        Path to the graphics default file.

    Returns
    -------
    dict
        Dictionary of default report-object settings, keyed by object
        type (via ``iterateGraphicsDefaults``).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> readGraphicsDefaults('Graphics_Defaults.xml')
    """

    # Parse the XML tree and pull out every <ReportObject> element, then
    # let iterateGraphicsDefaults build the keyed settings dictionary
    # from them (grouped by their 'Type' child tag).
    tree = ET.parse(GD_file)
    root = tree.getroot()
    gd_reportObjects = root.findall('ReportObject')
    reportObjects = iterateGraphicsDefaults(gd_reportObjects, 'Type')

    return reportObjects

def readDefaultLineStyle(linefile):
    """
    Parse the default line-style XML file into a settings dictionary.

    Parameters
    ----------
    linefile : str
        Path to the line-style default file.

    Returns
    -------
    dict
        Dictionary of default line-style settings, keyed by line type
        name (via ``iterateGraphicsDefaults``).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> readDefaultLineStyle('defaultLineStyles.xml')
    """

    tree = ET.parse(linefile)
    root = tree.getroot()
    def_lineTypes = root.findall('LineType')
    lineTypes = iterateGraphicsDefaults(def_lineTypes, 'Name')
    return lineTypes

def findTargetinChapterDefFile(flags, chapter, default=''):
    """
    Find the first matching tag's text within a chapter XML element.

    Parameters
    ----------
    flags : list of str
        Candidate tag names to search for, in priority order (used to
        support inconsistent capitalization in source XML files).
    chapter : xml.etree.ElementTree.Element
        Chapter (or section) XML element to search within.
    default : str, optional
        Value to use if none of the flags are found (default ``''``).

    Returns
    -------
    str
        The text of the first matching tag found, or ``default`` if
        none matched.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Notes
    -----
    Marked with a ``#TODO: make case insensitive`` comment in the
    original source.

    Examples
    --------
    >>> findTargetinChapterDefFile(['header', 'Header', 'HEADER'], section)
    'Monthly Averages'
    """

    targettext = default
    grouptext_flags = flags
    # Try each candidate tag name in order and stop at the first match;
    # this lets the same field be found regardless of capitalization
    # variants used across different chapter definition files.
    for flag in grouptext_flags:
        findtext = chapter.find(flag)
        if isinstance(findtext, ET.Element):
            # found a matching tag, use its text and stop searching
            targettext = findtext.text
            break
    return targettext

def readBCPathsMap(bcpathsmapfile):
    """
    Read the boundary-conditions path map as a formatted table.

    Parameters
    ----------
    bcpathsmapfile : str
        Path to the boundary conditions mapping file.

    Returns
    -------
    pandas.DataFrame
        The parsed boundary conditions table.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> readBCPathsMap('bc_paths_map.csv')
    """

    # delegate directly to the general-purpose formatted table reader
    bcpathsmap = readFormattedTable_Pandas(bcpathsmapfile)
    return bcpathsmap

def readChapterDefFile(CD_file):
    """
    Parse the chapter definitions XML file into a list of chapter dicts.

    Parameters
    ----------
    CD_file : str
        Path to the chapter definitions XML file.

    Returns
    -------
    list of dict
        One dictionary per chapter, each containing ``'name'``,
        ``'region'``, ``'sections'``, ``'grouptext'``, ``'resolution'``,
        ``'debug'``, ``'memberiteration'``, and ``'groupmembers'`` keys.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> readChapterDefFile('Shasta.xml')
    """

    Chapters = []
    tree = ET.parse(CD_file)
    root = tree.getroot()
    # Each top-level XML child represents one chapter; build a settings
    # dict for it and append to the running Chapters list.
    for chapter in root:
        ChapterDef = {}
        # Name/Region are read directly (with a blank fallback if
        # missing), while the remaining settings below use
        # findTargetinChapterDefFile to tolerate inconsistent tag
        # capitalization across different source files.
        try: chap_name = chapter.find('Name').text
        except: chap_name = ''
        try: chap_region = chapter.find('Region').text
        except: chap_region = ''
        ChapterDef['name'] = chap_name
        ChapterDef['region'] = chap_region
        ChapterDef['sections'] = []

        # each of the following settings tolerates several capitalization variants of its tag name
        grouptext_flags = ['text', 'Text', 'TEXT']
        ChapterDef['grouptext'] = findTargetinChapterDefFile(grouptext_flags, chapter)

        resolution_flags = ['resolution', 'Resolution', 'RESOLUTION']
        ChapterDef['resolution'] = findTargetinChapterDefFile(resolution_flags, chapter, default='high')

        debug_flags = ['debug', 'Debug', 'DEBUG']
        ChapterDef['debug'] = findTargetinChapterDefFile(debug_flags, chapter, default='false')

        memberiteration_flags = ['memberiteration', 'Memberiteration', 'MemberIteration', 'MEMBERITERATION', 'memberIteration']
        ChapterDef['memberiteration'] = findTargetinChapterDefFile(memberiteration_flags, chapter, default='false')

        groupmembers_flags = ['groupmembers', 'Groupmembers', 'GroupMembers', 'GROUPMEMBERS', 'groupMembers']
        ChapterDef['groupmembers'] = findTargetinChapterDefFile(groupmembers_flags, chapter, default='true')

        # Every <Section> under this chapter gets its own header plus a
        # list of report objects (plots/tables/etc.) parsed via
        # iterateChapterDefintions.
        cd_sections = chapter.findall('Sections/Section')
        for section in cd_sections:
            section_objects = {}

            headerflags = ['header', 'Header', 'HEADER']
            section_objects['header'] = findTargetinChapterDefFile(headerflags, section)

            section_objects['objects'] = []
            sec_objects = section.findall('Object')
            section_objects['objects'] = iterateChapterDefintions(sec_objects)
            ChapterDef['sections'].append(section_objects)

        Chapters.append(ChapterDef)

    return Chapters

def readCollectionsDSSData(dss_file, pathname, members, startdate, enddate, debug):
    """
    Read a DSS ensemble/collection record for one or more members.

    Parameters
    ----------
    dss_file : str
        Path to the DSS file.
    pathname : str
        DSS path with a ``*`` wildcard in the F-part (collection
        marker).
    members : list or 'all'
        List of member numbers to read, or the string ``'all'`` to read
        every member found in the file.
    startdate : datetime.datetime
        Start of the window to read data for.
    enddate : datetime.datetime
        End of the window to read data for.
    debug : bool
        Passed through to logging calls.

    Returns
    -------
    times : numpy.ndarray
        Timestamps for the collection (shared across all members).
    collection_values : dict
        Dictionary of value arrays keyed by member number.
    units : str or None
        Units reported by the DSS record.
    members : list
        The list of member numbers actually read.

    Raises
    ------
    None
        This function does not propagate exceptions; any failure while
        reading is caught, logged, and treated as "no data available".

    Examples
    --------
    >>> times, values, units, members = readCollectionsDSSData(dss_file, pathname, 'all', startdate, enddate, False)
    """

    try:
        if os.path.exists(dss_file):
            fid = HecDss.Open(dss_file)
            if pathname.split('/')[4] != '*': #make sure date field is blank
                # normalize the D-part (date field) to a wildcard before searching
                pns = pathname.split('/')
                pns[4] = '*'
                pathname = '/'.join(pns)
            # Get every pathname matching the collection wildcard, then
            # normalize the D-part (index 4, the date field) back to '*'
            # so duplicate records spanning different date blocks are
            # collapsed into one unique pathname per member.
            collection_pn = fid.getPathnameList(pathname)
            collection_pn = list(set(['/'.join(WF.ReplaceListAtIdx(n.split('/'), 4, '*')) for n in collection_pn]))
            WF.print2stdout(f'Found {len(collection_pn)} records in collection for {pathname}', debug=debug)
            if len(collection_pn) == 0:
                # no matching records at all in the collection
                WF.print2stdout(f'No records in collection for {pathname}', debug=debug)
                fid.close()
                return [], {}, None, []
            if members == 'all':
                # Extract the member number from each matched pathname's
                # F-part (e.g. 'C:000123|...' -> 123).
                members = [int(n.split('/')[6].split('|')[0].replace('C:', '')) for n in collection_pn]
            else:
                # explicit member list given, coerce every entry to int
                members = [int(n) for n in members]

            members.sort()
            # Read each member's time series individually, since DSS
            # collection records are stored as separate paths per member.
            for i, member in enumerate(members):
                # build the specific DSS path for this member, substituting the wildcard with its formatted number
                member_frmt = WF.formatMembers(member)
                CID_pathname_fpart = pathname.split('/')[6].replace('*|', f'C:{member_frmt}|')
                CID_pathname_split = pathname.split('/')
                CID_pathname_split[6] = CID_pathname_fpart
                CID_pathname = '/'.join(CID_pathname_split)
                WF.print2stdout(f'Currently working on {member}', debug=debug)
                ts = fid.read_ts(CID_pathname, window=(startdate, enddate), regular=True, trim_missing=False)
                values = np.array(ts.values)
                values[ts.nodata] = np.nan

                if i == 0:  # set vars like times and units that are always the same for all collection
                    # only need to capture the shared times/units once, on the first member
                    times = np.array(ts.pytimes)
                    units = ts.units
                    collection_values = {}

                if ts.empty: #if empty, it must be the path or time window. DSS record must exist
                    # nothing came back for this member, fall back to an all-NaN placeholder
                    WF.print2stdout('Invalid Timeseries record path of {0} or time window of {1} - {2}'.format(CID_pathname, startdate, enddate), debug=debug)
                    WF.print2stdout('Please check these parameters and rerun.', debug=debug)
                    values = np.full(len(times), np.nan)
                else:
                    values = np.asarray(values, dtype=np.float64)
                    if units == '':
                        # some members may report their units separately, fill in if not yet captured
                        units = ts.units

                collection_values[member] = values
            fid.close()
            return times, collection_values, units, members
        else:
            # DSS file itself doesn't exist at all
            WF.print2stdout(f'DSS file {dss_file} not found.', debug=True)
            return [], {}, None, []
    except:
        # Any unexpected failure reading the collection (corrupt file,
        # unexpected path structure, etc.) is logged and treated as "no
        # data" rather than crashing the whole report.
        WF.print2stdout(f'Unable to get data from {dss_file} {pathname}')
        WF.print2stdout(traceback.format_exc(), debug=debug)
        return [], {}, None, []

def readDSSData(dss_file, pathname, startdate, enddate, debug):
    """
    Read a single (non-collection) DSS time series record.

    Uses `pydsstools <https://github.com/gyanz/pydsstools>`_ to read the
    record.

    Example
    -------
    ::

        dss_file = "example.dss"
        pathname = "/REGULAR/TIMESERIES/FLOW//1HOUR/Ex1/"
        startDate = "15JUL2019 19:00:00"
        endDate = "15AUG2019 19:00:00"

    Parameters
    ----------
    dss_file : str
        Full path to the DSS file.
    pathname : str
        DSS path to read.
    startdate : datetime.datetime
        Start of the window to read data for.
    enddate : datetime.datetime
        End of the window to read data for.
    debug : bool
        Passed through to logging calls.

    Returns
    -------
    times : numpy.ndarray
        Timestamps for the series.
    values : numpy.ndarray
        Values for the series.
    units : str or None
        Units reported by the DSS record.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> times, values, units = readDSSData(dss_file, pathname, startdate, enddate, False)
    """

    # DSS requires date strings in this specific format
    startDatestr = startdate.strftime('%d%b%Y %H:%M:%S')
    endDatestr = enddate.strftime('%d%b%Y %H:%M:%S')

    if not os.path.exists(dss_file):
        # can't read anything without the source file
        WF.print2stdout('DSS file not found!', dss_file, debug=debug)
        return [], [], None

    fid = HecDss.Open(dss_file)
    ts = fid.read_ts(pathname,window=(startDatestr,endDatestr),regular=True,trim_missing=False)
    if ts.empty: #if empty, it must be the path or time window. DSS record must exist
        # nothing came back for the requested path/window
        WF.print2stdout('Invalid Timeseries record path of {0} or time window of {1} - {2}'.format(pathname, startDatestr, endDatestr), debug=debug)
        WF.print2stdout('Please check these parameters and rerun.', debug=debug)
        return [], [], None
    values = np.array(ts.values)
    values[ts.nodata] = np.nan

    made_ts = False
    if ts.dtype == 'Regular TimeSeries':
        # For regular-interval series, build the timestamp array
        # manually by stepping through the interval, rather than trusting
        # ts.pytimes (which can be slower/inconsistent); this is
        # validated below by checking the resulting length matches.
        interval_seconds = ts.interval
        times = []
        current_time = ts.startPyDateTime
        end_time = ts.endDateTime #bugged where the pydatetime shows start time..
        try:
            end_pytime = dt.datetime.strptime(end_time, '%d%b%Y %H:%M:%S')
        except ValueError:
            # DSS sometimes reports hour 24 (midnight of the next day)
            # instead of hour 00; roll it over manually since strptime
            # can't parse "24:" as an hour.
            end_time_repl = end_time.replace(' 24:', ' 23:')
            end_pytime = dt.datetime.strptime(end_time_repl, '%d%b%Y %H:%M:%S')
            end_pytime += dt.timedelta(hours=1)
        # step through the interval manually to build the full timestamp array
        while current_time <= end_pytime:
            times.append(current_time)
            current_time += dt.timedelta(seconds=interval_seconds)
        if len(times) == len(values):
            # manual build produced the expected length, safe to use
            made_ts = True

    if not made_ts:
        # Fall back to the (slower) pydsstools-provided timestamps for
        # irregular series, or if the manual regular-series build above
        # didn't produce a matching-length array.
        times = np.asarray(ts.pytimes)
        WF.print2stdout('Irregular DSS detected with {0} in {1}'.format(pathname, dss_file), debug=debug)
        WF.print2stdout('Recommend changing to regular time series for speed increases.', debug=debug)
    else:
        times = np.asarray(times)

    units = ts.units

    return times, values, units

def formatPyDSSToolsDates(datestring):
    """
    Parse a DSS-style date string into a datetime, handling hour-24.

    Parameters
    ----------
    datestring : str
        Date string in ``'%d%b%Y %H:%M:%S'`` format (e.g.
        ``'15JUL2019 19:00:00'``), possibly using hour ``24`` to mean
        midnight of the following day.

    Returns
    -------
    datetime.datetime
        The parsed datetime.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> formatPyDSSToolsDates('15JUL2019 24:00:00')
    datetime.datetime(2019, 7, 16, 0, 0)
    """

    try:
        ts_stime = dt.datetime.strptime(datestring, '%d%b%Y %H:%M:%S')
    except ValueError:
        # strptime can't parse hour "24"; rewrite it to "00" of the next
        # day and roll the date forward manually instead.
        ts_stime_splt = datestring.split(' ')
        if ts_stime_splt[1][:2] == '24':
            ts_stime_splt[1] = '00' + ts_stime_splt[1][2:]
            datestring = ' '.join(ts_stime_splt)
            ts_stime = dt.datetime.strptime(datestring, '%d%b%Y %H:%M:%S')
            ts_stime += dt.timedelta(days=1)
    return ts_stime

def readW2ResultsFile(output_file_name, jd_dates, run_path, targetfieldidx=1):
    """
    Read a CE-QUAL-W2 output text file and interpolate to given Julian dates.

    W2 output files are comma-delimited with a fixed 3-line header; the
    column to read is selectable since different output files use
    different layouts.

    Parameters
    ----------
    output_file_name : str
        Name of the W2 output file (relative to ``run_path``).
    jd_dates : array_like
        Julian dates (W2's native date format) to interpolate values at.
    run_path : str
        Directory containing the W2 run's output files.
    targetfieldidx : int, optional
        Column index (0-based) in the file to read values from
        (default ``1``).

    Returns
    -------
    numpy.ndarray
        Interpolated values at each requested Julian date (NaN where
        interpolation failed, e.g. out of range).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> readW2ResultsFile('spr.opt', jd_dates, run_path)
    """

    # allocate the output array, pre-filled with NaN for anything interpolation can't cover
    out_vals = np.full(len(jd_dates), np.nan)
    ofn_path = os.path.join(run_path, output_file_name)
    dates = []
    values = []
    skiplines = 3 #not sure if this is always true?
    # Read the raw file, skipping the fixed header, and pull the date
    # (always column 0) and the requested value column.
    with open(ofn_path, 'r') as o:
        for i, line in enumerate(o):
            if i >= skiplines:
                sline = line.split(',')
                dates.append(float(sline[0].strip()))
                values.append(float(sline[targetfieldidx].strip()))

    # Build a linear interpolator over the raw file's own Julian dates,
    # then sample it at each requested jd_dates value.
    if len(dates) > 1:
        val_interp = interp1d(dates, values)
    for j, jd in enumerate(jd_dates):
        try:
            out_vals[j] = val_interp(jd)
        except ValueError:
            # Requested date falls outside the interpolation range;
            # leave that entry as NaN.
            continue

    return out_vals

def readFormattedTable_Pandas(filename):
    """
    Read a specially-formatted single-page table with headers.

    Currently supports CSV files. Empty rows are dropped.

    Parameters
    ----------
    filename : str
        Path to the table file.

    Returns
    -------
    pandas.DataFrame
        The parsed table, or an empty DataFrame if the file doesn't
        exist.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> readFormattedTable_Pandas('table.csv')
    """

    if os.path.exists(filename):
        ext = filename.split('.')[-1]
        if ext.lower() == 'csv':
            # currently only CSV format is supported
            df = pd.read_csv(filename)
        df.dropna(inplace=True) #sometimes theres extra rows..
        return df
    else:
        # file doesn't exist, nothing to read
        WF.print2stdout(f'{filename} not found.')
        return pd.DataFrame()

def readTextProfile(observed_data_filename, timestamps, starttime=None, endtime=None):
    """
    Read an observed-data text file into per-timestamp temperature profiles.

    Parameters
    ----------
    observed_data_filename : str
        Path to the observed profile CSV file (columns: date, value,
        depth).
    timestamps : list, numpy.ndarray, or other
        If a list/array, only the profiles closest to these specific
        timestamps are returned; otherwise every profile found in the
        file is returned.
    starttime : datetime.datetime, optional
        Skip rows before this time.
    endtime : datetime.datetime, optional
        Stop reading once rows exceed this time.

    Returns
    -------
    values : list of numpy.ndarray
        Value arrays, one per matched timestamp/profile.
    depths : list of numpy.ndarray
        Depth arrays, one per matched timestamp/profile.
    times : numpy.ndarray
        The timestamps corresponding to each returned profile.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> values, depths, times = readTextProfile('observed.csv', 'all', starttime, endtime)
    """

    # accumulators for the completed profiles found so far
    t = []
    wt = []
    d = []
    # accumulators for the profile currently being built (shares one timestamp)
    t_profile = []
    wt_profile = []
    d_profile = []
    # cache of the last-seen raw date string/parsed datetime, to skip re-parsing repeat rows
    last_dtstr = ''
    last_dt = dt.datetime(1933, 10, 15)
    hold_dt = dt.datetime(1933, 10, 15) #https://www.onthisday.com/date/1933/october/15 sorry Steve
    if not os.path.exists(observed_data_filename):
        # can't read anything without the source file
        WF.print2stdout('Observed data at {0} does not exist.'.format(observed_data_filename))
        return [], [], []
    # Stream the file line by line, grouping consecutive rows sharing
    # the same date string into a single profile (t_profile/wt_profile/
    # d_profile), and flushing each completed profile onto the running
    # t/wt/d output lists whenever the date changes.
    with open(observed_data_filename, 'r') as odf:
        for j, line in enumerate(odf):
            if j == 0:
                # first line is the header row, capture it and move on
                headers = line.strip().split(',')
                continue
            sline = line.split(',')
            dt_str = sline[0]
            if dt_str == last_dtstr:
                # Same date string as the previous row: reuse the
                # already-parsed datetime instead of re-parsing (this is
                # a meaningful speedup for files with many rows per date).
                dt_tmp = last_dt
            else:
                dt_tmp = pendulum.parse(dt_str, strict=False).replace(tzinfo=None)
                last_dtstr = dt_str
                last_dt = dt_tmp
            if starttime > dt_tmp:
                # row is before the requested window, skip it
                continue
            if endtime < dt_tmp:
                # row is past the requested window, nothing more to read
                break
            # if (dt_tmp.year != hold_dt.year or dt_tmp.month != hold_dt.month or dt_tmp.day != hold_dt.day): #if its a new date
            if (dt_tmp != hold_dt): #if its a new date
                # New timestamp encountered: flush the just-completed
                # profile (if any) and start a fresh one.
                if len(t_profile) != 0 and len(wt_profile) != 0 and len(d_profile) != 0:
                    t.append(np.array(t_profile))
                    wt.append(np.array(wt_profile))
                    d.append(np.array(d_profile))
                t_profile = [dt_tmp]
                wt_profile = [float(sline[1])]
                d_profile = [float(sline[2])]
            else:
                # Same timestamp as the previous row: append this
                # depth/value pair onto the current profile.
                # if float(sline[2]) not in d_profile:
                t_profile.append(dt_tmp)
                wt_profile.append(float(sline[1]))
                d_profile.append(float(sline[2]))
            hold_dt = dt_tmp

    # Flush the final profile after the loop ends (it wouldn't have been
    # appended by the "new date" check above, since there's no next row).
    if len(t_profile) != 0 and len(wt_profile) != 0 and len(d_profile) != 0:
        t.append(np.array(t_profile))
        wt.append(np.array(wt_profile))
        d.append(np.array(d_profile))

    if isinstance(timestamps, (list, np.ndarray)):
        # Caller wants specific timestamps: match each requested
        # timestamp to its closest available profile (or an empty
        # profile if none is close enough).
        wtn = []
        dn = []
        ts = []
        if len(t) > 0:
            cti = getClosestProfileTime(timestamps, [n[0] for n in t])

            # build the output arrays, using an empty profile wherever no close match was found
            for ci, i in enumerate(cti):
                if i != None:
                    wtn.append(np.asarray(wt[i]))
                    dn.append(np.asarray(d[i]))
                    ts.append(timestamps[ci])
                else:
                    wtn.append(np.array([]))
                    dn.append(np.array([]))
                    ts.append(timestamps[ci])

        return wtn, dn, np.asarray(ts)
    else:
        # No specific timestamps requested: return every profile found.
        return wt, d, np.asarray(t)

def getTextProfileDates(observed_data_filename, starttime, endtime):
    """
    Extract the unique available dates from an observed profile file.

    Parameters
    ----------
    observed_data_filename : str
        Path to the observed profile CSV file.
    starttime : datetime.datetime
        Only include dates on or after this time.
    endtime : datetime.datetime
        Only include dates on or before this time.

    Returns
    -------
    numpy.ndarray
        Array of unique datetimes found within the requested window
        (empty if the file doesn't exist).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getTextProfileDates('observed.csv', starttime, endtime)
    """

    t = []
    if not os.path.exists(observed_data_filename):
        # can't read anything without the source file
        return t
    # First pass: collect every unique raw date string in the file
    # (deferring the more expensive pendulum parsing until after
    # de-duplication).
    with open(observed_data_filename, 'r') as odf:
        for j, line in enumerate(odf):
            if j == 0:
                # skip the header row
                continue
            sline = line.split(',')
            dt_str = sline[0]
            if dt_str not in t:
                t.append(dt_str)
    t_frmt = []
    # Second pass: parse each unique date string and keep only those
    # falling within the requested time window.
    for tdate in t:
        dt_tmp = pendulum.parse(tdate, strict=False).replace(tzinfo=None) #trying this to see if it will work
        if starttime <= dt_tmp <= endtime: #get time window
            if dt_tmp not in t_frmt:
                t_frmt.append(dt_tmp)

    return np.asarray(t_frmt)

def getClosestProfileTime(timestamps, dates):
    """
    Find the index of the closest available date to each target timestamp.

    Parameters
    ----------
    timestamps : list of datetime.datetime
        Target timestamps to match.
    dates : list of datetime.datetime
        Available dates to search (e.g. from a profile file).

    Returns
    -------
    list
        Index into ``dates`` closest to each timestamp, or ``None`` for
        a given timestamp if the closest match is more than a day away.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getClosestProfileTime(timestamps, dates)
    """

    cdi = []
    for timestamp in timestamps:
        # Build a {time_difference_seconds: index} map and take the
        # index with the smallest difference.
        cloz_dict = {
            abs(timestamp.timestamp() - date.timestamp()) : di
            for di, date in enumerate(dates)}
        res = cloz_dict[min(cloz_dict.keys())]
        if abs(timestamp.timestamp() - dates[res].timestamp()) > 86400: #seconds in a day
            # Closest match is still more than a day away; treat as "no
            # match" rather than silently using a far-off profile.
            cdi.append(None)
        else:
            cdi.append(res)
    return cdi

def getClosestTime(timestamps, dates):
    """
    Convert target timestamp(s) into array indices via regular-interval math.

    Assumes ``dates`` is a regularly-spaced time series; computes each
    index directly from elapsed time rather than searching, for speed.

    Parameters
    ----------
    timestamps : datetime.datetime, list, or numpy.ndarray
        Target timestamp(s) to convert to index/indices.
    dates : array_like of datetime.datetime
        The regularly-spaced reference date array.

    Returns
    -------
    list or int
        List of indices (if ``timestamps`` was a list/array) or a
        single index (if ``timestamps`` was a scalar), or ``[]`` if
        ``dates`` is empty.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getClosestTime(timestamp, dates)
    42
    """

    cdi = []
    if len(dates) > 0:
        t0 = dates[0]
    else:
        # nothing to index into at all
        return []
    if len(dates) > 1:
        t_interval = dates[1] - t0 #timedelta
    t_interval_seconds = t_interval.total_seconds()
    if isinstance(timestamps, (list, np.ndarray)):
        # Convert each requested timestamp to an index by dividing its
        # elapsed time from t0 by the series' regular interval.
        for timestamp in timestamps:
            ts_diff = timestamp - t0
            index = int(round(ts_diff.total_seconds() / t_interval_seconds))
            cdi.append(index)
    else:
        # single scalar timestamp given, compute its index directly
        ts_diff = timestamps - t0
        index = int(round(ts_diff.total_seconds() / t_interval_seconds))
        cdi = index
    return cdi

def getchildren(root, returnkeyless=False):
    """
    Recursively parse an XML element into a nested dict/list structure.

    Applies WAT's XML-to-settings convention: a leaf element becomes a
    plain value; a group of same-named repeated child elements becomes
    a list; and a group of differently-named child elements becomes a
    dict.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        XML element (section) to parse.
    returnkeyless : bool, optional
        If ``True``, returns just the parsed value (unwrapped from its
        tag-name key); if ``False``, returns a one-key dict keyed by
        ``root``'s (lowercased) tag name (default ``False``).

    Returns
    -------
    dict or list
        The parsed settings structure.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getchildren(root)
    """

    children = {}
    if len(root) == 0:
        # Leaf element (no children): store its text value directly,
        # using None for empty/whitespace-only text.
        try:
            if len(root.text.strip()) == 0:
                # whitespace-only text, treat as unset
                children[root.tag.lower()] = None
            else:
                children[root.tag.lower()] = root.text.strip()
        except:
            # root.text was None entirely (e.g. self-closing tag)
            children[root.tag.lower()] = root.text
    else:
        if len(Counter([n.tag.lower() for n in root])) > 1: #if the amount of diff subroots > 1
            # Multiple differently-named children: this element becomes
            # a dict of those children's parsed values.
            children[root.tag.lower()] = {}
        else: #if there is only 1 subroot
            if len(root.text.strip()) == 0: #if the text len of root is 0, we have subitems
                if len([n.tag.lower() for n in root]) > 1: #if we have more than 1 of the same subroot
                    # Multiple children sharing the same tag name: this
                    # becomes a list (e.g. multiple <Line> elements
                    # under <Lines>).
                    children[root.tag.lower()] = []
                else: #otherwise, we have a single dictionary
                    subroot = root[0]
                    #if the subroots is just the root, but singular, aka lines -> line, reaches -> reach
                    if subroot.tag.lower() == root.tag.lower()[:-1] or subroot.tag.lower() == root.tag.lower()[:-2]:
                        # Singular/plural naming pattern (e.g. <Lines>
                        # containing one <Line>): treat as a
                        # single-element list for consistency, since
                        # other cases with multiple <Line>s produce a
                        # list too.
                        children[root.tag.lower()] = []
                    else:
                        children[root.tag.lower()] = {}
            else:
                children[root.tag.lower()] = []

        # Recurse into each child, appending to the list or merging into
        # the dict depending on which container type was chosen above.
        for subroot in root:
            if isinstance(children[root.tag.lower()], list):
                children[root.tag.lower()].append(getchildren(subroot, returnkeyless=True))
            elif isinstance(children[root.tag.lower()], dict):
                children[root.tag.lower()].update(getchildren(subroot))

    if returnkeyless:
        # Strip the wrapping {tag: value} key, used when this call is
        # itself building an item for a parent list.
        children = children[root.tag.lower()]
    return children

def iterateGraphicsDefaults(root, main_key):
    """
    Parse a list of graphics-default XML elements, keyed by a named tag.

    Parameters
    ----------
    root : list of xml.etree.ElementTree.Element
        Elements to iterate (e.g. every ``<ReportObject>`` or
        ``<LineType>``).
    main_key : str
        Tag name whose text value should be used as this entry's
        dictionary key (e.g. ``'Type'`` or ``'Name'``).

    Returns
    -------
    dict
        Dictionary keyed by each element's ``main_key`` text (lowercased),
        each value a dict of that element's remaining settings parsed
        via ``getchildren``.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> iterateGraphicsDefaults(reportObjects, 'Type')
    """

    out = {}
    for cr in root:
        # Use the element's main_key child (e.g. Type/Name) as the
        # dictionary key, then parse every OTHER child as its settings.
        key = cr.find(main_key).text.lower()
        out[key.lower()] = {}
        for child in cr:
            if child.tag == main_key:
                # this is the key field itself, already used above, skip it
                continue
            else:
                out[key.lower()][child.tag.lower()] = getchildren(child, returnkeyless=True)
    return out

def iterateChapterDefintions(root):
    """
    Parse a list of chapter-definition XML elements into settings dicts.

    Parameters
    ----------
    root : list of xml.etree.ElementTree.Element
        Elements to iterate (e.g. every ``<Object>`` in a section).

    Returns
    -------
    list of dict
        One settings dict per element, each key/value parsed via
        ``getchildren``.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> iterateChapterDefintions(sec_objects)
    """

    out = []
    for cr in root:
        keylist = {}
        # parse every child element into this object's settings dict
        for child in cr:
            keylist[child.tag.lower()] = getchildren(child, returnkeyless=True)
        out.append(keylist)
    return out

def findDefaultCSVFile(Report, Simulation):
    """
    Build the expected path to a simulation's default CSV file.

    The file name pattern depends on the report type (validation,
    comparison, or forecast).

    Parameters
    ----------
    Report : object
        The main Report Generator instance; used for ``reportType`` and
        ``studyDir``.
    Simulation : dict
        Simulation settings dictionary; must contain ``'basename'``.

    Returns
    -------
    str
        The expected CSV file path (not guaranteed to exist).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> findDefaultCSVFile(Report, Simulation)
    ```
    """

    simbasename = Simulation['basename'].replace(' ', '_')
    # File naming convention differs by report type.
    if Report.reportType == 'validation':
        simulation_file = os.path.join(Report.studyDir, 'reports', '{0}.csv'.format(simbasename))
    elif Report.reportType == 'comparison':
        simulation_file = os.path.join(Report.studyDir, 'reports', '{0}_comparison.csv'.format(simbasename))
    elif Report.reportType == 'forecast':
        simulation_file = os.path.join(Report.studyDir, 'reports', '{0}_forecast.csv'.format(simbasename))

    return simulation_file

def readReportCSVFile(Report, Simulation):
    """
    Read a simulation's report CSV file describing which programs/XMLs to run.

    Parameters
    ----------
    Report : object
        The main Report Generator instance.
    Simulation : dict
        Simulation settings dictionary; must contain ``'csvfile'``
        (path, or ``None`` to use the default path) and
        ``'basename'``.

    Returns
    -------
    dict
        Dictionary keyed by line order number, each entry containing
        ``'xmlfile'``, ``'programs'``, ``'keywords'``, ``'order'``,
        ``'numtimesprogramused'``, and ``'deprecated_method'``. Exits
        the script if the CSV file cannot be found.

    Raises
    ------
    SystemExit
        Raised (via ``sys.exit(1)``) if the expected CSV file cannot be
        found on disk.

    Examples
    --------
    >>> readReportCSVFile(Report, Simulation)
    """
    if Simulation["csvfile"] is None: #if they didnt specify a csv file, use the default
        # no explicit CSV given, compute the expected default path instead
        Simulation["csvfile"] = findDefaultCSVFile(Report, Simulation)
        WF.print2stdout('Attempting to read default {0}'.format(Simulation["csvfile"]))
    else:
        WF.print2stdout('Attempting to read specified {0}'.format(Simulation["csvfile"]))

    if not os.path.exists(Simulation["csvfile"]):
        # can't proceed without the CSV file at all
        WF.print2stderr(f'Could not find CSV file: {Simulation["csvfile"]}')
        WF.print2stderr(f'Please create {Simulation["csvfile"]} run report again.')
        sys.exit(1)
    csv_info = {}
    program_used = {}
    accepted_lines = 0
    use_deprecated = False
    # Parse each CSV row into program(s), XML file, and optional
    # keywords; track how many times each program combination has been
    # seen so far (numtimesprogramused), used elsewhere to disambiguate
    # repeated program usage.
    with open(Simulation["csvfile"], 'r') as csvf:
        for i, line in enumerate(csvf):
            if len(line.strip()) >= 2: #needs to at least have a filename and report type
                accepted_lines += 1
                sline = line.strip().split(',')
                programs_raw = sline[0].strip().lower()
                programs = [n for n in programs_raw.split('|') if n != '']
                xmlfile = sline[1].strip().lower()
                # If the second column isn't actually an XML file, this
                # is an old-style CSV; fall back to the deprecated parser
                # entirely rather than mixing formats.
                use_deprecated = checkforDeprecatedCSV(xmlfile)
                if use_deprecated:
                    break
                keywords = [str(n).lower() for n in sline[2:] if n != ''] #optional keywords for CSV files to use to match simulations
                if programs_raw not in program_used:
                    # first time seeing this exact program combination
                    program_used[programs_raw] = 1
                else:
                    # seen this combination before, increment the counter
                    program_used[programs_raw] += 1

                csv_info[accepted_lines] = {'xmlfile': xmlfile,
                                            'programs': programs,
                                            'keywords': keywords,
                                            'order': accepted_lines,
                                            'numtimesprogramused': program_used[programs_raw],
                                            'deprecated_method': False}
    if use_deprecated:
        # detected an old-style CSV mid-parse, restart entirely with the deprecated parser
        csv_info = readSimulationFile_deprecated(Simulation["csvfile"])
    return csv_info

def checkforDeprecatedCSV(xmlfile):
    """
    Detect whether a CSV row uses the pre-6.0.0 (deprecated) format.

    Parameters
    ----------
    xmlfile : str
        The second column of a CSV row, expected to be an XML file
        path in the current format.

    Returns
    -------
    bool
        ``True`` if ``xmlfile`` doesn't end in ``.xml`` (indicating the
        deprecated format is in use), ``False`` otherwise.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> checkforDeprecatedCSV('shasta.xml')
    False
    """
    if not xmlfile.split('.')[-1] == 'xml':
        # doesn't look like a valid XML filename at all, must be the old format
        WF.print2stdout('Deprecated CSV file detected as of 6.0.0. Using old style.')
        WF.print2stdout('Please update CSV style to the following:')
        WF.print2stdout('Program (ressim, cequalw2), XML file path, keywords (optional)')
        return True
    return False

def readSimulationFile_deprecated(simulationfile):
    """
    Read a simulation CSV file using the pre-6.0.0 format.

    DEPRECATED METHOD: DO NOT USE EXCEPT AS A BACKUP.

    Simulation CSV files are named after the simulation, and consist of
    program, model alternative name, and then region(s).

    Parameters
    ----------
    simulationfile : str
        Path to the (old-format) simulation CSV file.

    Returns
    -------
    dict
        Dictionary keyed by line number, each containing ``'xmlfile'``,
        ``'programs'``, ``'modelaltnames'``, and
        ``'deprecated_method'`` (``True``). Exits the script if the
        file cannot be found.

    Raises
    ------
    SystemExit
        Raised (via ``sys.exit(1)``) if ``simulationfile`` cannot be
        found on disk.

    Examples
    --------
    >>> readSimulationFile_deprecated('legacy_simulation.csv')
    """

    WF.print2stdout('Attempting to read {0} using old method'.format(simulationfile))
    if not os.path.exists(simulationfile):
        # can't proceed without the source file at all
        WF.print2stderr(f'Could not find CSV file: {simulationfile}')
        WF.print2stderr(f'Please create {simulationfile} in the Reports Directory and run report again.')
        sys.exit(1)
    csv_info = {}
    with open(simulationfile, 'r') as sf:
        for i, line in enumerate(sf):
            if len(line.strip()) > 0:
                sline = line.strip().split(',')
                #iterate through sline in reverse until non '' is found
                # The XML file is always the last non-empty column, but
                # trailing commas can leave empty columns after it; scan
                # backwards to find the true last populated column.
                for si, s in enumerate(sline[::-1]):
                    if len(s.strip()) > 0:
                        csv_info[i] = {'xmlfile': sline[len(sline)-1-si].strip()} #subtract 1
                        break
                # csv_info[i] = {'xmlfile': sline[-1].strip()} #comparison reports always put xml last
                sline = sline[:len(sline)-si-1]
                csv_info[i]['programs'] = []
                csv_info[i]['modelaltnames'] = []
                # Remaining columns alternate: program, model alt name,
                # program, model alt name, ...
                for si, s in enumerate(sline):
                    if len(s.strip()) > 1:
                        if si % 2 == 0: #even
                            csv_info[i]['programs'].append(s.strip())
                        else: #odd
                            csv_info[i]['modelaltnames'].append(s.strip())
                csv_info[i]['deprecated_method'] = True

    return csv_info

def getReportType(Report):
    """
    Get the report type from the Report object.

    Historically this was inferred from the CSV file's parent directory
    name (see the commented-out legacy logic below); it's now taken
    directly from ``Report.reportType``.

    Parameters
    ----------
    Report : object
        The main Report Generator instance.

    Returns
    -------
    str
        The report type (e.g. ``'validation'``, ``'comparison'``,
        ``'forecast'``).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getReportType(Report)
    'validation'
    """
    # current implementation is trivial; historical directory-based inference kept below for reference
    return Report.reportType
    # if Report.reportCSV is None:
    #     return Report.reportType
    # else:
    #     path_sep = Report.reportCSV.split(os.path.sep)
    #     csv_origin = path_sep[-2].lower()
    #     if csv_origin.lower() in ['validation_report']: #todo: confirm these
    #         reportType = 'validation'
    #     elif csv_origin.lower() in ['comparison_report']:
    #         reportType = 'comparison'
    #     elif csv_origin.lower() in ['forecast_report']:
    #         reportType = 'forecast'
    #     else:
    #         WF.print2stdout(f'Warning: unable to identify report type from file containing CSV file: {csv_origin}')
    #         WF.print2stdout('Please use one of the applicable following: validation, comparison, forecast')
    #         WF.print2stdout('Using validation for now.')
    #         reportType = 'validation'
    #
    #     return reportType



def readSimulationInfo(Report, simulationInfoFile):
    """
    Read the WAT-generated simulation info XML into the Report object.

    Populates study-level attributes (report type, directories,
    description) and builds ``Report.Simulations``, a list of per-
    simulation settings dictionaries used for the rest of report
    generation.

    Parameters
    ----------
    Report : object
        The main Report Generator instance; updated in place with
        ``Simulations``, ``reportType``, ``studyDir``, ``observedDir``,
        ``installDir``, ``outputDir``, ``description``, ``studyname``,
        ``SimulationGroup``, ``iscomp``, ``isforecast``, ``All_IDs``,
        and ``reportCSV``.
    simulationInfoFile : str
        Full path to the simulation information XML file produced by
        the WAT.

    Returns
    -------
    None
        This function does not return a value; it updates ``Report``
        in place.

    Raises
    ------
    SystemExit
        Raised (via ``WF.checkExists``) if ``simulationInfoFile`` does
        not exist.

    Examples
    --------
    >>> readSimulationInfo(Report, 'SimulationInfo.xml')
    """

    # bail out early with a clear error if the required XML file is missing
    WF.checkExists(simulationInfoFile)

    Report.Simulations = []
    tree = ET.parse(simulationInfoFile)
    root = tree.getroot()

    # Top-level study/report settings, shared across all simulations.
    Report.reportType = root.find('ReportType').text.lower()
    Report.studyDir = root.find('Study/Directory').text
    Report.observedDir = root.find('Study/ObservedData').text
    Report.installDir = root.find('Study/InstallDirectory').text
    Report.outputDir = root.find('Study/WriteDirectory').text
    Report.description = root.find('Study/Description').text
    try:
        Report.studyname = root.find('Study/Name').text #todo: remove after backwards compat
    except:
        # older files may not have a Study/Name element at all
        Report.studyname = None

    # SimulationGroup is optional (absent in older/legacy simulation
    # info files); fall back to None values if not present.
    SimulationGroupRoot = root.find('SimulationGroup')
    Report.SimulationGroup = {}
    if SimulationGroupRoot != None: #legacy for old files
        Report.SimulationGroup['Name'] = SimulationGroupRoot.find('Name').text
        Report.SimulationGroup['Description'] = SimulationGroupRoot.find('Description').text #Todo: test to make sure this works with nones
    else:
        Report.SimulationGroup['Name'] = None
        Report.SimulationGroup['Description'] = None

    Report.iscomp = False
    Report.isforecast = False
    # Normalize legacy/alternate report type names to the current set
    # of recognized values.
    if Report.reportType == 'single':
        Report.reportType = 'validation'
    if Report.reportType == 'alternativecomparison':
        Report.iscomp = True
        Report.reportType = 'comparison'
    elif Report.reportType == 'forecast':
        Report.isforecast = True

    Report.All_IDs = []
    SimRoot = root.find('Simulations')
    # Build one settings dictionary per <Simulation> element.
    for simulation in SimRoot:
        simulationInfo = {'name': simulation.find('Name').text,
                          'basename': simulation.find('BaseName').text,
                          'directory': simulation.find('Directory').text,
                          'dssfile': simulation.find('DSSFile').text,
                          'starttime': simulation.find('StartTime').text,
                          'endtime': simulation.find('EndTime').text,
                          'lastcomputed': simulation.find('LastComputed').text,
                          'Description': simulation.find('Description').text
                          }

        # AnalysisPeriod and WatAlternative are also optional/legacy
        # sub-elements; same None-fallback pattern as SimulationGroup.
        analysisperiodRoot = simulation.find('AnalysisPeriod')
        simulationInfo['AnalysisPeriod'] = {}
        if analysisperiodRoot != None:  # legacy for old files
            simulationInfo['AnalysisPeriod']['Name'] = analysisperiodRoot.find('Name').text
            simulationInfo['AnalysisPeriod']['Description'] = analysisperiodRoot.find('Description').text
        else:
            simulationInfo['AnalysisPeriod']['Name'] = None
            simulationInfo['AnalysisPeriod']['Description'] = None

        watAlternativeRoot = simulation.find('WatAlternative')
        simulationInfo['WatAlternative'] = {}
        if watAlternativeRoot != None:  # legacy for old files
            simulationInfo['WatAlternative']['Name'] = watAlternativeRoot.find('Name').text
            simulationInfo['WatAlternative']['Description'] = watAlternativeRoot.find('Description').text
        else:
            simulationInfo['WatAlternative']['Name'] = None
            simulationInfo['WatAlternative']['Description'] = None

        try:
            Report.reportCSV = simulation.find('CsvFile').text
        except AttributeError:
            # no CsvFile element present, no explicit CSV path given
            Report.reportCSV = None
        simulationInfo['csvfile'] = Report.reportCSV

        # Forecast reports additionally define ensemble sets (groups of
        # forecast members) per simulation.
        if Report.isforecast:
            ensemblesets = getchildren(simulation.find('EnsembleSets'), returnkeyless=True)
            simulationInfo['ensemblesets'] = ensemblesets

        # ID defaults to the report's base ID if not explicitly given
        # (e.g. for single/validation reports with only one simulation).
        try:
            simulationInfo['ID'] = simulation.find('ID').text
        except AttributeError:
            simulationInfo['ID'] = Report.base_id
        Report.All_IDs.append(simulationInfo['ID'])

        modelAlternatives = getchildren(simulation.find('ModelAlternatives'), returnkeyless=True)
        simulationInfo['modelalternatives'] = modelAlternatives
        Report.Simulations.append(simulationInfo)

def readGraphicsDefaultFile(Report):
    """
    Locate and parse the study's Graphics_Defaults.xml file.

    Parameters
    ----------
    Report : object
        The main Report Generator instance; updated in place with
        ``Report.graphicsDefault``.

    Returns
    -------
    None
        This function does not return a value; it updates ``Report``
        in place.

    Raises
    ------
    SystemExit
        Raised (via ``WF.checkExists``) if the expected
        ``Graphics_Defaults.xml`` file does not exist.

    Examples
    --------
    >>> readGraphicsDefaultFile(Report)
    """

    # build the expected path and confirm it exists before parsing
    graphicsDefaultfile = os.path.join(Report.studyDir, 'reports', 'Graphics_Defaults.xml')
    WF.checkExists(graphicsDefaultfile)
    Report.graphicsDefault = readGraphicsDefaults(graphicsDefaultfile)

def readDefinitionsFile(Report, simorder):
    """
    Locate and parse the chapter definitions XML file for a simulation.

    Parameters
    ----------
    Report : object
        The main Report Generator instance; updated in place with
        ``Report.ChapterDefinitions``.
    simorder : dict
        Simulation CSV row dictionary; must contain ``'xmlfile'``.

    Returns
    -------
    None
        This function does not return a value; it updates ``Report``
        in place.

    Raises
    ------
    SystemExit
        Raised (via ``WF.checkExists``) if the expected chapter
        definitions file does not exist.

    Examples
    --------
    >>> readDefinitionsFile(Report, simorder)
    """

    # build the expected path and confirm it exists before parsing
    ChapterDefinitionsFile = os.path.join(Report.studyDir, 'reports', simorder['xmlfile'])
    WF.checkExists(ChapterDefinitionsFile)
    Report.ChapterDefinitions = readChapterDefFile(ChapterDefinitionsFile)

# def readComparisonSimulationsCSV(Report):
#     '''
#     Reads in the simulation CSV but for comparison plots. Comparison plots have '_comparison' appended to the end of them,
#     but are built in general the same as regular Simulation CSV files.
#     :return:
#     '''
#
#     simulation_file = os.path.join(Report.studyDir, 'reports', '{0}_comparison.csv'.format(Report.SimulationVariables[Report.base_id]['baseSimulationName'].replace(' ', '_')))
#     Report.SimulationCSV = readSimulationFile_deprecated(simulation_file)

def readForecastSimulationsCSV(Report):
    """
    Read the forecast-specific simulation CSV file (``*_forecast.csv``).

    Forecast plots use a CSV named with a ``'_forecast'`` suffix, but
    otherwise follow the same (deprecated-style) format as regular
    Simulation CSV files.

    Parameters
    ----------
    Report : object
        The main Report Generator instance; updated in place with
        ``Report.SimulationCSV``.

    Returns
    -------
    None
        This function does not return a value; it updates ``Report``
        in place.

    Raises
    ------
    SystemExit
        Raised (via ``readSimulationFile_deprecated``) if the expected
        forecast CSV file does not exist.

    Examples
    --------
    >>> readForecastSimulationsCSV(Report)
    """

    # build the expected forecast-suffixed CSV path and read it via the deprecated-format parser
    simulation_file = os.path.join(Report.studyDir, 'reports', '{0}_forecast.csv'.format(Report.SimulationVariables[Report.base_id]['baseSimulationName'].replace(' ', '_')))
    Report.SimulationCSV = readSimulationFile_deprecated(simulation_file)

def readTemplate(Report, templatefilename):
    """
    Read a report template XML file into a settings dictionary.

    Parameters
    ----------
    Report : object
        The main Report Generator instance; used for ``studyDir``.
    templatefilename : str
        Name of the template XML file (relative to
        ``<studyDir>/reports``).

    Returns
    -------
    dict
        Dictionary of template object settings (via
        ``iterateGraphicsDefaults``), or an empty dict if the template
        file doesn't exist.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> readTemplate(Report, 'MonthlyTable_Template.xml')
    """

    templatefile = os.path.join(Report.studyDir, 'reports', templatefilename)
    if os.path.exists(templatefile):
        tree = ET.parse(templatefile)
        root = tree.getroot()
        templateObjects = root.findall('Object')
        reportObjects = iterateGraphicsDefaults(templateObjects, 'Type')
        return reportObjects
    else:
        # template file doesn't exist, nothing to parse
        WF.print2stdout(f'Template file {templatefile} not found.')
        return {}

def readScalarTable(scalartablepath):
    """
    Read a value-scaling lookup table from a CSV file.

    Headers are ignored (auto-detected via failed float conversion, so
    they don't strictly need to be present). Supported row formats::

        name, scaled_by, scalar
        # OR
        scaled_by, scalar

    e.g.::

        1out, 363, .34
        # OR
        363, .43

    Parameters
    ----------
    scalartablepath : str
        Path to the scalar table CSV file.

    Returns
    -------
    dict
        Dictionary mapping each "scaled by" target value (float) to its
        corresponding scalar (float). Exits the script if a row has
        fewer than 2 usable columns.

    Raises
    ------
    SystemExit
        Raised (via ``sys.exit(1)``) if a row in the file has fewer
        than 2 columns.

    Examples
    --------
    >>> readScalarTable('scalar_table.csv')
    {363.0: 0.34}
    """

    scalars = {}
    scalartable = np.genfromtxt(scalartablepath, delimiter=',', dtype=None, encoding="utf8")
    # Rows can have either 2 columns (target, scalar) or 3+ columns
    # (name, target, scalar, ...); either way, take the last two
    # relevant columns as [target, scalar]. Rows that fail to parse as
    # floats are assumed to be a header row and skipped.
    for line in scalartable:
        if len(line) == 2:
            try:
                # simple [target, scalar] row format
                line_flt = [float(n) for n in line]
                scalars[line_flt[0]] = line_flt[1]
            except:
                #probably header line
                continue
        elif len(line) > 2:
            try:
                # [name, target, scalar, ...] row format, use columns 1 and 2
                line_flt = [float(n) for n in line[1:3]]
                scalars[line_flt[0]] = line_flt[1]
            except:
                #probably header line
                continue
        else:
            # Fewer than 2 columns: the file is malformed and can't be
            # used, so abort rather than silently proceed with bad data.
            WF.print2stderr(f'Scalar table {scalartablepath} formatted incorrectly. Should be [target, scalar]')
            WF.print2stderr('Now exiting')
            sys.exit(1)

    return scalars