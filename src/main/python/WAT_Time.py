import pandas as pd
import numpy as np
import datetime as dt
import pendulum

import WAT_Functions as WF
import WAT_Time as WT


def changeTimeSeriesInterval(times, values, Line_info, startYear):
    '''
    Change the interval of a time series, resampling values according to the
    averaging/accumulation type defined in the line settings.

    If a `type` is defined in `Line_info` (INST-VAL, INST-CUM, PER-AVER, or
    PER-CUM), that type is used to resample the data to the interval given
    in `Line_info['interval']`. If no interval is defined, the data is
    returned unchanged.

    Parameters
    ----------
    times : list or numpy.ndarray
        List of times associated with `values`. May be datetime objects or
        jdate (Julian date) numeric values.
    values : numpy.ndarray or dict
        Values to be resampled. May be a 1-D array, a 2-D array (multiple
        value sets sharing the same `times`), or a dict of value arrays
        keyed by member/scenario name.
    Line_info : dict
        Settings dictionary for the line/data series being processed. May
        contain the keys `type` (averaging type) and `interval` (target
        DSS-style time interval, e.g. '1HOUR').
    startYear : int
        Year used as the reference point when converting jdate values to
        datetime objects.

    Returns
    -------
    new_times : numpy.ndarray or list
        Resampled times, in the same format (datetime or jdate) as the
        input `times`.
    new_values : numpy.ndarray or dict
        Resampled values, in the same structure (array or dict) as the
        input `values`.

    Raises
    ------
    None
        This function does not explicitly raise exceptions, but will return
        the original `times`/`values` unchanged if resampling cannot be
        performed (e.g. mismatched lengths, missing interval, or unknown
        averaging type).

    Notes
    -----
    Per an August 2024 change from Reclamation, the default averaging type
    when none is specified has been changed from INST-VAL to PER-AVER.

    Examples
    --------
    >>> import datetime as dt
    >>> import numpy as np
    >>> times = [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)]
    >>> values = np.array([1.0, 2.0])
    >>> Line_info = {'type': 'PER-AVER', 'interval': '1DAY'}
    >>> new_times, new_values = changeTimeSeriesInterval(times, values, Line_info, 2020)
    '''

    # flag to track whether we need to convert back to jdate format at the end
    convert_to_jdate = False
    # nothing to do if there are no times supplied
    if len(times) == 0:
        return times, values

    # jdates come in as numbers, so convert them to datetime objects for processing
    if isinstance(times[0], (int, float)): #check for jdate, this is easier in dt..
        times = JDateToDatetime(times, startYear)
        convert_to_jdate = True

    # if a type is defined but no interval, there's nothing to resample to, so bail out early
    if 'type' in Line_info.keys() and 'interval' not in Line_info.keys():
        # WF.print2stdout('Defined Type but no interval..')
        if convert_to_jdate:
            return DatetimeToJDate(times), values
        else:
            return times, values

    # determine the averaging type to use (INST-CUM, INST-VAL, PER-AVER, PER-CUM)
    if 'type' in Line_info:
        avgtype = Line_info['type'].upper()
    else:
        # avgtype = 'INST-VAL'
        # default averaging type per Reclamation guidance (updated 8/12/24)
        avgtype = 'PER-AVER'

    # if values is a dict, recurse into each member/scenario's values separately
    if isinstance(values, dict):
        new_values = {}
        for key in values:
            new_times, new_values[key] = changeTimeSeriesInterval(times, values[key], Line_info, startYear)
    # if values is 2-D, recurse row by row and stack results back together
    elif len(values.shape) == 2:
        for vi, valueset in enumerate(values):
            new_times, changed_vals = changeTimeSeriesInterval(times, valueset, Line_info, startYear)
            if vi == 0:
                # allocate output array once we know the resampled length
                new_values = np.empty([values.shape[0], changed_vals.shape[0]])
            new_values[vi] = changed_vals
        # new_values = new_values.T

    else:
        # base case: values is a 1-D array, so do the actual resampling here
        if 'interval' in Line_info:
            interval = Line_info['interval'].upper()
            pd_interval = getPandasTimeFreq(interval)
        else:
            # WF.print2stdout('No time interval Defined.')
            return times, values

        # sanity check that times and values line up before resampling
        if len(values.shape) == 1:
            if len(values) != len(times):
                WF.print2stdout('Time and Value arrays not the same length')
                return times, values

        if avgtype == 'INST-VAL':
            #at the point in time, find intervals and use values
            if len(values.shape) == 1:
                # build a dataframe so pandas can handle the resampling logic
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                # only resample if the data isn't already at the target frequency
                if df.index.inferred_freq != pd_interval:
                    df = df.resample(pd_interval, origin='end_day').asfreq()
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()

        elif avgtype == 'INST-CUM':
            if len(values.shape) == 1:
                # cumulative sum first, then resample down to the desired interval
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                df = df.cumsum(skipna=True).resample(pd_interval, origin='end_day').asfreq()
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()

        elif avgtype == 'PER-AVER':
            #average over the period
            if len(values.shape) == 1:
                # resample and average values within each new interval
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                if df.index.inferred_freq != pd_interval:
                    df = df.resample(pd_interval, origin='end_day').mean()
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()

        elif avgtype == 'PER-CUM':
            #cum over the period
            if len(values.shape) == 1:
                # resample and sum values within each new interval
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                if df.index.inferred_freq != pd_interval:
                    df = df.resample(pd_interval, origin='end_day').sum()
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()
        else:
            # WF.print2stdout('INVALID INPUT TYPE DETECTED', avgtype)
            # unrecognized averaging type, so return the data unchanged
            return times, values

    # convert times back to jdate format if that's what was originally supplied
    if convert_to_jdate:
        return WT.DatetimeToJDate(new_times), new_values
    else:
        return new_times, new_values


def defineStartEndYears(Report):
    '''
    Define the start and end years for the simulation, storing them as
    attributes on the report object so they can be substituted in for
    flagged values elsewhere in the reporting code.

    End times that fall exactly on the first moment of a year (i.e.
    Dec 31 @ 24:00 represented as Jan 1 @ 00:00 of the following year)
    are treated as belonging to the prior year, since the simulation
    doesn't meaningfully extend into the new year.

    Parameters
    ----------
    Report : object
        Instance from the main report script. Must have `StartTime` and
        `EndTime` attributes as `datetime` objects.

    Returns
    -------
    None
        This function does not return a value. Instead, it sets the
        following attributes directly on `Report`:
            Report.startYear : int
            Report.endYear : int
            Report.years : list or range
            Report.years_str : str

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> defineStartEndYears(Report)
    >>> Report.years_str
    '2015-2018'
    '''

    tw_start = Report.StartTime
    tw_end = Report.EndTime
    #check if endTime is on the first day of the year at midnight
    if tw_end == dt.datetime(tw_end.year, 1, 1, 0, 0):
        tw_end += dt.timedelta(seconds=-1) #if its this day just go back

    # store the resolved start/end years on the report object
    Report.startYear = tw_start.year
    Report.endYear = tw_end.year
    if Report.startYear == Report.endYear:
        # single-year simulation, so years is just a one-item list
        Report.years_str = str(Report.startYear)
        Report.years = [Report.startYear]
    else:
        # multi-year simulation, build a range and a "start-end" display string
        Report.years = range(tw_start.year, tw_end.year + 1)
        Report.years_str = "{0}-{1}".format(Report.startYear, Report.endYear)


def defineStartEndMonths(Report):
    '''
    Define the start and end months for the simulation, storing them as
    attributes on the report object so they can be substituted in for
    flagged values elsewhere in the reporting code.

    If the end date falls exactly on the first day of a month, the end
    date is shifted back one second so it is treated as belonging to the
    previous month, giving an accurate representation of the covered
    period.

    Parameters
    ----------
    Report : object
        Instance from the main report script. Must have `StartTime` and
        `EndTime` attributes as `datetime` objects.

    Returns
    -------
    None
        This function does not return a value. Instead, it sets the
        following attributes directly on `Report`:
            Report.startMonth : str
            Report.endMonth : str
            Report.months_str : str
            Report.months : list

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> defineStartEndMonths(Report)
    >>> Report.months_str
    'Jan-Mar'
    '''

    tw_start = Report.StartTime
    tw_end = Report.EndTime

    # if end time lands exactly on the 1st of a month at midnight, back it up one second
    if tw_end == dt.datetime(tw_end.year, tw_end.month, 1, 0, 0):
        tw_end += dt.timedelta(seconds=-1) #go back one day

    # get abbreviated month names (e.g. 'Jan') for start and end
    Report.startMonth = tw_start.strftime("%b")
    Report.endMonth = tw_end.strftime("%b")

    if tw_start.year == tw_end.year and tw_start.month == tw_end.month:
        # simulation only spans a single month
        Report.months_str = Report.startMonth
        Report.months = [Report.startMonth]
    else:
        # simulation spans multiple months, so build up the list month by month
        start_month = dt.datetime(tw_start.year, tw_start.month, 1)
        end_month = dt.datetime(tw_end.year, tw_end.month, 1)
        months = []
        while start_month <= end_month:
            months.append(start_month.strftime("%b"))
            # step forward by 31 days then snap back to the 1st, guaranteeing month advance
            start_month += dt.timedelta(days=31)
            start_month = dt.datetime(start_month.year, start_month.month, 1)

        Report.months = months
        Report.months_str = "{0}-{1}".format(Report.months[0], Report.months[-1])


def setMultiRunStartEndYears(Report):
    '''
    Set the overall simulation start and end times by comparing the start
    and end times of all defined simulation runs, narrowing the window to
    the overlapping period common to all runs.

    Parameters
    ----------
    Report : object
        Instance from the main report script. Must have a
        `SimulationVariables` dict keyed by simulation ID, where each
        value is itself a dict containing `StartTime` and `EndTime`
        datetime objects. Must also have `StartTime` and `EndTime`
        attributes that will be narrowed in place.

    Returns
    -------
    None
        This function does not return a value. Instead, it updates
        `Report.StartTime` and `Report.EndTime` in place to reflect the
        overlapping time window across all runs.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> setMultiRunStartEndYears(Report)
    Start and End time set to 2015-01-01 00:00:00 - 2018-12-31 00:00:00
    '''

    # loop through every simulation run and narrow the overall window to the overlap
    for simID in Report.SimulationVariables.keys():
        if Report.SimulationVariables[simID]['StartTime'] > Report.StartTime:
            Report.StartTime = Report.SimulationVariables[simID]['StartTime']
        if Report.SimulationVariables[simID]['EndTime'] < Report.EndTime:
            Report.EndTime = Report.SimulationVariables[simID]['EndTime']
    WF.print2stdout('Start and End time set to {0} - {1}'.format(Report.StartTime, Report.EndTime))


def setSimulationDateTimes(Report, ID):
    '''
    Parse and set the simulation start and end times for a given run ID
    from their string representations. If a timestamp uses '24:00'
    (representing midnight at the end of a day), it is converted to the
    equivalent '00:00' timestamp on the following day.

    Parameters
    ----------
    Report : object
        Instance from the main report script. Must have a
        `SimulationVariables` dict keyed by simulation ID, where each
        value is a dict containing `StartTimeStr` and `EndTimeStr` string
        timestamps.
    ID : str or int
        Selected run ID used to look up the relevant entry in
        `Report.SimulationVariables`.

    Returns
    -------
    None
        This function does not return a value. Instead, it sets the
        following keys on `Report.SimulationVariables[ID]`:
            'StartTime' : datetime.datetime
            'EndTime' : datetime.datetime

    Raises
    ------
    None
        This function does not explicitly raise exceptions, but relies on
        `StartTimeStr`/`EndTimeStr` being in the expected
        '%d %B %Y, %H:%M' format.

    Examples
    --------
    >>> setSimulationDateTimes(Report, 'Run1')
    >>> Report.SimulationVariables['Run1']['StartTime']
    datetime.datetime(2020, 4, 1, 0, 0)
    '''

    StartTimeStr = Report.SimulationVariables[ID]['StartTimeStr']
    EndTimeStr = Report.SimulationVariables[ID]['EndTimeStr']

    # handle the '24:00' edge case by rolling it forward to the next day at midnight
    if '24:00' in StartTimeStr:
        tstrtmp = StartTimeStr.replace('24:00', '23:00')
        StartTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
        StartTime += dt.timedelta(hours=1)
    else:
        StartTime = dt.datetime.strptime(StartTimeStr, '%d %B %Y, %H:%M')
    Report.SimulationVariables[ID]['StartTime'] = StartTime

    # same '24:00' handling for the end time
    if '24:00' in EndTimeStr:
        tstrtmp = EndTimeStr.replace('24:00', '23:00')
        EndTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
        EndTime += dt.timedelta(hours=1)
    else:
        EndTime = dt.datetime.strptime(EndTimeStr, '%d %B %Y, %H:%M')
    Report.SimulationVariables[ID]['EndTime'] = EndTime


def makeRegularTimesteps(starttime, endtime, debug, days=15):
    '''
    Build a regularly spaced time series for profile plots when no
    explicit timesteps are defined.

    Parameters
    ----------
    starttime : datetime.datetime
        Start time for the new time series.
    endtime : datetime.datetime
        End time for the new time series.
    debug : bool
        Flag controlling whether debug messages are printed via
        `WF.print2stdout`.
    days : int, optional
        Interval, in days, between generated timesteps. Default is 15.

    Returns
    -------
    numpy.ndarray
        Array of generated timestep datetimes, with the first (potentially
        invalid) timestep removed.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> import datetime as dt
    >>> makeRegularTimesteps(dt.datetime(2020, 1, 1), dt.datetime(2020, 3, 1), debug=False)
    array([...], dtype=object)
    '''

    timesteps = []
    WF.print2stdout('No Timesteps found. Setting to Regular interval', debug=debug)
    # step forward from starttime by the given interval until we pass endtime
    cur_date = starttime
    while cur_date < endtime:
        timesteps.append(cur_date)
        cur_date += dt.timedelta(days=days)
    # drop the first timestep since it may just duplicate the start time
    return np.asarray(timesteps[1:]) #remove first timestep, may be invalid


def datetime2Ordinal(indate):
    '''
    Convert a datetime object to its ordinal (fractional day) value.

    Parameters
    ----------
    indate : datetime.datetime
        Datetime object to convert.

    Returns
    -------
    float
        Ordinal value representing the date, with hours and minutes
        expressed as a fraction of a day.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> import datetime as dt
    >>> datetime2Ordinal(dt.datetime(2020, 1, 2, 12, 0))
    737426.5
    '''

    # convert to ordinal day, then add fractional day for hours and minutes
    ord = indate.toordinal() + float(indate.hour) / 24. + float(indate.minute) / (24. * 60.)
    return ord


def getIdxForTimestamp(time_Array, t_in):
    '''
    Find the index of the timestep in `time_Array` nearest to the given
    target timestamp.

    Parameters
    ----------
    time_Array : list or numpy.ndarray
        Array of datetime values to search.
    t_in : datetime.datetime
        Target timestamp to locate within `time_Array`.

    Returns
    -------
    int
        Index of the nearest timestep in `time_Array`, or -1 if the
        nearest timestep is more than one day away.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Notes
    -----
    A warning is printed via `WF.print2stdout` if the nearest available
    timestep is more than 12 hours from the target timestamp, though the
    index is still returned in that case.

    Examples
    --------
    >>> import datetime as dt
    >>> import numpy as np
    >>> times = np.array([dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)])
    >>> getIdxForTimestamp(times, dt.datetime(2020, 1, 1, 1, 0))
    0
    '''

    # convert every timestamp in the array to an ordinal (fractional day) value
    ords = np.asarray([n.toordinal() + float(n.hour) / 24. + float(n.minute) / (24. * 60.) for n in time_Array])
    # convert the target timestamp the same way so they're comparable
    t_in_ord = t_in.toordinal() + float(t_in.hour) / 24. + float(t_in.minute) / (24. * 60.)
    tol_1hr = 0.04166666662786156  # 1 hour tolerance
    tol_12hrs = 0.5
    tol_1day = 1.0  # 1 day tolerance
    # find how far away the closest available timestep is
    min_diff = np.min(np.abs(ords - t_in_ord))
    if min_diff > tol_1day:
        WF.print2stdout('nearest time step > 1 day away')
        return -1
    if min_diff > tol_12hrs:
        WF.print2stdout(f'Warning: timestep {t_in} more than 12 hours away from closest timestep.')
    # locate the index of the closest matching timestep
    timestep = np.where(np.abs(ords - t_in_ord) == min_diff)[0][0]
    return timestep


def filterTimestepByYear(timestamps, year):
    '''
    Filter a list of timestamps to only those belonging to a given year.

    Parameters
    ----------
    timestamps : list
        List of datetime objects to filter.
    year : int or str
        Target year to filter by. If the special value 'ALLYEARS' is
        given, no filtering is performed.

    Returns
    -------
    list
        List of timestamps from the given year, or all timestamps if
        `year` is 'ALLYEARS'.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> import datetime as dt
    >>> timestamps = [dt.datetime(2019, 1, 1), dt.datetime(2020, 1, 1)]
    >>> filterTimestepByYear(timestamps, 2020)
    [datetime.datetime(2020, 1, 1, 0, 0)]
    '''

    # special case: no filtering requested, return everything
    if year == 'ALLYEARS':
        return timestamps
    # otherwise, keep only timestamps matching the target year
    return [n for n in timestamps if n.year == year]


def getPandasTimeFreq(intervalstring):
    '''
    Translate a DSS-formatted time interval string into the format
    expected by `pandas.DataFrame.resample()`.

    The translation is based on the unit portion of the interval string,
    so for example '15MIN' becomes '15T', and '6MON' becomes '6M'.

    Parameters
    ----------
    intervalstring : str
        DSS interval string, such as '1HOUR' or '1DAY'.

    Returns
    -------
    str
        Pandas-compatible time interval string. If the unit cannot be
        identified, the original `intervalstring` is returned unchanged.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getPandasTimeFreq('1HOUR')
    '1H'
    >>> getPandasTimeFreq('15MIN')
    '15T'
    '''

    intervalstringlr = intervalstring.lower()
    # check each supported unit in turn and translate to the pandas equivalent code
    if 'min' in intervalstringlr:
        for j in ['min', 'mins', 'minute', 'minutes']:
            if j in intervalstringlr:
                replaceflag = j
        timeint = intervalstringlr.replace(replaceflag,'') + 'T'
        return timeint
    elif 'hour' in intervalstringlr:
        for j in ['hour', 'hours']:
            if j in intervalstringlr:
                replaceflag = j
        timeint = intervalstringlr.replace(replaceflag,'') + 'H'
        return timeint
    elif 'day' in intervalstringlr:
        for j in ['day', 'days']:
            if j in intervalstringlr:
                replaceflag = j
        timeint = intervalstringlr.replace(replaceflag,'') + 'D'
        return timeint
    elif 'mon' in intervalstringlr:
        for j in ['mon', 'mons', 'month', 'months']:
            if j in intervalstringlr:
                replaceflag = j
        timeint = intervalstringlr.replace(replaceflag,'') + 'M'
        return timeint
    elif 'week' in intervalstringlr:
        for j in ['week', 'weeks']:
            if j in intervalstringlr:
                replaceflag = j
        timeint = intervalstringlr.replace(replaceflag,'') + 'W'
        return timeint
    elif 'year' in intervalstringlr:
        for j in ['year', 'years']:
            if j in intervalstringlr:
                replaceflag = j
        timeint = intervalstringlr.replace(replaceflag,'') + 'A'
        return timeint
    else:
        # WF.print2stdout('Unidentified time interval')
        # unit not recognized, so just hand the original string back unchanged
        return intervalstring


def buildTimeSeries(startTime, endTime, interval):
    '''
    Build a regular time series between two dates at a given interval.

    Parameters
    ----------
    startTime : datetime.datetime
        Start of the time series.
    endTime : datetime.datetime
        End of the time series.
    interval : str
        DSS-style interval string (e.g. '1HOUR', '1DAY') describing the
        spacing between generated timestamps.

    Returns
    -------
    numpy.ndarray
        Array of datetime objects spanning `startTime` to `endTime` at the
        given interval, inclusive of both endpoints.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Notes
    -----
    TODO noted in the original source: if `startTime` isn't on the hour
    but the interval is hourly (or similar), consider whether `startTime`
    should be adjusted to align with the interval.

    Examples
    --------
    >>> import datetime as dt
    >>> buildTimeSeries(dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2), '1DAY')
    array([datetime.datetime(2020, 1, 1, 0, 0), datetime.datetime(2020, 1, 2, 0, 0)],
          dtype=object)
    '''

    # translate the DSS interval into a pandas-compatible frequency string
    intervalinfo = getPandasTimeFreq(interval)
    # generate the full date range, inclusive of both start and end
    ts = pd.date_range(startTime, endTime, freq=intervalinfo, inclusive='both')
    # convert pandas Timestamps back to plain python datetime objects
    ts = np.asarray([t.to_pydatetime() for t in ts])
    return ts


def JDateToDatetime(dates, startyear):
    '''
    Convert jdate (Julian-style day-of-year) values to datetime objects.

    Parameters
    ----------
    dates : float, int, datetime.datetime, list, or numpy.ndarray
        Jdate value(s) to convert. If already a `datetime.datetime` or a
        collection of `datetime.datetime` objects, the input is returned
        unchanged.
    startyear : int
        Reference year used as the base for the jdate conversion (day 1.0
        corresponds to January 1 of this year).

    Returns
    -------
    datetime.datetime or numpy.ndarray or original type
        A single converted `datetime.datetime` if `dates` was a scalar, an
        array of converted `datetime.datetime` objects if `dates` was a
        list/array of numeric jdates, or the original `dates` unchanged if
        conversion doesn't apply.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Notes
    -----
    Jdate day 1.0 corresponds to the first day of `startyear`, so 1 must
    be subtracted from the jdate value before it is used as a day offset
    to avoid introducing an extra day.

    Examples
    --------
    >>> JDateToDatetime(1.5, 2020)
    datetime.datetime(2020, 1, 1, 12, 0)
    '''

    # jdate day 1.0 corresponds to Jan 1 of startyear
    first_year_Date = dt.datetime(startyear, 1, 1, 0, 0)
    #JDATES first day is at 1.0, so we need to subtract 1 or else we get an extra day..
    if isinstance(dates, (float, int)):
        # single scalar jdate value
        dtime = first_year_Date + dt.timedelta(days=dates-1)
        return dtime
    elif isinstance(dates, dt.datetime):
        # already a datetime, nothing to convert
        return dates
    elif isinstance(dates, (list, np.ndarray)):
        if len(dates) == 0:
            return dates
        elif isinstance(dates[0], dt.datetime):
            # already datetimes, nothing to convert
            return dates
        # convert each jdate value in the collection to a datetime
        dtimes = np.asarray([first_year_Date + dt.timedelta(days=n-1) for n in dates])
        return dtimes

    else:
        # unrecognized type, return unchanged
        return dates


def DatetimeToJDate(dates):
    '''
    Convert datetime objects to jdate (Julian-style day-of-year) values.

    Parameters
    ----------
    dates : float, int, datetime.datetime, list, or numpy.ndarray
        Datetime value(s) to convert. If already numeric (float or int),
        the input is returned unchanged, since it is assumed to already
        be in jdate format.

    Returns
    -------
    float or list or original type
        A single jdate value if `dates` was a scalar `datetime.datetime`,
        a list of jdate values if `dates` was a list/array of
        `datetime.datetime` objects, or the original `dates` unchanged if
        conversion doesn't apply.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Notes
    -----
    The jdate is computed relative to January 1 of the year of the first
    date in the collection (or the year of the single date supplied), with
    day 1.0 representing that reference date.

    Examples
    --------
    >>> import datetime as dt
    >>> DatetimeToJDate(dt.datetime(2020, 1, 2, 12, 0))
    2.5
    '''

    if len(dates) == 0:
        return dates
    elif isinstance(dates, (float, int)):
        # already numeric, assume already in jdate format
        return dates
    elif isinstance(dates, (list, np.ndarray)):
        if isinstance(dates[0], (float, int)):
            # already numeric, assume already in jdate format
            return dates
        # compute jdate relative to Jan 1 of the first date's year
        jdates = [((n.replace(tzinfo=None) - dt.datetime(dates[0].year, 1, 1, 0, 0)).total_seconds() / (24*60*60)+1) for n in dates]
        return jdates
    elif isinstance(dates, dt.datetime):
        # single datetime, compute jdate relative to Jan 1 of its own year
        jdate = (dates.replace(tzinfo=None) - dt.datetime(dates.year, 1, 1, 0, 0)).total_seconds() / (24*60*60) + 1
        return jdate
    else:
        # unrecognized type, return unchanged
        return dates


def translateDateFormat(lim, dateformat, fallback, StartTime, EndTime, debug=False):
    '''
    Translate a limit value between datetime and jdate formats, attempting
    multiple parsing strategies and falling back to a default value if all
    parsing attempts fail.

    Parameters
    ----------
    lim : int, float, str, or datetime.datetime
        Limit value to translate, typically an axis limit provided by a
        user or configuration file.
    dateformat : str
        Desired output format, either 'datetime' or 'jdate'.
    fallback : int, float, datetime.datetime, or None
        Value to fall back to if translation of `lim` fails. Typically the
        report's start time or end time.
    StartTime : datetime.datetime
        Start time of the report, used to validate parsed datetime limits
        and as a reference year for jdate conversion.
    EndTime : datetime.datetime
        End time of the report, used to validate parsed datetime limits.
    debug : bool, optional
        Flag controlling whether debug/status messages are printed via
        `WF.print2stdout`. Default is False.

    Returns
    -------
    lim_frmt or fallback
        The translated limit value in the requested `dateformat`, or the
        `fallback` value if translation could not be completed. If
        `dateformat` is 'jdate' and parsing as a jdate succeeds directly,
        that float value is returned.

    Raises
    ------
    None
        This function does not propagate exceptions; parsing failures are
        caught internally and reported via `WF.print2stdout`.

    Examples
    --------
    >>> import datetime as dt
    >>> translateDateFormat('Apr 2014 1 12:00', 'datetime', None,
    ...                      dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31))
    datetime.datetime(2014, 4, 1, 12, 0)
    '''

    if dateformat.lower() == 'datetime': #if want datetime
        if isinstance(lim, dt.datetime):
            # already the right type, nothing to do
            return lim
        else:
            try:
                # first, try parsing lim as a flexible date string
                lim_frmt = pendulum.parse(lim, strict=False).replace(tzinfo=None)#try simple date formatting.
                if not StartTime <= lim_frmt <= EndTime: #check for false negative
                    raise IndexError
                return lim_frmt
            except IndexError:
                WF.print2stdout('Xlim of {0} not between start and endtime {1} - {2}'.format(lim_frmt, StartTime,
                                                                                          EndTime), debug=debug)
            except:
                WF.print2stdout('Error Reading Limit: {0} as a dt.datetime object.'.format(lim), debug=debug)
                WF.print2stdout('If this is wrong, try format: Apr 2014 1 12:00', debug=debug)

            # date string parsing failed, so try treating lim as a jdate instead
            WF.print2stdout('Trying as Jdate..', debug=debug)
            try:
                lim_frmt = float(lim)
                lim_frmt = JDateToDatetime(lim_frmt, StartTime.year)
                WF.print2stdout('JDate {0} as {1} Accepted!'.format(lim, lim_frmt), debug=debug)
                return lim_frmt
            except:
                WF.print2stdout('Limit value of {0} also invalid as jdate.'.format(lim), debug=debug)

            # both parsing attempts failed, so fall back to the provided default
            if fallback != None and fallback != '':
                WF.print2stdout('Setting to fallback {0}.'.format(fallback), debug=debug)
            else:
                WF.print2stdout('Setting to fallback.', debug=debug)
            return fallback

    elif dateformat.lower() == 'jdate':
        try:
            # simplest case: lim is already numeric and can be used directly
            return float(lim)
        except:
            WF.print2stdout('Error Reading Limit: {0} as a jdate.'.format(lim), debug=debug)
            WF.print2stdout('If this is wrong, try format: 180', debug=debug)
            WF.print2stdout('Trying as Datetime..', debug=debug)
            # numeric parsing failed, try treating lim as a datetime or date string instead
            if isinstance(lim, (dt.datetime, str)):
                try:
                    if isinstance(lim, str):
                        lim_frmt = pendulum.parse(lim, strict=False).replace(tzinfo=None)
                        WF.print2stdout('Datetime {0} as {1} Accepted!'.format(lim, lim_frmt), debug=debug)
                    else:
                        lim_frmt = lim
                    WF.print2stdout('converting to jdate..', debug=debug)
                    lim_frmt = DatetimeToJDate(lim_frmt)
                    WF.print2stdout('Converted to jdate!', lim_frmt, debug=debug)
                    return lim_frmt
                except:
                    WF.print2stdout('Error Reading Limit: {0} as a dt.datetime object.'.format(lim), debug=debug)
                    WF.print2stdout('If this is wrong, try format: Apr 2014 1 12:00', debug=debug)

                # both attempts failed, fall back to the provided default (converted to jdate)
                fallback = DatetimeToJDate(fallback)

                if fallback != None and fallback != '':
                    WF.print2stdout('Setting to fallback {0}.'.format(fallback), debug=debug)
                else:
                    WF.print2stdout('Setting to fallback.', debug=debug)
                return fallback


def trimWindow(dates, values, window):
    """
    Trim the dates and values to the given window
    Parameters
    ----------
    dates: list
        Dates that go with the values that will be trimmed
    values: dict
        Dictionary of sets of values that will be trimmed
    window: str
        Month window to trim to

    Returns
    -------
    dates: list
        Dates that were trimmed
    values: dict
        Dictionary of sets of values that were trimmed

    Raises
    ------
    None
        This function does not explicitly raise exceptions, though a
        `KeyError` may occur if `window` contains month names not present
        in the internal month-name-to-number mapping.

    Examples
    --------
    >>> import numpy as np
    >>> dates = np.array([...])  # array of datetime objects
    >>> values = {'member1': np.array([...])}
    >>> trimmed_dates, trimmed_values = trimWindow(dates, values, 'May-November')
    """

    # dictionary to convert the month names to the month number
    c_month_to_number = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    # the window will be something like "May-November" so split into the two month
    s_start, s_end = window.split('-')

    # convert months to numbers
    i_start_month, i_end_month = c_month_to_number[s_start], c_month_to_number[s_end]

    # get the list of numbers to include
    # if we do not cross the year change
    if i_start_month <= i_end_month:
        window = list(range(i_start_month, i_end_month + 1))

    # if we do, we need start to 12 and 1 to end
    else:
        window  = list(range(i_start_month, 13)) + list(range(1, i_end_month + 1))

    # find the indices of dates whose month falls within the target window
    il_indeces = [i for i, date in enumerate(dates) if date.month in window]

    # trim the dates and values
    dates = dates[il_indeces]
    values = {member: vals[il_indeces] for member, vals in values.items()}

    return dates, values