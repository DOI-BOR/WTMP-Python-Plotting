'''
* Copyright 2022 United States Bureau of Reclamation (USBR).
* United States Department of the Interior
* All Rights Reserved. USBR PROPRIETARY/CONFIDENTIAL.
* Source may not be released without written approval
* from USBR

Created on 7/15/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''

import pandas as pd
import numpy as np
import datetime as dt
import pendulum

import WAT_Functions as WF
import WAT_Time as WT

def changeTimeSeriesInterval(times, values, Line_info, startYear):
    '''
    changes time series of time series data. If type is defined, use that to average data.
    #update 8/12/24 - per reclamation default has changed from INST-VAL to PER-AVER -Kayla
    :param times: list of times
    :param values: list of values
    :param Line_info: settings dictionary for line
    :return: new times and values
    '''

    convert_to_jdate = False
    if len(times) == 0:
        return times, values

    if isinstance(times[0], (int, float)): #check for jdate, this is easier in dt..
        times = JDateToDatetime(times, startYear)
        convert_to_jdate = True

    if 'type' in Line_info.keys() and 'interval' not in Line_info.keys():
        # WF.print2stdout('Defined Type but no interval..')
        if convert_to_jdate:
            return DatetimeToJDate(times), values
        else:
            return times, values

    # INST-CUM, INST-VAL, PER-AVER, PER-CUM)
    if 'type' in Line_info:
        avgtype = Line_info['type'].upper()
    else:
        # avgtype = 'INST-VAL'
        avgtype = 'PER-AVER'

    if isinstance(values, dict):
        new_values = {}
        for key in values:
            new_times, new_values[key] = changeTimeSeriesInterval(times, values[key], Line_info, startYear)
    elif len(values.shape) == 2:
        for vi, valueset in enumerate(values):
            new_times, changed_vals = changeTimeSeriesInterval(times, valueset, Line_info, startYear)
            if vi == 0:
                new_values = np.empty([values.shape[0], changed_vals.shape[0]])
            new_values[vi] = changed_vals
        # new_values = new_values.T

    else:
        if 'interval' in Line_info:
            interval = Line_info['interval'].upper()
            pd_interval = getPandasTimeFreq(interval)
        else:
            # WF.print2stdout('No time interval Defined.')
            return times, values

        if len(values.shape) == 1:
            if len(values) != len(times):
                WF.print2stdout('Time and Value arrays not the same length')
                return times, values

        if avgtype == 'INST-VAL':
            #at the point in time, find intervals and use values
            if len(values.shape) == 1:
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                if df.index.inferred_freq != pd_interval:
                    df = df.resample(pd_interval, origin='end_day').asfreq()
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()

        elif avgtype == 'INST-CUM':
            if len(values.shape) == 1:
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                df = df.cumsum(skipna=True).resample(pd_interval, origin='end_day').asfreq()
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()

        elif avgtype == 'PER-AVER':
            #average over the period
            if len(values.shape) == 1:
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                if df.index.inferred_freq != pd_interval:
                    df = df.resample(pd_interval, origin='end_day').mean()
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()

        elif avgtype == 'PER-CUM':
            #cum over the period
            if len(values.shape) == 1:
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                if df.index.inferred_freq != pd_interval:
                    df = df.resample(pd_interval, origin='end_day').sum()
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()
        else:
            # WF.print2stdout('INVALID INPUT TYPE DETECTED', avgtype)
            return times, values

    if convert_to_jdate:
        return WT.DatetimeToJDate(new_times), new_values
    else:
        return new_times, new_values

def defineStartEndYears(Report):
    '''
    defines start and end years for the simulation so they can be replaced by flagged values.
    end dates that end on the first of the year with no min seconds (aka Dec 31 @ 24:00) have their end
    years set to be the year prior, as its not fair to really call them that next year
    :param Report: instance from main report script
    :return: class variables
                self.startYear
                self.endYear
                self.years
                self.years_str
    '''

    tw_start = Report.StartTime
    tw_end = Report.EndTime
    #check if endTime is on the first day of the year at midnight
    if tw_end == dt.datetime(tw_end.year, 1, 1, 0, 0):
        tw_end += dt.timedelta(seconds=-1) #if its this day just go back

    Report.startYear = tw_start.year
    Report.endYear = tw_end.year
    if Report.startYear == Report.endYear:
        Report.years_str = str(Report.startYear)
        Report.years = [Report.startYear]
    else:
        Report.years = range(tw_start.year, tw_end.year + 1)
        Report.years_str = "{0}-{1}".format(Report.startYear, Report.endYear)

def defineStartEndMonths(Report):
    '''
    Defines the start and end months for the simulation, so they can be replaced by flagged values.
    If the end date falls on the first day of the month(ex: Jan 1 ),
    it adjusts the end date to the last day of the previous month to accurately represent the period
    :param Report: instance from main report script that includes `StartTime` and `EndTime` attributes as `datetime` objects.
    :return: class variables
                self.startMonth
                self.endMonth
                self.months_str
                self.months
    '''

    tw_start = Report.StartTime
    tw_end = Report.EndTime

    if tw_end == dt.datetime(tw_end.year, tw_end.month, 1, 0, 0):
        tw_end += dt.timedelta(seconds=-1) #go back one day

    Report.startMonth = tw_start.strftime("%b")
    Report.endMonth = tw_end.strftime("%b")

    if tw_start.year == tw_end.year and tw_start.month == tw_end.month:
        Report.months_str = Report.startMonth
        Report.months = [Report.startMonth]
    else:
        start_month = dt.datetime(tw_start.year, tw_start.month, 1)
        end_month = dt.datetime(tw_end.year, tw_end.month, 1)
        months = []
        while start_month <= end_month:
            months.append(start_month.strftime("%b"))
            start_month += dt.timedelta(days=31)
            start_month = dt.datetime(start_month.year, start_month.month, 1)

        Report.months = months
        Report.months_str = "{0}-{1}".format(Report.months[0], Report.months[-1])

def setMultiRunStartEndYears(Report):
    '''
    sets start and end times by looking at all possible runs. Picks overlapping time periods only.
    :param Report: instance from main report script
    '''

    for simID in Report.SimulationVariables.keys():
        if Report.SimulationVariables[simID]['StartTime'] > Report.StartTime:
            Report.StartTime = Report.SimulationVariables[simID]['StartTime']
        if Report.SimulationVariables[simID]['EndTime'] < Report.EndTime:
            Report.EndTime = Report.SimulationVariables[simID]['EndTime']
    WF.print2stdout('Start and End time set to {0} - {1}'.format(Report.StartTime, Report.EndTime))

def setSimulationDateTimes(Report, ID):
    '''
    sets the simulation start time and dates from string format. If timestamp says 24:00, converts it to be correct
    Datetime format of the next day at 00:00
    :param Report: instance from main report script
    :param ID: selected run ID
    :return: class varables
                self.StartTime
                self.EndTime
    '''

    StartTimeStr = Report.SimulationVariables[ID]['StartTimeStr']
    EndTimeStr = Report.SimulationVariables[ID]['EndTimeStr']

    if '24:00' in StartTimeStr:
        tstrtmp = StartTimeStr.replace('24:00', '23:00')
        StartTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
        StartTime += dt.timedelta(hours=1)
    else:
        StartTime = dt.datetime.strptime(StartTimeStr, '%d %B %Y, %H:%M')
    Report.SimulationVariables[ID]['StartTime'] = StartTime

    if '24:00' in EndTimeStr:
        tstrtmp = EndTimeStr.replace('24:00', '23:00')
        EndTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
        EndTime += dt.timedelta(hours=1)
    else:
        EndTime = dt.datetime.strptime(EndTimeStr, '%d %B %Y, %H:%M')
    Report.SimulationVariables[ID]['EndTime'] = EndTime

def makeRegularTimesteps(starttime, endtime, debug, days=15):
    '''
    makes regular time series for profile plots if there are no times defined
    :param starttime: start time for new timeseries
    :param endtime: end time for new time series
    :param days: interval for profile time series, 15 is default
    :return: timestep list
    '''

    timesteps = []
    WF.print2stdout('No Timesteps found. Setting to Regular interval', debug=debug)
    cur_date = starttime
    while cur_date < endtime:
        timesteps.append(cur_date)
        cur_date += dt.timedelta(days=days)
    return np.asarray(timesteps[1:]) #remove first timestep, may be invalid

def datetime2Ordinal(indate):
    '''
    converts datetime objects to ordinal values
    :param indate: datetime object
    :return: ordinal
    '''

    ord = indate.toordinal() + float(indate.hour) / 24. + float(indate.minute) / (24. * 60.)
    return ord

def getIdxForTimestamp(time_Array, t_in):
    '''
    finds timestep for date
    :param time_Array: array of time values
    :param t_in: time step
    :param offset: time series offset for ordinal
    :return: timestep index
    '''

    ords = np.asarray([n.toordinal() + float(n.hour) / 24. + float(n.minute) / (24. * 60.) for n in time_Array])
    t_in_ord = t_in.toordinal() + float(t_in.hour) / 24. + float(t_in.minute) / (24. * 60.)
    tol_1hr = 0.04166666662786156  # 1 hour tolerance
    tol_12hrs = 0.5
    tol_1day = 1.0  # 1 day tolerance
    min_diff = np.min(np.abs(ords - t_in_ord))
    if min_diff > tol_1day:
        WF.print2stdout('nearest time step > 1 day away')
        return -1
    if min_diff > tol_12hrs:
        WF.print2stdout(f'Warning: timestep {t_in} more than 12 hours away from closest timestep.')
    timestep = np.where(np.abs(ords - t_in_ord) == min_diff)[0][0]
    return timestep

def filterTimestepByYear(timestamps, year):
    '''
    returns only timestamps from the given year. Otherwise, just return all timestamps
    :param timestamps: list of dates
    :param year: target year
    :return:
        timestamps: list of selected timestamps
    '''

    if year == 'ALLYEARS':
        return timestamps
    return [n for n in timestamps if n.year == year]

def getPandasTimeFreq(intervalstring):
    '''
    Reads in the DSS formatted time intervals and translates them to a format pandas.resample() understands
    bases off of the time interval, so 15MIN becomes 15T, or 6MON becomes 6M
    :param intervalstring: DSS interval string such as 1HOUR or 1DAY
    :return: pandas time interval
    '''

    intervalstringlr = intervalstring.lower()
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
        return intervalstring

def buildTimeSeries(startTime, endTime, interval):
    '''
    builds a regular time series using the start and end time and a given interval
    #TODO: if start time isnt on the hour, but the interval is, change start time to be hourly?
    :param startTime: datetime object
    :param endTime: datetime object
    :param interval: DSS interval
    :return: list of time series dates
    '''

    intervalinfo = getPandasTimeFreq(interval)
    ts = pd.date_range(startTime, endTime, freq=intervalinfo, inclusive='both')
    ts = np.asarray([t.to_pydatetime() for t in ts])
    return ts

def JDateToDatetime(dates, startyear):
    '''
    converts jdate dates to datetime values
    :param dates: list of jdate dates
    :return:
        dtimes: list of dates
        dtime: single date
        dates: original date if unable to convert
    '''

    first_year_Date = dt.datetime(startyear, 1, 1, 0, 0)
    #JDATES first day is at 1.0, so we need to subtract 1 or else we get an extra day..
    if isinstance(dates, (float, int)):
        dtime = first_year_Date + dt.timedelta(days=dates-1)
        return dtime
    elif isinstance(dates, dt.datetime):
        return dates
    elif isinstance(dates, (list, np.ndarray)):
        if len(dates) == 0:
            return dates
        elif isinstance(dates[0], dt.datetime):
            return dates
        dtimes = np.asarray([first_year_Date + dt.timedelta(days=n-1) for n in dates])
        return dtimes

    else:
        return dates

def DatetimeToJDate(dates):
    '''
    converts datetime dates to jdate values
    :param dates: list of datetime dates
    :param time_offset: model time offset
    :return:
        jdates: list of dates
        jdate: single date
        dates: original date if unable to convert
    '''

    if len(dates) == 0:
        return dates
    elif isinstance(dates, (float, int)):
        return dates
    elif isinstance(dates, (list, np.ndarray)):
        if isinstance(dates[0], (float, int)):
            return dates
        jdates = [((n.replace(tzinfo=None) - dt.datetime(dates[0].year, 1, 1, 0, 0)).total_seconds() / (24*60*60)+1) for n in dates]
        return jdates
    elif isinstance(dates, dt.datetime):
        jdate = (dates.replace(tzinfo=None) - dt.datetime(dates.year, 1, 1, 0, 0)).total_seconds() / (24*60*60) + 1
        return jdate
    else:
        return dates

def translateDateFormat(lim, dateformat, fallback, StartTime, EndTime, debug=False):
    '''
    translates date formats between datetime and jdate, as desired
    :param lim: limit value, either int or datetime
    :param dateformat: desired date format, either 'datetime' or 'jdate'
    :param fallback: if setting translation fails, use backup, usually starttime or endtime
    :param StartTime: start time
    :param EndTime: end time
    :param time_offset: offset for conversion
    :return:
        lim: original limit, if translate fails
        lim_fmrt: translated limit
    '''

    if dateformat.lower() == 'datetime': #if want datetime
        if isinstance(lim, dt.datetime):
            return lim
        else:
            try:
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

            WF.print2stdout('Trying as Jdate..', debug=debug)
            try:
                lim_frmt = float(lim)
                lim_frmt = JDateToDatetime(lim_frmt, StartTime.year)
                WF.print2stdout('JDate {0} as {1} Accepted!'.format(lim, lim_frmt), debug=debug)
                return lim_frmt
            except:
                WF.print2stdout('Limit value of {0} also invalid as jdate.'.format(lim), debug=debug)

            if fallback != None and fallback != '':
                WF.print2stdout('Setting to fallback {0}.'.format(fallback), debug=debug)
            else:
                WF.print2stdout('Setting to fallback.', debug=debug)
            return fallback

    elif dateformat.lower() == 'jdate':
        try:
            return float(lim)
        except:
            WF.print2stdout('Error Reading Limit: {0} as a jdate.'.format(lim), debug=debug)
            WF.print2stdout('If this is wrong, try format: 180', debug=debug)
            WF.print2stdout('Trying as Datetime..', debug=debug)
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

    il_indeces = [i for i, date in enumerate(dates) if date.month in window]

    # trim the dates and values
    dates = dates[il_indeces]
    values = {member: vals[il_indeces] for member, vals in values.items()}

    return dates, values