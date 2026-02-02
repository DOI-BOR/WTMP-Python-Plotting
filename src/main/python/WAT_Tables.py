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

import datetime as dt
import pickle
import numpy as np
from matplotlib.colors import to_hex
from scipy import interpolate

import WAT_Functions as WF
import WAT_Time as WT

class Tables(object):

    def __init__(self, Report):
        '''
        Class to control table objects
        :param Report: self class from main Report Generator script
        '''
        self.Report = Report
        if self.Report.reportType == 'forecast':
            self.defineForecastTableHeaders()

    def buildHeadersByTimestamps(self, timestamps, years):
        '''
        build headers for profile line stat tables by timestamp
        convert to Datetime, no matter what. We can convert back..
        Filter by year, using year input. If ALLYEARS, no data is filtered.
        :param timestamps: list of available timesteps
        :param years: used to filter down to the year, or if ALLYEARS, allow all years
        :return: list of headers
        '''

        headers = []
        headers_i = []

        for year in years:
            h = []
            hi = []
            for ti, timestamp in enumerate(timestamps):
                if isinstance(timestamp, dt.datetime):
                    if year == timestamp.year:
                        h.append(timestamp)
                        hi.append(ti)

                elif isinstance(timestamp, float):
                    ts_dt = WT.JDateToDatetime(timestamp, self.Report.startYear)
                    if year == ts_dt.year:
                        h.append(str(timestamp))
                        hi.append(ti)
            headers.append(h)
            headers_i.append(hi)

        return headers, headers_i

    def buildErrorStatsTable(self, object_settings, data_settings):
        '''
        builds error stats table
        :param object_settings: dictionary of settings for object
        :param data_settings: dictionary containing data metadata
        :return: headers and rows for table
        '''

        headers = []
        rows = []
        for ri, row in enumerate(object_settings['rows']):
            rows.append(row.split('|')[0])

        if self.Report.iscomp: #comp run
            for i, header in enumerate(object_settings['headers']):
                curheader = pickle.loads(pickle.dumps(header, -1))
                for datakey in data_settings.keys():
                    ds = data_settings[datakey]
                    dk_flag = ds['flag']
                    dk_keys = ds.keys()
                    isused = False
                    if '%%{0}%%'.format(dk_flag) in curheader:
                        #like %%Computed%%%%SimulationName%%, %%Observed%%, etc..
                        if 'label' in dk_keys: #if theres a label, just use that, easy
                            # curheader = curheader.replace('%%{0}%%'.format(dk_flag), ds['label'])
                            curheader = ds['label']
                        elif 'ID' in dk_keys: #otherwise we will go find the settings and search for flags that are model spec
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
                        for ri, row in enumerate(object_settings['rows']):
                            srow = row.split('|')[1:][i]
                            rows[ri] += '|{0}'.format(srow.replace(dk_flag, datakey))

        else:
            headers = object_settings['headers']
            rows = object_settings['rows']

        return headers, rows

    def buildMonthlyStatsTable(self, object_settings, data_settings):
        '''
        builds monthly stats table
        :param object_settings: dictionary of settings for object
        :param data_settings: dictionary containing data metadata
        :return: headers and rows for table
        '''

        headers = []
        rows = []
        for ri, row in enumerate(object_settings['rows']):
            rows.append(row.split('|')[0])
        if 'headers' in object_settings.keys():
            assigned_headers = object_settings['headers']
        else:
            assigned_headers = []
        #unlike error tables, we need to build by row here..

        if self.Report.iscomp: #comp run
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
            headers = object_settings['headers']
            rows = object_settings['rows']

        return headers, rows

    def buildSingleStatTable(self, object_settings, data):
        '''
        builds headings and rows for single stat table
        :param object_settings: dictionary containing settings for table
        :param data: dictionary contianing data
        :return: headings and rows used to build tables
        '''

        headers = []
        rows = []
        months = [n for n in self.Report.Constants.mo_str_3]
        stat = object_settings['statistic']
        datakeys = list(data.keys())
        if len(datakeys) == 0:
            hasdata = False
        else:
            hasdata = True

        if stat in ['mean', 'count']:
            numflagsneeded = 1
        else:
            numflagsneeded = 2

        if len(datakeys) < numflagsneeded:
            hasdata = False

        if self.Report.iscomp:
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
            headers = months

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

        for year in object_settings['years']:
            row = f'{year}'
            for month in months:
                if not hasdata:
                    row += f'|{missingmarker}'
                else:
                    if numflagsneeded == 1:
                        for datakey in computed_keys:
                            row += f'|%%{stat}.{datakey}.MONTH={month.upper()}%%'
                    else:
                        if len(computed_keys) == 0:
                            row += f'|{missingmarker}'
                        else:
                            for cflag in computed_keys:
                                if len(observed_keys) == 0:
                                    row += f'|{missingmarker}'
                                else:
                                    for oflag in observed_keys:
                                        row += f'|%%{stat}.{cflag}.MONTH={month.upper()}.{data[oflag]["flag"]}.MONTH={month.upper()}%%'
            rows.append(row)

        return headers, rows

    def buildFormattedTable(self, data):
        '''
        builds headers and rows for formatted tables
        :param data: pandas dataframe of table
        :return: headers and rows
        '''

        headers = data.columns #ez
        rows = []
        for i, row in data.iterrows():
            built_row = ''
            for rowval in row.values:
                if built_row == '':
                    built_row = str(rowval)
                else:
                    built_row += f'|{rowval}'
            rows.append(built_row)
        return headers, rows

    def buildProfileStatsTable(self, object_settings, timestamp, data):
        '''
        builds a table header and rows for profile statistics
        :param object_settings: dictionary containing settings
        :param timestamp: current timestamp to configure table for
        :param data: dictionary contianing data
        :return: list of headers, list of rows
        '''

        headers = []
        rows = []
        for ri, row in enumerate(object_settings['rows']):
            rows.append(row.split('|')[0])

        if self.Report.iscomp: #comp run
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
            headers = [timestamp]
            rows = object_settings['rows']

        return headers, rows

    def filterTableData(self, data, object_settings):
        '''
        filters out data through xlims, ylims and omitvalues.
        :param data: dictionary of data
        :param object_settings: settings for current object containing limits
        :return: filtered data dictionary
        '''

        xmax = None
        xmin = None
        ymax = None
        ymin = None

        if 'xlims' in object_settings.keys():
            if 'max' in object_settings['xlims'].keys():
                # xmax = float(object_settings['xlims']['max'])
                if object_settings['xlims']['max'] is None:
                    xmax = self.Report.EndTime
                else:
                    xmax = WF.updateFlaggedValues(object_settings['xlims']['max'],'%%year%%', str(max(self.Report.years)))
                    xmax = WT.translateDateFormat(xmax, 'datetime', self.Report.EndTime, self.Report.StartTime,
                                                  self.Report.EndTime, debug=self.Report.debug)

            if 'min' in object_settings['xlims'].keys():
                # xmin = float(object_settings['xlims']['min'])
                if object_settings['xlims']['min'] is None:
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

            filtbylims = True
            if 'filterbylimits' in line.keys():
                if line['filterbylimits'].lower() == 'false':
                    filtbylims = False
            else:
                if 'filterbylimits' in object_settings.keys():
                    if object_settings['filterbylimits'].lower() == 'false':
                        filtbylims = False

            if 'omitvalue' in line.keys():
                omitvalues = [float(line['omitvalue'])]
            elif 'omitvalues' in line.keys():
                omitvalues = [float(n) for n in line['omitvalues']]
            else:
                omitvalues = None

            ### FALSE WHEN OUT OF BOUNDS, TRUE WHEN KEEP

            if xmax != None and filtbylims:

                xmax_filt = (dates <= xmax)
            else:
                xmax_filt = np.full(dates.shape, True)

            if xmin != None and filtbylims:
                xmin_filt = (dates >= xmin)
            else:
                xmin_filt = np.full(dates.shape, True)

            if ymax != None and filtbylims:
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
                    for key, vs in values.items():
                        omit_filt[key] = []
                        for omitval in omitvalues:
                            omitval_filt = (vs != omitval)
                            omit_filt[key] = np.append(omitvals_filt, omitval_filt)
                else:
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
        '''
        converts units for table data
        :param data: dictionary containing data
        :param data_settings: data metadata
        :param object_settings: settings for plot/table
        :return: data with updated units
        '''

        for datapath in data.keys():
            values = data[datapath]['values']
            units = data_settings[datapath]['units']
            if 'parameter' in data_settings[datapath].keys():
                units = WF.configureUnits(object_settings, data_settings[datapath]['parameter'], units)
            if 'unitsystem' in object_settings.keys():
                data[datapath]['values'], data_settings[datapath]['units'] = WF.convertUnitSystem(values, units, object_settings['unitsystem'], debug=self.Report.debug)

        return data, data_settings

    def getStatsLineData(self, row, data_dict, year='ALLYEARS', data_key=None):
        '''
        takes rows for tables and replaces flags with the correct data, computing stat analysis if needed
        :param row: row section string
        :param data_dict: dictionary of data that could be used
        :param year: selected year, or 'ALL'
        :return: new row value
        '''

        data = {}

        rrow = row.replace('%%', '')
        s_row = rrow.split('.')
        sr_month = ''
        curflag = None
        for sr in s_row:
            if sr in data_dict.keys():
                curflag = sr
                curvalues = data_dict[sr]['values']
                curdates = np.array(data_dict[sr]['dates'])
                if data_key != None:
                    data[curflag] = {'values': curvalues[data_key], 'dates': curdates}
                else:
                    if isinstance(curvalues, dict):
                        WF.print2stdout('Unable to get data for row. Expected key but none found.', debug=self.Report.debug)
                        return data, sr_month
                    else:
                        data[curflag] = {'values': np.asarray(curvalues), 'dates': curdates}
            else:
                if '=' in sr:
                    sr_spl = sr.split('=')
                    if sr_spl[0].lower() == 'month':
                        sr_month = sr_spl[1]
                        try:
                            sr_month = int(sr_month)
                        except ValueError:
                            try:
                                sr_month = self.Report.Constants.month2num[sr_month.lower()]
                            except KeyError:
                                WF.print2stdout('Invalid Entry for {0}'.format(sr), debug=self.Report.debug)
                                WF.print2stdout('Try using interger values or 3 letter monthly code.', debug=self.Report.debug)
                                WF.print2stdout('Ex: MONTH=1 or MONTH=JAN', debug=self.Report.debug)
                                continue
                        if curflag == None:
                            WF.print2stdout('Invalid Table row for {0}'.format(row), debug=self.Report.debug)
                            WF.print2stdout('Data Key not contained within {0}'.format(data_dict.keys()), debug=self.Report.debug)
                            WF.print2stdout('Please check Datapaths in the XML file, or modify the rows to have the correct flags'
                                  ' for the data present', debug=self.Report.debug)
                            return data, ''

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
                                    yearvals = []
                                    yeardates = []

                                if len(yeardates) > 0:
                                    s_idx, e_idx = WF.getMonthlyFilterIdx(yeardates, sr_month)

                                    newvals = np.append(newvals, yearvals[s_idx:e_idx+1])
                                    newdates = np.append(newdates, yeardates[s_idx:e_idx+1])

                        data[curflag]['values'] = newvals
                        data[curflag]['dates'] = newdates

        if year != 'ALLYEARS':
            for flag in data.keys():
                if len(data[flag]['dates']) == 0:
                    continue
                s_idx, e_idx = WF.getYearlyFilterIdx(data[flag]['dates'], year)
                if None not in [s_idx, e_idx]:
                    data[flag]['values'] = data[flag]['values'][s_idx:e_idx+1]
                    data[flag]['dates'] = data[flag]['dates'][s_idx:e_idx+1]
                else:
                    data[flag]['values'] = []
                    data[flag]['dates'] = []

        return data, sr_month

    def getStatsLine(self, row, data):
        '''
        takes rows for tables and replaces flags with the correct data, computing stat analysis if needed
        :param row: row section string
        :param data: dictionary of data that could be used
        :return:
            out_stat: stat value
            stat: string name for stat
        '''

        # stat_flag_Req = {'%%meanbias': 2,
        #                  '%%mae': 2,
        #                  '%%rmse': 2,
        #                  '%%nse': 2,
        #                  '%%count': 2, #can also be 1
        #                  '%%mean': 1}

        flags = list(data.keys())

        if len(flags) > 0:
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
            WF.print2stdout(f'Insufficient data for row {row}.', debug=self.Report.debug)
            WF.print2stdout(f'Flags: {flags}', debug=self.Report.debug)
            return np.nan, ''

        out_stat = np.nan

        for key in data.keys():
            if len(data[key]) == 0:
                WF.print2stdout(f'No data in dataset {key}.', debug=self.Report.debug)
                return np.nan, ''

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
            if '%%' in row:
                WF.print2stdout('Unable to convert flag in row', row, debug=self.Report.debug)
            return row, ''

        return out_stat, stat

    def matchThresholdToStat(self, stat, object_settings):
        '''
        matches prescribed threshold values to statistic value
        :param stat: string name of stat
        :param object_settings: dictionary containing object settings
        :return:
        '''

        thresholds = []
        if 'tablecolors' in object_settings.keys() or 'thresholds' in object_settings.keys():
            if 'tablecolors' in object_settings.keys():
                WF.print2stdout('The flag "tablecolors" is deprecated as of 5.4.26. Please use "thresholds" instead, '
                                'including the specified "statistic" flag within the <threshold> object.')
                modflag = 'tablecolors'
            else:
                modflag = 'thresholds'
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
        '''
        matches how numbers should be formatted by what stat they are using user settings in tables
        :param stat: current statistical value
        :param settings: dictionary of settings to parse
        :return: formatting
        '''

        numberFormats_default = []
        numberFormats_statspec = []

        if 'numberformats' in settings.keys():
            for numberformat in settings['numberformats']:
                if 'stats' in numberformat:
                    if stat.lower() in [n.lower() for n in numberformat['stats']]:
                        numberFormats_statspec.append(numberformat)
                else:
                    numberFormats_default.append(numberformat)
        if isinstance(stat, str):
            if stat.lower() == 'count':
                if len(numberFormats_statspec) == 0:
                    numberFormats_statspec.append({'decimalplaces': 0})

        if len(numberFormats_statspec) > 0:
            return numberFormats_statspec
        else:
            return numberFormats_default

    def formatThreshold_deprec(self, object_settings):
        '''
        DEPRECATED AS OF 5.4.26
        organizes settings for thresholds for stat tables. Fills in missing values with defaults
        :param object_settings: dictionary containing settings for thresholds
        :return: list of dictionary objects for each threshold
        '''

        default_color = '#a6a6a6' #default, grey
        default_when = 'under' #default
        accepted_threshold_conditions = ['under', 'over']
        thresholds = []

        if 'thresholds' in object_settings.keys():
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
        '''
        loops over thresholds from object settings and formats them
        :param object_settings: dictionary of settings for object
        :return: list of formatted thresholds
        '''

        thresholds = []
        if 'thresholds' in object_settings.keys():
            for threshold in object_settings['thresholds']:
                thf = self.formatThreshold(threshold)
                if len(thf.keys()) > 0:
                    thresholds.append(thf)
        return thresholds

    def formatThreshold(self, threshold):
        '''
        organizes settings for threshold for stat tables. Fills in missing values with defaults
        :param threshold: dictionary containing settings for potential threshold
        :return: list of dictionary objects for each threshold
        '''

        default_color = '#a6a6a6' #default, grey
        default_when = 'under' #default
        accepted_threshold_conditions = ['under', 'over']

        threshold_settings = {}

        if 'value' in threshold.keys():
            threshold_settings['value'] = float(threshold['value'])
        else:
            return {} #dont record this threshold

        if 'color' in threshold.keys():
            threshold_settings['color'] = self.formatThresholdColor(threshold['color'], default=default_color)
        else:
            threshold_settings['color'] = default_color

        if 'colorwhen' in threshold.keys():
            if any([n.lower() == threshold['colorwhen'].lower() for n in accepted_threshold_conditions]):
                threshold_settings['colorwhen'] = threshold['colorwhen'].lower()
            else:
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
        '''
        formats input color to either turn to hex or use default
        :param in_color: string to test if legit
        :param default: color to use if in_color fails
        :return:hex color
        '''

        threshold_color = default
        if in_color.startswith('#'):
            threshold_color = in_color
        else:
            try:
                threshold_color = to_hex(in_color)
            except ValueError:
                WF.print2stdout(f'Invalid color of {in_color}', debug=self.Report.debug)

        return threshold_color

    def getTableDates(self, year, object_settings, month='None'):
        '''
        gets start and end dates from lines in tables for logging
        :param year: selected year int or 'all' string
        :param object_settings: dictionary of item setting
        :param month: selected month (for monthly table) or None
        :return: start and end date
        '''

        xmin = 'NONE'
        xmax = 'NONE'
        if 'xlims' in object_settings.keys():
            if 'min' in object_settings['xlims'].keys():
                if object_settings['xlims']['min'] is not None:
                    xmin = WF.updateFlaggedValues(object_settings['xlims']['min'],'%%year%%', str(min(self.Report.years)))
                    xmin = WT.translateDateFormat(xmin, 'datetime', self.Report.StartTime,
                                                  self.Report.StartTime, self.Report.EndTime,
                                                  debug=self.Report.debug)
                    xmin = xmin.strftime('%d %b %Y')
            if 'max' in object_settings['xlims'].keys():
                if object_settings['xlims']['max'] is not None:
                    xmax = WF.updateFlaggedValues(object_settings['xlims']['max'],'%%year%%', str(max(self.Report.years)))
                    xmax = WT.translateDateFormat(xmax, 'datetime', self.Report.EndTime,
                                                  self.Report.StartTime, self.Report.EndTime,
                                                  debug=self.Report.debug)
                    xmax = xmax.strftime('%d %b %Y')

        if xmin != 'NONE':
            start_date = xmin
        elif year == self.Report.startYear:
            start_date = self.Report.StartTime.strftime('%d %b %Y')
        else:
            if str(year).lower() == 'allyears':
                start_date = '01 Jan {0}'.format(self.Report.startYear)
            else:
                start_date = '01 Jan {0}'.format(year)

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
            try:
                month = int(month)
            except ValueError:
                month = self.Report.Constants.month2num[month.lower()]

            try:
                start_date = dt.datetime.strptime(start_date, '%d %b %Y').replace(month=month).strftime('%d %b %Y')
            except ValueError:
                start_date = dt.datetime.strptime(start_date, '%d %b %Y')
                start_date = start_date.replace(day=1)
                start_date = start_date.replace(month=month+1)
                start_date -= dt.timedelta(days=1)
                start_date = start_date.strftime('%d %b %Y')
            try:
                end_date = dt.datetime.strptime(end_date, '%d %b %Y').replace(month=month).strftime('%d %b %Y')
            except ValueError:
                end_date = dt.datetime.strptime(end_date, '%d %b %Y')
                end_date = end_date.replace(day=1)
                end_date = end_date.replace(month=month+1)
                end_date -= dt.timedelta(days=1)
                end_date = end_date.strftime('%d %b %Y')

        return start_date, end_date

    def convertHeaderFormats(self, headers, object_settings):
        '''
        converts the formats of headers for profile line data tables to the correct format
        if the dateformat is selected, returns a formatted string.
        if Jdate, the string of the float value is used
        Datetime if not specified
        :param headers: list of datetime objects for headers
        :param object_settings: user defined settings for current object
        :return: list of new headers
        '''

        if 'dateformat' not in object_settings.keys():
            object_settings['dateformat'] = 'datetime'

        new_headers = []
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
        '''
        formats the primary key of a table if its a collection
        :param data: dictionary of incoming data
        :param object_settings: dictionary of settings about object
        :return: dtat dictionary
        '''

        if 'formatprimaryascollection' in object_settings.keys():
            if object_settings['formatprimaryascollection'].lower() == 'true':
                primarykey = object_settings['primarykey']
                for datakey in data.keys():
                    df = data[datakey]
                    for i, row in df.iterrows():
                        df.loc[i, primarykey] = WF.formatMembers(row[primarykey])
                    data[datakey] = df
        return data

    def formatStatsProfileLineData(self, row, data_dict, interpolation, usedepth, index):
        '''
        formats Profile line statistics for table using user inputs
        finds the highest and lowest overlapping profile points and uses them as end points, then interpolates
        :param row: Row line from inputs. String seperated by '|' and using flags surrounded by '%%'
        :param data_dict: dictionary containing available line data to be used
        :param interpolation: number of values to interpolate to. this way each dataset has values at the same levels
                            and there is enough data to do stats over.
        :param usedepth: string bool for using depth or elevation fields
        :param index: date index for profile to use
        :return:
            out_data: dictionary containing values and depths/elevations
        '''

        rrow = row.replace('%%', '')
        s_row = rrow.split('.')
        flags = []
        out_data = {}
        for sr in s_row:
            if sr in data_dict.keys():
                flags.append(sr)

        useflagforinterp = False
        if isinstance(interpolation, str):
            # if interpolation in [n.lower() for n in flags]:
            if interpolation in flags:
                useflagforinterp = True
            else:
                WF.print2stdout(f'Flag for output {interpolation} not found in data flags {flags}. Defaulting to '
                                f'interpolating both at 30 pt resolution', debug=self.Report.debug)
                interpolation = 30

        if usedepth.lower() == 'true':
            y_flag = 'depths'
        else:
            y_flag = 'elevations'

        if not useflagforinterp:
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
                output_interp_yvalues = []
            elif bottom == top:
                output_interp_yvalues = []
            else:
                if usedepth.lower() == 'true':
                    #build elev profiles
                    output_interp_yvalues = np.arange(top, bottom, (bottom-top) / float(interpolation))
                else:
                    output_interp_yvalues = np.arange(bottom, top, (top-bottom) / float(interpolation))


        for flag in flags:
            out_data[flag] = {}
            #interpolate over all values and then get interp values

            if len(data_dict[flag]['values'][index]) < 2:
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
                    WF.print2stdout(f'Insufficient {y_flag} points for row {flag} in {row}', debug=self.Report.debug)
                    out_data[flag]['values'] = []
                    out_data[flag]['depths'] = []
                    out_data[flag]['elevations'] = []
                    continue

            else:
                if len(data_dict[interpolation][y_flag][index]) == 0:
                    WF.print2stdout(f'Insufficient {y_flag} points for row {interpolation} in {row}', debug=self.Report.debug)
                    out_data[flag]['values'] = []
                    out_data[flag]['depths'] = []
                    out_data[flag]['elevations'] = []
                    continue

            if not np.all(data_dict[flag][y_flag][index][:-1] != data_dict[flag][y_flag][index][1:]): #check for duplicate yvals
                WF.print2stdout(f'Found duplicate values in {y_flag} for {flag} at index {index}', debug=self.Report.debug)
                duplicatemask = data_dict[flag][y_flag][index][:-1] != data_dict[flag][y_flag][index][1:]
                duplicatemask = np.insert(duplicatemask, 0, True)
                data_dict[flag][y_flag][index] = data_dict[flag][y_flag][index][duplicatemask]
                data_dict[flag]['values'][index] = data_dict[flag]['values'][index][duplicatemask]

            if useflagforinterp:
                if flag == interpolation:
                    out_data[flag][y_flag] = data_dict[flag][y_flag][index]
                    out_data[flag]['values'] = data_dict[flag]['values'][index]
                else:
                    f_interp = interpolate.interp1d(data_dict[flag][y_flag][index], data_dict[flag]['values'][index],
                                                    bounds_error=False, fill_value=np.nan)
                    out_data[flag][y_flag] = data_dict[interpolation][y_flag][index]
                    out_data[flag]['values'] = f_interp(data_dict[interpolation][y_flag][index])
            else:

                f_interp = interpolate.interp1d(data_dict[flag][y_flag][index], data_dict[flag]['values'][index], fill_value='extrapolate')
                out_data[flag]['values'] = f_interp(output_interp_yvalues)
                out_data[flag][y_flag] = output_interp_yvalues

        return out_data

    def replaceComparisonSettings(self, object_settings, iscomp):
        '''
        replaces normal settings with comparison settings if found to be comparison plot
        :param object_settings: dictionary containing settings
        :param iscomp: boolean, is comparison run or not
        :return: settings with replaced settings
        '''

        replace_flags = {'comparisonheaders': 'headers'}
        replaced_defaults = object_settings['replaced_defaults']
        if iscomp:
            for comparisonflag in replace_flags.keys():
                normalflag = replace_flags[comparisonflag]
                if comparisonflag in object_settings.keys():
                    if normalflag in replaced_defaults:
                        continue
                    object_settings[normalflag] = object_settings[comparisonflag]

        return object_settings

    def replaceIllegalJasperCharacters(self, tablelist):
        '''
        replaces characters that Jasper cant put in the report
        :param tablelist: list of table values
        :return: new formatted list
        '''

        illegal_chars = {"<": "&#60;",
                         ">": "&#62;"}
        newtablelist = []
        for tl in tablelist:
            replaced = False
            for key, char in illegal_chars.items():
                if key in tl:
                    newtablelist.append(tl.replace(key, char))
                    replaced = True
            if not replaced:
                newtablelist.append(tl)
        return newtablelist

    def replaceIllegalJasperCharactersHeadings(self, headers):
        '''
        replaces illegal characters in headings
        :param headers: list of headers
        :return: formatted list
        '''

        return self.replaceIllegalJasperCharacters(headers)

    def replaceIllegalJasperCharactersRows(self, rows):
        '''
        replaces illegal characters in rows
        :param rows: string of rows separated with |
        :return: formatted rows
        '''

        new_rows = []
        for row in rows:
            new_rows.append('|'.join(self.replaceIllegalJasperCharacters(row.split('|'))))
        return new_rows

    def configureHeadingsGroups(self, headings):
        '''
        finds indicies of groupings for headers
        :param headings: list of headings
        :return: indicies of headers
        '''

        headings_groups = []
        for h in headings:
            if h[0] not in headings_groups:
                headings_groups.append(h[0])
        headings_i = [[] for n in headings_groups]
        for hgi, hgroup in enumerate(headings_groups):
            for hi, h in enumerate(headings):
                if str(h[0]) == str(hgroup):
                    headings_i[hgi].append(hi)
        return headings_i

    def configureRowsForCollection(self, rows, object_settings):
        '''
        configures rows for collections
        :param rows: list of rows to format
        :param object_settings: dictionary of settings about object
        :return:
        '''

        formatted_rows = []
        #figure out members first
        if 'members' in object_settings.keys(): #if a subset
            members = object_settings['members']
        else: #otherwise, get them all
            members = self.Report.allMembers
        for row in rows:
            if '%%member%%' in row:
                for member in members:
                    srow = row.split('|')
                    frow = []
                    for sr in srow:
                        if '%%member%%' in sr:
                            sr = sr.replace('%%member%%', f'%%member.{member}%%')
                        frow.append(sr)
                    formatted_rows.append('|'.join(frow))
            else:
                formatted_rows.append(row)
        return formatted_rows

    def writeTable(self, table_constructor):
        '''
        writes table from table constructor
        :param table_constructor: dictionary of values for table
        '''

        lastdatecol = ''
        for i in range(max(table_constructor.keys())+1):
            if i not in table_constructor.keys():
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
            self.Report.XML.writeDateColumnEnd() #write date column end ifccomp
        self.Report.XML.writeTableEnd()




    def writeMissingTableItemsWarning(self, description):
        '''
        writes warning into report that some table items cannot be written due to not having all data
        :param description: description of table
        '''

        self.Report.makeTextBox({'text': f'Some items in Table "{description}" not generated due to insufficient data.'})

    def writeMissingTableWarning(self, description):
        '''
        writes warning in table if table cannot be written due to missing data
        :param description: description of table
        '''

        self.Report.makeTextBox({'text': f'\nTable "{description}" not generated due to insufficient data.'})

    def defineForecastTableHeaders(self):
        '''
        defines the fancy names for forecast table headers
        :return: dictionary of names
        '''

        #{name from XML | display name}
        self.forecastTableHeaders = {'name': 'Name',
                                     'operationsname': 'Operations',
                                     'metname': 'Met Set',
                                     'temptargetname': 'Temp Target',
                                     'member': 'Member Number'
                                     }

    def confirmForecastTableHeaders(self, columns):
        '''
        confirms that user enteries for forecast tables are valid. If not, removes them from the table.
        :param columns: user inputted columns
        :return: list of approved columns
        '''

        rejected_columns = []
        approved_columns = []
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
        '''
        iterates through forecast headers and formats them to their fancy names
        :param headers: list of headers
        :return:
        '''

        formatted_headers = []
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
        row_split = [n.replace('%', '') for n in row_val.split('.')]
        for key in row_data:
            if key in row_split:
                check = WF.checkData(row_data[key]['values'])
                if not check:
                    return True
        return False


    def getStat(self, row_val):
        srv = row_val.split('.')[0].lower().replace('%', '')
        return srv
