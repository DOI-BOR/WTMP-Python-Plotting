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

import numpy as np
from functools import reduce
from scipy import interpolate

import WAT_Functions as WF
import WAT_Reader as WR
import WAT_Time as WT

class Profiles(object):

    def __init__(self, Report):
        '''
        Class to control profile objects
        :param Report: self class from main Report Generator script
        '''
        self.Report = Report

    def getProfileDates(self, Line_info, StartTime, EndTime):
        '''
        gets dates from observed text profiles
        :param Line_info: dictionary containing line information, must include filename
        :return: list of times
        '''

        if 'filename' in Line_info.keys(): #Get data from Observed
            times = WR.getTextProfileDates(Line_info['filename'], StartTime, EndTime) #TODO: set up for not observed data??
            return times

        WF.print2stdout('Illegal Dates selection.', debug=self.Report.debug)
        return []

    def getProfileInterpResolution(self, object_settings, default=30):
        '''
        Gets the resolution value to interpolate profile data over for table stats. default value of 30
        if not defined.
        :param object_settings: currently selected object settings dictionary
        :param default: used if not defined in user settings
        :return: interpolation int value
        '''

        keys = [n['flag'] for n in object_settings[object_settings['datakey']]]
        if 'resolution' in object_settings.keys():
            resolution = object_settings['resolution']
            return int(resolution)
        elif 'interpolationsource' in object_settings.keys():
            resolution = object_settings['interpolationsource']
            if resolution in keys:
                return resolution
            else:
                WF.print2stdout(f'InterpolationSource {resolution} not found in keys. Setting to default value '
                                f'resolution: {default}', debug=self.Report.debug)
                resolution = default
                return int(resolution)
        else:
            if 'Observed' in keys:
                return 'Observed'
            else:
                WF.print2stdout(f'Resolution not defined. Setting to default value resolution: {default}', debug=self.Report.debug)
                resolution = default
                return int(resolution)

    def getProfileTimestamps(self, object_settings, StartTime, EndTime):
        '''
        Gets timestamps based off of user settings in XML file and reads/builds them.
        :param object_settings: currently selected object settings dictionary
        :return: list of timestamp values to be plotted
        '''

        if isinstance(object_settings['datessource_flag'], str):
            timestamps = []
            for line in object_settings[object_settings['datakey']]:
                if line['flag'] == object_settings['datessource_flag']:
                    timestamps = self.getProfileDates(line, StartTime, EndTime)
        elif isinstance(object_settings['datessource_flag'], dict): #single date instance..
            timestamps = []
            if 'dates' in object_settings['datessource'].keys():
                datekey = 'dates'
            elif 'date' in object_settings['datessource'].keys():
                datekey = 'date'
            tstamp_dates = object_settings['datessource'][datekey]
            for d in tstamp_dates:
                dfrmt = WT.translateDateFormat(d, 'datetime', None, StartTime, EndTime, None, debug=self.Report.debug)
                if dfrmt != None:
                    timestamps.append(dfrmt)
                else:
                    WF.print2stdout('Invalid Timestamp', d, debug=self.Report.debug)

        elif isinstance(object_settings['datessource_flag'], list): #single date instance..
            timestamps = []
            tstamp_dates = object_settings['datessource']
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
        '''
        gets month indexes for each year based off timestamps for profile tables
        :param object_settings: dictionary containing settings for current object
        :return: list of lists for years/months and timestamps
        '''

        timestamp_indexes = []
        for year in years:
            year_idx = []
            for mon in range(1,13):
                mon_idx = []
                for ti, timestamp in enumerate(object_settings['timestamps']):
                    if timestamp.year == year and timestamp.month == mon:
                        mon_idx.append(ti)
                year_idx.append(mon_idx)
            timestamp_indexes.append(year_idx)
        return timestamp_indexes

    def convertDepthsToElevations(self, data, wse_data, timestamp_index=None):
        '''
        handles data to convert depths into elevations for observed data
        :param data: dictionary containing values for lines
        :param wse_data: contains info about WSE for conversion
        :return: object settings dictionary with updated elevation data
        '''

        for ld in data.keys():
            # found_elevs = False
            if data[ld]['elevations'] == []:
                noelev_flag = ld
                wse_data_key = ld + '_wse'

                converted_elevations = []

                otherkey = None
                if 'Computed' in data.keys():
                    if len(data['Computed']['elevations']) > 0:
                        otherkey = 'Computed'
                else:
                    for key in data.keys():
                        if len(data[key]['elevations']) > 0:
                            otherkey = key
                            break

                use_index = False
                if timestamp_index != None:
                    use_index = True
                    timesteps = [timestamp_index]  # use the index directly
                else:
                    timesteps = data[ld]['times']

                for tsi, ts in enumerate(timesteps):
                    if wse_data_key in wse_data.keys():
                        if use_index:
                            try:
                                wse_at_timestep = wse_data[wse_data_key]['elevations'][ts]
                                e = self.convertObsDepths2Elevations(data[noelev_flag]['depths'][tsi],
                                                                     wse_at_timestep)
                            except IndexError:
                                e = np.full_like(data[noelev_flag]['depths'], fill_value=np.nan)
                        else:
                            wse_at_timestep = self.matchProfileTimestamps(ts, wse_data[wse_data_key], onflag='elevations')['elevations']
                            e = self.convertObsDepths2Elevations(data[noelev_flag]['depths'][tsi],
                                                                 wse_at_timestep)
                    else:
                        e = np.full_like(data[noelev_flag]['depths'][tsi], fill_value=np.nan)
                    if np.all(np.isnan(e)): #if theyre all nan
                        if otherkey != None:
                            try:
                                selected_elevation_data = data[otherkey]['elevations'][tsi]
                                maxelev = WF.getMaxWSEFromElev(selected_elevation_data)
                                e = self.convertObsDepths2Elevations(data[noelev_flag]['depths'][tsi],
                                                                     maxelev)
                            except IndexError:
                                e = np.full_like(data[noelev_flag]['depths'][tsi], fill_value=np.nan)
                        else:
                            e = np.full_like(data[noelev_flag]['depths'][tsi], fill_value=np.nan)

                    converted_elevations.append(e)
                data[noelev_flag]['elevations'] = converted_elevations
        return data

    def convertElevationsToDepths(self, data, wse_data, timestamp_index=None):
        '''
        handles data to convert depths into elevations for observed data
        :param data: dictionary containing values for lines
        :param wse_data: contains info about WSE for conversion
        :return: object settings dictionary with updated elevation data
        '''

        for ld in data.keys():
            if data[ld]['depths'] == []:
                nodepth_flag = ld
                wse_data_key = ld + '_wse'

                converted_depths = []

                otherkey = None
                if 'Computed' in data.keys():
                    if len(data['Computed']['depths']) > 0:
                        otherkey = 'Computed'
                else:
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

                for tsi, ts in enumerate(timesteps):
                    if wse_data_key in wse_data.keys():
                        if use_index:
                            try:
                                wse_at_timestep = wse_data[wse_data_key]['elevations'][ts]
                                d = self.convertObsElevations2Depths(data[nodepth_flag]['elevations'][tsi],
                                                                     wse_at_timestep)
                            except IndexError:
                                d = np.full_like(data[nodepth_flag]['elevations'], fill_value=np.nan)
                        else:
                            wse_at_timestep = self.matchProfileTimestamps(ts, wse_data[wse_data_key], onflag='elevations')['elevations']
                            d = self.convertObsElevations2Depths(data[nodepth_flag]['elevations'][tsi],
                                                                 wse_at_timestep)
                    else:
                        d = np.full_like(data[nodepth_flag]['elevations'], fill_value=np.nan)
                    if np.all(np.isnan(d)): #if theyre all nan
                        if otherkey != None:
                            try:
                                selected_elevation_data = data[otherkey]['elevations'][tsi]
                                maxelev = WF.getMaxWSEFromElev(selected_elevation_data)
                                d = self.convertObsElevations2Depths(data[nodepth_flag]['elevations'][tsi],
                                                                     maxelev)
                            except IndexError:
                                d = np.full_like(data[nodepth_flag]['elevations'], fill_value=np.nan)
                        else:
                            d = np.full_like(data[nodepth_flag]['elevations'], fill_value=np.nan)

                    converted_depths.append(d)
                data[nodepth_flag]['depths'] = converted_depths
        return data

    def convertObsDepths2Elevations(self, input_depths, max_wse):
        '''
        calculate observed elevations based on model elevations and obs depths
        :param input_depths: array of depths for observed data at timestep
        :param reference_elevs: array of model elevations at timestep
        :return: array of observed elevations
        '''

        obs_elev = []
        if isinstance(max_wse, (list, np.ndarray)):
            if len(max_wse) == 0:
                return np.full(len(input_depths), np.nan)
            if np.all(np.isnan(max_wse)):
                return np.full(len(input_depths), np.nan)
        elif np.isnan(max_wse):
            return np.full(len(input_depths), np.nan)
        for depth in input_depths:
            obs_elev.append(max_wse - depth)
        return np.asarray(obs_elev)

    def convertObsElevations2Depths(self, input_elevs, max_wse):
        '''
        calculate observed elevations based on model elevations and obs depths
        :param obs_depths: array of depths for observed data at timestep
        :param model_elevs: array of model elevations at timestep
        :return: array of observed elevations
        '''

        out_depth = []
        if len(max_wse) == 0:
            out_depth.append(np.full(len(input_elevs), np.nan)) #make nan boys
        else:
            for i, e in enumerate(input_elevs):
                d = []
                for elev in e:
                    d.append(max_wse - elev)
                out_depth.append(np.asarray(d))
        return out_depth

    def convertProfileDataUnits(self, object_settings, data, line_settings):
        '''
        converts the units of profile data if unitsystem is defined
        :param object_settings: user defined settings for current object
        :return: object_setting dictionaries with updated units and values
        '''

        if 'unitsystem' not in object_settings.keys():
            WF.print2stdout('Unit system (unitsystem) not defined.', debug=self.Report.debug)
            # return data, line_settings
        else:
            for flag in data.keys():
                if line_settings[flag]['units'] == None:
                    continue
                else:
                    profiles = data[flag]['values']
                    profileunits = line_settings[flag]['units']
                    for pi, profile in enumerate(profiles):
                        profile, newunits = WF.convertUnitSystem(profile, profileunits, object_settings['unitsystem'])
                        profiles[pi] = profile
                    line_settings[flag]['units'] = newunits
        if 'y_unitsystem' not in object_settings.keys():
            WF.print2stdout('Y Unit system (y_unitsystem) not defined.', debug=self.Report.debug)
        else:
            for flag in data.keys():
                if line_settings[flag]['y_units'] == None:
                    continue
                else:
                    yunits = line_settings[flag]['y_units']
                    for pi, profile in enumerate(data[flag]['depths']):
                        profile, newunits = WF.convertUnitSystem(profile, yunits, object_settings['y_unitsystem'])
                        data[flag]['depths'][pi] = profile
                    for pi, profile in enumerate(data[flag]['elevations']):
                        profile, newunits = WF.convertUnitSystem(profile, yunits, object_settings['y_unitsystem'])
                        data[flag]['elevations'][pi] = profile
                    line_settings[flag]['y_units'] = newunits
        return data, line_settings

    def filterProfileData(self, data, line_settings, object_settings):
        '''
        filters profile data by ylims, xlims, and omitvalues
        :param data: dictionary containing data
        :param line_settings: dictionary contining settings for line
        :param object_settings: dictionary containing settings for object
        :return: filtered profile dictionary
        '''

        xmax = None
        xmin = None
        ymax = None
        ymin = None

        if 'usedepth' in object_settings.keys():
            if object_settings['usedepth'].lower() == 'true':
                yflag = 'depths'
                other_yflag = 'elevations'
            else:
                yflag = 'elevations'
                other_yflag = 'depths'
        else:
            WF.print2stdout('UseDepth flag not set. Cannot filter properly.', debug=self.Report.debug)
            return data, object_settings

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

            filtbylims = False
            if 'filterbylimits' in cur_line_settings.keys():
                if cur_line_settings['filterbylimits'].lower() == 'true':
                    filtbylims = True
            else:
                if 'filterbylimits' in object_settings.keys():
                    if object_settings['filterbylimits'].lower() == 'true':
                        filtbylims = True

            if 'omitvalue' in cur_line_settings.keys():
                omitvalues = [float(cur_line_settings['omitvalue'])]
            elif 'omitvalues' in cur_line_settings.keys():
                omitvalues = [float(n) for n in cur_line_settings['omitvalues']]
            else:
                omitvalues = None

            for pi, profile in enumerate(cur_data['values']):
                ydata = cur_data[yflag][pi]

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
                    omitvals_filt = []
                    for omitval in omitvalues:
                        omitval_filt = np.where(profile != omitval)
                        omitvals_filt = np.append(omitvals_filt, omitval_filt)
                else:
                    omitvals_filt = np.arange(len(profile))

                master_filter = reduce(np.intersect1d, (xmax_filt, xmin_filt, ymax_filt, ymin_filt, omitvals_filt)).astype(int)

                data[lineflag]['values'][pi] = profile[master_filter]
                try:
                    if len(cur_data[other_yflag][pi]) == len(cur_data[yflag][pi]):
                        #if there isnt enough data, dont filter it the same. They need to be the same.
                        data[lineflag][other_yflag][pi] = cur_data[other_yflag][pi][master_filter]
                except IndexError:
                    if len(cur_data[other_yflag]) != len(cur_data[yflag]):
                        WF.print2stdout(f'Cannot filter {other_yflag} due to no values', debug=self.Report.debug)
                    else:
                        WF.print2stdout(f'Cannot filter {other_yflag} due to different number of values compared to '
                                        f'{yflag}. {len(cur_data[other_yflag][pi])}: {len(cur_data[yflag][pi])}',
                                        debug=self.Report.debug)
                data[lineflag][yflag][pi] = ydata[master_filter]

        return data, object_settings

    def stackProfileIndicies(self, exist_data, new_data):
        '''
        takes an existing array of data and adds another array
        for contour plots of several reaches split into different groups
        stacks them together so they function as a single reach
        exist_data is existing array
        new data is data to be added to it
        :param exist_data: dictionary containing existing data
        :param new_data: data to be added to existing data
        :return: modified exist_data
        '''

        for runflag in new_data.keys():
            if runflag not in exist_data.keys():
                exist_data[runflag] = {}
            for itemflag in new_data[runflag]:
                if itemflag not in exist_data[runflag].keys():
                    exist_data[runflag][itemflag] = new_data[runflag][itemflag]
                else:
                    if isinstance(new_data[runflag][itemflag], list):
                        exist_data[runflag][itemflag] += new_data[runflag][itemflag]
                    elif isinstance(new_data[runflag][itemflag], np.ndarray):
                        exist_data[runflag][itemflag] = np.append(exist_data[runflag][itemflag], new_data[runflag][itemflag])
        return exist_data

    def normalize2DElevations(self, vals, elevations):
        '''
        interpolates reservoir data in order to normalize the list of elevations for W2 runs
        :param vals: list of lists of values at timestamps/elevations
        :param elevations: list of lists of elevations at timestamps
        :return: new values, new elevations
        '''

        newvals = []
        top_elev = np.nanmax([np.nanmax(n) for n in elevations if ~np.all(np.isnan(n))])
        bottom_elev = np.nanmin([np.nanmin(n) for n in elevations if ~np.all(np.isnan(n))])
        new_elevations = np.linspace(bottom_elev, top_elev, elevations.shape[1])
        for vi, v in enumerate(vals):
            valelev_interp = interpolate.interp1d(elevations[vi], v, bounds_error=False, fill_value ="extrapolate")
            newvals.append(valelev_interp(new_elevations))
        return np.asarray(newvals), np.asarray(new_elevations)

    def matchProfileTimestamps(self, input_timestamps, timeseries_dict, onflag='values'):
        '''
        gets values from time series that allign with profile dates
        :param input_timestamps: profile timesteps in a list
        :param timeseries_dict: dictionary containing times and values
        :param onflag: optional flag incase 'values' is the incorrect flag in timeseries_dict
        :return: new dict with selected values and dates
        '''

        output = {}
        timestamp_idx = WR.getClosestTime(input_timestamps, timeseries_dict['dates'])
        output[onflag] = timeseries_dict[onflag][timestamp_idx]
        output['dates'] = timeseries_dict['dates'][timestamp_idx]
        for key in timeseries_dict.keys():
            if key not in output.keys():
                output[key] = timeseries_dict[key]
        return output

    def checkProfileValidity(self, data, object_settings, combineyears=False, includeallyears=False):
        '''
        checks profile data validity by checking for data outisde a wide margin, or min amount of points, or
        surface grouping. prints large amounts of messages if invalid
        :param data: dict containing data
        :param object_settings: dict containing settings for object
        :param combineyears: boolean
        :param includeallyears: boolean
        :return: warning info
        '''

        if not self.Report.debug:
            return {}
        if 'warnings' not in object_settings.keys():
            object_settings['warnings'] = {}
        range_percent_threshold = 1 #percent of the range to use for clustering detection
        percent_vals_under_threshold = 25 #percent of values in threshold for clustering detection
        minimum_number_values = 5 #min amount of points

        if 'splitbyyear' in object_settings.keys():
            if object_settings['splitbyyear'].lower() == 'false':
                combineyears = True
        if 'includeallyears' in object_settings.keys():
            if object_settings['includeallyears'].lower() == 'true':
                includeallyears = True

        for di, d in enumerate(data.keys()):
            WF.print2stdout(f'Assessing Dataset: {d}', debug=self.Report.debug)
            if d not in object_settings['warnings'].keys():
                object_settings['warnings'][d] = {}
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
                WF.print2stdout('No values for dataset.', debug=self.Report.debug)
                continue

            yvalues = data[d][yflag]

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

                    if yvalset[0] > yvalset[-1]:
                        increasing = False
                    else:
                        increasing = True

                    if len(yvalset) > len(list(set(yvalset))):
                        has_duplicates = True

                    if len(yvalset) < minimum_number_values:
                        enough_points = False

                    for yvi, yval in enumerate(yvalset):
                        if yvi == 0:
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

                        if (threshold_number_vals / number_of_points) * 100 > percent_vals_under_threshold:
                            WF.print2stdout(f'Profile may contain top clustering.', debug=self.Report.debug)
                            object_settings['warnings'][d][yearflag].append('clustering')
                else:
                    WF.print2stdout(f'No values for {d} for {data[d]["times"][yvsi]}', debug=self.Report.debug)
                    continue
            if combineyears or includeallyears:
                allwarnings = []
                for yearflag in object_settings['warnings'][d].keys():
                    for warningflag in object_settings['warnings'][d][yearflag]:
                        allwarnings.append(warningflag)
                if combineyears: #only output all years
                    object_settings['warnings'][d] = {'ALLYEARS': allwarnings}
                elif includeallyears:
                    object_settings['warnings'][d]['ALLYEARS'] = allwarnings
            for yearkey in object_settings['warnings'][d].keys():
                object_settings['warnings'][d][yearkey] = list(set(object_settings['warnings'][d][yearkey]))

        return object_settings['warnings']

    def writeWarnings(self, warnings, year):
        '''
        writes profile warning messages into report
        :param warnings:
        :param year:
        :return:
        '''
        for key in warnings.keys():
            if len(warnings[key][year]) > 0:
                message = self.formatProfileWarningMessages(warnings[key][year], key)
                self.Report.makeTextBox({'text': message})

    def formatProfileWarningMessages(self, warnings, key):
        '''
        formats warnings about profiles
        :param warnings: list of issues with profiles
        :param key: key for which profile to complain about
        :return: formatted message
        '''

        message = f'Some profiles in {key} may be invalid due to'
        if len(warnings) > 2:
            for wi, warn in enumerate(warnings):
                if wi == len(warnings) - 1:
                    message += f' and {warn}.'
                else:
                    message += f' {warn},'
        elif len(warnings) == 2:
            message += f' {warnings[0]} and {warnings[1]}'
        else:
            message += f' {warnings[0]}.'
        return message

    def confirmValidDepths(self, data):
        '''
        checks to confirm profile tables can use depths
        :param data: dictionary of profile data
        :return: boolean value
        '''

        usedepths = 'True' #innocent until proven guilty
        for key in data.keys():
            numDepthProfiles = len(data[key]['depths'])
            numElevProfiles = len(data[key]['elevations'])
            if numDepthProfiles == 0 and numElevProfiles > 0:
                WF.print2stdout(f'Not using depths for {key}.', debug=self.Report.debug)
                usedepths = 'False'
            for dpi, depthprofile in enumerate(data[key]['depths']):
                if len(depthprofile) == 0 and len(data[key]['elevations'][dpi]) > 0:
                    WF.print2stdout(f'Not using depths for {key}.', debug=self.Report.debug)
                    usedepths = 'False'
        return usedepths

    def snapTo0Depth(self, data, line_settings):
        '''
        takes depth data and adds an entry in the data using the last known value
        :param data: dictionary of profile data
        :param line_settings: dictionary of line settings
        :return:
        '''

        for key in data.keys():
            if 'snapto0depth' in line_settings[key].keys():
                if line_settings[key]['snapto0depth'].lower() == 'true':
                    for dsi, depthset in enumerate(data[key]['depths']):
                        if len(depthset) == 0:
                            continue
                        if 0.0 not in depthset:
                            distance_from_wse = min(depthset)
                            min_depth_i = np.where(depthset == distance_from_wse)
                            max_elevation = max(data[key]['elevations'][dsi])
                            wse_elevation = max_elevation + distance_from_wse
                            data[key]['elevations'][dsi][min_depth_i] = wse_elevation
                            data[key]['depths'][dsi][min_depth_i] = 0.0

        return data
