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

    def __init__(self, W2_path, alt_name, alt_Dir, starttime, endtime, Report):
        '''
        Class Builder init.
        :param W2_path: path to W2 run
        :param alt_name: name of run alternative for pathing, ex: 'Shasta from DSS 14'
        :param alt_Dir: directory of the alternative
        :param starttime: starttime of simulation
        :param endtime: endtime of simulation
        :param interval_min: output time series interval in minutes (60 = 1HOUR, 15 = 4 outputs an hour)
        '''

        self.W2_path = W2_path
        self.alt_name = alt_name #confirm this terminology
        self.run_path = alt_Dir
        self.starttime = starttime
        self.endtime = endtime
        self.Report = Report
        # self.interval_min = interval_min #output time series

        control_file_csv = os.path.join(self.run_path, 'w2_con.csv') #this should always be the same, UNTIL IT WASNT
        control_file_npt = os.path.join(self.run_path, 'w2_con.npt') #this should always be the same, UNTIL IT WASNT
        if os.path.exists(control_file_csv):
            self.control_file = control_file_csv
            self.control_file_type = 'csv'
        elif os.path.exists(control_file_npt):
            self.control_file = control_file_npt
            self.control_file_type = 'npt'
        else:
            WF.print2stderr('Unknown or missing W2 control file.')
            sys.exit(1)

        self.readControlFile()
        # dates are output irregular, so we need to build a regular time series to interpolate to
        self.buildTimes()
        if self.control_file_type == 'npt': #turn off, make user input full W2 file instead of trying to build it?
            self.getOutputFileName_NPT() #get the W2 sanctioned output file name convention
        elif self.control_file_type == 'csv':
            self.getOutputFileName_CSV()

    def buildTimes(self):
        '''
        builds two different timeseries to snap irregular data to, because W2 allows for two different output intervals.
        WDO is for qwo and two files (flow and temp).
        :return:
        '''

        self.getInterval()
        self.getW2StartTime()
        self.wdo_jd_dates, self.wdo_dt_dates = self.buildTimesbyInterval(self.tmstrtJDate,
                                                                         self.tmendJDate,
                                                                         self.wdo_interval)

        #Above case is ONLY for QWO files
        self.jd_dates, self.dt_dates = self.buildTimesbyInterval(self.tmstrtJDate,
                                                                 self.tmendJDate,
                                                                 self.tsr_interval)

    def readControlFile(self):
        '''
        Open control file lines and format them. Control file lines are split into "groups", usually based off of a
        header, and then the values. Control files are split into sections based off of spaces in the control file
        :return: set class variables:
                    self.cf_lines
                    self.line_sections
        '''

        self.cf_lines = self.getControlFileLines(self.control_file)
        if self.control_file_type == 'npt':
            self.line_sections = self.formatNPTCFLines(self.cf_lines)
        else: #csv
            self.line_sections = self.formatCSVCFLines(self.cf_lines)

    def getInterval(self):
        '''
        gets the interval from the W2 control file. there are two intervals, one for irregular data and one for the rest.
        irregular one is for qwo and two files (WDO).
        :return: sets tsr interval and wdo interval
        '''

        if self.control_file_type == 'npt':
            tsr_interval = float(self.getNPTControlVariable(self.line_sections, 'TSR FREQ')[0])
            wdo_interval = float(self.getNPTControlVariable(self.line_sections, 'WITH FRE')[0])
        else:
            tsr = self.getCSVControlVariable(self.line_sections, 'TSR')
            tsr_interval = float(tsr[5])
            wdo = self.getCSVControlVariable(self.line_sections, 'WDO')
            wdo_interval = float(wdo[5])
        self.tsr_interval = tsr_interval
        self.wdo_interval = wdo_interval

    def getW2StartTime(self):
        '''
        gets the time window from W2 con file
        :return:
        '''

        if self.control_file_type == 'npt':
            timecon = self.getNPTControlVariable(self.line_sections, 'TIME CON')
            self.tmstrt = float(timecon['TMSTRT'])
            self.tmend = float(timecon['TMEND'])
            self.tmyear = int(timecon['YEAR'])
            startend = WT.JDateToDatetime([self.tmstrt, self.tmend], self.tmyear)
            self.tmstrtJDate = startend[0]
            self.tmendJDate = startend[1]
        else:
            self.tmstrt = float(self.getCSVControlVariable(self.line_sections, 'TMSTRT'))
            self.tmend = float(self.getCSVControlVariable(self.line_sections, 'TMEND'))
            self.tmyear = int(self.getCSVControlVariable(self.line_sections, 'YEAR'))
            startend = WT.JDateToDatetime([self.tmstrt, self.tmend], self.tmyear)
            self.tmstrtJDate = startend[0]
            self.tmendJDate = startend[1]

    def get_tempprofile_layers(self):
        '''
        gets profile layers from the control file
        :return: set class variables:
                    self.layers
        '''

        if self.control_file_type == 'npt':
            self.layers = np.asarray([float(n) for n in self.getNPTControlVariable(self.line_sections, 'TSR LAYE') if n != ''])
            self.segments = np.asarray([int(n) for n in self.getNPTControlVariable(self.line_sections, 'TSR SEG') if n != ''])

        else:
            self.layers = float(self.getCSVControlVariable(self.line_sections, 'TSR LAYE'))
            self.segments = int(self.getCSVControlVariable(self.line_sections, 'TSR SEG'))



    def getOutputFileName_NPT(self):
        '''
        gets the name of output files
        :return:
        '''

        # self.output_file_name = self.getControlVariable(self.line_sections, 'TSR FILE')[0]
        output_file_name = self.getNPTControlVariable(self.line_sections, 'TSR FILE')
        output_file_name = ''.join(output_file_name) #sometimes this goes multi line, but it shouldnt
        # if len(output_file_name) > 0:
        self.output_file_name = output_file_name

    def getOutputFileName_CSV(self):
        '''
        gets the name of output files
        :return:
        '''

        output_file_name = self.getCSVControlVariable(self.line_sections, 'TSR')
        if len(output_file_name) > 0:
            self.output_file_name = output_file_name[3]

    def getControlFileLines(self, control_file):
        '''
        reads control file
        :param input_file: full path to control file
        :return: np array of all lines in control file
        '''

        file_read = open(control_file, 'r')
        file_lines = file_read.readlines()
        file_read.close()
        return np.asarray(file_lines)

    def formatNPTCFLines(self, cf_lines):
        '''
        seperates control file lines into sections, based off of spaces in the file. each section is 8 spaces. Control files are generally
        formatted like:

        NPT FORMAT:

        NWB	 NBR	 IMX	 KMX	 NPROC	 CLOSEC
        1	  4	      83	 135	     1	     ON

        OR

        WD1
        OFF
        52
        644.35
        2
        135

        File then splits both of the sections above into two seperate sections into a list for easier parsing. Sections
        are split based on the spaces between them.

        :param cf_lines: control file lines from self.get_control_file_lines()
        :return: a list of sections

        #scenarios (all of this contained in dict by first header item (sections))
            #1. subitems with all headers the same (minus first): dictionary of lists
            #2. subitems with all headers different (minus first): dictionary of dictionaries
            #3. no subitems with headers all the same: list
            #4. no subitems with different headers: dict
        '''

        sections = {}
        got_headers = False
        sections_contents_template = []
        sections_contents = []
        main_flag = None
        other_headers = []
        for line in cf_lines[10:]: #skip the first ten lines, theyre garb.
            # line = line.strip()
            line = [line[i:i+8] for i in range(0, len(line), 8)] #npt files are spaced 8 chars wide
            if line[-1] == '\n':
                line = line[:-1]
            line = [n.strip() for n in line]

            if (len(line) == 1 and line[0] == '') or (len(line) == 0):
                #store contents and reset
                if main_flag != None:
                    sections[main_flag] = sections_contents
                else:
                    continue
                    # print('Main flag none. Skip.')
                got_headers = False
                sections_contents_template = []
                sections_contents = []
                main_flag = None
                other_headers = []

            else:

                if not got_headers:
                    #this is our header
                    main_flag = line[0]
                    other_headers = line[1:]
                    if len(line) > 1: #if there are more headers
                        other_headers = Counter(line[1:])
                        if max(other_headers.values()) == 1: #every other header is unique
                            # sections_contents_template = {n: {} for n in other_headers}
                            sections_contents_template = {}
                        else:
                            sections_contents_template = []
                    else:
                        sections[main_flag] = []
                    got_headers = True
                else:
                    if line[0] != '':
                        #we have a sub item, and not just a big list, or some features
                        subitem = line[0]
                        subitem_contents = line[1:]
                        sections_contents = sections_contents_template
                        if isinstance(sections_contents, dict):
                            subsections_contents = {n: {} for n in other_headers}
                            for i, key in enumerate(other_headers):
                                try:
                                    subsections_contents[key] = subitem_contents[i]
                                except IndexError:
                                    subsections_contents[key] = ''
                            sections_contents[subitem] = subsections_contents
                        elif isinstance(sections_contents, list):
                            sections_contents += subitem_contents
                    else:
                        subitem_contents = line[1:]
                        sections_contents = sections_contents_template
                        if isinstance(sections_contents, dict):
                            for i, key in enumerate(other_headers):
                                try:
                                    sections_contents[key] = subitem_contents[i]
                                except IndexError:
                                    sections_contents[key] = ''
                        elif isinstance(sections_contents, list):
                            sections_contents += subitem_contents

        return sections

    def formatCSVCFLines(self, cf_lines):
        '''
        seperates control file lines into sections, based off of spaces in the file. Control files are generally
        formatted like:

        CSV FORMAT:

        NWB	 NBR	 IMX	 KMX	 NPROC	 CLOSEC
        1	  4	      83	 135	     1	     ON

        OR

        WD1
        OFF
        52
        644.35
        2
        135

        File then splits both of the sections above into two seperate sections into a list for easier parsing. Sections
        are split based on the spaces between them.

        :param cf_lines: control file lines from self.get_control_file_lines()
        :return: a list of sections
        '''

        sections = []
        small_section = []
        for line in cf_lines:
            if self.control_file_type == 'csv':
                line = line.strip().split(',')
            else:
                line = line.strip().split('   ')

            if len(line) > 0:
                if line[0] == '':
                    line = []

            line = [n.strip() for n in line if n != '']


                # line = ''.join(list(filter((',').__ne__, list(line))))
            if len(line) == 0 and len(small_section) == 0:
                continue
            if len(line) == 0 and len(small_section) != 0:
                #check section here
                if len(small_section) > 2:
                    header = small_section[0]
                    body = []
                    for n in small_section[1:]:
                        if len(n) > 1:
                            body.append(n)
                        else:

                            body.append(n[0])
                    small_section = [header, body]
                if len(small_section) > 1:
                    if len(small_section[0]) > len(small_section[1]):
                        small_section[1] += [''] * (len(small_section[0]) - len(small_section[1]))
                sections.append(small_section)
                small_section = []
            else:
                small_section.append(line)

        return sections

    def getCSVControlVariable(self, lines_sections, variable):
        '''
        Parses the split control file sections from self.format_cf_lines() for a wanted card. Cards usually preface
        headers in the contro file, see docuemntation. For the give example below...

        DLT MAX   DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX
                  3600.00

        DLT FRN     DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF
                   0.900

        If the user wanted the DLT Max value, they would search 'DLT MAX'. If there is only one instance of the flag,
        then return a single array. Else, output a list of arrays and let the user narrow it down from there.

        :param lines_sections: formatted control line sections from self.format_cf_lines()
        :param variable: intended variable card to find
        :param pref_output_type: preferred variable type for output (i.e. np.float, np.str, etc). Converts all values
                                 in output to intended type. If it fails (aka trying to convert a string value to a
                                 float, return strings instead.
        :return: either list of np arrays for multi output, or a single np.array
        '''

        variable_lines_idx = [i for i, line in enumerate(lines_sections) if variable in line[0]]
        outputs = []
        for var_line_idx in variable_lines_idx:
            # for line in lines_sections[var_line_idx].split('\n')[1:]: #skip header
            line = lines_sections[var_line_idx]
            if len(line[0]) != len(line[1]): #for cases of vert stack in csv
                cur_otpt = line[1]
            else:
                idx = np.where(np.asarray(line[0]) == variable)
                cur_otpt = np.asarray(line[1])[idx]
                if len(cur_otpt) > 1:
                    for item in cur_otpt:
                        if item != '':
                            cur_otpt = np.asarray(line[1])[idx][0]
            outputs.append(cur_otpt)

        if len(outputs) > 1:
            return outputs[0][0]
        return outputs[0]

    def getNPTControlVariable(self, lines_sections, variable):
        '''
        Parses the split control file sections from self.format_cf_lines() for a wanted card. Cards usually preface
        headers in the contro file, see docuemntation. For the give example below...

        DLT MAX   DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX
                  3600.00

        DLT FRN     DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF
                   0.900

        If the user wanted the DLT Max value, they would search 'DLT MAX'. If there is only one instance of the flag,
        then return a single array. Else, output a list of arrays and let the user narrow it down from there.

        :param lines_sections: formatted control line sections from self.format_cf_lines()
        :param variable: intended variable card to find
        :param pref_output_type: preferred variable type for output (i.e. np.float, np.str, etc). Converts all values
                                 in output to intended type. If it fails (aka trying to convert a string value to a
                                 float, return strings instead.
        :return: either list of np arrays for multi output, or a single np.array
        '''

        if variable in lines_sections.keys():
            return lines_sections[variable]
        else:
            return None


    def buildTimesbyInterval(self, start_day, end_day, interval):
        '''
        Creates a regular time series. W2 time series are irregular, so we'll create a regular time series for output
        and then interpolate over it. Return a list of jdates (days past the jan 1, starting at 1) and a list of
        datetime values
        :param start_day: start day of the simulation (jdate)
        :param end_day: end day of the simulations (jdate)
        :param interval: the desired output time series interval in minutes, aka 60=Hourly, 15=15MIN (4 per hour)
        :return: list of dates in jdate format and datetime format
        '''

        # turns out jdates can be passed into timedelta (decimals) and it works correctly. Just subtract 1 becuase jdates
        # start at 1
        dt_dates = pd.date_range(start_day,end_day,freq=dt.timedelta(seconds=np.floor(dt.timedelta(interval).total_seconds()))).to_pydatetime()
        jd_dates = np.asarray(WT.DatetimeToJDate(dt_dates))

        return jd_dates, dt_dates

    def readProfileData(self, seg, timesteps, resultsfile=None):
        '''
        Gets the temperature profile values from the output files.
        organizes results into np arrays. Arrays are full of Nan values by default for all possible values, then filled
        in where data applies. Water temps are organized into 2d arrays of dates - layers. This way, a user can index
        one date and get the temperature layers for that time step.
        Output values from W2 model come out in an irregular time series for some reason, so we will take the values,
        then interpolate them and find the interpolated values at the times we are looking for. Hopefully, these will
        still be very close to the values. This is how the pervious output method would put these into DSS.
        Water surface elevations are also needed for calculations of depth and elevation in main plotting script.
        These are the same for all layers at one time step, for valid layers.
        :return: array of water temperatures, elevations and depths
        '''

        if self.control_file_type == 'npt':
            values, elevations, depths, dates = self.readProfileData_NPT(seg, timesteps)
        elif self.control_file_type == 'csv':
            values, elevations, depths, dates = self.readProfileData_CSV(seg, timesteps, resultsfile)

        return values, elevations, depths, dates

    def readProfileData_CSV(self, seg, timesteps, resultsfile=None):
        '''
        output for CSV W2 runs are not always in the spr.csv file. Headers look like this
        Constituent	Julian_day	Depth	Elevation	Seg_111	Elevation	Seg_113
        :param seg:
        :param timesteps:
        :return:
        '''

        if resultsfile == None:
            resultsfile = self.getResultsFile_CSV()
        outputfile = os.path.join(self.run_path, resultsfile)
        if not os.path.exists(outputfile):
            WF.print2stdout(f'Results file {outputfile} does not exist.', debug=self.Report.debug)
            return [], [], [], []
        output = pd.read_csv(outputfile)
        segment_header = f'Seg_{seg}'
        # segment_index = np.where(output.columns.startswith(segment_header))[0] #should only ever be 1
        segment_index = [i for i, col in enumerate(output.columns) if col.startswith(segment_header)] #should only ever be 1
        if len(segment_index) == 0:
            WF.print2stdout(f'ERROR: segment {seg} not found in output file {outputfile}', debug=self.Report.debug)
            return [], [], [], []
        segment_elevation_header = output.columns[segment_index[0]-1]
        segment_value_header = output.columns[segment_index[0]] #stupid segment header can have a space at the end...
        all_jdates = output['Julian_day'].values #csv depths are always the same for all segments output
        all_dtdates = WT.JDateToDatetime(all_jdates, self.starttime.year) #csv depths are always the same for all segments output
        all_depths = output['Depth'].values #csv jdates are always the same for all segments output
        all_values = output[segment_value_header].values
        all_elevations = output[segment_elevation_header].values
        unique_dates = np.asarray(list(set(all_dtdates)))

        if len(unique_dates) == 0:
            WF.print2stdout('No values found in output.', debug=self.Report.debug)
            return [], [], [], []

        if isinstance(timesteps, (list, np.ndarray)):
            select_values = []
            select_elevations = []
            select_depths = []
            select_times = []
            for t, time in enumerate(timesteps):
                timestep = WT.getIdxForTimestamp(unique_dates, time)
                if timestep > -1:#timestep in model
                    indicies = np.where(all_dtdates == unique_dates[timestep])
                    values = all_values[indicies]
                    elevations = all_elevations[indicies]
                    depths = all_depths[indicies]
                    # WSE = elevations[timestep] #Meters #get WSE
                    if not WF.checkData(elevations): #if elevations is bad, skip usually first timestep...
                        select_elevations.append(np.array([]))
                        select_depths.append(np.array([]))
                        select_values.append(np.array([]))
                        select_times.append(time)
                        continue
                    select_elevations.append(elevations[:] * 3.28084)
                    select_depths.append(depths[:] * 3.28084)
                    select_values.append(values[:])
                    select_times.append(time)

                else: #if timestep NOT in model, add empties
                    select_values.append(np.array([])) #find WTs
                    select_elevations.append(np.array([]))
                    select_depths.append(np.array([]))
                    select_times.append(time)

            select_values, select_elevations, select_depths = self.matchProfileLengths(select_values, select_elevations, select_depths)
            return select_values, select_elevations, select_depths, select_times,
        else:
            return [], [], [], sorted(unique_dates)


    def readProfileData_NPT(self, seg, timesteps):
        '''
        Reads profile data from NPT file
        :param seg: segment number
        :param timesteps: timesteps to get data for
        :return: profile data
        '''

        self.get_tempprofile_layers() #get the output layers. out at 2m depths

        wt = []
        WS_Elev = []
        profile_jdates = []

        for i in range(1,len(self.layers)+1):
            # WF.print2stdout('{0} of {1}'.format(i, len(self.layers)+1))
            ofn = '{0}_{1}_seg{2}.{3}'.format(self.output_file_name.split('.')[0],
                                              i,
                                              seg,
                                              self.output_file_name.split('.')[1])
            if self.segments[i-1] != int(seg):
                continue
            ofn_path = os.path.join(self.run_path, ofn)
            if not os.path.exists(ofn_path):
                WF.print2stdout('File {0} not found'.format(ofn_path))
                continue
            headerline=0
            with open(ofn_path) as ofnf:
                for li, line in enumerate(ofnf):
                    if line.lower().startswith('jday'):
                        headerline=li
                        break

            op_file = pd.read_csv(ofn_path, header=headerline, skip_blank_lines=False)
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
                                new_dates.append(jd)
                        else:
                            search_index += 1

                    # build, concatonate, and sort-in rows for the missing JDATE values
                    new_values = np.full((len(new_dates),len(op_file.columns)), np.nan)
                    for i, jd in enumerate(new_dates):
                        new_values[i,0] = jd
                    op_file = pd.concat([op_file, pd.DataFrame(columns=op_file.columns, data=new_values, index=range(1,len(new_dates)+1))])
                    op_file = op_file.sort_values('jday', ignore_index=True)

                wt_vals = op_file['t2(c)']
                elev_vals = op_file['elws(m)']
                wt.append(wt_vals.values)
                WS_Elev.append(elev_vals.values)

        max_len = len(self.jd_dates)
        wt = np.asarray([np.pad(array, (0, max_len - len(array)), mode='constant', constant_values=np.nan) for array in wt]).T
        WS_Elev = np.asarray([np.pad(array, (0, max_len - len(array)), mode='constant', constant_values=np.nan) for array in WS_Elev]).T

        segment_layers = np.asarray([n for i, n in enumerate(self.layers) if self.segments[i] == int(seg)])

        if isinstance(timesteps, (list, np.ndarray)):
            select_wt = []
            elevations = []
            depths = []
            times = []
            for t, time in enumerate(timesteps):
                e = []
                timestep = WT.getIdxForTimestamp(self.dt_dates, time)
                if timestep > -1:#timestep in model
                    try:
                        WSE = WS_Elev[timestep] #Meters #get WSE
                    except IndexError:
                        WSE = []
                    if not WF.checkData(WSE): #if WSE is bad, skip usually first timestep...
                        elevations.append(np.array([]))
                        depths.append(np.array([]))
                        select_wt.append(np.array([]))
                        times.append(time)
                        continue
                    WSE = WSE[np.where(~np.isnan(WSE))][0] #otherwise find valid
                    # WSE_array = np.full((self.layers.shape), WSE)
                    WSE_array = np.full((segment_layers.shape), WSE)
                    e = (WSE_array - segment_layers) * 3.28084
                    e = e[:len(wt[timestep])]
                    select_wt.append(wt[timestep][:]) #find WTs

                else: #if timestep NOT in model, add empties
                    select_wt.append(np.array([])) #find WTs
                    elevations.append(np.array([]))
                    depths.append(np.array([]))
                    times.append(time)
                elevations.append(np.asarray(e)) #then append for timestep
                depths.append((segment_layers * 3.28084)[:len(e)]) #append dpeths
                times.append(time) #get time
            select_wt, elevations, depths = self.matchProfileLengths(select_wt, elevations, depths)

            return select_wt, elevations, depths, np.asarray(times)
        else:
            elevations = ((WS_Elev - segment_layers) * 3.28084)[:len(wt)]
            return wt, elevations, [], self.dt_dates

    def readProfileTopwater(self, seg, timesteps):
        '''
        gets the WSE for each timestep to filter for profile contour plots
        :param seg: segment number for profile
        :param timesteps: list of timesteps, or 'all'
        :return: list of WSE
        '''

        self.get_tempprofile_layers() #get the output layers. out at 2m depths

        WS_Elev = np.full((len(self.layers), len(self.jd_dates)), np.nan)

        for i in range(1,len(self.layers)+1):
            ofn = '{0}_{1}_seg{2}.{3}'.format(self.output_file_name.split('.')[0],
                                              i,
                                              seg,
                                              self.output_file_name.split('.')[1])
            ofn_path = os.path.join(self.run_path, ofn)
            if not os.path.exists(ofn_path):
                WF.print2stdout('File {0} not found'.format(ofn_path))
                continue
            headerline=0
            with open(ofn_path) as ofnf:
                for li, line in enumerate(ofnf):
                    if line.lower().startswith('jday'):
                        headerline=li
                        break

            op_file = pd.read_csv(ofn_path, header=headerline, skip_blank_lines=False)
            op_file.columns = op_file.columns.str.lower()
            if len(op_file['jday']) > 1:
                Elev_interp = interp1d(op_file['jday'], op_file['elws(m)'])
                jdate_minmask = np.where(min(op_file['jday']) <= self.jd_dates)
                jdate_maxmask = np.where(max(op_file['jday']) >= self.jd_dates)
                jdate_msk = np.intersect1d(jdate_maxmask, jdate_minmask)
                wsElev_ts_Vals = np.full(len(self.jd_dates), np.nan)
                wsElev_ts_Vals[jdate_msk] = Elev_interp(self.jd_dates[jdate_msk])
                WS_Elev[i-1] = wsElev_ts_Vals

        WS_Elev = np.asarray(WS_Elev).T

        if isinstance(timesteps, (list, np.ndarray)):

            WSE_out = []
            for t, time in enumerate(timesteps):
                timestep = WT.getIdxForTimestamp(self.dt_dates, time)
                if timestep > -1:#timestep in model
                    WSE = WS_Elev[timestep] #Meters #get WSE
                    if not WF.checkData(WSE): #if WSE is bad, skip usually first timestep...
                        WSE_out.append(np.nan)
                        continue
                    WSE = WSE[np.where(~np.isnan(WSE))][0] #otherwise find valid
                    WSE_out.append(WSE)

                else: #if timestep NOT in model, add empties
                    WSE_out.append(np.nan)

            return WSE_out
        else:
            return WS_Elev[:,0] * 3.28084

    def readStructuredTimeSeries(self, output_file_name, structure_nums, skiprows=2):
        """
        output files usually have header with several repeat headers for each structure
         Branch:           1  # of structures:          23  outlet temperatures
        JDAY      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL
        structures are determined by number of each header
        :param output_file_name: name of output file in run path
        :param structure_nums: number values of structures we want to output
        :param skiprows: how many header rows to skip
        :return: dates, values
        """

        ofn_path = os.path.join(self.run_path, output_file_name)
        if not os.path.exists(ofn_path):
            WF.print2stdout(f'File {ofn_path} not found!')
            return [], []

        structure_nums = [int(n) for n in structure_nums]
        values = {}

        with open(ofn_path, 'r') as o:
            for i, line in enumerate(o):
                if i == skiprows-1:
                    headers = line.strip().lower().replace(',','').split()[1:] #skipjdate..
                    break
        header_count = Counter(headers)
        headers = list(set(headers))

        stsf = pd.read_csv(ofn_path, header=skiprows-1, delim_whitespace=True)
        stsf.columns = stsf.columns.str.lower()
        stsf.columns = [n.replace(',','') for n in stsf.columns]
        for structure_num in structure_nums:
            if structure_num < 0:
                structure_num = min(header_count.values()) + structure_num+1 #reverse index the fun way, use min len incase doesnt match for some reason
            if structure_num not in values.keys():
                values[structure_num] = {}
            for header in headers:
                if structure_num == 1:
                    hname = header
                else:
                    hname = header+'.{0}'.format(structure_num-1)
                vals = np.asarray([float(str(n).replace(',','')) for n in stsf[hname].tolist()])
                values[structure_num][header.lower()] = vals

        dates = stsf['jday'].tolist()
        dates = np.asarray([float(str(n).replace(',', '')) for n in dates])

        return dates, values

    def filterByParameter(self, values, line_info):
        '''
        W2 results files have multiple parameters in a single file, so we can return many parameters
        this grabs the parameter defined in the line
        :param values: dictionary of lists of values
        :param line_info: line settings dictionary containing values
        :return: values list, parameter from line settings
        '''

        headerparam = {'flow': 'q(m3/s)',
                       'temperature': 't(c)',
                       'waterlevel': 'elevcl'}

        if 'parameter' not in line_info.keys():
            WF.print2stdout('Parameter not specified.', debug=self.Report.debug)
            WF.print2stdout('Line Info:', line_info, debug=self.Report.debug)
            return values, ''
        new_values = {}
        target_header = headerparam[line_info['parameter']]
        for key in values.keys():
            new_values[key] = values[key][target_header]
        return new_values, line_info['parameter']

    def matchProfileLengths(self, select_val, elevations, depths):
        '''
        Matches lengths for values and elevs. Sometimes values not output at certain elevs
        :param select_val: selected values
        :param elevations: selected elevations
        :param depths: selected depths
        :return: trimmed values, elevations, depths
        '''

        len_val = len(select_val)
        len_elev = len(elevations)
        len_depth = len(depths)
        min_len = min((len_val, len_elev, len_depth))
        return select_val[:min_len], elevations[:min_len], depths[:min_len]

    def readTimeSeries(self, output_file_name, column=1, skiprows=3, **kwargs):
        '''
        get the output time series for W2 at a specified location. Like the temperature profiles, output freq is
        variable so we'll interpolate over a regular time series
        :param output_file_name: full path to the output file.
        :param targetfieldidx: Number of column to grab data from (0 index). Usually date, value.
        :return: np.array of dates, and the values
        '''

        try:
            column = int(column)
        except:
            pass

        try:
            skiprows = int(skiprows)
        except:
            pass


        ofn_path = os.path.join(self.run_path, output_file_name)

        if not os.path.exists(ofn_path):
            WF.print2stdout('Data File not found!', ofn_path)
            return [], []

        if output_file_name.lower().startswith(('qwo', 'two')):
            jd_dates = self.wdo_jd_dates
            dt_dates = self.wdo_dt_dates
        else:
            jd_dates = self.jd_dates
            dt_dates = self.dt_dates

        if isinstance(column, str):
            header = linecache.getline(ofn_path, int(skiprows)).strip().replace(' ','').lower().split(',') #1 indexed, for some reason
            cidx = np.where(np.asarray(header) == column.replace(' ','').lower())[0]
            if len(cidx) > 0:
                column = cidx[0]
            else:
                WF.print2stdout(f'Header {column} not found in file', debug=self.Report.debug)
                return [], []

        dates = []
        values = []
        with open(ofn_path, 'r') as o:
            for i, line in enumerate(o):
                if i >= int(skiprows):
                    sline = line.split(',')
                    if len(sline) == 1: #not csv TODO: figure out this but better..
                        sline = line.split()
                    dates.append(float(sline[0].strip()))
                    # if isinstance(column, int):
                    values.append(float(sline[column].strip()))
                    # elif isinstance(column, str):
                        # header = linecache.getline(ofn_path, int(skiprows)).strip().lower().split() #1 indexed, for some reason
                        # cidx = np.where(np.asarray(header) == column.lower())[0]
                        # values.append(float(sline[cidx].strip()))


        if len(dt_dates) > len(values):
            dt_dates = dt_dates[:len(values)] #if the interval is off and its shifted, you better believe theres a missing value here. for fun.

        if len(dt_dates) < len(values): #in the event data file has full year of output and the time window changes
            values = values[:len(dt_dates)]

        return dt_dates, np.asarray(values)

    def readSegment(self, filename, parameter):
        '''
        Temporary until we figure out how to do W2 contours
        :param filename: file for output
        :param parameter: parameter to get data for
        :return:
        '''

        read_param = self.getParameterFileStr(parameter)
        if read_param == None:
            return [], [], []
        ofn_path = os.path.join(self.run_path, filename)

        output_values = np.array([])
        segments = []
        dates = []
        checkForVar = False
        record_vals = False
        gotvalues = True
        with open(ofn_path, 'r') as otf:
            for line in otf:
                if checkForVar and line.strip() != '':
                    if parameter in line.lower():
                        recordVals = True
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
                    checkForVar = True

        # otf.split('\n')

    def getParameterFileStr(self, parameter):
        '''
        gets parameter file name based on the input name
        :param parameter: desired parameter
        :return: formatted internal name
        '''

        #input:output
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

        if parameter.lower() not in fileparams:
            WF.print2stdout('Parameter {0} not in acceptable parameters.'.format(parameter), debug=self.Report.debug)
            return None
        else:
            return fileparams[parameter.lower()]

    def getResultsFile_CSV(self):
        '''
        Finds the output file name in CSV files for profiles
        Ryan Miles says this is fixed and can be calculated, so here we are.
        Reference email form 10/11/2022 at 10:35
        :return: results file name
        '''

        rootrow = 768

        structureoffsetline = [int(i) for i in self.cf_lines[135].strip().split(',') if i]
        structureoffset = (max(5, max(structureoffsetline))) * 6

        constituentsoffsetline = [int(i) for i in self.cf_lines[21].strip().split(',') if i]
        #Offset = NGC + NSS + NAL + (NBOD * 3) + 32 + NZP + 4
        #NGC, NSS, NAL, NEP, NBOD, NMC, NZP
        NGC = constituentsoffsetline[0]
        NSS = constituentsoffsetline[1]
        NAL = constituentsoffsetline[2]
        NEP = constituentsoffsetline[3]
        NBOD = constituentsoffsetline[4]
        NMC = constituentsoffsetline[5]
        NZP = constituentsoffsetline[6]

        constituentsoffset = NGC + NSS + NAL + (NBOD * 3) + 32 + NZP + 4

        #Offset = Max(5, NEP) * 3
        epiphytonoffset = (max(5, NEP)) * 3

        #Offset = Max(5, NAL) + Max(5, NZP)
        zooplanktonoffset = max(5, NAL) + max(5, NZP)

        #Offset = Max(5, NMC) * 3
        macrophyteoffset = max(5, NMC) * 3

        totaloffset = structureoffset + constituentsoffset + epiphytonoffset + zooplanktonoffset + macrophyteoffset
        inputfile_idx = rootrow + totaloffset
        results_file = self.cf_lines[inputfile_idx-1].strip().split()[0] #excel is 1 idx..
        return results_file