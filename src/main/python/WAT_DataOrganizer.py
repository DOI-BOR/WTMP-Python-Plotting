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
import numpy as np
import pandas as pd
import traceback
import pickle
import itertools
from collections import Counter

import WAT_Functions as WF
import WAT_Reader as WDR
import WAT_Time as WT
import WAT_Reader as WR
import WAT_ResSim_Results as WRSS
import WAT_Constants as WC

constants = WC.WAT_Constants()

class DataOrganizer(object):

    def __init__(self, Report):
        '''
        class supports getting data from file sources and orgainzing data into memory dictionary for easy recall
        :param Report: self from main Report Generator script
        '''

        self.Report = Report #self from main
        self.intializeMemory()

    def intializeMemory(self):
        '''
        initializes the main dictionary to store data
        '''

        self.Memory = {}

    def buildMemoryKey(self, data_info):
        '''
        creates uniform name for csv log output for data
        determines how to build the file name from the input type
        :param data_info: information about line
        :return: name for memory key, or null if can't be determined
        '''

        # very_special_flags = f'{self.Report.SimulationName.replace(" ", "").replace(":", "")}_{self.Report.baseSimulationName.replace(" ", "").replace(":", "")}'
        very_special_flags = WF.sanitizeText(self.Report.SimulationName)

        if 'dss_path' in data_info.keys(): #Get data from DSS record
            if 'dss_filename' in data_info.keys():
                outname = WF.sanitizeText(os.path.basename(data_info['dss_filename'])[:-4])
                dssnamesplit = data_info['dss_path'].split('/')
                dssname_pick = WF.sanitizeText(f"{dssnamesplit[1]}_{dssnamesplit[2]}_{dssnamesplit[3]}_{dssnamesplit[5]}_{dssnamesplit[-2]}")
                outname = very_special_flags + '_' + outname + '_' + dssname_pick

        elif 'w2_file' in data_info.keys():
            if 'structurenumbers' in data_info.keys():
                outname = WF.sanitizeText(os.path.basename(data_info['w2_file']).split('.')[0])
                if isinstance(data_info['structurenumbers'], dict):
                    structure_nums = [data_info['structurenumbers']['structurenumber']]
                elif isinstance(data_info['structurenumbers'], str):
                    structure_nums = [data_info['structurenumbers']]
                elif isinstance(data_info['structurenumbers'], (list, np.ndarray)):
                    structure_nums = data_info['structurenumbers']
                else:
                    structure_nums = ''
                outname += '_Struct_' + '_'.join(structure_nums) + f'_{very_special_flags}'
            else:
                if 'column' in data_info.keys():
                    very_special_flags += f'_Colf{data_info["column"].replace(" ", "")}'
                outname = f"{os.path.basename(data_info['w2_file']).split('.')[0]}_{very_special_flags}"

        elif 'h5file' in data_info.keys():
            h5name = WF.sanitizeText(os.path.basename(data_info['h5file']).split('.h5')[0] + 'h5')
            if 'easting' in data_info.keys() and 'northing' in data_info.keys():
                outname = 'externalh5_{0}_{1}_{2}_{3}_{4}'.format(h5name, WF.sanitizeText(data_info['parameter']), data_info['easting'], data_info['northing'], very_special_flags)
            elif 'ressimname' in data_info.keys():
                outname = 'externalh5_{0}_{1}_{2}_{3}'.format(h5name, WF.sanitizeText(data_info['parameter']), WF.sanitizeText(data_info['ressimresname']), very_special_flags)

        elif 'easting' in data_info.keys() and 'northing' in data_info.keys():
            outname = '{0}_{1}_{2}_{3}'.format(WF.sanitizeText(data_info['parameter']), data_info['easting'], data_info['northing'], very_special_flags)

        elif 'filename' in data_info.keys(): #Get data from Observed Profile
            outname = WF.sanitizeText(os.path.basename(data_info['filename']).split('.')[0].replace(' ', '_')) + f'_{very_special_flags}'

        elif 'w2_segment' in data_info.keys():
            outname = 'W2_{0}_{1}_profile'.format(self.Report.ModelAlt.output_file_name.split('.')[0], data_info['w2_segment']) + f'_{very_special_flags}'

        elif 'ressimresname' in data_info.keys():
            outname = '{0}_{1}_{2}_{3}'.format(WF.sanitizeText(os.path.basename(self.Report.ModelAlt.h5fname).split('.')[0] +'_h5'),
                                               WF.sanitizeText(data_info['parameter']), WF.sanitizeText(data_info['ressimresname']), very_special_flags)
            if 'target' in data_info.keys():
                outname += '_trgt'
                if 'parameter' in data_info['target'].keys():
                    outname += data_info['target']['parameter'][:4]
                if 'value' in data_info['target'].keys():
                    outname += data_info['target']['value']
        else:
            outname = 'NULL'

        return outname[:150]

    #################################################################
    #TimeSeries Functions
    #################################################################

    def updateTimeSeriesDataDictionary(self, data, line_settings, line):
        '''
        organizes line information and places it into a data dictionary
        :param data: dictionary containing line data
        :param line_settings: dictionary containing line settings for all lines
        :param line: dictionary containing line settings for current line
        :return: updated data dictionary
        '''

        dates, values, metadata = self.getTimeSeries(line, makecopy=False) #TODO: update
        datacheck = False
        if WF.checkData(values):
            datacheck = True
            flag = line['flag']
            if flag in line_settings.keys() or flag in data.keys():
                count = 1
                newflag = flag + '_{0}'.format(count)
                while newflag in data.keys():
                    count += 1
                    newflag = flag + '_{0}'.format(count)
                WF.print2stdout(f'The current flag was {flag}', debug=self.Report.debug)
                flag = newflag
                WF.print2stdout(f'The new flag is {newflag}', debug=self.Report.debug)
            datamem_key = self.buildMemoryKey(line)
            if 'units' in line.keys() and metadata['units'] == None:
                metadata['units'] = line['units']

            line_settings[flag] = {'logoutputfilename': datamem_key}

            data[flag] = {'values': values,
                          'dates': dates}

            #add flags and settings to linesettings..
            line_settings[flag].update(metadata)
            line_settings[flag].update(line)

        return data, line_settings, datacheck

    def getProfileWSE(self, settings, onflag='lines'):
        '''
        gets the Water surface elevation from time series if <wse> flag in line
        WSE is named <flag it was under> + _wse for easy coordination
        :param settings: dictionary containing lines
        :param onflag: optional flag if <lines> is not the main datasource (i.e., datapath)
        :return: new data dictionary with WSE data
        '''

        wse_data = {}
        for dataobject in settings[onflag]:
            if 'wse' in dataobject.keys():
                dates, values, metadata = self.getTimeSeries(dataobject['wse'], makecopy=False)
                datamem_key = self.buildMemoryKey(dataobject['wse'])
                new_key = dataobject['flag'] + '_wse'
                wse_data[new_key] = {'elevations': values,
                                      'dates': dates,
                                      'logoutputfilename': datamem_key}
                wse_data[new_key].update(metadata)

        return wse_data

    def getMembers(self, object_settings, data_settings):
        '''
        Gets members to use from settings
        :param object_settings: dictionary of settings
        :param data_settings: dictionary of settings describing the data
        :return: list of members to use
        '''

        if self.Report.memberiteration: #if its a forecast iteration, grab the current iteration
            members = [self.Report.member]
        elif 'members' in object_settings.keys(): #if user defined, use the user defined ones
            members = [WF.formatMembers(n) for n in object_settings['members']]
        elif self.Report.reportType == 'forecast': #use the forecasts defined
            members = self.Report.allMembers
        else: #otherwise, we just use everything that we have (if multi datasets, get the overlapping
            members = []
            for i, ds in enumerate(data_settings.keys()):
                if i == 0:
                    members = data_settings[ds]['members']
                else:
                    members = np.intersect1d(data_settings[ds]['members'], members)
        return members

    def filterTimeSeries(self, data, line_settings):
        '''
        filters data read from W2 results files to a target elevation
        :param data: dictionary of data
        :param line_settings: dictionary containing settings for all lines
        :return: modified data dictionary
        '''

        for d in data.keys():
            if 'target_elevation' in line_settings[d].keys():
                target_elevation = float(line_settings[d]['target_elevation'])
                values = data[d]['values']
                if isinstance(values, dict): #w2 results kept in dict with elev
                    if self.Report.reportType == 'forecast': #has to be from DSS
                        if 'elevation' in line_settings[d].keys():
                            for member in values.keys():
                                member_values = values[member]
                                if 'flag' not in line_settings[d]['elevation'].keys():
                                    line_settings[d]['elevation']['flag'] = line_settings[d]['flag'] + '_elev'
                                    line_settings[d]['elevation']['members'] = [member]
                                    elev_times, elev_values, elev_metadata = self.getTimeSeries(line_settings[d]['elevation'])
                                    targelev_failed = np.where(elev_values[member] != target_elevation)
                                    if len(elev_times) == len(data[d]['dates']):
                                        member_values[targelev_failed] = 0.
                                        data[d]['values'][member] = member_values
                                    else:
                                        WF.print2stdout(f'Values and Elevations in {d} member {member} different. Equalizing.',
                                                        debug=self.Report.debug)
                                        mainvalues, elev_data = WF.matchData(
                                                                        {'dates': data[d]['dates'], 'values': member_values},
                                                                        {'dates': elev_times, 'values': elev_values[member]})
                                        targelev_failed = np.where(elev_data['values'] != target_elevation)
                                        mainvalues['values'][targelev_failed] = 0.
                                        data[d]['values'][member] = mainvalues['values']
                                        data[d]['dates'] = mainvalues['dates']
                    else:
                        for sn in values.keys():
                            targelev_failed = np.where(values[sn]['elevcl'] != target_elevation)
                            data[d]['values'][sn]['q(m3/s)'][targelev_failed] = 0.
                elif isinstance(values, (list, np.ndarray)):
                    if 'elevation' in line_settings[d].keys():
                        if 'flag' not in line_settings[d]['elevation'].keys():
                            line_settings[d]['elevation']['flag'] = line_settings[d]['flag']+'_elev'
                            elev_times, elev_values, elev_metadata = self.getTimeSeries(line_settings[d]['elevation'])
                            targelev_failed = np.where(elev_values != target_elevation)
                            if len(elev_times) == len(data[d]['values']):
                                data[d]['values'][targelev_failed] = 0.
                            else:
                                WF.print2stdout(f'Values and Elevations in {d} different. Equalizing.', debug=self.Report.debug)
                                mainvalues, elev_data = WF.matchData({'dates': data[d]['dates'], 'values': data[d]['values']},
                                                                     {'dates': elev_times, 'values': elev_values})
                                targelev_failed = np.where(elev_data['values'] != target_elevation)
                                mainvalues['values'][targelev_failed] = 0.
                                data[d]['values'] = mainvalues['values']
                                data[d]['dates'] = mainvalues['dates']

            if 'filters' in line_settings[d].keys():
                for filter in line_settings[d]['filters']:
                    if 'value' not in filter.keys():
                        WF.print2stdout('Value not defined in filter. Not using filter.', debug=self.Report.debug)
                    else:
                        value = float(filter['value'])
                    use_filter_ts = False
                    if np.any([n.lower() in ['w2_file', 'dss_path', 'easting', 'h5file', 'ressimresname'] for n in filter.keys()]):
                        use_filter_ts = True
                        filter_times, filter_values, filter_metadata = self.getTimeSeries(filter)
                    if use_filter_ts:
                        data_to_filter = filter_values
                    else:
                        data_to_filter = data[d]['values']

                    if 'when' in filter.keys():
                        if filter['when'].lower() == 'under':
                            filtermask = np.where(data_to_filter < value)
                        elif filter['when'].lower() == 'over':
                            filtermask = np.where(data_to_filter > value)
                        elif filter['when'].lower() == 'equals':
                            filtermask = np.where(data_to_filter == value)
                    else:
                        WF.print2stdout('When condition not set in filter. Assuming equals.', debug=self.Report.debug)
                        filtermask = np.where(data_to_filter == value)
                    try:
                        data[d]['values'][filtermask] = np.nan
                    except IndexError:
                        WF.print2stdout('Filter and Data Index not equal. Not Filtering data.', debug=self.Report.debug)
                        WF.print2stdout('Confirm that Data and Filter are on the same timeseries interval', debug=self.Report.debug)

            if 'xlims' in line_settings[d].keys():
                xlims = line_settings[d]['xlims']
                data[d]['dates'], data[d]['values'] = WF.applyXLimits(self.Report, data[d]['dates'], data[d]['values'], xlims)

        return data

    def getTimeSeriesDataDictionary(self, settings):
        '''
        Gets profile line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        line_settings = {}
        if 'lines' in settings.keys():
            for line in settings['lines']:
                numtimesused = 0

                if self.Report.memberiteration:
                    # if 'members' not in line.keys():
                    line['members'] = [self.Report.member]
                    line['allmembers'] = self.Report.allMembers

                if 'flag' not in line.keys():
                    WF.print2stdout('Flag not set for line (Computed/Observed/etc)', debug=self.Report.debug)
                    WF.print2stdout('Not plotting Line:', line, debug=self.Report.debug)
                    continue

                elif line['flag'].lower() == 'computed':
                    for ID in self.Report.accepted_IDs:
                        curline = pickle.loads(pickle.dumps(line, -1))
                        curline = self.Report.configureSettingsForID(ID, curline)
                        if not self.Report.checkModelType(curline):
                            continue
                        curline['numtimesused'] = numtimesused
                        data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, curline)
                        if success:
                            numtimesused += 1
                else:
                    if self.Report.reportType == 'forecast':
                        line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')

                    else:
                        if self.Report.currentlyloadedID != self.Report.base_id: #for comparison plotting mostly
                            line = self.Report.configureSettingsForID(self.Report.base_id, line)
                        else:
                            line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')
                    line['numtimesused'] = numtimesused
                    if not self.Report.checkModelType(line):
                        continue
                    data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, line)
                    if success:
                        numtimesused += 1

        return data, line_settings

    def checkForIdenticalMembers(self, data, metadata):
        '''
        checks if all datasets are the same for all members
        :param data: dictionary containing data
        :param settings: dictionary containing settings
        :return: boolean
        '''

        metadata['identicalmembers'] = []
        for member in metadata['members']:
            membergroup = []
            if not any([member in n for n in metadata['identicalmembers']]): #check to see if we've already found a match
                main_member_data = WF.ignoreNans(data[member])
                for othermember in metadata['members']: #check all other members in the list
                    if othermember != member:
                        if np.all(main_member_data == WF.ignoreNans(data[othermember])):
                            if member not in membergroup:
                                membergroup.append(member)
                            membergroup.append(othermember)
            if len(membergroup) > 0:
                metadata['identicalmembers'].append(membergroup)
        return metadata

    def checkForDuplicateObject(self, settings, member):
        '''
        checks for duplicate plot settings in a list of line settings
        :param line_settings: list of  settings
        :return: boolean
        '''

        fully_duplicate = True
        if len(settings) == 0:
            return False
        for name, setting in settings.items():
            if 'identicalmembers' in setting:
                if not any([member in n for n in setting['identicalmembers']]):
                    fully_duplicate = False
                    break
        return fully_duplicate

    def checkForLowestMember(self, line_settings, member, linekey=None):
        '''
        checks for the lowest member in a list of line settings
        :param line_settings: list of line settings
        :return: boolean
        '''
        other_members = self.getOtherMembers(line_settings, member, linekey)
        if member < min(other_members):
            return True
        return False

    def getOtherMembers(self, line_settings, member, linekey=None):
        if linekey == None:
            linekey = list(line_settings.keys())[0]  # if we are here, then all the lines have the duplicate member group
        line_members = line_settings[linekey]['identicalmembers']
        other_members = []
        for membergroup in line_members:
            if member in membergroup:
                other_members = [n for n in membergroup if n != member]
        return other_members

    def getStraightLineValue(self, settings):
        '''
        reads settings to get value for straight lines on plots, either by getting value at timestamp or configuring
        timestamp
        :param settings: dictionary of settings for plots
        :return: dictionary with settings for straight lines
        '''

        straightlines = {}
        types_of_straightlines = ['hlines', 'vlines']
        for tosl in types_of_straightlines:
            if tosl in settings.keys():
                straightlines[tosl] = {}
                for line in settings[tosl]:
                    if 'value' in line.keys(): #if defined single value for all plots
                        value = float(line['value'])
                        const_key = f'constant_{value}'
                        if 'timestamps' in settings.keys():
                            values = [value] * len(settings['timestamps'])
                        else:
                            values = [value]
                        straightlines[tosl][const_key] = {'values': values,
                                                          'numtimesused': 0}
                        for key in line.keys():
                            if key not in straightlines[tosl][const_key].keys():
                                straightlines[tosl][const_key][key] = line[key]
                        if 'units' not in straightlines[tosl][const_key].keys():
                            straightlines[tosl][const_key]['units'] = None
                    else:
                        timeserieslines, timeserieslinesettings = self.getTimeSeriesDataDictionary({'lines': [line]})
                        for timeserieslinekey in timeserieslines.keys():
                            values = timeserieslines[timeserieslinekey]['values']
                            dates = timeserieslines[timeserieslinekey]['dates']
                            if 'timestamps' in settings.keys():
                                idx = WR.getClosestTime(settings['timestamps'], dates)
                                v_idx = []
                                for id in idx:
                                    if id > len(values):
                                        v_idx.append(np.nan)
                                    else:
                                        v_idx.append(values[id])
                                straightlines[tosl][timeserieslinekey] = {'values': v_idx}
                            for key in timeserieslinesettings[timeserieslinekey].keys():
                                if key not in straightlines[tosl][timeserieslinekey].keys():
                                    straightlines[tosl][timeserieslinekey][key] = timeserieslinesettings[timeserieslinekey][key]

        return straightlines

    def getTimeSeries(self, Line_info, makecopy=True):
        '''
        gets time series data from defined sources
        :param Line_info: dictionary of line setttings containing datasources
        :param makecopy: flag that determines if the data is grabbed or just copied
        :return: dates, values, units
        #TODO: move memory grab outside of functions so only called once
        #TODO: check intervals when grabbing from memory
        '''

        metadata = {'collection': False,
                    'frommemory': False,
                    'partialmemory': False
                    }

        if 'dss_path' in Line_info.keys(): #Get data from DSS record
            if 'dss_filename' not in Line_info.keys():
                WF.print2stdout('DSS_Filename not set for Line: {0}'.format(Line_info), debug=self.Report.debug)
                return np.array([]), np.array([]), metadata
            else:
                datamem_key = self.buildMemoryKey(Line_info)

                if Line_info['dss_path'].split('/')[6].startswith('*|'):

                    metadata['collection'] = True
                    metadata['allmembers'] = False
                    if 'members' in Line_info.keys():
                        wanted_members = Line_info['members']
                    elif self.Report.reportType == 'forecast': #uses the ones defined in the forecast
                        wanted_members = self.Report.allMembers
                    else: #otherwise, grab ALL in the dssfile
                        wanted_members = 'all'
                        metadata['allmembers'] = True
                    #grab all the members on the first go, we will be using the data
                    #then we can keep track of duplicates. Save to memory, but only return the wanted ones
                    metadata['members'] = self.Report.allMembers #keep track of the original series, as this can change

                if datamem_key in self.Memory.keys():
                    WF.print2stdout('Reading {0} from memory'.format(datamem_key), debug=self.Report.debug) #noisy
                    if makecopy:
                        datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                    else:
                        datamementry = self.Memory[datamem_key]
                    times = datamementry['times']
                    values = datamementry['values']
                    metadata = datamementry['metadata']

                    metadata['frommemory'] = True #did we get data from memory
                    members_to_grab = [] #reset, but we still know our members
                    if metadata['collection']:
                        if not metadata['allmembers']: #check if we've ever grabbed them all. if we did, no need to go back
                            for member in wanted_members:
                                # if member not in metadata['members']:
                                if member not in datamementry['metadata']['members']:
                                    members_to_grab.append(member)
                                    metadata['frommemory'] = False
                                    metadata['partialmemory'] = True
                                else:
                                    metadata['partialmemory'] = True

                        if len(members_to_grab) > 0:
                            WF.print2stdout(f'Not all members in memory. Getting remaining: {members_to_grab}', debug=self.Report.debug)

                if not metadata['frommemory']:
                    if metadata['collection']:
                        if metadata['partialmemory']: #if we've only grabbed some of them...
                            coll_times, coll_values, units, coll_members = WDR.readCollectionsDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                                                                              members_to_grab, self.Report.StartTime,
                                                                                                              self.Report.EndTime, self.Report.debug)

                            values.update(coll_values)
                            members = list(set(metadata['members'] + coll_members))
                        else:
                            times, values, units, members = WDR.readCollectionsDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                                                          metadata['members'], self.Report.StartTime,
                                                                                          self.Report.EndTime, self.Report.debug)
                        metadata['members'] = members #todo: look at this closer

                        #grab the wanted guys
                        # values = [{key: values[key]} for key in wanted_members] #this is done below, but this looks cool
                    else:
                        times, values, units = WDR.readDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                               self.Report.StartTime, self.Report.EndTime,
                                                               self.Report.debug)
                    metadata['units'] = units
                    self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                                 'values': pickle.loads(pickle.dumps(values, -1)),
                                                 'metadata': pickle.loads(pickle.dumps(metadata, -1))}

                if np.any(values == None):
                    return np.array([]), np.array([]), metadata
                elif len(values) == 0:
                    return np.array([]), np.array([]), metadata

        elif 'w2_file' in Line_info.keys():
            if self.Report.program.lower() != 'cequalw2':
                return np.array([]), np.array([]), None
            datamem_key = self.buildMemoryKey(Line_info)
            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                metadata = datamementry['metadata']
                metadata['frommemory'] = True

            if not metadata['frommemory']:
                if 'structurenumbers' in Line_info.keys():
                    # Ryan Miles: yeah looks like it's str_brX.npt, and X is 1-# of branches (which is defined in the control file)
                    times, values = self.Report.ModelAlt.readStructuredTimeSeries(Line_info['w2_file'], Line_info['structurenumbers'])
                else:
                    times, values = self.Report.ModelAlt.readTimeSeries(Line_info['w2_file'], **Line_info)

                if 'units' in Line_info.keys():
                    metadata['units'] = Line_info['units']
                elif 'parameter' in Line_info.keys():
                    plotunits = constants.units[Line_info['parameter'].lower()]
                    metadata['units'] = plotunits['metric']
                else:
                    metadata['units'] = None

                self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                            'values': pickle.loads(pickle.dumps(values, -1)),
                                            'metadata': pickle.loads(pickle.dumps(metadata, -1))}

        elif 'h5file' in Line_info.keys() and 'easting' in Line_info.keys() and 'northing' in Line_info.keys():
            datamem_key = self.buildMemoryKey(Line_info)
            if 'subdomain' in Line_info.keys():
                subdomain = Line_info['subdomain']
            else:
                subdomain = None
            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                metadata = datamementry['metadata']
                metadata['frommemory'] = True

            if not metadata['frommemory']:
                filename = Line_info['h5file']
                if not os.path.exists(filename):
                    WF.print2stdout('ERROR: H5 file does not exist:', filename, debug=self.Report.debug)
                    return [], [], None
                externalResSim = WRSS.ResSim_Results('', '', '', '', self.Report, external=True)
                externalResSim.openH5File(filename)
                externalResSim.load_time() #load time vars from h5
                externalResSim.loadSubdomains()
                times, values, units = externalResSim.readTimeSeries(Line_info['parameter'],
                                                              float(Line_info['easting']),
                                                              float(Line_info['northing']),
                                                              subdomain=subdomain)
                if 'units' in Line_info.keys():
                    metadata['units'] = Line_info['units']
                else:
                    metadata['units'] = units

                self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                            'values': pickle.loads(pickle.dumps(values, -1)),
                                            'metadata': pickle.loads(pickle.dumps(metadata, -1))}

        elif 'easting' in Line_info.keys() and 'northing' in Line_info.keys():
            datamem_key = self.buildMemoryKey(Line_info)
            if 'subdomain' in Line_info.keys():
                subdomain = Line_info['subdomain']
            else:
                subdomain = None
            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                metadata = datamementry['metadata']
                metadata['frommemory'] = True

            if not metadata['frommemory']:
                times, values, units = self.Report.ModelAlt.readTimeSeries(Line_info['parameter'],
                                                                    float(Line_info['easting']),
                                                                    float(Line_info['northing']),
                                                                    subdomain=subdomain)
                if 'units' in Line_info.keys():
                    metadata['units'] = Line_info['units']
                else:
                    metadata['units'] = units

                self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                             'values': pickle.loads(pickle.dumps(values, -1)),
                                             'metadata': pickle.loads(pickle.dumps(metadata, -1))}

        elif "ressimresname" in Line_info.keys():
            datamem_key = self.buildMemoryKey(Line_info)
            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                metadata = datamementry['metadata']
                metadata['frommemory'] = True

            if not metadata['frommemory']:
                times = []
                values = []
                if self.Report.program.lower() != 'ressim':
                    WF.print2stdout('Incorrect model type for line using ResSimResName', debug=self.Report.debug)
                    return [], [], metadata

                if 'target' in Line_info.keys():
                    if 'parameter' not in Line_info.keys():
                        WF.print2stdout('No parameter for profile target timeseries.', debug=self.Report.debug)
                        WF.print2stdout('Assuming output is elevation.', debug=self.Report.debug)
                        metadata['parameter'] = 'elevation'
                    else:
                        metadata['parameter'] = Line_info['parameter']
                    times, values, units = self.Report.ModelAlt.getProfileTargetTimeseries(Line_info['ressimresname'],
                                                                                    Line_info['parameter'],
                                                                                    metadata['target'])
                    metadata['type'] = 'target'
                    metadata['units'] = units

                elif 'fwa' in Line_info.keys():
                    if Line_info['fwa'].lower() == 'true': #not sure what to do otherwise..
                        if 'parameter' not in Line_info.keys():
                            WF.print2stdout('No parameter for FWA reservoir timeseries.', debug=self.Report.debug)
                            WF.print2stdout('Assuming output is temperature.', debug=self.Report.debug)
                            metadata['parameter'] = 'temperature'
                        else:
                            metadata['parameter'] = Line_info['parameter']
                        times, values, units = self.Report.ModelAlt.getFWAReservoirOutputTimeseries(Line_info['ressimresname'],
                                                                                             metadata['parameter'])
                        metadata['type'] = 'fwa'
                if 'units' in Line_info.keys():
                    metadata['units'] = Line_info['units']
                else:
                    metadata['units'] = None


        else:
            WF.print2stdout('No Data Defined for line', debug=self.Report.debug)
            return np.array([]), np.array([]), metadata

        if metadata['collection']:
            metadata = self.checkForIdenticalMembers(values, metadata)
            if wanted_members != 'all':
                values = WF.filterByMember(values, wanted_members)
            #we should check for duplicates here

        if 'omitvalue' in Line_info.keys():
            omitval = float(Line_info['omitvalue'])
            values = WF.NaNOmittedValues(values, omitval, debug=self.Report.debug)
        elif 'omitvalues' in Line_info.keys():
            omitvals = [float(n) for n in Line_info['omitvalues']]
            for omitval in omitvals:
                values = WF.NaNOmittedValues(values, omitval, debug=self.Report.debug)

        if 'interval' in Line_info.keys():
            times, values = WT.changeTimeSeriesInterval(times, values, Line_info, self.Report.startYear)
            metadata['interval_mod'] = True

        # if a window is provided, trim to it
        if 'window' in Line_info.keys():
            times, values = WT.trimWindow(times, values, Line_info['window'])
            metadata['window_mod'] = True

        return times, values, metadata

    def computeCollectionEnvelopes(self, values, envelopes):
        '''
        calculates and plots envelopes for collection data as specified
        :param values: collection value dictionary
        :param envelopes: settings for envelopes
        :return: computed envelope value dictionaries
        '''

        collected_envelopes = {}
        for envelope in envelopes:
            if 'percent' in envelope.keys():
                envelope_tag = envelope['percent']
                collected_envelopes[envelope_tag] = []
        for vi in range(len(values[list(values.keys())[0]])): #for each value in a single set of values
            for envelope in collected_envelopes.keys():
                try:
                    quantile = float(envelope)
                    quantile = quantile / 100 #envlopes come in as 0-100, but we need 0 - 1
                    assert(0. <= quantile <= 1.)
                    quantilevals = []
                    for key, vs in values.items():
                        quantilevals.append(vs[vi])
                    if np.all(np.isnan(quantilevals)):
                        collected_envelopes[envelope].append(np.nan)
                    else:
                        collected_envelopes[envelope].append(np.nanquantile(quantilevals, quantile))
                except AssertionError:
                    if quantile < 0.:
                        WF.print2stdout(f'Envelope {envelope} under 0. Envelopes must be between 0 and 1.', debug=self.Report.debug)
                    elif quantile > 1.:
                        WF.print2stdout(f'Envelope {envelope} over 1. Envelopes must be between 0 and 1.', debug=self.Report.debug)
                    else:
                        WF.print2stdout(f'Unknown Collection plot envelope {envelope}. Skipping.', debug=self.Report.debug)
                    collected_envelopes.pop(envelope)
                except:
                    WF.print2stdout(f'Unknown Collection plot envelope {envelope}. Skipping.', debug=self.Report.debug)
                    collected_envelopes.pop(envelope)

        return collected_envelopes

    #################################################################
    #Profile Functions
    #################################################################

    def getProfileDataDictionary(self, settings):
        '''
        Gets profile line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        line_settings = {}
        missing = []
        timestamps = settings['timestamps']
        for line in settings[settings['datakey']]:
            numtimesused = 0
            if 'flag' not in line.keys():
                WF.print2stdout('Flag not set for line (Computed/Observed/etc)', debug=self.Report.debug)
                WF.print2stdout('Not plotting Line:', line, debug=self.Report.debug)
                continue
            elif line['flag'].lower() == 'computed':
                # for ID in self.SimulationVariables.keys():
                for ID in self.Report.accepted_IDs:
                    curline = pickle.loads(pickle.dumps(line, -1))
                    curline = self.Report.configureSettingsForID(ID, curline)
                    curline['numtimesused'] = numtimesused
                    curline['ID'] = ID
                    if not self.Report.checkModelType(curline):
                        continue
                    data, line_settings, success = self.updateProfileDataDictionary(data, line_settings, curline, timestamps)
                    if success:
                        numtimesused += 1
                    else:
                        missing.append(curline['flag'])
            else:
                if self.Report.currentlyloadedID != self.Report.base_id:
                    line = self.Report.configureSettingsForID(self.Report.base_id, line)
                else:
                    line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')
                line['numtimesused'] = numtimesused
                if not self.Report.checkModelType(line):
                    continue
                data, line_settings, success = self.updateProfileDataDictionary(data, line_settings, line, timestamps)
                if success:
                    numtimesused += 1
                else:
                    missing.append(line['flag'])

        return data, line_settings, missing
    def updateProfileDataDictionary(self, data, line_settings, profile, timestamps):
        '''
        organizes line information and places it into a data dictionary
        :param data: dictionary containing data
        :param line_settings: dictionary containing all line settings
        :param profile: dictionary containing line settings
        :param timestamps: list of dates to get data
        :return: updated data dictionary, updated line_settings dictionary
        '''

        vals, elevations, depths, times, metadata = self.getProfileValues(profile, timestamps) #Test this speed for grabbing all profiles and then choosing
        datacheck = False
        if len(vals) > 0:
            datacheck = True
            datamem_key = self.buildMemoryKey(profile)

            if profile['flag'] in line_settings.keys() or profile['flag'] in data.keys():
                datakey = '{0}_{1}'.format(profile['flag'], profile['numtimesused'])
            else:
                datakey = profile['flag']

            line_settings[datakey] = {'logoutputfilename': datamem_key,
                                      }

            line_settings[datakey].update(metadata)
            line_settings[datakey].update(profile)

            data[datakey] = {'values': vals,
                             'elevations': elevations,
                             'depths': depths,
                             'times': times
                             }

            for key in profile.keys():
                if key not in line_settings[datakey].keys():
                    line_settings[datakey][key] = profile[key]

        return data, line_settings, datacheck

    def getProfileValues(self, Profile_info, timesteps):
        '''
        reads in profile data from various sources for profile plots at given timesteps
        attempts to get elevations if possible
        :param Profile_info: dictionary containing settings for line
        :param timesteps: given list of timesteps to extract data at
        :return: values, elevations, depths, flag
        '''

        metadata = {'frommemory': False,
                    'subset': False,
                    'flag': Profile_info['flag'],
                    'units': None,
                    'y_units': None,
                    'isprofile': True
                    }

        if 'units' in Profile_info.keys():
            metadata['units'] = Profile_info['units']

        datamemkey = self.buildMemoryKey(Profile_info)
        values, elevations, depths, times = [], [], [], []

        if isinstance(timesteps, str):
            metadata['subset'] = False
        else:
            metadata['subset'] = True

        if datamemkey in self.Memory.keys():
            dm = pickle.loads(pickle.dumps(self.Memory[datamemkey], -1))
            subset = dm['metadata']['subset'] #if when grabbed, all timesteps were grabbed, or specific ones
            metadata['frommemory'] = True
            WF.print2stdout(f'retrieving {datamemkey} profile from datamemory', debug=self.Report.debug)
            if isinstance(timesteps, str): #if looking for all
                if not subset: #the last time data was grabbed, it was not a subset, aka all
                    metadata['units'] = dm['units']
                    values, elevations, depths, times = dm['values'], dm['elevations'], dm['depths'], dm['times']
                else:
                    WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)
                    metadata['frommemory'] = False
            elif np.array_equal(timesteps, dm['times']):
                metadata['units'] = dm['metadata']['units']
                values, elevations, depths, times = dm['values'], dm['elevations'], dm['depths'], dm['times']
            else:
                WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)
                metadata['frommemory'] = False

        #read from source..
        if not metadata['frommemory']:
            if 'filename' in Profile_info.keys(): #Get data from Observed
                filename = Profile_info['filename']
                metadata['source'] = filename
                values, yvals, times = WDR.readTextProfile(filename, timesteps, self.Report.StartTime, self.Report.EndTime)
                if 'y_convention' in Profile_info.keys():
                    metadata['y_convention'] = Profile_info['y_convention']
                    if Profile_info['y_convention'].lower() == 'depth':
                         values, elevations, depths, times = values, [], yvals, times
                    elif Profile_info['y_convention'].lower() == 'elevation':
                        values, elevations, depths, times = values, yvals, [], times
                    else:
                        WF.print2stdout('Unknown value for flag y_convention: {0}'.format(Profile_info['y_convention']), debug=self.Report.debug)
                        WF.print2stdout('Please use "depth" or "elevation"', debug=self.Report.debug)
                        WF.print2stdout('Assuming depths...', debug=self.Report.debug)
                        values, elevations, depths, times = values, [], yvals, times
                else:
                    WF.print2stdout('No value for flag y_convention', debug=self.Report.debug)
                    WF.print2stdout('Assuming depths...', debug=self.Report.debug)
                    values, elevations, depths, times = values, [], yvals, times

            elif 'h5file' in Profile_info.keys() and 'ressimresname' in Profile_info.keys():
                filename = Profile_info['h5file']
                metadata['source'] = filename
                if not os.path.exists(filename):
                    WF.print2stdout('ERROR: H5 file does not exist:', filename, debug=self.Report.debug)
                externalResSim = WRSS.ResSim_Results('', '', '', '', self.Report, external=True)
                externalResSim.openH5File(filename)
                externalResSim.load_time() #load time vars from h5
                externalResSim.loadSubdomains()
                values, elevations, depths, times, units = externalResSim.readProfileData(Profile_info['ressimresname'],
                                                                                          Profile_info['parameter'], timesteps)
                metadata['units'] = units

            elif 'w2_segment' in Profile_info.keys():
                if self.Report.program.lower() == 'cequalw2':
                    if 'w2_file' in Profile_info.keys():
                        resultsfile = Profile_info['w2_file']
                    else:
                        resultsfile = None
                    metadata['source'] = resultsfile
                    values, elevations, depths, times = self.Report.ModelAlt.readProfileData(Profile_info['w2_segment'],
                                                                                             timesteps,
                                                                                             resultsfile=resultsfile)
                    metadata['units'] = 'c' #W2 outputs in metric
                    times = WT.JDateToDatetime(times, self.Report.startYear)

            elif 'ressimresname' in Profile_info.keys():
                if self.Report.program.lower() == 'ressim':
                    metadata['source'] = Profile_info['ressimresname']
                    values, elevations, depths, times, units = self.Report.ModelAlt.readProfileData(Profile_info['ressimresname'],
                                                                                                   Profile_info['parameter'], timesteps,
                                                                                                   )
                    metadata['units'] = units

        self.Memory[datamemkey] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                   'values': pickle.loads(pickle.dumps(values, -1)),
                                   'elevations': pickle.loads(pickle.dumps(elevations, -1)),
                                   'depths': pickle.loads(pickle.dumps(depths, -1)),
                                   'metadata': pickle.loads(pickle.dumps(metadata, -1))}


        if len(values) == 0:
            WF.print2stdout('No Data Defined for Profile', debug=self.Report.debug)
            WF.print2stdout('Profile:', Profile_info, debug=self.Report.debug)

        return values, elevations, depths, times, metadata
        # return [], [], [], [], metadata

    def getReservoirContourDataDictionary(self, settings):
        '''
        Gets Contour Reservoir data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        data_settings = {}
        for datapath in settings['datapaths']:
            for ID in self.Report.accepted_IDs:
                curreach = pickle.loads(pickle.dumps(datapath, -1))
                curreach = self.Report.configureSettingsForID(ID, curreach)
                if not self.Report.checkModelType(curreach):
                    continue
                values, elevations, depths, dates, metadata = self.getProfileValues(curreach, 'all')
                topwater = self.getProfileTopWater(curreach, 'all')
                if 'interval' in curreach.keys():
                    dates_change, values = WT.changeTimeSeriesInterval(dates, values, curreach,
                                                                       self.Report.startYear)
                    dates_change, topwater = WT.changeTimeSeriesInterval(dates, topwater, curreach,
                                                                         self.Report.startYear)
                    dates = dates_change
                if WF.checkData(values):
                    if 'flag' in datapath.keys():
                        flag = datapath['flag']
                    elif 'label' in datapath.keys():
                        flag = datapath['label']
                    else:
                        flag = 'reservoir_{0}'.format(ID)
                    if flag in data.keys():
                        count = 1
                        newflag = flag + '_{0}'.format(count)
                        while newflag in data.keys():
                            count += 1
                            newflag = flag + '_{0}'.format(count)
                        WF.print2stdout('The current flag is {0}'.format(flag), debug=self.Report.debug)
                        flag = newflag
                        WF.print2stdout('The new flag is {0}'.format(newflag), debug=self.Report.debug)
                    datamem_key = self.buildMemoryKey(datapath)

                    data_settings[flag] = {'logoutputfilename': datamem_key,
                                           'ID': ID,
                                           }

                    data[flag] = {'values': values,
                                  'dates': dates,
                                  'elevations': elevations,
                                  'topwater': topwater,
                                  'ID': ID}

                    data_settings[flag].update(metadata)
                    data_settings[flag].update(curreach)

                    for key in datapath.keys():
                        if key not in data_settings[flag].keys():
                            data_settings[flag][key] = datapath[key]
        #reset
        self.Report.loadCurrentID(self.Report.base_id)
        self.Report.loadCurrentModelAltID(self.Report.base_id)
        return data, data_settings

    def getProfileTopWater(self, profile, timesteps):
        '''
        gets topwater elevations for timestamps
        :param profile: dictionary containing settings for line
        :param timesteps: given list of timesteps to extract data at
        :return: values, elevations, depths, flag
        '''

        datamemkey = self.buildMemoryKey(profile)
        if datamemkey in self.Memory.keys():
            dm = pickle.loads(pickle.dumps(self.Memory[datamemkey], -1))
            WF.print2stdout('retrieving profile topwater from datamem', debug=self.Report.debug)
            if isinstance(timesteps, str): #if looking for all
                if dm['subset'] == 'false': #the last time data was grabbed, it was not a subset, aka all
                    return dm['topwater']
                else:
                    WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)
            elif np.array_equal(timesteps, dm['times']):
                return dm['topwater']
            else:
                WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)

        if 'filename' in profile.keys(): #Get data from Observed
            filename = profile['filename']
            values, yvals, times = WDR.readTextProfile(filename, timesteps, self.Report.StartTime, self.Report.EndTime)
            if 'y_convention' in profile.keys():
                if profile['y_convention'].lower() == 'elevation':
                    return [yval[0] for yval in yvals]
                elif profile['y_convention'].lower() == 'depth':
                    WF.print2stdout('Unable to get topwater from depth.', debug=self.Report.debug)
                    return []
                else:
                    WF.print2stdout('Unknown value for flag y_convention: {0}'.format(profile['y_convention']), debug=self.Report.debug)
                    WF.print2stdout('Please use "elevation"', debug=self.Report.debug)
                    WF.print2stdout('Assuming elevations...', debug=self.Report.debug)
                    return [yval[0] for yval in yvals]
            else:
                WF.print2stdout('No value for flag y_convention', debug=self.Report.debug)
                WF.print2stdout('Assuming elevation...', debug=self.Report.debug)
                return [yval[0] for yval in yvals]

        elif 'h5file' in profile.keys() and 'ressimresname' in profile.keys():
            filename = profile['h5file']
            if not os.path.exists(filename):
                WF.print2stdout('ERROR: H5 file does not exist:', filename, debug=self.Report.debug)
                return []
            externalResSim = WRSS.ResSim_Results('', '', '', '', self.Report, external=True)
            externalResSim.openH5File(filename)
            externalResSim.load_time() #load time vars from h5
            externalResSim.loadSubdomains()
            topwater = externalResSim.readProfileTopwater(profile['ressimresname'], timesteps)
            return topwater

        elif 'w2_segment' in profile.keys():
            if self.Report.program.lower() != 'cequalw2':
                return []
            topwater = self.Report.ModelAlt.readProfileTopwater(profile['w2_segment'], timesteps)
            return topwater

        elif 'ressimresname' in profile.keys():
            if self.Report.program.lower() != 'ressim':
                return []
            topwater = self.Report.ModelAlt.readProfileTopwater(profile['ressimresname'], timesteps)
            return topwater

        WF.print2stdout('No Data Defined for line', debug=self.Report.debug)
        WF.print2stdout('Profile:', profile, debug=self.Report.debug)
        return []

    def commitProfileDataToMemory(self, data, line_settings, object_settings):
        '''
        commits updated data to data memory dictionary that keeps track of data
        :param data: dictionary containing data
        :param line_settings: dictionary containing settings about lines
        :param object_settings:  dicitonary of user defined settings for current object
        '''

        for line in data.keys():
            write = False
            values = pickle.loads(pickle.dumps(data[line]['values'], -1))
            depths = pickle.loads(pickle.dumps(data[line]['depths'], -1))
            elevations = pickle.loads(pickle.dumps(data[line]['elevations'], -1))
            metadata = pickle.loads(pickle.dumps(line_settings[line]['metadata'], -1))
            metadata['isprofile'] = True
            datamem_key = line_settings[line]['logoutputfilename']
            if datamem_key not in self.Memory.keys():
                write = True
            else:
                if not np.array_equal(object_settings['timestamps'], self.Memory[datamem_key]['times']):
                    write = True

            if write:
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
        '''
        Grabs data from time series data to be used in tables
        :param object_settings: currently selected object settings dictionary
        :return: dictionary object containing info from each data source and list of units
        :type: determines the type of data we are grabbing.
            Timeseries: date/time values that will be organzied into data
            Formatted: existing tables to be read in as is
        '''

        data = {}
        line_settings = {}
        temp_flag_number = 1
        for dp in object_settings['datapaths']:
            numtimesused = 0
            if 'flag' not in dp.keys():
                WF.print2stdout('Flag not set for line (Computed/Observed/etc)', debug=self.Report.debug)
                temp_flag = f'Flag{str(temp_flag_number).zfill(6)}'
                WF.print2stdout(f'Using Temporary flag {temp_flag}, but table may not work as intended', debug=self.Report.debug)
                dp['flag'] = temp_flag
                temp_flag_number += 1
            if type.lower() == 'timeseries':
                if dp['flag'].lower() == 'computed':
                    for ID in self.Report.accepted_IDs:
                        cur_dp = pickle.loads(pickle.dumps(dp, -1))
                        cur_dp = self.Report.configureSettingsForID(ID, cur_dp)
                        cur_dp['numtimesused'] = numtimesused
                        cur_dp['ID'] = ID
                        if not self.Report.checkModelType(cur_dp):
                            continue
                        data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, cur_dp)
                        if success:
                            numtimesused += 1
                            dp_used = True
                else:
                    if self.Report.currentlyloadedID != self.Report.base_id:
                        dp = self.Report.configureSettingsForID(self.Report.base_id, dp)
                    else:
                        dp = WF.replaceflaggedValues(self.Report, dp, 'modelspecific')
                    dp['numtimesused'] = numtimesused
                    if not self.Report.checkModelType(dp):
                        continue
                    data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, dp)
                    if success:
                        numtimesused += 1


            elif type.lower() == 'formatted':
                dp = self.Report.configureSettingsForID(self.Report.base_id, dp)
                if 'filename' in dp.keys():
                    data[dp['flag']] = WR.readFormattedTable_Pandas(dp['filename'])
                line_settings[dp['flag']] = {}
                datamem_key = self.buildMemoryKey(dp)
                line_settings[dp['flag']]['logoutputfilename'] = datamem_key
                line_settings[dp['flag']].update(dp)

        # if self.Report.memberiteration:
        #     line_settings = self.checkForIdenticalMembers(data, line_settings)

        return data, line_settings

    def filterFormattedTable(self, data, object_settings, primarykey=None):
        '''
        Filters file based on user defined settings
        flags to filter on:
        headers - select what headers to be included in the table. All by default
        selectbyfirstcell - select by a list given by user cooresponding with the first row
        reportType: Forecast - checks for the member in the first row and filters on it
        :param data: pandas dataframe of table
        :param object_settings: dictionary of settings for table
        :param primarykey: primary key of table if we've already figured it out
        :return: formatted pandas DF with excluded data
        '''

        if data.empty:
            return data

        if primarykey == None:
            primarykey = self.getPrimaryTableKey(data, object_settings)

        if 'headers' in object_settings.keys():
            selected_headers = object_settings['headers']
            columns = data.columns
            for column in columns:
                if column not in selected_headers:
                    data.drop(column, axis=1, inplace=True)

        if 'filters' in object_settings.keys():
            selected_rows = object_settings['filters']
            if 'formatprimaryascollection' in object_settings.keys():
                if object_settings['formatprimaryascollection'].lower() == 'true':
                    selected_rows = [WF.formatMembers(n) for n in selected_rows] #match the table if theyve been converted

            for index, row in data.iterrows():
                if row[primarykey] not in selected_rows:
                    data.drop(index=index, inplace=True)

        if self.Report.reportType == 'forecast':
            if 'formatprimaryascollection' in object_settings.keys():
                if object_settings['formatprimaryascollection'].lower() == 'true':
                    for index, row in data.iterrows():
                        if row[primarykey] not in self.Report.allMembers:
                            data.drop(index=index, inplace=True)

        return data

    def mergeFormattedTables(self, data, data_settings, object_settings):
        '''
        combines multiple formatted tables by a common key
        :param data: dictionary containing pandas DFs
        :param data_settings: settings dict about the data
        :param object_settings: settings dict about the object
        :return: main table object and settings
        '''

        table_keys = list(data.keys())

        if len(table_keys) < 2: #todo: can this be zero?
            main_table = data[table_keys[0]]
            main_data_settings = data_settings[table_keys[0]]

        else:
            if 'merge_on' in object_settings.keys():
                merge_on = object_settings['merge_on']
            else:
                common_keys = [list(data[n].columns) for n in table_keys]
                common_keys = list(itertools.chain(*common_keys))
                common_keys_count = Counter(common_keys)
                merge_on = None
                for ckc in common_keys_count.keys(): #TODO: come back and improve this logic
                    if common_keys_count[ckc] == len(table_keys):
                        merge_on = ckc
            if merge_on != None:
                main_table = data[table_keys[0]]
                main_data_settings = data_settings[table_keys[0]]
                for tk in table_keys[1:]:
                    main_table = pd.merge(main_table, data[tk], on=merge_on)
                    main_data_settings.update(data_settings[tk])
                #Todo: does the flag matter?

            else:
                WF.print2stdout('Unable to find common key in tables.', debug=self.Report.debug)
                main_table = data[table_keys[0]]
                main_data_settings = data_settings[table_keys[0]]

        return main_table, main_data_settings

    def getPrimaryTableKey(self, data, object_settings):
        '''
        finds the primary key for a table
        :param data: pandas DF containing formatted text table data
        :param object_settings: dictionary containing settings for the object
        :return: primary key
        '''

        if 'primarykey' in object_settings:
            primarykey = object_settings['primarykey']
        elif 'merge_on' in object_settings.keys():
            primarykey = object_settings['merge_on']
        else:
            if isinstance(data, dict):
                firstkey = list(data.keys())[0]
                primarykey = list(data[firstkey].columns)
                if len(primarykey) == 0:
                    primarykey = firstkey
                else:
                    primarykey = primarykey[0]
            elif isinstance(data, pd.DataFrame):
                primarykey = list(data.columns)[0]
            WF.print2stdout('Unable to establish table primary key based on input.', debug=self.Report.debug)
            WF.print2stdout('To fix, specify a "primarykey" flag in the input file.', debug=self.Report.debug)
            WF.print2stdout(f'Using first column, {primarykey}.', debug=self.Report.debug)
        return primarykey

    #################################################################
    #Contour Functions
    #################################################################

    def getContours(self, settings):
        '''
        Retrieves Contour data from sources
        :param settings: dictionary for object settings
        :return: times, values, units distance. All objects are 1D/2D arrays
        '''

        metadata = {'iscontour': True,
                    'units': None,
                    'frommemory': False,
                    'interval': None}

        datamem_key = self.buildMemoryKey(settings)
        values, distance, times = [], [], []

        if datamem_key in self.Memory.keys():
            WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
            datamem_entry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
            times = datamem_entry['dates']
            values = datamem_entry['values']
            metadata = datamem_entry['metadata']
            distance = datamem_entry['distance']
            if 'interval' in settings.keys():
                if settings['interval'].lower() != metadata['interval']:
                    WF.print2stdout('incorrect interval in memory. Re-extracting..', debug=self.Report.debug)
                    metadata['frommemory'] = False


        if not metadata['frommemory']:
            if 'ressimresname' in settings.keys(): #Ressim subdomain
                checkdomain = self.Report.ModelAlt.checkSubdomain(settings['ressimresname'])
                if not checkdomain:
                    return [], [], [], metadata
                times, values, distance = self.Report.ModelAlt.readSubdomain(settings['parameter'],
                                                                             settings['ressimresname'])
        elif 'w2_file' in settings.keys():
            times, values, distance = self.Report.ModelAlt.readSegment(settings['w2_file'],
                                                                       settings['parameter'])

        if 'interval' in settings.keys():
            times, values = WT.changeTimeSeriesInterval(times, values, settings, self.Report.startYear)

        self.Memory[datamem_key] = {'dates': pickle.loads(pickle.dumps(times, -1)),
                                    'values': pickle.loads(pickle.dumps(values, -1)),
                                    'distance': pickle.loads(pickle.dumps(distance, -1)),
                                    'metadata': pickle.loads(pickle.dumps(metadata, -1))}

        return times, values, distance, metadata

    def getContourDataDictionary(self, settings):
        '''
        Gets Contour line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        data_settings = {}
        # missing = []
        for reach in settings['reaches']:
            for ID in self.Report.accepted_IDs:
                curreach = pickle.loads(pickle.dumps(reach, -1))
                curreach = self.Report.configureSettingsForID(ID, curreach)
                if not self.Report.checkModelType(curreach):
                    continue
                dates, values, distance, metadata = self.getContours(curreach)
                if WF.checkData(values):
                    if 'flag' in reach.keys():
                        flag = reach['flag']
                    elif 'label' in reach.keys():
                        flag = reach['label']
                    else:
                        flag = 'reach_{0}'.format(ID)
                    if flag in data.keys():
                        count = 1
                        newflag = flag + '_{0}'.format(count)
                        while newflag in data.keys():
                            count += 1
                            newflag = flag + '_{0}'.format(count)
                        WF.print2stdout('The current flag is {0}'.format(flag), debug=self.Report.debug)
                        flag = newflag
                        WF.print2stdout('The new flag is {0}'.format(newflag), debug=self.Report.debug)
                    datamem_key = self.buildMemoryKey(reach)

                    if 'y_scalar' in settings.keys():
                        y_scalar = float(settings['y_scalar'])
                        distance *= y_scalar

                    data_settings[flag] = {'distance': distance,
                                           'ID': ID,
                                           'logoutputfilename': datamem_key}

                    data[flag] = {'values': values,
                                  'dates': dates,
                                  'ID': ID}

                    data_settings[flag].update(metadata)
                    data_settings[flag].update(reach)
                # else:
                    # missing.append(curreach['flag'])

        #reset
        self.Report.loadCurrentID(self.Report.base_id)
        self.Report.loadCurrentModelAltID(self.Report.base_id)
        return data, data_settings

    #################################################################
    #Gate Functions
    #################################################################

    def getGateDataDictionary(self, settings, makecopy=True):
        '''
        Gets profile line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :param makecopy: optional flag that determines if data is grabbed or copied. turn off for speed
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        line_data = {}
        if 'gateops' in settings.keys():
            for gi, gateop in enumerate(settings['gateops']):

                if 'flag' in gateop.keys():
                    if gateop['flag'] not in data.keys():
                        data[gateop['flag']] = {}
                        line_data[gateop['flag']] = {}
                        gateopkey = gateop['flag']
                elif 'label' in gateop.keys():
                    if gateop['label'] not in data.keys():
                        data[gateop['label']] = {}
                        line_data[gateop['flag']] = {}
                        gateopkey = gateop['label']
                else:
                    if 'GATEOP_{0}'.format(gi) not in data.keys():
                        gateopkey = 'GATEOP_{0}'.format(gi)
                        data[gateopkey] = {}
                        line_data[gateopkey] = {}

                data[gateopkey]['gates'] = {}
                line_data[gateopkey]['gates'] = {}
                for gate in gateop['gates']:
                    dates, values, _ = self.getTimeSeries(gate, makecopy=makecopy)
                    if 'flag' in gate.keys():
                        flag = gate['flag']
                    else:
                        flag = 'gate'
                    if flag in data[gateopkey]['gates'].keys():
                        count = 1
                        newflag = flag + '_{0}'.format(count)
                        while newflag in data[gateopkey]['gates'].keys():
                            count += 1
                            newflag = flag + '_{0}'.format(count)
                        flag = newflag
                    datamem_key = self.buildMemoryKey(gate)
                    value_msk = np.where(values==0)
                    values[value_msk] = np.nan
                    if 'flag' in gateop.keys():
                        gategroup = gateop['flag']
                    else:
                        gategroup = 'gategroup_{0}'.format(gi)
                    data[gateopkey]['gates'][flag] = {'values': values,
                                                      'dates': dates}

                    line_data[gateopkey]['gates'][flag] = {'logoutputfilename': datamem_key,
                                                          'gategroup': gategroup}

                    for key in gate.keys():
                        if key not in line_data[gateopkey]['gates'][flag].keys():
                            line_data[gateopkey]['gates'][flag][key] = gate[key]

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
        '''
        writes out the data used in figures to csv files for later use and checking
        '''

        for key in self.Memory.keys():
            cleankey = WF.cleanFileName(key)
            csv_name = os.path.join(self.Report.CSVPath, '{0}.csv'.format(cleankey))
            try:
                metadata = self.Memory[key]['metadata']
                if 'isprofile' in metadata:
                    # if self.Memory[key]['isprofile'] == True:
                    if metadata['isprofile'] == True:
                        alltimes = self.Memory[key]['times']
                        allvalues = self.Memory[key]['values']
                        alltimes = WF.matcharrays(alltimes, allvalues)
                        allelevs = self.Memory[key]['elevations']
                        alldepths = self.Memory[key]['depths']
                        if len(allelevs) == 0: #elevations may not always fall out
                            allelevs = WF.matcharrays(allelevs, alldepths)
                        units = metadata['units']
                        values = WF.getListItems(allvalues)
                        times = WF.getListItems(alltimes)
                        elevs = WF.getListItems(allelevs)
                        depths = WF.getListItems(alldepths)
                        if isinstance(values, (list, np.ndarray)):
                            df = pd.DataFrame({'Dates': times, 'Values ({0})'.format(units): values, 'Elevations': elevs,
                                               'Depths': depths})
                        elif isinstance(values, dict):
                            colvals = {}
                            colvals['Dates'] = times
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

                    allvalues = self.Memory[key]['values']
                    alltimes = self.Memory[key]['times']
                    # metadata = self.Memory[key]['metadata']
                    units = metadata['units']
                    times = WF.getListItems(alltimes)
                    if isinstance(allvalues, (list, np.ndarray)):
                        multidimensional = False
                        if isinstance(allvalues, list):
                            if len(allvalues) > 0:
                                if isinstance(allvalues[0], (list, np.ndarray)):
                                    multidimensional = True
                        else:
                            if len(allvalues.shape) == 2:
                                multidimensional = True
                        if not multidimensional:
                            values = WF.getListItems(allvalues)
                            df = pd.DataFrame({'Dates': times, 'Values ({0})'.format(units): values})
                        else:
                            df_dict = {'Dates': times}
                            for vi, v in enumerate(allvalues):
                                values = WF.getListItems(v)
                                df_dict[f'Values {vi} ({units})'] = values
                            df = pd.DataFrame(df_dict)
                    elif isinstance(allvalues, dict):
                        colvals = {'Dates': times}
                        # values = WF.getListItems(allvalues)
                        for key, values in allvalues.items():
                            if units != None:
                                colvals[f'{key} ({units})'] = values
                            else:
                                colvals[f'{key}'] = values
                        df = pd.DataFrame(colvals)

                df.to_csv(csv_name, index=False)

            except:
                WF.print2stdout(f'ERROR WRITING CSV FILE {csv_name}')
                WF.print2stdout(traceback.format_exc(), debug=self.Report.debug)

    def scaleValuesByTable(self, data, line_settings):
        '''
        scales values based on table target values. Looks for SPECIFIC values, not ranges.
        :param data: dictionary containing data values
        :param line_settings: dictionary containing line settings
        :return: updated dictionary
        '''

        for d in data.keys():
            if 'scalartable' in line_settings[d].keys() and 'scalefrom' in line_settings[d].keys():
                WF.print2stdout(f'Scalar table found for {d}', debug=self.Report.debug)
                tablevalues = WDR.readScalarTable(line_settings[d]['scalartable'])
                scalefromflag = line_settings[d]['flag']+'_scalefrom'
                scalefrom_dates, scalefrom_values, scalefrom_metadata = self.getTimeSeries(line_settings[d]['scalefrom'])
                if len(scalefrom_values) != len(data[d]['values']):
                    WF.print2stdout(f'Values and Scaledby in different time intervals. Equalizing..', debug=self.Report.debug)
                    base_data, scaledFrom_data = WF.matchData({'dates': data[d]['dates'], 'values': data[d]['values']},
                                                         {'dates': scalefrom_dates, 'values': scalefrom_values})
                else:
                    base_data = data[d]
                    scaledFrom_data = {'values': scalefrom_values,
                                       'dates': scalefrom_dates,
                                       'metadata': scalefrom_metadata}

                #MAKE SURE NPARRAY
                for target in tablevalues.keys():
                    if target in scaledFrom_data['values']:
                        scalar = tablevalues[target]
                        target_i = [i for i, n in enumerate(scaledFrom_data['values']) if n == target]
                        base_data['values'][target_i] *= scalar

                data[d]['values'] = base_data['values']
                data[d][scalefromflag] = scaledFrom_data

        return data

