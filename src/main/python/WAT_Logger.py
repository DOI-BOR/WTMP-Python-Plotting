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
import pandas as pd

import WAT_Functions as WF

class WAT_Logger(object):

    def __init__(self, Report):
        '''
        class that handles logging
        '''

        self.Report = Report
        self.buildLogFile()

    def buildLogFile(self):
        '''
        builts the log dictionary for conisistent dictionary values
        '''

        self.Log = {'type': [], 'name': [], 'description': [], 'value': [], 'units': [], 'observed_data_path': [],
                    'start_time': [], 'end_time': [], 'compute_time': [], 'program': [], 'alternative_name': [],
                    'fpart': [], 'program_directory': [], 'region': [], 'value_start_date': [], 'value_end_date': [],
                    'function': [], 'logoutputfilename': []}

    def equalizeLog(self):
        '''
        ensure that all arrays are the same length with a '' character
        :return: append self.Log object
        '''

        longest_array_len = 0
        for key in self.Log.keys():
            if len(self.Log[key]) > longest_array_len:
                longest_array_len = len(self.Log[key])
        for key in self.Log.keys():
            if len(self.Log[key]) < longest_array_len:
                num_entries = longest_array_len - len(self.Log[key])
                for i in range(num_entries):
                    self.Log[key].append('')

    def writeLogFile(self, images_path):
        '''
        Writes out logfile data to csv file in report dir
        :param images_path: path to images file
        '''

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

        df.to_csv(os.path.join(images_path, 'Log.csv'), index=False)

    def addLogEntry(self, keysvalues, isdata=False):
        '''
        adds an entry to the log file. If data, add an entry for all lists.
        :param keysvalues: dictionary containing values and headers
        :param isdata: if True, adds blanks for all data rows to keep things consistent
        '''

        for key in keysvalues.keys():
            self.Log[key].append(keysvalues[key])
        if isdata:
            allkeys = ['type', 'name', 'function', 'description', 'value', 'units',
                       'value_start_date', 'value_end_date', 'logoutputfilename']
            for key in allkeys:
                if key not in keysvalues.keys():
                    self.Log[key].append('')

    def addSimLogEntry(self, accepted_IDs, SimulationVariables, observedDir):
        '''
        adds entries for a simulation with relevenat metadata
        :param accepted_IDs: IDs to add
        :param SimulationVariables: variables for each ID
        :param observedDir: observed dir path
        :return:
        '''

        for ID in accepted_IDs:
            WF.print2stdout('ID:', ID)
            WF.print2stdout('Simvars:', SimulationVariables[ID])
            self.Log['observed_data_path'].append(observedDir)
            self.Log['start_time'].append(SimulationVariables[ID]['StartTimeStr'])
            self.Log['end_time'].append(SimulationVariables[ID]['EndTimeStr'])
            self.Log['compute_time'].append(SimulationVariables[ID]['LastComputed'])
            self.Log['program'].append(SimulationVariables[ID]['program'])
            self.Log['alternative_name'].append(SimulationVariables[ID]['modelAltName'])
            self.Log['fpart'].append(SimulationVariables[ID]['alternativeFpart'])
            self.Log['program_directory'].append(SimulationVariables[ID]['alternativeDirectory'])
