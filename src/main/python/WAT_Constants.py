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

import calendar
import datetime as dt
import WAT_Functions as WF
from scipy import interpolate

class WAT_Constants(object):

    def __init__(self):
        '''
        Class that sets default constants that dont change, like conversions.
        '''

        self.defineUnits()
        self.defineMonths()
        self.defineTimeIntervals()
        self.defineDefaultColors()
        self.defineUnitConversions()
        self.defineModelSpecificVariables()
        self.saturatedDO()

    def defineUnits(self):
        '''
        creates dictionary with units for vars for labels
        #TODO: expand this
        :return: set class variable
                    self.units
                    self.unit_alt_names
                    self.english_units
                    self.metric_units
        '''

        self.units = {'temperature': {'metric':"c", 'english':"f"},
                      'temp': {'metric':'c', 'english':"f"},
                      'do_sat': {'metric': '%', 'english': '%'},
                      'flow': {'metric': 'm3/s', 'english': 'cfs'},
                      'storage': {'metric': 'm3', 'english': 'af'},
                      'stor': {'metric': 'm3', 'english': 'af'},
                      'elevation': {'metric': 'm', 'english': 'ft'},
                      'elev': {'metric': 'm', 'english': 'ft'},
                      'ec':  {'metric': 'us/cm', 'english': 'us/cm'},
                      'electrical conductivity': {'metric': 'us/cm', 'english': 'us/cm'},
                      'salinity': {'metric': 'psu', 'english': 'psu'},
                      'sal': {'metric': 'psu', 'english': 'psu'},
                      }

        self.unit_alt_names = {'f': ['f', 'faren', 'degf', 'fahrenheit', 'fahren', 'deg f', '°f'],
                                'c': ['c', 'cel', 'celsius', 'deg c', 'degc', '°c'],
                                'm3/s': ['m3/s', 'm3s', 'metercubedpersecond', 'cms'],
                                'cfs': ['cfs', 'cubicftpersecond', 'f3/s', 'f3s'],
                                'm': ['m', 'meters', 'mtrs'],
                                'ft': ['ft', 'feet'],
                                'm3': ['m3', 'meters cubed', 'meters3', 'meterscubed', 'meters-cubed'],
                                'af': ['af', 'acrefeet', 'acre-feet', 'acfeet', 'acft', 'ac-ft', 'ac/ft'],
                                'm/s': ['mps', 'm/s', 'meterspersecond', 'm/second'],
                                'ft/s': ['ft/s', 'fps', 'feetpersecond', 'feet/s']}

        self.units_fancy_flags_internal = {'f': u"\u00b0F",
                                            'c': u"\u00b0C"}

        self.units_fancy_flags_external = {'f': r"&#176;F",
                                          'c': r"&#176;C"}
        self.english_units = {self.units[key]['metric']: self.units[key]['english'] for key in self.units.keys()}
        self.metric_units = {v: k for k, v in self.english_units.items()}


    def normalize_unit(self, unit):
        """
        Normalizes a unit to its standard form based on self.unit_alt_names.
        :param unit: str - the unit to normalize
        :return: str - the normalized unit
        """
        WF.print2stdout(f'Normalizing unit: {unit}')
        testing_units = unit.strip()
        if testing_units == unit:
            WF.print2stdout(f'@@@@@ Units have no extra spaces')
        else:
            WF.print2stdout(f'@@@@@ Units have extra spaces')

        for standard, alt_names in self.unit_alt_names.items():
            if unit.lower().strip() in alt_names:
                WF.print2stdout(f'Match found for {unit}, normalized to {standard}')  # Add this
                return standard
        WF.print2stdout(f'No match found for {unit}, returning original')  # Add this
        return unit  # Return the original if no match is found


    def defineDefaultColors(self):
        '''
        sets up a list of default colors to use in the event that colors are not set up in the graphics default file
        for a line
        Color Changes based off https://davidmathlogic.com/colorblind/#%2388CCEE-%23882255-%23117733-%2344AA99-%23DDCC77-%23CC6677-%23AA4499-%23332288
        :return: class variable
                    self.def_colors
        '''

        # self.def_colors = ['#003E51', '#FF671F', '#007396', '#215732', '#C69214', '#4C12A1', '#DDCBA4', '#9A3324']
        self.def_colors = ['#88CCEE', '#882255', '#117733', '#44AA99', '#DDCC77', '#CC6677', '#AA4499', '#332288']
        #                     blue       red       green    light green   yellow     salmon    redpink, purple

    def defineMonths(self):
        '''
        defines month 3 letter codes for table labels, and reference dicts for months and numbers (aka Jan: 1)
        :return: class variables
                    self.month2num
                    self.num2month
                    self.mo_str_3
        '''

        self.month2num = {month.lower(): index for index, month in enumerate(calendar.month_abbr) if month}
        self.num2month = {index: month.lower() for index, month in enumerate(calendar.month_abbr) if month}
        self.mo_str_3 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def defineTimeIntervals(self):
        '''
        sets up a dictionary for DSS intervals and their associated timedelta amount for setting up regular
        interval time series arrays
        :return: class variable
                    self.time_intervals
        '''

        self.time_intervals = {'1MIN': [dt.timedelta(minutes=1),'np'],
                               '2MIN': [dt.timedelta(minutes=2),'np'],
                               '5MIN': [dt.timedelta(minutes=5),'np'],
                               '6MIN': [dt.timedelta(minutes=6),'np'],
                               '10MIN': [dt.timedelta(minutes=10),'np'],
                               '12MIN': [dt.timedelta(minutes=12),'np'],
                               '15MIN': [dt.timedelta(minutes=15),'np'],
                               '30MIN': [dt.timedelta(minutes=30),'np'],
                               '1HOUR': [dt.timedelta(hours=1),'np'],
                               '2HOUR': [dt.timedelta(hours=2),'np'],
                               '3HOUR': [dt.timedelta(hours=3),'np'],
                               '4HOUR': [dt.timedelta(hours=4),'np'],
                               '5HOUR': [dt.timedelta(hours=5),'np'],
                               '6HOUR': [dt.timedelta(hours=6),'np'],
                               '1DAY': [dt.timedelta(days=1),'np'],
                               '1MON': ['1M', 'pd'],
                               '2MON': ['2M', 'pd'],
                               '6MON': ['6M', 'pd'],
                               '1YEAR': ['1Y', 'pd']}

    def defineUnitConversions(self):
        '''
        dictionary for unit conversion, key is source units, then conversion
        '''

        self.conversion = {'m3/s': 35.314666213,
                          'cfs': 0.0283168469997284,
                          'm': 3.28084,
                          'ft': 0.3048,
                          'm3': 0.000810714,
                          'af': 1233.48}

    def defineModelSpecificVariables(self):
        '''
        defines model specific input types
        '''

        self.model_specific_vars = {'ressimresname': 'ressim',
                                   'xy': 'ressim',
                                   'w2_segment': 'cequalw2',
                                   'w2_file': 'cequalw2'}

    def saturatedDO(self):
        '''
        creates a precalculated interpolated field for saturated DO based off of temperature
        '''

        self.sat_data_do = [14.60, 14.19, 13.81, 13.44, 13.09, 12.75, 12.43, 12.12, 11.83, 11.55, 11.27, 11.01, 10.76, 10.52, 10.29,
                       10.07, 9.85, 9.65, 9.45, 9.26, 9.07, 8.90, 8.72, 8.56, 8.40, 8.24, 8.09, 7.95, 7.81, 7.67, 7.54, 7.41,
                       7.28, 7.16, 7.05, 6.93, 6.82, 6.71, 6.61, 6.51, 6.41, 6.31, 6.22, 6.13, 6.04, 5.95]
        self.sat_data_temp = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
                         22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
                         42., 43., 44., 45.]

        self.satDO_interp = interpolate.interp1d(self.sat_data_temp, self.sat_data_do,
                                        fill_value=(self.sat_data_do[0], self.sat_data_do[-1]), bounds_error=False)


