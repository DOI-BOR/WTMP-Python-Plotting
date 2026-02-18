'''
* Copyright 2022 United States Bureau of Reclamation (USBR).
* United States Department of the Interior
* All Rights Reserved. USBR PROPRIETARY/CONFIDENTIAL.
* Source may not be released without written approval
* from USBR

Created on 7/16/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''

import numpy as np
import math
import os, sys
from scipy import interpolate
from scipy.constants import convert_temperature
from sklearn.metrics import mean_absolute_error
import re
import pickle
import datetime as dt
from collections import Counter
from matplotlib.colors import is_color_like
import itertools

import WAT_Constants as WC
import WAT_Time as WT
import WAT_Reader as WR

constants = WC.WAT_Constants()

def print2stdout(*a, debug=True):
    '''
    prints standard message to the console standard out
    :param a: print message
    '''

    if debug:
        print(*a, file=sys.stdout)

def print2stderr(*a):
    '''
    prints error message to console standard error
    :param a: print message
    '''

    print(*a, file=sys.stderr)

def printVersion(VERSIONNUMBER):
    '''
    print current version number
    '''

    print2stdout(f'VERSION: {VERSIONNUMBER}')

def checkExists(infile):
    '''
    checks if important file exists, and if not, exit script with error
    :param infile: file path
    '''

    if not os.path.exists(infile):
        print2stderr(f'ERROR: {infile} does not exist')
        sys.exit(1)

def cleanMissing(indata):
    '''
    TODO: merge with omit values function?
    removes data with -901. flags
    :param indata: array of data to be cleaned
    :return: cleaned data array
    '''

    indata[indata == -901.] = np.nan
    return indata

def cleanComputed(indata):
    '''
    TODO: merge with omit values function?
    removes data with -9999 flags
    :param indata: array of data to be cleaned
    :return: cleaned data array
    '''

    indata[indata == -9999.] = np.nan
    return indata

def cleanOutputDirectory(dir_name, filetype):
    '''
    removes all files with a prescribed file type from a given directory
    mainly used to erase old images from output directory
    :param dir_name: full path to directory to clean
    :param filetype: file type to delete
    '''

    files_in_directory = os.listdir(dir_name)
    filtered_files = [file for file in files_in_directory if file.endswith(filetype)]
    for file in filtered_files:
        path_to_file = os.path.join(dir_name, file)
        try:
            os.remove(path_to_file)
        except:
            print2stdout('Failed to delete', path_to_file)
            print2stdout('Continuing..')

def calcDOSaturation(temp, diss_ox, DOSat_Interp):
    '''
    calulates dissolved oxygen saturation. uses a series of pre computed DO values interpolated
    :param temp: temperature value
    :param diss_ox: dissolved oxygen
    :return: dissolved oxygen value
    '''

    do_sat = DOSat_Interp(temp)
    return diss_ox / do_sat * 100.

def calcComputedDOSat(vtemp, vdo, DOSat_Interp):
    '''
    calculates the computed dissolved saturated oxygen
    :param vtemp: temperature values
    :param vdo: values for dissovled oxy
    :return: DOSat values
    '''

    v = np.zeros_like(vtemp)
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = calcDOSaturation(vtemp[j], vdo[j], DOSat_Interp)
    return v

def calcObservedDOSat(ttemp, vtemp, vdo, ):
    '''
    calc dissolved saturated oxygen for observed data
    :param ttemp: times for data
    :param vtemp: temperature values
    :param vdo: values for dissovled oxy
    :return: time and DOSat values
    '''

    v = np.zeros_like(vtemp)
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = calcDOSaturation(vtemp[j], vdo[j])
    return ttemp, v

def getSubplotConfig(n_profiles, plots_per_row):
    '''
    get subplot configs to figure out numb plots per page and num pages.
    :param n_profiles: number of total plots
    :param plots_per_row: number of plots per row
    :return: subplot rows, subplot columns
    '''

    factor = n_profiles / plots_per_row
    if factor < 1:
        return 1, n_profiles
    else:
        return math.ceil(factor), plots_per_row

def matchData(data1, data2):
    '''
    matches two sets of data to have the same length
    if one is shorter than the other, the short one is interpolated
    :param data1: dictionary containing dates and values flags
    :param data2: dictionary containing dates and values flags
    :return: Two dictionaries containing dates and values flags (data1 and data2)
    '''

    if 'dates' in data1.keys() and 'dates' in data2.keys():
        y_key = 'dates'
    elif 'depths' in data1.keys() and 'depths' in data2.keys():
        y_key = 'depths'
    elif 'elevations' in data1.keys() and 'elevations' in data2.keys():
        y_key = 'elevations'

    v_1 = data1['values']
    if isinstance(v_1, list):
        v_1 = np.asarray(v_1)

    if y_key == 'dates':
        t_1 = [n.timestamp() for n in data1[y_key]]
    else:
        t_1 = data1[y_key]

    v_2 = data2['values']
    if isinstance(v_2, list):
        v_2 = np.asarray(v_2)

    if y_key == 'dates':
        t_2 = [n.timestamp() for n in data2[y_key]]
    else:
        t_2 = data2[y_key]

    if len(v_1) == 0 or len(v_2) == 0:
        return data1, data2

    if len(v_1) == len(v_2):
        return data1, data2

    elif len(v_1) > len(v_2):
        f_interp = interpolate.interp1d(t_2, v_2, bounds_error=False, fill_value=np.nan)
        v2_interp = f_interp(t_1)
        msk = np.isfinite(v2_interp)
        v_1_msk = v_1[msk]
        v_2_msk = v2_interp[msk]
        data1['values'] = v_1_msk
        data2['values'] = v_2_msk
        data2[y_key] = data1[y_key][msk]
        data1[y_key] = data1[y_key][msk]
        return data1, data2

    elif len(v_2) > len(v_1):
        f_interp = interpolate.interp1d(t_1, v_1, bounds_error=False, fill_value=np.nan)
        v1_interp = f_interp(t_2)
        msk = np.isfinite(v1_interp)
        v_1_msk = v1_interp[msk]
        v_2_msk = np.asarray(v_2)[msk]
        data1['values'] = v_1_msk
        data1[y_key] = data2[y_key][msk]
        data2[y_key] = data2[y_key][msk]
        data2['values'] = v_2_msk
        return data1, data2

def printSimulationInfo(simulation):
    print2stdout(f'\nSimulation: {simulation["name"]} - {simulation["basename"]} - {simulation["ID"]}')
    print2stdout('Directory: {}'.format(simulation['directory']))
    print2stdout('DSS file: {}'.format(simulation['dssfile']))
    print2stdout('starttime: {}'.format(simulation['starttime']))
    print2stdout('endtime: {}'.format(simulation['endtime']))
    if 'csvfile' in simulation.keys():
        print2stdout('csvfile: {}'.format(simulation['csvfile']))
    if 'modelalternatives' in simulation.keys() and len(simulation['modelalternatives']) > 0:
        print2stdout('Model Alternatives:')
        for modelalt in simulation['modelalternatives']:
            print2stdout('\t{0} - {1}'.format(modelalt['name'], modelalt['program']))

def checkData(dataset, flag=None):
    '''
    checks datasets to ensure that they are valid for representation. Checks for a given flag, any length and
    that its not all NaNs
    :param dataset: input dataset list, array or dict
    :param flag: flag to get data from dataset if dict
    :return: boolean value on data viability
    '''

    if isinstance(dataset, dict):
        if flag != None:
            if flag not in dataset.keys():
                return False
            elif len(dataset[flag]) == 0:
                return False
            elif checkAllNaNs(dataset[flag]):
                return False
            else:
                return True
        else:
            multicheck = False
            for key in dataset.keys():
                if isinstance(dataset[key], dict):
                    check = checkData(dataset[key])
                    if check:
                        multicheck = True
                else:
                    check = checkData(dataset[key], flag=key)
                    if check:
                        multicheck = True
                if multicheck == False:
                    print2stdout(f'Invalid at {key}')
                    # return False
            if multicheck: #just need 1 valid
                return True
            else:
                return False

    elif isinstance(dataset, list) or isinstance(dataset, np.ndarray):
        if len(dataset) == 0:
            return False
        elif checkAllNaNs(dataset):
            return False
        else:
            return True
    else:
        return False

def checkAllNaNs(values):
    '''
    checks value sets for all nan values. Some NaN is okay, all is not.
    :param values: list or array of values
    :return: boolean
    '''

    if np.all(np.isnan(values)):
        return True
    else:
        return False

def removeNaNs(data1, data2, flag='values'):
    '''
    removes data points from both datasets where either ones are nans. This way, the nans dont throw off any stat
    analysis
    :param data1: input dataset list, array or dict
    :param data2: input dataset list, array or dict
    :param flag: flag to get data from dataset if dict
    :return:
    '''

    if isinstance(data1, dict):
        d1_msk = np.where(~np.isnan(data1[flag]))
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        d1_msk = np.where(~np.isnan(data1))

    if isinstance(data2, dict):
        d2_msk = np.where(~np.isnan(data2[flag]))
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        d2_msk = np.where(~np.isnan(data2))

    msk = np.intersect1d(d1_msk, d2_msk)

    if isinstance(data1, dict):
        data1[flag] = np.asarray(data1[flag])[msk]
        for otherflag in data1.keys():
            if otherflag != flag:
                data1[otherflag] = np.asarray(data1[otherflag])[msk]
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        data1 = np.asarray(data1)[msk]

    if isinstance(data2, dict):
        data2[flag] = np.asarray(data2[flag])[msk]
        for otherflag in data2.keys():
            if otherflag != flag:
                data2[otherflag] = np.asarray(data2[otherflag])[msk]
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        data2 = np.asarray(data2)[msk]

    return data1, data2

def removeINFs(data1, data2, flag='values'):
    '''
    removes data points from both datasets where either ones are infinity. This way, the nans dont throw off any stat
    analysis
    :param data1: input dataset list, array or dict
    :param data2: input dataset list, array or dict
    :param flag: flag to get data from dataset if dict
    :return:
    '''

    if isinstance(data1, dict):
        d1_msk = np.where(~np.isinf(data1[flag]))
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        d1_msk = np.where(~np.isinf(data1))

    if isinstance(data2, dict):
        d2_msk = np.where(~np.isinf(data2[flag]))
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        d2_msk = np.where(~np.isinf(data2))

    msk = np.intersect1d(d1_msk, d2_msk)

    if isinstance(data1, dict):
        data1[flag] = np.asarray(data1[flag])[msk]
        for otherflag in data1.keys():
            if otherflag != flag:
                data1[otherflag] = np.asarray(data1[otherflag])[msk]
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        data1 = np.asarray(data1)[msk]

    if isinstance(data2, dict):
        data2[flag] = np.asarray(data2[flag])[msk]
        for otherflag in data2.keys():
            if otherflag != flag:
                data2[otherflag] = np.asarray(data2[otherflag])[msk]
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        data2 = np.asarray(data2)[msk]

    return data1, data2

def calcMAE(data1, data2):
    '''
    calculates the mean absolute error for two datasets
    :param data1: dataset, list array or dict
    :param data2: dataset, list array or dict
    :return: MAE value
    '''

    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan

    data1_val = np.array(data1['values'], dtype=np.float64)
    data2_val = np.array(data2['values'], dtype=np.float64)

    return mean_absolute_error(data2_val, data1_val)

def calcMeanBias(data1, data2):
    '''
    calculates the mean bias for two datasets
    :param data1: dataset, list array or dict
    :param data2: dataset, list array or dict
    :return: Meanbias value
    '''

    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan

    data1_val = np.array(data1['values'], dtype=np.float64)
    data2_val = np.array(data2['values'], dtype=np.float64)

    diff = data1_val - data2_val
    count = len(data1_val)
    mean_diff = np.sum(diff) / count
    return mean_diff

def calcRMSE(data1, data2):
    '''
    calculates the root mean square error for two datasets
    :param data1: dataset, list array or dict
    :param data2: dataset, list array or dict
    :return: RMSE value
    '''

    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan

    data1_val = np.array(data1['values'], dtype=np.float64)
    data2_val = np.array(data2['values'], dtype=np.float64)

    diff = data1_val - data2_val
    count = len(data1_val)

    rmse = np.sqrt(np.sum(diff ** 2) / count)


    return rmse

def calcNSE(data1, data2):
    '''
    Nash-Sutcliffe Efficiency (NSE) as per `Nash and Sutcliffe, 1970
    <https://doi.org/10.1016/0022-1694(70)90255-6>`_.

    :Calculation Details:
        .. math::
           E_{\\text{NSE}} = 1 - \\frac{\\sum_{i=1}^{N}[e_{i}-s_{i}]^2}
           {\\sum_{i=1}^{N}[e_{i}-\\mu(e)]^2}

        where *N* is the length of the *simulations* and *evaluation*
        periods, *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, and *μ* is the arithmetic mean.

        source: https://pypi.org/project/hydroeval
    calculates the NSE for two datasets
    :param data1: dataset, list array or dict
    :param data2: dataset, list array or dict
    :return: NSE value
    '''

    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan
    # nash = nse(data1['values'], data2['values'])
    data1_val = np.array(data1['values'], dtype=np.float64)
    data2_val = np.array(data2['values'], dtype=np.float64)

    ### STEVE
    nse_ = 1 - (
            np.sum((data2_val - data1_val) ** 2, axis=0, dtype=float)
            / np.sum((data2_val - np.mean(data2_val)) ** 2, dtype=float)
               )

    ### MIKE DEAS
    # nse_ = 1 - (
    #         np.sum((data1['values'] - data2['values']) ** 2, axis=0, dtype=np.float64)
    #         / np.sum((data2['values'] - np.mean(data2['values'])) ** 2, dtype=np.float64)
    # )

    if np.isinf(nse_):
        nse_ = np.nan

    return nse_

def getMultiDatasetCount(data1, data2):
    '''
    get the count of data for 2 datasets, usually when being compared. Nan if they cannot be made same length
    :param data1: list of values
    :param data2: list of values
    :return: length of values to compare, or nan
    '''

    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan
    if len(data1['values']) != len(data2['values']):
        return np.nan
    return len(data1['values'])

def getCount(data1):
    '''
    calculates the length of datasets
    :param data1: dataset, list array or dict
    :return: count value
    '''

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    return len(np.where(~np.isnan(data1['values']))[0])

def calcMean(data1):
    '''
    calculates the mean of datasets
    :param data1: dataset, list array or dict
    :return: mean value
    '''

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    data1_msk = np.where(np.isinf(data1['values']))
    data1['values'][data1_msk] = np.nan
    return(np.nanmean(data1['values']))

def calcMax(data1):
    '''
    calculates the maximum of datasets
    :param data1: dataset, list array or dict
    :return: mean value
    '''

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    data1_msk = np.where(np.isinf(data1['values']))
    data1['values'][data1_msk] = np.nan
    return(np.nanmax(data1['values']))

def calcMin(data1):
    '''
    calculates the maximum of datasets
    :param data1: dataset, list array or dict
    :return: mean value
    '''

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    data1_msk = np.where(np.isinf(data1['values']))
    data1['values'][data1_msk] = np.nan
    return(np.nanmin(data1['values']))

def convertTempUnits(values, units):
    '''
    convert temperature units between c and f
    :param values: temp values
    :param units: units string
    :return: converted units
    '''

    if units.lower() in ['f', 'faren', 'degf', 'fahrenheit', 'fahren', 'deg f']:
        values = convert_temperature(values, 'F', 'C')
        return values
    elif units.lower() in ['c', 'cel', 'celsius', 'deg c', 'degc']:
        values = convert_temperature(values, 'C', 'F')
        return values
    else:
        # print2stdout('Undefined temp units:', units)
        return values

def filterContourOverTopWater(values, elevations, topwater):
    '''
    takes values for contour reservoir plots and nan's out any values over topwater. Ressim duplicates data
    to the top of the domain instead of cutting it off
    :param values: list of values at each timestep
    :param elevations: elevations to find closest index to top water
    :param topwater: water surface elevation at each timestep
    :return: values with nans
    '''

    for twi, tw in enumerate(topwater):
        elevationtopwateridx = (np.abs(elevations - tw)).argmin()
        values[twi][elevationtopwateridx+1:] = np.nan
    return values

def replaceflaggedValues(Report, settings, itemset, include=[], exclude=[], forjasper=False):
    '''
    recursive function to replace flagged values in settings, include and exclude only work on first order flags, aka
    the main flag in the dictionary, not anything nested inside of it
    :param settings: dict, list or string containing settings, potentially with flags
    :return:
        settings: dict, list or string with flags replaced
    '''

    if isinstance(settings, str):
        if '%%' in settings:
            newval = replaceFlaggedValue(Report, settings, itemset, forjasper=forjasper)
            settings = newval
    elif isinstance(settings, dict):
        for key in settings.keys():
            if len(exclude) > 0:
                if key in exclude:
                    continue
            if len(include) > 0:
                if key not in include:
                    continue
            if isinstance(settings[key], dict):
                settings[key] = replaceflaggedValues(Report, settings[key], itemset, forjasper=forjasper)
            elif isinstance(settings[key], list):
                new_list = []
                for item in settings[key]:
                    new_list.append(replaceflaggedValues(Report, item, itemset, forjasper=forjasper))
                settings[key] = new_list
            else:
                try:
                    if '%%' in settings[key]:
                        newval = replaceFlaggedValue(Report, settings[key], itemset, forjasper=forjasper)
                        settings[key] = newval
                except TypeError:
                    continue
    elif isinstance(settings, list):
        for i, item in enumerate(settings):
            if len(exclude) > 0:
                if item in exclude:
                    continue
            if len(include) > 0:
                if item not in include:
                    continue
            if isinstance(item, str):
                if '%%' in item:
                    settings[i] = replaceFlaggedValue(Report, item, itemset, forjasper=forjasper)


    return settings

def parseForTextFlags(text):
    '''
    replaces text formatting flags with Jasper appropriate flags. Currently supports bold, italic and underline, in
    any order.
    examples: %%ui%%, %%i%%, %%bui%%
    :param text: formatted text
    :return:
    '''

    start_font_change_front = "&#60;style"
    start_font_change_back = "&#62;"
    end_font_change = "&#60;/style&#62;"
    flag_defs = {'b': "isBold='true'",
                 'u': "isUnderline='true'",
                 'i': "isItalic='true'"}
    flag_permutations = list(itertools.permutations(flag_defs.keys()))

    start_flags = []
    end_flags = []
    for flag_permutation in flag_permutations:
        for L in range(1, len(flag_permutation) + 1):
            for subset in itertools.combinations(flag_permutation, L):
                start_flag = f'%%{"".join(subset)}%%'
                end_flag = f'%%/{"".join(subset)}%%'
                if start_flag not in start_flags:
                    start_flags.append(start_flag)
                    end_flags.append(end_flag)

    #find all idx of start flags
    for flag in start_flags:
        flag_idx = [m.start() for m in re.finditer(flag, text)]
        if len(flag_idx) > 0:
            output_from_flag = start_font_change_front
            for flagitem in flag:
                if flagitem != '%':
                    output_from_flag += f' {flag_defs[flagitem]}'
            output_from_flag += start_font_change_back
            flag_idx.reverse()
            for idx in flag_idx: #do it backwards so the flags don't interupt the idx of each other
                text = text[:idx] + output_from_flag + text[idx + len(flag):]

    # find all idx of end flags
    for flag in end_flags:
        flag_idx = [m.start() for m in re.finditer(flag, text)]
        if len(flag_idx) > 0:
            flag_idx.reverse()
            for idx in flag_idx:
                text = text[:idx] + end_font_change + text[idx + len(flag):]

    return text

def replaceFlaggedValue(Report, value, itemset, forjasper=False):
    '''
    replaces strings with flagged values with known paths
    flags are now case insensitive with more intelligent matching. yay.
    needs to use '[1:-1]' for paths, otherwise things like /t in a path C:/trains will be taken literal
    :param value: string potentially containing flagged value
    :return:
        value: string with potential flags replaced
    '''

    if itemset == 'general':
        flagged_values = {'%%region%%': Report.ChapterRegion,
                          '%%observedDir%%': Report.observedDir,
                          '%%startyear%%': str(Report.startYear),
                          '%%endyear%%': str(Report.endYear),
                          '%%startmonth%%': str(Report.startMonth),
                          '%%endmonth%%': str(Report.endMonth),
                          '%%studydir%%': str(Report.studyDir),
                          '%%studyname%%': Report.studyname,
                          '%%simulationgroup%%': Report.SimulationGroup['Name'],
                          }

    elif itemset == 'modelspecific':
        flagged_values = {'%%ModelDSS%%': Report.DSSFile,
                          '%%Fpart%%': Report.alternativeFpart,
                          '%%program%%': Report.program,
                          '%%plugin%%': Report.program,
                          '%%modelAltName%%': Report.modelAltName,
                          '%%SimulationName%%': Report.SimulationName,
                          '%%SimulationDir%%': Report.SimulationDir,
                          '%%baseSimulationName%%': Report.baseSimulationName,
                          '%%starttime%%': Report.StartTimeStr,
                          '%%endtime%%': Report.EndTimeStr,
                          '%%LastComputed%%': Report.LastComputed,
                          '%%id%%': Report.currentlyloadedID,
                          '%%studyname%%': Report.studyname,
                          '%%analysisperiod%%': Report.AnalysisPeriod['Name'],
                          '%%watalternative%%': Report.WatAlternative['Name'],
                          }

    elif itemset == 'fancytext':
        if forjasper:
            flagged_values = {'%%gt%%': '&gt;',
                               '%%gte%%': '&ge;',
                               '%%greaterthan%%': '&gt;',
                               '%%greaterthanequalto%%': '&ge;',
                               '%%lt%%': '&lt;',
                               '%%lte%%': '&le;',
                               '%%lessthan%%': '&lt;',
                               '%%lessthanequalto%%': '&le;',
                               '%%amp%%': '&amp;',
                               '%%degrees%%': '&#176;',
                               '%%b%%': "&#60;style isBold='true'&#62;",
                               '%%/b%%': "&#60;/style&#62;"}

        else:
            flagged_values = {'%%gt%%': '>',
                               '%%gte%%': u'\u2265',
                               '%%greaterthan%%': '>',
                               '%%greaterthanequalto%%': u'\u2265',
                               '%%lt%%': '<',
                               '%%lte%%': u'\u2264',
                               '%%lessthan%%': '<',
                               '%%lessthanequalto%%': u'\u2264',
                               '%%amp%%': '&',
                               '%%degrees%%': u'\u00b0'}

    else:
        print2stderr('Invalid flag itemset: {0}'.format(itemset))
        return value

    for fv in flagged_values.keys():
        pattern = re.compile(re.escape(fv), re.IGNORECASE)
        value = pattern.sub(repr(flagged_values[fv])[1:-1], value) #this seems weird with [1:-1] but paths wont work otherwise
    return value

def selectContourByID(contoursbyID, ID):
    '''
    selects contour data based on the ID
    :param contoursbyID: dictionary containing all contours
    :param ID: selected ID ('base', 'alt_1')
    :return: list of contour keys that apply to that ID
    '''

    output_contours = {}
    for key in contoursbyID:
        if contoursbyID[key]['ID'] == ID:
            output_contours[key] = contoursbyID[key]
    return output_contours

def stackContours(contours, contours_settings):
    '''
    stacks data for multiple contour reaches so they appear as a single reach. Adds distances to stay consistent.
    keeps track of the distances in which defined reaches change
    :param contours: dictionary containing reach contour data
    :return:
        output_values: values at each timestep/distance
        output_dates: list of dates for data
        output_distance:distances for each cell center from source
        transitions: names and locations where the reaches change
    '''

    output_values = np.array([])
    output_dates = np.array([])
    output_distance = np.array([])
    transitions = {}
    for contourname in contours.keys():
        contour = contours[contourname]
        contour_settings = contours_settings[contourname]
        if len(output_values) == 0:
            output_values = pickle.loads(pickle.dumps(contour['values'], -1))
        else:
            output_values = np.append(output_values, contour['values'][1:, :], axis=0)
        if len(output_dates) == 0:
            output_dates = contour['dates']
        if len(output_distance) == 0:
            output_distance = contour_settings['distance']
            transitions[contourname] = 0
        else:
            last_distance = output_distance[-1]
            current_distances = contour_settings['distance'][1:] + last_distance
            output_distance = np.append(output_distance, current_distances)
            transitions[contourname] = current_distances[0]
    return output_values, output_dates, output_distance, transitions

def mergeLines(data, data_settings, plot_settings):
    '''
    reads in mergeline settings and combines time series as defined. returns a single time series based on the
    controller flag. then removes lines if dictated.
    :param data: dictionary containing data
    :param plot_settings: list of settings including mergeline settings
    :return: updated data dictionary
    '''

    removekeys = []
    if 'mergelines' in plot_settings.keys():
        for mergeline in plot_settings['mergelines']:
            dataflags = [n.lower() for n in mergeline['flags']]
            if 'controller' in mergeline.keys():
                #Controller matches the flag defined in data[keys]
                controller = mergeline['controller'].lower()
                if controller not in [data_settings[n]['flag'].lower() for n in data.keys()]: #do it this way so if theres comp runs we can still make this work
                    print2stdout('Mergeline Controller {0} not found in data {1}'.format(controller, data.keys()))
                    print2stdout('Not Running Merge.')
                    continue
            else:
                controller = data_settings[dataflags[0]]['flag'].lower()
            # otherflags = [data_settings[n]['flag'] for n in dataflags if n != controller]
            data_keys_with_controller = [n for n in data.keys() if data_settings[n]['flag'].lower() == controller]
            data_keys_for_otherflags = [n for n in data.keys() if data_settings[n]['flag'].lower() != controller
                                        and data_settings[n]['flag'].lower() in dataflags]

            if 'math' in mergeline.keys():
                math = mergeline['math'].lower()
            else:
                math = 'add'
                print2stdout('no Mergeline math flag. Set to add by default.')

            for datakey_controller in data_keys_with_controller:
                baseunits = data_settings[datakey_controller]['units']
                for datakey_otherflag in data_keys_for_otherflags:
                    if data_settings[datakey_otherflag]['units'] != baseunits:
                        print2stdout('WARNING: Attempting to merge lines with differing units')
                        print2stdout('{0}: {1} and {2}: {3}'.format(datakey_otherflag, data[datakey_otherflag]['units'], controller, baseunits))
                        print2stdout('If incorrect, please modify/append input settings to ensure lines '
                              'are converted prior to merging.')
                    data[datakey_controller], data[datakey_otherflag] = matchData(data[datakey_controller], data[datakey_otherflag])
                    if data_settings[datakey_controller]['collection']:
                        if data_settings[datakey_otherflag]['collection']:
                            members = list(set(data_settings[datakey_controller]['members'] + data_settings[datakey_otherflag]['members']))
                            for member in members:
                                data[datakey_controller]['values'][member] = doMathOn2Datasets(
                                    data[datakey_controller]['values'][member],
                                    data[datakey_otherflag]['values'][member], math)
                        else: #add non collection onto a collection
                            members = data_settings[datakey_controller]['members']
                            for member in members:
                                data[datakey_controller]['values'][member] = doMathOn2Datasets(
                                    data[datakey_controller]['values'][member],
                                    data[datakey_otherflag]['values'], math)
                    else:
                        if data_settings[datakey_otherflag]['collection']:
                            print2stderr(f'Unable to merge collection ({datakey_controller}) onto non collection ({datakey_otherflag}')
                        else:
                            data[datakey_controller]['values'] = doMathOn2Datasets(data[datakey_controller]['values'],
                                                                                 data[datakey_otherflag]['values'], math)

            if 'keeplines' in mergeline.keys():
                if mergeline['keeplines'].lower() == 'false':
                    for flag in data_keys_for_otherflags:
                        removekeys.append(flag)
        for flag in removekeys:
            data.pop(flag)
            data_settings.pop(flag)
    return data, data_settings

def doMathOn2Datasets(data1, data2, math):
    if math == 'add':
        data1 += data2
    elif math == 'multiply':
        data1 *= data2
    elif math == 'divide':
        data1 /= data2
    elif math == 'subtract':
        data1 -= data2
    return data1

def filterDataByYear(data, year, extraflag=None):
    '''
    filters data by a given year. Used when splitting by year
    :param data: dictionary containing data to use
    :param year: selected year
    :return:dictionary containing fultered data
    '''

    if year != 'ALLYEARS':
        for flag in data.keys():
            if len(data[flag]['dates']) > 0:
                s_idx, e_idx = getYearlyFilterIdx(data[flag]['dates'], year)
                if None not in [s_idx, e_idx]:
                    if len(data[flag]['values'].shape) == 1:
                        data[flag]['values'] = data[flag]['values'][s_idx:e_idx+1]
                    else:
                        data[flag]['values'] = data[flag]['values'][:,s_idx:e_idx + 1]
                    data[flag]['dates'] = data[flag]['dates'][s_idx:e_idx+1]
                else:
                    data[flag]['values'] = []
                    data[flag]['dates'] = []
                if extraflag != None:
                    if len(data[flag][extraflag].shape) == 1:
                        data[flag][extraflag] = data[flag][extraflag][s_idx:e_idx+1]
                    else:
                        data[flag][extraflag] = data[flag][extraflag][:, s_idx:e_idx + 1]
    return data

def getYearlyFilterIdx(dates, year):
    '''
    finds start and end index for a given year for a list of dates
    :param dates: list of datetime objects
    :param year: target year
    :return: start and end index for that year
    '''

    start_date = dates[0]
    end_date = dates[-1]
    if isinstance(year, str):
        yrsplit = year.split('-')
        s_year_date = dt.datetime(int(yrsplit[0]),1,1,0,0)
        e_year_date = dt.datetime(int(yrsplit[1]),12,31,23,59)
    else:
        s_year_date = dt.datetime(year,1,1,0,0)
        e_year_date = dt.datetime(year,12,31,23,59)

    if start_date != end_date:
        interval = (dates[1] - start_date).total_seconds()
        if start_date.year == s_year_date.year:
            s_idx = 0
        elif start_date.year > s_year_date.year: #if the filter year is bigger than the start year (aka data for
            s_idx = None
        else:
            s_idx = round(int((s_year_date - start_date).total_seconds() / interval))
            if s_idx < 0:
                s_idx = 0
        if end_date.year == e_year_date.year:
            e_idx = len(dates)
        elif start_date.year > e_year_date.year:
            e_idx = None
        else:
            e_idx = round(int((e_year_date - start_date).total_seconds() / interval))
            if e_idx < 0:
                e_idx = 0

        return s_idx, e_idx
    else:
        return 0, -1

def getObjectAllYears(years_list):
    '''
    creates a formatted string for objects describing the years used. uses start and end year if mulit-year
    :param years_list: list of years
    :return: formatted string
    '''

    if len(years_list) == 1:
        outputstring = str(years_list[0])
    else:
        outputstring = f'{years_list[0]}-{years_list[1]}'
    return outputstring

def getMonthlyFilterIdx(dates, month):
    '''
    finds start and end index for a given month for a list of dates
    :param dates: list of datetime objects
    :param month: target month
    :return: start and end index for that year
    '''
    start_date = dates[0]
    end_date = dates[-1]
    s_month_date = dt.datetime(start_date.year, month,1,0,0)
    if month == 12:
        e_month_date = dt.datetime(start_date.year+1,1,1,0,0) - dt.timedelta(seconds=1)
    else:
        e_month_date = dt.datetime(start_date.year,month+1,1,0,0) - dt.timedelta(seconds=1)

    if start_date.month > month or end_date.month < month:
        print2stdout(f'Desired month prior to start date month. {start_date.month}, {month}')
        return 0, 0

    interval = (dates[1] - start_date).total_seconds()
    if start_date.month == month:
        s_idx = 0
    else:
        s_idx = round(int((s_month_date - start_date).total_seconds() / interval))
    if end_date.month == month:
        e_idx = len(dates)
    else:
        e_idx = round(int((e_month_date - start_date).total_seconds() / interval))
    if s_idx < 0:
        print2stdout(f'SIdx less than zero for {month}. Contact developer.')
        s_idx = 0
    if e_idx < 0:
        print2stdout(f'EIdx less than zero for {month}. Contact developer.')
        e_idx = 0
    return s_idx, e_idx

def getUnitsList(line_settings, mod=''):
    '''
    creates a list of units from defined lines in user defined settings
    :param object_settings: currently selected object settings dictionary
    :return: units_list: list of used units
    '''

    units_list = []
    for flag in line_settings.keys():
        units = line_settings[flag][mod+'units']
        if units != None:
            units_list.append(units)
    return units_list

def getUsedIDs(data):
    '''
    Finds all IDs that are actually used by data
    :param data: dictionary for data
    :return: list of IDs
    '''

    IDs = []
    for key in data.keys():
        ID = data[key]['ID']
        if ID not in IDs:
            IDs.append(ID)
    return IDs

def getAllMonthIdx(timestamp_indexes, i):
    '''
    collects all indecies for all years for a given month
    :param timestamp_indexes: list of indicies that fall in every month for every year ([[], [], []])
    :param i: month index (0-11)
    :return: list of indicies for a month for all years
    '''

    out_idx = []
    for yearlist in timestamp_indexes:
        out_idx += yearlist[i]
    return out_idx

def ignoreNans(values):
    '''
    ignores nans in a list of values
    :param values: list of values
    :return: list with no nans
    '''
    v = np.asarray(values)
    return v[~np.isnan(v)]

def getPlotUnits(unitslist, object_settings, axis='x'):
    '''
    gets units for the plot. Either looks at data already plotted units, or if there are no defined units
    in the plotted data, look for a parameter flag
    :param object_settings: dictionary with plot settings
    :return: string units value
    '''

    param_flag = 'parameter' if axis == 'x' else 'y_parameter'
    unitsystem_flag = 'unitsystem' if axis == 'x' else 'y_unitsystem'
    if param_flag in object_settings.keys():
        try:
            plotunits = constants.units[object_settings[param_flag].lower()]
            if isinstance(plotunits, dict):
                if unitsystem_flag in object_settings.keys():
                    plotunits = plotunits[object_settings[unitsystem_flag].lower()]
                else:
                    plotunits = plotunits['metric']
        except KeyError:
            plotunits = ''

    elif len(unitslist) > 0:
        plotunits = getMostCommon(unitslist)

    else:
        plotunits = ''

    plotunits = translateUnits(plotunits)
    return plotunits

def getMostCommon(listvars):
    '''
    gets most common instance of a var in a list
    :param listvars: list of variables
    :return: value that is most common in the list
    '''

    occurence_count = Counter(listvars)
    if len(occurence_count) == 0:
        most_common_interval = None
    else:
        most_common_interval = occurence_count.most_common(1)[0][0]
    return most_common_interval

def translateUnits(units):
    '''
    translates possible units to better known flags for consistancy in the script and conversion purposes
    :param units: units string
    :return: units string
    '''

    if units != None:
        for key in constants.unit_alt_names.keys():
            if units.lower().strip() in constants.unit_alt_names[key]:
                return key

    return units

def convertUnitSystem(values, units, target_unitsystem, debug=False):
    '''
    converts unit systems if defined english/metric
    :param values: list of values
    :param units: units of select values
    :param target_unitsystem: unit system to convert to
    :return:
        values: either converted units if successful, or original values if unsuccessful
        units: original units if unsuccessful
        new_units: new converted units if successful
    '''

    units = translateUnits(units)

    english_units = constants.english_units
    metric_units = constants.metric_units

    if units == None:
        print2stdout('Units undefined.', debug=debug)
        return values, units

    if target_unitsystem.lower() == 'english':
        if units.lower() in english_units.keys():
            new_units = english_units[units.lower()]
            print2stdout('Converting {0} to {1}'.format(units, new_units), debug=debug)
        elif units.lower() in english_units.values():
            print2stdout('Values already in target unit system. {0} {1}'.format(units, target_unitsystem), debug=debug)
            return values, units
        else:
            print2stdout('Units not found in definitions. Not Converting.', debug=debug)
            return values, units

    elif target_unitsystem.lower() == 'metric':
        if units.lower() in metric_units.keys():
            new_units = metric_units[units.lower()]
            print2stdout('Converting {0} to {1}'.format(units, new_units), debug=debug)
        elif units.lower() in metric_units.values():
            print2stdout('Values already in target unit system. {0} {1}'.format(units, target_unitsystem), debug=debug)
            return values, units
        else:
            print2stdout('Units not found in definitions. Not Converting.', debug=debug)
            return values, units

    else:
        print2stdout('Target Unit System undefined.', target_unitsystem, debug=debug)
        print2stdout('Try english or metric', debug=debug)
        return values, units

    if units == new_units:
        print2stdout('data already in target unit system.', debug=debug)
        return values, units

    if units.lower() in ['c', 'f']:
        if isinstance(values, (list, np.ndarray)):
            new_values = convertTempUnits(values, units)
        elif isinstance(values, dict):
            new_values = {}
            for key, vs in values.items():
                new_values[key] = convertTempUnits(vs, units)
        else:
            new_values = convertTempUnits(float(values), units)

    elif units.lower() in constants.conversion.keys():
        conversion_factor = constants.conversion[units.lower()]
        if isinstance(values, (list, np.ndarray)):
            new_values = values * conversion_factor
        elif isinstance(values, dict):
            new_values = {}
            for key, vs in values.items():
                new_values[key] = vs * conversion_factor
        else: #must be a single value???
            new_values = float(values) * conversion_factor
    elif new_units.lower() in constants.conversion.keys():
        conversion_factor = 1/constants.conversion[units.lower()]
        if isinstance(values, (list, np.ndarray)):
            new_values = values * conversion_factor
        elif isinstance(values, dict):
            new_values = {}
            for key, vs in values.items():
                new_values[key] = vs * conversion_factor
        else: #must be a single value???
            new_values = float(values) * conversion_factor
    else:
        print2stdout('Undefined Units conversion for units {0}.'.format(units), debug=debug)
        print2stdout('No Conversions taking place.', debug=debug)
        return values, units

    return new_values, new_units

def updateFlaggedValues(settings, flaggedvalue, replacevalue):
    '''
    iterates and updates specific flagged values with a replacement value
    :param settings: dictionary, list or str settings
    :param flaggedvalue: flagged value to look for and replace
    :param replacevalue: value to replace flagged value with
    :return: updated settings
    '''

    if isinstance(settings, list):
        new_list = []
        for item in settings:
            item = updateFlaggedValues(item, flaggedvalue, replacevalue)
            new_list.append(item)
        return new_list

    if isinstance(settings, np.ndarray):
        new_list = []
        for item in settings:
            item = updateFlaggedValues(item, flaggedvalue, replacevalue)
            new_list.append(item)
        return np.asarray(new_list, dtype=settings.dtype)

    elif isinstance(settings, dict):
        for key in settings.keys():
            settings[key] = updateFlaggedValues(settings[key], flaggedvalue, replacevalue)
        return settings

    elif isinstance(settings, str):
        pattern = re.compile(re.escape(flaggedvalue), re.IGNORECASE)
        settings = pattern.sub(repr(replacevalue)[1:-1], settings) #this seems weird with [1:-1] but paths wont work otherwise
        return settings

    else:
        #this gets REALLY noisy.
        #lots is set up to not be replaceable, so uncomment at your own risk
        # print('Cannot set {0}'.format(flaggedvalue))
        # print('Input Not recognized type', settings)
        return settings

def configureUnits(object_settings, parameter, units):
    '''
    configure units from line settings
    :param object_settings:  dicitonary of user defined settings for current object
    :param line: current line settings
    :param units: current units of line
    :return: units
    '''

    if units == None:
        try:
            units = constants.units[parameter.lower()]
        except KeyError:
            units = None

    if isinstance(units, dict):
        if 'unitsystem' in object_settings.keys():
            units = units[object_settings['unitsystem'].lower()]
        else:
            units = None
    return units

def ValueSum(dates, values):
    '''
    finds buzzplot targets defined and returns the flow sums
    :param dates: list of dates
    :param values: list of dicts of values @ structures
    :param target: target value
    :return: sum of values
    '''

    if isinstance(values, (list, np.ndarray)):
        return values
    sum_vals = []
    for i, d in enumerate(dates):
        sum = 0.0
        for sn in values.keys():
            # if values[sn]['elevcl'][i] == target:
            if not np.isnan(values[sn]['q(m3/s)'][i]):
                sum += values[sn]['q(m3/s)'][i]
        sum_vals.append(sum)
    return np.asarray(sum_vals)

def getObjectYears(Report, object_settings, allowIncludeAllYears=True):
    '''
    formats years settings for plots/tables. Figures out years used, if split by year, and year strings
    years is set to "ALLYEARS" if not split by year. This tells other parts of script to include all.
    :param object_settings: currently selected object settings dictionary
    :return:
        split_by_year: boolean if plots/tables should be split up year to year or all at once
        years: list of years that are used
        yearstr: list of years as strings, or set of years (ex: 2013-2016)
    '''

    split_by_year = False
    yearstr = ''
    if 'splitbyyear' in object_settings.keys():
        if object_settings['splitbyyear'].lower() == 'true':
            split_by_year = True
            years = [int(year) for year in Report.years]
            yearstr = [str(year) for year in years]
    if not split_by_year:
        yearstr = [Report.years_str]
        years = ['ALLYEARS']

    if 'yearblocks' in object_settings.keys():
        try:
            yearblocks = int(object_settings['yearblocks'])
            startyear = Report.startYear
            endyear = Report.startYear
            if yearblocks > (Report.endYear - Report.startYear + 1):
                print2stdout(f'Yearblock setting of {yearblocks} is larger than total number of years of {(Report.endYear - Report.startYear + 1)}. Not using yearblocks.', debug=Report.debug)
            else:
                while endyear < Report.endYear:
                    if endyear != Report.startYear:
                        startyear = endyear + 1
                    endyear = startyear + yearblocks - 1 #last year
                    if endyear > Report.endYear:
                        endyear = Report.endYear
                    if startyear == endyear:
                        frmtyear = startyear
                    else:
                        frmtyear = f'{startyear}-{endyear}'
                    if frmtyear not in years:
                        years.append(frmtyear)
                        yearstr.append(str(frmtyear))

        except TypeError:
            print2stdout(f"Invalid yearblock value: {object_settings['yearblocks']}", debug=Report.debug)

    if allowIncludeAllYears:
        if 'includeallyears' in object_settings.keys():
            if object_settings['includeallyears'].lower() == 'true':
                if 'ALLYEARS' not in years:
                    if len(years) > 1: #if theres only one year in here, please don't do another copy of that..
                        years.append('ALLYEARS')
                        yearstr.append(Report.years_str)

    return split_by_year, years, yearstr

def correctDuplicateLabels(linedata):
    '''
    changes the name of data internally if it is duplicated. Mostly used for comparison plots where "computed"
    may be used several times. Appends numbers to the end
    :param linedata: dictionary with settings
    :return: updated dictionary
    '''

    for line in linedata.keys():
        if 'label' in linedata[line].keys():
            curlabel = linedata[line]['label']
            if 'numtimesused' in linedata[line].keys():
                lineidx = linedata[line]['numtimesused']
                if lineidx > 0: #leave the first guy alone..
                    for otherline in linedata.keys():
                        if otherline != line:
                            if linedata[otherline]['label'] == curlabel:
                                linedata[line]['label'] = '{0} {1}'.format(curlabel, lineidx) #append the number
    return linedata

def getParameterCount(line, object_settings):
    '''
    Returns parameter used in dataset and keeps a running total of used parameters in dataset
    :param line: current line in linedata dataset
    :param object_settings: currently selected object settings dictionary
    :return:
        param: current parameter if available else None
        param_count: running count of used parameters
    '''

    if 'param_count' not in object_settings.keys():
        param_count = {}
    else:
        param_count = object_settings['param_count']

    if 'parameter' in line.keys():
        param = line['parameter'].lower()
    else:
        param = None
    if param not in param_count.keys():
        param_count[param] = 0
    else:
        param_count[param] += 1

    return param, param_count

def copyKeysBetweenDicts(to_dict, from_dict, ignore=[]):
    '''
    Copies settings from an object to an Axis. That way, users can define settings once in the main flags and have
    it cascade down to all axis, unless defined in the axis.
    :param to_dict: settings to copy to
    :param from_dict: settings to copy from
    :param ignore: list of keys to not copy
    :return: updated dictionary (to_dict)
    '''

    for key in from_dict.keys():
        if key not in ignore:
            if key not in to_dict.keys():
                to_dict[key] = from_dict[key]
    return to_dict

def getTimeInterval(times):
    '''
    attempts to find out the time interval of the time series by finding the most common time interval
    :param times: list of times
    :return:
    '''

    t_ints = []
    for i, t in enumerate(times):
        if i == 0: #skip 1
            last_time = t
        else:
            t_ints.append(t - last_time)

    return getMostCommon(t_ints)

def confirmColor(user_color, default_color, debug=False):
    '''
    confirms the color choice is valid. If not, checks to see if common mistake with spaces.
    If neither work, return a default color.
    :param user_color: desired color for line to check
    :param default_color: backup color we know works
    :return: color string
    '''

    if not is_color_like(user_color):
        if not is_color_like(user_color.replace(' ', '')):
            print2stdout('Invalid color with {0}'.format(user_color), debug=debug)
            print2stdout('Replacing with default color', debug=debug)
            return default_color
        else:
            print2stdout('Misspelling in color with {0}'.format(user_color), debug=debug)
            print2stdout('Replacing with {0}'.format(user_color.replace(' ', '')), debug=debug)
            return user_color.replace(' ', '')
    else:
        return user_color

def fixDuplicateColors(line_settings):
    '''
    when doing comparison runs, we can end up with multiple runs with the same lines set
    settings can be set to a list of colors, like linecolors instead of linecolor.
    finds the correct index for each line ,or chooses a default color
    :param line_settings: dictionary containing line settings
    :return: line settings with updated color settings
    '''

    lineusedcount = line_settings['numtimesused']
    if lineusedcount >= len(constants.def_colors):
        defcol_idx = lineusedcount%len(constants.def_colors)
    else:
        defcol_idx = lineusedcount
    if line_settings['drawline'].lower() == 'true':
        if lineusedcount > 0: #if more than one, the color specified is already used. Use a new color..
            if 'linecolors' in line_settings.keys():
                if lineusedcount > len(line_settings['linecolors']):
                    lc_idx = lineusedcount%len(line_settings['linecolors'])
                else:
                    lc_idx = lineusedcount
                try:
                    line_settings['linecolor'] = line_settings['linecolors'][lc_idx]
                except IndexError:
                    Warning('Index Error in linecolors. Using default color')
                    line_settings['linecolor'] = constants.def_colors[defcol_idx]
            else:
                line_settings['linecolor'] = constants.def_colors[defcol_idx]

        else: #case where first line, but linecolor isnt defined, but linecolorS is
            #so it used default color INSTEAD of the desired colro...
            if 'linecolors' in line_settings.keys():
                line_settings['linecolor'] = line_settings['linecolors'][0]
            elif 'linecolor' not in line_settings.keys():
                line_settings['linecolor'] = constants.def_colors[0]

    if line_settings['drawpoints'].lower() == 'true':
        if lineusedcount > 0: #if more than one, the color specified is already used. Use a new color..
            if 'pointfillcolors' in line_settings.keys():
                if isinstance(line_settings['pointfillcolors'], dict):
                    line_settings['pointfillcolors'] = [line_settings['pointfillcolors']['pointfillcolor']]
                if lineusedcount > len(line_settings['pointfillcolors']):
                    pfc_idx = lineusedcount % len(line_settings['pointfillcolors'])
                else:
                    pfc_idx = lineusedcount
                line_settings['pointfillcolor'] = line_settings['pointfillcolors'][pfc_idx]
            if 'pointlinecolors' in line_settings.keys():
                if isinstance(line_settings['pointlinecolors'], dict):
                    line_settings['pointlinecolors'] = [line_settings['pointlinecolors']['pointlinecolor']]
                if lineusedcount > len(line_settings['pointlinecolors']):
                    plc_idx = lineusedcount % len(line_settings['pointlinecolors'])
                else:
                    plc_idx = lineusedcount
                line_settings['pointlinecolor'] = line_settings['pointlinecolors'][plc_idx]

            if 'pointfillcolor' not in line_settings.keys():
                if 'pointlinecolor' in line_settings.keys():
                    line_settings['pointfillcolor'] = line_settings['pointlinecolor']
                else:
                    line_settings['pointfillcolor'] = constants.def_colors[defcol_idx]

            if 'pointlinecolor' not in line_settings.keys():
                if 'pointfillcolor' in line_settings.keys():
                    line_settings['pointlinecolor'] = line_settings['pointfillcolor']
                else:
                    line_settings['pointlinecolor'] = constants.def_colors[defcol_idx]

        else: #case where first line, but linecolor isnt defined, so it used default color...
            if 'pointfillcolors' in line_settings.keys():
                if isinstance(line_settings['pointfillcolors'], dict):
                    line_settings['pointfillcolors'] = [line_settings['pointfillcolors']['pointfillcolor']]
                line_settings['pointfillcolor'] = line_settings['pointfillcolors'][0]
            if 'pointlinecolors' in line_settings.keys():
                if isinstance(line_settings['pointlinecolors'], dict):
                    line_settings['pointlinecolors'] = [line_settings['pointlinecolors']['pointlinecolor']]
                line_settings['pointlinecolor'] = line_settings['pointlinecolors'][0]

    return line_settings

def applyXLimits(Report, dates, values, xlims):
    '''
    if the filterbylimits flag is true, filters out values outside of the xlimits
    :param dates: list of dates
    :param values: list of values
    :param xlims: dictionary of xlims, containing potentially min and/or max
    :return: filtered dates and values
    '''

    if isinstance(dates[0], (int, float)):
        wantedformat = 'jdate'
    elif isinstance(dates[0], dt.datetime):
        wantedformat = 'datetime'
    if 'min' in xlims.keys():
        datemin = WT.translateDateFormat(xlims['min'], wantedformat, Report.StartTime, Report.StartTime, Report.EndTime)
        for i, d in enumerate(dates):
            if datemin > d:
                if isinstance(values, (int, np.ndarray)):
                    values[i] = np.nan #exclude
                elif isinstance(values, dict):
                    for key in values.keys():
                        values[key][i] = np.nan
    if 'max' in xlims.keys():
        datemax = WT.translateDateFormat(xlims['max'], wantedformat, Report.EndTime,
                                     Report.StartTime, Report.EndTime)
        for i, d in enumerate(dates):
            if datemax < d:
                if isinstance(values, (int, np.ndarray)):
                    values[i] = np.nan #exclude
                elif isinstance(values, dict):
                    for key in values.keys():
                        values[key][i] = np.nan

    return dates, values

def applyYLimits(dates, values, ylims):
    '''
    if the filterbylimits flag is true, filters out values outside of the ylimits
    :param dates: list of dates
    :param values: list of values
    :param ylims: dictionary of ylims, containing potentially min and/or max
    :return: filtered dates and values
    '''

    if 'min' in ylims.keys():
        for i, v in enumerate(values):
            if float(ylims['min']) > v:
                values[i] = np.nan #exclude

    if 'max' in ylims.keys():
        for i, v in enumerate(values):
            if float(ylims['max']) < v:
                values[i] = np.nan #exclude

    return dates, values

def getGateOperationTimes(gatedata):
    '''
    gets times when gates are operational
    :param gatedata: dictionary containing gate data
    :return: list of dates with dates operational
    '''

    operationIndex = np.array([], dtype=int)
    for gatelevel in gatedata.keys():
        gate0 = list(gatedata[gatelevel]['gates'].keys())[0]
        gateops_datamask = np.zeros(len(gatedata[gatelevel]['gates'][gate0]['values']), dtype=bool) #assume everything closed
        for gi, gate in enumerate(gatedata[gatelevel]['gates']):
            curgate = gatedata[gatelevel]['gates'][gate]
            msk = ~np.isnan(curgate['values'])
            gateops_datamask = gateops_datamask | msk #change when differnt

        operationIndex = np.append(operationIndex, np.where(gateops_datamask[:-1] != gateops_datamask[1:])[0])

    return curgate['dates'][np.unique(operationIndex)]

def matcharrays( array1, array2):
    '''
    iterative recursive function that aims to line up arrays of different lengths. Takes in variable input so that
    if there are lists of lists with a single date (aka profiles), alligns those so each elevation value has a date
    assigned to it for easy output
    :param array1: np.array or list of values, generally values
    :param array2: np.array or list of values, generally dates
    :return: array1 with correct length
    '''

    if isinstance(array1, (list, np.ndarray)) and isinstance(array2, (list, np.ndarray)):
        if len(np.asarray(array1, dtype=object).shape) < len(np.asarray(array2, dtype=object).shape):
            if len(array1) == 0:
                new_array1 = np.full_like(array2, fill_value=np.nan)
            else:
                new_array1 = np.array([])
                for i, ar2 in enumerate(array2):
                    new_array1 = np.append(new_array1, np.asarray([array1[i]] * len(ar2)))
            return new_array1
        #if both are lists..
        elif len(array1) < len(array2):
            '''
            either ['Date1', 'Date2'], ['1,2,3'] OR ['Date1'], [1,2,3] OR ['DATE1'], [[1,2,3], [1,2,3,4]]
             OR ['Date1', 'Date2'], [[1,2,3], [2,4,5],[6,
             or [], [1,2,3]
            scenario 1: shouldnt ever happen
            scenario 2: do Date1 for each item in array2
            scenario 3: do date1 for each value in each subarray in 2 '''

            if len(array1) == 1: #solo date
                new_array1 = []
                for subarray2 in array2:
                    new_array1.append(matcharrays(array1[0], subarray2))
                return new_array1
            elif len(array1) == 0: #no data
                new_array1 = []
                for subarray2 in array2:
                    new_array1.append(matcharrays('', subarray2))
                return new_array1

            else:
                print2stdout('ERROR') #If the Len of the arrays are offset, then there should only ever be 1 date
        elif len(array1) == len(array2):
            new_array1 = []
            for i, subarray1 in enumerate(array1):
                new_array1.append(matcharrays(subarray1, array2[i]))
            return new_array1
        else:
            print2stdout('Array 1 is bigger than array2')
            print2stdout(len(array1))
            print2stdout(len(array2))
            new_array1 = []
            for i in range(len(array2)):
                new_array1.append(array1[i])
            return new_array1

    #GOAL LOOP
    elif isinstance(array1, (str, dt.datetime, int, float)) and isinstance(array2, (list, np.ndarray)):
        # array1 is a single value, array2 is a list of values
        new_array1 = []
        for subarray2 in array2:
            if isinstance(subarray2, (list, np.ndarray)):
                new_array1.append(matcharrays(array1, subarray2))
            else:
                new_array1.append(array1)
        return new_array1

    else:
        return array1

def pickByParameter(values, line):
    '''
    some data (W2) has multiple parameters coming from a single results file, we can't know which one we want at
    the moment. This grabs the right parameter based on input
    :param values: dictionary of values
    :param line: dictionary of line settings
    :return:
        values: list of values
    '''

    w2_param_dict = {'temperature': 't(c)',
                     'elevation': 'elevcl',
                     'flow': 'q(m3/s)'}

    if 'parameter' not in line.keys():
        print2stdout("Parameter not set for line.")
        print2stdout("using the first set of values, {0}".format(list(values.keys())[0]))
        return values[list(values.keys())[0]]
    else:
        if line['parameter'].lower() not in w2_param_dict.keys():
            print2stdout('Parameter {0} not found in dict in pickByParameter(). {1}'.format(line['parameter'].lower(), w2_param_dict.keys()))
            print2stdout("using the first set of values, {0}".format(list(values.keys())[0]))
            return values[list(values.keys())[0]]
        else:
            p = line['parameter'].lower()
            param_key = w2_param_dict[p]
            return values[param_key]

def prioritizeKey(firstchoice, secondchoice, key, backup=None):
    '''
    looks for a key in two sets of settings. If the key is in first choice, use that one. then check the second choice.
    if it neither, just use backup.
    :param firstchoice: dictionary
    :param secondchoice: dictionary
    :param key: settings key to look for in dictionaries
    :param backup: backup value if in neither.
    :return: setting from prioritized dictionary
    '''

    if key in firstchoice:
        return firstchoice[key]
    elif key in secondchoice:
        return secondchoice[key]
    else:
        return backup

def getListItems(listvals):
    '''
    recursive function to convert lists of lists into single lists for logging
    :param listvals: value object
    :return: list of values
    '''

    if isinstance(listvals, (list, np.ndarray)):
        outvalues = np.array([])
        for item in listvals:
            if isinstance(item, (list, np.ndarray)):
                vals = getListItems(item)
                for v in vals:
                    outvalues = np.append(outvalues, v)
                    # outvalues.append(v)
            else:
                return listvals #we just have a list of values, so we're good! return list
    elif isinstance(listvals, dict):
        outvalues = getListItemsFromDict(listvals)
    return outvalues

def cleanFileName(csvname):
    '''
    removes and replaces invalid characters in filenames with underscores
    :param csvname: potential name of file
    :return: sanitized file name
    '''

    pattern = r'[^\w\-_\. ]'
    # replace invalid characters with underscores
    sanitized_file_name = re.sub(pattern, '_', csvname)
    return sanitized_file_name

def getListItemsFromDict(indict):
    '''
    recursive function to convert dictionary of lists into single dictionary for logging. Keys are determined
    using original keys
    :param indict: value dictionary object
    :return: dictionary of values
    '''

    outdict = {}
    for key in indict:
        if isinstance(indict[key], dict):
            returndict = getListItemsFromDict(indict[key])
            returndict = {'{0}_{1}'.format(key, newkey): returndict[newkey] for newkey in returndict}
            for key in returndict.keys():
                outdict[key] = returndict[key]
        elif isinstance(indict[key], (list, np.ndarray)):
            outdict[key] = indict[key]
    return outdict

def NaNOmittedValues(values, omitval, debug):
    '''
    replaces a specified value in time series. Can be variable depending on data source (-99999, 0, 100, etc)
    :param values: array of values
    :param omitval: value to be omitted
    :return: new values
    '''

    if isinstance(values, dict):
        new_values = {}
        for key in values:
            new_values[key] = NaNOmittedValues(values[key], omitval, debug)
        return new_values
    else:
        if len(values) > 0:
            # Count the number of decimals after the value
            count_after_decimal = str(omitval)[::-1].find('.')

            # Add a check for a .0 in the value. If this occurs, treat the value like an integer and set the rounding to 0
            if '.0' in str(omitval) and count_after_decimal == 1:
                count_after_decimal = 0

            # Determine which values are within the rounding difference of the omitted value
            o_msk = np.where(np.round(values, count_after_decimal) == omitval)

            # Mask omitted values and set to nan
            values[o_msk] = np.nan

            # Convert back into an array
            new_values = np.asarray(values)

            # Log that some values have been omitted based on the filter
            print2stdout('Omitted {0} values of {1}'.format(len(o_msk[0]), omitval), debug=debug)

            # Return the updated values to the calling function
            return new_values
        else:
            print2stdout('No Values to omit.', debug=debug)
            return values

def replaceDefaults(Report, default_settings, object_settings):
    '''
    makes deep copies of default and defined settings so no settings are accidentally carried over
    replaces flagged values (%%) with easily identified variables
    iterates through settings and replaces all default settings with defined settings
    :param default_settings: default object settings dictionary
    :param object_settings: user defined settings dictionary
    :return:
        default_settings: dictionary of user and default settings
    '''

    default_settings = pickle.loads(pickle.dumps(replaceflaggedValues(Report, default_settings, 'general'), -1))
    object_settings = pickle.loads(pickle.dumps(replaceflaggedValues(Report, object_settings, 'general'), -1))
    replaced_flags = []
    for key in object_settings.keys():
        if key not in default_settings.keys(): #if defaults doesnt have key
            default_settings[key] = object_settings[key]
            replaced_flags.append(key)
        elif default_settings[key] == None: #if defaults has key, but is none
            default_settings[key] = object_settings[key]
            replaced_flags.append(key)
        elif isinstance(object_settings[key], list): #if settings is a list, aka rows or lines
            # if key.lower() == 'rows': #if the default has rows defined, just overwrite them.
            if key in default_settings.keys():
                default_settings[key] = object_settings[key]
                replaced_flags.append(key)
            elif key.lower() not in default_settings.keys():
                default_settings[key] = object_settings[key] #if the defaults dont have anything defined, fill it in
                replaced_flags.append(key)
        else:
            default_settings[key] = object_settings[key]
            replaced_flags.append(key)

    default_settings['replaced_defaults'] = replaced_flags
    return default_settings

def getDateSourceFlag(object_settings):
    '''
    Gets the datesource from object settings
    :param object_settings: currently selected object settings dictionary
    :return: datessource_flag
    '''

    if 'datessource' in object_settings.keys():
        datessource_flag = object_settings['datessource'] #determine how you want to get dates? either flag or list
    else:
        datessource_flag = [] #let it make timesteps

    return datessource_flag

def getMaxWSEFromElev(input_data):
    '''
    gets the max elevations for a timeseries
    :param input_data: list of elevations over timeseries
    :return: list of WSE
    '''

    try:
        return max(input_data)
    except ValueError:
        return np.nan
    # elevations = []
    # for e in input_data:
    #     elevations.append(max(e))
    # return elevations

def formatUnitsStrings(units, format='internal'):
    '''
    cleans strings and formats units
    :param units: string of units
    :param format: internal or eternal, determines which flags to use
    :return: formated units
    '''

    if units == None:
        return units
    if format == 'internal':
        units_list = constants.units_fancy_flags_internal
    elif format == 'external':
        units_list = constants.units_fancy_flags_external

    if units.lower() in units_list.keys():
        output = units_list[units.lower()]
    else:
        output = units
    return output

def formatTextFlags(text):
    flags = {"\\n": "\n"
             # "\\t": "\t", #replaces with square for laTex reasons
             # "\\r": "\r", #replaces with square for laTex reasons
             }
    for key, fixed in flags.items():
        text = text.replace(key, fixed)
    return text

def formatMembers(member):
    '''
    format members to have DSS notation of 6 characters with leading 0's
    :param member: single member or list of members, or regex match
    :return: formatted member
    '''

    if isinstance(member, re.Match):
        return member.group(1).zfill(6)
    elif isinstance(member, (np.ndarray, list)):
        frmted_members = []
        for me in member:
            frmted_members.append(str(me).zfill(6))
        return frmted_members
    else:
        return str(member).zfill(6)

def matchMemberToEnsembleSet(ensemblesets, member):
    '''
    finds the ensemble set a member belongs to
    :param ensemblesets: collection of ensemble sets
    :param member: member to get set for
    :return: selected ensemble set
    '''

    for ensembleset in ensemblesets:
        if member in ensembleset['members']:
            return ensembleset
    return {}


def getOriginalMemberNumber(member, ensembleset, s_dss_file, s_f_part, o_start_time, o_end_time, b_debug):
    """
    Gets the original member number based on the ensemble set and current member number. This will provide the correct schedule number instead of the collection start plus the schedule number.
    Parameters
    ----------
    Report: object
        Report generator object
    member: int
        member number that includes the collection
    ensembleset: dict
        Ensemble set that contains the memeber
    s_dss_file:str
        Path to DSS file with simulation data
    s_f_part: str
        Alternative f-part in DSS file to use
    o_start_time: datetime object
        Start date of simulation
    o_end_time: datetime object
        End date of simulation
    b_debug: bool
        Flag set in report, needed for DSS reader

    Returns
    -------
    s_original_member: str
        Original member number, as a string
    """

    # Check if there is a final schedule number in the dss file. This will only happen for the iterative W2 model
    # the path where the schedule number would be if it exists
    s_final_schedule_path = f"//W2_FOLSOM_SCHEDULE_FINAL/Count/01Apr2024/1Hour/{s_f_part}/"

    # try and pull the values
    times, values, units = WR.readDSSData(s_dss_file, s_final_schedule_path, o_start_time, o_end_time, b_debug)

    # if the values array is non-empty return the first non nan value
    if len(values) > 0:
        return str(int(values[~np.isnan(values)][0]))


    # get the index where the current member number is
    member_index = ensembleset['members'].index(member)

    # split the original members into a list
    sl_original_members = ensembleset['memberstoreport'].split(', ')

    # get the original member number
    s_original_member = sl_original_members[member_index]

    return s_original_member


def formatNumbers(number, numberformatsettings):
    '''
    formats numbers to use the correct amount of decimal places
    :param number: number to be formatted
    :param numberformatsettings: settings to format from
    :return: formatted number
    '''

    try:
        number = float(number)
    except:
        return number
    if np.isnan(number):
        return number

    for numberformat in numberformatsettings:
        if 'decimalplaces' in numberformat.keys():
            decplaces = int(numberformat['decimalplaces'])
            if 'max' in numberformat.keys() and 'min' in numberformat.keys():
                if float(numberformat['min']) < abs(number) <= float(numberformat['max']):
                    # print2stdout(f'Number {number} with settings {numberformat}')
                    return '{num:,.{digits}f}'.format(num=number, digits=decplaces)

            elif 'max' in numberformat.keys() and 'min' not in numberformat.keys():
                if abs(number) <= float(numberformat['max']):
                    return '{num:,.{digits}f}'.format(num=number, digits=decplaces)

            elif 'min' in numberformat.keys() and 'max' not in numberformat.keys():
                if float(numberformat['min']) < abs(number):
                    return '{num:,.{digits}f}'.format(num=number, digits=decplaces)
            else:
                return '{num:,.{digits}f}'.format(num=number, digits=decplaces)

    return f'{number:,.2f}'

def replaceAllFlags(Report, text):
    '''
    Replaces all flags in text with the correct values for text boxes
    :param Report: wat report generator main object
    :param text: text to parse
    :return: updated string
    '''

    text = replaceflaggedValues(Report, text, 'fancytext', forjasper=True) #these are text formatted and dont matter
    text = replaceflaggedValues(Report, text, 'general', forjasper=True) #these should be the same for ALL IDs

    starting_ID = Report.currentlyloadedID
    flag_objects = list(set(re.findall(r'%%(.*?)%%', text)))
    for fo in flag_objects:
        original_str = '%%{0}%%'.format(fo)

        if not Report.modelIndependent: #need to make sure its model independent and there are IDs to load
            if len(fo.split('.')) > 1:  # if its longer than 2 then its wanting a specific ID
                wanted_ID = fo.split('.')[-1]
                if Report.currentlyloadedID != wanted_ID and wanted_ID in Report.All_IDs:
                    Report.loadCurrentID(wanted_ID)
                if wanted_ID in Report.All_IDs: #if it is an ID flag, we want to trim it off
                    fo = '.'.join(fo.split('.')[:-1])
            flagged_text = '%%{0}%%'.format(fo) #format it back like its a flag, but we've got the right alts loaded

            flagged_text = replaceflaggedValues(Report, flagged_text, 'modelspecific', forjasper=True)
            flagged_text = formatDescriptionsForPrint(Report, flagged_text)

            text = text.replace(original_str, flagged_text)

            if starting_ID != Report.currentlyloadedID:
                Report.loadCurrentID(starting_ID)

        else:
            flagged_text = '%%{0}%%'.format(fo)
            flagged_text = formatDescriptionsForPrint(Report, flagged_text)
            text = text.replace(original_str, flagged_text)

    return text

def formatDescriptionsForPrint(Report, text):
    '''
    formats descriptions for printing
    :param text: text to be formatted
    :return: formatted text
    '''

    #format should be %%description.object%% where object is the object to get the description from
    desc_objects = list(set(re.findall(r'%%description\.(.*?)%%', text)))
    for do in desc_objects:
        do_low = do.lower()
        desciption_str = '%%description.{0}%%'.format(do)
        desc = getDescription(Report, do_low)
        desc = formatDescription(desc)
        text = text.replace(desciption_str, desc)
    return text

def formatDescription(description):
    '''
    formats description for printing
    :param description: description to be formatted
    :return: formatted description
    '''
    if description == None:
        return ''

    desc_split = description.split('\n')#handle newlines
    desc_list = []
    for item in desc_split:
        desc_list.append(item.strip())
    description = '\n'.join(desc_list)

    return formatTextFlags(description)

def getDescription(Report, do):
    '''
    gets the description from the report object
    :param do: object to get description from
    :return: description
    '''

    # starting_ID = Report.currentlyloadedID
    # if len(do.split('.')) > 1: #if its longer than 2 then its wanting a specific ID
    #     wanted_ID = do.split('.')[-1]
    #     if starting_ID != wanted_ID:
    #         Report.loadCurrentID(wanted_ID)
    #     do = '.'.join(do.split('.')[:-1])

    desc = ''

    if do == 'study':
        desc = Report.description
    elif do == 'simulationgroup':
        desc = Report.SimulationGroup['Description']
    elif do == 'simulation':
        # if Report.reportType == 'validation':
        desc = Report.SimulationDescription
        # else:
        #     desc = '' #TODO add comparison and forecast
    elif do == 'watalternative':
        desc = Report.WatAlternative['Description']
    elif do == 'analysisperiod':
        desc = Report.AnalysisPeriod['Description']

    elif do == 'modelalternative':
        desc = Report.ModelAltDescription

    else:
        print2stderr('No description found for object: {0}'.format(do))
        desc = ''

    # if starting_ID != Report.currentlyloadedID:
    #     Report.loadCurrentID(starting_ID)

    return desc


def checkJasperFiles(study_dir, install_dir):
    '''
    checks existing jasper and jrxml files, and if the jrxml files are newer, deletes jasper files so they can be regen
    :param study_dir: directory to look for jasper files
    '''

    #JRXML files can exist in two places, study and install. Study overwrites install.
    jrxml_study_directory = os.path.join(study_dir, 'reports', 'Jasper')
    jrxml_install_directory = os.path.join(install_dir, 'reports', 'Jasper')

    if os.path.exists(jrxml_study_directory): #if the study dir exsits
        files_in_study_directory = os.listdir(jrxml_study_directory) #then get files in study dir
    else:
        files_in_study_directory = [] #otherwise, there are none

    if os.path.exists(jrxml_install_directory): #then check the install dir
        files_in_install_directory = os.listdir(jrxml_install_directory) #get install dir files
    else:
        files_in_install_directory = [] #otherwise there are none

    jrxml_study_files = [file for file in files_in_study_directory if file.endswith('.jrxml')] #get jrxml files
    jrxml_install_files = [file for file in files_in_install_directory if file.endswith('.jrxml')] #default included in install

    for jrxml_file in jrxml_install_files: #should contain ALL files, as this is the base set
        jasper_file = os.path.join(study_dir, 'reports', 'JasperC', jrxml_file.split('.jrxml')[0] + '.jasper') #link to where compiled jasper file would be

        if jrxml_file in jrxml_study_files: #if the jrxml file is in the study dir, use that one
            jrxml_source = study_dir
        else: #otherwise use the one in the study fir
            jrxml_source = install_dir

        if os.path.exists(jasper_file):
            jrxml_time = os.path.getmtime(os.path.join(jrxml_source, 'reports', 'Jasper', jrxml_file))
            jasper_time = os.path.getmtime(jasper_file)

            if jrxml_time > jasper_time: #if the jasper if older than the jrxml
                print2stdout(f'\nNewer JRXML file detected for {jrxml_file}')
                print2stdout(f'Deleting {jasper_file}')
                os.remove(jasper_file)

def filterByMember(values, members):
    '''
    filters values by member
    :param values: dictionary of values
    :param members: list of members
    :return: filtered values
    '''

    filtered_values = {}

    for member in members:
        try:
            membervalues = values[member]
        except KeyError:
            print2stderr(f'Member {member} not found in values')
            continue
        filtered_values[member] = membervalues

    return filtered_values

def checkForCollections(data_settings):
    '''
    checks the data_settings for the collection flag
    :param data_settings: dictionary of settings for data
    :return: boolean
    '''

    for ds in data_settings.keys():
        if 'collection' in data_settings[ds].keys():
            if data_settings[ds]['collection']:
                return True
    return False

def organizePlotYears(object_settings):
    '''
    organizes years and year strings for report objects
    :param object_settings: settings to parse for year information
    :return: years, year strings
    '''

    if 'years' in object_settings.keys():
        years = []
        if 'yearstr' in object_settings.keys():
            yrstrs = []
            _isyrstr = True
        for yi, year in enumerate(object_settings['years']):
            if year == 'ALLYEARS':
                years.append(year)
                if _isyrstr:
                    yrstrs.append(object_settings['yearstr'][yi])
        for yi, year in enumerate(object_settings['years']):
            if isinstance(year, str):
                if '-' in year:
                    years.append(year)
                    if _isyrstr:
                        yrstrs.append(object_settings['yearstr'][yi])
        for yi, year in enumerate(object_settings['years']):
            if year != 'ALLYEARS':
                years.append(year)
                if _isyrstr:
                    yrstrs.append(object_settings['yearstr'][yi])
        if _isyrstr:
            return years, yrstrs
        else:
            return years, []
    return [], []

def sanitizeText(intext):
    '''
    cleans incoming text
    :param intext: text to be cleaned
    :return: clean text :)
    '''

    return str(intext).replace('.','').replace(' ', '').replace(':', '').replace("_","")

def calculateStorageFromElevation(values, curline):
    '''
    reads in elevation storage area file, and interpolates across it, then calcs storage based off of elevation vals
    :param values: elevation values
    :param curline: current line with info about file
    :return: storage interp values
    '''
    
    elevation_storage_area_file = curline['elevation_storage_area_file']
    elev_stor_area = np.loadtxt(elevation_storage_area_file, delimiter=',')
    elevstorcurve = interpolate.interp1d(elev_stor_area[:, 0], elev_stor_area[:, 1], bounds_error=False, fill_value=np.nan)
    return elevstorcurve(values)

def ReplaceListAtIdx(list, idx, replacevalue):
    '''
    replaces value in a list at a specified index
    :param list: list of values
    :param idx: index to replace
    :param replacevalue: value to replace with
    :return: corrected list
    '''

    list[idx] = replacevalue
    return list