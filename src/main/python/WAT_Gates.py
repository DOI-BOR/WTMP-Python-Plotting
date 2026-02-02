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

import WAT_Reader as WR

def getGateBlendDays(gateconfig, gatedata, timestamp):
    '''
    calculates gate blend days for a given timestamp. Gate blend days are the amount of days in the simulation where
    the combination of currently open gates have been open
    :param gateconfig: dictionary containing settings for gate configurations
    :param gatedata: dictionary containing information for gates
    :param timestamp: current timestamp to configure to
    :return: decimal days number
    '''

    if len(gateconfig) == 0:
        return 'N/A'

    gd_key = list(gatedata.keys())[0]
    curgate = gatedata[gd_key]['gates'][list(gatedata[gd_key]['gates'].keys())[0]]
    idx = WR.getClosestTime([timestamp], curgate['dates'])[0]
    datamask = np.ones(idx+1, dtype=bool)

    for gatelevel in gatedata.keys():
        for gatenumber in gatedata[gatelevel]['gates'].keys():
            current_op = gateconfig[gatelevel][gatenumber]
            if np.isnan(current_op):
                msk = np.isnan(gatedata[gatelevel]['gates'][gatenumber]['values'][:idx+1])
            else:
                msk = ~np.isnan(gatedata[gatelevel]['gates'][gatenumber]['values'][:idx+1])
            datamask = datamask & msk

    changeop = False
    for i in reversed(range(idx)):
        if not datamask[i]:
            changeop = True
            break
    timestep = (curgate['dates'][1] - curgate['dates'][0]).total_seconds() / 86400
    if changeop:
        decdays = (idx - i -1) * timestep
    else:
        decdays = idx * timestep

    return round(decdays, 3)

def getGateConfigurationDays(gateconfig, gatedata, timestamp):
    '''
    calculates gate configuration days for a given timestamp. Gate configuration days are the amount of days in the
    simulation where the configuration of currently open gates have been open
    :param gateconfig: dictionary containing settings for gate configurations
    :param gatedata: dictionary containing information for gates
    :param timestamp: current timestamp to configure to
    :return: decimal days number
    '''

    if len(gateconfig) == 0:
        return 'N/A'

    gd_key = list(gatedata.keys())[0]
    curgate = gatedata[gd_key]['gates'][list(gatedata[gd_key]['gates'].keys())[0]]
    idx = WR.getClosestTime([timestamp], curgate['dates'])[0]
    datamask = np.ones(idx+1, dtype=bool)

    for gatelevel in gatedata.keys():
        current_op_level = np.nan
        for gatenumber in gatedata[gatelevel]['gates'].keys():
            current_op = gateconfig[gatelevel][gatenumber]
            if not np.isnan(current_op):
                current_op_level = 1
                break

        datamask_gateLevel = np.zeros(idx+1, dtype=bool)
        for gatenumber in gatedata[gatelevel]['gates'].keys():
            msk = ~np.isnan(gatedata[gatelevel]['gates'][gatenumber]['values'][:idx+1]) #true when open
            datamask_gateLevel = datamask_gateLevel | msk

        if np.isnan(current_op_level): #if closed..
            datamask = datamask & ~datamask_gateLevel
        else:
            datamask = datamask & datamask_gateLevel #datamsk_gateLevel if true when open

    changeop = False
    for i in reversed(range(idx)):
        if not datamask[i]:
            changeop = True
            break
    timestep = (curgate['dates'][1] - curgate['dates'][0]).total_seconds() / 86400
    if changeop:
        decdays = (idx - i -1) * timestep
    else:
        decdays = idx * timestep

    return round(decdays, 3)
