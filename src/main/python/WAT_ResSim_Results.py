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

import os, sys
import numpy as np
import datetime as dt
import h5py

import WAT_Functions as WF
import WAT_Time as WT

class ResSim_Results(object):

    def __init__(self, simulationPath, alternativeName, starttime, endtime, Report, external=False):
        '''
        Class Builder init. Controls results and pathing for ResSim model results
        :param simulationPath: full path to ResSim simulation
        :param alternativeName: Name of selected Ressim Alternative run
        :param starttime: start time datetime object
        :param endtime: end time datetime object
        '''

        self.simulationPath = simulationPath
        self.alternativeName = alternativeName.replace(':', '_')
        self.starttime = starttime
        self.endtime = endtime
        self.external = external
        self.Report = Report

        if not self.external:
            self.getH5File()
            self.load_time() #load time vars from h5
            self.loadSubdomains()

    def getH5File(self):
        '''
        build h5 file name and open file
        :return: class variable
                    self.h
        '''

        h5filefrmt = self.alternativeName.replace(' ', '_')
        self.h5fname = os.path.join(self.simulationPath, 'rss', h5filefrmt + '.h5')
        self.openH5File(self.h5fname)

    def openH5File(self, h5fname):
        '''
        opens hdf5 file instance
        :param h5fname: filepath to hdf5 file
        '''

        if os.path.exists(h5fname):
            self.h = h5py.File(h5fname, 'r')
        else:
            WF.print2stderr(f'ERROR: missing results file {h5fname}')
            sys.exit(1)

    def load_time(self):
        '''
        Loads times from H5 file
        :return: set class variables
                    self.tstr - time string
                    self.t - times list
                    self.nt - num times
                    self.t_offset - time offset
        '''

        self.tstr = self.h['Results/Subdomains/Time Date Stamp']
        tstr0 = (self.tstr[0]).decode("utf-8")
        ttmp = self.h['Results/Subdomains/Time']
        jd_dates = ttmp[:]
        try:
            ttmp = dt.datetime.strptime(tstr0, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstr0.replace('24:00', '23:00')
            ttmp = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp += dt.timedelta(hours=1)
        dt_dates = []
        t_offset = ttmp.toordinal() + float(ttmp.hour) / 24. + float(ttmp.minute) / (24. * 60.)
        for t in jd_dates:
            dt_dates.append(ttmp + dt.timedelta(days=t))
        self.nt = len(dt_dates)

        self.dt_dates = np.asarray(dt_dates)
        self.jd_dates = np.asarray(jd_dates)
        self.t_offset = t_offset

    def loadSubdomains(self):
        '''
        creates a dictionary of all subdomains in the H5 file and grabes their XY coordinates for later reference
        :return: set class variables
                    self.subdomains
        '''

        self.subdomains = {}
        group = self.h['Geometry/Subdomains']
        for subdomain in group:
            dataset = self.h['Geometry/Subdomains/{0}/Cell Center Coordinate'.format(subdomain)]
            ncells = (np.shape(dataset))[0]
            x = np.array([dataset[i][0] for i in range(ncells)])
            y = np.array([dataset[i][1] for i in range(ncells)])
            z = np.array([dataset[i][2] for i in range(ncells)])
            self.subdomains[subdomain] = {'x': x, 'y': y, 'z': z}

    def readProfileData(self, resname, metric, timestamps):
        '''
        reads Ressim profile data from model
        :param resname: name of reservoir in h5 file
        :param metric: metric of data to be extracted
        :param times: timesteps to grab
        :return: vals, elevations, depths
        '''

        self.loadElevation(alt_subdomain_name=resname)

        if self.subdomain_read_success:

            vals = []
            elevations = []
            depths = []
            times = []
            # WF.print2stdout('UNIQUE TIMES:', unique_times)
            if isinstance(timestamps, (list, np.ndarray)):
                unique_times = [n for n in timestamps]
                for j, time_in in enumerate(unique_times):
                    timestep = WT.getIdxForTimestamp(self.dt_dates, time_in)
                    if timestep == -1:
                        depths.append(np.asarray([]))
                        elevations.append(np.asarray([]))
                        vals.append(np.asarray([]))
                        times.append(time_in)
                        # continue
                    else:
                        # WF.print2stdout('finding time for', time_in)
                        time_to_grab = self.dt_dates[timestep]
                        self.loadResults(time_to_grab, metric.lower(), alt_subdomain_name=resname)
                        ktop = self.getTopLayer(timestep) #get waterlevel top layer to know where to grab data from
                        v_el = self.vals[:ktop + 1]
                        el = self.elev[:ktop + 1]
                        d_step = []
                        e_step = []
                        v_step = []
                        for ei, e in enumerate(el):
                            d_step.append(np.max(el) - e)
                            e_step.append(e)
                            v_step.append(v_el[ei])
                        depths.append(np.asarray(d_step))
                        elevations.append(np.asarray(e_step))
                        vals.append(np.asarray(v_step))
                        times.append(time_in)
            else:
                self.loadResults('all', metric.lower(), alt_subdomain_name=resname)
                elevations = self.elev
                vals = np.asarray(self.vals)
                depths = np.array([])
                times = self.dt_dates

            return vals, elevations, depths, np.asarray(times), self.units

        else:
            return [], [], [], [], None

    def getTopLayer(self, timestep_index):
        '''
        grabs the top active layer of water for a given timestep
        :param timestep_index: timestep index to grab data at
        :return: returns index for top layer of water column
        '''

        elev = self.elev_ts[timestep_index] #elevations at a timestep
        for k in range(len(self.elev) - 1): #for each cell..
            cell_z = self.elev[k]  # layer midpoint
            cell_z1 = self.elev[k + 1]  # layer above midpoint
            top_of_cell_z = 0.5 * (cell_z + cell_z1)
            if elev < top_of_cell_z:
                break
        return k

    def loadElevation(self, alt_subdomain_name=None):
        '''
        loads elevations from the H5 file
        :param alt_subdomain_name: alternate field if the domain is not class defined subdomain name
        :return: set class variables
                    self.ncells - number of cells
                    self.elev - elevation time series for profile
                    self.elev_ts - elevation time series
        '''

        this_subdomain = self.subdomain_name if alt_subdomain_name is None else alt_subdomain_name
        subdomain_name = 'Geometry/Subdomains/' + this_subdomain + '/Cell Center Coordinate'
        if subdomain_name not in self.h.keys():
            WF.print2stdout(f"\nWARNING: Subdomain {this_subdomain} not found in results file.")
            self.elev = np.array([])
            self.elev_ts = np.array([])
            self.ncells = 0
            self.subdomain_read_success = False
        else:
            cell_center_xy = self.h[subdomain_name]
            self.ncells = (np.shape(cell_center_xy))[0]
            self.elev = np.array(cell_center_xy[:self.ncells, 2])
            elev_ts = self.h['Results/Subdomains/' + this_subdomain + '/Water Surface Elevation']
            self.elev_ts = np.array(elev_ts[:self.nt])
            self.subdomain_read_success = True

    def loadResults(self, t_in, metrc, alt_subdomain_name=None):
        '''
        loads results for a specific time step from h5 file
        :param t_in: time in datetime object
        :param metrc: metric to get data from
        :param alt_subdomain_name: alt field if data is grabbed from a different location than the default
        :return: set class vairables
                self.t_data - timestep
                self.vals - array of values
        '''

        this_subdomain = self.subdomain_name if alt_subdomain_name is None else alt_subdomain_name
        self.units = None
        if metrc.lower() == 'temperature':
            metric_name = 'Water Temperature'
            attrs = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name].attrs
            self.units = self.getUnitsFromAttrs(attrs)

            try:
                vals = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            except KeyError:
                # raise KeyError('WQ Simulation does not have results for metric: {0}'.format(metric_name))
                WF.print2stdout(f'\nWARNING: WQ Simulation does not have results for metric: {metric_name}')
                vals = []

        elif metrc == 'diss_oxy' or metrc.lower() == 'do':
            metric_name = 'Dissolved Oxygen'
            attrs = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name].attrs
            self.units = self.getUnitsFromAttrs(attrs)
            try:
                vals = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            except KeyError:
                raise KeyError('WQ Simulation does not have results for metric: {0}'.format(metric_name))

        elif metrc.lower() == 'do_sat':
            metric_name = 'Water Temperature'
            vt = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            metric_name = 'Dissolved Oxygen'
            vdo = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            vals = WF.calcComputedDOSat(vt, vdo, self.Report.Constants.satDO_interp)
            self.units = '%'

        elif metrc.lower() in ['elevation', 'wse']:
            self.loadElevation(alt_subdomain_name=this_subdomain)
            attrs = self.h['Results/Subdomains/' + this_subdomain + '/Water Surface Elevation'].attrs
            self.units = self.getUnitsFromAttrs(attrs)
            vals = self.elev

        elif metrc.lower() == 'outflow':
            metric_name = 'Total boundary outflow'
            try:
                vals = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
                attrs = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name].attrs
                self.units = self.getUnitsFromAttrs(attrs)
            except KeyError:
                # raise KeyError('WQ Simulation does not have results for metric: {0}'.format(metric_name))
                WF.print2stdout(f'\nWARNING: WQ Simulation does not have results for metric: {metric_name}')
                self.units = None
                vals = []

        if t_in != 'all':
            timestep = WT.getIdxForTimestamp(self.dt_dates, t_in) #get timestep index for current date
            if timestep == -1:
                WF.print2stdout('should never be here..')
            self.t_data = t_in
            self.vals = np.array([vals[timestep][i] for i in range(self.ncells)])
        else:
            self.vals = vals

    def getUnitsFromAttrs(self, attrs):

        units = None
        if 'Units' in attrs.keys():
            u = attrs['Units']
        elif 'units' in attrs.keys():
            u = attrs['units']

        try:
            u_i = u[0]
        except IndexError:
            return None

        if isinstance(u_i, np.bytes_):
            units = u_i.decode('utf-8')
        else:
            u_i = units

        return units

    def readTimeSeries(self, metric, x, y, subdomain=None):
        '''
        Gets Time series values from Ressim H5 files.
        :param metric: metric of data
        :param x: X coordinate of the observed data to be passed into self.find_computed_station_cell(xy) to find
                    location of modeled data
        :param y: Y coordinate of the observed data to be passed into self.find_computed_station_cell(xy) to find
                    location of modeled data
        :return: times and values arrays for selected metric and time window
        '''

        units = None
        i, subdomain_name = self.findComputedStationCell(x, y, subdomain=subdomain)
        if subdomain_name == None:
            WF.print2stdout(f'XY coords ({x}, {y}) not found', debug=self.Report.debug)
            return [], [], units

        accepted_metrics = ['flow', 'elevation', 'temperature', 'do', 'do_sat']
        metric = metric.lower().replace('_', '').replace(' ', '')
        if metric in ['flow']:
            dataset_name = 'Cell flow'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            attrs = dataset.attrs
            units = self.getUnitsFromAttrs(attrs)
            v = np.array(dataset[:, i])
            v = WF.cleanComputed(v)
        elif metric in ['elevation', 'wse', 'waterlevel', 'watersurfaceelevation']:
            dataset_name = 'Water Surface Elevation'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            attrs = dataset.attrs
            units = self.getUnitsFromAttrs(attrs)
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric in ['temperature', 'temp', 'tempc', 'tempcelsius', 'watertemperature']:
            dataset_name = 'Water Temperature'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            attrs = dataset.attrs
            units = self.getUnitsFromAttrs(attrs)
            v = np.array(dataset[:, i])
            v = WF.cleanComputed(v)
        elif metric in ['do', 'dissolvedoxygen']:
            dataset_name = 'Dissolved Oxygen'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            attrs = dataset.attrs
            units = self.getUnitsFromAttrs(attrs)
            v = np.array(dataset[:, i])
            v = WF.cleanComputed(v)
        elif metric in ['dosat', 'saturateddo', 'satdo', 'dissolvedoxygensaturation']:
            dataset_name = 'Water Temperature'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            vt = np.array(dataset[:, i])
            dataset_name = 'Dissolved Oxygen'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            units = '%'
            vdo = np.array(dataset[:, i])
            vt = WF.cleanComputed(vt)
            vdo = WF.cleanComputed(vdo)
            v = WF.calcComputedDOSat(vt, vdo, self.Report.Constants.satDO_interp)
        else:
            WF.print2stdout(f'ERROR: Metric {metric} not in accepted metrics list: {accepted_metrics}')
            return [], [], units

        if not hasattr(self, 't_computed'):
            self.loadComputedTime()
        istart = 0
        iend = -1
        return self.t_computed[istart:iend], v[istart:iend], units

    def readProfileTopwater(self, resname, timestamps):
        '''
        gets the WSE for each timestep to filter for profile contour plots
        :param resname: name of reservoir to get from H5 file
        :param timestamps: list of timesteps, or 'all'
        :return: list of WSE
        '''

        self.loadElevation(alt_subdomain_name=resname)

        if self.subdomain_read_success:

            if isinstance(timestamps, (list, np.ndarray)):
                topwater = []
                unique_times = [n for n in timestamps]
                for j, time_in in enumerate(unique_times):
                    timestep = WT.getIdxForTimestamp(self.dt_dates, time_in)
                    if timestep == -1:
                        topwater.append(np.nan)
                        # continue
                    else:
                        topwater.append(self.elev_ts[timestep])
            else:
                topwater = self.elev_ts[:]
            return np.asarray(topwater)
        else:
            return []

    def checkSubdomain(self, subdomain_name):
        '''
        checks to see if subdomain exists in model results
        :param subdomain_name: name of subdomain to check
        :return:
        '''

        dataset = 'Results/Subdomains/{0}'.format(subdomain_name)
        if dataset not in self.h.keys():
            return False
        else:
            return True

    def readSubdomain(self, metric, subdomain_name):
        '''
        reads the subdomain data from h5 file
        :param metric: metric of data to get
        :param subdomain_name: name of subdomain to extract data from
        :return: times, values, distances
        '''

        if f'Results/Subdomains/{subdomain_name}' not in self.h.keys():
            WF.print2stdout(f"\nWARNING: Subdomain {subdomain_name} not found in results file.")
            return [], [], []

        if metric.lower() == 'flow':
            dataset_name = 'Cell flow'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'elevation':
            dataset_name = 'Water Surface Elevation'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'temperature':
            dataset_name = 'Water Temperature'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'do':
            dataset_name = 'Dissolved Oxygen'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'do_sat':
            dataset_name = 'Water Temperature'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            vt = np.array(dataset[:])
            dataset_name = 'Dissolved Oxygen'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            vdo = np.array(dataset[:])
            vt = WF.cleanComputed(vt)
            vdo = WF.cleanComputed(vdo)
            v = WF.calcComputedDOSat(vt, vdo, self.Report.Constants.satDO_interp)

        distance = self.calcSubdomainDistances(subdomain_name)
        #add a value at the start and end to compensate for the start and end values
        v = np.insert(v, 0, v.T[:][0], 1)
        v = np.insert(v, -1, v.T[:][-1], 1)

        if not hasattr(self, 't_computed'):
            self.loadComputedTime()
        istart = 0
        iend = -1
        #transpose so it returns [time, values at each cell]
        return self.t_computed[istart:iend], v[istart:iend].T, distance

    def readModelTimeseriesData(self, data, metric):
        '''
        function to wrangle data and be universably 'callable' from main script
        :param data: dictionary containing information about data records
        :param metric: not needed for W2, but still passed in for Ressim results so a single function can be called
        :return: arrays of dates and values
        '''

        x = data['easting']
        y = data['northing']
        dates, vals = self.get_Timeseries(metric, xy=[x, y])
        return dates, vals

    def calcSubdomainDistances(self, subdomain):
        '''
        calculates subdomain distances. Either uses the cell length to calculate, or does a distance formula between
        cell centers. Using a cell length field is MUCH more accurate, but you gotta make do.
        :param subdomain: name of subdomain
        :return: list of distances
        '''

        cell_center_xy = self.h['Geometry/Subdomains/{0}/Cell Center Coordinate'.format(subdomain)]
        firstpoint = cell_center_xy[0]
        distance = []
        if 'Geometry/Subdomains/{0}/Cell Length'.format(subdomain) in self.h.keys():
            distance.append(0)
            cell_lengths = self.h['Geometry/Subdomains/{0}/Cell Length'.format(subdomain)]
            for cli, celllen in enumerate(cell_lengths):
                if cli == 0:
                    distance.append(celllen.item()/2)# distance from edge to first cell center
                else:
                    #half the len of current cell, half the len of last cell to get distnace between cell centers
                    #then add on the distance we've calc'd
                    distance.append(celllen.item()/2 + cell_lengths[cli-1].item()/2 + distance[-1])
                if cli == len(cell_lengths)-1:
                    #add the distance from the last cell center to the edge
                    distance.append(celllen.item()/2 + distance[-1])
            distance = np.asarray(distance)
        else:
            for cell in cell_center_xy:
                d = np.sqrt( (cell[0] - firstpoint[0])**2 + (cell[1] - firstpoint[1])**2)
                distance.append(d)
            #get roughly half the distance between first two cells. This is the closest we can get to half the cell distance
            first_distance_diff_half = (distance[1] - distance[0]) / 2

            #shift all distance so the first instance is now the cell center of the first point
            distance = np.asarray(distance) + first_distance_diff_half

            #then add 0 to the start, so it starts at 0
            distance = np.insert(distance, 0, 0)

            #then do the same for the backend
            last_distance_diff_half = (distance[-1] - distance[-2]) / 2

            distance = np.append(distance, distance[-1] + last_distance_diff_half)

        return distance

    def findComputedStationCell(self, easting, northing, subdomain=None):
        '''
        finds subdomains that are closest to observed station coordinates
        TODO: add some kind of tolerance or max distance?
        :param xy: XY coordinates for observed station
        :return: cell index and subdomain information closest to observed data
        '''

        nearest_dist = 1e20
        data_index = 0
        data_subdomain = None
        if subdomain == None:
            subdomains = self.subdomains.items()
        else:
            subdomains = [(subdomain, self.subdomains[subdomain])]
        for subdomain, sd_data in subdomains:
            isRes = self.checkForReservoir(subdomain, sd_data)
            if isRes:
                WF.print2stdout(f'{subdomain} found to be reservoir for {easting}, {northing}. Skipping.', debug=self.Report.debug)
                continue
            x = sd_data['x']
            y = sd_data['y']
            dist = np.sqrt((x - easting) * (x - easting) + (y - northing) * (y - northing))
            min_dist = np.min(dist)
            if min_dist < nearest_dist:
                min_cell = np.argmin(dist) #pulls the index, not the min. duh
                data_index = min_cell
                data_subdomain = subdomain
                nearest_dist = min_dist
                cell_x = x[data_index]
                cell_y = y[data_index]
        WF.print2stdout(f'Using {data_subdomain} for {easting}, {northing}.', debug=self.Report.debug)
        WF.print2stdout(f'Distance: {nearest_dist}.', debug=self.Report.debug)
        WF.print2stdout(f'XY: {cell_x},{cell_y}.', debug=self.Report.debug)
        return data_index, data_subdomain

    def loadComputedTime(self):
        '''
        loads computed time values, replacing 24 hr date values with 0000 the next day
        grabs all values instead of user defined, if none are defined
        TODO: is this still needed? require user input.
        :return: sets class variables
                    self.t_computed - list of times used in computation
        '''

        tstr = self.h['Results/Subdomains/Time Date Stamp']
        tstr0 = (tstr[0]).decode("utf-8")
        tstr1 = (tstr[1]).decode("utf-8")
        ttmp = self.h['Results/Subdomains/Time']
        nt = len(ttmp)
        try:
            ttmp0 = dt.datetime.strptime(tstr0, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstr0.replace('24:00', '23:00')
            ttmp0 = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp0 += dt.timedelta(hours=1)
        try:
            ttmp1 = dt.datetime.strptime(tstr1, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstr1.replace('24:00', '23:00')
            ttmp1 = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp1 += dt.timedelta(hours=1)
        delta_t = ttmp1 - ttmp0
        self.t_computed = []
        for j in range(nt):
            self.t_computed.append(ttmp0 + j * delta_t)
        self.t_computed = np.array(self.t_computed)

    def getProfileTargetTimeseries(self, ResName, parameter, target_info):
        '''
        get profile values for target parameter. Target info contains info about what parameter and value to output at
        examples: give elevation when profile is 15 C, give temperature at elevation 400 ft, etc..
        :param ResName: name of reservoir
        :param parameter: parameter we want output
        :param target_info: info about target we want to hit and output values at
        :return: times, output values
        '''

        target_parameter = target_info['parameter']
        target_value = float(target_info['value'])
        self.loadResults('all', target_parameter.lower(), alt_subdomain_name=ResName)
        target_param_values = self.vals
        self.loadResults('all', parameter.lower(), alt_subdomain_name=ResName)
        output_param_values = self.vals
        self.loadComputedTime()

        interval_seconds = (self.t_computed[1] - self.t_computed[0]).total_seconds()
        if interval_seconds == 3600: #hourly
            interval = 24
        elif interval_seconds == 900:
            interval = 96
        elif interval_seconds == 86400:
            interval = 1
        # interval = 1

        vals_skip = target_param_values[::interval]
        output_val_at_target = np.full(len(vals_skip), np.nan)
        for i, vsp in enumerate(vals_skip):
            for j, pv in enumerate(vsp[::-1]):
                if pv <= target_value:
                    toplayer = self.getTopLayer(interval*i)
                    real_layer = len(vsp) - j - 1
                    if toplayer < real_layer:
                        layer = toplayer
                    else:
                        layer = real_layer

                    if layer < toplayer:
                        layer_pls_1 = layer + 1

                        if len(output_param_values.shape) == 1:
                            layer_val = output_param_values[layer]
                            layer_val_pls_1 = output_param_values[layer_pls_1]
                        elif len(output_param_values.shape) == 2:
                            layer_val = output_param_values[interval*i][layer]
                            layer_val_pls_1 = output_param_values[interval*i][layer_pls_1]
                        interp_layer_val = layer_val + ((target_value - vsp[layer]) / (vsp[layer_pls_1] - vsp[layer])) * (layer_val_pls_1 - layer_val)
                        output_val_at_target[i] = interp_layer_val
                        # y's are the elevations and x's are the temperatures.

                    else:
                        if len(output_param_values.shape) == 1:
                            output_val_at_target[i] = output_param_values[layer]
                        elif len(output_param_values.shape) == 2:
                            output_val_at_target[i] = output_param_values[interval*i][layer]


                    break

        return self.t_computed[::interval], output_val_at_target, self.units

    def checkForReservoir(self, subdomain, sd_data):
        '''
        checks to see if given location is a reservoir
        :param subdomain: name of ressim subdomain
        :param sd_data: data about subdomain
        :return: boolean if res or not
        '''

        attrs = self.h['Geometry/Subdomains'][subdomain].attrs
        if 'Subdomain Type' in attrs.keys():
            subdomaintype = attrs.get('Subdomain Type')[0].decode()
            if subdomaintype == 'Reach_1D':
                return True
        x = sd_data['x']
        y = sd_data['y']
        z = sd_data['z']
        num_x_is1 = len(x) == 1
        unique_num_x_is1 = len(list(set(x))) == 1
        unique_num_y_is1 = len(list(set(y))) == 1
        unique_num_z_is1 = len(list(set(z))) == 1
        if not num_x_is1: #only 1 value.. shouldnt happen by ressim rules but lets be sure.
            if not unique_num_z_is1 and unique_num_x_is1 and unique_num_y_is1: #res have same x and y but differnt y
                return True
            else:
                return False
        else:
            return True #channels must have 2 cells so this is our best guess

    def checkForOutput(self, resname, output):
        '''
        Checks for Ressim output Total boundary outflow in HDF5 file for FWA outflow
        :param resname: name of reservoir name
        :return: boolean
        '''

        if output in self.h['Results/Subdomains/' + resname].keys():
            return True
        else:
            return False

    def getFWAReservoirOutputTimeseries(self, resname, parameter):
        '''
        calculates flow weighted average reservoir outflow for a parameter
        :param resname:
        :param parameter:
        :return: flow weighted time series
        '''

        hasOutflow = self.checkForOutput(resname, 'Total boundary outflow')
        if not hasOutflow:
            WF.print2stdout(f'Total Boundary Outflow not found in results file for {resname}', debug=self.Report.debug)
            WF.print2stdout('To turn on, check "Cell Flow" in the Output variables tab, under Water Quality tab in the'
                            'ResSim Alternative Editor', debug=self.Report.debug)
        self.loadResults('all', parameter.lower(), alt_subdomain_name=resname)
        param_vals = self.vals[:]
        self.loadResults('all', 'outflow', alt_subdomain_name=resname)
        outflow_vals = self.vals[:]
        outflow_sums = outflow_vals.sum(axis=1)
        param_times_flow = param_vals * outflow_vals
        param_times_flow_sum = param_times_flow.sum(axis=1)
        FWA_values = param_times_flow_sum / outflow_sums
        return self.t_computed, FWA_values, self.units
