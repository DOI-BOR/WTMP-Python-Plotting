import os, sys
import numpy as np
import datetime as dt
import h5py						# h5py provides read access to the HEC-ResSim HDF5 (.h5) results format.

import WAT_Functions as WF
import WAT_Time as WT


class ResSim_Results(object):
    """
    Reader for HEC-ResSim simulation results stored in HDF5 (.h5) files.

    Wraps a ResSim alternative's ``.h5`` results file and provides
    methods to read time series, vertical profiles, gate/structure
    time series, longitudinal subdomain contours, and target-based
    (elevation/value-triggered) time series. Also supports reading an
    "external" H5 file (e.g. from a different simulation or study) by
    opening it directly rather than deriving its path from a simulation
    alternative name.

    Attributes
    ----------
    simulationPath : str
        Full path to the ResSim simulation directory.
    alternativeName : str
        Name of the selected ResSim alternative run (colons replaced
        with underscores).
    starttime : datetime.datetime
        Start time of the report window.
    endtime : datetime.datetime
        End time of the report window.
    external : bool
        Whether this instance represents an externally-supplied H5
        file rather than one derived from the simulation path.
    Report : object
        The main Report Generator instance.
    h : h5py.File
        The opened HDF5 file handle (set once opened).
    """

    def __init__(self, simulationPath, alternativeName, starttime, endtime, Report, external=False):
        """
        Set up the ResSim results reader and (unless external) open its H5 file.

        Parameters
        ----------
        simulationPath : str
            Full path to the ResSim simulation directory.
        alternativeName : str
            Name of the selected ResSim alternative run.
        starttime : datetime.datetime
            Start time of the report window.
        endtime : datetime.datetime
            End time of the report window.
        Report : object
            The main Report Generator instance.
        external : bool, optional
            If ``True``, this instance represents an externally-supplied
            H5 file (opened later via ``openH5File`` rather than derived
            from ``simulationPath``/``alternativeName``) (default
            ``False``).

        Returns
        -------
        None
            This is a constructor and does not return a value.

        Raises
        ------
        SystemExit
            Raised (indirectly, via ``getH5File``/``openH5File``) if the
            expected H5 file does not exist and ``external`` is
            ``False``.

        Examples
        --------
        >>> results = ResSim_Results('/path/to/sim', 'Alt1', starttime, endtime, Report)
        """

        # store the basic run identity/context, sanitizing the alternative name for filesystem use
        self.simulationPath = simulationPath
        self.alternativeName = alternativeName.replace(':', '_')
        self.starttime = starttime
        self.endtime = endtime
        self.external = external
        self.Report = Report

        if not self.external:
            # For a "normal" (non-external) instance, immediately locate
            # and open this alternative's own H5 file and load its
            # shared time/subdomain metadata.
            self.getH5File()
            self.load_time() #load time vars from h5
            self.loadSubdomains()

    def getH5File(self):
        """
        Build this alternative's expected H5 file path and open it.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets ``self.h5fname`` and (via ``openH5File``) ``self.h``.

        Raises
        ------
        SystemExit
            Raised (via ``openH5File``) if the expected file does not
            exist.

        Examples
        --------
        >>> results.getH5File()
        """

        # build the standard H5 filename from the alternative name, spaces replaced with underscores
        h5filefrmt = self.alternativeName.replace(' ', '_')
        self.h5fname = os.path.join(self.simulationPath, 'rss', h5filefrmt + '.h5')
        self.openH5File(self.h5fname)

    def openH5File(self, h5fname):
        """
        Open an HDF5 results file for reading.

        Parameters
        ----------
        h5fname : str
            Path to the HDF5 file to open.

        Returns
        -------
        None
            Sets ``self.h`` to the opened ``h5py.File`` handle. Exits
            the script if the file doesn't exist.

        Raises
        ------
        SystemExit
            Raised (via ``sys.exit(1)``) if ``h5fname`` does not exist.

        Examples
        --------
        >>> results.openH5File('/path/to/results.h5')
        """

        if os.path.exists(h5fname):
            # open the file in read-only mode
            self.h = h5py.File(h5fname, 'r')
        else:
            # can't proceed at all without the source file
            WF.print2stderr(f'ERROR: missing results file {h5fname}')
            sys.exit(1)

    def load_time(self):
        """
        Load the shared timestamp array from the H5 file.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets ``self.tstr`` (raw time date-stamp strings),
            ``self.dt_dates`` (array of datetimes), ``self.jd_dates``
            (array of Julian-day offsets), ``self.nt`` (number of
            timesteps), and ``self.t_offset`` (Julian-date offset of the
            first timestamp).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.load_time()
        >>> results.nt
        8760
        """

        # pull the raw date-stamp strings and the Julian-day offset array from the H5 file
        self.tstr = self.h['Results/Subdomains/Time Date Stamp']
        tstr0 = (self.tstr[0]).decode("utf-8")
        ttmp = self.h['Results/Subdomains/Time']
        jd_dates = ttmp[:]
        try:
            ttmp = dt.datetime.strptime(tstr0, '%Y-%m-%d, %H:%M')
        except ValueError:
            # H5 sometimes reports hour 24 (midnight of the next day)
            # instead of hour 00; roll it over manually since strptime
            # can't parse "24:" as an hour.
            tstrtmp = tstr0.replace('24:00', '23:00')
            ttmp = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp += dt.timedelta(hours=1)
        dt_dates = []
        # compute the ordinal (fractional-day) offset of the reference start time
        t_offset = ttmp.toordinal() + float(ttmp.hour) / 24. + float(ttmp.minute) / (24. * 60.)
        # Every subsequent timestamp is stored as a Julian-day offset
        # from this same reference start time.
        for t in jd_dates:
            dt_dates.append(ttmp + dt.timedelta(days=t))
        self.nt = len(dt_dates)

        self.dt_dates = np.asarray(dt_dates)
        self.jd_dates = np.asarray(jd_dates)
        self.t_offset = t_offset

    def loadSubdomains(self):
        """
        Load every subdomain's cell-center coordinates from the H5 file.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets ``self.subdomains``, a dict keyed by subdomain name,
            each with ``'x'``, ``'y'``, and ``'z'`` coordinate arrays
            (one entry per model cell).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.loadSubdomains()
        >>> list(results.subdomains.keys())
        ['Shasta', 'Reach1']
        """

        self.subdomains = {}
        group = self.h['Geometry/Subdomains']
        # read every subdomain's cell-center coordinates one at a time
        for subdomain in group:
            dataset = self.h['Geometry/Subdomains/{0}/Cell Center Coordinate'.format(subdomain)]
            ncells = (np.shape(dataset))[0]
            x = np.array([dataset[i][0] for i in range(ncells)])
            y = np.array([dataset[i][1] for i in range(ncells)])
            z = np.array([dataset[i][2] for i in range(ncells)])
            self.subdomains[subdomain] = {'x': x, 'y': y, 'z': z}

    def readProfileData(self, resname, metric, timestamps):
        """
        Read a vertical profile at a reservoir for one or more timestamps.

        Parameters
        ----------
        resname : str
            Name of the reservoir/subdomain in the H5 file.
        metric : str
            Name of the metric/parameter to extract.
        timestamps : list, numpy.ndarray, or str
            Specific timestamps to extract, or the string ``'all'`` to
            read every available timestep.

        Returns
        -------
        vals : list of numpy.ndarray
            Profile values, one array per requested timestamp.
        elevations : list of numpy.ndarray
            Elevation for each value.
        depths : list of numpy.ndarray
            Depth (from the water surface) for each value.
        times : numpy.ndarray
            The timestamps corresponding to each profile.
        units : str or None
            Units of the returned values, or ``None`` if the subdomain
            couldn't be read.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> vals, elevations, depths, times, units = results.readProfileData('Shasta', 'temperature', 'all')
        """

        self.loadElevation(alt_subdomain_name=resname)

        if self.subdomain_read_success:

            vals = []
            elevations = []
            depths = []
            times = []
            # WF.print2stdout('UNIQUE TIMES:', unique_times)
            if isinstance(timestamps, (list, np.ndarray)):
                # Specific timestamps requested: read the model results
                # closest to each one individually.
                unique_times = [n for n in timestamps]
                for j, time_in in enumerate(unique_times):
                    timestep = WT.getIdxForTimestamp(self.dt_dates, time_in)
                    if timestep == -1:
                        # No matching timestep found in the model output;
                        # record an empty profile for this timestamp
                        # rather than skipping it, so the returned lists
                        # stay aligned with the requested timestamps.
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
                        # Only include cells up through the top (active)
                        # water layer; convert each cell's elevation into
                        # a depth-below-surface value using the profile's
                        # own max elevation as the water surface.
                        for ei, e in enumerate(el):
                            d_step.append(np.max(el) - e)
                            e_step.append(e)
                            v_step.append(v_el[ei])
                        depths.append(np.asarray(d_step))
                        elevations.append(np.asarray(e_step))
                        vals.append(np.asarray(v_step))
                        times.append(time_in)
            else:
                # 'all' (or similar): read the full time series at once
                # rather than looping per-timestamp.
                self.loadResults('all', metric.lower(), alt_subdomain_name=resname)
                elevations = self.elev
                vals = np.asarray(self.vals)
                depths = np.array([])
                times = self.dt_dates

            return vals, elevations, depths, np.asarray(times), self.units

        else:
            # subdomain couldn't be found/read at all
            return [], [], [], [], None

    def getTopLayer(self, timestep_index):
        """
        Find the index of the top active (water-covered) model layer.

        Parameters
        ----------
        timestep_index : int
            Index into the loaded elevation time series to check.

        Returns
        -------
        int
            Index of the top layer whose midpoint (averaged with the
            layer above) is still below the water surface elevation at
            this timestep.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.getTopLayer(0)
        12
        """

        elev = self.elev_ts[timestep_index] #elevations at a timestep
        # scan upward through the layers until the water surface is passed
        for k in range(len(self.elev) - 1): #for each cell..
            cell_z = self.elev[k]  # layer midpoint
            cell_z1 = self.elev[k + 1]  # layer above midpoint
            top_of_cell_z = 0.5 * (cell_z + cell_z1)
            if elev < top_of_cell_z:
                break
        return k

    def loadElevation(self, alt_subdomain_name=None):
        """
        Load a subdomain's cell-center elevations and water-surface time series.

        Parameters
        ----------
        alt_subdomain_name : str, optional
            Subdomain name to use instead of ``self.subdomain_name``, if
            given.

        Returns
        -------
        None
            Sets ``self.ncells`` (cell count), ``self.elev`` (per-cell
            elevation array), ``self.elev_ts`` (water surface elevation
            time series), and ``self.subdomain_read_success`` (whether
            the subdomain was found in the H5 file).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.loadElevation(alt_subdomain_name='Shasta')
        """

        # use the alternate subdomain name if given, otherwise the instance's own default
        this_subdomain = self.subdomain_name if alt_subdomain_name is None else alt_subdomain_name
        subdomain_name = 'Geometry/Subdomains/' + this_subdomain + '/Cell Center Coordinate'
        if subdomain_name not in self.h.keys():
            # requested subdomain doesn't exist at all in this results file
            WF.print2stdout(f"\nWARNING: Subdomain {this_subdomain} not found in results file.")
            self.elev = np.array([])
            self.elev_ts = np.array([])
            self.ncells = 0
            self.subdomain_read_success = False
        else:
            # read the per-cell elevations and the water-surface elevation time series
            cell_center_xy = self.h[subdomain_name]
            self.ncells = (np.shape(cell_center_xy))[0]
            self.elev = np.array(cell_center_xy[:self.ncells, 2])
            elev_ts = self.h['Results/Subdomains/' + this_subdomain + '/Water Surface Elevation']
            self.elev_ts = np.array(elev_ts[:self.nt])
            self.subdomain_read_success = True

    def loadResults(self, t_in, metrc, alt_subdomain_name=None):
        """
        Load a metric's results (a single timestep, or the full series) from the H5 file.

        Parameters
        ----------
        t_in : datetime.datetime or str
            The specific timestamp to load, or ``'all'`` to load every
            timestep at once.
        metrc : str
            Name of the metric to load (e.g. ``'temperature'``,
            ``'diss_oxy'``/``'do'``, ``'do_sat'``,
            ``'elevation'``/``'wse'``, ``'outflow'``).
        alt_subdomain_name : str, optional
            Subdomain name to use instead of ``self.subdomain_name``, if
            given.

        Returns
        -------
        None
            Sets ``self.units`` and either ``self.t_data``/``self.vals``
            (single timestep, one value per cell) or ``self.vals``
            (full time series) depending on ``t_in``.

        Raises
        ------
        KeyError
            Raised for the ``'diss_oxy'``/``'do'`` metric if the
            requested dataset is missing from the results file.

        Examples
        --------
        >>> results.loadResults('all', 'temperature', alt_subdomain_name='Shasta')
        """

        # use the alternate subdomain name if given, otherwise the instance's own default
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
                # dataset genuinely missing, log and fall back to an empty result rather than crashing
                WF.print2stdout(f'\nWARNING: WQ Simulation does not have results for metric: {metric_name}')
                vals = []

        elif metrc == 'diss_oxy' or metrc.lower() == 'do':
            metric_name = 'Dissolved Oxygen'
            attrs = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name].attrs
            self.units = self.getUnitsFromAttrs(attrs)
            try:
                vals = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            except KeyError:
                # unlike temperature above, this metric raises instead of falling back silently
                raise KeyError('WQ Simulation does not have results for metric: {0}'.format(metric_name))

        elif metrc.lower() == 'do_sat':
            # Saturated DO isn't stored directly; compute it from the
            # temperature and dissolved-oxygen series.
            metric_name = 'Water Temperature'
            vt = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            metric_name = 'Dissolved Oxygen'
            vdo = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            vals = WF.calcComputedDOSat(vt, vdo, self.Report.Constants.satDO_interp)
            self.units = '%'

        elif metrc.lower() in ['elevation', 'wse']:
            # elevation is a byproduct of loading the subdomain's own elevation data
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
                # dataset genuinely missing, log and fall back to an empty result rather than crashing
                WF.print2stdout(f'\nWARNING: WQ Simulation does not have results for metric: {metric_name}')
                self.units = None
                vals = []

        if t_in != 'all':
            # Single timestep requested: locate its index and pull out
            # just that row (one value per cell).
            timestep = WT.getIdxForTimestamp(self.dt_dates, t_in) #get timestep index for current date
            if timestep == -1:
                # shouldn't normally happen given getIdxForTimestamp's nearest-match behavior
                WF.print2stdout('should never be here..')
            self.t_data = t_in
            self.vals = np.array([vals[timestep][i] for i in range(self.ncells)])
        else:
            # full time series requested, use everything as-is
            self.vals = vals

    def getUnitsFromAttrs(self, attrs):
        """
        Extract and decode a units string from an H5 dataset's attributes.

        Parameters
        ----------
        attrs : h5py.AttributeManager
            Attributes dict-like object for an H5 dataset, expected to
            contain a ``'Units'`` or ``'units'`` key.

        Returns
        -------
        str or None
            The decoded units string, or ``None`` if no units attribute
            was found or it couldn't be decoded.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.getUnitsFromAttrs(dataset.attrs)
        'C'
        """

        units = None
        if 'Units' in attrs.keys():
            u = attrs['Units']
        elif 'units' in attrs.keys():
            u = attrs['units']

        try:
            u_i = u[0]
        except IndexError:
            # no units attribute value present at all
            return None

        if isinstance(u_i, np.bytes_):
            # H5 stores strings as bytes; decode to a normal Python str.
            units = u_i.decode('utf-8')
        else:
            u_i = units

        return units

    def readTimeSeries(self, metric, x, y, subdomain=None):
        """
        Read a computed time series at the model cell closest to given coordinates.

        Parameters
        ----------
        metric : str
            Name of the metric to read (e.g. ``'flow'``, ``'elevation'``,
            ``'temperature'``, ``'do'``, ``'do_sat'``); several alternate
            spellings are recognized for each.
        x : float
            Easting coordinate of the target location, used to find the
            nearest model cell via ``findComputedStationCell``.
        y : float
            Northing coordinate of the target location.
        subdomain : str, optional
            Restrict the cell search to a specific subdomain, if given.

        Returns
        -------
        times : numpy.ndarray
            Timestamps for the series (empty if no matching cell/metric
            was found).
        values : numpy.ndarray
            Values for the series.
        units : str or None
            Units of the returned values.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> times, values, units = results.readTimeSeries('temperature', 620000.0, 4200000.0)
        """

        units = None
        # find the nearest model cell and its subdomain for the requested coordinates
        i, subdomain_name = self.findComputedStationCell(x, y, subdomain=subdomain)
        if subdomain_name == None:
            # no matching cell/subdomain found at all
            WF.print2stdout(f'XY coords ({x}, {y}) not found', debug=self.Report.debug)
            return [], [], units

        accepted_metrics = ['flow', 'elevation', 'temperature', 'do', 'do_sat']
        # Normalize the requested metric name (strip spaces/underscores,
        # lowercase) so many spelling variants all match.
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
            # Saturated DO isn't stored directly; compute it from the
            # temperature and dissolved-oxygen series at this cell.
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
            # unrecognized metric name at all
            WF.print2stdout(f'ERROR: Metric {metric} not in accepted metrics list: {accepted_metrics}')
            return [], [], units

        if not hasattr(self, 't_computed'):
            # Lazily load the computed-results timestamp array only once
            # (shared across every call to this method).
            self.loadComputedTime()
        istart = 0
        iend = -1
        return self.t_computed[istart:iend], v[istart:iend], units

    def readProfileTopwater(self, resname, timestamps):
        """
        Get the water-surface elevation at each requested timestamp.

        Used to filter reservoir contour plots down to only the water
        column actually present at each timestep.

        Parameters
        ----------
        resname : str
            Name of the reservoir/subdomain in the H5 file.
        timestamps : list, numpy.ndarray, or str
            Specific timestamps to look up, or ``'all'`` to return the
            full water-surface-elevation time series.

        Returns
        -------
        numpy.ndarray
            Water surface elevation at each requested timestamp (NaN
            where no matching timestep was found), or an empty list if
            the subdomain couldn't be read.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.readProfileTopwater('Shasta', 'all')
        """

        self.loadElevation(alt_subdomain_name=resname)

        if self.subdomain_read_success:

            if isinstance(timestamps, (list, np.ndarray)):
                # look up the closest matching timestep for each requested timestamp
                topwater = []
                unique_times = [n for n in timestamps]
                for j, time_in in enumerate(unique_times):
                    timestep = WT.getIdxForTimestamp(self.dt_dates, time_in)
                    if timestep == -1:
                        # no matching timestep found, record NaN rather than skipping
                        topwater.append(np.nan)
                        # continue
                    else:
                        topwater.append(self.elev_ts[timestep])
            else:
                # 'all' requested, return the full series directly
                topwater = self.elev_ts[:]
            return np.asarray(topwater)
        else:
            # subdomain couldn't be found/read at all
            return []

    def checkSubdomain(self, subdomain_name):
        """
        Check whether a subdomain exists in the model results.

        Parameters
        ----------
        subdomain_name : str
            Name of the subdomain to check.

        Returns
        -------
        bool
            ``True`` if the subdomain's results group exists in the H5
            file, ``False`` otherwise.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.checkSubdomain('Shasta')
        True
        """

        dataset = 'Results/Subdomains/{0}'.format(subdomain_name)
        if dataset not in self.h.keys():
            return False
        else:
            return True

    def readSubdomain(self, metric, subdomain_name):
        """
        Read a subdomain's full longitudinal contour (value vs. distance vs. time).

        Parameters
        ----------
        metric : str
            Name of the metric to read (``'flow'``, ``'elevation'``,
            ``'temperature'``, ``'do'``, or ``'do_sat'``).
        subdomain_name : str
            Name of the subdomain to extract data from.

        Returns
        -------
        times : numpy.ndarray
            Timestamps for the data.
        values : numpy.ndarray
            2-D array of values (time x distance).
        distance : numpy.ndarray
            Distance values for each cell along the subdomain, or an
            empty list if the subdomain wasn't found.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> times, values, distance = results.readSubdomain('temperature', 'Reach1')
        """

        if f'Results/Subdomains/{subdomain_name}' not in self.h.keys():
            # requested subdomain doesn't exist at all
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
            # Saturated DO isn't stored directly; compute it from the
            # temperature and dissolved-oxygen series across the domain.
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
        # Duplicate the first/last cell's values onto an extra padding
        # cell at each end, matching the padding added to `distance` in
        # calcSubdomainDistances, so the value and distance arrays stay
        # aligned in length.
        v = np.insert(v, 0, v.T[:][0], 1)
        v = np.insert(v, -1, v.T[:][-1], 1)

        if not hasattr(self, 't_computed'):
            # lazily load the computed-results timestamp array only once
            self.loadComputedTime()
        istart = 0
        iend = -1
        #transpose so it returns [time, values at each cell]
        return self.t_computed[istart:iend], v[istart:iend].T, distance

    def readModelTimeseriesData(self, data, metric):
        """
        Universal-call wrapper for reading a coordinate-based time series.

        Parameters
        ----------
        data : dict
            Settings dictionary containing ``'easting'`` and
            ``'northing'`` coordinates.
        metric : str
            Metric name; retained for a consistent call signature with
            the equivalent W2 method, though ResSim doesn't need it
            here beyond what ``get_Timeseries`` uses internally.

        Returns
        -------
        dates : array_like
            Timestamps for the series.
        vals : array_like
            Values for the series.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> dates, vals = results.readModelTimeseriesData(data, 'temperature')
        """

        # extract the target coordinates and delegate to the coordinate-based reader
        x = data['easting']
        y = data['northing']
        dates, vals = self.get_Timeseries(metric, xy=[x, y])
        return dates, vals

    def calcSubdomainDistances(self, subdomain):
        """
        Compute cumulative along-channel distance for every cell in a subdomain.

        Prefers using an explicit "Cell Length" field (much more
        accurate) when available; otherwise falls back to a straight-
        line distance formula between consecutive cell centers.

        Parameters
        ----------
        subdomain : str
            Name of the subdomain to compute distances for.

        Returns
        -------
        numpy.ndarray
            Cumulative distance for each cell edge/center, padded with
            an extra value at each end (matching the padding applied to
            the value array in ``readSubdomain``).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.calcSubdomainDistances('Reach1')
        """

        cell_center_xy = self.h['Geometry/Subdomains/{0}/Cell Center Coordinate'.format(subdomain)]
        firstpoint = cell_center_xy[0]
        distance = []
        if 'Geometry/Subdomains/{0}/Cell Length'.format(subdomain) in self.h.keys():
            # Preferred path: build cumulative distance directly from
            # each cell's known length, including the distance from the
            # channel edges to the first/last cell centers.
            distance.append(0)
            cell_lengths = self.h['Geometry/Subdomains/{0}/Cell Length'.format(subdomain)]
            for cli, celllen in enumerate(cell_lengths):
                if cli == 0:
                    # distance from the upstream edge to the first cell center
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
            # Fallback path: no explicit cell-length field, so estimate
            # distances via straight-line distance from the first cell
            # center to every other cell center.
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
        """
        Find the model cell closest to a given (easting, northing) coordinate.

        Parameters
        ----------
        easting : float
            Easting coordinate of the target location.
        northing : float
            Northing coordinate of the target location.
        subdomain : str, optional
            Restrict the search to a specific subdomain, if given
            (otherwise every subdomain is searched).

        Returns
        -------
        data_index : int
            Index of the closest cell within its subdomain.
        data_subdomain : str or None
            Name of the subdomain containing the closest cell, or
            ``None`` if nothing was found.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        Marked with a ``#TODO: add some kind of tolerance or max
        distance?`` comment in the original source.

        Examples
        --------
        >>> data_index, subdomain = results.findComputedStationCell(620000.0, 4200000.0)
        """

        nearest_dist = 1e20
        data_index = 0
        data_subdomain = None
        if subdomain == None:
            # search every subdomain
            subdomains = self.subdomains.items()
        else:
            # restrict the search to just the requested subdomain
            subdomains = [(subdomain, self.subdomains[subdomain])]
        for subdomain, sd_data in subdomains:
            # Reservoir subdomains aren't valid targets for a
            # coordinate-based station lookup (reservoirs are handled
            # via profile methods instead); skip them.
            isRes = self.checkForReservoir(subdomain, sd_data)
            if isRes:
                WF.print2stdout(f'{subdomain} found to be reservoir for {easting}, {northing}. Skipping.', debug=self.Report.debug)
                continue
            x = sd_data['x']
            y = sd_data['y']
            dist = np.sqrt((x - easting) * (x - easting) + (y - northing) * (y - northing))
            min_dist = np.min(dist)
            if min_dist < nearest_dist:
                # this subdomain's closest cell beats the current best, update the running match
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
        """
        Build the computed-results timestamp array from the first two timestamps.

        Assumes a perfectly regular interval, derived from the gap
        between the first two timestamps, and extrapolates the rest.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets ``self.t_computed``, an array of ``nt`` regularly-
            spaced datetimes.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        Marked with a ``#TODO: is this still needed? require user
        input.`` comment in the original source.

        Examples
        --------
        >>> results.loadComputedTime()
        >>> len(results.t_computed)
        8760
        """

        # pull the first two raw date-stamp strings to determine the regular interval
        tstr = self.h['Results/Subdomains/Time Date Stamp']
        tstr0 = (tstr[0]).decode("utf-8")
        tstr1 = (tstr[1]).decode("utf-8")
        ttmp = self.h['Results/Subdomains/Time']
        nt = len(ttmp)
        try:
            ttmp0 = dt.datetime.strptime(tstr0, '%Y-%m-%d, %H:%M')
        except ValueError:
            # H5 sometimes reports hour 24 instead of hour 00 of the
            # next day; roll it over manually.
            tstrtmp = tstr0.replace('24:00', '23:00')
            ttmp0 = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp0 += dt.timedelta(hours=1)
        try:
            ttmp1 = dt.datetime.strptime(tstr1, '%Y-%m-%d, %H:%M')
        except ValueError:
            # same hour-24 handling as above, for the second timestamp
            tstrtmp = tstr1.replace('24:00', '23:00')
            ttmp1 = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp1 += dt.timedelta(hours=1)
        delta_t = ttmp1 - ttmp0
        self.t_computed = []
        # extrapolate every remaining timestamp using the fixed interval derived above
        for j in range(nt):
            self.t_computed.append(ttmp0 + j * delta_t)
        self.t_computed = np.array(self.t_computed)

    def getProfileTargetTimeseries(self, ResName, parameter, target_info):
        """
        Get the value of one parameter at the elevation/depth where another parameter hits a target.

        Examples: get the elevation where the profile first reaches
        15 degrees C, or get the temperature at the elevation where
        flow equals a target value.

        Parameters
        ----------
        ResName : str
            Name of the reservoir/subdomain.
        parameter : str
            Parameter to output the value of at the target location.
        target_info : dict
            Dictionary with ``'parameter'`` (the parameter to search
            for the target in) and ``'value'`` (the target value to
            match/cross).

        Returns
        -------
        times : numpy.ndarray
            Timestamps for the output series (subsampled to one value
            per day, per the interval logic below).
        output_val_at_target : numpy.ndarray
            The ``parameter`` value at the target-matching depth/
            elevation for each timestep (NaN where no match was found).
        units : str or None
            Units of the output parameter values.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> times, vals, units = results.getProfileTargetTimeseries('Shasta', 'temperature', {'parameter': 'elevation', 'value': '900'})
        """

        target_parameter = target_info['parameter']
        target_value = float(target_info['value'])
        # load both the target-searching parameter and the output parameter's full profile series
        self.loadResults('all', target_parameter.lower(), alt_subdomain_name=ResName)
        target_param_values = self.vals
        self.loadResults('all', parameter.lower(), alt_subdomain_name=ResName)
        output_param_values = self.vals
        self.loadComputedTime()

        # Determine how many timesteps make up one day, so the search
        # below only checks once per day rather than every model
        # timestep (a speed optimization, since profile searches are
        # relatively expensive).
        interval_seconds = (self.t_computed[1] - self.t_computed[0]).total_seconds()
        if interval_seconds == 3600: #hourly
            interval = 24
        elif interval_seconds == 900:
            interval = 96
        elif interval_seconds == 86400:
            interval = 1
        # interval = 1

        # subsample to one profile per day before running the (relatively expensive) search
        vals_skip = target_param_values[::interval]
        output_val_at_target = np.full(len(vals_skip), np.nan)
        # For each sampled day's profile, search from the bottom layer
        # upward (reversed) for the first layer whose target-parameter
        # value is at or below the target value, then interpolate the
        # output parameter's value at that exact target-crossing point.
        for i, vsp in enumerate(vals_skip):
            for j, pv in enumerate(vsp[::-1]):
                if pv <= target_value:
                    # Clamp the search to whichever is shallower: the
                    # actual top-of-water layer, or the layer found by
                    # the reversed search (guards against searching
                    # above the water surface).
                    toplayer = self.getTopLayer(interval*i)
                    real_layer = len(vsp) - j - 1
                    if toplayer < real_layer:
                        layer = toplayer
                    else:
                        layer = real_layer

                    if layer < toplayer:
                        # Target crossing is strictly between this layer
                        # and the one above it: linearly interpolate the
                        # output parameter's value at the exact
                        # crossing point.
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
                        # Target crossing is exactly at the top layer;
                        # no interpolation needed, just use that layer's
                        # value directly.
                        if len(output_param_values.shape) == 1:
                            output_val_at_target[i] = output_param_values[layer]
                        elif len(output_param_values.shape) == 2:
                            output_val_at_target[i] = output_param_values[interval*i][layer]

                    break

        return self.t_computed[::interval], output_val_at_target, self.units

    def checkForReservoir(self, subdomain, sd_data):
        """
        Determine whether a subdomain represents a reservoir (rather than a river reach).

        Parameters
        ----------
        subdomain : str
            Name of the ResSim subdomain.
        sd_data : dict
            Subdomain cell-coordinate data with ``'x'``, ``'y'``, and
            ``'z'`` arrays.

        Returns
        -------
        bool
            ``True`` if the subdomain is identified as a reservoir,
            ``False`` if it's identified as a river reach/channel.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.checkForReservoir('Shasta', sd_data)
        True
        """

        attrs = self.h['Geometry/Subdomains'][subdomain].attrs
        if 'Subdomain Type' in attrs.keys():
            # Preferred check: an explicit subdomain-type attribute.
            subdomaintype = attrs.get('Subdomain Type')[0].decode()
            if subdomaintype == 'Reach_1D':
                return True
        # Fallback heuristic (no explicit type attribute available):
        # reservoirs tend to have cells sharing the same x/y position but
        # differing z (vertical layers), whereas channels vary in x/y.
        x = sd_data['x']
        y = sd_data['y']
        z = sd_data['z']
        num_x_is1 = len(x) == 1
        unique_num_x_is1 = len(list(set(x))) == 1
        unique_num_y_is1 = len(list(set(y))) == 1
        unique_num_z_is1 = len(list(set(z))) == 1
        if not num_x_is1: #only 1 value.. shouldnt happen by ressim rules but lets be sure.
            if not unique_num_z_is1 and unique_num_x_is1 and unique_num_y_is1: #res have same x and y but differnt y
                # varying z but constant x/y across cells, looks like a reservoir
                return True
            else:
                return False
        else:
            # only a single cell present at all, best guess is a channel
            return True #channels must have 2 cells so this is our best guess

    def checkForOutput(self, resname, output):
        """
        Check whether a given output dataset exists for a subdomain.

        Used to check for "Total boundary outflow" before computing a
        flow-weighted-average time series.

        Parameters
        ----------
        resname : str
            Name of the reservoir/subdomain.
        output : str
            Name of the output dataset to check for.

        Returns
        -------
        bool
            ``True`` if the dataset exists, ``False`` otherwise.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> results.checkForOutput('Shasta', 'Total boundary outflow')
        True
        """

        if output in self.h['Results/Subdomains/' + resname].keys():
            return True
        else:
            return False

    def getFWAReservoirOutputTimeseries(self, resname, parameter):
        """
        Compute a flow-weighted-average time series for a reservoir output parameter.

        Parameters
        ----------
        resname : str
            Name of the reservoir/subdomain.
        parameter : str
            Name of the parameter to flow-weight-average (e.g.
            ``'temperature'``).

        Returns
        -------
        times : numpy.ndarray
            Timestamps for the series.
        FWA_values : numpy.ndarray
            The flow-weighted-average parameter value at each timestep.
        units : str or None
            Units of the parameter values.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> times, FWA_values, units = results.getFWAReservoirOutputTimeseries('Shasta', 'temperature')
        """

        # confirm the required outflow dataset exists, warning (but continuing) if not
        hasOutflow = self.checkForOutput(resname, 'Total boundary outflow')
        if not hasOutflow:
            WF.print2stdout(f'Total Boundary Outflow not found in results file for {resname}', debug=self.Report.debug)
            WF.print2stdout('To turn on, check "Cell Flow" in the Output variables tab, under Water Quality tab in the'
                            'ResSim Alternative Editor', debug=self.Report.debug)
        # load both the parameter series and the outflow series across every outlet
        self.loadResults('all', parameter.lower(), alt_subdomain_name=resname)
        param_vals = self.vals[:]
        self.loadResults('all', 'outflow', alt_subdomain_name=resname)
        outflow_vals = self.vals[:]
        # Flow-weighted average = sum(parameter * outflow) / sum(outflow)
        # across every outlet, at each timestep.
        outflow_sums = outflow_vals.sum(axis=1)
        param_times_flow = param_vals * outflow_vals
        param_times_flow_sum = param_times_flow.sum(axis=1)
        FWA_values = param_times_flow_sum / outflow_sums
        return self.t_computed, FWA_values, self.units