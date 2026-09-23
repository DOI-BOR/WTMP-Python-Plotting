import numpy as np
import os, sys, math, re, pickle, itertools
from scipy import interpolate
from scipy.constants import convert_temperature		# scipy provides general interpolation plus a dedicated temperature-unit converter; 
from sklearn.metrics import mean_absolute_error		# sklearn provides the mean-absolute-error metric used in calcMAE below.
import datetime as dt
from collections import Counter
from matplotlib.colors import is_color_like			# is_color_like validates whether a string/tuple is a usable matplotlib color (used by confirmColor/fixDuplicateColors below).

import WAT_Constants as WC
import WAT_Time as WT
import WAT_Reader as WR

# Shared constants instance (units, unit conversions, colors, etc.),
# used throughout this module's unit-conversion and formatting helpers.
constants = WC.WAT_Constants()


def print2stdout(*a, debug=True):
    """
    Print a message to stdout, gated by a debug flag.

    Parameters
    ----------
    *a
        Values to print (passed through to ``print``).
    debug : bool, optional
        Only prints if ``True`` (default ``True``).

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> print2stdout('Hello', debug=True)
    Hello
    """

    # This is the module's central logging function: nearly every other
    # file in this repo calls print2stdout(..., debug=self.Report.debug)
    # so log output can be toggled on/off globally via one debug flag.
    if debug:
        # only print when the caller's debug flag is enabled
        print(*a, file=sys.stdout)


def print2stderr(*a):
    """
    Print a message to stderr, unconditionally.

    Parameters
    ----------
    *a
        Values to print (passed through to ``print``).

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> print2stderr('Something went wrong')
    """

    # Unlike print2stdout, this always prints (no debug gate), since it's
    # reserved for genuine errors that should always be visible.
    print(*a, file=sys.stderr)


def printVersion(VERSIONNUMBER):
    """
    Print the current version number to stdout.

    Parameters
    ----------
    VERSIONNUMBER : str
        The version string to print.

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> printVersion('1.0.0')
    """

    # delegate to the shared debug-gated logger
    print2stdout(f'VERSION: {VERSIONNUMBER}')


def checkExists(infile):
    """
    Verify a required file exists, exiting the script if it does not.

    Parameters
    ----------
    infile : str
        Path to the file to check.

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    SystemExit
        Raised (via ``sys.exit(1)``) if ``infile`` does not exist.

    Examples
    --------
    >>> checkExists('/path/to/required_file.xml')
    """

    # A missing required file is treated as fatal: log to stderr and
    # terminate the whole script rather than continuing with bad state.
    if not os.path.exists(infile):
        print2stderr(f'ERROR: {infile} does not exist')
        sys.exit(1)


def cleanMissing(indata):
    """
    Replace the -901 "missing data" sentinel value with NaN.

    Parameters
    ----------
    indata : numpy.ndarray
        Array of data to clean.

    Returns
    -------
    numpy.ndarray
        The cleaned data array (modified in place and returned).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Notes
    -----
    Marked with a ``#TODO: merge with omit values function?`` comment
    in the original source.

    Examples
    --------
    >>> import numpy as np
    >>> cleanMissing(np.array([1.0, -901.0, 3.0]))
    array([ 1., nan,  3.])
    """

    # -901 is a sentinel value (commonly used by DSS/HEC data) meaning
    # "missing"; convert it to NaN so it doesn't get plotted/averaged as
    # a real value.
    indata[indata == -901.] = np.nan
    return indata


def cleanComputed(indata):
    """
    Replace the -9999 "missing computed data" sentinel value with NaN.

    Parameters
    ----------
    indata : numpy.ndarray
        Array of data to clean.

    Returns
    -------
    numpy.ndarray
        The cleaned data array (modified in place and returned).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Notes
    -----
    Marked with a ``#TODO: merge with omit values function?`` comment
    in the original source.

    Examples
    --------
    >>> import numpy as np
    >>> cleanComputed(np.array([1.0, -9999.0, 3.0]))
    array([ 1., nan,  3.])
    """

    # -9999 is a different sentinel value used by computed/model output
    # to flag missing data; same treatment as cleanMissing above.
    indata[indata == -9999.] = np.nan
    return indata


def cleanOutputDirectory(dir_name, filetype):
    """
    Delete every file of a given type from a directory.

    Mainly used to erase old images from the output directory before a
    fresh report run.

    Parameters
    ----------
    dir_name : str
        Full path to the directory to clean.
    filetype : str
        File extension/suffix to match for deletion.

    Returns
    -------
    None
        This function does not return a value; it deletes files as a
        side effect.

    Raises
    ------
    None
        This function does not propagate exceptions; a failure to
        delete an individual file is caught, logged, and skipped.

    Examples
    --------
    >>> cleanOutputDirectory('/path/to/output', '.png')
    """

    # Only remove files matching the given extension/suffix, leaving
    # everything else in the directory untouched.
    files_in_directory = os.listdir(dir_name)
    # narrow down to just the files matching the requested extension/suffix
    filtered_files = [file for file in files_in_directory if file.endswith(filetype)]
    # delete each matching file one at a time
    for file in filtered_files:
        path_to_file = os.path.join(dir_name, file)
        try:
            os.remove(path_to_file)
        except:
            # Don't let one locked/permission-denied file stop cleanup of
            # the rest; just log and move on.
            print2stdout('Failed to delete', path_to_file)
            print2stdout('Continuing..')


def calcDOSaturation(temp, diss_ox, DOSat_Interp):
    """
    Compute dissolved-oxygen saturation percentage at a given temperature.

    Parameters
    ----------
    temp : float
        Water temperature.
    diss_ox : float
        Measured dissolved oxygen concentration.
    DOSat_Interp : scipy.interpolate.interp1d
        Precomputed saturated-DO-vs-temperature interpolation function
        (from ``WAT_Constants``).

    Returns
    -------
    float
        Dissolved oxygen as a percentage of saturation.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calcDOSaturation(20.0, 8.0, DOSat_Interp)
    """

    # Look up the saturated DO concentration at this temperature (via the
    # precomputed interpolation curve from WAT_Constants), then express
    # the actual DO as a percentage of saturation.
    do_sat = DOSat_Interp(temp)
    return diss_ox / do_sat * 100.


def calcComputedDOSat(vtemp, vdo, DOSat_Interp):
    """
    Compute dissolved-oxygen saturation percentage for computed/model data.

    Parameters
    ----------
    vtemp : numpy.ndarray
        Temperature values.
    vdo : numpy.ndarray
        Dissolved oxygen values.
    DOSat_Interp : scipy.interpolate.interp1d
        Precomputed saturated-DO-vs-temperature interpolation function.

    Returns
    -------
    numpy.ndarray
        DO saturation percentage values, with NaN at any index where
        either input was NaN.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calcComputedDOSat(vtemp, vdo, DOSat_Interp)
    """

    # allocate the output array up front, same shape as the temperature input
    v = np.zeros_like(vtemp)
    # compute the saturation percentage index by index
    for j in range(len(v)):
        # Propagate NaN if either input is missing at this index, rather
        # than computing a bogus saturation percentage.
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = calcDOSaturation(vtemp[j], vdo[j], DOSat_Interp)
    return v


def calcObservedDOSat(ttemp, vtemp, vdo, ):
    """
    Compute dissolved-oxygen saturation percentage for observed data.

    Parameters
    ----------
    ttemp : array_like
        Timestamps for the data.
    vtemp : numpy.ndarray
        Temperature values.
    vdo : numpy.ndarray
        Dissolved oxygen values.

    Returns
    -------
    ttemp : array_like
        The input timestamps, unchanged.
    v : numpy.ndarray
        DO saturation percentage values, with NaN at any index where
        either input was NaN.

    Raises
    ------
    TypeError
        Would be raised by the internal call to ``calcDOSaturation``,
        which requires a third ``DOSat_Interp`` argument that this
        function does not supply. See Notes below.

    Notes
    -----
    Unlike ``calcComputedDOSat`` above, the internal call to
    ``calcDOSaturation`` omits the required ``DOSat_Interp`` argument.
    This appears to be a latent bug that would raise a ``TypeError`` if
    this function is actually invoked. Left unchanged here per the "no
    logic changes" scope of this documentation pass.

    Examples
    --------
    >>> calcObservedDOSat(ttemp, vtemp, vdo)
    """

    # allocate the output array up front, same shape as the temperature input
    v = np.zeros_like(vtemp)
    # compute the saturation percentage index by index
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            # propagate NaN if either input is missing at this index
            v[j] = np.nan
        else:
            # NOTE: unlike calcComputedDOSat above, this call omits the
            # DOSat_Interp argument even though calcDOSaturation requires
            # it; this appears to be a latent bug (would raise
            # TypeError if this function is actually called). Left
            # unchanged here per the "no logic changes" scope of this
            # documentation pass.
            v[j] = calcDOSaturation(vtemp[j], vdo[j])
    return ttemp, v


def getSubplotConfig(n_profiles, plots_per_row):
    """
    Determine the subplot grid (rows x columns) for a given plot count.

    Parameters
    ----------
    n_profiles : int
        Total number of plots to arrange.
    plots_per_row : int
        Maximum number of plots per row.

    Returns
    -------
    rows : int
        Number of subplot rows needed.
    cols : int
        Number of subplot columns to use.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getSubplotConfig(7, 3)
    (3, 3)
    """

    # If everything fits on fewer than one full row, use a single row
    # sized to just the number of profiles; otherwise, fill full rows of
    # plots_per_row and round up to a whole number of rows for the
    # remainder.
    factor = n_profiles / plots_per_row
    if factor < 1:
        # fewer plots than a single row holds, use exactly that many columns
        return 1, n_profiles
    else:
        # round up to a whole number of full rows
        return math.ceil(factor), plots_per_row


def matchData(data1, data2):
    """
    Align two time-series-like datasets to the same length via interpolation.

    If one dataset is shorter than the other, the shorter one is
    interpolated onto the longer one's axis; points outside the
    interpolation range are dropped from both so the two remain aligned.

    Parameters
    ----------
    data1 : dict
        Dictionary containing ``'values'`` plus a shared axis key
        (``'dates'``, ``'depths'``, or ``'elevations'``).
    data2 : dict
        Dictionary in the same form as ``data1``.

    Returns
    -------
    data1 : dict
        The (possibly resampled) first dataset.
    data2 : dict
        The (possibly resampled) second dataset, aligned with ``data1``.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> data1, data2 = matchData(data1, data2)
    """

    # Determine which axis key both datasets share (time series use
    # 'dates', profiles use 'depths' or 'elevations').
    if 'dates' in data1.keys() and 'dates' in data2.keys():
        y_key = 'dates'
    elif 'depths' in data1.keys() and 'depths' in data2.keys():
        y_key = 'depths'
    elif 'elevations' in data1.keys() and 'elevations' in data2.keys():
        y_key = 'elevations'

    # pull out data1's values, coercing to an array if given as a plain list
    v_1 = data1['values']
    if isinstance(v_1, list):
        v_1 = np.asarray(v_1)

    # Dates need to be converted to numeric (POSIX timestamp) values
    # before they can be used as the x-axis for interpolation.
    if y_key == 'dates':
        t_1 = [n.timestamp() for n in data1[y_key]]
    else:
        # depths/elevations are already numeric, use directly
        t_1 = data1[y_key]

    # pull out data2's values, coercing to an array if given as a plain list
    v_2 = data2['values']
    if isinstance(v_2, list):
        v_2 = np.asarray(v_2)

    if y_key == 'dates':
        t_2 = [n.timestamp() for n in data2[y_key]]
    else:
        # depths/elevations are already numeric, use directly
        t_2 = data2[y_key]

    if len(v_1) == 0 or len(v_2) == 0:
        # nothing usable in one or both datasets, return them unchanged
        return data1, data2

    if len(v_1) == len(v_2):
        # Already the same length; assume they're aligned and do nothing.
        return data1, data2

    elif len(v_1) > len(v_2):
        # data2 is shorter/coarser: interpolate it onto data1's axis,
        # then keep only the points where the interpolation succeeded
        # (i.e. fell within data2's original range) so both datasets end
        # up the same length and aligned on data1's axis.
        f_interp = interpolate.interp1d(t_2, v_2, bounds_error=False, fill_value=np.nan)
        v2_interp = f_interp(t_1)
        # only keep points where interpolation actually produced a value
        msk = np.isfinite(v2_interp)
        v_1_msk = v_1[msk]
        v_2_msk = v2_interp[msk]
        data1['values'] = v_1_msk
        data2['values'] = v_2_msk
        data2[y_key] = data1[y_key][msk]
        data1[y_key] = data1[y_key][msk]
        return data1, data2

    elif len(v_2) > len(v_1):
        # Mirror image of the branch above: data1 is shorter, so
        # interpolate it onto data2's axis instead.
        f_interp = interpolate.interp1d(t_1, v_1, bounds_error=False, fill_value=np.nan)
        v1_interp = f_interp(t_2)
        # only keep points where interpolation actually produced a value
        msk = np.isfinite(v1_interp)
        v_1_msk = v1_interp[msk]
        v_2_msk = np.asarray(v_2)[msk]
        data1['values'] = v_1_msk
        data1[y_key] = data2[y_key][msk]
        data2[y_key] = data2[y_key][msk]
        data2['values'] = v_2_msk
        return data1, data2


def printSimulationInfo(simulation):
    """
    Log a summary of a simulation's key settings to stdout.

    Parameters
    ----------
    simulation : dict
        Simulation settings dictionary; expected to contain at least
        ``'name'``, ``'basename'``, ``'ID'``, ``'directory'``,
        ``'dssfile'``, ``'starttime'``, and ``'endtime'``, plus optional
        ``'csvfile'`` and ``'modelalternatives'`` keys.

    Returns
    -------
    None
        This function does not return a value; it logs directly to
        stdout.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> printSimulationInfo(simulation)
    """
    # log the top-level simulation identity/settings
    print2stdout(f'\nSimulation: {simulation["name"]} - {simulation["basename"]} - {simulation["ID"]}')
    print2stdout('Directory: {}'.format(simulation['directory']))
    print2stdout('DSS file: {}'.format(simulation['dssfile']))
    print2stdout('starttime: {}'.format(simulation['starttime']))
    print2stdout('endtime: {}'.format(simulation['endtime']))
    if 'csvfile' in simulation.keys():
        # optional CSV data source, only log if present
        print2stdout('csvfile: {}'.format(simulation['csvfile']))
    if 'modelalternatives' in simulation.keys() and len(simulation['modelalternatives']) > 0:
        # log every configured model alternative and its program type
        print2stdout('Model Alternatives:')
        for modelalt in simulation['modelalternatives']:
            print2stdout('\t{0} - {1}'.format(modelalt['name'], modelalt['program']))


def checkData(dataset, flag=None):
    """
    Check whether a dataset is valid for plotting/tabulation.

    Valid means present, non-empty, and not entirely NaN. For dicts
    without a specific ``flag``, recurses into every entry and considers
    the dataset valid if at least one entry is valid.

    Parameters
    ----------
    dataset : dict, list, or numpy.ndarray
        The dataset to check.
    flag : str, optional
        If ``dataset`` is a dict, checks specifically the value at this
        key rather than every key.

    Returns
    -------
    bool
        ``True`` if the dataset (or the specified flag's data) is
        usable, ``False`` otherwise.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> import numpy as np
    >>> checkData(np.array([1.0, 2.0, np.nan]))
    True
    >>> checkData(np.array([np.nan, np.nan]))
    False
    """

    if isinstance(dataset, dict):
        if flag != None:
            # Checking a specific key within the dict: must exist, be
            # non-empty, and not be entirely NaN.
            if flag not in dataset.keys():
                return False
            elif len(dataset[flag]) == 0:
                return False
            elif checkAllNaNs(dataset[flag]):
                return False
            else:
                return True
        else:
            # No specific flag given: recursively check every entry
            # (including nested dicts, e.g. per-member collections), and
            # consider the whole dataset valid if AT LEAST ONE entry is
            # valid.
            multicheck = False
            # check every key, tracking whether any single one is valid
            for key in dataset.keys():
                if isinstance(dataset[key], dict):
                    # nested dict, recurse without a specific flag
                    check = checkData(dataset[key])
                    if check:
                        multicheck = True
                else:
                    # leaf value, check this specific key
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
        # Not a dict, list, or array (e.g. None) - never valid.
        return False


def checkAllNaNs(values):
    """
    Check whether every value in a set is NaN.

    Some NaN values in a series are fine; all-NaN is not.

    Parameters
    ----------
    values : array_like
        Values to check.

    Returns
    -------
    bool
        ``True`` if every value is NaN, ``False`` otherwise.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> import numpy as np
    >>> checkAllNaNs([np.nan, np.nan])
    True
    """

    if np.all(np.isnan(values)):
        return True
    else:
        return False


def removeNaNs(data1, data2, flag='values'):
    """
    Drop indices from both datasets where either one is NaN.

    Keeping NaN-mismatched points would throw off downstream statistics
    (MAE, RMSE, etc.), so this filters both datasets down to only the
    indices valid in both.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        First dataset.
    data2 : dict, list, or numpy.ndarray
        Second dataset.
    flag : str, optional
        If the datasets are dicts, the key holding the values to check
        for NaN (default ``'values'``); every other array in the dict
        is filtered with the same mask.

    Returns
    -------
    data1 : dict, list, or numpy.ndarray
        The filtered first dataset.
    data2 : dict, list, or numpy.ndarray
        The filtered second dataset, aligned with ``data1``.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> data1, data2 = removeNaNs(data1, data2)
    """

    # Build a mask of non-NaN indices for each dataset's value array.
    if isinstance(data1, dict):
        d1_msk = np.where(~np.isnan(data1[flag]))
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        d1_msk = np.where(~np.isnan(data1))

    if isinstance(data2, dict):
        d2_msk = np.where(~np.isnan(data2[flag]))
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        d2_msk = np.where(~np.isnan(data2))

    # Only keep indices that are valid (non-NaN) in BOTH datasets, so the
    # two returned series stay aligned index-for-index.
    msk = np.intersect1d(d1_msk, d2_msk)

    if isinstance(data1, dict):
        data1[flag] = np.asarray(data1[flag])[msk]
        # Apply the same mask to every other array in the dict (e.g.
        # dates) so they stay aligned with the filtered values.
        for otherflag in data1.keys():
            if otherflag != flag:
                data1[otherflag] = np.asarray(data1[otherflag])[msk]
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        data1 = np.asarray(data1)[msk]

    if isinstance(data2, dict):
        data2[flag] = np.asarray(data2[flag])[msk]
        # apply the same shared mask to every other array in this dict too
        for otherflag in data2.keys():
            if otherflag != flag:
                data2[otherflag] = np.asarray(data2[otherflag])[msk]
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        data2 = np.asarray(data2)[msk]

    return data1, data2


def removeINFs(data1, data2, flag='values'):
    """
    Drop indices from both datasets where either one is +/-infinity.

    Same rationale and structure as ``removeNaNs``, but for infinite
    values instead of NaN.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        First dataset.
    data2 : dict, list, or numpy.ndarray
        Second dataset.
    flag : str, optional
        If the datasets are dicts, the key holding the values to check
        for infinity (default ``'values'``).

    Returns
    -------
    data1 : dict, list, or numpy.ndarray
        The filtered first dataset.
    data2 : dict, list, or numpy.ndarray
        The filtered second dataset, aligned with ``data1``.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> data1, data2 = removeINFs(data1, data2)
    """

    # Identical structure to removeNaNs above, but masking out +/-inf
    # values instead of NaN.
    if isinstance(data1, dict):
        d1_msk = np.where(~np.isinf(data1[flag]))
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        d1_msk = np.where(~np.isinf(data1))

    if isinstance(data2, dict):
        d2_msk = np.where(~np.isinf(data2[flag]))
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        d2_msk = np.where(~np.isinf(data2))

    # only keep indices valid (non-infinite) in both datasets
    msk = np.intersect1d(d1_msk, d2_msk)

    if isinstance(data1, dict):
        data1[flag] = np.asarray(data1[flag])[msk]
        # apply the same shared mask to every other array in this dict too
        for otherflag in data1.keys():
            if otherflag != flag:
                data1[otherflag] = np.asarray(data1[otherflag])[msk]
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        data1 = np.asarray(data1)[msk]

    if isinstance(data2, dict):
        data2[flag] = np.asarray(data2[flag])[msk]
        # apply the same shared mask to every other array in this dict too
        for otherflag in data2.keys():
            if otherflag != flag:
                data2[otherflag] = np.asarray(data2[otherflag])[msk]
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        data2 = np.asarray(data2)[msk]

    return data1, data2


def calcMAE(data1, data2):
    """
    Compute the mean absolute error between two datasets.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        First dataset.
    data2 : dict, list, or numpy.ndarray
        Second dataset.

    Returns
    -------
    float
        The MAE value, or NaN if either dataset is unusable after
        alignment/cleaning.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calcMAE(data1, data2)
    """

    # Standard prep pipeline shared by every comparison statistic in this
    # module: align lengths, drop NaN/inf points, and bail out with NaN
    # if either dataset is unusable afterward.
    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        # one or both datasets are unusable after cleaning, nothing to compute
        return np.nan

    # coerce to plain float arrays for the sklearn metric
    data1_val = np.array(data1['values'], dtype=np.float64)
    data2_val = np.array(data2['values'], dtype=np.float64)

    return mean_absolute_error(data2_val, data1_val)


def calcMeanBias(data1, data2):
    """
    Compute the mean bias (data1 - data2) between two datasets.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        First dataset.
    data2 : dict, list, or numpy.ndarray
        Second dataset.

    Returns
    -------
    float
        The mean bias value, or NaN if either dataset is unusable after
        alignment/cleaning.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calcMeanBias(data1, data2)
    """

    # same alignment/cleaning pipeline used by every comparison statistic
    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan

    # coerce to plain float arrays for the arithmetic below
    data1_val = np.array(data1['values'], dtype=np.float64)
    data2_val = np.array(data2['values'], dtype=np.float64)

    # Mean bias = average of (data1 - data2), i.e. how much data1 tends
    # to run higher (positive) or lower (negative) than data2.
    diff = data1_val - data2_val
    count = len(data1_val)
    mean_diff = np.sum(diff) / count
    return mean_diff


def calcRMSE(data1, data2):
    """
    Compute the root-mean-square error between two datasets.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        First dataset.
    data2 : dict, list, or numpy.ndarray
        Second dataset.

    Returns
    -------
    float
        The RMSE value, or NaN if either dataset is unusable after
        alignment/cleaning.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calcRMSE(data1, data2)
    """

    # same alignment/cleaning pipeline used by every comparison statistic
    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan

    # coerce to plain float arrays for the arithmetic below
    data1_val = np.array(data1['values'], dtype=np.float64)
    data2_val = np.array(data2['values'], dtype=np.float64)

    diff = data1_val - data2_val
    count = len(data1_val)

    # Standard RMSE: square root of the mean squared difference.
    rmse = np.sqrt(np.sum(diff ** 2) / count)

    return rmse


def calcNSE(data1, data2):
    """
    Compute the Nash-Sutcliffe Efficiency (NSE) between two datasets.

    As per `Nash and Sutcliffe, 1970
    <https://doi.org/10.1016/0022-1694(70)90255-6>`_:

    .. math::
       E_{\\text{NSE}} = 1 - \\frac{\\sum_{i=1}^{N}[e_{i}-s_{i}]^2}
       {\\sum_{i=1}^{N}[e_{i}-\\mu(e)]^2}

    where *N* is the length of the *simulations* and *evaluation*
    periods, *e* is the *evaluation* series, *s* is (one of) the
    *simulations* series, and *mu* is the arithmetic mean.

    source: https://pypi.org/project/hydroeval

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        First (simulated) dataset.
    data2 : dict, list, or numpy.ndarray
        Second (evaluation/reference) dataset.

    Returns
    -------
    float
        The NSE value, or NaN if either dataset is unusable after
        alignment/cleaning, or if the computed value is infinite.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calcNSE(data1, data2)
    """

    # same alignment/cleaning pipeline used by every comparison statistic
    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan
    # nash = nse(data1['values'], data2['values'])
    # coerce to plain float arrays for the arithmetic below
    data1_val = np.array(data1['values'], dtype=np.float64)
    data2_val = np.array(data2['values'], dtype=np.float64)

    ### STEVE
    # NSE = 1 - (sum of squared error between the two series) / (sum of
    # squared deviation of data2 from its own mean). An NSE of 1 is a
    # perfect match; 0 means data1 predicts no better than the mean of
    # data2; negative means worse than just using the mean.
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
        # Guard against divide-by-zero (e.g. data2 is constant, so its
        # variance is 0) producing +/-inf instead of a usable value.
        nse_ = np.nan

    return nse_


def getMultiDatasetCount(data1, data2):
    """
    Get the count of comparable values across two datasets.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        First dataset.
    data2 : dict, list, or numpy.ndarray
        Second dataset.

    Returns
    -------
    int or float
        Number of values to compare, or NaN if the datasets can't be
        made the same length or are otherwise unusable.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getMultiDatasetCount(data1, data2)
    """

    # same alignment/cleaning pipeline used by every comparison statistic
    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    data1, data2 = removeINFs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan
    if len(data1['values']) != len(data2['values']):
        # still mismatched even after alignment, nothing usable to count
        return np.nan
    return len(data1['values'])


def getCount(data1):
    """
    Count the number of non-NaN values in a dataset.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        The dataset to count.

    Returns
    -------
    int or float
        Count of valid values, or NaN if the dataset is unusable.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getCount(data1)
    """

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    # count every index that is not NaN
    return len(np.where(~np.isnan(data1['values']))[0])


def calcMean(data1):
    """
    Compute the mean of a dataset, ignoring NaN and infinite values.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        The dataset to average.

    Returns
    -------
    float
        The mean value, or NaN if the dataset is unusable.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calcMean(data1)
    """

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    # Treat +/-inf the same as missing data (NaN) before averaging, so a
    # single infinite value can't blow up the mean.
    data1_msk = np.where(np.isinf(data1['values']))
    data1['values'][data1_msk] = np.nan
    return(np.nanmean(data1['values']))


def calcMax(data1):
    """
    Compute the maximum of a dataset, ignoring NaN and infinite values.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        The dataset to evaluate.

    Returns
    -------
    float
        The maximum value, or NaN if the dataset is unusable.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calcMax(data1)
    """

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    # treat +/-inf the same as missing data before taking the max
    data1_msk = np.where(np.isinf(data1['values']))
    data1['values'][data1_msk] = np.nan
    return(np.nanmax(data1['values']))


def calcMin(data1):
    """
    Compute the minimum of a dataset, ignoring NaN and infinite values.

    Parameters
    ----------
    data1 : dict, list, or numpy.ndarray
        The dataset to evaluate.

    Returns
    -------
    float
        The minimum value, or NaN if the dataset is unusable.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calcMin(data1)
    """

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    # treat +/-inf the same as missing data before taking the min
    data1_msk = np.where(np.isinf(data1['values']))
    data1['values'][data1_msk] = np.nan
    return(np.nanmin(data1['values']))


def convertTempUnits(values, units):
    """
    Convert temperature values between Celsius and Fahrenheit.

    Parameters
    ----------
    values : array_like or float
        Temperature value(s) to convert.
    units : str
        The CURRENT units of ``values`` (recognized C or F name
        variants); values are converted to the other unit.

    Returns
    -------
    array_like or float
        The converted temperature value(s), or ``values`` unchanged if
        ``units`` isn't recognized.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> convertTempUnits(32.0, 'F')
    0.0
    """

    # Whichever unit the input is currently in, convert to the OTHER one
    # (F -> C or C -> F); unrecognized unit strings are passed through
    # unchanged.
    if units.lower() in ['f', 'faren', 'degf', 'fahrenheit', 'fahren', 'deg f']:
        values = convert_temperature(values, 'F', 'C')
        return values
    elif units.lower() in ['c', 'cel', 'celsius', 'deg c', 'degc']:
        values = convert_temperature(values, 'C', 'F')
        return values
    else:
        # print2stdout('Undefined temp units:', units)
        # unrecognized units string, return unchanged
        return values


def filterContourOverTopWater(values, elevations, topwater):
    """
    NaN-out contour values above the actual water surface elevation.

    ResSim duplicates data up to the top of the model domain instead of
    cutting it off at the actual water surface, so this trims the excess.

    Parameters
    ----------
    values : numpy.ndarray
        2-D array of values at each timestep/elevation.
    elevations : numpy.ndarray
        Elevation values used to find the index closest to the water
        surface.
    topwater : array_like
        Water surface elevation at each timestep.

    Returns
    -------
    numpy.ndarray
        The ``values`` array, with everything above each timestep's
        water surface set to NaN.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> filterContourOverTopWater(values, elevations, topwater)
    """

    # For every timestep, find the elevation index closest to the actual
    # water surface, then NaN out everything above it (ResSim otherwise
    # repeats the top-of-domain value up past the real water surface).
    for twi, tw in enumerate(topwater):
        # index of the elevation value nearest the actual water surface
        elevationtopwateridx = (np.abs(elevations - tw)).argmin()
        # everything above the water surface index gets NaN'd out
        values[twi][elevationtopwateridx+1:] = np.nan
    return values


def replaceflaggedValues(Report, settings, itemset, include=[], exclude=[], forjasper=False):
    """
    Recursively replace ``%%flag%%`` placeholders throughout a settings
    structure.

    ``include``/``exclude`` only apply at the first (top) level of
    recursion, i.e. the main keys in the top-level dict, not anything
    nested inside it.

    Parameters
    ----------
    Report : object
        The main Report Generator instance, passed through to the
        single-value flag resolver.
    settings : dict, list, or str
        Settings structure potentially containing flags.
    itemset : str
        Which flag set to resolve against (e.g. ``'general'``,
        ``'modelspecific'``, ``'fancytext'``).
    include : list, optional
        If non-empty, only these top-level keys are processed (default
        ``[]``, meaning all keys).
    exclude : list, optional
        Top-level keys to skip (default ``[]``).
    forjasper : bool, optional
        Passed through to the flag resolver to select Jasper-specific
        formatting for certain flags (default ``False``).

    Returns
    -------
    dict, list, or str
        The ``settings`` structure with flags replaced.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> settings = replaceflaggedValues(Report, settings, 'general')
    """

    if isinstance(settings, str):
        # Base case: a plain string. Only bother calling the (more
        # expensive) single-value replacer if a '%%' flag marker is
        # actually present.
        if '%%' in settings:
            newval = replaceFlaggedValue(Report, settings, itemset, forjasper=forjasper)
            settings = newval
    elif isinstance(settings, dict):
        # recurse into every key of the dict
        for key in settings.keys():
            # include/exclude only apply at this top level of recursion,
            # not to nested dicts/lists reached via recursive calls below.
            if len(exclude) > 0:
                if key in exclude:
                    continue
            if len(include) > 0:
                if key not in include:
                    continue
            if isinstance(settings[key], dict):
                # nested dict, recurse into it directly
                settings[key] = replaceflaggedValues(Report, settings[key], itemset, forjasper=forjasper)
            elif isinstance(settings[key], list):
                # nested list, recurse into each item and rebuild the list
                new_list = []
                for item in settings[key]:
                    new_list.append(replaceflaggedValues(Report, item, itemset, forjasper=forjasper))
                settings[key] = new_list
            else:
                try:
                    if '%%' in settings[key]:
                        # leaf string containing a flag marker, resolve it
                        newval = replaceFlaggedValue(Report, settings[key], itemset, forjasper=forjasper)
                        settings[key] = newval
                except TypeError:
                    # settings[key] wasn't a string (e.g. int/float/bool),
                    # so '%%' in ... raised TypeError; nothing to replace.
                    continue
    elif isinstance(settings, list):
        # recurse into every item of the list, replacing in place by index
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
    """
    Replace inline text-formatting flags with Jasper style tags.

    Currently supports bold, italic, and underline flags in any order
    (e.g. ``%%ui%%``, ``%%i%%``, ``%%bui%%``).

    Parameters
    ----------
    text : str
        Formatted text potentially containing style flags.

    Returns
    -------
    str
        The text with style flags replaced by Jasper-style tags.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> parseForTextFlags('%%b%%Bold text%%/b%%')
    "&#60;style isBold='true'&#62;Bold text&#60;/style&#62;"
    """

    # building blocks for the Jasper-style opening/closing style tags
    start_font_change_front = "&#60;style"
    start_font_change_back = "&#62;"
    end_font_change = "&#60;/style&#62;"
    # maps each single-letter flag to its corresponding Jasper style attribute
    flag_defs = {'b': "isBold='true'",
                 'u': "isUnderline='true'",
                 'i': "isItalic='true'"}
    # Generate every possible ordering/combination of the b/u/i flags
    # (e.g. 'b', 'u', 'i', 'bu', 'ub', 'bui', 'iub', ...) so that a
    # user-written flag like %%ui%% or %%bui%% is recognized regardless
    # of the order the letters were typed in.
    flag_permutations = list(itertools.permutations(flag_defs.keys()))

    # build the full list of possible start/end flag strings from every permutation/combination
    start_flags = []
    end_flags = []
    for flag_permutation in flag_permutations:
        for L in range(1, len(flag_permutation) + 1):
            for subset in itertools.combinations(flag_permutation, L):
                start_flag = f'%%{"".join(subset)}%%'
                end_flag = f'%%/{"".join(subset)}%%'
                if start_flag not in start_flags:
                    # avoid duplicate entries since permutations can produce the same combination
                    start_flags.append(start_flag)
                    end_flags.append(end_flag)

    #find all idx of start flags
    for flag in start_flags:
        flag_idx = [m.start() for m in re.finditer(flag, text)]
        if len(flag_idx) > 0:
            # Build the Jasper-style opening tag with one style attribute
            # per letter in this flag combination (e.g. %%bi%% ->
            # isBold='true' isItalic='true').
            output_from_flag = start_font_change_front
            for flagitem in flag:
                if flagitem != '%':
                    output_from_flag += f' {flag_defs[flagitem]}'
            output_from_flag += start_font_change_back
            # Replace matches back-to-front so earlier replacements don't
            # shift the string indices found for later ones.
            flag_idx.reverse()
            for idx in flag_idx: #do it backwards so the flags don't interupt the idx of each other
                text = text[:idx] + output_from_flag + text[idx + len(flag):]

    # find all idx of end flags
    for flag in end_flags:
        flag_idx = [m.start() for m in re.finditer(flag, text)]
        if len(flag_idx) > 0:
            # same back-to-front replacement strategy as the start flags above
            flag_idx.reverse()
            for idx in flag_idx:
                text = text[:idx] + end_font_change + text[idx + len(flag):]

    return text


def replaceFlaggedValue(Report, value, itemset, forjasper=False):
    """
    Replace known ``%%flag%%`` placeholders in a single string.

    Flags are matched case-insensitively. Uses ``repr(...)[1:-1]`` for
    the replacement value so backslash-containing paths (e.g.
    ``C:/trains``) aren't misinterpreted as escape codes.

    Parameters
    ----------
    Report : object
        The main Report Generator instance, used to look up the actual
        values for recognized flags.
    value : str
        String potentially containing flagged values.
    itemset : {'general', 'modelspecific', 'fancytext'}
        Which flag set to resolve against.
    forjasper : bool, optional
        If ``True``, uses Jasper/XML-safe replacements (e.g. HTML
        entities) for the ``'fancytext'`` itemset (default ``False``).

    Returns
    -------
    str
        The ``value`` string with any recognized flags replaced.

    Raises
    ------
    None
        This function does not explicitly raise exceptions; an
        unrecognized ``itemset`` is logged and returns ``value``
        unchanged.

    Examples
    --------
    >>> replaceFlaggedValue(Report, '%%region%%', 'general')
    ```
    """

    if itemset == 'general':
        # Flags that resolve to general report/study-level information,
        # independent of any specific model/simulation.
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
        # Flags that resolve to the currently-loaded simulation/model
        # alternative's specific settings (DSS file, program type, etc.).
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
        # Symbol/comparison-operator flags; rendered as HTML entities
        # when destined for Jasper (forjasper=True) and as literal
        # unicode characters otherwise (e.g. for plot labels).
        if forjasper:
            # HTML-entity versions for embedding directly in Jasper/XML output
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
            # literal unicode versions for use outside of Jasper (e.g. matplotlib labels)
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
        # unrecognized itemset, log and return unchanged
        print2stderr('Invalid flag itemset: {0}'.format(itemset))
        return value

    for fv in flagged_values.keys():
        # Case-insensitive replace of every occurrence of this flag.
        # repr(...)[1:-1] strips the surrounding quotes from repr() while
        # keeping backslash-escaping intact, so path strings containing
        # sequences like '\t' aren't misinterpreted as escape codes by
        # re.sub's replacement-string handling.
        pattern = re.compile(re.escape(fv), re.IGNORECASE)
        value = pattern.sub(repr(flagged_values[fv])[1:-1], value) #this seems weird with [1:-1] but paths wont work otherwise
    return value


def selectContourByID(contoursbyID, ID):
    """
    Select contour entries matching a given simulation ID.

    Parameters
    ----------
    contoursbyID : dict
        Dictionary of all contours, each entry with an ``'ID'`` key.
    ID : str
        Selected ID (e.g. ``'base'``, ``'alt_1'``).

    Returns
    -------
    dict
        Subset of ``contoursbyID`` whose entries match ``ID``.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> selectContourByID(contoursbyID, 'base')
    """

    # build a filtered dict containing only the entries matching this ID
    output_contours = {}
    for key in contoursbyID:
        if contoursbyID[key]['ID'] == ID:
            output_contours[key] = contoursbyID[key]
    return output_contours


def stackContours(contours, contours_settings):
    """
    Stack multiple contour reaches so they display as a single reach.

    Adds cumulative distances to keep the reaches consistent, and keeps
    track of the distances at which each defined reach transitions to
    the next.

    Parameters
    ----------
    contours : dict
        Dictionary containing reach contour data keyed by reach name.
    contours_settings : dict
        Per-reach settings dictionary; each entry must contain
        ``'distance'``.

    Returns
    -------
    output_values : numpy.ndarray
        Stacked values at each timestep/distance.
    output_dates : numpy.ndarray
        Dates for the data.
    output_distance : numpy.ndarray
        Cumulative distance for each cell center from the source.
    transitions : dict
        Reach name to distance mapping, marking where each reach
        begins.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> values, dates, distance, transitions = stackContours(contours, contours_settings)
    """

    # accumulators built up reach by reach
    output_values = np.array([])
    output_dates = np.array([])
    output_distance = np.array([])
    transitions = {}
    for contourname in contours.keys():
        contour = contours[contourname]
        contour_settings = contours_settings[contourname]
        if len(output_values) == 0:
            # First reach: seed the output with a full copy.
            output_values = pickle.loads(pickle.dumps(contour['values'], -1))
        else:
            # Subsequent reaches: append along the distance axis, but
            # skip the first cell (index 0) since it duplicates the last
            # cell of the previous reach at the junction point.
            output_values = np.append(output_values, contour['values'][1:, :], axis=0)
        if len(output_dates) == 0:
            # dates are shared across reaches, only need to capture them once
            output_dates = contour['dates']
        if len(output_distance) == 0:
            # first reach establishes the baseline distance axis
            output_distance = contour_settings['distance']
            transitions[contourname] = 0
        else:
            # Offset this reach's distances by the running total so far
            # (again skipping the duplicated junction cell), and record
            # where the transition to this reach begins.
            last_distance = output_distance[-1]
            current_distances = contour_settings['distance'][1:] + last_distance
            output_distance = np.append(output_distance, current_distances)
            transitions[contourname] = current_distances[0]
    return output_values, output_dates, output_distance, transitions


def mergeLines(data, data_settings, plot_settings):
    """
    Combine time series lines per the object's mergeline settings.

    Reads ``'mergelines'`` settings, combines the specified data series
    using the configured math operation into the controller line, and
    optionally removes the merged-in lines afterward.

    Parameters
    ----------
    data : dict
        Dictionary of data keyed by flag.
    data_settings : dict
        Per-flag settings dictionary.
    plot_settings : dict
        Object settings, potentially containing a ``'mergelines'`` list.

    Returns
    -------
    data : dict
        The updated data dictionary.
    data_settings : dict
        The updated settings dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> data, data_settings = mergeLines(data, data_settings, plot_settings)
    """

    # keys marked for removal after being merged into their controller
    removekeys = []
    if 'mergelines' in plot_settings.keys():
        # process every configured mergeline group
        for mergeline in plot_settings['mergelines']:
            # normalize the flags being merged to lowercase for comparison
            dataflags = [n.lower() for n in mergeline['flags']]
            if 'controller' in mergeline.keys():
                #Controller matches the flag defined in data[keys]
                controller = mergeline['controller'].lower()
                if controller not in [data_settings[n]['flag'].lower() for n in data.keys()]: #do it this way so if theres comp runs we can still make this work
                    # requested controller isn't actually present in the data, skip this mergeline entirely
                    print2stdout('Mergeline Controller {0} not found in data {1}'.format(controller, data.keys()))
                    print2stdout('Not Running Merge.')
                    continue
            else:
                # No explicit controller: default to the first flag in
                # the mergeline's own flag list.
                controller = data_settings[dataflags[0]]['flag'].lower()
            # otherflags = [data_settings[n]['flag'] for n in dataflags if n != controller]
            # Resolve which data keys correspond to the controller flag
            # vs. the other flags being merged into it (there can be
            # multiple data keys per flag, e.g. one per simulation ID).
            data_keys_with_controller = [n for n in data.keys() if data_settings[n]['flag'].lower() == controller]
            data_keys_for_otherflags = [n for n in data.keys() if data_settings[n]['flag'].lower() != controller
                                        and data_settings[n]['flag'].lower() in dataflags]

            if 'math' in mergeline.keys():
                # explicit math operation given for this mergeline
                math = mergeline['math'].lower()
            else:
                # no operation specified, default to addition
                math = 'add'
                print2stdout('no Mergeline math flag. Set to add by default.')

            # apply the math operation for every controller/other-flag combination
            for datakey_controller in data_keys_with_controller:
                baseunits = data_settings[datakey_controller]['units']
                for datakey_otherflag in data_keys_for_otherflags:
                    if data_settings[datakey_otherflag]['units'] != baseunits:
                        # Warn but proceed anyway; the math operation
                        # below will still run on raw values regardless
                        # of a unit mismatch.
                        print2stdout('WARNING: Attempting to merge lines with differing units')
                        print2stdout('{0}: {1} and {2}: {3}'.format(datakey_otherflag, data[datakey_otherflag]['units'], controller, baseunits))
                        print2stdout('If incorrect, please modify/append input settings to ensure lines '
                              'are converted prior to merging.')
                    # align the controller and other-flag series in time before combining them
                    data[datakey_controller], data[datakey_otherflag] = matchData(data[datakey_controller], data[datakey_otherflag])
                    if data_settings[datakey_controller]['collection']:
                        if data_settings[datakey_otherflag]['collection']:
                            # Both sides are forecast collections: apply
                            # the math per-member across the union of
                            # both collections' members.
                            members = list(set(data_settings[datakey_controller]['members'] + data_settings[datakey_otherflag]['members']))
                            for member in members:
                                data[datakey_controller]['values'][member] = doMathOn2Datasets(
                                    data[datakey_controller]['values'][member],
                                    data[datakey_otherflag]['values'][member], math)
                        else: #add non collection onto a collection
                            # The "other" side is a single series (not a
                            # collection): apply it to every member of
                            # the controller collection.
                            members = data_settings[datakey_controller]['members']
                            for member in members:
                                data[datakey_controller]['values'][member] = doMathOn2Datasets(
                                    data[datakey_controller]['values'][member],
                                    data[datakey_otherflag]['values'], math)
                    else:
                        if data_settings[datakey_otherflag]['collection']:
                            # Can't average/combine a collection down
                            # into a single controller series
                            # unambiguously; refuse this combination.
                            print2stderr(f'Unable to merge collection ({datakey_controller}) onto non collection ({datakey_otherflag}')
                        else:
                            # Simple case: both sides are single series.
                            data[datakey_controller]['values'] = doMathOn2Datasets(data[datakey_controller]['values'],
                                                                                 data[datakey_otherflag]['values'], math)

            if 'keeplines' in mergeline.keys():
                if mergeline['keeplines'].lower() == 'false':
                    # The merged-in lines are no longer needed for
                    # display on their own; mark them for removal.
                    for flag in data_keys_for_otherflags:
                        removekeys.append(flag)
        # remove every merged-away line from both data dictionaries
        for flag in removekeys:
            data.pop(flag)
            data_settings.pop(flag)
    return data, data_settings


def doMathOn2Datasets(data1, data2, math):
    """
    Combine two value arrays element-wise using a named operation.

    Parameters
    ----------
    data1 : numpy.ndarray
        The base/controller values (modified and returned).
    data2 : numpy.ndarray
        The values to combine into ``data1``.
    math : str
        One of ``'add'``, ``'multiply'``, ``'divide'``, or
        ``'subtract'``; any other value leaves ``data1`` unchanged.

    Returns
    -------
    numpy.ndarray
        ``data1`` after applying the requested operation with ``data2``.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> import numpy as np
    >>> doMathOn2Datasets(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 'add')
    array([2., 3.])
    """
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
    """
    Filter data down to a given year (used when splitting plots by year).

    Parameters
    ----------
    data : dict
        Dictionary of data keyed by flag, each with ``'dates'``/
        ``'values'`` arrays.
    year : int, str, or 'ALLYEARS'
        The target year (or "YYYY-YYYY" range string); if
        ``'ALLYEARS'``, no filtering is applied.
    extraflag : str, optional
        An additional array key (e.g. ``'elevations'``) to slice the
        same way as ``'values'``.

    Returns
    -------
    dict
        The filtered ``data`` dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> data = filterDataByYear(data, 2020)
    """

    if year != 'ALLYEARS':
        # only filter if a specific year (or range) was requested
        for flag in data.keys():
            if len(data[flag]['dates']) > 0:
                s_idx, e_idx = getYearlyFilterIdx(data[flag]['dates'], year)
                if None not in [s_idx, e_idx]:
                    # Slice values/dates down to just the target year's
                    # range. Handle both 1-D (single series) and 2-D
                    # (e.g. multi-series/contour) value arrays.
                    if len(data[flag]['values'].shape) == 1:
                        data[flag]['values'] = data[flag]['values'][s_idx:e_idx+1]
                    else:
                        data[flag]['values'] = data[flag]['values'][:,s_idx:e_idx + 1]
                    data[flag]['dates'] = data[flag]['dates'][s_idx:e_idx+1]
                else:
                    # No data falls within the requested year; empty it out.
                    data[flag]['values'] = []
                    data[flag]['dates'] = []
                if extraflag != None:
                    # Apply the same slicing to an additional parallel
                    # array (e.g. elevations) if one was specified.
                    if len(data[flag][extraflag].shape) == 1:
                        data[flag][extraflag] = data[flag][extraflag][s_idx:e_idx+1]
                    else:
                        data[flag][extraflag] = data[flag][extraflag][:, s_idx:e_idx + 1]
    return data


def getYearlyFilterIdx(dates, year):
    """
    Find the start/end array indices covering a given year.

    Parameters
    ----------
    dates : array_like of datetime.datetime
        The full date array to index into.
    year : int or str
        Target year, or a "YYYY-YYYY" range string.

    Returns
    -------
    s_idx : int or None
        Start index for the year (``None`` if the year starts after the
        data ends).
    e_idx : int or None
        End index for the year (``None`` if the year ends before the
        data starts).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> s_idx, e_idx = getYearlyFilterIdx(dates, 2020)
    """

    start_date = dates[0]
    end_date = dates[-1]
    # 'year' can be a single year (int) or a "YYYY-YYYY" range string.
    if isinstance(year, str):
        # parse the start/end years out of the range string
        yrsplit = year.split('-')
        s_year_date = dt.datetime(int(yrsplit[0]),1,1,0,0)
        e_year_date = dt.datetime(int(yrsplit[1]),12,31,23,59)
    else:
        # single year given, use Jan 1 through Dec 31 of that year
        s_year_date = dt.datetime(year,1,1,0,0)
        e_year_date = dt.datetime(year,12,31,23,59)

    if start_date != end_date:
        # Use the regular sampling interval to convert a target
        # date directly into an index via elapsed-time math, rather than
        # scanning the whole array.
        interval = (dates[1] - start_date).total_seconds()
        if start_date.year == s_year_date.year:
            # data already starts in the target year, start from index 0
            s_idx = 0
        elif start_date.year > s_year_date.year: #if the filter year is bigger than the start year (aka data for
            # data starts after the target year even begins, nothing to return
            s_idx = None
        else:
            # compute the start index directly from elapsed time
            s_idx = round(int((s_year_date - start_date).total_seconds() / interval))
            if s_idx < 0:
                s_idx = 0
        if end_date.year == e_year_date.year:
            # data already ends in the target year, go all the way to the end
            e_idx = len(dates)
        elif start_date.year > e_year_date.year:
            # data starts after the target year even ends, nothing to return
            e_idx = None
        else:
            # compute the end index directly from elapsed time
            e_idx = round(int((e_year_date - start_date).total_seconds() / interval))
            if e_idx < 0:
                e_idx = 0

        return s_idx, e_idx
    else:
        # Only a single timestamp in the whole series; treat it as
        # spanning the entire (trivial) range.
        return 0, -1


def getObjectAllYears(years_list):
    """
    Format a list of years into a display string.

    Parameters
    ----------
    years_list : list
        List of years; uses a start-end range format if more than one.

    Returns
    -------
    str
        A single year as a string, or a "start-end" range string for
        multi-year lists.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getObjectAllYears([2020])
    '2020'
    >>> getObjectAllYears([2018, 2020])
    '2018-2020'
    """

    if len(years_list) == 1:
        # single year, just render it directly
        outputstring = str(years_list[0])
    else:
        # multiple years, render as a start-end range string
        outputstring = f'{years_list[0]}-{years_list[1]}'
    return outputstring


def getMonthlyFilterIdx(dates, month):
    """
    Find the start/end array indices covering a given month.

    Parameters
    ----------
    dates : array_like of datetime.datetime
        The full date array to index into.
    month : int
        Target month (1-12).

    Returns
    -------
    s_idx : int
        Start index for the month.
    e_idx : int
        End index for the month.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> s_idx, e_idx = getMonthlyFilterIdx(dates, 4)
    """
    start_date = dates[0]
    end_date = dates[-1]
    # first moment of the target month, in the same year as the data start
    s_month_date = dt.datetime(start_date.year, month,1,0,0)
    if month == 12:
        # December's "next month" would roll into the following year;
        # compute the last second of December directly instead.
        e_month_date = dt.datetime(start_date.year+1,1,1,0,0) - dt.timedelta(seconds=1)
    else:
        # last second of the target month, computed as one second before the next month starts
        e_month_date = dt.datetime(start_date.year,month+1,1,0,0) - dt.timedelta(seconds=1)

    if start_date.month > month or end_date.month < month:
        # target month falls entirely outside the data's range
        print2stdout(f'Desired month prior to start date month. {start_date.month}, {month}')
        return 0, 0

    # Same elapsed-time-to-index conversion approach as getYearlyFilterIdx.
    interval = (dates[1] - start_date).total_seconds()
    if start_date.month == month:
        # data already starts in the target month, start from index 0
        s_idx = 0
    else:
        # compute the start index directly from elapsed time
        s_idx = round(int((s_month_date - start_date).total_seconds() / interval))
    if end_date.month == month:
        # data already ends in the target month, go all the way to the end
        e_idx = len(dates)
    else:
        # compute the end index directly from elapsed time
        e_idx = round(int((e_month_date - start_date).total_seconds() / interval))
    if s_idx < 0:
        # shouldn't normally happen, log for troubleshooting and clamp to 0
        print2stdout(f'SIdx less than zero for {month}. Contact developer.')
        s_idx = 0
    if e_idx < 0:
        # shouldn't normally happen, log for troubleshooting and clamp to 0
        print2stdout(f'EIdx less than zero for {month}. Contact developer.')
        e_idx = 0
    return s_idx, e_idx


def getUnitsList(line_settings, mod=''):
    """
    Collect the units used across a set of lines.

    Parameters
    ----------
    line_settings : dict
        Per-line settings dictionary, each with a units key.
    mod : str, optional
        Prefix for the units key to look up (e.g. ``'y_'`` for
        ``'y_units'``) (default ``''``, i.e. plain ``'units'``).

    Returns
    -------
    list
        List of unit strings used by the lines (excluding ``None``).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getUnitsList(line_settings)
    ['c', 'cfs']
    """

    # collect every non-None units value found across the lines
    units_list = []
    for flag in line_settings.keys():
        # `mod` allows selecting a prefixed units key (e.g. 'y_units'
        # instead of 'units') for secondary-axis lookups.
        units = line_settings[flag][mod+'units']
        if units != None:
            units_list.append(units)
    return units_list


def getUsedIDs(data):
    """
    Find all simulation IDs actually referenced by a data dictionary.

    Parameters
    ----------
    data : dict
        Data dictionary keyed by flag, each with an ``'ID'`` key.

    Returns
    -------
    list
        List of unique IDs found in ``data``.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getUsedIDs(data)
    ['base', 'alt_1']
    """

    # build a list of unique IDs found across every entry in data
    IDs = []
    for key in data.keys():
        ID = data[key]['ID']
        if ID not in IDs:
            IDs.append(ID)
    return IDs


def getAllMonthIdx(timestamp_indexes, i):
    """
    Flatten a given month's timestamp indices across every year.

    Parameters
    ----------
    timestamp_indexes : list
        Nested list of indices, structured as
        ``[year_index][month_index] -> list of indices``.
    i : int
        Month index (0-11) to collect.

    Returns
    -------
    list
        Combined list of indices for month ``i`` across all years.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getAllMonthIdx(timestamp_indexes, 0)
    """

    # accumulate this month's indices across every year in the list
    out_idx = []
    for yearlist in timestamp_indexes:
        # Flatten the requested month's indices across every year.
        out_idx += yearlist[i]
    return out_idx


def ignoreNans(values):
    """
    Return a value array with NaN entries removed.

    Parameters
    ----------
    values : array_like
        Input values.

    Returns
    -------
    numpy.ndarray
        The input values with NaN entries filtered out.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> import numpy as np
    >>> ignoreNans([1.0, np.nan, 3.0])
    array([1., 3.])
    """
    # coerce to an array, then mask out every NaN entry
    v = np.asarray(values)
    return v[~np.isnan(v)]


def getPlotUnits(unitslist, object_settings, axis='x'):
    """
    Determine the units to display for a plot axis.

    Uses the axis's named parameter (if set) to look up units, or falls
    back to the most common unit already present in the plotted data.

    Parameters
    ----------
    unitslist : list
        List of unit strings already in use by the plotted data.
    object_settings : dict
        Plot settings dictionary; checked for ``'parameter'``/
        ``'unitsystem'`` (x axis) or ``'y_parameter'``/
        ``'y_unitsystem'`` (y axis).
    axis : {'x', 'y'}, optional
        Which axis to resolve units for (default ``'x'``).

    Returns
    -------
    str
        The resolved (and translated) unit string, or ``''`` if none
        could be determined.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getPlotUnits(['c', 'c', 'f'], {'parameter': 'temperature'})
    'c'
    """

    # Support both the primary (x) axis and a secondary (y) axis by
    # switching which settings keys are consulted.
    param_flag = 'parameter' if axis == 'x' else 'y_parameter'
    unitsystem_flag = 'unitsystem' if axis == 'x' else 'y_unitsystem'
    if param_flag in object_settings.keys():
        try:
            # A named parameter (e.g. 'temperature') takes priority:
            # look up its unit for the requested unit system (metric by
            # default).
            plotunits = constants.units[object_settings[param_flag].lower()]
            if isinstance(plotunits, dict):
                if unitsystem_flag in object_settings.keys():
                    # explicit unit system requested, resolve to that specific unit
                    plotunits = plotunits[object_settings[unitsystem_flag].lower()]
                else:
                    # default to metric if no unit system was specified
                    plotunits = plotunits['metric']
        except KeyError:
            # parameter name not recognized, nothing to resolve
            plotunits = ''

    elif len(unitslist) > 0:
        # No parameter specified; fall back to whatever unit is most
        # common among the data actually being plotted.
        plotunits = getMostCommon(unitslist)

    else:
        # nothing to go on at all
        plotunits = ''

    # normalize the resolved unit string to its standard abbreviation
    plotunits = translateUnits(plotunits)
    return plotunits


def getMostCommon(listvars):
    """
    Find the most frequently occurring value in a list.

    Parameters
    ----------
    listvars : list
        List of variables to count.

    Returns
    -------
    object or None
        The most common value in the list, or ``None`` if the list is
        empty.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getMostCommon(['c', 'c', 'f'])
    'c'
    """

    # count occurrences of every distinct value in the list
    occurence_count = Counter(listvars)
    if len(occurence_count) == 0:
        # nothing to count at all
        most_common_interval = None
    else:
        # pull out the single most frequent value
        most_common_interval = occurence_count.most_common(1)[0][0]
    return most_common_interval


def translateUnits(units):
    """
    Normalize a unit string to its standard abbreviation.

    Parameters
    ----------
    units : str or None
        Units string to translate.

    Returns
    -------
    str or None
        The standardized unit abbreviation, or the original ``units``
        string if no match was found (or it was ``None``).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> translateUnits('Fahrenheit')
    'f'
    """

    if units != None:
        # search every standard unit's alternate-name list for a case-insensitive match
        for key in constants.unit_alt_names.keys():
            if units.lower().strip() in constants.unit_alt_names[key]:
                return key

    return units


def convertUnitSystem(values, units, target_unitsystem, debug=False):
    """
    Convert values between the english and metric unit systems.

    Parameters
    ----------
    values : array_like, dict, or float
        Value(s) to convert.
    units : str
        Current units of ``values``.
    target_unitsystem : {'english', 'metric'}
        Unit system to convert to.
    debug : bool, optional
        Passed through to logging calls (default ``False``).

    Returns
    -------
    values : array_like, dict, or float
        The converted values if successful, or the original ``values``
        if unsuccessful.
    units : str
        The new converted units if successful, or the original
        ``units`` if unsuccessful.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> values, units = convertUnitSystem(values, 'c', 'english')
    """

    # normalize the current units string first
    units = translateUnits(units)

    english_units = constants.english_units
    metric_units = constants.metric_units

    if units == None:
        # nothing to convert without knowing the current units
        print2stdout('Units undefined.', debug=debug)
        return values, units

    # Determine the target unit string based on the requested unit
    # system, bailing out early (unchanged) if units are already in the
    # target system, unrecognized, or the system itself is invalid.
    if target_unitsystem.lower() == 'english':
        if units.lower() in english_units.keys():
            # current units are metric, translate to their english counterpart
            new_units = english_units[units.lower()]
            print2stdout('Converting {0} to {1}'.format(units, new_units), debug=debug)
        elif units.lower() in english_units.values():
            # already in the target system, nothing to do
            print2stdout('Values already in target unit system. {0} {1}'.format(units, target_unitsystem), debug=debug)
            return values, units
        else:
            # units not recognized at all, can't convert
            print2stdout('Units not found in definitions. Not Converting.', debug=debug)
            return values, units

    elif target_unitsystem.lower() == 'metric':
        if units.lower() in metric_units.keys():
            # current units are english, translate to their metric counterpart
            new_units = metric_units[units.lower()]
            print2stdout('Converting {0} to {1}'.format(units, new_units), debug=debug)
        elif units.lower() in metric_units.values():
            # already in the target system, nothing to do
            print2stdout('Values already in target unit system. {0} {1}'.format(units, target_unitsystem), debug=debug)
            return values, units
        else:
            # units not recognized at all, can't convert
            print2stdout('Units not found in definitions. Not Converting.', debug=debug)
            return values, units

    else:
        # invalid target unit system string given
        print2stdout('Target Unit System undefined.', target_unitsystem, debug=debug)
        print2stdout('Try english or metric', debug=debug)
        return values, units

    if units == new_units:
        # source and target happen to be identical, nothing to convert
        print2stdout('data already in target unit system.', debug=debug)
        return values, units

    if units.lower() in ['c', 'f']:
        # Temperature needs its own conversion formula (not a simple
        # multiplicative factor), so route it through convertTempUnits
        # regardless of value shape (array, per-member dict, or scalar).
        if isinstance(values, (list, np.ndarray)):
            new_values = convertTempUnits(values, units)
        elif isinstance(values, dict):
            # per-member collection, convert each member's values individually
            new_values = {}
            for key, vs in values.items():
                new_values[key] = convertTempUnits(vs, units)
        else:
            # single scalar value
            new_values = convertTempUnits(float(values), units)

    elif units.lower() in constants.conversion.keys():
        # All other supported units are simple multiplicative
        # conversions, looked up from constants.conversion (source unit
        # -> factor to convert to its counterpart system).
        conversion_factor = constants.conversion[units.lower()]
        if isinstance(values, (list, np.ndarray)):
            new_values = values * conversion_factor
        elif isinstance(values, dict):
            # per-member collection, apply the factor to each member individually
            new_values = {}
            for key, vs in values.items():
                new_values[key] = vs * conversion_factor
        else: #must be a single value???
            new_values = float(values) * conversion_factor
    elif new_units.lower() in constants.conversion.keys():
        # The source units weren't directly in the conversion table, but
        # the target units are; use the reciprocal of that factor.
        conversion_factor = 1/constants.conversion[units.lower()]
        if isinstance(values, (list, np.ndarray)):
            new_values = values * conversion_factor
        elif isinstance(values, dict):
            # per-member collection, apply the reciprocal factor to each member
            new_values = {}
            for key, vs in values.items():
                new_values[key] = vs * conversion_factor
        else: #must be a single value???
            new_values = float(values) * conversion_factor
    else:
        # no usable conversion factor found in either direction
        print2stdout('Undefined Units conversion for units {0}.'.format(units), debug=debug)
        print2stdout('No Conversions taking place.', debug=debug)
        return values, units

    return new_values, new_units


def updateFlaggedValues(settings, flaggedvalue, replacevalue):
    """
    Recursively replace a specific flagged value throughout a settings structure.

    Parameters
    ----------
    settings : dict, list, numpy.ndarray, or str
        Settings structure to search and update.
    flaggedvalue : str
        The flag string to look for and replace.
    replacevalue : str
        The value to replace ``flaggedvalue`` with.

    Returns
    -------
    dict, list, numpy.ndarray, or str
        The updated ``settings`` structure.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> updateFlaggedValues(settings, '%%flag%%', 'value')
    """

    # Recursively walk lists, arrays, and dicts, applying the string
    # substitution at every leaf string found.
    if isinstance(settings, list):
        # rebuild the list, recursing into each item
        new_list = []
        for item in settings:
            item = updateFlaggedValues(item, flaggedvalue, replacevalue)
            new_list.append(item)
        return new_list

    if isinstance(settings, np.ndarray):
        # rebuild the array, recursing into each item, preserving the original dtype
        new_list = []
        for item in settings:
            item = updateFlaggedValues(item, flaggedvalue, replacevalue)
            new_list.append(item)
        return np.asarray(new_list, dtype=settings.dtype)

    elif isinstance(settings, dict):
        # recurse into every value in the dict
        for key in settings.keys():
            settings[key] = updateFlaggedValues(settings[key], flaggedvalue, replacevalue)
        return settings

    elif isinstance(settings, str):
        # Case-insensitive replace of the flag; repr()[1:-1] preserves
        # backslashes in replacevalue (e.g. Windows paths) literally
        # rather than having them interpreted as regex escape codes.
        pattern = re.compile(re.escape(flaggedvalue), re.IGNORECASE)
        settings = pattern.sub(repr(replacevalue)[1:-1], settings) #this seems weird with [1:-1] but paths wont work otherwise
        return settings

    else:
        #this gets REALLY noisy.
        #lots is set up to not be replaceable, so uncomment at your own risk
        # print('Cannot set {0}'.format(flaggedvalue))
        # print('Input Not recognized type', settings)
        # Anything else (int, float, bool, None, etc.) can't contain a
        # text flag; return unchanged.
        return settings


def configureUnits(object_settings, parameter, units):
    """
    Resolve the units to use for a line, given its parameter and settings.

    Parameters
    ----------
    object_settings : dict
        Settings dictionary for the current object; checked for
        ``'unitsystem'``.
    parameter : str
        Name of the data parameter (e.g. ``'temperature'``), used to
        look up default units if ``units`` is not already set.
    units : str or None
        Currently known units for the line, if any.

    Returns
    -------
    str or None
        The resolved units string, or ``None`` if it couldn't be
        determined.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> configureUnits({'unitsystem': 'metric'}, 'temperature', None)
    'c'
    """

    if units == None:
        # No explicit units given: try to look them up from the
        # parameter name via the shared constants table.
        try:
            units = constants.units[parameter.lower()]
        except KeyError:
            # parameter not recognized at all
            units = None

    if isinstance(units, dict):
        # constants.units entries are {'metric':..., 'english':...}
        # dicts; resolve to the specific unit string for the requested
        # unit system (or leave undefined if no system was specified).
        if 'unitsystem' in object_settings.keys():
            units = units[object_settings['unitsystem'].lower()]
        else:
            units = None
    return units


def ValueSum(dates, values):
    """
    Sum flow values across gate structures at each timestep.

    Used for finding buzz-plot targets (defined flow sums).

    Parameters
    ----------
    dates : array_like
        List of dates.
    values : list, numpy.ndarray, or dict
        Either a flat array of values (returned unchanged), or a dict
        of per-structure value dicts each containing a ``'q(m3/s)'``
        array.

    Returns
    -------
    numpy.ndarray or list
        The summed flow values at each timestep (or ``values``
        unchanged if it was already a flat array).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> ValueSum(dates, values)
    """

    if isinstance(values, (list, np.ndarray)):
        # Already a single flat series (not per-structure); nothing to
        # sum, return as-is.
        return values
    # accumulate the summed flow at each timestep across every structure
    sum_vals = []
    for i, d in enumerate(dates):
        sum = 0.0
        for sn in values.keys():
            # if values[sn]['elevcl'][i] == target:
            # Sum flow across every structure at this timestep, skipping
            # NaN (inactive/closed) values.
            if not np.isnan(values[sn]['q(m3/s)'][i]):
                sum += values[sn]['q(m3/s)'][i]
        sum_vals.append(sum)
    return np.asarray(sum_vals)


def getObjectYears(Report, object_settings, allowIncludeAllYears=True):
    """
    Determine which year(s)/blocks a plot or table should iterate over.

    If not split by year, ``years`` is set to ``['ALLYEARS']`` to signal
    that all data should be included at once.

    Parameters
    ----------
    Report : object
        The main Report Generator instance.
    object_settings : dict
        Currently selected object settings dictionary; checked for
        ``'splitbyyear'``, ``'yearblocks'``, and ``'includeallyears'``.
    allowIncludeAllYears : bool, optional
        Whether an ``'includeallyears'`` setting should be honored
        (default ``True``).

    Returns
    -------
    split_by_year : bool
        Whether plots/tables should be split up year to year.
    years : list
        List of years (and/or year-block strings, and/or
        ``'ALLYEARS'``) to use.
    yearstr : list
        Matching list of years formatted as strings (e.g. "2013-2016"
        for blocks).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> split_by_year, years, yearstr = getObjectYears(Report, object_settings)
    """

    split_by_year = False
    yearstr = ''
    if 'splitbyyear' in object_settings.keys():
        if object_settings['splitbyyear'].lower() == 'true':
            # split into individual years, using the report's own year list
            split_by_year = True
            years = [int(year) for year in Report.years]
            yearstr = [str(year) for year in years]
    if not split_by_year:
        # Not splitting by year: represent the whole report period as a
        # single "ALLYEARS" pseudo-year.
        yearstr = [Report.years_str]
        years = ['ALLYEARS']

    if 'yearblocks' in object_settings.keys():
        # yearblocks groups consecutive years into multi-year chunks
        # (e.g. blocks of 3: 2010-2012, 2013-2015, ...) in addition to
        # (or instead of) individual years.
        try:
            yearblocks = int(object_settings['yearblocks'])
            startyear = Report.startYear
            endyear = Report.startYear
            if yearblocks > (Report.endYear - Report.startYear + 1):
                # requested block size is larger than the whole reporting period, can't use blocks
                print2stdout(f'Yearblock setting of {yearblocks} is larger than total number of years of {(Report.endYear - Report.startYear + 1)}. Not using yearblocks.', debug=Report.debug)
            else:
                # step through the full report period in yearblocks-sized chunks
                while endyear < Report.endYear:
                    if endyear != Report.startYear:
                        startyear = endyear + 1
                    endyear = startyear + yearblocks - 1 #last year
                    if endyear > Report.endYear:
                        # Clip the final block to the report's actual end
                        # year, so a leftover partial block doesn't run
                        # past the data.
                        endyear = Report.endYear
                    if startyear == endyear:
                        # single-year block, format as a plain year
                        frmtyear = startyear
                    else:
                        # multi-year block, format as a "start-end" range string
                        frmtyear = f'{startyear}-{endyear}'
                    if frmtyear not in years:
                        years.append(frmtyear)
                        yearstr.append(str(frmtyear))

        except TypeError:
            # yearblocks setting wasn't a valid integer
            print2stdout(f"Invalid yearblock value: {object_settings['yearblocks']}", debug=Report.debug)

    if allowIncludeAllYears:
        if 'includeallyears' in object_settings.keys():
            if object_settings['includeallyears'].lower() == 'true':
                # Add an extra "all years combined" entry on top of the
                # individual/blocked years, unless it's already the only
                # entry (avoids a redundant duplicate).
                if 'ALLYEARS' not in years:
                    if len(years) > 1: #if theres only one year in here, please don't do another copy of that..
                        years.append('ALLYEARS')
                        yearstr.append(Report.years_str)

    return split_by_year, years, yearstr


def correctDuplicateLabels(linedata):
    """
    Disambiguate repeated line labels by appending a number.

    Mostly used for comparison plots where "computed" may be used
    several times across different simulations.

    Parameters
    ----------
    linedata : dict
        Dictionary of line settings keyed by flag, each with
        ``'label'`` and ``'numtimesused'`` keys.

    Returns
    -------
    dict
        The updated ``linedata`` dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> linedata = correctDuplicateLabels(linedata)
    """

    # process every line, only renaming later occurrences of a repeated label
    for line in linedata.keys():
        if 'label' in linedata[line].keys():
            curlabel = linedata[line]['label']
            if 'numtimesused' in linedata[line].keys():
                lineidx = linedata[line]['numtimesused']
                if lineidx > 0: #leave the first guy alone..
                    # Only rename lines after the first occurrence of a
                    # given label, so the original label stays clean for
                    # the first line and duplicates get a numeric suffix.
                    for otherline in linedata.keys():
                        if otherline != line:
                            if linedata[otherline]['label'] == curlabel:
                                linedata[line]['label'] = '{0} {1}'.format(curlabel, lineidx) #append the number
    return linedata


def getParameterCount(line, object_settings):
    """
    Get a line's parameter and update the running per-parameter count.

    Parameters
    ----------
    line : dict
        Current line's settings dictionary.
    object_settings : dict
        Currently selected object settings dictionary; checked for/
        updated with a ``'param_count'`` dict.

    Returns
    -------
    param : str or None
        The line's parameter (lowercased), or ``None`` if not set.
    param_count : dict
        The updated running count of each parameter seen so far.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> param, param_count = getParameterCount(line, object_settings)
    """

    if 'param_count' not in object_settings.keys():
        # first time this object has been asked for a parameter count
        param_count = {}
    else:
        param_count = object_settings['param_count']

    if 'parameter' in line.keys():
        param = line['parameter'].lower()
    else:
        # no parameter set on this line at all
        param = None
    if param not in param_count.keys():
        # first time seeing this parameter, start its count at 0
        param_count[param] = 0
    else:
        # already seen this parameter before, increment its count
        param_count[param] += 1

    return param, param_count


def copyKeysBetweenDicts(to_dict, from_dict, ignore=[]):
    """
    Cascade settings from a parent object down to a child (e.g. an axis).

    Lets users define a setting once in the main object flags and have
    it cascade down to all axes, unless overridden in the axis itself.

    Parameters
    ----------
    to_dict : dict
        Settings dictionary to copy into (e.g. an axis's settings).
    from_dict : dict
        Settings dictionary to copy from (e.g. the parent object's
        settings).
    ignore : list, optional
        Keys to skip copying (default ``[]``).

    Returns
    -------
    dict
        The updated ``to_dict`` dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> to_dict = copyKeysBetweenDicts(to_dict, from_dict)
    """

    for key in from_dict.keys():
        if key not in ignore:
            # Only fill in keys that aren't already explicitly set on
            # to_dict, so more-specific (e.g. per-axis) settings always
            # win over the inherited/cascaded ones.
            if key not in to_dict.keys():
                to_dict[key] = from_dict[key]
    return to_dict


def getTimeInterval(times):
    """
    Determine a time series' regular interval by finding the most common gap.

    Parameters
    ----------
    times : array_like of datetime.datetime
        List of timestamps.

    Returns
    -------
    datetime.timedelta or None
        The most common interval between consecutive timestamps, or
        ``None`` if it can't be determined.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getTimeInterval(times)
    datetime.timedelta(seconds=3600)
    """

    # collect the gap between every consecutive pair of timestamps
    t_ints = []
    for i, t in enumerate(times):
        if i == 0: #skip 1
            # nothing to compare the very first timestamp against
            last_time = t
        else:
            # Record the gap between each consecutive pair of timestamps.
            t_ints.append(t - last_time)

    # The most frequently occurring gap is taken as the series' regular
    # interval (robust to occasional gaps/duplicates in the data).
    return getMostCommon(t_ints)


def confirmColor(user_color, default_color, debug=False):
    """
    Validate a user-supplied color, correcting common mistakes.

    Parameters
    ----------
    user_color : str
        Desired color for the line, to validate.
    default_color : str
        Backup color known to be valid, used as a last resort.
    debug : bool, optional
        Passed through to logging calls (default ``False``).

    Returns
    -------
    str
        ``user_color`` if valid; a space-stripped version of it if that
        fixes a common typo; or ``default_color`` if neither works.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> confirmColor('light blue', '#88CCEE')
    'lightblue'
    """

    if not is_color_like(user_color):
        # Common mistake: a color name typed with a stray space (e.g.
        # "light blue" instead of "lightblue"); try stripping spaces
        # before giving up and falling back to the default.
        if not is_color_like(user_color.replace(' ', '')):
            # not fixable by stripping spaces either, fall back to the safe default
            print2stdout('Invalid color with {0}'.format(user_color), debug=debug)
            print2stdout('Replacing with default color', debug=debug)
            return default_color
        else:
            # stripping spaces fixed it, use the corrected version
            print2stdout('Misspelling in color with {0}'.format(user_color), debug=debug)
            print2stdout('Replacing with {0}'.format(user_color.replace(' ', '')), debug=debug)
            return user_color.replace(' ', '')
    else:
        # already a valid color, use it as-is
        return user_color


def fixDuplicateColors(line_settings):
    """
    Resolve a repeated line's color from a color list or the default cycle.

    When doing comparison runs, multiple simulations can end up sharing
    the same line settings; this picks a distinct color for each
    repetition, either from a user-supplied ``linecolors``/
    ``pointfillcolors``/``pointlinecolors`` list, or from the default
    color cycle.

    Parameters
    ----------
    line_settings : dict
        Settings dictionary for the line; must contain
        ``'numtimesused'``, ``'drawline'``, and ``'drawpoints'``.

    Returns
    -------
    dict
        The updated ``line_settings`` dictionary with resolved color
        keys.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> line_settings = fixDuplicateColors(line_settings)
    """

    lineusedcount = line_settings['numtimesused']
    # Wrap the default-color index around the palette length so any
    # number of repeated lines still gets a (repeating) valid color.
    if lineusedcount >= len(constants.def_colors):
        defcol_idx = lineusedcount%len(constants.def_colors)
    else:
        defcol_idx = lineusedcount
    if line_settings['drawline'].lower() == 'true':
        if lineusedcount > 0: #if more than one, the color specified is already used. Use a new color..
            if 'linecolors' in line_settings.keys():
                # A list of colors was given (linecolorS): pick the entry
                # for this repetition, wrapping around if there are more
                # repeats than colors in the list.
                if lineusedcount > len(line_settings['linecolors']):
                    lc_idx = lineusedcount%len(line_settings['linecolors'])
                else:
                    lc_idx = lineusedcount
                try:
                    line_settings['linecolor'] = line_settings['linecolors'][lc_idx]
                except IndexError:
                    # something went wrong indexing into the color list, fall back to the default palette
                    Warning('Index Error in linecolors. Using default color')
                    line_settings['linecolor'] = constants.def_colors[defcol_idx]
            else:
                # No color list given: fall back to the generic default
                # color cycle so repeated lines are still distinguishable.
                line_settings['linecolor'] = constants.def_colors[defcol_idx]

        else: #case where first line, but linecolor isnt defined, but linecolorS is
            #so it used default color INSTEAD of the desired colro...
            if 'linecolors' in line_settings.keys():
                # first occurrence but a color list was still given, use its first entry
                line_settings['linecolor'] = line_settings['linecolors'][0]
            elif 'linecolor' not in line_settings.keys():
                # nothing given at all, fall back to the first default palette color
                line_settings['linecolor'] = constants.def_colors[0]

    if line_settings['drawpoints'].lower() == 'true':
        # Same repeated-line color-cycling logic as above, but for
        # point fill/line colors.
        if lineusedcount > 0: #if more than one, the color specified is already used. Use a new color..
            if 'pointfillcolors' in line_settings.keys():
                if isinstance(line_settings['pointfillcolors'], dict):
                    # A single-item XML list can parse as a dict instead
                    # of a list; normalize it to a one-element list.
                    line_settings['pointfillcolors'] = [line_settings['pointfillcolors']['pointfillcolor']]
                if lineusedcount > len(line_settings['pointfillcolors']):
                    pfc_idx = lineusedcount % len(line_settings['pointfillcolors'])
                else:
                    pfc_idx = lineusedcount
                line_settings['pointfillcolor'] = line_settings['pointfillcolors'][pfc_idx]
            if 'pointlinecolors' in line_settings.keys():
                if isinstance(line_settings['pointlinecolors'], dict):
                    # normalize a single-item XML dict into a one-element list, same as fill colors above
                    line_settings['pointlinecolors'] = [line_settings['pointlinecolors']['pointlinecolor']]
                if lineusedcount > len(line_settings['pointlinecolors']):
                    plc_idx = lineusedcount % len(line_settings['pointlinecolors'])
                else:
                    plc_idx = lineusedcount
                line_settings['pointlinecolor'] = line_settings['pointlinecolors'][plc_idx]

            # If only one of fill/line color ended up set, mirror it to
            # the other so points aren't left with a missing color key;
            # only fall back to the default palette if neither is set.
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
                    # normalize a single-item XML dict into a one-element list
                    line_settings['pointfillcolors'] = [line_settings['pointfillcolors']['pointfillcolor']]
                line_settings['pointfillcolor'] = line_settings['pointfillcolors'][0]
            if 'pointlinecolors' in line_settings.keys():
                if isinstance(line_settings['pointlinecolors'], dict):
                    # normalize a single-item XML dict into a one-element list
                    line_settings['pointlinecolors'] = [line_settings['pointlinecolors']['pointlinecolor']]
                line_settings['pointlinecolor'] = line_settings['pointlinecolors'][0]

    return line_settings


def applyXLimits(Report, dates, values, xlims):
    """
    NaN-out values outside a configured x (date) limit window.

    Parameters
    ----------
    Report : object
        The main Report Generator instance, used for its overall
        StartTime/EndTime.
    dates : array_like
        List of dates (as Julian-date numbers or datetime objects).
    values : numpy.ndarray or dict
        Values aligned with ``dates`` (a plain array, or a dict of
        per-member arrays).
    xlims : dict
        Dictionary of x-limits, with optional ``'min'``/``'max'`` keys.

    Returns
    -------
    dates : array_like
        The (unmodified) dates array.
    values : numpy.ndarray or dict
        The values array, with out-of-range entries set to NaN.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> dates, values = applyXLimits(Report, dates, values, {'min': 'Apr 2014'})
    """

    # Dates can be represented either as Julian-date numbers or as real
    # datetime objects depending on the source; detect which so the
    # xlims min/max strings get parsed into the matching format.
    if isinstance(dates[0], (int, float)):
        wantedformat = 'jdate'
    elif isinstance(dates[0], dt.datetime):
        wantedformat = 'datetime'
    if 'min' in xlims.keys():
        # resolve the minimum limit into the same date format used by this series
        datemin = WT.translateDateFormat(xlims['min'], wantedformat, Report.StartTime, Report.StartTime, Report.EndTime)
        for i, d in enumerate(dates):
            if datemin > d:
                # NaN out (rather than remove) anything before the
                # minimum, preserving array length/alignment with dates.
                if isinstance(values, (int, np.ndarray)):
                    values[i] = np.nan #exclude
                elif isinstance(values, dict):
                    # per-member collection, NaN out this index for every member
                    for key in values.keys():
                        values[key][i] = np.nan
    if 'max' in xlims.keys():
        # resolve the maximum limit into the same date format used by this series
        datemax = WT.translateDateFormat(xlims['max'], wantedformat, Report.EndTime,
                                     Report.StartTime, Report.EndTime)
        for i, d in enumerate(dates):
            if datemax < d:
                # same NaN-out treatment as the minimum limit above
                if isinstance(values, (int, np.ndarray)):
                    values[i] = np.nan #exclude
                elif isinstance(values, dict):
                    for key in values.keys():
                        values[key][i] = np.nan

    return dates, values


def applyYLimits(dates, values, ylims):
    """
    NaN-out values outside a configured y (value) limit window.

    Parameters
    ----------
    dates : array_like
        List of dates (unmodified; returned for API symmetry with
        ``applyXLimits``).
    values : numpy.ndarray
        Values to filter.
    ylims : dict
        Dictionary of y-limits, with optional ``'min'``/``'max'`` keys.

    Returns
    -------
    dates : array_like
        The (unmodified) dates array.
    values : numpy.ndarray
        The values array, with out-of-range entries set to NaN.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> dates, values = applyYLimits(dates, values, {'min': '0', 'max': '100'})
    """

    if 'min' in ylims.keys():
        # NaN-out every value below the configured minimum
        for i, v in enumerate(values):
            if float(ylims['min']) > v:
                values[i] = np.nan #exclude

    if 'max' in ylims.keys():
        # NaN-out every value above the configured maximum
        for i, v in enumerate(values):
            if float(ylims['max']) < v:
                values[i] = np.nan #exclude

    return dates, values


def getGateOperationTimes(gatedata):
    """
    Find timestamps where any gate's operating status changed.

    Parameters
    ----------
    gatedata : dict
        Dictionary of gate operation data, keyed by gate level then
        ``'gates'`` then gate name.

    Returns
    -------
    numpy.ndarray
        Dates where the combined (any-gate-operating) status flipped
        between closed and open.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getGateOperationTimes(gatedata)
    """

    # accumulate transition indices across every gate level/group
    operationIndex = np.array([], dtype=int)
    for gatelevel in gatedata.keys():
        # use the first gate in this group just to get the correct array length
        gate0 = list(gatedata[gatelevel]['gates'].keys())[0]
        gateops_datamask = np.zeros(len(gatedata[gatelevel]['gates'][gate0]['values']), dtype=bool) #assume everything closed
        for gi, gate in enumerate(gatedata[gatelevel]['gates']):
            curgate = gatedata[gatelevel]['gates'][gate]
            # A non-NaN value means this gate was operating at this
            # timestep; OR the masks together so the combined mask marks
            # "at least one gate in this group was operating".
            msk = ~np.isnan(curgate['values'])
            gateops_datamask = gateops_datamask | msk #change when differnt

        # Find every index where the combined operating-status flips
        # (closed->open or open->closed) - these are the "interesting"
        # transition timestamps to report.
        operationIndex = np.append(operationIndex, np.where(gateops_datamask[:-1] != gateops_datamask[1:])[0])

    return curgate['dates'][np.unique(operationIndex)]


def matcharrays( array1, array2):
    """
    Recursively align array1's shape to match array2's nesting.

    Handles variable/ragged input so that, for example, lists of lists
    representing a single date's profile values get aligned so each
    elevation value has an associated date for easy tabular output.

    Parameters
    ----------
    array1 : array_like
        Values (or dates) to align; generally the "shorter"/less-nested
        array (e.g. one date per profile).
    array2 : array_like
        Values (or dates) to align against; generally the "longer"/
        more-nested array (e.g. per-elevation values).

    Returns
    -------
    array_like
        ``array1`` reshaped/broadcast to align with ``array2``'s
        nesting structure.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> matcharrays(dates, values)
    """

    if isinstance(array1, (list, np.ndarray)) and isinstance(array2, (list, np.ndarray)):
        if len(np.asarray(array1, dtype=object).shape) < len(np.asarray(array2, dtype=object).shape):
            # array2 is "deeper" (more nested) than array1, e.g. array1
            # is a flat list of dates and array2 is a list of per-date
            # value arrays: repeat each array1 entry once per item in the
            # corresponding array2 sub-array so lengths match up.
            if len(array1) == 0:
                # nothing in array1 at all, build a matching NaN-filled placeholder
                new_array1 = np.full_like(array2, fill_value=np.nan)
            else:
                # repeat each array1 value once per item in its matching array2 sub-array
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
                # Single date, multiple value sub-arrays: recursively
                # broadcast that one date across every sub-array.
                new_array1 = []
                for subarray2 in array2:
                    new_array1.append(matcharrays(array1[0], subarray2))
                return new_array1
            elif len(array1) == 0: #no data
                # No dates at all: broadcast an empty placeholder instead.
                new_array1 = []
                for subarray2 in array2:
                    new_array1.append(matcharrays('', subarray2))
                return new_array1

            else:
                # shouldn't normally happen given the offset-length assumption above
                print2stdout('ERROR') #If the Len of the arrays are offset, then there should only ever be 1 date
        elif len(array1) == len(array2):
            # Same top-level length: recurse pairwise into each matching
            # sub-array in case further nested alignment is needed.
            new_array1 = []
            for i, subarray1 in enumerate(array1):
                new_array1.append(matcharrays(subarray1, array2[i]))
            return new_array1
        else:
            # array1 is longer than array2 - shouldn't normally happen;
            # truncate array1 down to array2's length as a fallback.
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
        # Base recursion case: a single scalar being broadcast across a
        # list - repeat it once per item, recursing further if a given
        # array2 item is itself a sub-array.
        new_array1 = []
        for subarray2 in array2:
            if isinstance(subarray2, (list, np.ndarray)):
                # this item is itself nested, recurse further
                new_array1.append(matcharrays(array1, subarray2))
            else:
                # plain scalar item, just repeat the value directly
                new_array1.append(array1)
        return new_array1

    else:
        # Nothing left to align (e.g. array2 isn't a list/array); return
        # array1 unchanged.
        return array1


def pickByParameter(values, line):
    """
    Select the correct parameter's column from multi-parameter W2 results.

    Some data sources (CE-QUAL-W2 structure results) return multiple
    parameters from a single results file; this picks out the right one
    based on the line's ``'parameter'`` setting.

    Parameters
    ----------
    values : dict
        Dictionary of values keyed by raw W2 column name.
    line : dict
        Line settings dictionary; checked for ``'parameter'``.

    Returns
    -------
    array_like
        The values for the selected parameter (or the first available
        set of values if the parameter wasn't specified/recognized).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> pickByParameter(values, {'parameter': 'temperature'})
    """

    # Maps a human-readable parameter name to the raw column key used in
    # CE-QUAL-W2's structure-result files.
    w2_param_dict = {'temperature': 't(c)',
                     'elevation': 'elevcl',
                     'flow': 'q(m3/s)'}

    if 'parameter' not in line.keys():
        # no parameter specified at all, fall back to the first available column
        print2stdout("Parameter not set for line.")
        print2stdout("using the first set of values, {0}".format(list(values.keys())[0]))
        return values[list(values.keys())[0]]
    else:
        if line['parameter'].lower() not in w2_param_dict.keys():
            # parameter given but not recognized, fall back to the first available column
            print2stdout('Parameter {0} not found in dict in pickByParameter(). {1}'.format(line['parameter'].lower(), w2_param_dict.keys()))
            print2stdout("using the first set of values, {0}".format(list(values.keys())[0]))
            return values[list(values.keys())[0]]
        else:
            # parameter recognized, look up its raw column key and return that column
            p = line['parameter'].lower()
            param_key = w2_param_dict[p]
            return values[param_key]


def prioritizeKey(firstchoice, secondchoice, key, backup=None):
    """
    Look up a key in two settings dicts, preferring the first.

    Parameters
    ----------
    firstchoice : dict
        Dictionary to check first.
    secondchoice : dict
        Dictionary to check if ``key`` isn't in ``firstchoice``.
    key : str
        Settings key to look up.
    backup : object, optional
        Value to return if ``key`` is in neither dict (default
        ``None``).

    Returns
    -------
    object
        The value found, or ``backup`` if not found in either
        dictionary.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> prioritizeKey({'a': 1}, {'a': 2, 'b': 3}, 'b')
    3
    """

    if key in firstchoice:
        # first choice wins if it has the key
        return firstchoice[key]
    elif key in secondchoice:
        # fall back to the second choice
        return secondchoice[key]
    else:
        # neither dict has it, use the backup value
        return backup


def getListItems(listvals):
    """
    Recursively flatten a nested list/array structure into a flat array.

    Parameters
    ----------
    listvals : list, numpy.ndarray, or dict
        Value structure to flatten. If a dict, delegates to
        ``getListItemsFromDict``.

    Returns
    -------
    numpy.ndarray or dict
        The flattened values (an array for list/array input, or a
        flattened dict for dict input).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getListItems([[1, 2], [3, 4]])
    array([1., 2., 3., 4.])
    """

    if isinstance(listvals, (list, np.ndarray)):
        # accumulate flattened values here
        outvalues = np.array([])
        for item in listvals:
            if isinstance(item, (list, np.ndarray)):
                # Nested list found: flatten it recursively and append
                # its values one at a time.
                vals = getListItems(item)
                for v in vals:
                    outvalues = np.append(outvalues, v)
                    # outvalues.append(v)
            else:
                # As soon as we hit a plain (non-nested) list, it's
                # already flat, so just return it directly instead of
                # rebuilding it element-by-element.
                return listvals #we just have a list of values, so we're good! return list
    elif isinstance(listvals, dict):
        # delegate dict flattening to the dedicated helper
        outvalues = getListItemsFromDict(listvals)
    return outvalues


def cleanFileName(csvname):
    """
    Sanitize a string for use as a file name.

    Parameters
    ----------
    csvname : str
        Potential file name to sanitize.

    Returns
    -------
    str
        The sanitized file name, with invalid characters replaced by
        underscores.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> cleanFileName('bad:name?.csv')
    'bad_name_.csv'
    """

    # match any character that isn't a word character, dash, underscore, dot, or space
    pattern = r'[^\w\-_\. ]'
    # replace invalid characters with underscores
    sanitized_file_name = re.sub(pattern, '_', csvname)
    return sanitized_file_name


def getListItemsFromDict(indict):
    """
    Recursively flatten a dict of (possibly nested) lists for logging.

    Parameters
    ----------
    indict : dict
        Dictionary of values, potentially with nested dicts.

    Returns
    -------
    dict
        Flattened dictionary; nested dict keys are prefixed with their
        parent key (e.g. ``'member1_values'``).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getListItemsFromDict(indict)
    """

    outdict = {}
    for key in indict:
        if isinstance(indict[key], dict):
            # Nested dict: flatten recursively and prefix each resulting
            # key with the parent key (e.g. 'member1_values') so nested
            # structure is preserved in the flat output.
            returndict = getListItemsFromDict(indict[key])
            returndict = {'{0}_{1}'.format(key, newkey): returndict[newkey] for newkey in returndict}
            for key in returndict.keys():
                outdict[key] = returndict[key]
        elif isinstance(indict[key], (list, np.ndarray)):
            # leaf list/array value, copy it directly into the flat output
            outdict[key] = indict[key]
    return outdict


def NaNOmittedValues(values, omitval, debug):
    """
    Replace a specific sentinel value in a series with NaN.

    The sentinel value varies by data source (e.g. -99999, 0, 100), so
    it's passed in explicitly rather than hardcoded.

    Parameters
    ----------
    values : dict, list, or numpy.ndarray
        Array (or dict of arrays) of values to filter.
    omitval : float
        The sentinel value to replace with NaN.
    debug : bool
        Passed through to logging calls.

    Returns
    -------
    dict, list, or numpy.ndarray
        The updated values, with matching entries set to NaN.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> import numpy as np
    >>> NaNOmittedValues(np.array([1.0, -99999.0]), -99999.0, False)
    array([ 1., nan])
    """

    if isinstance(values, dict):
        # per-member collection, apply the same omit logic to each member independently
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
            # nothing to filter on an empty array
            print2stdout('No Values to omit.', debug=debug)
            return values


def replaceDefaults(Report, default_settings, object_settings):
    """
    Merge user-defined settings over defaults, replacing flags first.

    Deep-copies both inputs so no settings are accidentally shared
    between calls, resolves ``%%flag%%`` placeholders, then overlays
    every user-defined setting onto the defaults.

    Parameters
    ----------
    Report : object
        The main Report Generator instance, used for flag resolution.
    default_settings : dict
        Default object settings dictionary.
    object_settings : dict
        User-defined settings dictionary.

    Returns
    -------
    dict
        The merged settings dictionary (defaults overridden by user
        settings), with a ``'replaced_defaults'`` key listing which
        keys were overridden.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> merged = replaceDefaults(Report, default_settings, object_settings)
    """

    # Deep-copy both inputs (via pickle round-trip) after resolving any
    # %%flag%% placeholders, so mutating the result below can't leak back
    # into the caller's original default/object settings dicts.
    default_settings = pickle.loads(pickle.dumps(replaceflaggedValues(Report, default_settings, 'general'), -1))
    object_settings = pickle.loads(pickle.dumps(replaceflaggedValues(Report, object_settings, 'general'), -1))
    # track which keys ended up overridden by the user, for debugging/traceability
    replaced_flags = []
    for key in object_settings.keys():
        if key not in default_settings.keys(): #if defaults doesnt have key
            # default doesn't have this key at all, add it directly
            default_settings[key] = object_settings[key]
            replaced_flags.append(key)
        elif default_settings[key] == None: #if defaults has key, but is none
            # default has the key but it's unset, fill it in with the user's value
            default_settings[key] = object_settings[key]
            replaced_flags.append(key)
        elif isinstance(object_settings[key], list): #if settings is a list, aka rows or lines
            # if key.lower() == 'rows': #if the default has rows defined, just overwrite them.
            # User-defined lists (e.g. rows/lines) always fully replace
            # the default's list rather than merging item-by-item.
            if key in default_settings.keys():
                default_settings[key] = object_settings[key]
                replaced_flags.append(key)
            elif key.lower() not in default_settings.keys():
                default_settings[key] = object_settings[key] #if the defaults dont have anything defined, fill it in
                replaced_flags.append(key)
        else:
            # Any other user-defined setting simply overrides the default.
            default_settings[key] = object_settings[key]
            replaced_flags.append(key)

    # Keep a record of which keys were overridden by the user, for
    # debugging/traceability.
    default_settings['replaced_defaults'] = replaced_flags
    return default_settings


def getDateSourceFlag(object_settings):
    """
    Get the configured date source setting for an object.

    Parameters
    ----------
    object_settings : dict
        Currently selected object settings dictionary.

    Returns
    -------
    str, dict, or list
        The ``'datessource'`` setting if defined, otherwise an empty
        list (signaling that timesteps should be auto-generated).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getDateSourceFlag({'datessource': 'dss'})
    'dss'
    """

    if 'datessource' in object_settings.keys():
        # explicit date source configured, use it directly
        datessource_flag = object_settings['datessource'] #determine how you want to get dates? either flag or list
    else:
        # no explicit source, signal the caller to auto-generate timesteps
        datessource_flag = [] #let it make timesteps

    return datessource_flag


def getMaxWSEFromElev(input_data):
    """
    Get the maximum elevation from a set of values.

    Parameters
    ----------
    input_data : array_like
        Elevation values over a timeseries/profile.

    Returns
    -------
    float
        The maximum elevation, or NaN if ``input_data`` is empty.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getMaxWSEFromElev([100.0, 105.0, 98.0])
    105.0
    """

    try:
        return max(input_data)
    except ValueError:
        # max() on an empty sequence raises ValueError; treat that as
        # "no data" rather than crashing.
        return np.nan
    # elevations = []
    # for e in input_data:
    #     elevations.append(max(e))
    # return elevations


def formatUnitsStrings(units, format='internal'):
    """
    Format a unit string with its special/fancy symbol representation.

    Parameters
    ----------
    units : str or None
        The unit string to format.
    format : {'internal', 'external'}, optional
        Which symbol set to use: ``'internal'`` for unicode symbols
        (Python/matplotlib rendering), ``'external'`` for HTML entities
        (Jasper/XML rendering) (default ``'internal'``).

    Returns
    -------
    str or None
        The formatted unit string, or the original ``units`` if no
        special formatting was found (or ``units`` was ``None``).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> formatUnitsStrings('f')
    '\u00b0F'
    """

    if units == None:
        # nothing to format
        return units
    if format == 'internal':
        # Internal (Python/matplotlib) rendering uses actual unicode
        # degree symbols.
        units_list = constants.units_fancy_flags_internal
    elif format == 'external':
        # External (Jasper/XML) rendering uses HTML numeric entities
        # instead, since the degree symbol may not survive XML encoding.
        units_list = constants.units_fancy_flags_external

    if units.lower() in units_list.keys():
        # found a fancy representation for this unit, use it
        output = units_list[units.lower()]
    else:
        # no fancy representation defined, use the plain unit string as-is
        output = units
    return output


def formatTextFlags(text):
    """
    Replace literal escape-sequence text with actual control characters.

    Handles text that was read in with a literal backslash-n
    (``"\\n"``) rather than an actual newline character, converting it
    to a real ``\n`` so it renders correctly. Tab and carriage-return
    replacements are intentionally left disabled (see commented-out
    lines) because they render as square/placeholder glyphs in LaTeX
    output.

    Parameters
    ----------
    text : str
        Text potentially containing literal escape sequences.

    Returns
    -------
    str
        Text with recognized escape sequences replaced.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> formatTextFlags('line one\\\\nline two')
    'line one\\nline two'
    """
    # only literal backslash-n is currently converted; tab/carriage-return are disabled (see docstring)
    flags = {"\\n": "\n"
             # "\\t": "\t", #replaces with square for laTex reasons
             # "\\r": "\r", #replaces with square for laTex reasons
             }
    for key, fixed in flags.items():
        text = text.replace(key, fixed)
    return text


def formatMembers(member):
    """
    Format ensemble member identifiers into 6-character DSS notation.

    Parameters
    ----------
    member : re.Match, list, numpy.ndarray, str, or int
        A single member, a list/array of members, or a regex match
        object (when called from within a ``re.sub`` callback).

    Returns
    -------
    str or list of str
        The zero-padded 6-character member identifier(s).

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> formatMembers(5)
    '000005'
    """

    if isinstance(member, re.Match):
        # Called from within a re.sub callback: extract the first
        # capture group and zero-pad it.
        return member.group(1).zfill(6)
    elif isinstance(member, (np.ndarray, list)):
        # format every member in the list/array individually
        frmted_members = []
        for me in member:
            frmted_members.append(str(me).zfill(6))
        return frmted_members
    else:
        # single scalar member, format directly
        return str(member).zfill(6)


def matchMemberToEnsembleSet(ensemblesets, member):
    """
    Find which ensemble set a given member belongs to.

    Parameters
    ----------
    ensemblesets : list of dict
        Collection of ensemble set dictionaries, each with a
        ``'members'`` list.
    member : str or int
        Member identifier to look up.

    Returns
    -------
    dict
        The matching ensemble set dictionary, or an empty dict if no
        match is found.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> matchMemberToEnsembleSet(ensemblesets, 'member_1')
    """

    # scan every ensemble set for one containing the requested member
    for ensembleset in ensemblesets:
        if member in ensembleset['members']:
            return ensembleset
    # no matching ensemble set found
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

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getOriginalMemberNumber(member, ensembleset, s_dss_file, s_f_part, o_start_time, o_end_time, False)
    """

    # Check if there is a final schedule number in the dss file. This will only happen for the iterative W2 model
    # the path where the schedule number would be if it exists
    s_final_schedule_path = f"//W2_FOLSOM_SCHEDULE_FINAL/Count/01Apr2024/1Hour/{s_f_part}/"

    # try and pull the values
    times, values, units = WR.readDSSData(s_dss_file, s_final_schedule_path, o_start_time, o_end_time, b_debug)

    # if the values array is non-empty return the first non nan value
    if len(values) > 0:
        # found an actual schedule value in DSS, use the first valid one directly
        return str(int(values[~np.isnan(values)][0]))

    # get the index where the current member number is
    member_index = ensembleset['members'].index(member)

    # split the original members into a list
    sl_original_members = ensembleset['memberstoreport'].split(', ')

    # get the original member number
    s_original_member = sl_original_members[member_index]

    return s_original_member


def formatNumbers(number, numberformatsettings):
    """
    Format a number to the correct number of decimal places by rule.

    Parameters
    ----------
    number : float or object
        Number to be formatted; non-numeric or NaN input is returned
        unchanged.
    numberformatsettings : list of dict
        Ordered list of formatting rules, each optionally with
        ``'decimalplaces'``, ``'min'``, and/or ``'max'`` keys.

    Returns
    -------
    str or object
        The formatted number string (comma-separated with the matching
        rule's decimal places), or the original ``number`` if it wasn't
        numeric.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> formatNumbers(1234.5678, [{'max': '10000', 'decimalplaces': '2'}])
    '1,234.57'
    """

    try:
        number = float(number)
    except:
        # Not a numeric value at all (e.g. already a formatted string);
        # return it unchanged.
        return number
    if np.isnan(number):
        # NaN can't be meaningfully formatted, return as-is
        return number

    # Walk the ordered list of formatting rules and use the first one
    # whose min/max range (if any) matches this number's magnitude.
    for numberformat in numberformatsettings:
        if 'decimalplaces' in numberformat.keys():
            decplaces = int(numberformat['decimalplaces'])
            if 'max' in numberformat.keys() and 'min' in numberformat.keys():
                # rule has both bounds, check the number falls within them
                if float(numberformat['min']) < abs(number) <= float(numberformat['max']):
                    # print2stdout(f'Number {number} with settings {numberformat}')
                    return '{num:,.{digits}f}'.format(num=number, digits=decplaces)

            elif 'max' in numberformat.keys() and 'min' not in numberformat.keys():
                # rule only has an upper bound
                if abs(number) <= float(numberformat['max']):
                    return '{num:,.{digits}f}'.format(num=number, digits=decplaces)

            elif 'min' in numberformat.keys() and 'max' not in numberformat.keys():
                # rule only has a lower bound
                if float(numberformat['min']) < abs(number):
                    return '{num:,.{digits}f}'.format(num=number, digits=decplaces)
            else:
                # No min or max given: this rule applies unconditionally.
                return '{num:,.{digits}f}'.format(num=number, digits=decplaces)

    # No matching rule found; fall back to 2 decimal places.
    return f'{number:,.2f}'


def replaceAllFlags(Report, text):
    """
    Replace every ``%%flag%%`` placeholder in text with its resolved value.

    Parameters
    ----------
    Report : object
        The main Report Generator instance.
    text : str
        Text to parse for flags (used in text boxes).

    Returns
    -------
    str
        The text with every recognized flag replaced.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> replaceAllFlags(Report, 'Region: %%region%%')
    ```
    """

    # First pass: replace flags that don't depend on any particular
    # simulation ID (symbol/comparison flags and general report flags).
    text = replaceflaggedValues(Report, text, 'fancytext', forjasper=True) #these are text formatted and dont matter
    text = replaceflaggedValues(Report, text, 'general', forjasper=True) #these should be the same for ALL IDs

    # remember the currently loaded ID so it can be restored after per-ID flag resolution
    starting_ID = Report.currentlyloadedID
    # Find every remaining %%...%% flag left in the text so each can be
    # resolved individually (some may need a specific simulation ID
    # loaded first).
    flag_objects = list(set(re.findall(r'%%(.*?)%%', text)))
    for fo in flag_objects:
        original_str = '%%{0}%%'.format(fo)

        if not Report.modelIndependent: #need to make sure its model independent and there are IDs to load
            if len(fo.split('.')) > 1:  # if its longer than 2 then its wanting a specific ID
                # A flag like '%%SimulationName.alt_1%%' targets a
                # specific simulation ID; temporarily switch the report's
                # active ID before resolving the flag.
                wanted_ID = fo.split('.')[-1]
                if Report.currentlyloadedID != wanted_ID and wanted_ID in Report.All_IDs:
                    Report.loadCurrentID(wanted_ID)
                if wanted_ID in Report.All_IDs: #if it is an ID flag, we want to trim it off
                    fo = '.'.join(fo.split('.')[:-1])
            flagged_text = '%%{0}%%'.format(fo) #format it back like its a flag, but we've got the right alts loaded

            # resolve the model-specific flag now that the right ID (if any) is loaded
            flagged_text = replaceflaggedValues(Report, flagged_text, 'modelspecific', forjasper=True)
            flagged_text = formatDescriptionsForPrint(Report, flagged_text)

            text = text.replace(original_str, flagged_text)

            if starting_ID != Report.currentlyloadedID:
                # Restore whichever ID was active before this flag's
                # resolution needed a different one.
                Report.loadCurrentID(starting_ID)

        else:
            # Model-independent report: no per-ID switching needed.
            flagged_text = '%%{0}%%'.format(fo)
            flagged_text = formatDescriptionsForPrint(Report, flagged_text)
            text = text.replace(original_str, flagged_text)

    return text


def formatDescriptionsForPrint(Report, text):
    """
    Replace ``%%description.object%%`` flags with their resolved descriptions.

    Parameters
    ----------
    Report : object
        The main Report Generator instance.
    text : str
        Text potentially containing description flags.

    Returns
    -------
    str
        The text with description flags replaced by formatted
        descriptions.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> formatDescriptionsForPrint(Report, '%%description.study%%')
    ```
    """

    #format should be %%description.object%% where object is the object to get the description from
    desc_objects = list(set(re.findall(r'%%description\.(.*?)%%', text)))
    for do in desc_objects:
        do_low = do.lower()
        desciption_str = '%%description.{0}%%'.format(do)
        # look up and format the requested description, then substitute it in
        desc = getDescription(Report, do_low)
        desc = formatDescription(desc)
        text = text.replace(desciption_str, desc)
    return text


def formatDescription(description):
    """
    Clean up whitespace and text flags in a description for printing.

    Parameters
    ----------
    description : str or None
        Description text to format.

    Returns
    -------
    str
        The formatted description (empty string if ``description`` was
        ``None``), with each line's leading/trailing whitespace
        stripped and text flags resolved.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> formatDescription('  line one  \\n  line two  ')
    'line one\\nline two'
    """
    if description == None:
        # nothing to format
        return ''

    # Strip leading/trailing whitespace from each line independently
    # (rather than the whole block at once) so indentation quirks in the
    # source XML don't carry through to the printed description.
    desc_split = description.split('\n')#handle newlines
    desc_list = []
    for item in desc_split:
        desc_list.append(item.strip())
    description = '\n'.join(desc_list)

    return formatTextFlags(description)


def getDescription(Report, do):
    """
    Get a named description from the Report object.

    Parameters
    ----------
    Report : object
        The main Report Generator instance.
    do : str
        Name of the description object to fetch (e.g. ``'study'``,
        ``'simulationgroup'``, ``'simulation'``, ``'watalternative'``,
        ``'analysisperiod'``, ``'modelalternative'``).

    Returns
    -------
    str
        The requested description, or an empty string if ``do`` isn't
        recognized.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getDescription(Report, 'study')
    ```
    """

    # starting_ID = Report.currentlyloadedID
    # if len(do.split('.')) > 1: #if its longer than 2 then its wanting a specific ID
    #     wanted_ID = do.split('.')[-1]
    #     if starting_ID != wanted_ID:
    #         Report.loadCurrentID(wanted_ID)
    #     do = '.'.join(do.split('.')[:-1])

    desc = ''

    # Map each recognized description-object name to the corresponding
    # attribute on the Report object.
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
        # unrecognized description object name
        print2stderr('No description found for object: {0}'.format(do))
        desc = ''

    # if starting_ID != Report.currentlyloadedID:
    #     Report.loadCurrentID(starting_ID)

    return desc


def checkJasperFiles(study_dir, install_dir):
    """
    Delete stale compiled Jasper files whose .jrxml source is newer.

    Parameters
    ----------
    study_dir : str
        Directory to look for study-specific Jasper/jrxml files.
    install_dir : str
        Directory to look for default install-provided jrxml files.

    Returns
    -------
    None
        Deletes stale ``.jasper`` files from disk as needed.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> checkJasperFiles('/path/to/study', '/path/to/install')
    """

    #JRXML files can exist in two places, study and install. Study overwrites install.
    jrxml_study_directory = os.path.join(study_dir, 'reports', 'Jasper')
    jrxml_install_directory = os.path.join(install_dir, 'reports', 'Jasper')

    if os.path.exists(jrxml_study_directory): #if the study dir exsits
        # study directory exists, list its files
        files_in_study_directory = os.listdir(jrxml_study_directory) #then get files in study dir
    else:
        files_in_study_directory = [] #otherwise, there are none

    if os.path.exists(jrxml_install_directory): #then check the install dir
        # install directory exists, list its files
        files_in_install_directory = os.listdir(jrxml_install_directory) #get install dir files
    else:
        files_in_install_directory = [] #otherwise there are none

    # narrow both directory listings down to just the .jrxml source files
    jrxml_study_files = [file for file in files_in_study_directory if file.endswith('.jrxml')] #get jrxml files
    jrxml_install_files = [file for file in files_in_install_directory if file.endswith('.jrxml')] #default included in install

    for jrxml_file in jrxml_install_files: #should contain ALL files, as this is the base set
        # build the path to the corresponding compiled jasper file
        jasper_file = os.path.join(study_dir, 'reports', 'JasperC', jrxml_file.split('.jrxml')[0] + '.jasper') #link to where compiled jasper file would be

        if jrxml_file in jrxml_study_files: #if the jrxml file is in the study dir, use that one
            # study version overrides the install version, per the priority note above
            jrxml_source = study_dir
        else: #otherwise use the one in the study fir
            jrxml_source = install_dir

        if os.path.exists(jasper_file):
            # compare source vs. compiled modification times
            jrxml_time = os.path.getmtime(os.path.join(jrxml_source, 'reports', 'Jasper', jrxml_file))
            jasper_time = os.path.getmtime(jasper_file)

            if jrxml_time > jasper_time: #if the jasper if older than the jrxml
                # Stale compiled Jasper file: delete it so it gets
                # rebuilt from the newer .jrxml source on the next run.
                print2stdout(f'\nNewer JRXML file detected for {jrxml_file}')
                print2stdout(f'Deleting {jasper_file}')
                os.remove(jasper_file)


def filterByMember(values, members):
    """
    Filter a values dict down to a specific list of members.

    Parameters
    ----------
    values : dict
        Dictionary of values keyed by member.
    members : list
        List of member keys to keep.

    Returns
    -------
    dict
        Dictionary containing only the requested (and present) members.

    Raises
    ------
    None
        This function does not explicitly raise exceptions; a requested
        member missing from ``values`` is logged and skipped.

    Examples
    --------
    >>> filterByMember(values, ['member_1', 'member_2'])
    """

    filtered_values = {}

    for member in members:
        try:
            membervalues = values[member]
        except KeyError:
            # Requested member isn't present in the data; skip it rather
            # than failing the whole filter operation.
            print2stderr(f'Member {member} not found in values')
            continue
        filtered_values[member] = membervalues

    return filtered_values


def checkForCollections(data_settings):
    """
    Check whether any data setting is flagged as a forecast collection.

    Parameters
    ----------
    data_settings : dict
        Dictionary of settings for data, each optionally containing a
        ``'collection'`` flag.

    Returns
    -------
    bool
        ``True`` if any entry is flagged as a collection, ``False``
        otherwise.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> checkForCollections(data_settings)
    False
    """

    # return True as soon as any entry is found flagged as a collection
    for ds in data_settings.keys():
        if 'collection' in data_settings[ds].keys():
            if data_settings[ds]['collection']:
                return True
    return False


def organizePlotYears(object_settings):
    """
    Reorder a plot/table's configured years into a priority order.

    Orders 'ALLYEARS' first, then multi-year ranges (containing '-'),
    then remaining individual years, rather than preserving the
    original settings order.

    Parameters
    ----------
    object_settings : dict
        Settings to parse for year information; must contain
        ``'years'`` and optionally ``'yearstr'``.

    Returns
    -------
    years : list
        Reordered list of years.
    yrstrs : list
        Matching list of year strings (empty list if ``'yearstr'``
        wasn't present in ``object_settings``).

    Raises
    ------
    NameError
        May be raised if ``'years'`` is present in ``object_settings``
        but ``'yearstr'`` is not; see Notes below.

    Notes
    -----
    ``_isyrstr`` is only assigned when ``'yearstr'`` is present in
    ``object_settings``; if ``'years'`` is present but ``'yearstr'`` is
    not, ``_isyrstr`` is referenced later while undefined, which would
    raise a ``NameError``. This matches the source file as written and
    has not been changed here, per the "no logic changes" scope of this
    documentation pass.

    Examples
    --------
    >>> years, yrstrs = organizePlotYears(object_settings)
    """

    if 'years' in object_settings.keys():
        years = []
        # NOTE: `_isyrstr` is only assigned when 'yearstr' is present in
        # object_settings; if 'years' is present but 'yearstr' is not,
        # `_isyrstr` is referenced below while undefined, which would
        # raise a NameError. This matches the source file as written; 
        # not changed here per the "no logic changes" scope of this
        # documentation pass.
        if 'yearstr' in object_settings.keys():
            yrstrs = []
            _isyrstr = True
        # Build the years list in a specific priority order: 'ALLYEARS'
        # first, then multi-year ranges (containing '-'), then remaining
        # individual years - rather than preserving the original order
        # from object_settings['years'].
        for yi, year in enumerate(object_settings['years']):
            if year == 'ALLYEARS':
                # ALLYEARS entries go first
                years.append(year)
                if _isyrstr:
                    yrstrs.append(object_settings['yearstr'][yi])
        for yi, year in enumerate(object_settings['years']):
            if isinstance(year, str):
                if '-' in year:
                    # multi-year range strings go second
                    years.append(year)
                    if _isyrstr:
                        yrstrs.append(object_settings['yearstr'][yi])
        for yi, year in enumerate(object_settings['years']):
            if year != 'ALLYEARS':
                # remaining individual years go last
                years.append(year)
                if _isyrstr:
                    yrstrs.append(object_settings['yearstr'][yi])
        if _isyrstr:
            return years, yrstrs
        else:
            return years, []
    return [], []


def sanitizeText(intext):
    """
    Strip characters unsafe for generated file names.

    Parameters
    ----------
    intext : object
        Text (or any value convertible to ``str``) to clean.

    Returns
    -------
    str
        The cleaned text, with periods, spaces, colons, and
        underscores removed.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> sanitizeText('My Sim: v1.0')
    'MySimv10'
    """

    # Strip characters that are problematic in generated file names
    # (periods, spaces, colons, underscores).
    return str(intext).replace('.','').replace(' ', '').replace(':', '').replace("_","")


def calculateStorageFromElevation(values, curline):
    """
    Interpolate storage volume from elevation using a storage curve file.

    Parameters
    ----------
    values : array_like
        Elevation values to convert.
    curline : dict
        Current line settings; must contain
        ``'elevation_storage_area_file'`` (a 2-column CSV of elevation,
        storage).

    Returns
    -------
    numpy.ndarray
        Interpolated storage values corresponding to ``values``.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> calculateStorageFromElevation(values, curline)
    """
    
    # Elevation-storage curve file: two columns (elevation, storage);
    # build an interpolation function and evaluate it at every requested
    # elevation value.
    elevation_storage_area_file = curline['elevation_storage_area_file']
    elev_stor_area = np.loadtxt(elevation_storage_area_file, delimiter=',')
    elevstorcurve = interpolate.interp1d(elev_stor_area[:, 0], elev_stor_area[:, 1], bounds_error=False, fill_value=np.nan)
    return elevstorcurve(values)


def ReplaceListAtIdx(list, idx, replacevalue):
    """
    Replace a value in a list at a specified index.

    Parameters
    ----------
    list : list
        List of values (modified in place).
    idx : int
        Index to replace.
    replacevalue : object
        Value to place at ``idx``.

    Returns
    -------
    list
        The updated list.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> ReplaceListAtIdx([1, 2, 3], 1, 99)
    [1, 99, 3]
    """

    # replace the value in place at the given index
    list[idx] = replacevalue
    return list