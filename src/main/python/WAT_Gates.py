import numpy as np

import WAT_Reader as WR


def getGateBlendDays(gateconfig, gatedata, timestamp):
    """
    Compute the number of days the current gate blend has been in effect.

    "Gate blend days" is how long (in fractional days) the exact
    combination of currently open/closed individual gates has been
    continuously in effect, as of ``timestamp``.

    Parameters
    ----------
    gateconfig : dict
        Settings dictionary describing the current gate configuration,
        keyed by gate level then gate number.
    gatedata : dict
        Dictionary of gate operation time series data.
    timestamp : datetime.datetime
        The timestamp to evaluate the blend duration as of.

    Returns
    -------
    float or str
        The number of days (rounded to 3 decimals) the current gate
        blend has been in effect, or ``'N/A'`` if ``gateconfig`` is
        empty.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getGateBlendDays(gateconfig, gatedata, timestamp)
    12.5
    """

    if len(gateconfig) == 0:
        # no gate configuration to evaluate at all
        return 'N/A'

    # Use the dates from the first available gate as the shared time axis
    # (all gates in gatedata are assumed to share the same timestamps).
    gd_key = list(gatedata.keys())[0]
    curgate = gatedata[gd_key]['gates'][list(gatedata[gd_key]['gates'].keys())[0]]
    # find the array index closest to the requested timestamp
    idx = WR.getClosestTime([timestamp], curgate['dates'])[0]
    # Start with everything "matching" up through the target index, then
    # AND in each individual gate's open/closed mask below.
    datamask = np.ones(idx+1, dtype=bool)

    # narrow the mask down gate by gate, requiring every gate's state to match the current configuration
    for gatelevel in gatedata.keys():
        for gatenumber in gatedata[gatelevel]['gates'].keys():
            current_op = gateconfig[gatelevel][gatenumber]
            # current_op being NaN means this specific gate is currently
            # closed; build a mask of every timestep where that gate's
            # actual open/closed state matches the current state (NaN
            # value in the data == closed, non-NaN == open).
            if np.isnan(current_op):
                # gate is currently closed, match every timestep where it was also closed
                msk = np.isnan(gatedata[gatelevel]['gates'][gatenumber]['values'][:idx+1])
            else:
                # gate is currently open, match every timestep where it was also open
                msk = ~np.isnan(gatedata[gatelevel]['gates'][gatenumber]['values'][:idx+1])
            # combine this gate's mask into the running overall match mask
            datamask = datamask & msk

    # Walk backward from the target timestamp to find the most recent
    # point where the exact combination of open/closed gates changed.
    changeop = False
    for i in reversed(range(idx)):
        if not datamask[i]:
            # found the most recent mismatch, i.e. the last time the blend changed
            changeop = True
            break
    # Convert the index-based duration into a fractional day count using
    # the series' regular timestep.
    timestep = (curgate['dates'][1] - curgate['dates'][0]).total_seconds() / 86400
    if changeop:
        # blend duration is the number of steps since the last change
        decdays = (idx - i -1) * timestep
    else:
        # No change was ever found going all the way back; the current
        # gate combination has been in effect since the start of the
        # available data.
        decdays = idx * timestep

    return round(decdays, 3)


def getGateConfigurationDays(gateconfig, gatedata, timestamp):
    """
    Compute the number of days the current gate-level configuration has
    been in effect.

    Unlike ``getGateBlendDays`` (which tracks each individual gate's
    exact state), this tracks whether each GATE LEVEL as a whole has
    been open or closed (i.e. whether any gate within the level is
    open), as of ``timestamp``.

    Parameters
    ----------
    gateconfig : dict
        Settings dictionary describing the current gate configuration,
        keyed by gate level then gate number.
    gatedata : dict
        Dictionary of gate operation time series data.
    timestamp : datetime.datetime
        The timestamp to evaluate the configuration duration as of.

    Returns
    -------
    float or str
        The number of days (rounded to 3 decimals) the current gate
        level configuration has been in effect, or ``'N/A'`` if
        ``gateconfig`` is empty.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> getGateConfigurationDays(gateconfig, gatedata, timestamp)
    8.25
    """

    if len(gateconfig) == 0:
        # no gate configuration to evaluate at all
        return 'N/A'

    # Use the dates from the first available gate as the shared time axis
    gd_key = list(gatedata.keys())[0]
    curgate = gatedata[gd_key]['gates'][list(gatedata[gd_key]['gates'].keys())[0]]
    # find the array index closest to the requested timestamp
    idx = WR.getClosestTime([timestamp], curgate['dates'])[0]
    # start with everything "matching" through the target index, narrowed down level by level below
    datamask = np.ones(idx+1, dtype=bool)

    # Unlike getGateBlendDays (which tracks each individual gate's exact
    # open/closed state), this tracks whether each GATE LEVEL (a group of
    # gates, e.g. all outlets at one elevation) is open at all - i.e. any
    # gate within the level is open - rather than which specific gates
    # within it are open.
    for gatelevel in gatedata.keys():
        # determine whether this level is currently open (any gate within it open) or closed
        current_op_level = np.nan
        for gatenumber in gatedata[gatelevel]['gates'].keys():
            current_op = gateconfig[gatelevel][gatenumber]
            if not np.isnan(current_op):
                # As soon as any gate in this level is currently open,
                # the whole level counts as "open".
                current_op_level = 1
                break

        # Build a mask of every timestep where ANY gate in this level was
        # open (true == at least one gate open).
        datamask_gateLevel = np.zeros(idx+1, dtype=bool)
        for gatenumber in gatedata[gatelevel]['gates'].keys():
            # non-NaN values mean this gate was open at that timestep
            msk = ~np.isnan(gatedata[gatelevel]['gates'][gatenumber]['values'][:idx+1]) #true when open
            # OR every gate's mask together, so this marks "any gate in the level was open"
            datamask_gateLevel = datamask_gateLevel | msk

        if np.isnan(current_op_level): #if closed..
            # Currently this whole level is closed: match timesteps where
            # the level was ALSO closed (invert the "any gate open" mask).
            datamask = datamask & ~datamask_gateLevel
        else:
            # Currently this level is open: match timesteps where it was
            # also open.
            datamask = datamask & datamask_gateLevel #datamsk_gateLevel if true when open

    # Same backward-scan-for-the-last-change logic as getGateBlendDays.
    changeop = False
    for i in reversed(range(idx)):
        if not datamask[i]:
            # found the most recent mismatch, i.e. the last time the configuration changed
            changeop = True
            break
    # convert the index-based duration into a fractional day count using the series' regular timestep
    timestep = (curgate['dates'][1] - curgate['dates'][0]).total_seconds() / 86400
    if changeop:
        # configuration duration is the number of steps since the last change
        decdays = (idx - i -1) * timestep
    else:
        # no change found, configuration has held since the start of the available data
        decdays = idx * timestep

    return round(decdays, 3)