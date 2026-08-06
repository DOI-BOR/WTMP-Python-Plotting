# Standard library imports used for calendar/month lookups and time deltas.
import calendar
import datetime as dt

import WAT_Functions as WF		# WAT_Functions provides shared helper functions (e.g. logging via print2stdout).
from scipy import interpolate	# scipy interpolation is used to build the saturated dissolved-oxygen lookup curve.


class WAT_Constants(object):
    """
    Container for constants shared across the WTMP plotting/reporting code.

    An instance of this class is created once and passed around (or
    instantiated locally) so that unit tables, color palettes, month
    lookups, DSS time-interval mappings, unit-conversion factors,
    model-specific variable flags, and a saturated dissolved-oxygen (DO)
    interpolation function are all available from a single object instead
    of being redefined in every module that needs them.

    Attributes
    ----------
    units : dict
        Maps variable names to their metric/english unit abbreviations.
    unit_alt_names : dict
        Maps standardized unit abbreviations to lists of alternate
        spellings/formats.
    units_fancy_flags_internal : dict
        Maps temperature unit abbreviations to unicode degree-symbol
        strings, for internal use (e.g. matplotlib labels).
    units_fancy_flags_external : dict
        Maps temperature unit abbreviations to HTML-entity degree-symbol
        strings, for external use (e.g. XML/HTML report output).
    english_units : dict
        Maps metric unit abbreviations directly to their english
        equivalents.
    metric_units : dict
        Maps english unit abbreviations directly to their metric
        equivalents.
    def_colors : list of str
        Colorblind-safe hex color palette used as the default line color
        cycle.
    month2num : dict
        Maps lowercase 3-letter month abbreviations to month numbers.
    num2month : dict
        Maps month numbers to lowercase 3-letter month abbreviations.
    mo_str_3 : list of str
        Capitalized 3-letter month codes, in calendar order.
    time_intervals : dict
        Maps DSS-style interval strings to a two-element list of
        ``[step_size, library_flag]``.
    conversion : dict
        Maps a unit abbreviation to the multiplicative factor used to
        convert a value in that unit to its counterpart unit.
    model_specific_vars : dict
        Maps input-file variable/column names to the model type
        ('ressim' or 'cequalw2') that produces them.
    sat_data_do : list of float
        Reference dissolved-oxygen saturation values (mg/L).
    sat_data_temp : list of float
        Reference temperature values (degrees C) corresponding to
        ``sat_data_do``.
    satDO_interp : scipy.interpolate.interp1d
        Interpolation function for looking up saturated DO at an
        arbitrary temperature.

    Notes
    -----
    None of the values set here are expected to change at runtime; they
    are effectively read-only lookup tables computed once in ``__init__``.
    """

    def __init__(self):
        '''
        Initialize a WAT_Constants instance by populating all of the
        constant lookup tables used throughout the WTMP plotting and
        reporting code.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This is a constructor and does not return a value.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> constants = WAT_Constants()
        >>> constants.units['flow']['english']
        'cfs'
        '''

        # Each of these helper methods populates a distinct group of class
        # attributes (units, months, time intervals, colors, unit
        # conversions, model-specific variable names, and the saturated DO
        # curve). They are split into separate methods purely for
        # readability/organization.
        self.defineUnits()
        self.defineMonths()
        self.defineTimeIntervals()
        self.defineDefaultColors()
        self.defineUnitConversions()
        self.defineModelSpecificVariables()
        self.saturatedDO()

    def defineUnits(self):
        """
        Create the lookup dictionaries mapping variables to units.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets the instance attributes ``self.units``,
            ``self.unit_alt_names``, ``self.units_fancy_flags_internal``,
            ``self.units_fancy_flags_external``, ``self.english_units``,
            and ``self.metric_units``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        Marked with a ``#TODO: expand this`` comment in the original
        source, suggesting more variables/units may need to be added over
        time.

        Examples
        --------
        >>> constants = WAT_Constants()
        >>> constants.units['temperature']['metric']
        'c'
        """

        # Maps each supported variable name (e.g. 'flow', 'elevation') to
        # its metric and english unit abbreviations. Several keys map to
        # the same physical quantity (e.g. 'temp' and 'temperature') so
        # that either naming convention used elsewhere in the code works.
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

        # For each standardized unit abbreviation, list the alternate
        # spellings/formats that should be recognized as equivalent (used
        # by normalize_unit() below to clean up units coming from input
        # files that may not follow a consistent convention).
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

        # Degree symbol renderings for temperature units: one for internal
        # use (unicode, e.g. matplotlib labels) and one for external use
        # (HTML entity, e.g. embedding in XML/HTML report output).
        self.units_fancy_flags_internal = {'f': u"\u00b0F",
                                            'c': u"\u00b0C"}

        # HTML-entity version of the degree symbol, used when writing
        # unit labels directly into the XML/HTML report output.
        self.units_fancy_flags_external = {'f': r"&#176;F",
                                          'c': r"&#176;C"}
        # Build quick lookup dicts that go directly from a metric unit to
        # its english equivalent, and vice versa, derived from self.units.
        self.english_units = {self.units[key]['metric']: self.units[key]['english'] for key in self.units.keys()}
        # Reverse of english_units, mapping english abbreviations back to metric.
        self.metric_units = {v: k for k, v in self.english_units.items()}

    def normalize_unit(self, unit):
        """
        Normalize a unit string to its standard abbreviation.

        Looks up the (case-insensitive, whitespace-trimmed) input unit
        against the alternate-name lists in ``self.unit_alt_names`` and
        returns the standard key that the alternate name belongs to. If
        no match is found, the original string is returned unchanged.

        Parameters
        ----------
        unit : str
            The unit string to normalize (e.g. ``"Fahrenheit"``,
            ``"cfs"``, ``" m3/s "``).

        Returns
        -------
        str
            The standardized unit abbreviation (e.g. ``"f"``, ``"cfs"``,
            ``"m3/s"``) if a match is found, otherwise the original
            ``unit`` string.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> constants = WAT_Constants()
        >>> constants.normalize_unit('Fahrenheit')
        'f'
        >>> constants.normalize_unit('unknown_unit')
        'unknown_unit'
        """
        # log the raw (un-normalized) unit string for debugging/traceability
        WF.print2stdout(f'Normalizing unit: {unit}')
        # Strip whitespace first so trailing/leading spaces from source
        # files don't cause a failed match below.
        testing_units = unit.strip()
        if testing_units == unit:
            # no leading/trailing whitespace was present on the original string
            WF.print2stdout(f'@@@@@ Units have no extra spaces')
        else:
            # original string had extra whitespace that has now been stripped
            WF.print2stdout(f'@@@@@ Units have extra spaces')

        # Search every standard unit's list of alternate spellings for a
        # case-insensitive match against the trimmed input string.
        for standard, alt_names in self.unit_alt_names.items():
            if unit.lower().strip() in alt_names:
                # match found, log it and return the standardized abbreviation
                WF.print2stdout(f'Match found for {unit}, normalized to {standard}')  # Add this
                return standard
        # No alternate-name match was found; fall back to the raw input.
        WF.print2stdout(f'No match found for {unit}, returning original')  # Add this
        return unit  # Return the original if no match is found

    def defineDefaultColors(self):
        """
        Set up the fallback line-color palette.

        Color choices are based on the colorblind-safe palette at
        https://davidmathlogic.com/colorblind/#%2388CCEE-%23882255-%23117733-%2344AA99-%23DDCC77-%23CC6677-%23AA4499-%23332288

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets the instance attribute ``self.def_colors``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> constants = WAT_Constants()
        >>> constants.def_colors[0]
        '#88CCEE'
        """

        # Previous (non-colorblind-safe) palette, kept commented out for
        # reference in case the color scheme needs to be reverted.
        # self.def_colors = ['#003E51', '#FF671F', '#007396', '#215732', '#C69214', '#4C12A1', '#DDCBA4', '#9A3324']
        # Colorblind-safe palette (see URL above) used as the fallback line
        # color cycle when a plot's graphics defaults don't specify colors.
        self.def_colors = ['#88CCEE', '#882255', '#117733', '#44AA99', '#DDCC77', '#CC6677', '#AA4499', '#332288']
        #                     blue       red       green    light green   yellow     salmon    redpink, purple

    def defineMonths(self):
        """
        Define month name/number lookups and 3-letter month codes.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets the instance attributes ``self.month2num``,
            ``self.num2month``, and ``self.mo_str_3``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> constants = WAT_Constants()
        >>> constants.month2num['jan']
        1
        >>> constants.mo_str_3[0]
        'Jan'
        """

        # Build lowercase-month-abbreviation -> month-number and
        # month-number -> lowercase-month-abbreviation dictionaries from
        # Python's calendar module (calendar.month_abbr[0] is empty, so
        # it's filtered out by the `if month` check).
        self.month2num = {month.lower(): index for index, month in enumerate(calendar.month_abbr) if month}
        # reverse mapping of month2num, keyed by month number instead of name
        self.num2month = {index: month.lower() for index, month in enumerate(calendar.month_abbr) if month}
        # Capitalized 3-letter month codes used directly as table/axis labels.
        self.mo_str_3 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def defineTimeIntervals(self):
        """
        Build the DSS-interval-string to timedelta/library-flag lookup.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets the instance attribute ``self.time_intervals``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> constants = WAT_Constants()
        >>> constants.time_intervals['1HOUR']
        [datetime.timedelta(seconds=3600), 'np']
        """

        # Maps each DSS-style interval string (e.g. '1HOUR', '1DAY') to a
        # two-element list: [step size, library flag]. Sub-monthly
        # intervals use a datetime.timedelta and are tagged 'np' (built
        # with numpy-based regular time arrays); monthly/yearly intervals
        # use a pandas offset alias string and are tagged 'pd' (built with
        # pandas date_range, since timedelta can't represent variable-length
        # months/years).
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
                               # monthly/yearly intervals use pandas offset aliases ('pd') instead
                               # of timedelta, since calendar months/years have variable lengths
                               '1MON': ['1M', 'pd'],
                               '2MON': ['2M', 'pd'],
                               '6MON': ['6M', 'pd'],
                               '1YEAR': ['1Y', 'pd']}

    def defineUnitConversions(self):
        """
        Build the unit-conversion factor dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets the instance attribute ``self.conversion``, keyed by
            source unit with each value the multiplicative factor to
            its counterpart unit.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> constants = WAT_Constants()
        >>> constants.conversion['m3/s']
        35.314666213
        """

        # Multiplicative factors used to convert FROM the metric unit
        # given by each key TO the corresponding english unit (e.g.
        # multiply a value in m3/s by 35.314666213 to get cfs). Also
        # includes the reverse direction (cfs, ft, af) for converting
        # english units back to metric.
        self.conversion = {'m3/s': 35.314666213,
                          'cfs': 0.0283168469997284,
                          'm': 3.28084,
                          'ft': 0.3048,
                          'm3': 0.000810714,
                          'af': 1233.48}

    def defineModelSpecificVariables(self):
        """
        Map model-specific input variable names to their model type.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets the instance attribute ``self.model_specific_vars``.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> constants = WAT_Constants()
        >>> constants.model_specific_vars['w2_segment']
        'cequalw2'
        """

        # Maps input-file variable/column names to the model type
        # ('ressim' or 'cequalw2') that produces them, so downstream code
        # can tell which model a given input variable came from.
        self.model_specific_vars = {'ressimresname': 'ressim',
                                   'xy': 'ressim',
                                   'w2_segment': 'cequalw2',
                                   'w2_file': 'cequalw2'}

    def saturatedDO(self):
        """
        Build a precomputed saturated-dissolved-oxygen interpolation curve.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets the instance attributes ``self.sat_data_do``,
            ``self.sat_data_temp``, and ``self.satDO_interp`` (the
            interpolation function itself).

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        The reference values represent standard dissolved-oxygen
        saturation (mg/L) at sea-level atmospheric pressure, tabulated at
        1-degree Celsius increments from 0 to 45 degrees C. The resulting
        interpolation function clamps out-of-range temperatures to the
        nearest table endpoint rather than raising an error.

        Examples
        --------
        >>> constants = WAT_Constants()
        >>> float(constants.satDO_interp(20.0))
        9.07
        """

        # Reference table of dissolved-oxygen saturation (mg/L) at 1-degree
        # Celsius increments from 0 to 45 degrees C (standard DO saturation
        # values at sea-level atmospheric pressure).
        self.sat_data_do = [14.60, 14.19, 13.81, 13.44, 13.09, 12.75, 12.43, 12.12, 11.83, 11.55, 11.27, 11.01, 10.76, 10.52, 10.29,
                       10.07, 9.85, 9.65, 9.45, 9.26, 9.07, 8.90, 8.72, 8.56, 8.40, 8.24, 8.09, 7.95, 7.81, 7.67, 7.54, 7.41,
                       7.28, 7.16, 7.05, 6.93, 6.82, 6.71, 6.61, 6.51, 6.41, 6.31, 6.22, 6.13, 6.04, 5.95]
        # Corresponding temperature values (degrees C) for each saturated
        # DO value above; index-aligned with self.sat_data_do.
        self.sat_data_temp = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
                         22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
                         42., 43., 44., 45.]

        # Build a 1-D interpolation function so saturated DO can be looked
        # up for any temperature within (or clamped to) the table range.
        # bounds_error=False + fill_value means temperatures outside
        # [0, 45] C are clamped to the first/last table values instead of
        # raising an error.
        self.satDO_interp = interpolate.interp1d(self.sat_data_temp, self.sat_data_do,
                                        fill_value=(self.sat_data_do[0], self.sat_data_do[-1]), bounds_error=False)