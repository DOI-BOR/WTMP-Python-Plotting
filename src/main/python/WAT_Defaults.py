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

import os
import matplotlib as mpl

import WAT_Constants as WC
import WAT_Functions as WF
import WAT_Reader as WR
import WAT_Plots as WP

constants = WC.WAT_Constants()

def getDefaultDefaultLineStyles(i):
    '''
    creates a default line style based off of the number line and default colors
    used if param is undefined or not in defaults file
    :param i: count of line on the plot
    :return: dictionary with line settings
    '''

    while i >= len(constants.def_colors):
        i = i - len(constants.def_colors)
    return {'linewidth': 2, 'linecolor': constants.def_colors[i],
            'linestylepattern': 'solid', 'alpha': 1.0, 'zorder': 4}

def getDefaultDefaultForecastTableHeaders():
    '''
    incase nothing is entered for forecast tables, arranges the columns
    :return: list of table headers
    '''

    return ['member', 'operationsname', 'metname', 'temptargetname']

def getDefaultDefaultPointStyles(i):
    '''
    creates a default point style based off of the number points and default colors
    used if param is undefined or not in defaults file
    :param i: count of points on the plot
    :return: dictionary with point settings
    '''

    while i >= len(constants.def_colors):
        i = i - len(constants.def_colors)
    return {'pointfillcolor': constants.def_colors[i], 'pointlinecolor': constants.def_colors[i], 'symboltype': 1,
            'symbolsize': 5, 'numptsskip': 0, 'alpha': 1.0}

def getDefaultDefaultTextStyles():
    '''
    creates a default line style based off of the number line and default colors
    used if param is undefined or not in defaults file
    :return: dictionary with line settings
    '''

    return {'fontsize': 9, 'fontcolor': 'black', 'alpha': 1.0, 'horizontalalignment': 'left'}

def getDefaultLineSettings(defaultLineStyles, LineSettings, param, i, debug=False):
    '''
    gets line settings and adds missing needed settings with defaults. Then translates java style inputs to
    python commands. Gets colors and styles.
    :param LineSettings: dictionary object containing settings and flags for lines/points
    :param param: parameter of data in order to grab default
    :param i: number of line on the plot in order to get the right sequential color
    :return:
        LineSettings: dictionary containing keys describing how the line/points are drawn
    '''

    LineSettings = getDrawFlags(LineSettings)
    if LineSettings['drawline'].lower() == 'true':
        if param != None:
            if param.lower() in defaultLineStyles.keys():
                while i >= len(defaultLineStyles[param.lower()]['lines']):
                    i = i - len(defaultLineStyles[param.lower()]['lines'])

                default_lines = defaultLineStyles[param.lower()]['lines'][i]

                for key in default_lines.keys():
                    if key not in LineSettings.keys():
                        LineSettings[key] = default_lines[key]

        default_default_lines = getDefaultDefaultLineStyles(i)
        for key in default_default_lines.keys():
            if key not in LineSettings.keys():
                LineSettings[key] = default_default_lines[key]

        LineSettings['linecolor'] = WF.confirmColor(LineSettings['linecolor'], default_default_lines['linecolor'], debug=debug)
        LineSettings = WP.translateLineStylePatterns(LineSettings)

    if LineSettings['drawpoints'] == 'true':
        if param in defaultLineStyles.keys():
            while i >= len(defaultLineStyles[param]['lines']):
                i = i - len(defaultLineStyles[param]['lines'])
            default_lines = defaultLineStyles[param]['lines'][i]
            for key in default_lines.keys():
                if key not in LineSettings.keys():
                    LineSettings[key] = default_lines[key]

        default_default_points = getDefaultDefaultPointStyles(i)

        for key in default_default_points.keys():
            if key not in LineSettings.keys():
                LineSettings[key] = default_default_points[key]

        LineSettings['pointfillcolor'] = WF.confirmColor(LineSettings['pointfillcolor'], default_default_points['pointfillcolor'], debug=debug)
        LineSettings['pointlinecolor'] = WF.confirmColor(LineSettings['pointlinecolor'], default_default_points['pointlinecolor'], debug=debug)

        try:
            if int(LineSettings['numptsskip']) == 0:
                LineSettings['numptsskip'] = 1
        except ValueError:
            WF.print2stdout('Invalid setting for numptsskip.', LineSettings['numptsskip'], debug=debug)
            WF.print2stdout('defaulting to 25', debug=debug)
            LineSettings['numptsskip'] = 25

        LineSettings = WP.translatePointStylePatterns(LineSettings)

    return LineSettings

def getDefaultGateLineSettings(GateLineSettings, i, debug=False):
    '''
    gets line settings and adds missing needed settings with defaults. Then translates java style inputs to
    python commands. Gets colors and styles.
    :param GateLineSettings: dictionary object containing settings and flags for gates
    :param i: number of line on the plot in order to get the right sequential color
    :return:
        GateLineSettings: dictionary containing keys describing how the line/points are drawn
    '''

    GateLineSettings = getDrawFlags(GateLineSettings)
    if GateLineSettings['drawline'] == 'true':
        default_default_lines = getDefaultDefaultLineStyles(i)
        for key in default_default_lines.keys():
            if key not in GateLineSettings.keys():
                GateLineSettings[key] = default_default_lines[key]

        GateLineSettings = WP.translateLineStylePatterns(GateLineSettings)
        GateLineSettings['linecolor'] = WF.confirmColor(GateLineSettings['linecolor'], default_default_lines['linecolor'], debug=debug)

    if GateLineSettings['drawpoints'] == 'true':
        default_default_points = getDefaultDefaultPointStyles(i)
        for key in default_default_points.keys():
            if key not in GateLineSettings.keys():
                GateLineSettings[key] = default_default_points[key]
        try:
            if int(GateLineSettings['numptsskip']) == 0:
                GateLineSettings['numptsskip'] = 1
        except ValueError:
            WF.print2stdout('Invalid setting for numptsskip.', GateLineSettings['numptsskip'], debug=debug)
            WF.print2stdout('defaulting to 25', debug=debug)
            GateLineSettings['numptsskip'] = 25

        GateLineSettings['pointlinecolor'] = WF.confirmColor(GateLineSettings['pointlinecolor'], default_default_lines['pointlinecolor'], debug=debug)
        GateLineSettings['pointfillcolor'] = WF.confirmColor(GateLineSettings['pointfillcolor'], default_default_lines['pointfillcolor'], debug=debug)
        GateLineSettings = WP.translatePointStylePatterns(GateLineSettings)

    return GateLineSettings

def getDefaultContourLineSettings(contour_settings):
    '''
    contains bare essentials for contour plots just incase nothing is specified
    :param contour_settings: dictionary containing settings
    :return:
    '''

    default_contour_settings = {'linecolor': 'grey',
                                'linewidth': 1,
                                'linestylepattern': 'solid',
                                'alpha': 1,
                                'contourlinetext': 'false',
                                'fontsize': 10,
                                'text_inline': 'true',
                                'inline_spacing': 10,
                                'legend': 'false'}

    for key in default_contour_settings.keys():
        if key not in contour_settings:
            contour_settings[key] = default_contour_settings[key]
            if key == 'text_inline':
                if contour_settings[key].lower() == 'true':
                    contour_settings[key] = True
                else:
                    contour_settings[key] = False

    contour_settings = WP.translateLineStylePatterns(contour_settings)

    return contour_settings

def getDefaultContourSettings(object_settings, debug=False):
    '''
    gets default settings for contours adn overwrites missing settings in object settings to be sure
    contour plots have everything that they need
    :param object_settings: dictionary containing settings
    :return: updated settings dictionary
    '''

    # defaultColormap = mpl.cm.get_cmap('jet')
    defaultColormap = mpl.colormaps['jet']
    default_colorbar_settings = {'colormap': defaultColormap,
                                 'bins': 10,
                                 'numticks': 5}

    if 'colorbar' in object_settings.keys():
        if 'colormap' in object_settings['colorbar'].keys():
            try:
                # usercolormap = mpl.cm.get_cmap(object_settings['colorbar']['colormap'])
                usercolormap = mpl.colormaps[object_settings['colorbar']['colormap']]
                object_settings['colormap'] = usercolormap
            except ValueError:
                WF.print2stdout('User selected invalid colormap:', object_settings['colorbar']['colormap'], debug=debug)
                WF.print2stdout('Tip: make sure capitalization is correct!', debug=debug)
                WF.print2stdout('Defaulting to Jet.', debug=debug)
                object_settings['colormap'] = defaultColormap
    else:
        object_settings['colorbar'] = {}

    for key in default_colorbar_settings.keys():
        if key not in object_settings['colorbar']:
            object_settings['colorbar'][key] = default_colorbar_settings[key]

    return object_settings

def getDefaultStraightLineSettings(LineSettings, debug):
    '''
    gets line settings and adds missing needed settings with defaults. Then translates java style inputs to
    python commands. Gets colors and styles.
    :param LineSettings: dictionary object containing settings and flags for lines/points
    :param param: parameter of data in order to grab default
    :return:
        LineSettings: dictionary containing keys describing how the line/points are drawn
    '''
    LineSettings = getDrawFlags(LineSettings)
    default_default_lines = getDefaultDefaultLineStyles(0)
    default_default_lines['linecolor'] = 'black' #don't need different colors by default..
    for key in default_default_lines.keys():
        if key not in LineSettings.keys():
            LineSettings[key] = default_default_lines[key]

    LineSettings = WP.translateLineStylePatterns(LineSettings)
    LineSettings['linecolor'] = WF.confirmColor(LineSettings['linecolor'], default_default_lines['linecolor'], debug=debug)

    return LineSettings

def getDefaultTextSettings(TextSettings, debug):
    '''
    gets text settings and adds missing needed settings with defaults. Then translates java style inputs to
    python commands. Gets colors and styles.
    :param TextSettings: dictionary object containing settings and flags for text
    :return:
        LineSettings: dictionary containing keys describing how the line/points are drawn
    '''

    default_default_text = getDefaultDefaultTextStyles()
    for key in default_default_text.keys():
        if key not in TextSettings.keys():
            TextSettings[key] = default_default_text[key]

    TextSettings['fontcolor'] = WF.confirmColor(TextSettings['fontcolor'], default_default_text['fontcolor'], debug=debug)

    return TextSettings

#################################################################
# Helper Functions #
#################################################################

def getDrawFlags(LineSettings):
    '''
    reads line settings dictionary to look for defined settings of lines or points to determine if either or both
    should be drawn. If nothing is explicitly stated, then draw lines with default settings.
    :param LineSettings: dictionary object containing settings and flags for lines/points
    :return:
        LineSettings: dictionary containing keys describing how the line/points are drawn
    '''

    #unless explicitly stated, look for key identifiers to draw lines or not
    LineVars = ['linecolor', 'linestylepattern', 'linewidth']
    PointVars = ['pointfillcolor', 'pointlinecolor', 'symboltype', 'symbolsize', 'numptsskip', 'markersize']

    if 'drawline' not in LineSettings.keys():
        for var in LineVars:
            if var in LineSettings.keys():
                LineSettings['drawline'] = 'true'
                break
        if 'drawline' not in LineSettings.keys():
            LineSettings['drawline'] = 'false'

    if 'drawpoints' not in LineSettings.keys():
        for var in PointVars:
            if var in LineSettings.keys():
                LineSettings['drawpoints'] = 'true'
                break
        if 'drawpoints' not in LineSettings.keys():
            LineSettings['drawpoints'] = 'false'

    if LineSettings['drawpoints'] == 'false' and LineSettings['drawline'] == 'false':
        LineSettings['drawline'] = 'true' #gotta do something..

    return LineSettings

def readDefaultLineStylesFile(Report):
    '''
    sets up path for default line styles file and reads the xml
    :return: class variable
                self.defaultLineStyles
    '''

    defaultLinesFile = os.path.join(Report.studyDir, 'reports', 'defaultLineStyles.xml')
    WF.checkExists(defaultLinesFile)
    # defaultLinesFile = os.path.join(self.default_dir, 'defaultLineStyles.xml') #TODO: implement with build
    defaultLineStyles = WR.readDefaultLineStyle(defaultLinesFile)
    return defaultLineStyles