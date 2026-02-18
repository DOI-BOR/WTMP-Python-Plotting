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

import pickle
import datetime as dt
import numpy as np
import matplotlib as mpl
from collections import Counter

import WAT_Functions as WF
import WAT_Time as WT
import WAT_Defaults as WD

class Plots(object):

    def __init__(self, Report):
        '''
        class that controls plotting functions
        :param Report: self from main Report Generator script
        '''

        self.Report = Report

    def confirmAxis(self, object_settings):
        '''
        Checks for an axis item in object settings. If not, make one.
        :param object_settings: dictionary containing settings for current object
        :return: object settings but with empty axis.
        '''

        if 'axs' not in object_settings.keys():
            object_settings['axs'] = [{}] #empty axis object
        return object_settings

    def seperateCollectionLines(self, line_draw_settings):
        '''
        takes a collection of lines and seperates them into their own instances to be plotted seperatly
        :param line_draw_settings: settings for lines
        :return: updated settings
        '''

        if 'members' in line_draw_settings:
            collection_draw_settings = {}
            members = line_draw_settings['members']
            for mi, member in enumerate(members):
                collection_draw_settings[member] = {}
                collection_draw_settings[member].update(line_draw_settings)
                collection_draw_settings[member]['numtimesused'] = mi
                if '%%member%%' not in collection_draw_settings[member]['label']:
                    collection_draw_settings[member]['label'] = f"{collection_draw_settings[member]['label']}: {member}"
                else:
                    # get the ensemble set for the current member
                    curr_ensemble_set = WF.matchMemberToEnsembleSet(self.Report.ensembleSets, member)
                    collection_draw_settings[member]['label'] = collection_draw_settings[member]['label'].replace('%%member%%', WF.getOriginalMemberNumber(member, curr_ensemble_set, self.Report.DSSFile,
                                                                                                                                                     self.Report.alternativeFpart, self.Report.StartTime, self.Report.EndTime, self.Report.debug))
                collection_draw_settings[member] = WF.fixDuplicateColors(collection_draw_settings[member])
            return collection_draw_settings

        else:
            WF.print2stdout('Unable to get members. Cannot seperate collection lines.', debug=self.Report.debug)
            return line_draw_settings

    def setTimeSeriesXlims(self, cur_obj_settings, yearstr, years):
        '''
        gets the xlimits for time series. This can be dependent on year, so needs to be looped over.
        :param cur_obj_settings: current plotting object settings dictionary
        :param yearstr: current year string
        :param years: list of years
        :return: updated cur_obj_settings dict
        '''

        if 'ALLYEARS' not in years:
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)
        else:
            if 'xlims' in cur_obj_settings.keys():
                if 'min' in cur_obj_settings['xlims']:
                    cur_obj_settings['xlims']['min'] = WF.updateFlaggedValues(cur_obj_settings['xlims']['min'],
                                                                              '%%year%%', str(self.Report.years[0]))
                if 'max' in cur_obj_settings['xlims']:
                    cur_obj_settings['xlims']['max'] = WF.updateFlaggedValues(cur_obj_settings['xlims']['max'],
                                                                              '%%year%%', str(self.Report.years[-1]))
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)

        return cur_obj_settings

    def getRelativeMasterSet(self, linedata, line_settings):
        '''
        organizes data and gets it on the same interval
        :param linedata: dictionary containing data
        :return: set of data on same interavl and units, settings for relative lines
        '''

        #add all the data together. then we can use this when plotting it to get %
        #TODO: deal with irregular intervals
        intervals = {}
        biggest_interval = None
        type = 'INST-VAL'
        for line in linedata.keys():
            if 'interval' in line_settings[line].keys():
                td = WF.getTimeInterval(linedata[line]['dates'])
                if line_settings[line]['interval'].upper() not in intervals.keys():
                    intervals[line_settings[line]['interval'].upper()] = td
                if biggest_interval == None:
                    biggest_interval = line_settings[line]['interval'].upper()
                    if 'type' in line_settings[line].keys():
                        type = line_settings[line]['type'].upper()
                else:
                    if td > intervals[biggest_interval]:
                        biggest_interval = line_settings[line]['interval'].upper()
                        if line_settings[line]['type'] in line.keys():
                            type = line_settings[line]['type'].upper()

        RelativeLineSettings = {'interval': biggest_interval,
                                'type': type}
        RelativeMasterSet = []
        units = []
        for li, line in enumerate(linedata.keys()):
            curline = pickle.loads(pickle.dumps(linedata[line], -1))
            curline_settings = pickle.loads(pickle.dumps(line_settings[line], -1))
            curline['values'], curline_settings['units'] = WF.convertUnitSystem(curline['values'], curline_settings['units'], 'metric', debug=self.Report.debug) #just make everything metric..
            units.append(curline_settings['units'])
            if li == 0:
                if biggest_interval != None:
                    _, RelativeMasterSet = WT.changeTimeSeriesInterval(curline['dates'], curline['values'],
                                                                                RelativeLineSettings,
                                                                                self.Report.startYear)
                else:
                    RelativeMasterSet = curline['values']
            else:
                if biggest_interval != None:
                    curline['interval'] = biggest_interval
                    curline['type'] = type
                    _, newvals = WT.changeTimeSeriesInterval(curline['dates'], curline['values'],
                                                              RelativeLineSettings,
                                                              self.Report.startYear)
                    RelativeMasterSet += newvals
                else:
                    RelativeMasterSet += curline['values']

        RelativeLineSettings['units'] = WF.getMostCommon(units)

        return RelativeMasterSet, RelativeLineSettings

    def plot(self, dates, values, curax, line_draw_settings):
        '''
        plots lines, taking into account markers and lines
        :param dates: list of dates
        :param values: list of values
        :param curax: axis to plot on
        :param line_draw_settings: settings that configure how line is drawn
        '''

        if line_draw_settings['drawline'].lower() == 'true' and line_draw_settings['drawpoints'].lower() == 'true':
            self.plotLinesAndPoints(dates, line_draw_settings, curax, line_draw_settings)
        elif line_draw_settings['drawline'].lower() == 'true':
            self.plotLines(dates, values, curax, line_draw_settings)
        elif line_draw_settings['drawpoints'].lower() == 'true':
            self.plotPoints(dates, values, curax, line_draw_settings)

    def plotLinesAndPoints(self, x, y, curaxis, settings):
        '''
        plots lines and points on line on given axis
        :param x: data list for x axis (dates or otherwise)
        :param y: data list for y axis
        :param curaxis: current axis object
        :param settings: dictionary containing settings for plot object
        '''

        curaxis.plot(x, y, label=settings['label'], c=settings['linecolor'],
                     lw=settings['linewidth'], ls=settings['linestylepattern'],
                     marker=settings['symboltype'], markerfacecolor=settings['pointfillcolor'],
                     markeredgecolor=settings['pointlinecolor'], markersize=float(settings['symbolsize']),
                     markevery=int(settings['numptsskip']), zorder=float(settings['zorder']),
                     alpha=float(settings['alpha']))

    def plotLines(self, x, y, curaxis, settings):
        '''
        plots lines on line on given axis
        :param x: data list for x axis (dates or otherwise)
        :param y: data list for y axis
        :param curaxis: current axis object
        :param settings: dictionary containing settings for plot object
        '''

        curaxis.plot(x, y, label=settings['label'], c=settings['linecolor'],
                     lw=settings['linewidth'], ls=settings['linestylepattern'],
                     zorder=float(settings['zorder']),
                     alpha=float(settings['alpha']))

    def plotPoints(self, x, y, curaxis, settings):
        '''
        plots points on line on given axis
        :param x: data list for x axis (dates or otherwise)
        :param y: data list for y axis
        :param curaxis: current axis object
        :param settings: dictionary containing settings for plot object
        '''

        curaxis.scatter(x[::int(settings['numptsskip'])], y[::int(settings['numptsskip'])],
                        marker=settings['symboltype'], facecolor=settings['pointfillcolor'],
                        edgecolor=settings['pointlinecolor'], s=float(settings['symbolsize']),
                        label=settings['label'], zorder=float(settings['zorder']),
                        alpha=float(settings['alpha']))

    def plotCollectionEnvelopes(self, dates, values, curax, settings):
        '''
        iterates and plots envelopes
        :param dates: list of dates
        :param values: list of values containing envelopes
        :param curax: current axis to plot on
        :param settings: settings for object
        '''

        if 'envelopes' in settings.keys() and len(values.keys()) > 0:
            collection_evelopes = self.Report.Data.computeCollectionEnvelopes(values, settings['envelopes'])
            for envelope_settings in settings['envelopes']:
                envelope = envelope_settings['percent']
                envelope_settings = WF.replaceDefaults(self, settings, envelope_settings)
                if envelope in collection_evelopes.keys():
                    envelope_vals = collection_evelopes[envelope]
                    self.plotLines(dates, envelope_vals, curax, envelope_settings)

    def formatDateXAxis(self, curax, object_settings, twin=False):
        '''
        formats the xaxis to be jdate or datetime and sets up xlimits. also sets up secondary xaxis
        :param curax: current plot axis
        :param object_settings: dictionary of settings
        :param twin: if true, will configure top axis
        :return: sets xlimits for axis
        '''

        useplot = True
        if twin:
            if 'xlims2' in object_settings.keys():
                xlims_flag = 'xlims2'
            else:
                WF.print2stdout('Using Same Xlims for top and bottom.', debug=self.Report.debug)
                xlims_flag = 'xlims'
        else:
            xlims_flag = 'xlims'

        if xlims_flag in object_settings.keys():
            xlims = object_settings[xlims_flag]#should be min max flags in here

            if 'min' in xlims.keys():
                xmin = xlims['min']
                if '-' in self.Report.years_str: #multiyear plots use 2008-2019 format
                    if isinstance(xmin, str): #if this gets replaced it will only be a str
                        if self.Report.years_str in xmin: #check for the offender
                            xmin = xmin.replace(self.Report.years_str, str(self.Report.startYear))

            else:
                xmin = self.Report.StartTime

            if 'max' in xlims.keys():
                xmax = xlims['max']
                if '-' in self.Report.years_str: #multiyear plots use 2008-2019 format
                    if isinstance(xmax, str): #if this gets replaced it will only be a str
                        if self.Report.years_str in xmax: #check for the offender
                            xmax = xmax.replace(self.Report.years_str, str(self.Report.endYear))
            else:
                xmax = self.Report.EndTime

            current_xlims = curax.get_xlim()
            current_xlims = [n.replace(tzinfo=None) for n in mpl.dates.num2date(current_xlims)]

            if current_xlims[0] < self.Report.StartTime:
                starttime = self.Report.StartTime
            else:
                starttime = current_xlims[0]

            if current_xlims[1] > self.Report.EndTime:
                endtime = self.Report.EndTime
            else:
                endtime = current_xlims[1]
            xmin = WT.translateDateFormat(xmin, 'datetime', starttime, starttime, endtime, debug=self.Report.debug)

            try:
                xmax = float(xmax)
                tmp_starttime = current_xlims[1] - dt.timedelta(seconds=1)
                if starttime < tmp_starttime: #check for end of year
                    starttime = dt.datetime(tmp_starttime.year, 1, 1, 0, 0)
            except:
                pass
            xmax = WT.translateDateFormat(xmax, 'datetime', endtime, starttime, endtime, debug=self.Report.debug)

            if xmax > current_xlims[1]:
                xmax = current_xlims[1]
            if xmin < current_xlims[0]:
                xmin = current_xlims[0]
            if xmin > current_xlims[1]:
                useplot = False
            if xmax < current_xlims[0]:
                useplot = False
            curax.set_xlim(left=xmin, right=xmax)

        else:
            WF.print2stdout('No Xlims flag set for {0}'.format(xlims_flag), debug=self.Report.debug)
            WF.print2stdout('Not setting Xlims.', debug=self.Report.debug)

        return useplot

    def formatTickLabels(self, ticks, ticksettings):
        '''
        changes ticks based on input settings
        :param ticks: existing ticks
        :param ticksettings: dictionary contianing settings for ticks
        :return: formatted ticks
        '''

        newticklabels = []
        allticksinteger = False
        if np.all([float(tick).is_integer() if isinstance(tick, (int, float)) else False for tick in ticks]):
            allticksinteger=True

        for tick in ticks:
            if isinstance(tick, (int, float)):

                if 'numdecimals' in ticksettings.keys():
                    numdecimals = int(ticksettings['numdecimals'])
                elif allticksinteger:
                    numdecimals = 0
                else:
                    numdecimals = 2

                newticklabels.append('{num:,.{digits}f}'.format(num=tick, digits=numdecimals))

            elif isinstance(tick, dt.datetime):
                if 'datetimeformat' in ticksettings.keys():
                    datetimeformat = ticksettings['datetimeformat']
                else:
                    datetimeformat = '%m/%d/%Y'
                tick_str = tick.strftime(datetimeformat)
                newticklabels.append(tick_str)

            else:
                newticklabels.append(str(tick))

        return newticklabels

    def formatTimeSeriesXticks(self, curax, xtick_settings, axis_settings, dateformatflag='dateformat'):
        '''
        applies tick settings to Xaxis for time series
        :param curax: current axis object
        :param xtick_settings: dictionary containing settings for ticks
        :param axis_settings: dictionary containing settings for ticks
        :param dateformatflag: flag to get dateformat from settings
        '''

        xmin, xmax = curax.get_xlim()

        xmin = mpl.dates.num2date(xmin).replace(tzinfo=None)
        xmax = mpl.dates.num2date(xmax).replace(tzinfo=None)

        if 'fontsize' in xtick_settings.keys():
            xticksize = float(xtick_settings['fontsize'])
        elif 'fontsize' in axis_settings.keys():
            xticksize = float(axis_settings['fontsize'])
        else:
            xticksize = 10

        if 'rotation' in xtick_settings.keys():
            rotation = float(xtick_settings['rotation'])
        else:
            rotation = 0

        curax.tick_params(axis='x', labelsize=xticksize, rotation=rotation)

        if 'onmonths' in xtick_settings.keys():
            if isinstance(xtick_settings['onmonths'], dict):
                xtick_settings['onmonths'] = [xtick_settings['onmonths']['month']]
            bymonthday = [1]

            if 'ondays' in xtick_settings.keys():
                if isinstance(xtick_settings['ondays'], dict):
                    xtick_settings['ondays'] = [xtick_settings['ondays']['day']]
                bymonthday = [int(n) for n in xtick_settings['ondays']]

            try:
                locator = mpl.dates.MonthLocator([int(n) for n in xtick_settings['onmonths']], bymonthday=bymonthday)
            except ValueError:
                WF.print2stdout('Invalid month values. Please use integer representation of Months (aka 1, 3, 5, etc...)', debug=self.Report.debug)
                formatted_months = [self.Report.Constants.month2num[n.lower()] for n in xtick_settings['onmonths']]
                locator = mpl.dates.MonthLocator(formatted_months, bymonthday=bymonthday)

            curax.xaxis.set_major_locator(locator)

        elif 'ondays' in xtick_settings.keys():
            if isinstance(xtick_settings['ondays'], dict):
                xtick_settings['ondays'] = [xtick_settings['ondays']['day']]
            locator = mpl.dates.DayLocator([int(n) for n in xtick_settings['ondays']])
            curax.xaxis.set_major_locator(locator)

        elif 'spacing' in xtick_settings.keys():
            xtickspacing = xtick_settings['spacing']
            try:
                xtickspacing = float(xtickspacing)
                xtickspacing = f'{xtickspacing}D'
            except ValueError:
                xtickspacing = xtick_settings['spacing']

            newxticks = WT.buildTimeSeries(xmin.replace(tzinfo=None), xmax.replace(tzinfo=None), xtickspacing)

            newxticklabels = self.formatTickLabels(newxticks, xtick_settings)
            curax.set_xticks(newxticks)
            curax.set_xticklabels(newxticklabels)

        if 'datetimeformat' in xtick_settings.keys():
            if axis_settings[dateformatflag].lower() == 'datetime':
                datetimeformat = xtick_settings['datetimeformat']
                fmt = mpl.dates.DateFormatter(datetimeformat)
                curax.xaxis.set_major_formatter(fmt)

        current_xticks = mpl.dates.num2date(curax.get_xticks())
        if dateformatflag in axis_settings.keys():
            if axis_settings[dateformatflag].lower() == 'jdate':
                if isinstance(current_xticks[0], dt.datetime):
                    jdateticklabels = WT.DatetimeToJDate(current_xticks)
                    newxticklabels = self.formatTickLabels(jdateticklabels, xtick_settings)
                    curax.set_xticks(current_xticks)
                    curax.set_xticklabels(newxticklabels)

    def formatYTicks(self, ax, ax_settings, gatedata={}, gate_placement=10, axis='left'):
        '''
        formats yticks on an axis
        :param ax: current axis
        :param ax_settings: settings for axis
        :param gatedata: data for gates
        :param gate_placement: how far away by default to plot gates
        :param axis: left or right
        '''

        if axis == 'left':
            ylimflag = 'ylims'
            yticksflag = 'yticks'
        else:
            ylimflag = 'ylims2'
            yticksflag = 'yticks2'

        ymin, ymax = ax.get_ylim()

        if ylimflag in ax_settings.keys():
            if 'min' in ax_settings[ylimflag]:
                ymin = float(ax_settings[ylimflag]['min'])

            if 'max' in ax_settings[ylimflag]:
                ymax = float(ax_settings[ylimflag]['max'])

        if len(gatedata.keys()) != 0:
            ymax = gate_placement
            ymin = 0

        ax.set_ylim(bottom=ymin)
        ax.set_ylim(top=ymax)


        if yticksflag in ax_settings.keys():
            ytick_settings = ax_settings[yticksflag]
            if 'fontsize' in ytick_settings.keys():
                yticksize = float(ytick_settings['fontsize'])
            elif 'fontsize' in ax_settings.keys():
                yticksize = float(ax_settings['fontsize'])
            else:
                yticksize = 10
            ax.tick_params(axis='y', labelsize=yticksize)

            if 'spacing' in ytick_settings.keys():
                ytickspacing = ytick_settings['spacing']

                if float(ytickspacing).is_integer():
                    ytickspacing = int(ytickspacing)
                else:
                    ytickspacing = float(ytickspacing)

                if 'ylims' not in ax_settings.keys():
                    ymax = int(np.ceil(ymax))
                    ymin = int(np.floor(ymin))
                else:
                    if 'min' not in ax_settings[ylimflag].keys():
                        ymin = int(np.floor(ymin))
                    if 'max' not in ax_settings[ylimflag].keys():
                        ymax = int(np.ceil(ymax))

                newyticks = np.arange(ymin, (ymax+ytickspacing), ytickspacing)
                newyticklabels = self.formatTickLabels(newyticks, ytick_settings)
                ax.set_yticks(newyticks)
                ax.set_yticklabels(newyticklabels)

                ax.set_ylim(bottom=min(newyticks))
                ax.set_ylim(top=max(newyticks))
                return

        newyticklabels = self.formatTickLabels(ax.get_yticks(), {})
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(newyticklabels)
        ax.set_ylim(bottom=ymin)
        ax.set_ylim(top=ymax)

    def formatXTicks(self, ax, ax_settings, axis='bottom'):
        '''
        formats xticks for an axis
        :param ax: current axis
        :param ax_settings: settings for axis
        :param axis: top or bottom
        '''

        if axis == 'bottom':
            xlimflag = 'xlims'
            xticksflag = 'xticks'
        else:
            xlimflag = 'xlims2'
            xticksflag = 'xticks2'

        xmin, xmax = ax.get_xlim()
        if 'xlims' in ax_settings.keys():
            if 'min' in ax_settings[xlimflag]:
                xmin = float(ax_settings[xlimflag]['min'])
            if 'max' in ax_settings[xlimflag]:
                xmax = float(ax_settings[xlimflag]['max'])

        if xticksflag in ax_settings.keys():
            xtick_settings = ax_settings[xticksflag]
            if 'fontsize' in xtick_settings.keys():
                xticksize = float(xtick_settings['fontsize'])
            elif 'fontsize' in ax_settings.keys():
                xticksize = float(ax_settings['fontsize'])
            else:
                xticksize = 10
            ax.tick_params(axis='x', labelsize=xticksize)

            if 'spacing' in xtick_settings.keys():
                xtickspacing = xtick_settings['spacing']
                if float(xtickspacing).is_integer():
                    xtickspacing = int(xtickspacing)
                else:
                    xtickspacing = float(xtickspacing)
                newxticks = np.arange(xmin, (xmax+xtickspacing), xtickspacing)
                newxticklabels = self.formatTickLabels(newxticks, xtick_settings)
                ax.set_xticks(newxticks)
                ax.set_xticklabels(newxticklabels)

        ax.set_xlim(left=xmin)
        ax.set_xlim(right=xmax)

    def plotHorizontalLines(self, straightlines, ax, object_settings, timestamp_index=0):
        '''
        organizes given data and plots horizontal lines on a given axis
        :param straightlines: dictionary potentionally containing <hlines> field
        :param ax: current axis
        :param object_settings: dictionary containing information for the entire plot
        :param timestamp_index: optional datetime index if hline is a timeseries
        '''

        if 'hlines' in straightlines.keys():
            hlines = straightlines['hlines']
            hlines = WF.correctDuplicateLabels(hlines)
            for key in hlines.keys():
                hline_settings = hlines[key]
                value = hline_settings['values'][timestamp_index]
                units = hline_settings['units']
                if 'parameter' in hline_settings.keys():
                    if object_settings['usedepth'].lower() == 'true':
                        if hline_settings['parameter'].lower() == 'elevation':
                            valueconv = self.Report.Profiles.convertElevationsToDepths({'depths': [],
                                                                                       'elevations': [value]},
                                                                                       {},
                                                                                       timestamp_index=timestamp_index)
                            if len(valueconv['hline']['depths']) == 0:
                                WF.print2stdout('Unable to convert horizontal line elevations to depths.', debug=self.Report.debug)
                                value = np.nan
                            else:
                                value = valueconv['hline']['depths'][0]
                    elif object_settings['usedepth'].lower() == 'false':
                        if hline_settings['parameter'].lower() == 'depth':
                            valueconv = self.Report.Profiles.convertDepthsToElevations({'depths': [value],
                                                                                        'elevations': []},
                                                                                        {},
                                                                                        timestamp_index=timestamp_index)
                            if len(valueconv['hline']['depths']) == 0:
                                WF.print2stdout('Unable to convert horizontal line depths to elevations.', debug=self.Report.debug)
                                value = np.nan
                            else:
                                value = valueconv['hline']['elevations'][0]

                #currently cant convert these units..
                if units != None:
                    if 'y_unitsystem' in object_settings.keys():
                        unitsystem = object_settings['y_unitsystem']
                    else:
                        unitsystem = object_settings['unitsystem']
                    valueconv, units = WF.convertUnitSystem(value, units, unitsystem, debug=self.Report.debug)
                    value = valueconv

                ### instead, use scalar to be manual
                if 'scalar' in hline_settings.keys():
                    value *= float(hline_settings['scalar'])

                hline_settings = WD.getDefaultStraightLineSettings(hline_settings, self.Report.debug)
                hline_settings = WF.fixDuplicateColors(hline_settings) #used the line, used param, then double up so subtract 1

                if 'label' not in hline_settings.keys():
                    hline_settings['label'] = None
                if 'zorder' not in hline_settings.keys():
                    hline_settings['zorder'] = 3

                ax.axhline(value, label=hline_settings['label'], c=hline_settings['linecolor'],
                           lw=hline_settings['linewidth'], ls=hline_settings['linestylepattern'],
                           zorder=float(hline_settings['zorder']),
                           alpha=float(hline_settings['alpha']))

    def plotVerticalLines(self, straightlines, ax, object_settings, timestamp_index=0, isdate=True):
        '''
        organizes given data and plots vertical lines on a given axis
        :param straightlines: dictionary potentionally containing <vlines> field
        :param ax: current axis
        :param object_settings: dictionary containing information for the entire plot
        :param isdate: optional if input value is a date, in which case we may need to format it
        :param timestamp_index: optional datetime index if vline is a timeseries
        '''

        if 'vlines' in straightlines.keys():
            vlines = straightlines['vlines']
            vlines = WF.correctDuplicateLabels(vlines)
            for key in vlines.keys():
                vline_settings = vlines[key]
                value = vline_settings['values'][timestamp_index]
                units = vline_settings['units']

                if isdate:
                    value = WT.translateDateFormat(value, 'datetime', '',
                                                   self.Report.StartTime, self.Report.EndTime,
                                                   debug=self.Report.debug)

                if 'label' not in vline_settings.keys():
                    vline_settings['label'] = None
                if 'zorder' not in vline_settings.keys():
                    vline_settings['zorder'] = 3

                if units != None:
                    valueconv, units = WF.convertUnitSystem(value, units, object_settings['unitsystem'], debug=self.Report.debug)
                    value = valueconv

                vline_settings = WD.getDefaultStraightLineSettings(vline_settings, self.Report.debug)
                vline_settings = WF.fixDuplicateColors(vline_settings) #used the line, used param, then double up so subtract 1

                ax.axvline(value, label=vline_settings['label'], c=vline_settings['linecolor'],
                           lw=vline_settings['linewidth'], ls=vline_settings['linestylepattern'],
                           zorder=float(vline_settings['zorder']),
                           alpha=float(vline_settings['alpha']))

    def fixEmptyYAxis(self, ax, ax2, keepblankax, keepblankax2):
        '''
        checks for empty axis and if there is, removes the yticks on that side
        :param ax: left axis object
        :param ax2: right axis object
        :param keepblankax: keeps left axis even if there is no data
        :param keepblankax2: keeps right axis even if there is no data
        '''

        ax_lines, _ = ax.get_legend_handles_labels()
        ax2_lines, _ = ax2.get_legend_handles_labels()
        if len(ax_lines) == 0 and keepblankax.lower() != 'true':
            ax.set_yticks([])
            ax.set_yticklabels([])
        if len(ax2_lines) == 0 and keepblankax2.lower() != 'true':
            ax2.set_yticks([])
            ax2.set_yticklabels([])

    def setInitialXlims(self, ax, year):
        '''
        sets xlimits for plots depending on plot limits to then be modified
        :param ax: axis object
        :param year: current plot year, or ALLYEARS
        '''

        if year == 'ALLYEARS':
            xmin = self.Report.StartTime
            xmax = self.Report.EndTime
        else:
            if isinstance(year, str):
                yrsplit = year.split('-')
                tmpmin = dt.datetime(int(yrsplit[0]), 1, 1, 0, 0)
                tmpmax = dt.datetime(int(yrsplit[1])+1, 1, 1, 0, 0)
            else:
                tmpmin = dt.datetime(year, 1, 1, 0, 0)
                tmpmax = dt.datetime(year+1, 1, 1, 0, 0)
            if tmpmin < self.Report.StartTime:
                xmin = self.Report.StartTime
            else:
                xmin = tmpmin
            if tmpmax > self.Report.EndTime:
                xmax = self.Report.EndTime
            else:
                xmax = tmpmax

        ax.set_xlim(left=xmin, right=xmax)

    def copyYTicks(self, ax, ax2, units, ax_settings):
        '''
        copies yticks from axis 1 to use for axis 2, if we want to use the same ticks, but in a different unitsystem
        :param ax: axis 1 to get ticks from
        :param ax2: axis 2 to duplicate ticks to
        :param units: units of ticks from ax
        :param ax_settings: settings for current axis
        '''

        axylims = ax.get_ylim()
        axyticks = ax.get_yticks()
        axyticklabels = ax.get_yticklabels()

        if 'unitsystem2' in ax_settings.keys():
            axylims, _ = WF.convertUnitSystem(axylims, units, ax_settings['unitsystem2'], debug=self.Report.debug)
            axyticks, _ = WF.convertUnitSystem(axyticks, units, ax_settings['unitsystem2'], debug=self.Report.debug)
            axyticklabels = self.formatTickLabels(axyticks, {})

        ax2.set_ylim(axylims)
        ax2.set_yticks(axyticks)
        ax2.set_yticklabels(axyticklabels)


def translateLineStylePatterns(LineSettings):
    '''
    translates java line style patterns to python friendly commands.
    :param LineSettings: dictionary containing keys describing how the line/points are drawn
    :return:
        LineSettings: dictionary containing keys describing how the line/points are drawn
    '''

    #java|python
    linestylesdict = {'dash': 'dashed',
                      'dash dot': 'dashdot',
                      'dash dot-dot': (0, (3, 5, 1, 5, 1, 5)), #this one doesnt get a string name?
                      'dot': 'dotted',
                      'solid': 'solid'}

    if 'linestylepattern' in LineSettings.keys():
        if LineSettings['linestylepattern'].lower() in linestylesdict.values(): #existing python values
            LineSettings['linestylepattern'] = LineSettings['linestylepattern'].lower() #use python but lower it
        else:
            try:
                LineSettings['linestylepattern'] = linestylesdict[LineSettings['linestylepattern'].lower()]
            except KeyError:
                WF.print2stdout('Invalid lineStylePattern:', LineSettings['linestylepattern'])
                WF.print2stdout('Defaulting to Solid.')
                LineSettings['linestylepattern'] = 'solid'
    else:
        WF.print2stdout('lineStylePattern undefined for line. Using solid')
        LineSettings['linestylepattern'] = 'solid'

    return LineSettings

def translatePointStylePatterns(LineSettings):
    '''
    translates java point style patterns to python friendly commands.
    :param LineSettings: dictionary containing keys describing how the line/points are drawn
    :return:
        LineSettings: dictionary containing keys describing how the line/points are drawn
    '''

    #java|python
    #https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    pointstylesdict = {1: 's', #square
                       2: 'o', #circle
                       3: '^', #triangle up
                       4: 'v', #triangle down
                       5: 'D', #diamond
                       6: '*' #star
                       }

    if 'symboltype' in LineSettings.keys():
        if LineSettings['symboltype'] in pointstylesdict.values(): #existing python values
            LineSettings['symboltype'] = LineSettings['symboltype'] #needs to be case sensitive..
        else:
            try:
                LineSettings['symboltype'] = pointstylesdict[int(LineSettings['symboltype'])]
            except:
                WF.print2stdout('Invalid Symboltype:', LineSettings['symboltype'])
                WF.print2stdout('Defaulting to Square.')
                LineSettings['symboltype'] = 's'

    else:
        WF.print2stdout('Symbol not defined. Defaulting to Square.')
        LineSettings['symboltype'] = 's'

    return LineSettings
