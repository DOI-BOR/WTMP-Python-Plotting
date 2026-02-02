"""
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
"""

VERSIONNUMBER = '6.0.26'

import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates
import numpy as np
import pickle
import itertools
import traceback
import time
import re

import WAT_Reader as WR
import WAT_Functions as WF
import WAT_XML_Utils as WXMLU
import WAT_Logger as WL
import WAT_Constants as WC
import WAT_Time as WT
import WAT_Defaults as WD
import WAT_DataOrganizer as WDO
import WAT_ResSim_Results as WRSS
import WAT_W2_Results as WW2
import WAT_Profiles as WProfile
import WAT_Tables as WTable
import WAT_Plots as WPlot
import WAT_Gates as WGates

import warnings

# warnings.simplefilter('error') #turn on for debugging warnings
warnings.filterwarnings("always")

mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
mpl.use("Agg")


class MakeAutomatedReport(object):
    """
    class to organize data and generate XML file for Jasper processing in conjunction with WAT. Takes in a simulation
    information file output from WAT and develops the report from there.
    """

    def __init__(self, simulationInfoFile, batdir):
        """
        organizes input data and generates XML report
        :param simulationInfoFile: full path to simulation information XML file output from WAT.
        """

        WF.printVersion(VERSIONNUMBER)
        self.simulationInfoFile = simulationInfoFile
        self.WriteLog = True #TODO we're testing this.
        self.batdir = batdir
        WR.readSimulationInfo(self, simulationInfoFile)  # Read file output by WAT
        self.definePaths()
        self.Constants = WC.WAT_Constants()
        self.cleanOutputDirs()
        WF.checkJasperFiles(self.studyDir, self.installDir)
        WR.readGraphicsDefaultFile(self)  # Read graphical component defaults
        self.defaultLineStyles = WD.readDefaultLineStylesFile(self)
        self.WAT_log = WL.WAT_Logger(self)
        self.reportType = WR.getReportType(self)
        # self.reportCSV = WR.readReportCSVFile(self)

        self.modelOrder = 0
        # chapterkeys = list(self.reportCSV.keys())
        # chapterkeys.sort() #these are always numbers, so it works

        if self.reportType == 'validation':

            for simulation in self.Simulations:
                self.reportCSV = WR.readReportCSVFile(self, simulation)
                # self.modelOrder = 0
                chapterkeys = list(self.reportCSV.keys())
                chapterkeys.sort()  # these are always numbers, so it works

                WF.printSimulationInfo(simulation)
                self.base_id = self.Simulations[0]['ID']
                self.initSimulationDict()
                self.setSimulationVariables(simulation)
                self.loadCurrentID(self.base_id)  # Load the data for the current sim, we do 1 at a time here.
                WF.checkExists(self.SimulationDir)
                WT.defineStartEndYears(self)
                WT.defineStartEndMonths(self)
                self.cleanOutputDirs()
                self.initializeXML()
                self.writeXMLIntroduction()
                for chapterkey in chapterkeys:
                    WF.print2stdout(f'Running XML file {self.reportCSV[chapterkey]["xmlfile"]}')
                    self.setSimulationCSVVars(self.reportCSV[chapterkey])
                    WR.readDefinitionsFile(self, self.reportCSV[chapterkey])
                    self.loadModelAlts(self.reportCSV[chapterkey])
                    self.initializeDataOrganizer()
                    self.loadCurrentModelAltID(self.base_id)
                    self.WAT_log.addSimLogEntry(self.accepted_IDs, self.SimulationVariables, self.observedDir)
                    self.writeChapter()
                    self.appendXMLModelIntroduction(chapterkey)
                    self.Data.writeDataFiles()
                self.fixXMLModelIntroduction()
                self.XML.writeReportEnd()
                self.WAT_log.equalizeLog()

        elif self.reportType == 'comparison':
            self.initSimulationDict()
            for simulation in self.Simulations:
                WF.printSimulationInfo(simulation)
                self.setSimulationVariables(simulation)
                WF.checkExists(simulation['directory'])
            self.base_id = 'base'  # this should always be base?
            self.loadCurrentID(self.base_id)  # load the data for the current sim, we do 1 at a time here.
            WT.setMultiRunStartEndYears(self)  # find the start and end time
            WT.defineStartEndYears(self)  # format the years correctly after they are set
            WT.defineStartEndMonths(self)  # format the months correctly after they are set
            self.cleanOutputDirs()
            self.initializeXML()
            self.reportCSV = WR.readReportCSVFile(self, [sim for sim in self.Simulations if sim['ID'] == self.base_id][
                0])  # use the base
            chapterkeys = list(self.reportCSV.keys())
            chapterkeys.sort()  # these are always numbers, so it works
            self.writeXMLIntroduction()
            for chapterkey in chapterkeys:
                WF.print2stdout(f'Running XML file {self.reportCSV[chapterkey]["xmlfile"]}')
                self.setSimulationCSVVars(self.reportCSV[chapterkey])
                WR.readDefinitionsFile(self, self.reportCSV[chapterkey])
                self.loadModelAlts(self.reportCSV[chapterkey])
                self.initializeDataOrganizer()
                self.loadCurrentModelAltID(self.base_id)
                self.WAT_log.addSimLogEntry(self.accepted_IDs, self.SimulationVariables, self.observedDir)
                self.writeChapter()
                self.appendXMLModelIntroduction(chapterkey)
                self.Data.writeDataFiles()
            self.fixXMLModelIntroduction()
            self.XML.writeReportEnd()
            self.WAT_log.equalizeLog()

        elif self.reportType == 'forecast':
            self.initSimulationDict()
            self.organizeMembers()
            self.Ensemble = None
            self.member = None
            self.plot_name = None

            self.original_section_header = None
            self.skip_identical_plot = False

            for simulation in self.Simulations:
                WF.printSimulationInfo(simulation)
                WF.checkExists(simulation['directory'])
                self.setSimulationVariables(simulation)
                # self.base_id = self.Simulations[0]['ID']
                self.base_id = 'base'
                self.loadCurrentID(self.base_id)  # load the first simulation
                WT.setMultiRunStartEndYears(self)  # find the start and end time
                WT.defineStartEndYears(self)  # format the years correctly after they are set
                WT.defineStartEndMonths(self)  # format the months correctly after they are set
                # WR.readForecastSimulationsCSV(self) #read to determine order/sims/regions in report
                self.reportCSV = WR.readReportCSVFile(self,[sim for sim in self.Simulations if sim['ID'] == self.base_id][0])  # use the base
                self.cleanOutputDirs()
                self.initializeXML()
                self.writeXMLIntroduction()
                self.initializeDataOrganizer()  # Do this here because forecast runs are likely to overlap over chapters

                chapterkeys = list(self.reportCSV.keys())
                chapterkeys.sort()  # these are always numbers, so it works
                for chapterkey in chapterkeys:
                    WF.print2stdout(f'Running XML file {self.reportCSV[chapterkey]["xmlfile"]}')
                    self.setSimulationCSVVars(self.reportCSV[chapterkey])
                    WR.readDefinitionsFile(self, self.reportCSV[chapterkey])
                    self.loadModelAlts(self.reportCSV[chapterkey])
                    self.loadCurrentModelAltID(self.base_id)
                    self.WAT_log.addSimLogEntry(self.accepted_IDs, self.SimulationVariables, self.observedDir)
                    self.writeChapter()
                    self.Data.writeDataFiles()
                    self.appendXMLModelIntroduction(chapterkey)
                self.fixXMLModelIntroduction()
                self.XML.writeReportEnd()
                self.WAT_log.equalizeLog()

        else:
            WF.print2stderr('UNKNOWN REPORT TYPE:', self.reportType)
            sys.exit(1)

        self.WAT_log.writeLogFile(self.images_path)
        self.Data.writeDataFiles()

    def definePaths(self):
        """
        defines run specific paths
        used to contain more paths, but not needed. Consider moving.
        :return: set class variables
                    self.images_path
        """

        self.images_path = os.path.join(self.outputDir, 'Images')
        if not os.path.exists(self.images_path):
            try:
                os.makedirs(self.images_path)
                WF.print2stdout(f'{self.images_path} created!')
            except:
                WF.print2stderr(f'Unable to make {self.images_path}')
                sys.exit(1)

        self.CSVPath = os.path.join(self.outputDir, 'CSVData')  #TODO: update

        if not os.path.exists(self.CSVPath):
            try:
                os.makedirs(self.CSVPath)
                WF.print2stdout(f'{self.CSVPath} created!')
            except:
                WF.print2stderr(f'Unable to make {self.CSVPath}')
                sys.exit(1)


    def makeTimeSeriesPlot(self, object_settings):
        """
        Takes in object settings to build time series plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making TimeSeries Plot.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Plots = WPlot.Plots(self)

        default_settings = self.loadDefaultPlotObject('timeseriesplot') # get default TS plot items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) # overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', exclude=['description'])
        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', include=['description'],
                                                  forjasper=True)

        # update the title and descriptions with anything from general or model specific
        object_settings = WF.replaceflaggedValues(self, object_settings, 'general', include=['description', 'title'])
        object_settings = WF.replaceflaggedValues(self, object_settings, 'modelspecific',
                                                  include=['description', 'title'])

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)
        object_settings = self.Plots.confirmAxis(object_settings)

        object_settings['years'], object_settings['yearstr'] = WF.organizePlotYears(object_settings)


        if 'description' not in object_settings.keys():
            object_settings['description'] = ''
        # else:
        #     object_settings['description'] = WF.parseForTextFlags(object_settings['description'])

        for yi, year in enumerate(object_settings['years']):
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))
            yearstr = object_settings['yearstr'][yi]

            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)
            if self.memberiteration:
                cur_obj_settings['member'] = self.member
                cur_obj_settings['memberiteration'] = self.memberiteration

            if len(cur_obj_settings['axs']) == 1:
                figsize = (12, 6)
                pageformat = 'half'
            else:
                figsize = (12, 14)
                pageformat = 'full'

            axis_weight = []
            for ax_settings in cur_obj_settings['axs']:
                if 'weight' in ax_settings.keys():
                    axis_weight.append(float(ax_settings['weight']))
                else:
                    axis_weight.append(1)

            if 'sharex' not in cur_obj_settings:
                cur_obj_settings['sharex'] == 'false'

            if cur_obj_settings['sharex'].lower() == 'true':
                fig, axes = plt.subplots(nrows=len(cur_obj_settings['axs']), sharex=True, figsize=figsize,
                                         gridspec_kw={'height_ratios': axis_weight})
            else:
                fig, axes = plt.subplots(ncols=1, nrows=len(cur_obj_settings['axs']), figsize=figsize,
                                         gridspec_kw={'height_ratios': axis_weight})

            left_sided_axes = []
            right_sided_axes = []
            useAx = []
            fullyduplicate = False

            for axi, ax_settings in enumerate(cur_obj_settings['axs']):

                ax_settings = WF.copyKeysBetweenDicts(ax_settings, cur_obj_settings, ignore=['axs'])

                if len(cur_obj_settings['axs']) == 1:
                    ax = axes
                else:
                    ax = axes[axi]

                ax_settings = self.Plots.setTimeSeriesXlims(ax_settings, yearstr, object_settings['years'])
                self.Plots.setInitialXlims(ax, year)

                ### Make Twin axis ###
                _usetwinx = False
                if 'twinx' in ax_settings.keys():
                    if ax_settings['twinx'].lower() == 'true':
                        _usetwinx = True

                _usetwiny = False
                if 'twiny' in object_settings.keys():
                    if object_settings['twiny'].lower() == 'true':
                        _usetwiny = True

                if _usetwinx:
                    ax2 = ax.twinx()

                unitslist = []
                unitslist2 = []
                stackplots = {}

                linedata, line_settings = self.Data.getTimeSeriesDataDictionary(ax_settings)

                if self.memberiteration:
                    if self.groupmembers:
                        if self.Data.checkForDuplicateObject(line_settings, self.member): #for member iteration plots, skip duplicates. If the first instance of a duplicate plot, plot it but change hte header + description
                            is_lowest = self.Data.checkForLowestMember(line_settings, self.member)
                            if is_lowest:
                                fullyduplicate = False #if the first instance of a duplicate plot, plot it but change the header description
                                other_members = self.Data.getOtherMembers(line_settings, self.member)
                                member_list_frmt = (f"{WF.getOriginalMemberNumber(self.member, WF.matchMemberToEnsembleSet(self.ensembleSets, self.member), self.DSSFile, self.alternativeFpart, self.StartTime, self.EndTime, self.debug)}, " +
                                                    ', '.join([WF.getOriginalMemberNumber(mem, WF.matchMemberToEnsembleSet(self.ensembleSets, mem), self.DSSFile, self.alternativeFpart, self.StartTime, self.EndTime, self.debug) for mem in other_members]))
                                cur_obj_settings['description'] = WF.updateFlaggedValues(cur_obj_settings['description'], '%%member%%', member_list_frmt)  # updates timeseriesplots png descriptions
                                ax_settings['title'] = WF.updateFlaggedValues(ax_settings['title'], '%%member%%', member_list_frmt)

                                # update met name in description and title
                                cur_obj_settings['description'] = WF.updateFlaggedValues(cur_obj_settings['description'], '%%metname%%',
                                                                                         WF.matchMemberToEnsembleSet(self.ensembleSets, self.member)['metname'])
                                ax_settings['title'] = WF.updateFlaggedValues(ax_settings['title'], '%%metname%%',
                                                                              WF.matchMemberToEnsembleSet(self.ensembleSets, self.member)['metname'])
                            else:
                                fullyduplicate = True
                                WF.print2stdout('Duplicate plot found. Skipping plot.')
                                plt.close('all')
                                return
                        else:
                            fullyduplicate = False
                            # get the ensemble set for the current member
                            curr_ensemble_set =  WF.matchMemberToEnsembleSet(self.ensembleSets, self.member)

                            # update the member number
                            cur_obj_settings['description'] = WF.updateFlaggedValues(cur_obj_settings['description'], '%%member%%', WF.getOriginalMemberNumber(self.member, curr_ensemble_set, self.DSSFile,
                                                                                                                                                               self.alternativeFpart, self.StartTime, self.EndTime, self.debug))
                            ax_settings['title'] = WF.updateFlaggedValues(ax_settings['title'], '%%member%%', WF.getOriginalMemberNumber(self.member, curr_ensemble_set, self.DSSFile,
                                                                                                                                         self.alternativeFpart, self.StartTime, self.EndTime, self.debug))

                            # update met name in description and title
                            cur_obj_settings['description'] = WF.updateFlaggedValues(cur_obj_settings['description'], '%%metname%%', curr_ensemble_set['metname'])
                            ax_settings['title'] = WF.updateFlaggedValues(ax_settings['title'], '%%metname%%', curr_ensemble_set['metname'])
                    else:
                        # get the ensemble set for the current member
                        curr_ensemble_set = WF.matchMemberToEnsembleSet(self.ensembleSets, self.member)

                        # update the member number
                        cur_obj_settings['description'] = WF.updateFlaggedValues(cur_obj_settings['description'], '%%member%%', WF.getOriginalMemberNumber(self.member, curr_ensemble_set, self.DSSFile,
                                                                                                                                                           self.alternativeFpart, self.StartTime, self.EndTime, self.debug))
                        ax_settings['title'] = WF.updateFlaggedValues(ax_settings['title'], '%%member%%', WF.getOriginalMemberNumber(self.member, curr_ensemble_set, self.DSSFile,
                                                                                                                                     self.alternativeFpart, self.StartTime, self.EndTime, self.debug))

                        # update met name in description and title
                        cur_obj_settings['description'] = WF.updateFlaggedValues(cur_obj_settings['description'], '%%metname%%', curr_ensemble_set['metname'])
                        ax_settings['title'] = WF.updateFlaggedValues(ax_settings['title'], '%%metname%%', curr_ensemble_set['metname'])

                linedata = WF.filterDataByYear(linedata, year)
                linedata = self.Data.filterTimeSeries(linedata, line_settings)
                linedata = self.Data.scaleValuesByTable(linedata, line_settings)
                linedata, line_settings = WF.mergeLines(linedata, line_settings, ax_settings)
                ax_settings = self.configureSettingsForID(self.base_id, ax_settings)
                gatedata, gate_settings = self.Data.getGateDataDictionary(ax_settings, makecopy=False)
                line_settings = WF.correctDuplicateLabels(line_settings)
                straightlines = self.Data.getStraightLineValue(ax_settings)

                isCollection = WF.checkForCollections(line_settings)

                for gateop in gatedata.keys():
                    gatedata[gateop]['gates'] = WF.filterDataByYear(gatedata[gateop]['gates'], year)

                if 'relative' in ax_settings.keys():
                    if ax_settings['relative'].lower() == 'true':
                        RelativeMasterSet, RelativeLineSettings = self.Plots.getRelativeMasterSet(linedata, line_settings)
                        if 'unitsystem' in ax_settings.keys():
                            RelativeMasterSet, RelativeLineSettings['units'] = WF.convertUnitSystem(RelativeMasterSet,
                                                                                                    RelativeLineSettings['units'],
                                                                                                    ax_settings['unitsystem'])

                for line in linedata:
                    curline = linedata[line]
                    curline_settings = line_settings[line]
                    parameter, ax_settings['param_count'] = WF.getParameterCount(curline_settings, ax_settings)
                    i = ax_settings['param_count'][parameter]

                    values = curline['values']
                    dates = curline['dates']
                    units = curline_settings['units']

                    if not curline_settings['collection']:  # please don't do this for collection plots
                        values = WF.ValueSum(dates, values)  # check for dict values and add them all together

                    curline_settings['stack'] = False
                    if 'linetype' in curline_settings.keys():
                        if curline_settings['linetype'].lower() == 'stacked':  # Stacked plots need to be added at the end.
                            if _usetwinx:
                                if 'yaxis' in curline_settings.keys():
                                    axis = curline_settings['yaxis'].lower()
                            else:
                                axis = 'left'  #if not twinx, then only can use left

                            curline_settings['stack'] = True

                    if units is None:
                        if parameter is not None:
                            try:
                                units = self.Constants.units[parameter]
                            except KeyError:
                                units = None

                    if isinstance(units, dict):
                        if 'unitsystem' in ax_settings.keys():
                            units = units[ax_settings['unitsystem'].lower()]
                        else:
                            units = None

                    if 'unitsystem' in ax_settings.keys():
                        values, units = WF.convertUnitSystem(values, units, ax_settings['unitsystem'])

                    chkvals = WF.checkData(values)
                    if not chkvals:
                        WF.print2stdout('Invalid Data settings for line:', line, debug=self.debug)
                        continue

                    dates = WT.JDateToDatetime(dates, self.startYear)

                    if 'elevation_storage_area_file' in curline_settings.keys():
                        values = WF.calculateStorageFromElevation(values, curline_settings)

                    if 'scalar' in curline_settings.keys():
                        try:
                            scalar = float(curline_settings['scalar'])
                            if self.reportType == 'forecast':
                                if isinstance(values, np.ndarray):
                                    values = scalar * values
                                elif isinstance(values, dict):
                                    for member in values.keys():
                                        values[member] = scalar * values[member]
                            else:
                                values = scalar * values
                        except ValueError:
                            WF.print2stdout('Invalid Scalar. {0}'.format(curline_settings['scalar']), debug=self.debug)
                            continue

                    line_draw_settings = WD.getDefaultLineSettings(self.defaultLineStyles, curline_settings, parameter, i, debug=self.debug)
                    line_draw_settings = WF.fixDuplicateColors(line_draw_settings) #used the line, used param, then double up so subtract 1

                    if 'zorder' not in line_draw_settings.keys():
                        line_draw_settings['zorder'] = 4

                    if 'label' not in line_draw_settings.keys():
                        line_draw_settings['label'] = ''
                    else:
                        line_draw_settings['label'] = WF.formatTextFlags(line_draw_settings['label'])

                    curax = ax
                    axis2 = False
                    if _usetwinx:
                        if 'yaxis' in line_draw_settings.keys():
                            if line_draw_settings['yaxis'].lower() == 'right':
                                curax = ax2
                                axis2 = True

                    if units != '' and units is not None:
                        if axis2:
                            unitslist2.append(units)
                        else:
                            unitslist.append(units)

                    if 'relative' in ax_settings:
                        if ax_settings['relative'].lower() == 'true':
                            if RelativeLineSettings['interval'] is not None:
                                dates, values = WT.changeTimeSeriesInterval(dates, values, RelativeLineSettings,
                                                                            self.startYear)
                            values = values / RelativeMasterSet

                    if isCollection:  #if we have a collection for the datasource
                        coloreach = False
                        plotallmembers = False
                        if 'coloreach' in curline_settings.keys():
                            if curline_settings['coloreach'].lower() == 'true':
                                coloreach = True
                        if 'plotallmembers' in curline_settings.keys():
                            if curline_settings['plotallmembers'].lower() == 'true':
                                plotallmembers = True
                        if not coloreach:
                            if not self.memberiteration or plotallmembers:
                                modifiedalpha = False
                                if line_draw_settings['alpha'] == 1.:
                                    modifiedalpha = True
                                    line_draw_settings['alpha'] = 0.25 #for collection plots, set to low opac for a jillion lines

                                for cIT, member in enumerate(curline_settings['members']):
                                    valueset = values[member]
                                    if cIT > 0:
                                        line_draw_settings['label'] = ''
                                    self.Plots.plot(dates, valueset, curax, line_draw_settings)
                                if modifiedalpha:
                                    line_draw_settings['alpha'] = 1.
                            elif self.memberiteration:
                                valueset = values[self.member]
                                if curline_settings['stack']:
                                    if axis not in stackplots.keys():  # left or right
                                        stackplots[axis] = []
                                    stackplots[axis].append({'values': valueset,
                                                             'dates': dates,
                                                             'label': line_draw_settings['label'],
                                                             'color': line_draw_settings['linecolor']})
                                else:
                                    self.Plots.plot(dates, valueset, curax, line_draw_settings)

                        else:
                            single_coll_line_settings = self.Plots.seperateCollectionLines(line_draw_settings)
                            for member in curline_settings['members']:
                                valueset = values[member]
                                coll_line_settings = single_coll_line_settings[member]
                                self.Plots.plot(dates, valueset, curax, coll_line_settings)

                        self.Plots.plotCollectionEnvelopes(dates, values, curax, line_draw_settings)

                    elif curline_settings['stack']:
                        if axis not in stackplots.keys():  # left or right
                            stackplots[axis] = []
                        stackplots[axis].append({'values': values,
                                                 'dates': dates,
                                                 'label': line_draw_settings['label'],
                                                 'color': line_draw_settings['linecolor']})

                    else:
                        self.Plots.plot(dates, values, curax, line_draw_settings)

                        self.WAT_log.addLogEntry({'type': line_draw_settings['label'] +
                                                  '_TimeSeries' if line_draw_settings['label'] != '' else 'Timeseries',
                                                  'name': self.ChapterRegion + '_' + yearstr,
                                                  'description': ax_settings['description'],
                                                  'units': units,
                                                  'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                             self.StartTime, self.EndTime,
                                                                                             debug=self.debug).strftime('%d %b %Y'),
                                                  'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                                           self.StartTime, self.EndTime,
                                                                                           debug=self.debug).strftime('%d %b %Y'),
                                                  'logoutputfilename': line_draw_settings['logoutputfilename']
                                                  },
                                                 isdata=True)

                # GATE DATA #
                if 'gatespacing' in ax_settings.keys():
                    gatespacing = float(ax_settings['gatespacing'])
                else:
                    gatespacing = 3
                gate_placement = gatespacing

                gategroup_labels = []
                gatelabels_positions = []
                gateop_rev = list(gatedata.keys())
                if len(gateop_rev) > 1:

                    gateline_pos = {}
                    line_pos = gatespacing
                    for gateop in gateop_rev[::-1]:
                        gate_line_settings = gatedata[gateop]
                        for gate in list(gate_line_settings['gates'].keys())[::-1]:
                            gateline_pos[f"{gateop}_{gate}"] = line_pos
                            line_pos += 1
                        line_pos += gatespacing

                    for ggi, gateop in enumerate(gateop_rev):
                        # gate_placement += ggi*gatespacing
                        gate_count = 0  # keep track of gate number in group
                        if 'label' in gate_settings[gateop]:
                            gategroup_label = WF.formatTextFlags(gate_settings[gateop]['label'])
                            gategroup_labels.append(gategroup_label)
                        elif 'flag' in gate_settings[gateop]:
                            gategroup_flag = WF.formatTextFlags(gate_settings[gateop]['flag'])
                            gategroup_labels.append(gategroup_flag)
                        else:
                            gategroup_labels.append(gateop)

                        gatelines_positions = []
                        for gate in gatedata[gateop]['gates'].keys():

                            curgate = gatedata[gateop]['gates'][gate]
                            curgate_settings = gate_settings[gateop]['gates'][gate]
                            values = curgate['values']
                            dates = curgate['dates']

                            if len(dates) > 0:

                                dates = WT.JDateToDatetime(dates, self.startYear)

                                gate_line_settings = WD.getDefaultGateLineSettings(curgate_settings, gate_count, debug=self.debug)

                                if 'zorder' not in gate_line_settings.keys():
                                    gate_line_settings['zorder'] = 4

                                if 'label' not in gate_line_settings.keys():
                                    gate_line_settings['label'] = '{0}_{1}'.format(gateop, gate_count)

                                line_placement = gateline_pos[f"{gateop}_{gate}"]
                                gatelines_positions.append(line_placement)
                                gatevalues = line_placement * values

                                curax = ax
                                if _usetwinx:
                                    if 'xaxis' in line_settings.keys():
                                        if 'xaxis'.lower() == 'right':
                                            curax = ax2

                                self.Plots.plot(dates, gatevalues, curax, gate_line_settings)

                                gate_count += 1  #keep track of gate number in group
                                gate_placement += 1  #keep track of gate palcement in space
                                self.WAT_log.addLogEntry({'type': gate_line_settings['label'] +
                                                          '_GateTimeSeries' if gate_line_settings['label'] != '' else 'GateTimeseries',
                                                          'name': self.ChapterRegion + '_' + yearstr,
                                                          'description': ax_settings['description'],
                                                          'units': 'BINARY',
                                                          'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                                     self.StartTime, self.EndTime,
                                                                                                     debug=self.debug).strftime('%d %b %Y'),
                                                          'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                                                   self.StartTime, self.EndTime,
                                                                                                   debug=self.debug).strftime('%d %b %Y'),
                                                          'logoutputfilename': curgate_settings['logoutputfilename']
                                                          },
                                                         isdata=True)

                        gatelabels_positions.append(np.average(gatelines_positions))
                        gate_placement += gatespacing

                if 'operationlines' in ax_settings.keys():
                    operationtimes = WF.getGateOperationTimes(gatedata)
                    axs_to_add_line = [ax]
                    if 'allaxis' in ax_settings['operationlines'].keys():
                        if ax_settings['operationlines']['allaxis'].lower() == 'true':
                            axs_to_add_line = axes

                    opline_settings = WD.getDefaultStraightLineSettings(ax_settings['operationlines'], self.debug)

                    for ax_to_add_line in axs_to_add_line:
                        for operationTime in operationtimes:
                            operationTime = WT.translateDateFormat(operationTime, 'datetime', '',
                                                                   self.StartTime, self.EndTime,
                                                                   debug=self.debug)

                            if 'zorder' not in opline_settings.keys():
                                opline_settings['zorder'] = 3

                            ax_to_add_line.axvline(operationTime, c=opline_settings['linecolor'],
                                                   lw=opline_settings['linewidth'],
                                                   ls=opline_settings['linestylepattern'],
                                                   zorder=float(opline_settings['zorder']),
                                                   alpha=float(opline_settings['alpha']))

                for stackplot_ax in stackplots.keys():
                    if stackplot_ax == 'left':
                        curax = ax
                    elif stackplot_ax == 'right':
                        curax = ax2

                    sps = stackplots[stackplot_ax]
                    stackvalues = [n['values'] for n in sps]
                    stackdates = [n['dates'] for n in sps]
                    stacklabels = [n['label'] for n in sps]
                    stackcolors = [n['color'] for n in sps]

                    matched_dates = list(set(stackdates[0]).intersection(*stackdates))  #find dates that ALL dates have.
                    matched_dates.sort()

                    if len(matched_dates) == 0:
                        WF.print2stdout('Mismatching dates for stack plot.', debug=self.debug)
                        WF.print2stdout(
                            'Please check inputs are on the date interval and time stamps are on the same hours.',
                            debug=self.debug)

                    # Now filter values associated with dates not in this list
                    for di, datelist in enumerate(stackdates):
                        mask_date_idx = [ni for ni, date in enumerate(datelist) if date in matched_dates]
                        stackvalues[di] = np.asarray(stackvalues[di])[mask_date_idx]

                    curax.stackplot(matched_dates, stackvalues, labels=stacklabels, colors=stackcolors, zorder=2)

                ### VERTICAL LINES ###
                self.Plots.plotVerticalLines(straightlines, ax, cur_obj_settings, isdate=True)

                ### Horizontal LINES ###
                self.Plots.plotHorizontalLines(straightlines, ax, cur_obj_settings)

                plotunits = WF.getPlotUnits(unitslist, ax_settings)
                if len(unitslist2) == 0 and 'unitsystem2' in ax_settings.keys():
                    _, plotunits2 = WF.convertUnitSystem([], plotunits, ax_settings['unitsystem2'], debug=self.debug)
                else:
                    plotunits2 = WF.getPlotUnits(unitslist2, ax_settings)
                ax_settings = WF.updateFlaggedValues(ax_settings, '%%units%%', WF.formatUnitsStrings(plotunits))
                ax_settings = WF.updateFlaggedValues(ax_settings, '%%units2%%', WF.formatUnitsStrings(plotunits2))

                if axi == 0:
                    if 'title' in ax_settings.keys():
                        if 'titlesize' in ax_settings.keys():
                            titlesize = float(ax_settings['titlesize'])
                        elif 'fontsize' in ax_settings.keys():
                            titlesize = float(ax_settings['fontsize'])
                        else:
                            titlesize = 15

                        title = WF.formatTextFlags(ax_settings['title'])

                        ax.set_title(title, fontsize=titlesize, wrap=True)

                if 'gridlines' in ax_settings.keys():
                    if ax_settings['gridlines'].lower() == 'true':
                        ax.grid(True)

                if 'ylabel' in ax_settings.keys():
                    if 'ylabelsize' in ax_settings.keys():
                        ylabsize = float(ax_settings['ylabelsize'])
                    elif 'fontsize' in ax_settings.keys():
                        ylabsize = float(ax_settings['fontsize'])
                    else:
                        ylabsize = 12
                    ylabel = WF.formatTextFlags(ax_settings['ylabel'])
                    ax.set_ylabel(ylabel, fontsize=ylabsize)

                if 'xlabel' in ax_settings.keys():
                    if 'xlabelsize' in ax_settings.keys():
                        xlabsize = float(ax_settings['xlabelsize'])
                    elif 'fontsize' in ax_settings.keys():
                        xlabsize = float(ax_settings['fontsize'])
                    else:
                        xlabsize = 12
                    xlabel = WF.formatTextFlags(ax_settings['xlabel'])
                    ax.set_xlabel(xlabel, fontsize=xlabsize)

                ############# xticks and lims #############

                useplot = self.Plots.formatDateXAxis(ax, ax_settings)
                if not useplot:
                    useAx.append(False)
                else:
                    useAx.append(True)

                xmin, xmax = ax.get_xlim()

                xmin = mpl.dates.num2date(xmin)
                xmax = mpl.dates.num2date(xmax)

                if 'xticks' in ax_settings.keys():
                    xtick_settings = ax_settings['xticks']
                else:
                    xtick_settings = {}
                self.Plots.formatTimeSeriesXticks(ax, xtick_settings, ax_settings)

                ax.set_xlim(left=xmin)
                ax.set_xlim(right=xmax)

                if _usetwiny:
                    ax2y = ax.twiny()
                    ax2y.set_xlim(left=xmin)
                    ax2y.set_xlim(right=xmax)
                    useplot = self.Plots.formatDateXAxis(ax2y, object_settings, twin=True)
                    xmin, xmax = ax2y.get_xlim()

                    xmin = mpl.dates.num2date(xmin)
                    xmax = mpl.dates.num2date(xmax)

                    if 'xticks2' in ax_settings.keys():
                        xtick_settings = ax_settings['xticks2']
                    else:
                        xtick_settings = {}
                    if 'copybottom' in xtick_settings.keys():
                        if xtick_settings['copybottom'].lower() == 'true':
                            ax2y.set_xticks(ax.get_xticks())

                    self.Plots.formatTimeSeriesXticks(ax2y, xtick_settings, ax_settings,
                                                      dateformatflag='dateformat2')

                    ax2y.set_xlim(left=xmin)
                    ax2y.set_xlim(right=xmax)

                    ax2y.grid(False)

                ############# yticks and lims #############
                self.Plots.formatYTicks(ax, ax_settings, gatedata, gate_placement)

                if _usetwinx:
                    if 'keepblankax' in ax_settings.keys():
                        keepblankax = ax_settings['keepblankax']
                    else:
                        keepblankax = 'false'
                    if 'keepblankax2' in ax_settings.keys():
                        keepblankax2 = ax_settings['keepblankax2']
                    else:
                        keepblankax2 = 'false'
                    self.Plots.fixEmptyYAxis(ax, ax2, keepblankax, keepblankax2)

                if len(gatelabels_positions) > 0:
                    ax.set_yticks(gatelabels_positions)
                    ax.set_yticklabels(gategroup_labels, rotation=90, va='center', ha='center')
                    ax.tick_params(axis='both', which='both', color='w')

                if _usetwinx:
                    if 'ylabel2' in ax_settings.keys():
                        if 'ylabelsize2' in ax_settings.keys():
                            ylabsize2 = float(ax_settings['ylabelsize2'])
                        elif 'fontsize' in cur_obj_settings.keys():
                            ylabsize2 = float(ax_settings['fontsize'])
                        else:
                            ylabsize2 = 12
                        ylabel2 = WF.formatTextFlags(ax_settings['ylabel2'])
                        ax2.set_ylabel(ylabel2, fontsize=ylabsize2)

                    copied_ticks = False
                    if 'sameyticks' in ax_settings:
                        if ax_settings['sameyticks'].lower() == 'true':
                            self.Plots.copyYTicks(ax, ax2, units, ax_settings)
                            copied_ticks = True
                    if not copied_ticks:
                        self.Plots.formatYTicks(ax2, ax_settings, gatedata, gate_placement, axis='right')

                    if len(ax.get_lines()) > 0:
                        ax2.grid(False)
                    else:
                        ax2.grid(True)
                    ax.set_zorder(ax2.get_zorder() + 1)  # axis called second will always be on top unless this
                    ax.patch.set_visible(False)

                if 'legend' in ax_settings.keys():
                    plt.gcf().canvas.draw()
                    if ax_settings['legend'].lower() == 'true':
                        if 'legendsize' in ax_settings.keys():
                            legsize = float(ax_settings['legendsize'])
                        elif 'fontsize' in ax_settings.keys():
                            legsize = float(ax_settings['fontsize'])
                        else:
                            legsize = 12

                        handles, labels = ax.get_legend_handles_labels()

                        plot_blank = False
                        if _usetwinx:
                            if len(handles) > 0:
                                if 'useblanklegendentry' in ax_settings.keys():
                                    if ax_settings['useblanklegendentry'].lower() == 'true':
                                        plot_blank = True
                                if plot_blank:
                                    empty_handle, = ax.plot([], [], color="w", alpha=0.0)
                                    handles.append(empty_handle)
                                    labels.append('')
                            ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
                            handles += ax2_handles
                            labels += ax2_labels
                            right_sided_axes.append([ax, ax2])
                            ax2ylabel = ax2.get_ylabel()
                            if ax2ylabel != '':
                                ax2setylabel = ax2.set_ylabel(ax2ylabel)
                                ylabel_x1 = ax2setylabel.get_window_extent().x1
                                right_offset = ax.get_window_extent().x0 / (ax.get_window_extent().width - ax2setylabel.get_window_extent().width)
                                right_offset *= 1.20
                            else:
                                right_offset = ax.get_window_extent().x0 / ax.get_window_extent().width

                        if 'numlegendcolumns' in ax_settings:
                            numcols = int(ax_settings['numlegendcolumns'])
                        else:
                            numcols = 1

                        if len(handles) > 0:

                            # If legend outside is set to right, put the legend to the right of the plot
                            if ax_settings['legend_outside'].lower() == 'right':  #TODO: calibrate the offset
                                if _usetwinx:

                                    ax.legend(handles=handles, labels=labels, loc='center left',
                                              bbox_to_anchor=(1 + right_offset / 2, 0.5), ncol=numcols,
                                              fontsize=legsize)

                                else:
                                    # right_sided_axes.append(ax)
                                    ax.legend(handles=handles, labels=labels, loc='center left',
                                              bbox_to_anchor=(1, 0.5), ncol=numcols, fontsize=legsize)

                            # If legend_outside is set to below, put the legend below the plot
                            elif ax_settings['legend_outside'].lower() == 'below':

                                # If there is an xlabel, the legend needs to be farther down to account for it
                                if 'xlabel' in ax_settings.keys():
                                    legened_y_offset = -0.15
                                else:
                                    legened_y_offset = -0.05

                                # the below legend does not change if we have a twin axis
                                ax.legend(handles=handles, labels=labels, loc='upper center',
                                              bbox_to_anchor=(0.5, legened_y_offset), ncol=numcols,
                                              fontsize=legsize)

                            else:
                                ax.legend(handles=handles, labels=labels, fontsize=legsize, ncol=numcols)


            if not any(useAx):
                WF.print2stdout(f'Plot for {year} not included due to xlimits.')

                print(f'Plot for {year} not included due to xlimits.')
                plt.close("all")
                continue

            plt.gcf().canvas.draw()  #refresh so we can get legend stuff

            plt.tight_layout()

            if 'spacebetweenaxis' in object_settings.keys():
                if object_settings['spacebetweenaxis'].lower() != 'true':
                    plt.subplots_adjust(wspace=0, hspace=0)
            else:
                plt.subplots_adjust(wspace=0, hspace=0)

            # if self.memberiteration:
            #     if fullyduplicate:
            #         WF.print2stdout(f'Skipping duplicate plot for {year} {self.member}')
            #         plt.close('all')
            #         return

            basefigname = os.path.join(self.images_path, 'TimeSeriesPlot' + '_' + self.ChapterRegion.replace(' ', '_')
                                       + '_' + yearstr)
            exists = True
            tempnum = 1
            tfn = basefigname
            while exists:
                if os.path.exists(tfn + '.png'):
                    tfn = basefigname + '_{0}'.format(tempnum)
                    tempnum += 1
                else:
                    exists = False
            figname = tfn + '.png'

            if self.highres:
                plt.savefig(figname, dpi=300)
            else:
                plt.savefig(figname)
            plt.close('all')


            if pageformat == 'half':
                self.XML.writeHalfPagePlot(os.path.basename(figname), cur_obj_settings['description'])
            if pageformat == 'full':
                self.XML.writeFullPagePlot(os.path.basename(figname), cur_obj_settings['description'])


        WF.print2stdout(f'Timeseries Plot took {time.time() - objectstarttime} seconds.')

    def set_identicals_member_key(self, plot_name, member):
        """
           Update `self.identical_members_key` with the formatted members for the given member in the specified plot.
           :param plot_name: The name of the plot to use for lookup
           :param member: The member to get the formatted string for
           """
        if plot_name in self.plot_identical_members_key:
            member_dict = self.plot_identical_members_key[plot_name]
            if member in member_dict:
                self.identical_members_key = member_dict[member]
            else:
                self.identical_members_key = f'Member {member} not found in plot {plot_name}'
        else:
            self.identical_members_key = f'Plot name {plot_name} not found in plot_identical_members_key'


    def check_for_identical_timeSeriesPlots(self, values_dict):
        """
        Compares members' data within each plot to identify identical and non-identical groups.

        :param values_dict: Dictionary of time series data with plot names as keys.
        :return: Updates internal variables with results for each plot.
        """

        plots_processed = 0

        for plot_name, plot_data in values_dict.items():
            member_keys = list(next(iter(plot_data.values())).keys())
            self.identical_groups_per_plot, self.non_identicals_per_plot = [], []
            self.checked_members = set() #reset for each plot

            # Compare each member against others
            for reference_member in member_keys:
                if reference_member in self.checked_members:
                    continue

                identical_to_current = [reference_member]

                for other_member in member_keys:
                    if other_member == reference_member or other_member in self.checked_members:
                        continue

                    if self.are_members_identical(plot_data, reference_member, other_member):
                        identical_to_current.append(other_member)

                # Update identical and non-identical groups
                if len(identical_to_current) > 1:
                    self.identical_groups_per_plot.append(sorted(identical_to_current))
                    self.checked_members.update(identical_to_current)
                else:
                    self.non_identicals_per_plot.append(reference_member)
                    self.checked_members.add(reference_member)

            # Handle members not yet checked
            remaining_members = set(member_keys) - self.checked_members
            self.non_identicals_per_plot.extend(remaining_members)

            self.process_plot_results(plot_name)
            plots_processed += 1

        # Final check to confirm all plots have been checked
        self.cur_section_members_all_checked = (plots_processed == len(values_dict))

        # Final check after processing all plots
        WF.print2stdout(f'#### After checking for identicals:  \nidentical_members_per_plot: \n{self.identical_members_per_plot}', debug=self.debug)
        WF.print2stdout(f'non_identical_members_per_plot: \n{self.non_identical_members_per_plot} ####', debug=self.debug)


    def are_members_identical(self, plot_data, reference_member, other_member):
        """
        Compares two members' data across all time series in the plot.

        :param plot_data: Dictionary of time series data for the plot.
        :param reference_member: The member to compare against.
        :param other_member: The member to compare with the reference.
        :return: True if members are identical, False otherwise.
        """
        for member_data in plot_data.values():
            if not np.array_equal(member_data[reference_member], member_data[other_member]):
                return False
        return True


    def process_plot_results(self, plot_name):
        """
        Stores results for identical and non-identical members for each plot.

        :param plot_name: The name of the plot being processed.
        """
        if self.identical_groups_per_plot:
            self.identical_members_per_plot[plot_name] = self.identical_groups_per_plot
            self.create_identical_members_keys(self.identical_groups_per_plot, plot_name)
            self.update_identical_members_list(self.identical_groups_per_plot)

        if self.non_identicals_per_plot:
            self.non_identical_members_per_plot[plot_name] = self.non_identicals_per_plot


    def update_identical_members_list(self, identical_groups):
        """
        Marks identical members and skips plotting for non-primary members.

        :param identical_groups: List of groups of identical members.
        """
        for group in identical_groups:
            if group not in self.complete_identical_members_groups:  # Avoid duplicates
                self.complete_identical_members_groups.append(group)
                for member in group:
                    if member != group[0]:
                        self.identical_members_do_not_plot.append(member) # Skip non-primary members


    def create_identical_members_keys(self, identical_groups, plot_name):
        """
        Creates and stores keys for identical members within each plot.

        :param identical_groups: Groups of identical members.
        :param plot_name: The plot being processed.
         plot_identical_members_key dict ex: {plot_name: {member1: 'member1, member2'}}
        """
        if plot_name not in self.plot_identical_members_key:
            self.plot_identical_members_key[plot_name] = {}

        # Format each group and update the dictionary
        for group in identical_groups:
            first_member = group[0]
            formatted_members = ', '.join(WF.formatMembers(group))
            if first_member in self.plot_identical_members_key[plot_name]:
                self.plot_identical_members_key[plot_name][first_member] += f', {formatted_members}'
            else:
                self.plot_identical_members_key[plot_name][first_member] = formatted_members


    def makeProfileStatisticsTable(self, object_settings):
        """
        Makes a table to compute stats based off of profile lines. Data is interpolated over a series of points
        determined by the user
        :param object_settings: currently selected object settings dictionary
        :return: writes table to XML
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Profile Stats Table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)
        self.Profiles = WProfile.Profiles(self)

        default_settings = self.loadDefaultPlotObject('profilestatisticstable')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', forjasper=True)

        object_settings['datakey'] = 'datapaths'

        ################# Get timestamps #################
        object_settings['datessource_flag'] = WF.getDateSourceFlag(object_settings)
        object_settings['timestamps'] = self.Profiles.getProfileTimestamps(object_settings, self.StartTime, self.EndTime)

        ################# Get units #################
        object_settings['plot_parameter'] = self.getPlotParameter(object_settings)

        ################# Get data #################
        data, line_settings, missing = self.Data.getProfileDataDictionary(object_settings)
        object_settings['warnings'] = self.Profiles.checkProfileValidity(data, object_settings)

        line_settings = WF.correctDuplicateLabels(line_settings)
        table_blueprint = pickle.loads(pickle.dumps(object_settings, -1))

        object_settings = self.configureSettingsForID(self.base_id, object_settings)

        ################# Get plot units #################
        data, line_settings = self.Profiles.convertProfileDataUnits(object_settings, data, line_settings)
        object_settings['units_list'] = WF.getUnitsList(line_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)
        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        table_blueprint = WF.updateFlaggedValues(table_blueprint, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        # self.Data.commitProfileDataToMemory(data, line_settings, object_settings)

        # object_settings['usedepth'] = self.Profiles.confirmValidDepths(data)
        # if object_settings['usedepth']:
        #     data = self.Profiles.snapTo0Depth(data, line_settings)

        if 'usedepth' not in object_settings.keys():
            object_settings['usedepth'] = 'true'

        if object_settings['usedepth'].lower() == 'false':
            wse_data = self.Data.getProfileWSE(object_settings, onflag='datapaths')
            data = self.Profiles.convertDepthsToElevations(data, wse_data)
        elif object_settings['usedepth'].lower() == 'true':
            wse_data = self.Data.getProfileWSE(object_settings, onflag='datapaths')
            data = self.Profiles.convertElevationsToDepths(data, wse_data=wse_data)

        data, object_settings = self.Profiles.filterProfileData(data, line_settings, object_settings)

        object_settings['resolution'] = self.Profiles.getProfileInterpResolution(object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings, allowIncludeAllYears=False)

        yrheaders, yrheaders_i = self.Tables.buildHeadersByTimestamps(object_settings['timestamps'], self.years)
        yrheaders = self.Tables.convertHeaderFormats(yrheaders, object_settings)

        if 'description' not in object_settings.keys():
            object_settings['description'] = ''
        # else:
        #     object_settings['description'] = WF.parseForTextFlags(object_settings['description'])

        if not object_settings['split_by_year']:  #if we don't want to split by year, just make a huge list
            yrheaders = [list(itertools.chain.from_iterable(yrheaders))]
            yrheaders_i = [list(itertools.chain.from_iterable(yrheaders_i))]
        for yi, yrheader_group in enumerate(yrheaders):
            year = object_settings['years'][yi]
            yearstr = object_settings['yearstr'][yi]
            table_constructor = {}

            if len(yrheader_group) == 0:
                WF.print2stdout('No data for', yearstr, debug=self.debug)
                continue

            object_desc = WF.updateFlaggedValues(object_settings['description'], '%%year%%', yearstr)

            for yhi, yrheader in enumerate(yrheader_group):
                header_i = yrheaders_i[yi][yhi]
                headings, rows = self.Tables.buildProfileStatsTable(table_blueprint, yrheader, line_settings)
                for hi, heading in enumerate(headings):
                    tcnum = len(table_constructor.keys())
                    table_constructor[tcnum] = {}
                    if self.iscomp:
                        table_constructor[tcnum]['datecolumn'] = yrheader
                    frmt_rows = []
                    threshold_colors = np.full(len(rows), None)
                    for ri, row in enumerate(rows):
                        s_row = row.split('|')
                        rowname = s_row[0]
                        row_val = s_row[hi + 1]
                        addasterisk = False
                        if '%%' in row_val:
                            stats_data = self.Tables.formatStatsProfileLineData(row_val, data,
                                                                                object_settings['resolution'],
                                                                                object_settings['usedepth'], header_i)
                            missing_data = self.Tables.checkForMissingData(row_val, stats_data)
                            if missing_data:
                                stat = self.Tables.getStat(row_val)
                                row_val = np.nan
                            else:
                                row_val, stat = self.Tables.getStatsLine(row_val, stats_data)
                            if not np.isnan(row_val) and row_val is not None:
                                thresholdsettings = self.Tables.matchThresholdToStat(stat, object_settings)

                                for thresh in thresholdsettings:
                                    if row_val < thresh['value']:
                                        if 'colorwhen' in thresh.keys():
                                            if thresh['colorwhen'] == 'under':
                                                threshold_colors[ri] = thresh['color']
                                                if 'addasterisk' in object_settings.keys():
                                                    if object_settings['addasterisk'].lower() == 'true':
                                                        addasterisk = True
                                            if thresh['when'] == 'under':
                                                if 'replacement' in thresh.keys():
                                                    row_val = thresh['replacement']
                                    elif row_val > thresh['value']:
                                        if thresh['colorwhen'] == 'over':
                                            threshold_colors[ri] = thresh['color']
                                            if 'addasterisk' in object_settings.keys():
                                                if object_settings['addasterisk'].lower() == 'true':
                                                    addasterisk = True
                                        if thresh['when'] == 'over':
                                            if 'replacement' in thresh.keys():
                                                row_val = thresh['replacement']

                            else:
                                if np.isnan(row_val):
                                    if 'missingmarker' in object_settings.keys():
                                        row_val = object_settings['missingmarker']

                            self.WAT_log.addLogEntry({'type': 'ProfileTableStatistic',
                                                      'name': ' '.join([self.ChapterRegion, heading, stat]),
                                                      'description': object_desc,
                                                      'value': row_val,
                                                      'function': stat,
                                                      'units': object_settings['plot_units'],
                                                      'value_start_date': yrheader,
                                                      'value_end_date': yrheader,
                                                      'logoutputfilename': ', '.join(
                                                          [line_settings[flag]['logoutputfilename'] for flag in
                                                           line_settings])
                                                      },
                                                     isdata=True)
                        numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings)
                        formattedNumber = WF.formatNumbers(row_val, numberFormat)
                        if addasterisk:
                            formattedNumber += '*'
                        frmt_rows.append('{0}|{1}'.format(rowname, formattedNumber))
                    table_constructor[tcnum]['rows'] = frmt_rows
                    table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                    table_constructor[tcnum]['header'] = heading

            keeptable = False
            keepall = True
            keepcolumn = {}
            missing_value_objects = ['nan', '-', 'none', '--']
            if 'missingmarker' in object_settings.keys():
                missing_value_objects.append(object_settings['missingmarker'])
            for row_num in table_constructor.keys():
                constructor = table_constructor[row_num]
                rows = constructor['rows']
                header = constructor['header']
                for row in rows:
                    srow = row.split('|')
                    if header not in keepcolumn.keys():
                        keepcolumn[header] = False
                        if srow[1].lower() not in missing_value_objects:
                            keepcolumn[header] = True

            for key in keepcolumn.keys():
                if keepcolumn[key] is True:
                    keeptable = True
                else:
                    keepall = False

            if keeptable:  #quick check if we're even writing a table.
                if not keepall:
                    new_table_constructor = {}
                    for row_num in table_constructor.keys():
                        constructor = table_constructor[row_num]
                        header = constructor['header']
                        if keepcolumn[header]:
                            new_table_constructor[row_num] = constructor
                else:
                    new_table_constructor = table_constructor

                # THEN write table
                if self.iscomp:
                    self.XML.writeDateControlledTableStart(object_desc, 'Statistics')
                else:
                    self.XML.writeTableStart(object_desc, 'Statistics')

                self.Tables.writeTable(new_table_constructor)
                if not keepall:
                    self.Tables.writeMissingTableItemsWarning(object_desc)
            else:
                self.Tables.writeMissingTableWarning(object_desc)
                WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)

            ignorewarnings = False
            if 'ignorewarnings' in object_settings.keys():
                if object_settings['ignorewarnings'] == 'true'.lower():
                    ignorewarnings = True
            if not ignorewarnings:
                # self.Profiles.writeWarnings(object_settings['warnings'], year)
                WF.print2stdout('Not currently writing out warnings to report', debug=self.debug)

        WF.print2stdout(f'Profile Stat Table took {time.time() - objectstarttime} seconds.')

    def makeProfilePlot(self, object_settings):
        """
        Takes in object settings to build profile plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Profile Plot.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Plots = WPlot.Plots(self)
        self.Profiles = WProfile.Profiles(self)

        default_settings = self.loadDefaultPlotObject('profileplot')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['datakey'] = 'lines'

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', exclude=['description'])
        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', include=['description'], forjasper=True)

        # object_settings['description'] = WF.parseForTextFlags(object_settings['description'])
        obj_desc = WF.updateFlaggedValues(object_settings['description'], '%%year%%', self.years_str)

        ################# Get timestamps #################
        object_settings['datessource_flag'] = WF.getDateSourceFlag(object_settings)
        object_settings['timestamps'] = self.Profiles.getProfileTimestamps(object_settings, self.StartTime, self.EndTime)

        ################# Get units #################
        object_settings['plot_parameter'] = self.getPlotParameter(object_settings)

        ################# Get data #################
        data, line_settings, missing = self.Data.getProfileDataDictionary(object_settings)

        straightlines = self.Data.getStraightLineValue(object_settings)

        line_settings = WF.correctDuplicateLabels(line_settings)

        object_settings = self.configureSettingsForID(self.base_id, object_settings)
        gatedata, gate_settings = self.Data.getGateDataDictionary(object_settings, makecopy=False)

        ################ convert yflags ################
        if object_settings['usedepth'].lower() == 'false':
            wse_data = self.Data.getProfileWSE(object_settings)
            data = self.Profiles.convertDepthsToElevations(data, wse_data)
        elif object_settings['usedepth'].lower() == 'true':
            wse_data = self.Data.getProfileWSE(object_settings)
            data = self.Profiles.convertElevationsToDepths(data, wse_data=wse_data)

        ################# Get plot units #################
        data, line_settings = self.Profiles.convertProfileDataUnits(object_settings, data, line_settings)
        object_settings['units_list'] = WF.getUnitsList(line_settings)
        object_settings['y_units_list'] = WF.getUnitsList(line_settings, mod='y_')
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)
        object_settings['y_plot_units'] = WF.getPlotUnits(object_settings['y_units_list'], object_settings, axis='y')
        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units']))
        object_settings = WF.updateFlaggedValues(object_settings, '%%y_units%%', WF.formatUnitsStrings(object_settings['y_plot_units']))

        # ################ convert yflags ################
        # if object_settings['usedepth'].lower() == 'false':
        #     wse_data = self.Data.getProfileWSE(object_settings)
        #     data = self.Profiles.convertDepthsToElevations(data, wse_data)
        # elif object_settings['usedepth'].lower() == 'true':
        #     wse_data = self.Data.getProfileWSE(object_settings)
        #     data = self.Profiles.convertElevationsToDepths(data, wse_data=wse_data)

        # self.Data.commitProfileDataToMemory(data, line_settings, object_settings)
        linedata, object_settings = self.Profiles.filterProfileData(data, line_settings, object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings, allowIncludeAllYears=False)

        object_settings['warnings'] = self.Profiles.checkProfileValidity(data, object_settings)

        ################ Build Plots ################
        for yi, year in enumerate(object_settings['years']):
            self.XML.writeProfilePlotStart(obj_desc)
            yearstr = object_settings['yearstr'][yi]

            t_stmps = WT.filterTimestepByYear(object_settings['timestamps'], year)

            prof_indices = [np.where(object_settings['timestamps'] == n)[0][0] for n in t_stmps]
            n = int(object_settings['profilesperrow']) * int(object_settings['rowsperpage']) #Get number of plots on page
            page_indices = [prof_indices[i * n:(i + 1) * n] for i in range((len(prof_indices) + n - 1) // n)]
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)  #TODO: reudce the settings

            for page_i, pgi in enumerate(page_indices):

                subplot_rows, subplot_cols = WF.getSubplotConfig(len(pgi), int(cur_obj_settings['profilesperrow']))
                n_nrow_active = np.ceil(len(pgi) / subplot_cols)

                fig, axs = plt.subplots(nrows=int(object_settings['rowsperpage']),
                                        ncols=int(object_settings['profilesperrow']), figsize=(9, 10))

                for i in range(n):

                    current_row = i // int(object_settings['profilesperrow'])
                    current_col = i % int(object_settings['profilesperrow'])

                    ax = axs[current_row, current_col]
                    if i + 1 > len(pgi):
                        ax.axis('off')
                        continue
                    else:
                        j = pgi[i]

                    if object_settings['usedepth'].lower() == 'true':
                        ax.invert_yaxis()

                    for li, line in enumerate(data.keys()):
                        try:
                            values = data[line]['values'][j]
                            if len(values) == 0:
                                WF.print2stdout(
                                    'No values for {0} on {1}'.format(line, object_settings['timestamps'][j]),
                                    debug=self.debug)
                                continue
                            msk = np.where(~np.isnan(values))
                            values = values[msk]
                        except IndexError:
                            WF.print2stdout('No values for {0} on {1}'.format(line, object_settings['timestamps'][j]),
                                            debug=self.debug)
                            continue

                        try:
                            if object_settings['usedepth'].lower() == 'true':
                                levels = data[line]['depths'][j][msk]
                            else:
                                levels = data[line]['elevations'][j][msk]
                            if not WF.checkData(levels):
                                WF.print2stdout('Non Viable depths/elevations for {0} on {1}'.format(line, object_settings['timestamps'][j]), debug=self.debug)
                                continue
                        except IndexError:
                            WF.print2stdout('Non Viable depths/elevations for {0} on {1}'.format(line, object_settings['timestamps'][j]), debug=self.debug)
                            continue

                        if not WF.checkData(values):
                            continue

                        current_ls = line_settings[line] #we have all the settings we need..
                        current_ls = WD.getDefaultLineSettings(self.defaultLineStyles, current_ls, object_settings['plot_parameter'], li, debug=self.debug)
                        current_ls = WF.fixDuplicateColors(current_ls) #used the line, used param, then double up so subtract 1

                        if current_ls['drawline'].lower() == 'true' and current_ls['drawpoints'].lower() == 'true':
                            self.Plots.plotLinesAndPoints(values, levels, ax, current_ls)

                        elif current_ls['drawline'].lower() == 'true':
                            self.Plots.plotLines(values, levels, ax, current_ls)

                        elif current_ls['drawpoints'].lower() == 'true':
                            self.Plots.plotPoints(values, levels, ax, current_ls)

                    ### HLINES ###
                    self.Plots.plotHorizontalLines(straightlines, ax, cur_obj_settings, timestamp_index=j)

                    ### VERTICAL LINES ###
                    self.Plots.plotVerticalLines(straightlines, ax, cur_obj_settings, timestamp_index=j, isdate=False)

                    show_xlabel, show_ylabel = self.getPlotLabelMasks(i, len(pgi), subplot_cols)

                    if show_ylabel:
                        if 'ylabel' in cur_obj_settings.keys():
                            if 'ylabelsize' in object_settings.keys():
                                ylabsize = float(object_settings['ylabelsize'])
                            elif 'fontsize' in object_settings.keys():
                                ylabsize = float(object_settings['fontsize'])
                            else:
                                ylabsize = 12
                            ylabel = WF.formatTextFlags(cur_obj_settings['ylabel'])
                            ax.set_ylabel(ylabel, fontsize=ylabsize)

                    if show_xlabel:
                        if 'xlabel' in cur_obj_settings.keys():
                            if 'xlabelsize' in object_settings.keys():
                                xlabsize = float(object_settings['xlabelsize'])
                            elif 'fontsize' in object_settings.keys():
                                xlabsize = float(object_settings['fontsize'])
                            else:
                                xlabsize = 12
                            if 'bottomtext' in cur_obj_settings:
                                labelpad = 12
                            else:
                                labelpad = 0
                            xlabel = WF.formatTextFlags(cur_obj_settings['xlabel'])
                            ax.set_xlabel(xlabel, fontsize=xlabsize, labelpad=labelpad)

                    show_xticks = True
                    if 'xlims' in object_settings.keys() and not show_xlabel:
                        if all([x in object_settings['xlims'].keys() for x in ['min', 'max']]):
                            show_xticks = False

                    self.Plots.formatXTicks(ax, object_settings)

                    if not show_xticks:
                        ax.set_xticklabels([])
                        ax.tick_params(axis='x', which='both', bottom=False)

                    show_yticks = True
                    if 'ylims' in object_settings.keys() and not show_ylabel:
                        if all([x in object_settings['ylims'].keys() for x in ['min', 'max']]):
                            show_yticks = False

                    self.Plots.formatYTicks(ax, object_settings)

                    if not show_yticks:
                        ax.set_yticklabels([])
                        ax.tick_params(axis='y', which='both', left=False)

                    if cur_obj_settings['gridlines'].lower() == 'true':
                        ax.grid(zorder=-9)

                    ### GATES ###
                    gateconfig = {}
                    if len(gatedata.keys()) > 0:
                        gatemsk = None
                        for ggi, gategroup in enumerate(gatedata.keys()):
                            gatetop = None
                            gatebottom = None
                            gatemiddle = None
                            gateop_has_value = False
                            gate_count = 0  #keep track of gate number in group
                            numgates = len(gatedata[gategroup]['gates'])
                            gatepoint_xpositions = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1],numgates+2)[1:-1] #+2 for start and end

                            cur_gateop = gatedata[gategroup]
                            cur_gateop_settings = gate_settings[gategroup]

                            if 'top' in cur_gateop_settings.keys():
                                gatetop = float(cur_gateop_settings['top'])

                            if 'bottom' in cur_gateop_settings.keys():
                                gatebottom = float(cur_gateop_settings['bottom'])

                            if 'middle' in cur_gateop_settings.keys():
                                gatemiddle = float(cur_gateop_settings['middle'])
                            elif gatetop is not None and gatebottom is not None:
                                gatemiddle = np.mean([gatetop, gatebottom])

                            if gatetop is None and gatebottom is not None and gatemiddle is None:  #bottom no top/middle
                                gatetop = gatebottom + float(object_settings['defaultgateenvelope'])
                                gatemiddle = gatebottom + float(object_settings['defaultgateenvelope']) / 2
                            elif gatetop is not None and gatebottom is None and gatemiddle is None:  #top no bottom/middle
                                gatebottom = gatetop - float(object_settings['defaultgateenvelope'])
                                gatemiddle = gatetop - float(object_settings['defaultgateenvelope']) / 2
                            elif gatetop is None and gatebottom is None and gatemiddle is not None:  #only middle
                                gatebottom = gatemiddle - float(object_settings['defaultgateenvelope']) / 2
                                gatetop = gatemiddle + float(object_settings['defaultgateenvelope']) / 2

                            for gate in cur_gateop['gates'].keys():

                                curgate = cur_gateop['gates'][gate]
                                curgate_settings = cur_gateop_settings['gates'][gate]

                                values = curgate['values']
                                dates = curgate['dates']

                                if len(values) == 0:
                                    continue

                                if 'dateformat' in cur_obj_settings.keys():
                                    if cur_obj_settings['dateformat'].lower() == 'datetime':
                                        if isinstance(dates[0], (int, float)):
                                            dates = WT.JDateToDatetime(dates, self.startYear)

                                if gatemsk is None:
                                    gatemsk = WR.getClosestTime([object_settings['timestamps'][j]], dates)
                                if len(gatemsk) == 0:
                                    value = np.nan
                                else:
                                    value = values[gatemsk[0]]

                                xpos = gatepoint_xpositions[gate_count]
                                if gategroup not in gateconfig.keys():
                                    gateconfig[gategroup] = {gate: value}
                                else:
                                    gateconfig[gategroup][gate] = value

                                if not np.isnan(value):
                                    gateop_has_value = True
                                    showgate = False
                                    if 'showgates' in cur_gateop.keys():
                                        if cur_gateop['showgates'].lower() == 'true':
                                            showgate = True
                                    elif 'showgates' in cur_obj_settings.keys():
                                        if cur_obj_settings['showgates'].lower() == 'true':
                                            showgate = True

                                    if showgate:
                                        ax.scatter(xpos, gatemiddle, edgecolor='black', facecolor='black', marker='o')

                                gate_count += 1  #keep track of gate number in group
                                self.WAT_log.addLogEntry(
                                    {'type': gate + '_GateTimeSeries' if gate != '' else 'GateTimeseries',
                                     'name': self.ChapterRegion + '_' + yearstr,
                                     'description': cur_obj_settings['description'],
                                     'units': 'BINARY',
                                     'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                self.StartTime, self.EndTime,
                                                                                debug=self.debug).strftime('%d %b %Y'),
                                     'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                              self.StartTime, self.EndTime,
                                                                              debug=self.debug).strftime('%d %b %Y'),
                                     'logoutputfilename': curgate_settings['logoutputfilename']
                                     },
                                    isdata=True)

                            if 'color' in cur_gateop_settings.keys():
                                color = cur_gateop_settings['color']
                                default_color = self.Constants.def_colors[ggi]
                                color = WF.confirmColor(color, default_color, debug=self.debug)
                            if 'top' in cur_gateop_settings.keys():
                                ax.axhline(gatetop, color=color, zorder=-7)
                            if 'bottom' in cur_gateop_settings.keys():
                                ax.axhline(gatebottom, color=color, zorder=-7)
                            if 'middle' in cur_gateop_settings.keys():
                                ax.axhline(gatemiddle, color=color, zorder=-7)

                            if gateop_has_value:
                                ax.axhspan(gatebottom, gatetop, alpha=0.5, color=color, zorder=-8)

                    cur_timestamp = object_settings['timestamps'][j]
                    if 'dateformat' in object_settings:
                        if object_settings['dateformat'].lower() == 'datetime':
                            cur_timestamp = WT.translateDateFormat(cur_timestamp, 'datetime', '',
                                                                   self.StartTime, self.EndTime,
                                                                   debug=self.debug)
                            ttl_str = cur_timestamp.strftime('%d %b %Y')
                        elif object_settings['dateformat'].lower() == 'jdate':
                            cur_timestamp = WT.translateDateFormat(cur_timestamp, 'jdate', '',
                                                                   self.StartTime, self.EndTime,
                                                                   debug=self.debug)
                            ttl_str = str(cur_timestamp)
                        else:
                            ttl_str = cur_timestamp
                    else:
                        ttl_str = object_settings['timestamps'][j].strftime('%d %b %Y') #should get set to datetime anyways..

                    if 'datetext' in cur_obj_settings.keys():
                        if cur_obj_settings['datetext'].lower() == 'true':
                            xbufr = 0.05
                            ybufr = 0.05
                            xl = ax.get_xlim()
                            yl = ax.get_ylim()
                            xtext = xl[0] + xbufr * (xl[1] - xl[0])
                            ytext = yl[1] - ybufr * (yl[1] - yl[0])
                            ax.text(xtext, ytext, ttl_str, ha='left', va='top', size=10,  #TODO: make this variable
                                    bbox=dict(boxstyle='round', facecolor='w', alpha=0.35),
                                    zorder=10)

                    if 'bottomtext' in cur_obj_settings.keys():
                        bottomtext_str = []
                        for text in cur_obj_settings['bottomtext']:
                            if text.lower() == 'date':
                                bottomtext_str.append(object_settings['timestamps'][j].strftime('%m/%d/%Y'))
                            elif text.lower() == 'gateconfiguration':
                                gateconfignum = WGates.getGateConfigurationDays(gateconfig, gatedata, object_settings['timestamps'][j])
                                if isinstance(gateconfignum, float):
                                    bottomtext_str.append(str('{num:,.{digits}f}'.format(num=gateconfignum, digits=3)))
                                else:
                                    bottomtext_str.append(gateconfignum)
                            elif text.lower() == 'gateblend':
                                gateblendnum = WGates.getGateBlendDays(gateconfig, gatedata, object_settings['timestamps'][j])
                                if isinstance(gateblendnum, float):
                                    bottomtext_str.append(str('{num:,.{digits}f}'.format(num=gateblendnum, digits=3)))
                                else:
                                    bottomtext_str.append(gateblendnum)
                            else:
                                bottomtext_str.append(text)
                        bottomtext = ', '.join(bottomtext_str)

                        if show_xticks:
                            bottomtext_y = -25
                        else:
                            bottomtext_y = -10
                        if 'bottomtextfontsize' in cur_obj_settings.keys():
                            bottomtextfontsize = float(cur_obj_settings['bottomtextfontsize'])
                        else:
                            bottomtextfontsize = 6
                        if 'bottomtextfontcolor' in cur_obj_settings.keys():
                            bottomtextfontcolor = WF.confirmColor(cur_obj_settings['bottomtextfontcolor'], 'red', debug= self.debug)
                        else:
                            bottomtextfontcolor = 'red'
                        ax.annotate(bottomtext, xy=(0.02, bottomtext_y), fontsize=bottomtextfontsize, color=bottomtextfontcolor, xycoords='axes points')

                plt.tight_layout()

                if 'legend' in cur_obj_settings.keys():
                    if cur_obj_settings['legend'].lower() == 'true':
                        leg_labels = []
                        leg_handles = []
                        for ax in axs:
                            for subax in ax:
                                leg_handle_labels = subax.get_legend_handles_labels()
                                for li in range(len(leg_handle_labels[0])):
                                    if leg_handle_labels[1][li] not in leg_labels:
                                        leg_labels.append(leg_handle_labels[1][li])
                                        leg_handles.append(leg_handle_labels[0][li])

                        if len(leg_labels) > 0:
                            if 'legendsize' in cur_obj_settings.keys():
                                legsize = float(cur_obj_settings['legendsize'])
                            elif 'fontsize' in cur_obj_settings.keys():
                                legsize = float(cur_obj_settings['fontsize'])
                            else:
                                legsize = 12

                            ncolumns = 3

                            # n_legends_row = np.ceil(len(linedata.keys()) / ncolumns) * .65
                            n_legends_row = np.ceil(len(leg_handles) / ncolumns) * .65
                            if n_legends_row < 1:
                                n_legends_row = 1

                            plt.subplots_adjust(bottom=.1 * n_legends_row)
                            # fig_ratio = (axs[int(n_nrow_active)-1,0].bbox.extents[1] - (fig.bbox.height * (.1025 * n_legends_row))) / fig.bbox.height
                            fig_ratio = (axs[int(n_nrow_active)-1,0].bbox.extents[1] - (fig.bbox.height * .055)) / fig.bbox.height
                            # fig_ratio += text_y * linestext
                            # plt.legend(bbox_to_anchor=(.5,fig_ratio), loc="lower center", fontsize=legsize,
                            plt.legend(bbox_to_anchor=(.5, fig_ratio), loc="upper center", fontsize=legsize,
                                       bbox_transform=fig.transFigure, ncol=ncolumns, handles=leg_handles,
                                       labels=leg_labels)


                basefigname = 'ProfilePlot_{0}_{1}_{2}_{3}_{4}'.format(self.ChapterName, yearstr,
                                                                       object_settings['plot_parameter'], self.program,
                                                                       page_i)

                exists = True
                tempnum = 1
                tfn = basefigname
                while exists:
                    if os.path.exists(os.path.join(self.images_path, tfn + '.png')):
                        tfn = basefigname + '_{0}'.format(tempnum)
                        tempnum += 1
                    else:
                        exists = False

                figname = tfn + '.png'

                if self.highres:
                    plt.savefig(os.path.join(self.images_path, figname), dpi=300)
                else:
                    plt.savefig(os.path.join(self.images_path, figname))

                plt.close('all')

                ################################################

                description = '{0}: {1} of {2}'.format(cur_obj_settings['description'], page_i + 1, len(page_indices))
                self.XML.writeProfilePlotFigure(figname, description)

                self.WAT_log.addLogEntry({'type': 'ProfilePlot',
                                          'name': self.ChapterRegion,
                                          'description': description,
                                          'units': object_settings['plot_units'],
                                          'value_start_date': WT.translateDateFormat(
                                              object_settings['timestamps'][pgi[0]],
                                              'datetime', '',
                                              self.StartTime, self.EndTime,
                                              debug=self.debug).strftime('%d %b %Y'),
                                          'value_end_date': WT.translateDateFormat(
                                              object_settings['timestamps'][pgi[-1]],
                                              'datetime', '',
                                              self.StartTime, self.EndTime,
                                              debug=self.debug).strftime('%d %b %Y'),
                                          'logoutputfilename': ', '.join(
                                              [line_settings[flag]['logoutputfilename'] for flag in line_settings])
                                          },
                                         isdata=True)

            self.XML.writeProfilePlotEnd()

            ignorewarnings = False
            if 'ignorewarnings' in object_settings.keys():
                if object_settings['ignorewarnings'] == 'true'.lower():
                    ignorewarnings = True
            if not ignorewarnings:
                # self.Profiles.writeWarnings(object_settings['warnings'], year)
                WF.print2stdout('Not currently writing out warnings to report', debug=self.debug)

        WF.print2stdout(f'Profile Plot took {time.time() - objectstarttime} seconds.')

    def makeErrorStatisticsTable(self, object_settings):
        """
        Takes in object settings to build error stats table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: writes to XML file
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Error Stats table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)

        default_settings = self.loadDefaultPlotObject('errorstatisticstable')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', exclude=['description', 'thresholds'])
        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', include=['description', 'thresholds'],
                                                  forjasper=True)

        # update the title and descriptions with anything from general or model specific
        object_settings = WF.replaceflaggedValues(self, object_settings, 'general', include=['description'])
        object_settings = WF.replaceflaggedValues(self, object_settings, 'modelspecific', include=['description'])

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)
        object_settings['allyearsstr'] = WF.getObjectAllYears(object_settings['years'])

        data, data_settings = self.Data.getTableDataDictionary(object_settings)

        fullyduplicate = False
        if self.memberiteration:
            if self.groupmembers:
                if self.Data.checkForDuplicateObject(data_settings, self.member):  # for member iteration plots, skip duplicates. If the first instance of a duplicate plot, plot it but change hte header + description
                    is_lowest = self.Data.checkForLowestMember(data_settings, self.member)
                    if is_lowest:
                        fullyduplicate = False  # if the first instance of a duplicate plot, plot it but change the header description
                        other_members = self.Data.getOtherMembers(data_settings, self.member)
                        member_list_frmt = (f"{WF.getOriginalMemberNumber(self.member, WF.matchMemberToEnsembleSet(self.ensembleSets, self.member), self.DSSFile, self.alternativeFpart, self.StartTime, self.EndTime, self.debug)}, " +
                                            ', '.join([WF.getOriginalMemberNumber(mem, WF.matchMemberToEnsembleSet(self.ensembleSets, mem), self.DSSFile, self.alternativeFpart, self.StartTime, self.EndTime, self.debug) for mem in other_members]))

                        object_settings['description'] = WF.updateFlaggedValues(object_settings['description'], '%%member%%', member_list_frmt)  # updates timeseriesplots png descriptions

                        # update met name in description
                        object_settings['description'] = WF.updateFlaggedValues(object_settings['description'], '%%metname%%',
                                                                                WF.matchMemberToEnsembleSet(self.ensembleSets, self.member)['metname'])
                    else:
                        fullyduplicate = True
                        WF.print2stdout('Duplicate object found. Skipping.', debug=self.debug)
                        return
                else:
                    fullyduplicate = False

                    # get the ensemble set for the current member
                    curr_ensemble_set = WF.matchMemberToEnsembleSet(self.ensembleSets, self.member)

                    # update the member number
                    object_settings['description'] = WF.updateFlaggedValues(object_settings['description'], '%%member%%', WF.getOriginalMemberNumber(self.member, curr_ensemble_set, self.DSSFile,
                                                                                                                                                     self.alternativeFpart, self.StartTime, self.EndTime, self.debug))

                    # update met name in description
                    object_settings['description'] = WF.updateFlaggedValues(object_settings['description'], '%%metname%%', curr_ensemble_set['metname'])
            else:
                # get the ensemble set for the current member
                curr_ensemble_set = WF.matchMemberToEnsembleSet(self.ensembleSets, self.member)

                # update the member number
                object_settings['description'] = WF.updateFlaggedValues(object_settings['description'], '%%member%%', WF.getOriginalMemberNumber(self.member, curr_ensemble_set, self.DSSFile,
                                                                                                                                                 self.alternativeFpart, self.StartTime, self.EndTime, self.debug))

                # update met name in description
                object_settings['description'] = WF.updateFlaggedValues(object_settings['description'], '%%metname%%', curr_ensemble_set['metname'])

        data, data_settings = WF.mergeLines(data, data_settings, object_settings)

        data = self.Data.filterTimeSeries(data, data_settings)

        object_settings = self.Tables.replaceComparisonSettings(object_settings, self.iscomp)

        headings, rows = self.Tables.buildErrorStatsTable(object_settings, data_settings)
        headings = self.Tables.replaceIllegalJasperCharactersHeadings(headings)
        rows = self.Tables.replaceIllegalJasperCharactersRows(rows)

        object_settings = self.configureSettingsForID(self.base_id, object_settings)

        data, data_settings = self.Tables.correctTableUnits(data, data_settings, object_settings)

        object_settings['units_list'] = WF.getUnitsList(data_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)

        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))
        rows = WF.updateFlaggedValues(rows, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))
        headings = WF.updateFlaggedValues(headings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        if 'primarykeyheader' in object_settings.keys():
            primarykeyheader = object_settings['primarykeyheader']
        else:
            primarykeyheader = 'Year'

        isCollection = WF.checkForCollections(data_settings)
        if isCollection:
            members = self.Data.getMembers(object_settings, data_settings)
            object_settings['members'] = members
            if self.memberiteration:
                # get the ensemble set for the current member
                curr_ensemble_set = WF.matchMemberToEnsembleSet(self.ensembleSets, self.member)

                object_settings = WF.updateFlaggedValues(object_settings, '%%member%%', WF.getOriginalMemberNumber(self.member, curr_ensemble_set, self.DSSFile,
                                                                                                                   self.alternativeFpart, self.StartTime, self.EndTime, self.debug))

                # update metname
                object_settings = WF.updateFlaggedValues(object_settings, '%%metname%%', curr_ensemble_set['metname'])

            rows = self.Tables.configureRowsForCollection(rows, object_settings)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
            # desc = WF.parseForTextFlags(desc)
            desc = WF.updateFlaggedValues(desc, '%%year%%', object_settings['allyearsstr'])
        else:
            desc = ''

        table_constructor = {}

        for yi, year in enumerate(object_settings['years']):  #iterate years. If iscomp, that;s the date header.
            yearlydata = self.Tables.filterTableData(data, object_settings)
            yearheader = object_settings['yearstr'][yi]
            for hi, header in enumerate(headings):
                tcnum = len(table_constructor.keys())
                table_constructor[tcnum] = {}
                if self.iscomp:
                    table_constructor[tcnum]['datecolumn'] = yearheader

                header_frmt = header.replace('%%year%%', yearheader)
                frmt_rows = []
                threshold_colors = np.full(len(rows), None)
                for ri, row in enumerate(rows):
                    s_row = row.split('|')
                    rowname = s_row[0]
                    row_val = s_row[hi + 1]
                    stat = None
                    addasterisk = False
                    if self.memberiteration:
                        member = self.member
                    else:
                        member = None
                    if '%%' in rowname:
                        if isCollection and '%%member' in rowname:
                            member = int(rowname.split('%%member.')[1].split('%%')[0])
                            curr_ensemble_set = WF.matchMemberToEnsembleSet(self.ensembleSets, member)
                            rowname = WF.updateFlaggedValues(rowname, f'%%member.{member}%%', WF.getOriginalMemberNumber(member, curr_ensemble_set, self.DSSFile,
                                                                                                                         self.alternativeFpart, self.StartTime, self.EndTime, self.debug))

                    if '%%' in row_val:
                        rowdata, sr_month = self.Tables.getStatsLineData(row_val, yearlydata, year=year, data_key=member)
                        if len(rowdata) == 0:
                            stat = self.Tables.getStat(row_val)
                            row_val = None
                        else:
                            missing_data = self.Tables.checkForMissingData(row_val, rowdata)
                            if missing_data:
                                stat = self.Tables.getStat(row_val)
                                row_val = np.nan
                            else:
                                row_val, stat = self.Tables.getStatsLine(row_val, rowdata)
                            if not np.isnan(row_val) and row_val is not None:
                                thresholdsettings = self.Tables.matchThresholdToStat(stat, object_settings)
                                for thresh in thresholdsettings:
                                    if row_val < thresh['value']:
                                        if 'colorwhen' in thresh.keys():
                                            if thresh['colorwhen'] == 'under':
                                                threshold_colors[ri] = thresh['color']
                                                if 'addasterisk' in object_settings.keys():
                                                    if object_settings['addasterisk'].lower() == 'true':
                                                        addasterisk = True
                                            if thresh['when'] == 'under':
                                                if 'replacement' in thresh.keys():
                                                    row_val = thresh['replacement']
                                    elif row_val > thresh['value']:
                                        if thresh['colorwhen'] == 'over':
                                            threshold_colors[ri] = thresh['color']
                                            if 'addasterisk' in object_settings.keys():
                                                if object_settings['addasterisk'].lower() == 'true':
                                                    addasterisk = True
                                        if thresh['when'] == 'over':
                                            if 'replacement' in thresh.keys():
                                                row_val = thresh['replacement']

                            else:
                                if 'missingmarker' in object_settings.keys():
                                    row_val = object_settings['missingmarker']

                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings)
                            self.WAT_log.addLogEntry({'type': 'Statistic',
                                                      'name': ' '.join([self.ChapterRegion, header_frmt, stat]),
                                                      'description': desc,
                                                      'value': row_val,
                                                      'function': stat,
                                                      'units': object_settings['plot_units'],
                                                      'value_start_date': WT.translateDateFormat(data_start_date,
                                                                                                 'datetime', '',
                                                                                                 self.StartTime,
                                                                                                 self.EndTime,
                                                                                                 debug=self.debug),
                                                      'value_end_date': WT.translateDateFormat(data_end_date,
                                                                                               'datetime', '',
                                                                                               self.StartTime,
                                                                                               self.EndTime,
                                                                                               debug=self.debug),
                                                      'logoutputfilename': ', '.join(
                                                          [data_settings[flag]['logoutputfilename'] for flag in
                                                           data_settings])
                                                      },
                                                     isdata=True)

                    header_frmt = '' if header_frmt is None else header_frmt
                    numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings)
                    formatted_number = WF.formatNumbers(row_val, numberFormat)
                    if addasterisk:
                        formatted_number += '*'
                    frmt_rows.append('{0}|{1}'.format(rowname, formatted_number))
                table_constructor[tcnum]['rows'] = frmt_rows
                table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                table_constructor[tcnum]['header'] = header_frmt

        keeptable = False
        keepall = True
        keepcolumn = {}
        missing_value_objects = ['nan', '-', 'none', '--']
        if 'missingmarker' in object_settings.keys():
            missing_value_objects.append(object_settings['missingmarker'])
        for row_num in table_constructor.keys():
            constructor = table_constructor[row_num]
            rows = constructor['rows']
            header = constructor['header']
            for row in rows:
                srow = row.split('|')
                if header not in keepcolumn.keys():
                    keepcolumn[header] = False
                if srow[1].lower() not in missing_value_objects:
                    keepcolumn[header] = True

        for key in keepcolumn.keys():
            if keepcolumn[key]:
                keeptable = True
            else:
                keepall = False

        if keeptable:  #quick check if we're even writing a table.
            if not keepall:
                new_table_constructor = {}
                for row_num in table_constructor.keys():
                    constructor = table_constructor[row_num]
                    header = constructor['header']
                    if keepcolumn[header]:
                        new_table_constructor[row_num] = constructor
            else:
                new_table_constructor = table_constructor

            #THEN write table
            if self.iscomp:
                self.XML.writeDateControlledTableStart(desc, primarykeyheader)
            else:
                WF.print2stdout(f'##### if self.iscomp: = False ##### , writing table start, should output Report_Element : control_point_tables ')
                self.XML.writeTableStart(desc, primarykeyheader)

            self.Tables.writeTable(new_table_constructor)
            if not keepall:
                self.Tables.writeMissingTableItemsWarning(desc)
        else:
            self.Tables.writeMissingTableWarning(desc)
            WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)

        WF.print2stdout(f'Error Stats table took {time.time() - objectstarttime} seconds.')

    def makeMonthlyStatisticsTable(self, object_settings):
        """
        Takes in object settings to build monthly stats table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: writes to XML file
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Monthly Stats Table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)

        default_settings = self.loadDefaultPlotObject('monthlystatisticstable')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', forjasper=True)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        data, data_settings = self.Data.getTableDataDictionary(object_settings)
        data, data_settings = WF.mergeLines(data, data_settings, object_settings)

        data = self.Data.filterTimeSeries(data, data_settings)

        object_settings = self.Tables.replaceComparisonSettings(object_settings, self.iscomp)

        headings, rows = self.Tables.buildMonthlyStatsTable(object_settings, data_settings)

        object_settings = self.configureSettingsForID(self.base_id, object_settings)

        data = self.Tables.filterTableData(data, object_settings)
        data, data_settings = self.Tables.correctTableUnits(data, data_settings, object_settings)
        object_settings['units_list'] = WF.getUnitsList(data_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)
        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        thresholds = self.Tables.getThresholdsfromSettings(object_settings)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
            # desc = WF.parseForTextFlags(desc)
            desc = WF.updateFlaggedValues(desc, '%%year%%', object_settings['yearstr'])
        else:
            desc = ''

        table_constructor = {}

        for yi, year in enumerate(object_settings['years']):  # Iterate years. If iscomp, that's the date header.

            yearheader = object_settings['yearstr'][yi]
            for hi, header in enumerate(headings):
                tcnum = len(table_constructor.keys())
                table_constructor[tcnum] = {}
                if self.iscomp:
                    table_constructor[tcnum]['datecolumn'] = yearheader

                header_frmt = header.replace('%%year%%', yearheader)

                frmt_rows = []
                threshold_colors = np.full(len(rows), None)
                for ri, row in enumerate(rows):
                    s_row = row.split('|')
                    rowname = s_row[0]
                    row_val = s_row[hi + 1]
                    stat = None
                    addasterisk = False
                    if '%%' in row_val:
                        rowdata, sr_month = self.Tables.getStatsLineData(row_val, data, year=year)
                        if len(rowdata) == 0:
                            stat = self.Tables.getStat(row_val)
                            row_val = None
                        else:
                            missing_data = self.Tables.checkForMissingData(row_val, rowdata)
                            if missing_data:
                                stat = self.Tables.getStat(row_val)
                                row_val = np.nan
                            else:
                                row_val, stat = self.Tables.getStatsLine(row_val, rowdata)
                            if not np.isnan(row_val) and row_val is not None:
                                for thresh in thresholds:
                                    if row_val < thresh['value']:
                                        if 'colorwhen' in thresh.keys():
                                            if thresh['colorwhen'] == 'under':
                                                threshold_colors[ri] = thresh['color']
                                                if 'addasterisk' in object_settings.keys():
                                                    if object_settings['addasterisk'].lower() == 'true':
                                                        addasterisk = True
                                            if thresh['when'] == 'under':
                                                if 'replacement' in thresh.keys():
                                                    row_val = thresh['replacement']
                                    elif row_val > thresh['value']:
                                        if thresh['colorwhen'] == 'over':
                                            threshold_colors[ri] = thresh['color']
                                            if 'addasterisk' in object_settings.keys():
                                                if object_settings['addasterisk'].lower() == 'true':
                                                    addasterisk = True
                                        if thresh['when'] == 'over':
                                            if 'replacement' in thresh.keys():
                                                row_val = thresh['replacement']

                            else:
                                if 'missingmarker' in object_settings.keys():
                                    row_val = object_settings['missingmarker']
                                else:
                                    row_val = '-'
                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings)
                            self.WAT_log.addLogEntry({'type': 'Statistic',
                                                      'name': ' '.join([self.ChapterRegion, header_frmt, stat]),
                                                      'description': desc,
                                                      'value': row_val,
                                                      'function': stat,
                                                      'units': object_settings['plot_units'],
                                                      'value_start_date': WT.translateDateFormat(data_start_date,
                                                                                                 'datetime', '',
                                                                                                 self.StartTime,
                                                                                                 self.EndTime,
                                                                                                 debug=self.debug),
                                                      'value_end_date': WT.translateDateFormat(data_end_date,
                                                                                               'datetime', '',
                                                                                               self.StartTime,
                                                                                               self.EndTime,
                                                                                               debug=self.debug),
                                                      'logoutputfilename': ', '.join(
                                                          [data_settings[flag]['logoutputfilename'] for flag in
                                                           data_settings])
                                                      },
                                                     isdata=True)

                    header_frmt = '' if header_frmt is None else header_frmt
                    numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings)
                    formattedNumber = WF.formatNumbers(row_val, numberFormat)
                    if addasterisk:
                        formattedNumber += '*'
                    frmt_rows.append('{0}|{1}'.format(rowname, formattedNumber))
                table_constructor[tcnum]['rows'] = frmt_rows
                table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                table_constructor[tcnum]['header'] = header_frmt

        keeptable = False
        keepall = True
        keepcolumn = {}
        missing_value_objects = ['nan', '-', 'none', '--']
        if 'missingmarker' in object_settings.keys():
            missing_value_objects.append(object_settings['missingmarker'])
        for row_num in table_constructor.keys():
            constructor = table_constructor[row_num]
            rows = constructor['rows']
            header = constructor['header']
            for row in rows:
                srow = row.split('|')
                if header not in keepcolumn.keys():
                    keepcolumn[header] = False
                if srow[1].lower() not in missing_value_objects:
                    keepcolumn[header] = True

        for key in keepcolumn.keys():
            if keepcolumn[key]:
                keeptable = True
            else:
                keepall = False

        if keeptable:  # quick check if we're even writing a table.
            if not keepall:
                new_table_constructor = {}
                for row_num in table_constructor.keys():
                    constructor = table_constructor[row_num]
                    header = constructor['header']
                    if keepcolumn[header]:
                        new_table_constructor[row_num] = constructor
            else:
                new_table_constructor = table_constructor

            # THEN write table
            if self.iscomp:
                self.XML.writeDateControlledTableStart(desc, 'Month')
            else:
                self.XML.writeTableStart(desc, 'Month')

            self.Tables.writeTable(new_table_constructor)
            if not keepall:
                self.Tables.writeMissingTableItemsWarning(desc)
        else:
            self.Tables.writeMissingTableWarning(desc)
            WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)

        WF.print2stdout(f'Monthly Stat Table took {time.time() - objectstarttime} seconds.')

    def makeSingleStatisticTable(self, object_settings):
        """
        takes in object settings to build Single Statistic table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Single Statistic Table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)

        default_settings = self.loadDefaultPlotObject('singlestatistictable') #get default SingleStat items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) #overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', forjasper=True)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        data, data_settings = self.Data.getTableDataDictionary(object_settings)
        data, data_settings = WF.mergeLines(data, data_settings, object_settings)

        data = self.Data.filterTimeSeries(data, data_settings)

        data = self.Tables.filterTableData(data, object_settings)
        data, data_settings = self.Tables.correctTableUnits(data, data_settings, object_settings)
        object_settings['units_list'] = WF.getUnitsList(data_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)
        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        object_settings = self.Tables.replaceComparisonSettings(object_settings, self.iscomp)

        thresholds = self.Tables.getThresholdsfromSettings(object_settings)

        object_settings_blueprint = pickle.loads(pickle.dumps(object_settings, -1))

        headings, rows = self.Tables.buildSingleStatTable(object_settings_blueprint, data_settings)
        object_settings = self.configureSettingsForID(self.base_id, object_settings)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
            # desc = WF.parseForTextFlags(desc)
            desc = WF.updateFlaggedValues(desc, '%%year%%', object_settings['yearstr'])
        else:
            desc = ''

        table_constructor = {}

        if self.iscomp:
            datecolumns = self.Constants.mo_str_3
        else:
            datecolumns = ['']  # if not comp run, we don't need date headings, months will be in headings

        for mi, month in enumerate(datecolumns):
            if len(headings) == 0:
                WF.print2stdout('No headings for table.', debug=self.debug)
                # self.XML.writeDateColumn(month)
            for i, header in enumerate(headings):
                tcnum = len(table_constructor.keys())
                table_constructor[tcnum] = {}
                if self.iscomp:  #if a comparison, write the date column.
                    table_constructor[tcnum]['datecolumn'] = month
                frmt_rows = []
                threshold_colors = np.full(len(rows), None)
                for ri, row in enumerate(rows):
                    s_row = row.split('|')
                    rowname = s_row[0]
                    if rowname == 'ALLYEARS':
                        rowname = 'All'
                    year = object_settings['years'][ri]
                    row_val = s_row[i + (len(headings) * mi) + 1]
                    stat = None
                    addasterisk = False
                    if '%%' in row_val:
                        rowdata, sr_month = self.Tables.getStatsLineData(row_val, data, year=year)
                        if len(rowdata) == 0:
                            stat = self.Tables.getStat(row_val)
                            row_val = np.nan
                        else:
                            missing_data = self.Tables.checkForMissingData(row_val, rowdata)
                            if missing_data:
                                stat = self.Tables.getStat(row_val)
                                row_val = np.nan
                            else:
                                row_val, stat = self.Tables.getStatsLine(row_val, rowdata)
                            if np.isnan(row_val):
                                if 'missingmarker' in object_settings.keys():
                                    row_val = object_settings['missingmarker']
                                else:
                                    row_val = '-'
                            else:
                                for thresh in thresholds:
                                    if row_val < thresh['value']:
                                        if 'colorwhen' in thresh.keys():
                                            if thresh['colorwhen'] == 'under':
                                                threshold_colors[ri] = thresh['color']
                                                if 'addasterisk' in object_settings.keys():
                                                    if object_settings['addasterisk'].lower() == 'true':
                                                        addasterisk = True
                                            if thresh['when'] == 'under':
                                                if 'replacement' in thresh.keys():
                                                    row_val = thresh['replacement']
                                    elif row_val > thresh['value']:
                                        if thresh['colorwhen'] == 'over':
                                            threshold_colors[ri] = thresh['color']
                                            if 'addasterisk' in object_settings.keys():
                                                if object_settings['addasterisk'].lower() == 'true':
                                                    addasterisk = True
                                        if thresh['when'] == 'over':
                                            if 'replacement' in thresh.keys():
                                                row_val = thresh['replacement']

                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings_blueprint, month=sr_month)
                            self.WAT_log.addLogEntry({'type': 'Statistic',
                                                      'name': ' '.join([self.ChapterRegion, header, stat]),
                                                      'description': object_settings_blueprint['description'],
                                                      'value': row_val,
                                                      'units': object_settings_blueprint['units_list'],
                                                      'function': stat,
                                                      'value_start_date': WT.translateDateFormat(data_start_date,
                                                                                                 'datetime', '',
                                                                                                 self.StartTime,
                                                                                                 self.EndTime,
                                                                                                 debug=self.debug),
                                                      'value_end_date': WT.translateDateFormat(data_end_date,
                                                                                               'datetime', '',
                                                                                               self.StartTime,
                                                                                               self.EndTime,
                                                                                               debug=self.debug),
                                                      'logoutputfilename': ', '.join(
                                                          [data_settings[flag]['logoutputfilename'] for flag in
                                                           data_settings])
                                                      },
                                                     isdata=True)

                    header = '' if header is None else header
                    numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings_blueprint)
                    formattedNumber = WF.formatNumbers(row_val, numberFormat)
                    if addasterisk:
                        formattedNumber += '*'
                    frmt_rows.append('{0}|{1}'.format(rowname, formattedNumber))
                table_constructor[tcnum]['rows'] = frmt_rows
                table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                table_constructor[tcnum]['header'] = header

        # Check for entire rows/columns that can be sniped
        keeptable = False
        keepall = True
        keepheader = {}
        missing_value_objects = ['nan', '-', 'none', '--']
        if 'missingmarker' in object_settings.keys():
            missing_value_objects.append(object_settings['missingmarker'])
        for row_num in table_constructor.keys():
            constructor = table_constructor[row_num]
            rows = constructor['rows']
            for row in rows:
                srow = row.split('|')
                if srow[0] not in keepheader.keys():
                    keepheader[srow[0]] = False
                if srow[1].lower() not in missing_value_objects:
                    keepheader[srow[0]] = True

        for key in keepheader.keys():
            if keepheader[key]:
                keeptable = True
            else:
                keepall = False

        if keeptable:  #quick check if we're even writing a table.
            if not keepall:
                for row_num in table_constructor.keys():
                    constructor = table_constructor[row_num]
                    rows = constructor['rows']
                    new_rows = []
                    new_thresh = []
                    for r, row in enumerate(rows):
                        srow = row.split('|')
                        if keepheader[srow[0]]:
                            new_rows.append(row)
                            new_thresh.append(constructor['thresholdcolors'][r])
                    table_constructor[row_num]['rows'] = new_rows
                    table_constructor[row_num]['thresholdcolors'] = new_thresh

            # THEN write table
            if self.iscomp:
                self.XML.writeDateControlledTableStart(desc, 'Year')
            else:
                self.XML.writeNarrowTableStart(desc, 'Year')

            self.Tables.writeTable(table_constructor)
            if not keepall:
                self.Tables.writeMissingTableItemsWarning(desc)
        else:
            self.Tables.writeMissingTableWarning(desc)
            WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)

        WF.print2stdout(f'Single Stat Table took {time.time() - objectstarttime} seconds.')

    def makeSingleStatisticProfileTable(self, object_settings):
        """
        takes in object settings to build Single Statistic profile table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Single Statistic Profile Table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)
        self.Profiles = WProfile.Profiles(self)

        default_settings = self.loadDefaultPlotObject('singlestatisticprofiletable') #get default SingleStat items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) #overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', forjasper=True)

        object_settings['datakey'] = 'datapaths'

        ################# Get timestamps #################
        object_settings['datessource_flag'] = WF.getDateSourceFlag(object_settings)
        object_settings['timestamps'] = self.Profiles.getProfileTimestamps(object_settings, self.StartTime, self.EndTime)
        object_settings['timestamp_index'] = self.Profiles.getProfileTimestampYearMonthIndex(object_settings, self.years)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        data, line_settings, missing = self.Data.getProfileDataDictionary(object_settings)
        line_settings = WF.correctDuplicateLabels(line_settings)
        object_settings['warnings'] = self.Profiles.checkProfileValidity(data, object_settings, combineyears=True)

        object_settings = self.Tables.replaceComparisonSettings(object_settings, self.iscomp)

        ################# Get plot units #################
        data, line_settings = self.Profiles.convertProfileDataUnits(object_settings, data, line_settings)
        object_settings['units_list'] = WF.getUnitsList(line_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)

        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        # self.Data.commitProfileDataToMemory(data, line_settings, object_settings)

        # object_settings['usedepth'] = self.Profiles.confirmValidDepths(data)
        # if object_settings['usedepth']:
        #     data = self.Profiles.snapTo0Depth(data, line_settings)
        if 'usedepth' not in object_settings.keys():
            object_settings['usedepth'] = 'true'

        if object_settings['usedepth'].lower() == 'false':
            wse_data = self.Data.getProfileWSE(object_settings, onflag='datapaths')
            data = self.Profiles.convertDepthsToElevations(data, wse_data)
        elif object_settings['usedepth'].lower() == 'true':
            wse_data = self.Data.getProfileWSE(object_settings, onflag='datapaths')
            data = self.Profiles.convertElevationsToDepths(data, wse_data=wse_data)

        data, object_settings = self.Profiles.filterProfileData(data, line_settings, object_settings)

        object_settings['resolution'] = self.Profiles.getProfileInterpResolution(object_settings)

        thresholds = self.Tables.getThresholdsfromSettings(object_settings)

        object_settings_blueprint = pickle.loads(pickle.dumps(object_settings, -1))

        headings, rows = self.Tables.buildSingleStatTable(object_settings_blueprint, line_settings)
        object_settings = self.configureSettingsForID(self.base_id, object_settings)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
            # desc = WF.parseForTextFlags(desc)
            desc = WF.updateFlaggedValues(desc, '%%year%%', object_settings['yearstr'])
        else:
            desc = ''

        table_constructor = {}

        if self.iscomp:
            datecolumns = self.Constants.mo_str_3
        else:
            datecolumns = ['']  # If not comp run we don't need date headings, months will be in headings

        for mi, month in enumerate(datecolumns):
            if len(headings) == 0:
                WF.print2stdout('No headings for table.', debug=self.debug)
            for i, header in enumerate(headings):
                tcnum = len(table_constructor.keys())
                table_constructor[tcnum] = {}
                if self.iscomp:
                    table_constructor[tcnum]['datecolumn'] = month
                frmt_rows = []
                threshold_colors = np.full(len(rows), None)
                for ri, row in enumerate(rows):
                    s_row = row.split('|')
                    rowname = s_row[0]
                    if rowname == 'ALLYEARS':
                        rowname = 'All'
                    year = object_settings['years'][ri]
                    row_val = s_row[i + (len(headings) * mi) + 1]
                    stat = None
                    addasterisk = False
                    if '%%' in row_val:
                        rowval_stats = {}
                        if year == 'ALLYEARS':
                            if self.iscomp:
                                data_idx = WF.getAllMonthIdx(object_settings_blueprint['timestamp_index'], mi)
                            else:
                                data_idx = WF.getAllMonthIdx(object_settings_blueprint['timestamp_index'], i)
                        else:
                            if self.iscomp:
                                data_idx = object_settings_blueprint['timestamp_index'][ri][mi]
                            else:
                                data_idx = object_settings_blueprint['timestamp_index'][ri][i]
                        for di in data_idx:
                            stats_data = self.Tables.formatStatsProfileLineData(row_val, data,
                                                                                object_settings_blueprint['resolution'],
                                                                                object_settings_blueprint['usedepth'],
                                                                                di)

                            rowval_stats = self.Profiles.stackProfileIndicies(rowval_stats, stats_data)

                        missing_data = self.Tables.checkForMissingData(row_val, rowval_stats)
                        if missing_data:
                            stat = self.Tables.getStat(row_val)
                            row_val = np.nan
                        else:
                            row_val, stat = self.Tables.getStatsLine(row_val, rowval_stats)
                        if np.isnan(row_val):
                            if 'missingmarker' in object_settings.keys():
                                row_val = object_settings['missingmarker']
                            else:
                                row_val = '-'
                        else:
                            for thresh in thresholds:
                                if row_val < thresh['value']:
                                    if 'colorwhen' in thresh.keys():
                                        if thresh['colorwhen'] == 'under':
                                            threshold_colors[ri] = thresh['color']
                                            if 'addasterisk' in object_settings.keys():
                                                if object_settings['addasterisk'].lower() == 'true':
                                                    addasterisk = True
                                        if thresh['when'] == 'under':
                                            if 'replacement' in thresh.keys():
                                                row_val = thresh['replacement']
                                elif row_val > thresh['value']:
                                    if thresh['colorwhen'] == 'over':
                                        threshold_colors[ri] = thresh['color']
                                        if 'addasterisk' in object_settings.keys():
                                            if object_settings['addasterisk'].lower() == 'true':
                                                addasterisk = True
                                    if thresh['when'] == 'over':
                                        if 'replacement' in thresh.keys():
                                            row_val = thresh['replacement']

                        if self.iscomp:
                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings_blueprint, month=month)
                        else:
                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings_blueprint, month=header)
                        self.WAT_log.addLogEntry({'type': 'ProfileTableStatistic',
                                                  'name': ' '.join([self.ChapterRegion, header, stat]),
                                                  'description': object_settings_blueprint['description'],
                                                  'value': row_val,
                                                  'function': stat,
                                                  'units': object_settings_blueprint['plot_units'],
                                                  'value_start_date': data_start_date,
                                                  'value_end_date': data_end_date,
                                                  'logoutputfilename': ', '.join(
                                                      [line_settings[flag]['logoutputfilename'] for flag in
                                                       line_settings])
                                                  },
                                                 isdata=True)

                    header = '' if header is None else header
                    numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings_blueprint)
                    formattedNumber = WF.formatNumbers(row_val, numberFormat)
                    if addasterisk:
                        formattedNumber += '*'
                    frmt_rows.append('{0}|{1}'.format(rowname, formattedNumber))
                table_constructor[tcnum]['rows'] = frmt_rows
                table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                table_constructor[tcnum]['header'] = header

        keeptable = False
        keepall = True
        keepheader = {}
        missing_value_objects = ['nan', '-', 'none', '--']
        if 'missingmarker' in object_settings.keys():
            missing_value_objects.append(object_settings['missingmarker'])
        for row_num in table_constructor.keys():
            constructor = table_constructor[row_num]
            rows = constructor['rows']
            for row in rows:
                srow = row.split('|')
                if srow[0] not in keepheader.keys():
                    keepheader[srow[0]] = False
                if srow[1].lower() not in missing_value_objects:
                    keepheader[srow[0]] = True

        for key in keepheader.keys():
            if keepheader[key]:
                keeptable = True
            else:
                keepall = False

        if keeptable:  # Quick check if we're even writing a table.
            if not keepall:
                for row_num in table_constructor.keys():
                    constructor = table_constructor[row_num]
                    rows = constructor['rows']
                    new_rows = []
                    new_thresh = []
                    for r, row in enumerate(rows):
                        srow = row.split('|')
                        if keepheader[srow[0]]:
                            new_rows.append(row)
                            new_thresh.append(constructor['thresholdcolors'][r])
                    table_constructor[row_num]['rows'] = new_rows
                    table_constructor[row_num]['thresholdcolors'] = new_thresh

            # THEN write table
            if self.iscomp:
                self.XML.writeDateControlledTableStart(desc, 'Year')
            else:
                self.XML.writeNarrowTableStart(desc, 'Year')

            self.Tables.writeTable(table_constructor)
            if not keepall:
                self.Tables.writeMissingTableItemsWarning(desc)

        else:
            self.Tables.writeMissingTableWarning(desc)
            WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)

        ignorewarnings = False
        if 'ignorewarnings' in object_settings.keys():
            if object_settings['ignorewarnings'] != 'true'.lower():
                ignorewarnings = True
        if not ignorewarnings:
            # self.Profiles.writeWarnings(object_settings['warnings'], 'ALLYEARS')
            WF.print2stdout('Not currently writing out warnings to report', debug=self.debug)

        WF.print2stdout(f'Single Profile Stat Table took {time.time() - objectstarttime} seconds.')

    def makeContourPlot(self, object_settings):
        """
        Takes in object settings to build contour plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Contour Plot.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Plots = WPlot.Plots(self)

        default_settings = self.loadDefaultPlotObject('contourplot') #get default TS plot items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) #overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', exclude=['description'])
        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', include=['description'],
                                                  forjasper=True)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        object_settings['years'], object_settings['yearstr'] = WF.organizePlotYears(object_settings)

        if 'description' not in object_settings.keys():
            object_settings['description'] = ''
        # else:
        #     object_settings['description'] = WF.parseForTextFlags(object_settings['description'])

        for yi, year in enumerate(object_settings['years']):
            useAx = []
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))

            yearstr = object_settings['yearstr'][yi]

            cur_obj_settings = self.Plots.setTimeSeriesXlims(cur_obj_settings, yearstr, object_settings['years'])

            # NOTES
            # Data structure:
            # 2D array of dates[distance from source]
            # array of dates corresponding to the number of the first D of array above
            # supplementary array for distances corrsponding to the second D of array above
            # ex
            # [[1,2,3,5],[2,3,4,2],[5,3,2,5]] #values per date at a distance
            # [01jan2016, 04Feb2016, 23May2016] #dates
            # [0, 19, 25, 35] #distances

            contoursbyID, contoursbyID_settings = self.Data.getContourDataDictionary(cur_obj_settings)
            contoursbyID = WF.filterDataByYear(contoursbyID, year)
            selectedContourIDs = WF.getUsedIDs(contoursbyID_settings)

            straightlines = self.Data.getStraightLineValue(cur_obj_settings)

            if len(selectedContourIDs) == 1:
                figsize = (12, 6)
                pageformat = 'half'
            else:
                figsize = (12, 12)
                pageformat = 'full'

            if pageformat == 'full':
                height_ratios = []
                for i in range(len(selectedContourIDs)):
                    if i == len(selectedContourIDs) - 1:
                        height_ratios.append(1)
                    else:
                        height_ratios.append(.75)
                fig, axes = plt.subplots(ncols=1, nrows=len(selectedContourIDs), sharex=True, figsize=figsize,
                                         gridspec_kw={'height_ratios': height_ratios})
            else:
                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize,
                                         )

            for IDi, ID in enumerate(selectedContourIDs):
                contour_plot_settings = pickle.loads(pickle.dumps(cur_obj_settings, -1))
                contour_plot_settings = self.configureSettingsForID(ID, contour_plot_settings)
                contours = WF.selectContourByID(contoursbyID, ID)
                contour_settings = WF.selectContourByID(contoursbyID_settings, ID)
                values, dates, distance, transitions = WF.stackContours(contours, contoursbyID_settings)
                if len(selectedContourIDs) == 1:
                    axes = [axes]

                ax = axes[IDi]

                for contourname in contour_settings:
                    c_set = contour_settings[contourname]
                    parameter, contour_plot_settings['param_count'] = WF.getParameterCount(c_set, contour_plot_settings)

                if 'units' in contour_plot_settings.keys():
                    units = contour_plot_settings['units']
                else:
                    if 'parameter' in contour_plot_settings.keys():
                        parameter = contour_plot_settings['parameter']
                    else:
                        parameter = ''
                        top_count = 0
                        for key in contour_plot_settings['param_count'].keys():
                            if contour_plot_settings['param_count'][key] > top_count:
                                parameter = key
                    try:
                        units = self.Constants.units[parameter]
                    except KeyError:
                        units = None

                if isinstance(units, dict):
                    if 'unitsystem' in contour_plot_settings.keys():
                        units = units[contour_plot_settings['unitsystem'].lower()]
                    else:
                        units = None

                if 'unitsystem' in contour_plot_settings.keys():
                    values, units = WF.convertUnitSystem(values, units, contour_plot_settings['unitsystem']) #TODO: confirm

                chkvals = WF.checkData(values)
                if not chkvals:
                    WF.print2stdout('Invalid Data settings for contour plot year {0}'.format(year), debug=self.debug)
                    continue

                dates = WT.JDateToDatetime(dates, self.startYear)

                if 'label' not in contour_plot_settings.keys():
                    contour_plot_settings['label'] = ''
                else:
                    contour_plot_settings['label'] = WF.formatTextFlags(contour_plot_settings['label'])

                if 'description' not in contour_plot_settings.keys():
                    contour_plot_settings['description'] = ''
                # else:
                #     contour_plot_settings['description'] = WF.parseForTextFlags(contour_plot_settings['description'])

                contour_plot_settings = WD.getDefaultContourSettings(contour_plot_settings, debug=self.debug)

                if 'min' in contour_plot_settings['colorbar']:
                    vmin = float(contour_plot_settings['colorbar']['min'])
                else:
                    vmin = np.nanmin(values)
                if 'max' in contour_plot_settings['colorbar']:
                    vmax = float(contour_plot_settings['colorbar']['max'])
                else:
                    vmax = np.nanmax(values)

                contr = ax.contourf(dates, distance, values, cmap=contour_plot_settings['colorbar']['colormap'],
                                    vmin=vmin, vmax=vmax,
                                    levels=np.linspace(vmin, vmax, int(contour_plot_settings['colorbar']['bins'])), #add one to get the desired number..
                                    extend='both') #the .T transposes the array so dates on bottom TODO:make extend variable

                self.WAT_log.addLogEntry({'type': contour_plot_settings['label'] +
                                          '_ContourPlot' if contour_plot_settings['label'] != '' else 'ContourPlot',
                                          'name': self.ChapterRegion + '_' + yearstr,
                                          'description': contour_plot_settings['description'],
                                          'units': units,
                                          'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                     self.StartTime, self.EndTime,
                                                                                     debug=self.debug).strftime('%d %b %Y'),
                                          'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                                   self.StartTime, self.EndTime,
                                                                                   debug=self.debug).strftime('%d %b %Y'),
                                          'logoutputfilename': 'NA'
                                          },
                                         isdata=True)

                contour_plot_settings = WF.updateFlaggedValues(contour_plot_settings, '%%units%%', WF.formatUnitsStrings(units))

                if 'contourlines' in contour_plot_settings.keys():
                    for contourline in contour_plot_settings['contourlines']:
                        if 'value' in contourline.keys():
                            val = float(contourline['value'])
                        else:
                            WF.print2stdout('No Value set for contour line.', debug=self.debug)
                            continue
                        contourline = WD.getDefaultContourLineSettings(contourline)
                        cs = ax.contour(contr, levels=[val], linewidths=[float(contourline['linewidth'])], colors=[contourline['linecolor']],
                                        linestyles=contourline['linestylepattern'], alpha=float(contourline['alpha']))
                        if 'contourlinetext' in contourline.keys():
                            if contourline['contourlinetext'].lower() == 'true':
                                ax.clabel(cs, inline_spacing=contourline['inline_spacing'],
                                          fontsize=contourline['fontsize'], inline=contourline['text_inline'])
                        if 'show_in_legend' in contourline.keys():
                            if contourline['show_in_legend'].lower() == 'true':
                                if 'label' in contourline.keys():
                                    label = WF.formatTextFlags(contourline['label'])
                                else:
                                    label = WF.formatTextFlags(str(val))
                                cl_leg = ax.plot([], [], c=contourline['linecolor'], ls=contourline['linestylepattern'],
                                                 alpha=float(contourline['alpha']), lw=float(contourline['linewidth']),
                                                 label=label)

                ### VERTICAL LINES ###
                self.Plots.plotVerticalLines(straightlines, ax, contour_plot_settings, isdate=True)

                ### Horizontal LINES ###
                self.Plots.plotHorizontalLines(straightlines, ax, contour_plot_settings)

                if self.iscomp:
                    if 'modeltext' in contour_plot_settings.keys():
                        modeltext = contour_plot_settings['modeltext']
                    else:
                        modeltext = self.SimulationName
                    plt.text(1.02, 0.5, modeltext, fontsize=12, transform=ax.transAxes, verticalalignment='center',
                             horizontalalignment='center', rotation='vertical')

                if 'gridlines' in contour_plot_settings.keys():
                    if contour_plot_settings['gridlines'].lower() == 'true':
                        ax.grid(True)

                if 'ylabel' in contour_plot_settings.keys():
                    if 'ylabelsize' in contour_plot_settings.keys():
                        ylabsize = float(contour_plot_settings['ylabelsize'])
                    elif 'fontsize' in contour_plot_settings.keys():
                        ylabsize = float(contour_plot_settings['fontsize'])
                    else:
                        ylabsize = 12
                    ylabel = WF.formatTextFlags(contour_plot_settings['ylabel'])
                    ax.set_ylabel(ylabel, fontsize=ylabsize)

                xmin, xmax = ax.get_xlim()
                # if contour_settings['dateformat'].lower() == 'datetime':
                xmin = mpl.dates.num2date(xmin)
                xmax = mpl.dates.num2date(xmax)
                if 'xticks' in contour_plot_settings.keys():
                    xtick_settings = contour_plot_settings['xticks']
                else:
                    xtick_settings = {}
                self.Plots.formatTimeSeriesXticks(ax, xtick_settings, contour_plot_settings)

                ax.set_xlim(left=xmin)
                ax.set_xlim(right=xmax)

                self.Plots.formatYTicks(ax, contour_plot_settings)

                if 'transitions' in contour_plot_settings.keys():
                    for transkey in transitions.keys():
                        transition_start = transitions[transkey]
                        trans_name = None
                        hline = WD.getDefaultStraightLineSettings(contour_plot_settings['transitions'], self.debug)

                        linecolor = WF.prioritizeKey(contours[transkey], hline, 'linecolor')
                        linestylepattern = WF.prioritizeKey(contours[transkey], hline, 'linestylepattern')
                        alpha = WF.prioritizeKey(contours[transkey], hline, 'alpha')
                        linewidth = WF.prioritizeKey(contours[transkey], hline, 'linewidth')

                        ax.axhline(y=transition_start, c=linecolor, ls=linestylepattern, alpha=float(alpha),
                                   lw=float(linewidth))
                        if 'name' in contour_plot_settings['transitions'].keys():
                            trans_flag = contour_plot_settings['transitions']['name'].lower() #blue:pink:white:pink:blue
                            text_settings = WD.getDefaultTextSettings(contour_plot_settings['transitions'], self.debug)

                            if trans_flag in contour_settings[transkey].keys():
                                trans_name = contour_settings[transkey][trans_flag]
                            if trans_name is not None:
                                if ax.get_ylim()[0] <= transition_start <= ax.get_ylim()[1]:
                                    trans_y_value = transition_start + ((ax.get_ylim()[1] - ax.get_ylim()[0]) * .01)

                                    fontcolor = WF.prioritizeKey(contour_settings[transkey], text_settings, 'fontcolor')
                                    fontsize = WF.prioritizeKey(contour_settings[transkey], text_settings, 'fontsize')
                                    horizontalalignment = WF.prioritizeKey(contour_settings[transkey], text_settings, 'horizontalalignment')
                                    text_x_pos = WF.prioritizeKey(contour_settings[transkey], text_settings, 'text_x_pos', backup=0.001)
                                    trans_x_value = mpl.dates.num2date(ax.get_xlim()[0] + ((ax.get_xlim()[1] - ax.get_xlim()[0]) * float(text_x_pos)))

                                    ax.text(trans_x_value, trans_y_value, trans_name, c=fontcolor, size=float(fontsize),
                                            horizontalalignment=horizontalalignment,
                                            verticalalignment='top')

                ax.invert_yaxis()

            # #stuff to call once per plot
            self.configureSettingsForID(self.base_id, cur_obj_settings)
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%units%%', WF.formatUnitsStrings(units))

            if 'title' in cur_obj_settings.keys():
                if 'titlesize' in cur_obj_settings.keys():
                    titlesize = float(object_settings['titlesize'])
                elif 'fontsize' in cur_obj_settings.keys():
                    titlesize = float(object_settings['fontsize'])
                else:
                    titlesize = 15
                title = WF.formatTextFlags(cur_obj_settings['title'])
                axes[0].set_title(title, fontsize=titlesize, wrap=True)

            if 'xlabel' in cur_obj_settings.keys():
                if 'xlabelsize' in cur_obj_settings.keys():
                    xlabsize = float(cur_obj_settings['xlabelsize'])
                elif 'fontsize' in cur_obj_settings.keys():
                    xlabsize = float(cur_obj_settings['fontsize'])
                else:
                    xlabsize = 12
                xlabel = WF.formatTextFlags(cur_obj_settings['xlabel'])
                axes[-1].set_xlabel(xlabel, fontsize=xlabsize)

            useplot = self.Plots.formatDateXAxis(axes[-1], cur_obj_settings)
            if not useplot:
                useAx.append(False)
            else:
                useAx.append(True)

            if 'legend' in cur_obj_settings.keys():
                if cur_obj_settings['legend'].lower() == 'true':
                    if 'legendsize' in cur_obj_settings.keys():
                        legsize = float(cur_obj_settings['legendsize'])
                    elif 'fontsize' in cur_obj_settings.keys():
                        legsize = float(cur_obj_settings['fontsize'])
                    else:
                        legsize = 12
                    if len(axes[0].get_legend_handles_labels()[0]) > 0:
                        plt.legend(fontsize=legsize)

            if not any(useAx):
                print(f'Plot for {year} not included due to xlimits.')
                plt.close("all")
                continue

            cbar = plt.colorbar(contr, ax=axes[-1], orientation='horizontal', aspect=50.)
            locs = np.linspace(vmin, vmax, int(contour_plot_settings['colorbar']['numticks']))
            cbar.set_ticks(locs)
            cbar.set_ticklabels(locs.round(2))
            if 'label' in contour_plot_settings['colorbar']:
                if 'labelsize' in contour_plot_settings['colorbar'].keys():
                    labsize = float(contour_plot_settings['colorbar']['labelsize'])
                elif 'fontsize' in contour_plot_settings['colorbar'].keys():
                    labsize = float(cur_obj_settings['colorbar']['fontsize'])
                else:
                    labsize = 12
                cbar_label = WF.formatTextFlags(contour_plot_settings['colorbar']['label'])
                cbar.set_label(cbar_label, fontsize=labsize)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.05)

            if 'description' not in cur_obj_settings.keys():
                cur_obj_settings['description'] = ''
            # else:
            #     cur_obj_settings['description'] = WF.parseForTextFlags(cur_obj_settings['description'])

            basefigname = os.path.join(self.images_path, 'ContourPlot' + '_' + self.ChapterRegion.replace(' ', '_')
                                       + '_' + yearstr)
            exists = True
            tempnum = 1
            tfn = basefigname
            while exists:
                if os.path.exists(tfn + '.png'):
                    tfn = basefigname + '_{0}'.format(tempnum)
                    tempnum += 1
                else:
                    exists = False
            figname = tfn + '.png'

            if self.highres:
                plt.savefig(figname, dpi=300)
            else:
                plt.savefig(figname)
            plt.close('all')

            if pageformat == 'full':
                self.XML.writeFullPagePlot(os.path.basename(figname), cur_obj_settings['description'])
            elif pageformat == 'half':
                self.XML.writeHalfPagePlot(os.path.basename(figname), cur_obj_settings['description'])

        WF.print2stdout(f'Contour Plot took {time.time() - objectstarttime} seconds.')

    def makeReservoirContourPlot(self, object_settings):
        """
        Takes in object settings to build reservoir contour plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Reservoir Contour Plot.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Plots = WPlot.Plots(self)

        default_settings = self.loadDefaultPlotObject('contourplot') #get default TS plot items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) #overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', exclude=['description'])
        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', include=['description'],
                                                  forjasper=True)

        object_settings['datakey'] = 'datapaths'

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        object_settings['years'], object_settings['yearstr'] = WF.organizePlotYears(object_settings)

        if 'description' not in object_settings.keys():
            object_settings['description'] = ''
        # else:
        #     object_settings['description'] = WF.parseForTextFlags(object_settings['description'])

        for yi, year in enumerate(object_settings['years']):
            useAx = []
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))
            yearstr = object_settings['yearstr'][yi]

            cur_obj_settings = self.Plots.setTimeSeriesXlims(cur_obj_settings, yearstr, object_settings['years'])

            # NOTES
            # Data structure:
            # 2D array of dates[distance from source]
            # array of dates corresponding to the number of the first D of array above
            # supplementary array for distances corrsponding to the second D of array above
            # ex
            # [[1,2,3,5],[2,3,4,2],[5,3,2,5]] #values per date
            # [01jan2016, 04Feb2016, 23May2016] #dates
            # [0, 19, 25, 35] #elevations
            # [0, 19, 25, 35] #top water elevations

            contoursbyID, contoursbyID_settings = self.Data.getReservoirContourDataDictionary(cur_obj_settings)
            contoursbyID = WF.filterDataByYear(contoursbyID, year, extraflag='topwater')
            selectedContourIDs = WF.getUsedIDs(contoursbyID)
            straightlines = self.Data.getStraightLineValue(cur_obj_settings)

            if len(selectedContourIDs) == 1:
                figsize = (12, 6)
                pageformat = 'half'
            else:
                figsize = (12, 12)
                pageformat = 'full'

            if pageformat == 'full':
                height_ratios = []
                for i in range(len(selectedContourIDs)):
                    if i == len(selectedContourIDs) - 1:
                        height_ratios.append(1)
                    else:
                        height_ratios.append(.75)
                fig, axes = plt.subplots(ncols=1, nrows=len(selectedContourIDs), sharex=True, figsize=figsize,
                                         gridspec_kw={'height_ratios': height_ratios})
            else:
                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize,
                                         )

            for IDi, ID in enumerate(selectedContourIDs):
                contour_plot_settings = pickle.loads(pickle.dumps(cur_obj_settings, -1))
                contour_plot_settings = self.configureSettingsForID(ID, contour_plot_settings)
                contours = WF.selectContourByID(contoursbyID, ID)
                contours_settings = WF.selectContourByID(contoursbyID_settings, ID)
                if len(contours.keys()) > 1:
                    WF.print2stdout(f'Too many reservoir keys defined. Using the first, {list(contours.keys())[0]}',
                                    debug=self.debug)
                elif len(contours.keys()) == 0:
                    WF.print2stdout(f'No reservoir keys defined.', debug=self.debug)
                    return
                contour = contours[list(contours.keys())[0]]
                contour_settings = contours_settings[list(contours.keys())[0]]

                values = contour['values']
                elevations = contour['elevations']
                dates = contour['dates']
                topwater = contour['topwater']

                if len(selectedContourIDs) == 1:
                    axes = [axes]

                ax = axes[IDi]

                parameter, contour_plot_settings['param_count'] = WF.getParameterCount(contour_settings, contour_plot_settings)

                if 'units' in contour_plot_settings.keys():
                    units = contour_settings['units']
                else:
                    if 'parameter' in contour_plot_settings.keys():
                        parameter = contour_plot_settings['parameter']
                    else:
                        parameter = ''
                        top_count = 0
                        for key in contour_plot_settings['param_count'].keys():
                            if contour_plot_settings['param_count'][key] > top_count:
                                parameter = key
                    try:
                        units = self.Constants.units[parameter]
                    except KeyError:
                        units = None

                if isinstance(units, dict):
                    if 'unitsystem' in contour_plot_settings.keys():
                        units = units[contour_plot_settings['unitsystem'].lower()]
                    else:
                        units = None

                if 'unitsystem' in contour_plot_settings.keys():
                    values, units = WF.convertUnitSystem(values, units, contour_plot_settings['unitsystem']) #TODO: confirm

                chkvals = WF.checkData(values)
                if not chkvals:
                    WF.print2stdout('Invalid Data settings for contour plot year {0}'.format(year), debug=self.debug)
                    continue

                dates = WT.JDateToDatetime(dates, self.startYear)

                if 'label' not in contour_plot_settings.keys():
                    contour_plot_settings['label'] = ''
                else:
                    contour_plot_settings['label'] = WF.formatTextFlags(contour_plot_settings['label'])

                contour_plot_settings = WD.getDefaultContourSettings(contour_plot_settings, debug=self.debug)

                if 'min' in contour_plot_settings['colorbar']:
                    vmin = float(contour_plot_settings['colorbar']['min'])
                else:
                    vmin = np.nanmin(values)
                if 'max' in contour_plot_settings['colorbar']:
                    vmax = float(contour_plot_settings['colorbar']['max'])
                else:
                    vmax = np.nanmax(values)

                values = WF.filterContourOverTopWater(values, elevations, topwater)

                contr = ax.contourf(dates, elevations, values.T, cmap=contour_plot_settings['colorbar']['colormap'],
                                    vmin=vmin, vmax=vmax,
                                    levels=np.linspace(vmin, vmax, int(contour_plot_settings['colorbar']['bins'])), #add one to get the desired number..
                                    extend='both') #the .T transposes the array so dates on bottom TODO:make extend variable

                # ax.plot(dates, topwater, c='red') #debug topwater
                self.WAT_log.addLogEntry({'type': contour_plot_settings['label'] +
                                          '_ContourPlot' if contour_plot_settings['label'] != '' else 'ContourPlot',
                                          'name': self.ChapterRegion + '_' + yearstr,
                                          'description': contour_plot_settings['description'],
                                          'units': units,
                                          'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                     self.StartTime, self.EndTime,
                                                                                     debug=self.debug).strftime('%d %b %Y'),
                                          'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                                   self.StartTime, self.EndTime,
                                                                                   debug=self.debug).strftime('%d %b %Y'),
                                          'logoutputfilename': 'NA'
                                          },
                                         isdata=True)

                contour_plot_settings = WF.updateFlaggedValues(contour_plot_settings, '%%units%%', WF.formatUnitsStrings(units))

                if 'contourlines' in contour_plot_settings.keys():
                    for contourline in contour_plot_settings['contourlines']:
                        if 'value' in contourline.keys():
                            val = float(contourline['value'])
                        else:
                            WF.print2stdout('No Value set for contour line.', debug=self.debug)
                            continue
                        contourline = WD.getDefaultContourLineSettings(contourline)
                        cs = ax.contour(contr, levels=[val], linewidths=[float(contourline['linewidth'])],
                                        colors=[contourline['linecolor']],
                                        linestyles=contourline['linestylepattern'], alpha=float(contourline['alpha']))
                        if 'contourlinetext' in contourline.keys():
                            if contourline['contourlinetext'].lower() == 'true':
                                ax.clabel(cs, inline_spacing=contourline['inline_spacing'],
                                          fontsize=contourline['fontsize'], inline=contourline['text_inline'])
                        if 'show_in_legend' in contourline.keys():
                            if contourline['show_in_legend'].lower() == 'true':
                                if 'label' in contourline.keys():
                                    label = contourline['label']
                                else:
                                    label = WF.formatTextFlags(str(val))
                                cl_leg = ax.plot([], [], c=contourline['linecolor'], ls=contourline['linestylepattern'],
                                                 alpha=float(contourline['alpha']), lw=float(contourline['linewidth']),
                                                 label=label)

                ### VERTICAL LINES ###
                self.Plots.plotVerticalLines(straightlines, ax, contour_plot_settings, isdate=True)

                ### Horizontal LINES ###
                self.Plots.plotHorizontalLines(straightlines, ax, contour_plot_settings)

                if self.iscomp:
                    if 'modeltext' in contour_plot_settings.keys():
                        modeltext = contour_plot_settings['modeltext']
                    else:
                        modeltext = self.SimulationName
                    plt.text(1.02, 0.5, modeltext, fontsize=12, transform=ax.transAxes, verticalalignment='center',
                             horizontalalignment='center', rotation='vertical')

                if 'gridlines' in contour_plot_settings.keys():
                    if contour_plot_settings['gridlines'].lower() == 'true':
                        ax.grid(True)

                if 'ylabel' in contour_plot_settings.keys():
                    if 'ylabelsize' in contour_plot_settings.keys():
                        ylabsize = float(contour_plot_settings['ylabelsize'])
                    elif 'fontsize' in contour_plot_settings.keys():
                        ylabsize = float(contour_plot_settings['fontsize'])
                    else:
                        ylabsize = 12
                    ylabel = WF.formatTextFlags(contour_plot_settings['ylabel'])
                    ax.set_ylabel(ylabel, fontsize=ylabsize)

                ############# xticks and lims #############

                useplot = self.Plots.formatDateXAxis(ax, contour_plot_settings)
                if not useplot:
                    useAx.append(False)
                else:
                    useAx.append(True)
                xmin, xmax = ax.get_xlim()

                xmin = mpl.dates.num2date(xmin)
                xmax = mpl.dates.num2date(xmax)

                if 'xticks' in contour_plot_settings.keys():
                    xtick_settings = contour_plot_settings['xticks']
                else:
                    xtick_settings = {}
                self.Plots.formatTimeSeriesXticks(ax, xtick_settings, contour_plot_settings)

                ax.set_xlim(left=xmin)
                ax.set_xlim(right=xmax)

                ############# yticks and lims #############
                self.Plots.formatYTicks(ax, contour_plot_settings)

            # #stuff to call once per plot
            self.configureSettingsForID(self.base_id, cur_obj_settings)
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%units%%', WF.formatUnitsStrings(units))

            if 'title' in cur_obj_settings.keys():
                if 'titlesize' in cur_obj_settings.keys():
                    titlesize = float(object_settings['titlesize'])
                elif 'fontsize' in cur_obj_settings.keys():
                    titlesize = float(object_settings['fontsize'])
                else:
                    titlesize = 15
                title = WF.formatTextFlags(cur_obj_settings['title'])
                axes[0].set_title(title, fontsize=titlesize, wrap=True)

            if 'xlabel' in cur_obj_settings.keys():
                if 'xlabelsize' in cur_obj_settings.keys():
                    xlabsize = float(cur_obj_settings['xlabelsize'])
                elif 'fontsize' in cur_obj_settings.keys():
                    xlabsize = float(cur_obj_settings['fontsize'])
                else:
                    xlabsize = 12
                xlabel = WF.formatTextFlags(cur_obj_settings['xlabel'])
                axes[-1].set_xlabel(xlabel, fontsize=xlabsize)

            useplot = self.Plots.formatDateXAxis(axes[-1], cur_obj_settings)
            if not useplot:
                useAx.append(False)
            else:
                useAx.append(True)

            if not any(useAx):
                print(f'Plot for {year} not included due to xlimits.')
                plt.close("all")
                continue

            if 'legend' in cur_obj_settings.keys():
                if cur_obj_settings['legend'].lower() == 'true':
                    if 'legendsize' in cur_obj_settings.keys():
                        legsize = float(cur_obj_settings['legendsize'])
                    elif 'fontsize' in cur_obj_settings.keys():
                        legsize = float(cur_obj_settings['fontsize'])
                    else:
                        legsize = 12
                    if len(axes[0].get_legend_handles_labels()[0]) > 0:
                        plt.legend(fontsize=legsize)

            cbar = plt.colorbar(contr, ax=axes[-1], orientation='horizontal', aspect=50.)
            locs = np.linspace(vmin, vmax, int(contour_plot_settings['colorbar']['numticks']))
            cbar.set_ticks(locs)
            cbar.set_ticklabels(locs.round(2))
            if 'label' in contour_plot_settings['colorbar']:
                if 'labelsize' in contour_plot_settings['colorbar'].keys():
                    labsize = float(contour_plot_settings['colorbar']['labelsize'])
                elif 'fontsize' in cur_obj_settings['colorbar'].keys():
                    labsize = float(cur_obj_settings['colorbar']['fontsize'])
                else:
                    labsize = 12
                cbar_label = WF.formatTextFlags(contour_plot_settings['colorbar']['label'])
                cbar.set_label(cbar_label, fontsize=labsize)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.05)

            basefigname = os.path.join(self.images_path, 'ContourPlot' + '_' + self.ChapterRegion.replace(' ', '_')
                                       + '_' + yearstr)
            exists = True
            tempnum = 1
            tfn = basefigname
            while exists:
                if os.path.exists(tfn + '.png'):
                    tfn = basefigname + '_{0}'.format(tempnum)
                    tempnum += 1
                else:
                    exists = False
            figname = tfn + '.png'

            if self.highres:
                plt.savefig(figname, dpi=300)
            else:
                plt.savefig(figname)
            plt.close('all')

            if pageformat == 'full':
                self.XML.writeFullPagePlot(os.path.basename(figname), cur_obj_settings['description'])
            elif pageformat == 'half':
                self.XML.writeHalfPagePlot(os.path.basename(figname), cur_obj_settings['description'])

        WF.print2stdout(f'Reservoir Contour Plot took {time.time() - objectstarttime} seconds.')

    def makeTextBox(self, object_settings):
        """
        Makes a text box object in the report
        :param object_settings: currently selected object settings dictionary
        :return:
        """

        objectstarttime = time.time()

        if 'text' not in object_settings.keys():
            WF.print2stdout('Failed to input textbox contents using <text> flag.', debug=self.debug)

        if isinstance(object_settings['text'], list):
            texts = object_settings['text']
        else:
            texts = [object_settings['text']]

        if 'iteration_sep' in object_settings.keys():
            iteration_sep = object_settings['iteration_sep']
        else:
            iteration_sep = '\n'

        exclude_programs = []
        if 'exclude' in object_settings.keys():
            exclude_programs = object_settings['exclude']
        if not isinstance(exclude_programs, list):
            exclude_programs = [exclude_programs]

        iteratesimulations = False
        if 'iteratesimulations' in object_settings.keys():
            if object_settings['iteratesimulations'].lower() == 'true':
                iteratesimulations = True

            # If we are doing a comaprison report, the user can specify sections to be repeated for each alt by wrapping the text in {}
        if self.reportType == 'comparison' and not self.modelIndependent:
            for text in texts:
                # object_settings['text'] = WF.parseForTextFlags(object_settings['text'])
                text = WF.parseForTextFlags(text)
            if '{' in text and '}' in text:  # iterate sections
                sections = list(set(re.findall(r'{(.*?)}', text)))
                for section in sections:
                    replace_section = []
                    for ID in self.SimulationVariables.keys():
                        flaggedvals = list(set(re.findall(r'%%(.*?)%%', text)))
                        tmp_text = section
                        for fv in flaggedvals:
                            tmp_text = tmp_text.replace(fv, fv + f'.{ID}')
                        replace_section.append(tmp_text)
                    replace_section = iteration_sep.join(replace_section)
                    text = text.replace(f'{section}', replace_section)
                text = text.replace('{', '')
                text = text.replace('}', '')
            text = WF.replaceAllFlags(self, text)
            self.XML.writeTextBox(text)

        elif self.reportType == 'comparison' and self.modelIndependent and iteratesimulations is True:
            for ID in self.SimulationVariables.keys():  # should be onle 1 simulation at this point.
                out_text = []
                for text in texts:
                    text = WF.parseForTextFlags(text)

                    self.modelIndependent = False  # turn off model independent flag for this iteration, so we can set the model alt var
                    self.loadCurrentID(ID)
                    if '{' in text and '}' in text:  # iterate model alts
                        sections = list(set(re.findall(r'{(.*?)}', text)))
                        replace_section = []
                        for section in sections:
                            for modelalt in self.SimulationVariables[ID]['ModelAlternatives']:
                                self.softLoadModelAlt(modelalt)
                                if self.program.lower() in [n.lower() for n in exclude_programs]:
                                    WF.print2stdout('Skipping model alt {modelalt["name"]} for program {self.program}.',
                                                    debug=self.debug)
                                    continue
                                ma_text = WF.replaceAllFlags(self, section)
                                ma_text += '\n'
                                replace_section.append(ma_text)
                            replace_section = iteration_sep.join(replace_section)
                            text = text.replace(f'{section}', replace_section)
                        text = text.replace('{', '')
                        text = text.replace('}', '')
                    out_text.append(text)

                out_text = iteration_sep.join(out_text)
                out_text = WF.replaceAllFlags(self, out_text)
                self.modelIndependent = True
                self.loadCurrentModelAltID('')  # reset to model independent
                text = out_text
                self.XML.writeTextBox(text)

        elif self.reportType != 'comparison' and self.modelIndependent and iteratesimulations:
            for text in texts:
                # object_settings['text'] = WF.parseForTextFlags(object_settings['text'])
                text = WF.parseForTextFlags(text)
                out_text = []
                for ID in self.SimulationVariables.keys():  # should be only 1 simulation at this point.
                    self.modelIndependent = False  # turn off model independent flag for this iteration, so we can set the model alt var
                    self.loadCurrentID(ID)
                    if '{' in text and '}' in text:  # iterate model alts
                        sections = list(set(re.findall(r'{(.*?)}', text)))
                        replace_section = []
                        for section in sections:
                            for modelalt in self.SimulationVariables[ID]['ModelAlternatives']:
                                self.softLoadModelAlt(modelalt)
                                if self.program.lower() in [n.lower() for n in exclude_programs]:
                                    WF.print2stdout('Skipping model alt {modelalt["name"]} for program {self.program}.', debug=self.debug)
                                    continue
                                ma_text = WF.replaceAllFlags(self, section)
                                ma_text += '\n'
                                replace_section.append(ma_text)
                            replace_section = iteration_sep.join(replace_section)
                            text = text.replace(f'{section}', replace_section)
                        text = text.replace('{', '')
                        text = text.replace('}', '')
                    out_text.append(text)

                    out_text = iteration_sep.join(out_text)
                    out_text = WF.replaceAllFlags(self, out_text)
                    self.modelIndependent = True
                    self.loadCurrentModelAltID('')  #reset to model independent
                    text = out_text
                self.XML.writeTextBox(text)
        else:
            for text in texts:
                text = WF.replaceAllFlags(self, text)
                self.XML.writeTextBox(text)

        WF.print2stdout(f'Text box took {time.time() - objectstarttime} seconds.')

    def makeTableFromFile(self, object_settings):
        """
        Makes a table from a formatted CSV file
        :param object_settings:
        :return:
        """

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Table From File.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)

        default_settings = self.loadDefaultPlotObject('tablefromfile')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', forjasper=True)

        data, data_settings = self.Data.getTableDataDictionary(object_settings, type='formatted')
        object_settings['primarykey'] = self.Data.getPrimaryTableKey(data, object_settings)
        data = self.Tables.formatPrimaryKey(data, object_settings)
        data, data_settings = self.Data.mergeFormattedTables(data, data_settings, object_settings)
        data = self.Data.filterFormattedTable(data, object_settings, primarykey=object_settings['primarykey'])
        headings, rows = self.Tables.buildFormattedTable(data)
        headings = self.Tables.replaceIllegalJasperCharactersHeadings(headings)
        rows = self.Tables.replaceIllegalJasperCharactersRows(rows)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
            # desc = WF.parseForTextFlags(desc)
        else:
            desc = ''

        # ID#|value|value..etc
        table_constructor = {}
        threshold_colors = np.full(len(rows), None)
        for hi, header in enumerate(headings):
            if header == object_settings['primarykey']:
                continue
            tcnum = len(table_constructor.keys())
            formatted_rows = []
            for ri, row in enumerate(rows):
                srow = row.split('|')
                rowfrmt = f'{srow[0]}|{srow[hi]}'
                formatted_rows.append(rowfrmt)
            table_constructor[tcnum] = {}
            table_constructor[tcnum]['rows'] = formatted_rows
            table_constructor[tcnum]['thresholdcolors'] = threshold_colors
            table_constructor[tcnum]['header'] = header

        if len(table_constructor) == 0:
            self.Tables.writeMissingTableWarning(desc)
            WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)
        else:
            self.XML.writeTableStart(desc, object_settings['primarykey'])
            self.Tables.writeTable(table_constructor)
        WF.print2stdout(f'Table from file took {time.time() - objectstarttime} seconds.')

    def makeForecastTable(self, object_settings):
        """
        Makes a table for forecast runs
        Can look 1 of two ways.
            1. If member iteration is turned on, it will be for a single forecast member.
            member number | Operations Name | Met Name | TempTargetName
            2. If it's off, that line will be repeated for each available member (unless specified).
        Order of rows and contents is also variable.
        :param object_settings:
        :return:
        """

        if self.reportType != 'forecast':
            WF.print2stdout('### WARNING ###')
            WF.print2stdout('Forecast Table object only available for Iterative Forecast report.')
            WF.print2stdout(f'Incompatible with selected report type: {self.reportType}')
            WF.print2stdout('Continuing without table.')
            return

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Forecast Table')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)

        default_settings = self.loadDefaultPlotObject('forecasttable')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()],
                                                     object_settings)

        object_settings = WF.replaceflaggedValues(self, object_settings, 'fancytext', forjasper=True)

        # Update with anything in general or model specific
        object_settings = WF.replaceflaggedValues(self, object_settings, 'general')
        object_settings = WF.replaceflaggedValues(self, object_settings, 'modelspecific')

        if 'description' in object_settings.keys():
            desc = object_settings['description']
            # desc = WF.parseForTextFlags(desc)
        else:
            desc = ''

        # Get members to plot
        members_to_plot = []
        if 'members' in object_settings.keys():
            members_to_plot = [int(n) for n in object_settings['members']]
        else:
            if self.memberiteration: #if we are looping through each member, we only want to table the one member, unless specified
                if self.multiMemberSection:
                    members_to_plot = self.membersInSection
                else:
                    members_to_plot = [self.member]
                desc = WF.updateFlaggedValues(desc, '%%member%%', WF.getOriginalMemberNumber(self.member, WF.matchMemberToEnsembleSet(self.ensembleSets, self.member), self.DSSFile,
                                                                                             self.alternativeFpart, self.StartTime, self.EndTime, self.debug))

            else:
                members_to_plot = self.allMembers

        # get columns/order, most likely using a default
        if 'headers' in object_settings.keys():
            headers = object_settings['headers']
        else:
            headers = WD.getDefaultDefaultForecastTableHeaders()
        headers = self.Tables.confirmForecastTableHeaders(headers)

        formatted_headers = self.Tables.formatForecastTableHeaders(headers)
        primarykey = headers[0]

        table_constructor = {}
        for ci, header in enumerate(headers[1:]):  #first column will be done automatically
            rows = []
            for mi, member in enumerate(members_to_plot):
                row = ''
                ensembleset = WF.matchMemberToEnsembleSet(self.ensembleSets, member)
                if len(ensembleset.keys()) == 0:  # if for some reason, we cant find the matching ensemble set? should never happen.
                    continue
                if primarykey == 'member':
                    primaryitem = WF.getOriginalMemberNumber(member, ensembleset, self.DSSFile, self.alternativeFpart, self.StartTime, self.EndTime, self.debug)
                else:
                    primaryitem = ensembleset[primarykey]
                row += primaryitem
                row += '|'
                if header == 'member':
                    row += WF.getOriginalMemberNumber(member, ensembleset, self.DSSFile, self.alternativeFpart, self.StartTime, self.EndTime, self.debug)
                else:
                    row += ensembleset[header]
                rows.append(row)
            table_constructor[ci] = {}
            table_constructor[ci]['rows'] = rows

            table_constructor[ci]['thresholdcolors'] = np.full(len(rows), None) #not used but needed to build out the table
            table_constructor[ci]['header'] = formatted_headers[ci+1] #add 1 becuase we skip the first

        self.XML.writeTableStart(desc, formatted_headers[0], limit=True)
        self.Tables.writeTable(table_constructor)
        WF.print2stdout(f'Forecast Table from file took {time.time() - objectstarttime} seconds.')

    def setSimulationCSVVars(self, simulationCSV):
        """
        Set variables pertaining to a specified simulation.
        :param simulationCSV: dictionary of specified simulation
        :return: class variables
                    self.program
                    self.modelAltName

        """
        if simulationCSV['deprecated_method']:
            self.programs = simulationCSV['programs']
            self.reportXML_modelAltNames = simulationCSV['modelaltnames']
            self.reportXMLFile = simulationCSV['xmlfile']
        else:
            self.programs = simulationCSV['programs']
            self.reportXMLFile = simulationCSV['xmlfile']
            self.reportXML_Keywords = simulationCSV['keywords']

    def setSimulationVariables(self, simulation):
        """
        Sets various class variables for selected variable
        Sets simulation dates and times
        :param simulation: simulation dictionary object for specified simulation
        :return: class variables
                    self.Data.Memory
                    self.SimulationName
                    self.baseSimulationName
                    self.simulationDir
                    self.DSSFile
                    self.StartTimeStr
                    self.EndTimeStr
                    self.LastComputed
                    self.ModelAlternatives
        """

        # self.Data.Memory = {}
        ID = simulation['ID']
        self.SimulationVariables[ID] = {}
        self.SimulationVariables[ID]['SimulationName'] = simulation['name']
        self.SimulationVariables[ID]['baseSimulationName'] = simulation['basename']
        simulation_dir = simulation['directory']  #todo: remove when mark fixes xml output
        simulation_dir_split = simulation_dir.split(os.path.sep)
        simulation_list = [s for s in simulation_dir_split if s != '']
        simulation_dir = os.path.sep.join(simulation_list)
        self.SimulationVariables[ID]['simulationDir'] = simulation_dir
        # self.SimulationVariables[ID]['simulationDir'] = simulation['directory']
        self.SimulationVariables[ID]['DSSFile'] = simulation['dssfile']
        self.SimulationVariables[ID]['StartTimeStr'] = simulation['starttime']
        self.SimulationVariables[ID]['EndTimeStr'] = simulation['endtime']
        self.SimulationVariables[ID]['LastComputed'] = simulation['lastcomputed']
        self.SimulationVariables[ID]['ModelAlternatives'] = simulation['modelalternatives']
        self.SimulationVariables[ID]['Description'] = simulation['Description']
        self.SimulationVariables[ID]['AnalysisPeriod'] = simulation['AnalysisPeriod']
        self.SimulationVariables[ID]['WatAlternative'] = simulation['WatAlternative']
        if self.reportType == 'forecast':
            self.SimulationVariables[ID]['ensemblesets'] = simulation['ensemblesets']
        else:
            self.SimulationVariables[ID]['ensemblesets'] = []

        WT.setSimulationDateTimes(self, ID)

    def organizeMembers(self):
        """
        Formats members part of an ensemble set and gets a list of all members
        :return:
        """

        self.allMembers = []
        for simulation in self.Simulations:
            simulation['ensemblesets'] = self.formatMembers(simulation['ensemblesets'])
            for ensembleset in simulation['ensemblesets']:
                for member in ensembleset['members']:
                    self.allMembers.append(member)
        self.allMembers.sort()

    def formatMembers(self, ensemblesets):
        """
        Formats members as part of a collection set. Takes the collection start and adds to the member number
        ex: member 4, collection start 5000, return 5004
        :param ensemblesets: dictionary for ensemble sets
        :return: formatted ensemble set dictionaries
        """

        formatted_ensemblesets = []
        for ensembleset in ensemblesets:
            members = ensembleset['memberstoreport']
            collectionstart = int(ensembleset['collectionsstart'])
            members = [int(n.strip()) for n in members.split(',')]
            members_formatted = []
            for member in members:
                members_formatted.append(member + collectionstart)
            ensembleset['members'] = members_formatted
            formatted_ensemblesets.append(ensembleset)
        return formatted_ensemblesets

    def getLineSettings(self, LineSettings, Flag):
        """
        Gets the correct line settings for the selected flag
        :param LineSettings: dictionary of settings
        :param Flag: selected flag to match line
        :return: deep copy of line
        """

        for line in LineSettings:
            if Flag == line['flag']:
                return pickle.loads(pickle.dumps(line, -1))

    def getPlotLabelMasks(self, idx, nprofiles, cols):
        """
        Gets plot label masks
        :param idx: page index
        :param nprofiles: number of profiles
        :param cols: number of columns
        :return: boolean fields for plotting
        """

        if idx >= nprofiles - cols:
            add_xlabel = True
        else:
            add_xlabel = False
        if idx % cols == 0:
            add_ylabel = True
        else:
            add_ylabel = False

        return add_xlabel, add_ylabel

    def getPlotParameter(self, object_settings):
        """
        Gets the plot parameters based on user settings. If explicitly stated, uses that. Otherwise, looks at the
        defined parameters in the linedata and grabs the most common one.
        :param object_settings: currently selected object settings dictionary
        :return: plot parameter if possible, otherwise None
        """

        if 'parameter' in object_settings.keys():
            plot_parameter = object_settings['parameter']
        else:
            params = []
            for line in object_settings['lines']:
                if 'parameter' in line.keys():
                    params.append(line['parameter'])
            if len(list(set(params))) == 1:
                plot_parameter = params[0]
            else:
                plot_parameter = None
        return plot_parameter

    def writeXMLIntroduction(self):
        """
        Writes the intro section for XML file. Creates a line in the intro for each model used
        """

        self.XML.writeIntroStart()
        for model in self.reportCSV.keys():
            # program = self.XML.writeIntroLine(self.reportCSV[model]['program'])
            self.XML.writeIntroLine('%%REPLACEINTRO_{0}%%'.format(model))
        self.XML.writeIntroEnd()

    def writeChapter(self):
        """

        Writes each chapter defined in the simulation CSV file to the XML file during initialization.

        This function:
        - Iterates through ChapterDefinitions(), extracting and setting chapter-specific attributes.
        - Calls writeSections() to process the sections of the chapter,
          which in turn calls iterateSections() to generate plots and figures.

        :return: Updates class variables

        """

        for Chapter in self.ChapterDefinitions:
            self.ChapterName = Chapter['name']
            self.ChapterRegion = Chapter['region']
            self.ChapterText = Chapter['grouptext']
            self.ChapterResolution = Chapter['resolution']
            self.debug_boolean = Chapter['debug']
            self.memberiteration_boolean = Chapter['memberiteration']
            self.groupmembers_boolean = Chapter['groupmembers']

            self.debug = False
            if self.debug_boolean.lower() == 'true':
                self.debug = True
                WF.print2stdout('Verbose mode activated!')
            else:
                WF.print2stdout('Quiet mode activated.')

            self.highres = True  #default
            if self.ChapterResolution.lower() == 'high':
                self.highres = True
                WF.print2stdout('Running High Res Mode!')
            elif self.ChapterResolution.lower() == 'low':
                self.highres = False
                WF.print2stdout('Running Low Res Mode!')

            self.memberiteration = False
            if self.memberiteration_boolean.lower() == 'true':
                self.memberiteration = True
                WF.print2stdout('member iteration mode activated!')
            else:
                WF.print2stdout('member iteration mode deactivated.', debug=self.debug)

            self.groupmembers = False
            if self.groupmembers_boolean.lower() == 'true':
                self.groupmembers = True
                WF.print2stdout('group member mode activated!')
            else:
                WF.print2stdout('group member mode deactivated.', debug=self.debug)

            self.WAT_log.addLogEntry({'region': self.ChapterRegion})
            self.XML.writeChapterStart(WF.replaceflaggedValues(self, self.ChapterName, 'fancytext', forjasper=True),
                                       WF.replaceflaggedValues(self, self.ChapterText, 'fancytext', forjasper=True))

            self.writeSections(Chapter)  # Triggers iterateSections for plot and table creation

            WF.print2stdout('\n################################')
            WF.print2stdout('Chapter Complete.')
            WF.print2stdout('################################\n')
            self.XML.writeChapterEnd()

    def writeSections(self, Chapter):
        """
        Processes sections for a chapter. If `self.memberiteration` is True, it loops
        through ensembles and members, handling a second pass for non-identical members.
        Otherwise, processes sections directly.

        :param Chapter: Dictionary with section details from the XML file.
        """

        for section in Chapter['sections']:
            # Reset variables for each section
            section_header = section['header']
            section_header = WF.replaceFlaggedValue(self, section_header, 'fancytext', forjasper=True)
            self.multiMemberSection = False
            if self.memberiteration:
                for ensemble in self.ensemblesets:
                    for member in ensemble['members']:
                        self.member = int(member)
                        if self.groupmembers:
                            self.membersInSection, section = self.preCheckSectionForDuplicates(section) #check for section being multi-member
                            if len(self.membersInSection) > 0:
                                self.multiMemberSection = True
                            else:
                                self.multiMemberSection = False

                            if self.multiMemberSection:
                                member_already_used = self.checkForMemberUsed(member, self.membersInSection)
                                if member_already_used:
                                    WF.print2stdout(f'Member {member} already used in previous section. Skipping.')
                                    continue

                                # get the original member numbs for each in the section
                                member_list_frmt = ', '.join([WF.getOriginalMemberNumber(mem, WF.matchMemberToEnsembleSet(self.ensembleSets, mem), self.DSSFile,
                                                                                         self.alternativeFpart, self.StartTime, self.EndTime, self.debug) for mem in self.membersInSection])

                                # this will cover anything in 'general' or 'modelspecific' which includes SimulationName
                                new_section_header = WF.replaceflaggedValues(self, section_header, 'modelspecific')
                                new_section_header = WF.replaceflaggedValues(self, new_section_header, 'general')
                                # replace met name
                                new_section_header = WF.updateFlaggedValues(new_section_header, '%%metname%%', ensemble['metname'])
                                # replace member number list
                                new_section_header = WF.updateFlaggedValues(new_section_header, '%%member%%', member_list_frmt)

                                self.original_section_header = new_section_header
                                self.XML.writeSectionHeader(new_section_header)
                                self.Ensemble = ensemble
                                self.iterateMultiMemberSection(section)
                                self.XML.writeSectionHeaderEnd()

                            else:
                                # this will cover anything in 'general' or 'modelspecific' which includes SimulationName
                                new_section_header = WF.replaceflaggedValues(self, section_header, 'modelspecific')
                                new_section_header = WF.replaceflaggedValues(self, new_section_header, 'general')
                                # replace met name
                                new_section_header = WF.updateFlaggedValues(new_section_header, '%%metname%%', ensemble['metname'])
                                # replace member number, do not format
                                new_section_header = WF.updateFlaggedValues(new_section_header, '%%member%%', WF.getOriginalMemberNumber(member, ensemble, self.DSSFile,
                                                                                                                                         self.alternativeFpart, self.StartTime, self.EndTime, self.debug))

                                self.original_section_header = new_section_header
                                self.XML.writeSectionHeader(new_section_header)
                                self.Ensemble = ensemble
                                self.iterateSection(section)
                                self.XML.writeSectionHeaderEnd()

                        else:
                            # this will cover anything in 'general' or 'modelspecific' which includes SimulationName
                            new_section_header = WF.replaceflaggedValues(self, section_header, 'modelspecific')
                            new_section_header = WF.replaceflaggedValues(self, new_section_header, 'general')
                            # replace met name
                            new_section_header = WF.updateFlaggedValues(new_section_header, '%%metname%%', ensemble['metname'])
                            # replace member number, do not format
                            new_section_header = WF.updateFlaggedValues(new_section_header, '%%member%%', WF.getOriginalMemberNumber(member, ensemble, self.DSSFile,
                                                                                                                                     self.alternativeFpart, self.StartTime, self.EndTime, self.debug))

                            self.original_section_header = new_section_header
                            self.XML.writeSectionHeader(new_section_header)
                            self.Ensemble = ensemble
                            self.iterateSection(section)
                            self.XML.writeSectionHeaderEnd()

            else:
                self.XML.writeSectionHeader(section_header)
                self.iterateSection(section)
                self.XML.writeSectionHeaderEnd()

    def iterateSection(self, section):
        """
        Iterates through objects in a section to build tables and plots
        :param section: dictionary read from XML file
        """

        for object in section['objects']:
            objtype = object['type'].lower()
            self.makeObject(object, objtype)

    def iterateMultiMemberSection(self, section):
        for object in section['objects']:
            objtype = object['type'].lower()
            if 'isduplicate' in object.keys():
                if object['isduplicate']:
                    self.makeObject(object, objtype)
                else:
                    original_member = self.member
                    for target_member in self.membersInSection:
                        self.member = target_member
                        self.makeObject(object, objtype)
                    self.member = original_member
            else:
                #shouldnt be here, but just in case..
                WF.print2stderr('WARNING: Object found to be multimember but not marked for duplicate TRUE/FALSE')
                self.makeObject(object, objtype)



    def makeObject(self, object, objtype):
        if objtype == 'timeseriesplot':
            self.makeTimeSeriesPlot(object)
        elif objtype == 'profileplot':
            self.makeProfilePlot(object)
        elif objtype == 'errorstatisticstable':
            self.makeErrorStatisticsTable(object)
        elif objtype == 'monthlystatisticstable':
            self.makeMonthlyStatisticsTable(object)
        elif objtype == 'profilestatisticstable':
            self.makeProfileStatisticsTable(object)
        elif objtype == 'contourplot':
            self.makeContourPlot(object)
        elif objtype == 'reservoircontourplot':
            self.makeReservoirContourPlot(object)
        elif objtype == 'singlestatistictable':
            self.makeSingleStatisticTable(object)
        elif objtype == 'singlestatisticprofiletable':
            self.makeSingleStatisticProfileTable(object)
        elif objtype == 'textbox':
            self.makeTextBox(object)
        elif objtype == 'tablefromfile':
            self.makeTableFromFile(object)
        elif objtype == 'forecasttable':
            self.makeForecastTable(object)
        else:
            WF.print2stdout('Section Type {0} not identified.'.format(objtype))
            WF.print2stdout('Skipping Section..')

        WF.print2stdout(f'##### Exiting Iterate Sections from object type: {objtype} #####')

    def cleanOutputDirs(self):
        """
        Cleans the images output directory, so png's from old reports aren't mistakenly added to new reports.
        Creates directory if it doesn't exist.
        """

        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
        if not os.path.exists(self.CSVPath):
            os.makedirs(self.CSVPath)

        WF.cleanOutputDirectory(self.images_path, '.png')
        WF.cleanOutputDirectory(self.images_path, '.csv')
        WF.cleanOutputDirectory(self.CSVPath, '.csv')

    def loadModelAlts(self, csvChapterSettings):
        """
        Loads info for specified model alts.
        Loads correct model program class from WDR
        :param csvChapterSettings: simulation alt dict object from self.simulation.
        :return: class variables
                self.alternativeFpart
                self.alternativeDirectory
                self.ModelAlt - WDR class that is program specific
        """

        self.accepted_IDs = []  #IDs are tied to Model Alts in Simulations
        self.modelIndependent = False

        if len(csvChapterSettings['programs']) == 0:
            # you can have chapters/sections in the report that are not tied to models, either of text of non-model results
            # such as boundary conditions or something
            WF.print2stdout('Model independent plotting occurring. If this is a mistake, check the input CSV for missing flags.')
            self.modelIndependent = True

        else:
            csv_programs = [n.lower() for n in csvChapterSettings['programs']]
            if csvChapterSettings['deprecated_method']:  #new method does not care about this
                csv_modelaltnames = [n.lower() for n in csvChapterSettings['modelaltnames']]
            for ID in self.SimulationVariables.keys():  #for each simulation
                if csvChapterSettings['deprecated_method']:  #TODO: remove this far enough into the future
                    approved_modelalts = [modelalt for modelalt in self.SimulationVariables[ID]['ModelAlternatives']
                                          if modelalt['name'].lower() in csv_modelaltnames and
                                          modelalt['program'].lower() in csv_programs]
                    if len(approved_modelalts) == 0:
                        # This chapter does not apply to this simulation if no model alts have the correct program. do not consider.
                        continue

                    approved_modelalt = approved_modelalts[0]  #if there are many, just do the first


                else:
                    # First, for each simulation, figure out which model alts per sim work
                    approved_modelalts = [modelalt for modelalt in self.SimulationVariables[ID]['ModelAlternatives']
                                          if modelalt['program'].lower() in csvChapterSettings['programs']]

                    if len(approved_modelalts) == 0:
                        # This chapter does not apply to this simulation if no model alts have the correct program. do no consider.
                        continue

                    if len(approved_modelalts) > 1:  # Now, try and filter by keywords
                        keyword_approved_modelalts = []
                        keywords = csvChapterSettings['keywords']
                        for keyword in keywords:  #for each keyword
                            for modelalt in approved_modelalts:  #for each already approved model alt
                                if keyword in modelalt['name'].lower():
                                    keyword_approved_modelalts.append(modelalt)
                        if len(keyword_approved_modelalts) > 0:  # if any worked, none will if there are no keyword or none that apply
                            approved_modelalts = keyword_approved_modelalts

                    if len(approved_modelalts) > 1:  #try and filter by order now
                        try:
                            approved_modelalts = [approved_modelalts[csvChapterSettings['numtimesprogramused']-1]] #starts at 1
                        except IndexError:
                            WF.print2stdout(f'Unable to confidently choose model alt for ID {ID}')
                            WF.print2stdout(f'Using model alt: {approved_modelalts[-1]["name"]} for {csvChapterSettings["xmlfile"]}')
                            WF.print2stdout('If this is incorrect, use keyword(s) in the CSV file to narrow down the selection.')
                            approved_modelalt = approved_modelalts[-1]

                    if len(approved_modelalts) == 1:
                        approved_modelalt = approved_modelalts[0]

                WF.print2stdout('Added {0} for ID {1}'.format(approved_modelalt['program'], ID))
                self.SimulationVariables[ID]['alternativeFpart'] = approved_modelalt['fpart']
                self.SimulationVariables[ID]['alternativeDirectory'] = approved_modelalt['directory']
                self.SimulationVariables[ID]['modelAltName'] = approved_modelalt['name']
                self.SimulationVariables[ID]['program'] = approved_modelalt['program']
                self.SimulationVariables[ID]['modelAltDesc'] = approved_modelalt['description']

                if self.SimulationVariables[ID]['program'].lower() == "ressim":
                    self.SimulationVariables[ID]['ModelAlt'] = WRSS.ResSim_Results(
                        self.SimulationVariables[ID]['simulationDir'],
                        self.SimulationVariables[ID]['alternativeFpart'],
                        self.StartTime, self.EndTime, self)
                elif self.SimulationVariables[ID]['program'].lower() == 'cequalw2':
                    self.SimulationVariables[ID]['ModelAlt'] = WW2.W2_Results(
                        self.SimulationVariables[ID]['simulationDir'],
                        self.SimulationVariables[ID]['modelAltName'],
                        self.SimulationVariables[ID]['alternativeDirectory'],
                        self.StartTime, self.EndTime, self)
                else:
                    self.SimulationVariables[ID]['ModelAlt'] == 'unknown'

                self.accepted_IDs.append(ID)
                self.modelIndependent = False

            if not self.modelIndependent:
                if len(self.accepted_IDs) == 0: #if we accept no IDs
                    WF.print2stderr('Incompatible input information from the WAT XML output file ({0}))'.format(self.simulationInfoFile))
                    WF.print2stderr('Please confirm inputs and run again.')
                    WF.print2stdout('CSV Programs: {0}'.format(csvChapterSettings['programs']))
                    WF.print2stdout('Simulation Programs: {0}'.format([n['program'] for n in self.SimulationVariables[ID]['ModelAlternatives']]))
                    if csvChapterSettings['deprecated_method'] == True:
                        WF.print2stdout('CSV Model Alt Names: {0}'.format(csvChapterSettings['modelaltnames']))
                        WF.print2stdout('Simulation Programs: {0}'.format([n['name'] for n in self.SimulationVariables[ID]['ModelAlternatives']]))
                    if self.reportType == 'comparison':
                        WF.print2stderr('If comparison plot, ensure that all programs are in the first cell line CSV file')
                        WF.print2stderr('Example line: CeQualW2|Ressim, format/Shasta_ResSim_TCD_comparison.XML')
                    WF.print2stderr('Now Exiting...')
                    sys.exit(1)

    def loadCurrentID(self, ID):
        """
        Loads model specific settings for a given ID
        :param ID: selected ID, such as 'base' or 'alt_1'
        """

        self.currentlyloadedID = ID
        self.SimulationName = self.SimulationVariables[ID]['SimulationName']
        self.baseSimulationName = self.SimulationVariables[ID]['baseSimulationName']
        self.SimulationDir = self.SimulationVariables[ID]['simulationDir']
        self.DSSFile = self.SimulationVariables[ID]['DSSFile']
        self.StartTimeStr = self.SimulationVariables[ID]['StartTimeStr']
        self.EndTimeStr = self.SimulationVariables[ID]['EndTimeStr']
        self.LastComputed = self.SimulationVariables[ID]['LastComputed']
        self.ModelAlternatives = self.SimulationVariables[ID]['ModelAlternatives']
        self.StartTime = self.SimulationVariables[ID]['StartTime']
        self.EndTime = self.SimulationVariables[ID]['EndTime']
        self.ensemblesets = self.SimulationVariables[ID]['ensemblesets']
        self.SimulationDescription = self.SimulationVariables[ID]['Description']
        if 'AnalysisPeriod' in self.SimulationVariables[ID].keys():
            self.AnalysisPeriod = self.SimulationVariables[ID]['AnalysisPeriod']
        if 'WatAlternative' in self.SimulationVariables[ID].keys():
            self.WatAlternative = self.SimulationVariables[ID]['WatAlternative']

    def softLoadModelAlt(self, modelalt):
        self.alternativeFpart = modelalt['fpart']
        self.modelAltName = modelalt['name']
        self.ModelAlt = modelalt
        self.ModelAltDescription = modelalt['description']
        self.program = modelalt['program']

    def loadCurrentModelAltID(self, ID):
        """
        Loads model alternative specific settings for a given ID
        :param ID: selected ID, such as 'base' or 'alt_1'
        """

        if not self.modelIndependent:
            self.alternativeFpart = self.SimulationVariables[ID]['alternativeFpart']
            self.alternativeDirectory = self.SimulationVariables[ID]['alternativeDirectory']
            self.modelAltName = self.SimulationVariables[ID]['modelAltName']
            self.program = self.SimulationVariables[ID]['program']
            self.ModelAlt = self.SimulationVariables[ID]['ModelAlt']
            self.ensembleSets = self.SimulationVariables[ID]['ensemblesets']
            self.ModelAltDescription = self.SimulationVariables[ID]['modelAltDesc']

            # WF.print2stdout('Model {0} Loaded'.format(ID), debug=self.debug) #noisy
        else:
            self.alternativeFpart = 'none'
            self.alternativeDirectory = 'none'
            self.modelAltName = 'none'
            self.program = 'none'
            self.ModelAlt = 'none'
            self.ModelAltDescription = None  #todo: why are these others texts?

    def initializeXML(self):
        """
        Creates a new version of the template XML file, initiates the XML class, and writes the cover page
        :return: sets class variables
                    self.XML
        """

        # new_xml = os.path.join(self.studyDir, 'reports', 'Datasources', 'USBRAutomatedReportOutput.xml') #required name for file
        new_xml = os.path.join(self.outputDir, 'Datasources', 'USBRAutomatedReportOutput.xml')  #required name for file
        print(f'Creating new XML at {new_xml}\n')

        self.XML = WXMLU.XMLReport(new_xml)
        if self.reportType == 'forecast':
            self.XML.writeCover('DRAFT Temperature Forecast Summary Report')
        elif self.reportType == 'alternativecomparison':
            self.XML.writeCover('DRAFT Temperature Validation Comparison Report')
        else:
            self.XML.writeCover('DRAFT Temperature Validation Summary Report')

    def initializeDataOrganizer(self):
        """
        Creates Data_Memory dictionary
        """

        self.Data = WDO.DataOrganizer(self)

    def initSimulationDict(self):
        """
        Creates simulationVariables dictionary
        """

        self.SimulationVariables = {}

    def loadDefaultPlotObject(self, plotobject):
        """
        Loads the graphic default options.
        :param plotobject: string specifying the default graphics object
        :return:
            plot_info: dict of object settings
        """

        if plotobject in self.graphicsDefault.keys():
            plot_info = pickle.loads(pickle.dumps(self.graphicsDefault[plotobject], -1))
        else:
            WF.print2stdout(f'Plotting object {plotobject} not found in graphics default file.', debug=self.debug)
            plot_info = {}
        return plot_info

    def getLineModelType(self, Line_info):
        """
        Attempts to figure out which model the data is for based off the model source as an attempt to filter out data
        retrieval attempts
        :param Line_info: dictionary of settings for line
        :return: model program name if possible
        """

        for var, ident in self.Constants.model_specific_vars.items():
            if var in Line_info.keys():
                return ident

        return 'undefined'  #no id either way.

    def appendXMLModelIntroduction(self, simorder):
        """
        Fixes intro in XML that shows which models are used for each region.
        Updates a flag with used models.
        :param simorder: number of simulation file
        :return:
        """

        if not self.modelIndependent:
            modelstrs = []
            for Chapter in self.ChapterDefinitions:
                chapname = Chapter['name']
                outstr = f'<Model ModelOrder="%%modelOrder%%" >{chapname}:'
                for cnt, ID in enumerate(self.accepted_IDs):
                    if cnt > 0:
                        outstr += ','
                    outstr += ' {0}'.format(self.SimulationVariables[ID]['program'])
                outstr += '</Model>\n'
                modelstrs.append(outstr)

            lastline = '%%REPLACEINTRO_{0}%%'.format(simorder)
            for i, ms in enumerate(modelstrs):
                tmpstr = ms.replace('%%modelOrder%%', str(self.modelOrder))
                self.XML.insertAfter(lastline, tmpstr)
                self.modelOrder += 1
                lastline = tmpstr

    def fixXMLModelIntroduction(self):
        """
        Removes extra flag values from the report introduction
        """

        self.XML.removeLine('%%REPLACEINTRO_')

    def checkModelType(self, line_info):
        """
         Checks to see if current data path configuration is congruent with currently loaded model ID.
        :param line_info: selected line or datapath
        :return: boolean
        """

        modeltype = self.getLineModelType(line_info)
        if modeltype == 'undefined':
            return True
        if modeltype.lower() != self.program.lower():
            return False
        return True

    def configureSettingsForID(self, ID, settings):
        """
        Loads settings for selected run ID. Mainly for comparison plots.
        Then replaces model specific flags in settings using loaded variables
        :param ID: selected ID, aka 'base' or 'alt_1'
        :param settings: dictionary of settings possibly containing flags to replace
        :return: settings with updated flags. also flags such as self.baseSimulationName are updated to current ID
        """

        self.loadCurrentID(ID)
        self.loadCurrentModelAltID(ID)
        settings = WF.replaceflaggedValues(self, settings, 'modelspecific')
        return settings

    def preCheckSectionForDuplicates(self, section):
        '''
        Check if any objects in the section are duplicates. If so, mark them as such and return a list of all members
        list of members then goes to create a group section for those members
        :param section: dict with info about each report section
        :return:
        list of duplicate members
        new section with flag confirming if duplicate or not
        '''

        duplicate_members = []
        objects_formated = []
        #check if we even need to do this, aka if any objects are duplicates and collect all members
        for oi, object in enumerate(section['objects']):
            if 'lines' in object.keys() or 'datapaths' in object.keys(): #for plots
                linedata, line_settings = self.Data.getTimeSeriesDataDictionary(object)
                if self.Data.checkForDuplicateObject(line_settings, self.member): #check if anything is fully duplicate
                    # islowest = self.Data.checkForLowestMember(line_settings, self.member)
                    # if islowest: #if not lowest, then we probably already did this
                    for member in [self.member] + self.Data.getOtherMembers(line_settings, self.member):
                        if member not in duplicate_members:
                            duplicate_members.append(member)
                    object['isduplicate'] = True
                else:
                    object['isduplicate'] = False
                objects_formated.append(object)
            else:
                objects_formated.append(object)
        section['object'] = objects_formated
        return duplicate_members, section

    def checkForMemberUsed(self, member, membersInSection):
        '''
        checks if the current member is the smallest in the batch. We generally only plot on the first member, if we are
        doing member iterations and grouping them
        :param member: current member
        :param membersInSection: all members in section
        :return: boolean
        '''

        if member != min(membersInSection): #we do everything on the first grouped member, so skip if were not them
            return True
        return False







if __name__ == '__main__':

    if '--version' in sys.argv:
        WF.print2stdout(VERSIONNUMBER)
        sys.exit(0)
    else:
        rundir = sys.argv[0]
        simInfoFile = sys.argv[1]

        # import cProfile
        # ar = cProfile.run('MakeAutomatedReport(simInfoFile, rundir)')

        try:
            MakeAutomatedReport(simInfoFile, rundir)
        except:
            WF.print2stderr(traceback.format_exc())
            sys.exit(1)
