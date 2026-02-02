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

class XMLReport(object):

    def __init__(self, XML_fn):
        '''
        init function for XML class. Makes XML file and primes the counters to keep track of figures and such
        :param XML_fn: name and path of desired XML file
        '''

        self.XML_fn = XML_fn
        self.makeXML()
        self.primeCounters()

#########################################################################################
                            #Main functions#
#########################################################################################

    def makeXML(self):
        '''
        creates and writes a fresh XML file for report
        '''

        with open(self.XML_fn, 'w') as XML:
            XML.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            XML.write('<USBR_Automated_Report>\n')

    def replaceinXML(self, StringToReplace, StringReplacing):
        '''
        replaces strings in XML file, mainly used for the run types in the intro
        :param StringToReplace: string that is going to be replaced
        :param StringReplacing: string to replace it with
        '''

        xml_text = []
        with open(self.XML_fn, 'r') as XML:
            for line in XML:
                if StringToReplace in line:
                    xml_text.append(line.replace(StringToReplace, StringReplacing))
                else:
                    xml_text.append(line)

        with open(self.XML_fn, 'w') as XML:
            for line in xml_text:
                XML.write(line)

    def insertAfter(self, StringToAddAfter, StringToAdd):
        '''
        adds a string after a target string
        :param StringToAddAfter: string to look for to add text after
        :param StringToAdd: string to add in after target string
        '''

        xml_text = []
        with open(self.XML_fn, 'r') as XML:
            for line in XML:
                if StringToAddAfter in line:
                    xml_text.append(line)
                    xml_text.append(StringToAdd)
                else:
                    xml_text.append(line)

        with open(self.XML_fn, 'w') as XML:
            for line in xml_text:
                XML.write(line)

    def removeLine(self, inputStr):
        '''
        removes target string from XML document
        :param inputStr: string to be removed
        '''

        xml_text = []
        with open(self.XML_fn, 'r') as XML:
            for line in XML:
                if inputStr in line:
                    continue
                else:
                    xml_text.append(line)
        with open(self.XML_fn, 'w') as XML:
            for line in xml_text:
                XML.write(line)

    def primeCounters(self):
        '''
        sets up counters for figures tables and sections. Some start at 0, some start at 1
        '''

        self.current_fig_num = 1
        self.current_table_num = 1
        self.current_reportelem_num = 0
        self.current_model_num = 0
        self.current_reportgroup_num = 0

    def writeCover(self, title):
        '''
        writes the cover for the report.
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Cover_Page><Title>{0}</Title></Cover_Page>\n'.format(title))

    def writeIntroStart(self):
        '''
        writes the start of the introduction section
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Group ReportGroupOrder="{0}" ReportGroupName="Introduction">\n'.format(self.current_reportgroup_num))
            XML.write('<Report_Subgroup ReportSubgroupOrder="0">\n')
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Introduction">\n'.format(self.current_reportelem_num))

        self.current_reportgroup_num += 1
        self.current_reportelem_num += 1

    def writeIntroLine(self, program):
        '''
        writes line in intro for a given program
        :param program: name of program to be added to model order
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Model ModelOrder="0" >{1}</Model>\n'.format(self.current_model_num, program))
        self.current_model_num += 1

    def writeIntroEnd(self):
        '''
        writes the end block for the intro file
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('</Report_Element>\n')
            XML.write('</Report_Subgroup>\n')
            XML.write('</Report_Group>\n')

    def writeChapterStart(self, ChapterName, ChapterText):
        '''
        writes the starting chunk of an individual chapter. chapter names automatically show up in the TOC
        :param ChapterName: Name of the chapter
        '''

        self.current_reportsubgroup_num = 0
        with open(self.XML_fn, 'a') as XML:
            if ChapterText != '':
                outputtext = f'<Report_Group ReportGroupOrder="{self.current_reportgroup_num}" ReportGroupName="{ChapterName}" GroupText="{ChapterText}">\n'.replace('\\n', '\n').replace('\\t', '\t')
            else:
                outputtext = f'<Report_Group ReportGroupOrder="{self.current_reportgroup_num}" ReportGroupName="{ChapterName}">\n'.replace('\\n', '\n').replace('\\t', '\t')
            XML.write(outputtext)
        self.current_reportgroup_num += 1

    def writeChapterEnd(self):
        '''
        writes the end block for a chapter
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('</Report_Group>\n')

    def writeReportEnd(self):
        '''
        write the end block of the report
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('</USBR_Automated_Report>\n')

    def writeSectionHeader(self, section_header):
        '''
        writes the header section for the section
        :param section_header: description string of section
        '''

        with open(self.XML_fn, 'a') as XML:
            if section_header != '':
                XML.write('<Report_Subgroup ReportSubgroupOrder="{0}" ReportSubgroupDescription="{1}">\n'.format(self.current_reportsubgroup_num, section_header))
            else:
                XML.write('<Report_Subgroup ReportSubgroupOrder="{0}">\n'.format(self.current_reportsubgroup_num))
        self.current_reportsubgroup_num += 1

    def writeSectionHeaderEnd(self):
        '''
        writes end section for header
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('</Report_Subgroup>\n')

    def writeHalfPagePlot(self, figname, figdesc):
        '''
        writes a time series plot png
        :param figname: name of the figure
        :param figdesc: description of the figure
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Control_Point_Plots">\n'.format(self.current_reportelem_num))
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(figdesc))
            XML.write('<Output_Image FigureNumber="{0}" FigureDescription="{1}">{2}</Output_Image>\n'.format(self.current_fig_num ,figdesc, figname))
            XML.write('</Output_Temp_Flow>\n')
            XML.write('</Report_Element>\n')

        self.current_reportelem_num += 1
        self.current_fig_num += 1

    def writeFullPagePlot(self, figname, figdesc):
        '''
        writes a time series plot png
        :param figname: name of the figure
        :param figdesc: description of the figure
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Reservoir_Profile">\n'.format(self.current_reportelem_num))
            XML.write('<Reservoir_Profiles Reservoir="{0}">\n'.format(figdesc))
            XML.write('<Profile_Image FigureNumber="{0}" FigureDescription="{1}">{2}</Profile_Image>\n'.format(self.current_fig_num ,figdesc, figname))
            XML.write('</Reservoir_Profiles>\n')
            XML.write('</Report_Element>\n')

        self.current_reportelem_num += 1
        self.current_fig_num += 1

    def writeTextBox(self, text):
        '''
        writes a textbox object in the XML report
        :param text: text string to be added
        :return:
        '''
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="TextBox">\n'.format(self.current_reportelem_num))
            outputtext = f'<Text>{text}</Text>\n'.replace('\\n', '\n').replace('\\t', '\t')
            XML.write(outputtext)
            XML.write('</Report_Element>\n')
        self.current_reportelem_num += 1

    def writeProfilePlotStart(self, reservoir):
        '''
        writes the starting block for a profile plot
        :param reservoir: name of the reservior
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Reservoir_Profile">\n'.format(self.current_reportelem_num))
            XML.write('<Reservoir_Profiles Reservoir="{0}">\n'.format(reservoir))

        self.current_reportelem_num += 1

    def writeProfilePlotFigure(self, figname, figdesc):
        '''
        writes the profile plot png
        :param figname: name of the png file
        :param figdesc: description of the figure
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Profile_Image FigureNumber="{0}" FigureDescription="{1}">{2}</Profile_Image>\n'.format(self.current_fig_num, figdesc, figname))
        self.current_fig_num += 1

    def writeProfilePlotEnd(self):
        '''
        writes end block for the profile plot
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('</Reservoir_Profiles>\n')
            XML.write('</Report_Element>\n')

    def writeTableStart(self, desc, type, limit=False):
        '''
        writes start of table block for a desired table type
        :param desc: description of table
        :param type: type of table (monthly or error)
        '''

        with open(self.XML_fn, 'a') as XML:
            if limit:
                XML.write('<Report_Element ReportElementOrder="{0}" Element="Control_Point_Tables_Four_Columns">\n'.format(self.current_reportelem_num))
            else:
                XML.write('<Report_Element ReportElementOrder="{0}" Element="Control_Point_Tables">\n'.format(self.current_reportelem_num))
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(desc))
            XML.write('<Output_Table TableNumber="{0}" TableDescription="{1}" TableType="{2}">\n'.format(self.current_table_num, desc, type))

        self.current_reportelem_num += 1
        self.current_table_num += 1
        self.column_order = 0

    def writeNarrowTableStart(self, desc, type):
        '''
        writes start of table block for a desired table type
        :param desc: description of table
        :param type: type of table (monthly or error)
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="12_Column_Control_Point_Tables">\n'.format(self.current_reportelem_num))
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(desc))
            XML.write('<Output_Table TableNumber="{0}" TableDescription="{1}" TableType="{2}">\n'.format(self.current_table_num, desc, type))

        self.current_reportelem_num += 1
        self.current_table_num += 1
        self.column_order = 0

    def writeDateControlledTableStart(self, desc, type):
        '''
        writes the start of a datecontrolledtable
        :param desc: description of table
        :param type: table type (text in top left)
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="DateControlledTable">\n'.format(self.current_reportelem_num))
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(desc))
            XML.write('<Output_Table TableNumber="{0}" TableDescription="{1}" TableType="{2}">\n'.format(self.current_table_num, desc, type))

        self.current_reportelem_num += 1
        self.current_table_num += 1
        self.column_order = 0
        self.datecolumn_order = 0

    def writeTableColumn(self, header, rows, thresholdcolors=[]):
        '''
        writes a full column of a table
        :param header: name of header for column
        :param rows: row values
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<Column Column_Order="{0}" Column_Name="{1}">\n'.format(self.column_order, header))
            for i, row in enumerate(rows):
                s_row = row.split('|')
                rowname = s_row[0]
                rowval = s_row[1]
                if len(thresholdcolors) != 0:
                    if thresholdcolors[i] != None:
                        XML.write('<Row Row_Order="{0}" Row_name="{1}" Background_Color="{2}">{3}</Row>\n'.format(i, rowname, thresholdcolors[i], rowval))
                        continue
                XML.write('<Row Row_Order="{0}" Row_name="{1}">{2}</Row>\n'.format(i, rowname, rowval))
            XML.write('</Column>\n')
        self.column_order += 1

    def writeDateColumn(self, header):
        '''
        writes datecolumn for datecontrolledtables
        :param header: string inside of datecolumn
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('<DateColumn DateColumn_Order="{0}" DateColumn_Name="{1}">\n'.format(self.datecolumn_order, header))
        self.datecolumn_order += 1

    def writeDateColumnEnd(self):
        '''
        write end of datecolumn for datecontrolledtable
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('</DateColumn>\n')

    def writeTableEnd(self):
        '''
        writes end block for table
        '''

        with open(self.XML_fn, 'a') as XML:
            XML.write('</Output_Table>\n')
            XML.write('</Output_Temp_Flow>\n')
            XML.write('</Report_Element>\n')

    #################################################################
    #End Class
    #################################################################


def fixXMLModelIntroduction(Report, simorder):
    '''
    Fixes intro in XML that shows what models are used for each region.
    Updates a flag with used models.
    :param simorder: number of simulation file
    :return:
    '''

    outstr = '{0}:'.format(Report.ChapterRegion)
    for cnt, ID in enumerate(Report.accepted_IDs):
        if cnt > 0:
            outstr += ','
        outstr += ' {0}'.format(Report.SimulationVariables[ID]['program'])
    Report.XML.replaceinXML('%%REPLACEINTRO_{0}%%'.format(simorder), outstr)
