class XMLReport(object):
    '''
    Builds and incrementally writes the custom XML report format used by
    Reclamation's WTMP reporting tools. The XML file is written to disk
    line-by-line as report elements (chapters, sections, tables, plots,
    text boxes, etc.) are added, and various counters are tracked to
    keep figure numbers, table numbers, and report element ordering
    correct throughout the document.

    Attributes
    ----------
    XML_fn : str
        Full path to the XML file being written.
    current_fig_num : int
        Running count of figures added to the report, starting at 1.
    current_table_num : int
        Running count of tables added to the report, starting at 1.
    current_reportelem_num : int
        Running count of report elements added, starting at 0.
    current_model_num : int
        Running count of models listed in the introduction, starting at 0.
    current_reportgroup_num : int
        Running count of report groups (chapters), starting at 0.
    current_reportsubgroup_num : int
        Running count of report subgroups (sections) within a chapter.
        Set when a new chapter is started.
    column_order : int
        Running count of columns added to the current table.
    datecolumn_order : int
        Running count of date columns added to the current date-controlled
        table.
    '''

    def __init__(self, XML_fn):
        '''
        Initialize the XML report class. Creates a fresh XML file at the
        given path and primes the internal counters used to track
        figures, tables, and other report elements.

        Parameters
        ----------
        XML_fn : str
            Name and path of the desired XML file.

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
        >>> xml_report = XMLReport('report.xml')
        '''

        # store the target XML file path for use by all writing methods
        self.XML_fn = XML_fn
        # create a fresh XML file with the required header/root tags
        self.makeXML()
        # reset all figure/table/section counters to their starting values
        self.primeCounters()

#########################################################################################
                            #Main functions#
#########################################################################################

    def makeXML(self):
        '''
        Create and write a fresh XML file for the report, writing the XML
        declaration and the opening root element tag.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value; it writes directly to
            `self.XML_fn`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though an
            `IOError` may occur if the file path is invalid or
            inaccessible.

        Examples
        --------
        >>> xml_report.makeXML()
        '''

        # open in write mode to truncate/create a fresh file
        with open(self.XML_fn, 'w') as XML:
            # write the required XML declaration line
            XML.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            # open the root element that will contain the whole report
            XML.write('<USBR_Automated_Report>\n')

    def replaceinXML(self, StringToReplace, StringReplacing):
        '''
        Replace all occurrences of a target string within the XML file,
        mainly used for substituting placeholder run-type text in the
        introduction.

        Parameters
        ----------
        StringToReplace : str
            String that is going to be replaced.
        StringReplacing : str
            String to replace it with.

        Returns
        -------
        None
            This function does not return a value; it rewrites
            `self.XML_fn` in place.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.replaceinXML('%%REPLACEINTRO_1%%', 'Sacramento: HEC-5Q')
        '''

        # read the whole file into memory line by line, substituting where needed
        xml_text = []
        with open(self.XML_fn, 'r') as XML:
            for line in XML:
                if StringToReplace in line:
                    # target string found in this line, perform the substitution
                    xml_text.append(line.replace(StringToReplace, StringReplacing))
                else:
                    # no match, keep the line unchanged
                    xml_text.append(line)

        # write the updated lines back out, overwriting the original file
        with open(self.XML_fn, 'w') as XML:
            for line in xml_text:
                XML.write(line)

    def insertAfter(self, StringToAddAfter, StringToAdd):
        '''
        Insert a new string immediately after every line containing a
        target string.

        Parameters
        ----------
        StringToAddAfter : str
            String to look for; new content is inserted after any line
            containing this string.
        StringToAdd : str
            String to insert immediately after the target line.

        Returns
        -------
        None
            This function does not return a value; it rewrites
            `self.XML_fn` in place.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.insertAfter('</Report_Group>\\n', '<!-- new chapter -->\\n')
        '''

        # read the whole file, inserting the new string right after any matching line
        xml_text = []
        with open(self.XML_fn, 'r') as XML:
            for line in XML:
                if StringToAddAfter in line:
                    # keep the original line, then insert the new content right after it
                    xml_text.append(line)
                    xml_text.append(StringToAdd)
                else:
                    # no match, keep the line unchanged
                    xml_text.append(line)

        # write the updated lines back out, overwriting the original file
        with open(self.XML_fn, 'w') as XML:
            for line in xml_text:
                XML.write(line)

    def removeLine(self, inputStr):
        '''
        Remove every line containing a target string from the XML
        document.

        Parameters
        ----------
        inputStr : str
            String to search for; any line containing this string is
            removed entirely.

        Returns
        -------
        None
            This function does not return a value; it rewrites
            `self.XML_fn` in place.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.removeLine('%%PLACEHOLDER%%')
        '''

        # read the file, skipping any line that contains the target string
        xml_text = []
        with open(self.XML_fn, 'r') as XML:
            for line in XML:
                if inputStr in line:
                    # matching line found, skip it entirely (i.e. remove it)
                    continue
                else:
                    # no match, keep the line
                    xml_text.append(line)
        # write the filtered lines back out, overwriting the original file
        with open(self.XML_fn, 'w') as XML:
            for line in xml_text:
                XML.write(line)

    def primeCounters(self):
        '''
        Set up the counters used to track figure numbers, table numbers,
        and various report element/group orderings. Some counters start
        at 0, others at 1, depending on how the downstream report
        template expects numbering to begin.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value. Instead, it sets the
            following attributes on the instance:
                self.current_fig_num : int
                self.current_table_num : int
                self.current_reportelem_num : int
                self.current_model_num : int
                self.current_reportgroup_num : int

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.primeCounters()
        >>> xml_report.current_fig_num
        1
        '''

        # figures and tables are numbered starting at 1 (matches document conventions)
        self.current_fig_num = 1
        self.current_table_num = 1
        # report elements, models, and report groups are numbered starting at 0
        self.current_reportelem_num = 0
        self.current_model_num = 0
        self.current_reportgroup_num = 0

    def writeCover(self, title):
        '''
        Write the cover page block for the report.

        Parameters
        ----------
        title : str
            Title text to display on the cover page.

        Returns
        -------
        None
            This function does not return a value; it appends directly to
            `self.XML_fn`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeCover('2020 Water Temperature Report')
        '''

        # append the cover page element containing the given title
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Cover_Page><Title>{0}</Title></Cover_Page>\n'.format(title))

    def writeIntroStart(self):
        '''
        Write the opening tags for the introduction section of the
        report, including the report group, subgroup, and element
        wrapper tags.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments:
                self.current_reportgroup_num
                self.current_reportelem_num

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeIntroStart()
        '''

        # open the report group / subgroup / element tags for the introduction
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Group ReportGroupOrder="{0}" ReportGroupName="Introduction">\n'.format(self.current_reportgroup_num))
            XML.write('<Report_Subgroup ReportSubgroupOrder="0">\n')
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Introduction">\n'.format(self.current_reportelem_num))

        # advance the group and element counters after writing this block
        self.current_reportgroup_num += 1
        self.current_reportelem_num += 1

    def writeIntroLine(self, program):
        '''
        Write a single line in the introduction listing a model/program
        used in the report.

        Parameters
        ----------
        program : str
            Name of the program/model to be added to the model list.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments `self.current_model_num`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Notes
        -----
        The `ModelOrder` attribute in the written XML is hardcoded to
        "0" rather than using `self.current_model_num`, even though the
        counter is still incremented afterward.

        Examples
        --------
        >>> xml_report.writeIntroLine('HEC-5Q')
        '''

        # write the model entry; note ModelOrder is hardcoded to 0 in the current implementation
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Model ModelOrder="0" >{1}</Model>\n'.format(self.current_model_num, program))
        # advance the model counter regardless
        self.current_model_num += 1

    def writeIntroEnd(self):
        '''
        Write the closing tags for the introduction section, matching the
        tags opened in `self.writeIntroStart()`.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value; it appends directly to
            `self.XML_fn`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeIntroEnd()
        '''

        # close the element, subgroup, and group tags opened by writeIntroStart()
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Report_Element>\n')
            XML.write('</Report_Subgroup>\n')
            XML.write('</Report_Group>\n')

    def writeChapterStart(self, ChapterName, ChapterText):
        '''
        Write the opening block for an individual chapter. Chapter names
        automatically appear in the report's table of contents.

        Parameters
        ----------
        ChapterName : str
            Name of the chapter, used as the report group name and shown
            in the table of contents.
        ChapterText : str
            Optional descriptive text for the chapter. If empty, no
            `GroupText` attribute is written.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn`, sets `self.current_reportsubgroup_num` to 0,
            and increments `self.current_reportgroup_num`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeChapterStart('Shasta Reservoir', 'Simulated using HEC-5Q')
        '''

        # each new chapter resets its own subgroup (section) counter
        self.current_reportsubgroup_num = 0
        with open(self.XML_fn, 'a') as XML:
            if ChapterText != '':
                # include GroupText attribute, and unescape literal \n and \t sequences
                outputtext = f'<Report_Group ReportGroupOrder="{self.current_reportgroup_num}" ReportGroupName="{ChapterName}" GroupText="{ChapterText}">\n'.replace('\\n', '\n').replace('\\t', '\t')
            else:
                # no chapter text provided, omit the GroupText attribute entirely
                outputtext = f'<Report_Group ReportGroupOrder="{self.current_reportgroup_num}" ReportGroupName="{ChapterName}">\n'.replace('\\n', '\n').replace('\\t', '\t')
            XML.write(outputtext)
        # advance the report group counter after writing this chapter's opening tag
        self.current_reportgroup_num += 1

    def writeChapterEnd(self):
        '''
        Write the closing tag for a chapter, matching the tag opened in
        `self.writeChapterStart()`.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value; it appends directly to
            `self.XML_fn`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeChapterEnd()
        '''

        # close the report group tag opened by writeChapterStart()
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Report_Group>\n')

    def writeReportEnd(self):
        '''
        Write the closing tag for the overall report, matching the root
        element opened in `self.makeXML()`.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value; it appends directly to
            `self.XML_fn`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeReportEnd()
        '''

        # close the root USBR_Automated_Report element
        with open(self.XML_fn, 'a') as XML:
            XML.write('</USBR_Automated_Report>\n')

    def writeSectionHeader(self, section_header):
        '''
        Write the opening tag for a report subgroup (section) within the
        current chapter.

        Parameters
        ----------
        section_header : str
            Description string for the section. If empty, no
            `ReportSubgroupDescription` attribute is written.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments `self.current_reportsubgroup_num`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeSectionHeader('Monthly Averages')
        '''

        with open(self.XML_fn, 'a') as XML:
            if section_header != '':
                # include the description attribute when one is provided
                XML.write('<Report_Subgroup ReportSubgroupOrder="{0}" ReportSubgroupDescription="{1}">\n'.format(self.current_reportsubgroup_num, section_header))
            else:
                # no description given, write a bare subgroup tag
                XML.write('<Report_Subgroup ReportSubgroupOrder="{0}">\n'.format(self.current_reportsubgroup_num))
        # advance the subgroup counter after writing this section's opening tag
        self.current_reportsubgroup_num += 1

    def writeSectionHeaderEnd(self):
        '''
        Write the closing tag for a report subgroup (section), matching
        the tag opened in `self.writeSectionHeader()`.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value; it appends directly to
            `self.XML_fn`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeSectionHeaderEnd()
        '''

        # close the report subgroup tag opened by writeSectionHeader()
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Report_Subgroup>\n')

    def writeHalfPagePlot(self, figname, figdesc):
        '''
        Write a half-page time series plot image block (typically a
        control point plot) into the report.

        Parameters
        ----------
        figname : str
            Name of the figure/image file.
        figdesc : str
            Description of the figure, also used as the location label.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments `self.current_reportelem_num`
            and `self.current_fig_num`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeHalfPagePlot('shasta_temp.png', 'Shasta Dam')
        '''

        # write the nested report element / output / image tags for this plot
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Control_Point_Plots">\n'.format(self.current_reportelem_num))
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(figdesc))
            XML.write('<Output_Image FigureNumber="{0}" FigureDescription="{1}">{2}</Output_Image>\n'.format(self.current_fig_num ,figdesc, figname))
            XML.write('</Output_Temp_Flow>\n')
            XML.write('</Report_Element>\n')

        # advance both the report element counter and the figure counter
        self.current_reportelem_num += 1
        self.current_fig_num += 1

    def writeFullPagePlot(self, figname, figdesc):
        '''
        Write a full-page time series plot image block (typically a
        reservoir profile plot) into the report.

        Parameters
        ----------
        figname : str
            Name of the figure/image file.
        figdesc : str
            Description of the figure, also used as the reservoir label.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments `self.current_reportelem_num`
            and `self.current_fig_num`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeFullPagePlot('shasta_profile.png', 'Shasta Reservoir')
        '''

        # write the nested report element / reservoir profile / image tags for this plot
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Reservoir_Profile">\n'.format(self.current_reportelem_num))
            XML.write('<Reservoir_Profiles Reservoir="{0}">\n'.format(figdesc))
            XML.write('<Profile_Image FigureNumber="{0}" FigureDescription="{1}">{2}</Profile_Image>\n'.format(self.current_fig_num ,figdesc, figname))
            XML.write('</Reservoir_Profiles>\n')
            XML.write('</Report_Element>\n')

        # advance both the report element counter and the figure counter
        self.current_reportelem_num += 1
        self.current_fig_num += 1

    def writeTextBox(self, text):
        '''
        Write a text box report element containing arbitrary descriptive
        text.

        Parameters
        ----------
        text : str
            Text string to be added inside the text box.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments `self.current_reportelem_num`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeTextBox('This chapter summarizes results for the American River.')
        '''
        with open(self.XML_fn, 'a') as XML:
            # open the report element wrapper for this text box
            XML.write('<Report_Element ReportElementOrder="{0}" Element="TextBox">\n'.format(self.current_reportelem_num))
            # unescape literal \n and \t sequences before writing the text content
            outputtext = f'<Text>{text}</Text>\n'.replace('\\n', '\n').replace('\\t', '\t')
            XML.write(outputtext)
            # close the report element wrapper
            XML.write('</Report_Element>\n')
        # advance the report element counter after writing this text box
        self.current_reportelem_num += 1

    def writeProfilePlotStart(self, reservoir):
        '''
        Write the opening block for a profile plot, used when multiple
        profile images will be added under a single reservoir grouping.

        Parameters
        ----------
        reservoir : str
            Name of the reservoir the profile plots belong to.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments `self.current_reportelem_num`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeProfilePlotStart('Shasta Reservoir')
        '''

        # open the report element and reservoir profile group tags
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Reservoir_Profile">\n'.format(self.current_reportelem_num))
            XML.write('<Reservoir_Profiles Reservoir="{0}">\n'.format(reservoir))

        # advance the report element counter after opening this profile plot group
        self.current_reportelem_num += 1

    def writeProfilePlotFigure(self, figname, figdesc):
        '''
        Write a single profile plot image tag within an already-open
        profile plot group (opened via `self.writeProfilePlotStart()`).

        Parameters
        ----------
        figname : str
            Name of the PNG image file.
        figdesc : str
            Description of the figure.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments `self.current_fig_num`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeProfilePlotFigure('shasta_may.png', 'May 2020')
        '''

        # write the profile image tag with the current figure number
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Profile_Image FigureNumber="{0}" FigureDescription="{1}">{2}</Profile_Image>\n'.format(self.current_fig_num, figdesc, figname))
        # advance the figure counter after writing this image
        self.current_fig_num += 1

    def writeProfilePlotEnd(self):
        '''
        Write the closing tags for a profile plot group, matching the
        tags opened in `self.writeProfilePlotStart()`.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value; it appends directly to
            `self.XML_fn`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeProfilePlotEnd()
        '''

        # close the reservoir profiles and report element tags
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Reservoir_Profiles>\n')
            XML.write('</Report_Element>\n')

    def writeTableStart(self, desc, type, limit=False):
        '''
        Write the opening block of a table, for a desired table type.

        Parameters
        ----------
        desc : str
            Description of the table, used as both the location label and
            table description.
        type : str
            Type of table (e.g. 'monthly' or 'error').
        limit : bool, optional
            If True, writes the table using the four-column control point
            table element instead of the standard control point table
            element. Default is False.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn`, increments `self.current_reportelem_num` and
            `self.current_table_num`, and resets `self.column_order` to 0.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeTableStart('Shasta Dam', 'monthly')
        '''

        with open(self.XML_fn, 'a') as XML:
            if limit:
                # four-column variant of the control point table element
                XML.write('<Report_Element ReportElementOrder="{0}" Element="Control_Point_Tables_Four_Columns">\n'.format(self.current_reportelem_num))
            else:
                # standard control point table element
                XML.write('<Report_Element ReportElementOrder="{0}" Element="Control_Point_Tables">\n'.format(self.current_reportelem_num))
            # open the location and table wrapper tags
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(desc))
            XML.write('<Output_Table TableNumber="{0}" TableDescription="{1}" TableType="{2}">\n'.format(self.current_table_num, desc, type))

        # advance the element and table counters, and reset column tracking for this new table
        self.current_reportelem_num += 1
        self.current_table_num += 1
        self.column_order = 0

    def writeNarrowTableStart(self, desc, type):
        '''
        Write the opening block of a narrow (12-column) table variant.

        Parameters
        ----------
        desc : str
            Description of the table, used as both the location label and
            table description.
        type : str
            Type of table (e.g. 'monthly' or 'error').

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn`, increments `self.current_reportelem_num` and
            `self.current_table_num`, and resets `self.column_order` to 0.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeNarrowTableStart('Shasta Dam', 'monthly')
        '''

        # write the 12-column control point table element and its wrapper tags
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="12_Column_Control_Point_Tables">\n'.format(self.current_reportelem_num))
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(desc))
            XML.write('<Output_Table TableNumber="{0}" TableDescription="{1}" TableType="{2}">\n'.format(self.current_table_num, desc, type))

        # advance the element and table counters, and reset column tracking for this new table
        self.current_reportelem_num += 1
        self.current_table_num += 1
        self.column_order = 0

    def writeDateControlledTableStart(self, desc, type):
        '''
        Write the opening block of a date-controlled table, which
        includes an additional date column tracking counter compared to
        standard tables.

        Parameters
        ----------
        desc : str
            Description of the table, used as both the location label and
            table description.
        type : str
            Table type (displayed as text in the top-left of the table).

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn`, increments `self.current_reportelem_num` and
            `self.current_table_num`, and resets `self.column_order` and
            `self.datecolumn_order` to 0.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeDateControlledTableStart('Shasta Dam', 'daily')
        '''

        # write the date-controlled table element and its wrapper tags
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="DateControlledTable">\n'.format(self.current_reportelem_num))
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(desc))
            XML.write('<Output_Table TableNumber="{0}" TableDescription="{1}" TableType="{2}">\n'.format(self.current_table_num, desc, type))

        # advance element/table counters and reset both column-tracking counters
        self.current_reportelem_num += 1
        self.current_table_num += 1
        self.column_order = 0
        self.datecolumn_order = 0

    def writeTableColumn(self, header, rows, thresholdcolors=[]):
        '''
        Write a full column of data into the currently open table,
        including optional per-row background color highlighting.

        Parameters
        ----------
        header : str
            Header name for the column.
        rows : list of str
            Row values, where each entry is a string formatted as
            `'rowname|rowvalue'`.
        thresholdcolors : list, optional
            List of colors (or `None`) parallel to `rows`, used to set the
            background color of individual rows that exceed some
            threshold. If empty, no color highlighting is applied.
            Default is an empty list.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments `self.column_order`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions, though a
            malformed `rows` entry lacking a `'|'` separator would raise
            an `IndexError`.

        Examples
        --------
        >>> xml_report.writeTableColumn('Jan', ['Shasta|45.2', 'Keswick|46.1'])
        '''

        with open(self.XML_fn, 'a') as XML:
            # open the column tag with its order and header name
            XML.write('<Column Column_Order="{0}" Column_Name="{1}">\n'.format(self.column_order, header))
            # write one Row tag per entry in rows, splitting name and value on '|'
            for i, row in enumerate(rows):
                s_row = row.split('|')
                rowname = s_row[0]
                rowval = s_row[1]
                if len(thresholdcolors) != 0:
                    if thresholdcolors[i] != None:
                        # a color was specified for this row, include the Background_Color attribute
                        XML.write('<Row Row_Order="{0}" Row_name="{1}" Background_Color="{2}">{3}</Row>\n'.format(i, rowname, thresholdcolors[i], rowval))
                        continue
                # no color specified (or thresholdcolors not provided), write a plain row
                XML.write('<Row Row_Order="{0}" Row_name="{1}">{2}</Row>\n'.format(i, rowname, rowval))
            # close the column tag
            XML.write('</Column>\n')
        # advance the column counter after writing this column
        self.column_order += 1

    def writeDateColumn(self, header):
        '''
        Write the opening tag for a date column, used within
        date-controlled tables.

        Parameters
        ----------
        header : str
            Text/label to display inside the date column header.

        Returns
        -------
        None
            This function does not return a value. It appends directly to
            `self.XML_fn` and increments `self.datecolumn_order`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeDateColumn('Date')
        '''

        # write the date column opening tag with its order and header text
        with open(self.XML_fn, 'a') as XML:
            XML.write('<DateColumn DateColumn_Order="{0}" DateColumn_Name="{1}">\n'.format(self.datecolumn_order, header))
        # advance the date column counter after writing this tag
        self.datecolumn_order += 1

    def writeDateColumnEnd(self):
        '''
        Write the closing tag for a date column, matching the tag opened
        in `self.writeDateColumn()`.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value; it appends directly to
            `self.XML_fn`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeDateColumnEnd()
        '''

        # close the date column tag opened by writeDateColumn()
        with open(self.XML_fn, 'a') as XML:
            XML.write('</DateColumn>\n')

    def writeTableEnd(self):
        '''
        Write the closing tags for a table, matching the tags opened in
        `self.writeTableStart()`, `self.writeNarrowTableStart()`, or
        `self.writeDateControlledTableStart()`.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value; it appends directly to
            `self.XML_fn`.

        Raises
        ------
        None
            This function does not explicitly raise exceptions.

        Examples
        --------
        >>> xml_report.writeTableEnd()
        '''

        # close the output table, location, and report element tags
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Output_Table>\n')
            XML.write('</Output_Temp_Flow>\n')
            XML.write('</Report_Element>\n')

    #################################################################
    #End Class
    #################################################################


def fixXMLModelIntroduction(Report, simorder):
    '''
    Fix a placeholder in the report's XML introduction that describes
    which model(s) are used for a given region, replacing it with the
    actual list of accepted simulation programs.

    Parameters
    ----------
    Report : object
        Instance from the main report script. Must have `ChapterRegion`
        (str), `accepted_IDs` (list of simulation IDs), `SimulationVariables`
        (dict keyed by simulation ID containing a 'program' key), and an
        `XML` attribute that is an `XMLReport` instance.
    simorder : int or str
        Number/identifier of the simulation file, used to build the
        placeholder string to search for.

    Returns
    -------
    None
        This function does not return a value; it updates the XML file
        via `Report.XML.replaceinXML()`.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Examples
    --------
    >>> fixXMLModelIntroduction(Report, 1)
    '''

    # start building the replacement string with the region name as a label
    outstr = '{0}:'.format(Report.ChapterRegion)
    # append each accepted simulation's program name, comma-separated
    for cnt, ID in enumerate(Report.accepted_IDs):
        if cnt > 0:
            # add a separating comma before every program after the first
            outstr += ','
        outstr += ' {0}'.format(Report.SimulationVariables[ID]['program'])
    # replace the placeholder in the XML with the fully built model list string
    Report.XML.replaceinXML('%%REPLACEINTRO_{0}%%'.format(simorder), outstr)