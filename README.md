# WTMP Python Plotting

This repository contains all the Python scripts related to plotting for the Bureau of Reclamation (USBR) Water Temperature Modeling Platform (WTMP) reporting. 
All the code to create the plots and XML files for the Jasper report is contained in this repository. 
This code gets compiled into an executable for the WAT to call to create the reports. 
The code reads in the XML and DSS outputs from the WTMP models and creates all necessary plots and an XML file with the report data for Jasper to read in.
These files are in Python.

The *WAT_Report_Generator.py* acts as the main file in this repository. It calls functions from the other .py files that are grouped together by similar function.
This code is used for both the W2 and ResSim models. Differences between the models are handled within the code.

## Dependencies
All Python dependencies are in the *environment.yml* file. The code is meant to be compiled and ran inside the WAT.

## Usage
### Usage withing the build process
To be added later.

### Post build implementation
After making any desired changes to the code, a new executable must be compiled and placed in the WAT build.
1. To create a conda environment, in a command prompt type: `conda env create -f environment.yml`
2. Activate the environment by running the line: `conda activate plotting-env`
3. Create an executable by running the line: `pyinstaller -y WAT_Report_Generator.py`
4. In the WAT build *HEC-WAT/AutomatedReport* directory, delete any old files.
5. From the *dist/WAT_Report_Generator* directory, copy the *_internal* directory and *WAT_Report_Generator.exe* file and paste them into the WAT build */HEC-WAT/AutomatedReport* directory.