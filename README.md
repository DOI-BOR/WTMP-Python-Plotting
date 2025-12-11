# USBR WTMP Python Plotting

This folder will contain all Python scripts related to plotting for the USBR WQ development. 

1. Check in the usbr-python-plotting folder the existence of an environment.yml file to build or update your conda environment. Make sure that the NumPy package version is 1.26.4 or newer and the msvc_runtime package is 14.40.33807 or newer. Do not include the versions of the other packages.

2. Build a new python conda environment by writing the following in the command prompt (cmd)
	 conda env create -f environment.yml
 
3. Activate and check the updated Python version and all packages specified in environment.yml by writing the following in the command prompt.
	conda activate plotting-env
	conda list

4. Create WAT_Report_Generator.exe by writing the following in the command prompt.
	pyinstaller -y WAT_Report_Generator.py

5. From the folder dist/WAT_Report_Generator, copy the folder _internal and file WAT_Report_Generator.exe and paste them at WTMP-2025.10.10-1/HEC-WAT/AutomatedReport folder. Also delete the other files and subfolders in this folder.