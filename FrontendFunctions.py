import PyNexus as PN
import AnalysisFunctions as AF

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import time
from matplotlib.pyplot import cm
import matplotlib.colors as mplcolors
from matplotlib.ticker import FormatStrFormatter

import numpy as np
from math import isclose

import os
import shutil
import csv

import base64
from IPython.display import clear_output, Javascript, display, HTML
import subprocess

try:
    import ipysheet
except:
    print('Careful: the module ipysheet is not installed!')

try:
    import xraylib
except:
    print('Careful: the module xraylib is not installed!')

__version__="0.12"

"""
-Here are defined all the functions relevant to the front end of JupyFluo,
i.e. the widgets (buttons, interactive plots), the self-generation of cells, conversion and saving of files ...
-Arguments are passed from the Notebook through an belonging to the class Experiment only.
"""


# General parameters for style of widgets
style = {'description_width': 'initial'}

def Print_version():
    print("Versions of modules used:")
    print("AnalysisFunctions: %s"%AF.__version__)
    print("FrontendFunctions: %s"%__version__)
    print("PyNexus: %s"%PN.__version__)
    print('Check that you are using the last versions of the modules and read the manual on:\n%s'
          %'https://github.com/ArnaudHemmerle/JupyFluo')
    print("")

def Check_files(expt):

    Print_version()

    if not os.path.exists(expt.working_dir):
        print(PN._RED+"The following folder does not exist and should be created:"+PN._RESET)
        print(expt.working_dir)
        print("Data analysis will be saved in this folder.")
        print("")
    if not os.path.exists(expt.recording_dir):
        print(PN._RED+"The following folder does not exist:"+PN._RESET)
        print(expt.recording_dir)
        print("This folder should contain the nexus files.")
        print("")
    if not os.path.exists(expt.notebook_name):
        print(PN._RED+"The following file does not exist:"+PN._RESET)
        print(expt.notebook_name)
        print("Check that you have assigned the correct notebook name to expt.notebook_name")
        print("")
    if not os.path.exists('DefaultPeaks.csv'):
        print(PN._RED+"The following file does not exist:"+PN._RESET)
        print('DefaultPeaks.csv')
        print("This file contains the peaks to be displayed by default and should be placed in the same folder as the notebook.")
        print("")
    if not os.path.exists('DefaultParameters.csv'):
        print(PN._RED+"The following file does not exist:"+PN._RESET)
        print('DefaultParameters.csv')
        print("This file contains the parameters to be displayed by default and should be placed in the same folder as the notebook.")
        print("")
    if not os.path.exists('f-si'):
        print(PN._RED+"The following file does not exist:"+PN._RESET)
        print('f-si')
        print("This file contains the scattering factor of Si and should be placed in the same folder as the notebook.")
        print("")
    if not os.path.exists('latex_template.tplx'):
        print(PN._RED+"The following file does not exist:"+PN._RESET)
        print('latex_template.tplx')
        print("This file contains the template for generating PDF and should be placed in the same folder as the notebook.")
        print("")

def Create_cell(code='', position='below', celltype='markdown', is_print=False, is_execute=True):
    """Create a cell in the IPython Notebook.
    code: unicode, Code to fill the new cell with.
    celltype: unicode, Type of cells "code" or "markdown".
    position: unicode, Where to put the cell "below" or "at_bottom"
    is_print: boolean, To decide if the cell is printed in the pdf report
    is_execute: boolean, To decide if the cell is executed after creation
    The code requires direct use of Javascript and is thus not usable in Jupyter Lab.
    """

    # Delay to ensure unique id
    time.sleep(0.1)

    encoded_code = (base64.b64encode(code.encode())).decode()

    # Create a unique id based on epoch time
    display_id = int(time.time()*1e9)

    javascript_code = """
                      var cell = IPython.notebook.insert_cell_{0}("{1}");
                      cell.set_text(atob("{2}"));
                      """.format(position, celltype, encoded_code)

    if not is_print:
        javascript_code += '\n'+'cell.metadata.tags = [\'notPrint\'];'

    if is_execute:
        javascript_code += '\n'+'cell.execute();'

    display(Javascript(javascript_code),display_id=display_id)

    # Necessary hack to avoid scan-execution of cells at notebook re-opening
    # See http://tiny.cc/fnf3nz
    display(Javascript(""" """), display_id=display_id, update=True)

def Generate_new_cells(expt):
    """
    Generate the title corresponding to the current scan.
    """

    position = 'at_bottom'

    """
    cells is an array of array used to create the predefined cells.
    Each array in cells should contain (code, position, celltype, is_print, is_execute)
    code is a string with the text/code to display in the cell
    celltype is a string, 'code' or 'markdown'
    is_execute is a boolean to tell if the cell should executed immediately after creation
    """

    cells = [
    ['# '+expt.id, position, 'markdown', True, True],
    ['Describe the scan here.', position, 'markdown', True, True],
    ['FF.Display_panel(expt)', position, 'code', False, True]
    ]

    for cell in cells:
        Create_cell(code=cell[0], position=cell[1], celltype=cell[2], is_print=cell[3], is_execute = cell[4])


def Delete_current_cell():
    """Delete the cell which called this function in the IPython Notebook."""

    display(Javascript(
        """
        var index = IPython.notebook.get_selected_cells_indices();
        IPython.notebook.delete_cell(index);
        """
    ))

def Export_nb_to_pdf(nb_name):
    """
    Save the current state of the notebook.
    Export the notebook to pdf using a command line through the OS.
    Return a boolean which is True if the export suceeded without error/warning.
    """

    # Save the notebook
    display(Javascript('IPython.notebook.save_checkpoint();'))

    # Uncomment to save the widgets as well (but the file will be big)
    #display(HTML('<script>Jupyter.menubar.actions._actions["widgets:save-with-widgets"].handler()</script>') )

    t0 = time.time()
    rc = 1
    while rc>0:
        if (time.time()-t0) > 300:
            # Timeout before PDF export is considered as failed
            export_done = False
            break
        else:
            time.sleep(3)
            command = 'jupyter nbconvert '
            command+= nb_name
            command+= ' --to pdf '
            command+= ' --TagRemovePreprocessor.remove_cell_tags=\"[\'notPrint\']\" ' # Remove the widgets from the PDF
            command+= ' --no-input '
            command+= ' --template latex_template.tplx ' # Custom template
            rc = subprocess.call(command,shell=True)
            if rc==0: export_done = True

    return export_done


################ ANALYSIS ###########################

class Scan:
    """
    Class Scan is used to pass arguments concerning the current scan only.
    """
    def __init__(self):
        pass


def Set_scan(expt):
    """
    1) Create widgets to allow for user to enter required params.
    2) Execute the spectrum extraction when clicking on the button.
    """

    # Create the list of nxs files in the folder
    expt.list_nxs_files = [file for file in sorted(os.listdir(expt.recording_dir)) if 'nxs' in file][::-1]
    if expt.list_nxs_files == []:
        print(PN._RED+'There is no nexus file in the recording folder.'+PN._RESET)
        print(PN._RED+'Recording folder: %s'%expt.recording_dir+PN._RESET)
        expt.list_nxs_files = ['SIRIUS_NoFileFound_00_00_00.nxs']

    def on_button_choose_scan_clicked(b):

        # Generate several identifiers for the scan
        expt.nxs = w_select_scan.value
        Set_scan_identifiers(expt)

        # Create a folder for saving params and results, if it does not already exist.
        if not os.path.exists(expt.working_dir+expt.id):
            os.mkdir(expt.working_dir+expt.id)

        # Check if the csv file for parameters already exists, if not copy the DefaultParameters.csv file
        if not os.path.isfile(expt.working_dir+expt.id+'/Parameters.csv'):
            shutil.copy('DefaultParameters.csv', expt.working_dir+expt.id+'/Parameters.csv')

        # Set the info for the panel generation
        expt.is_extract_done = False
        expt.is_fit_ready = False
        expt.is_fit_done = False

        # Generate the markdown cells
        Generate_new_cells(expt)

        Delete_current_cell()

    w_select_scan = widgets.Dropdown(
        options=expt.list_nxs_files,
        description='Select scan:',
        layout=widgets.Layout(width='400px'),
        style=style)

    button_choose_scan = widgets.Button(description="OK",layout=widgets.Layout(width='100px'))
    button_choose_scan.on_click(on_button_choose_scan_clicked)

    display(widgets.HBox([w_select_scan, button_choose_scan]))

def Display_panel(expt):
    """Display the panel to select the next step."""

    def on_button_new_scan_clicked(b):
        """Start the analysis of a new scan."""

        Delete_current_cell()

        Create_cell(code='scan = FF.Set_scan(expt)', position='at_bottom', celltype='code', is_print=False, is_execute = True)


    def on_button_export_clicked(b):
        """Export the notebook to PDF."""

        print('Export in progress...')

        export_done = Export_nb_to_pdf(expt.notebook_name)

        if export_done:
            print('Notebook exported to %s.pdf'%expt.notebook_name.split('.')[0])
        else:
            print("There was something wrong with the export to pdf. Please try again.")


    def on_button_set_params_clicked(b):
        """Set the parameters and extract the scan."""

        # Reset the buttons
        expt.is_extract_done = False
        expt.is_fit_ready = False
        expt.is_fit_done = False

        # Set the parameters
        Set_params(expt)

    def on_button_plot_peaks_clicked(b):
        """Plot the peaks."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_panel(expt)

        # Reput the sheet set peaks
        Set_peaks(expt)

        # Extract the info from the sheet
        Extract_groups(expt)

        # Display the peaks on the selected spectrum or on the sum
        if expt.is_peaks_on_sum:
            Display_peaks(expt)
        else:
            w = widgets.interact(Display_peaks,
                                 expt=widgets.fixed(expt),
                                 spectrum_index=widgets.IntText(value=0, step=1, description='Spectrum:'))


    def on_button_save_params_clicked(b):
        """Save the current params as default ones."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_panel(expt)

        # Copy the current params as the defaut file
        shutil.copy(expt.working_dir+expt.id+'/Parameters.csv','DefaultParameters.csv')

        print("Current set of parameters saved as default.")

    def on_button_load_params_clicked(b):
        """Load a set of params as the current one."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_panel(expt)

        list_params_files = [file for file in sorted(os.listdir('parameters/'))][::-1]

        w_select_files = widgets.Dropdown(options=list_params_files)

        def on_button_import_clicked(b):
            "Make the copy."

            # Copy the selected params as the current params file
            shutil.copy('parameters/'+w_select_files.value, expt.working_dir+expt.id+'/Parameters.csv')

            print(str(w_select_files.value)+" imported as current set of parameters.")

        button_import = widgets.Button(description="OK",layout=widgets.Layout(width='100px'))
        button_import.on_click(on_button_import_clicked)

        display(widgets.HBox([w_select_files, button_import]))


    def on_button_save_peaks_clicked(b):
        """Save the current peaks as default ones."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_panel(expt)

        # Copy the current params as the defaut file
        shutil.copy(expt.working_dir+expt.id+'/Peaks.csv','DefaultPeaks.csv')

        print("Current set of peaks saved as the default one.")

    def on_button_reload_peaks_clicked(b):
        """Reload the default peaks as current ones."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_panel(expt)

        # Copy the current params as the defaut file
        shutil.copy('DefaultPeaks.csv', expt.working_dir+expt.id+'/Peaks.csv')

        print("Default set of peaks saved as the current one.")


    def on_button_start_fit_clicked(b):
        """Start the fit."""

        # Clear the plots
        clear_output(wait=True)

        # Do the fit
        AF.Fit_spectrums(expt)

        expt.is_fit_done = True

        # Reput the boxes
        Display_panel(expt)

    def on_button_add_plot_clicked(b):
        """Create a new cell with the result to be added to the report."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_panel(expt)

        Choose_spectrum_to_plot(expt)


    def on_button_extract_mean_clicked(b):
        """Extract the mean values of the fitted parameters."""

        for name in expt.dparams_list:
            if name[:-5] in expt.list_isfit:
                  print(name[:-5], np.nanmean(expt.dparams_list[name]))

                  if name[:-5] == 'sl':
                      expt.sl = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'ct':
                      expt.ct = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'noise':
                      expt.noise = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'sfa0':
                      expt.sfa0 = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'sfa1':
                      expt.sfa1 = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'tfb0':
                      expt.tfb0 = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'tfb1':
                      expt.tfb1 = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'twc0':
                      expt.twc0 = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'twc1':
                      expt.twc1 = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'fG':
                      expt.fG = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'fA':
                      expt.fA = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'fB':
                      expt.fB = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'gammaA':
                      expt.gammaA = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'gammaB':
                      expt.gammaB = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'epsilon':
                      expt.epsilon = np.nanmean(expt.dparams_list[name])

                  if name[:-5] == 'fano':
                      expt.fano = np.nanmean(expt.dparams_list[name])

        # Save the updated values

        # Prepare the header of the csv file
        with open(expt.working_dir+expt.id+'/Parameters.csv', "w", newline='') as f:
            writer = csv.writer(f,delimiter=';',dialect='excel')
            header = np.array([
                    'is_fluospectrum00',
                    'is_fluospectrum01',
                    'is_fluospectrum02',
                    'is_fluospectrum03',
                    'is_fluospectrum04',
                    'ind_first_channel',
                    'ind_last_channel',
                    'ind_first_spectrum',
                    'ind_last_spectrum',
                    'gain',
                    'eV0',
                    'beam_energy',
                    'is_ipysheet',
                    'delimiter',
                    'fitstuck_limit',
                    'min_strength',
                    'is_fast',
                    'list_isfit_str',
                    'sl',
                    'ct',
                    'noise',
                    'sfa0',
                    'sfa1',
                    'tfb0',
                    'tfb1',
                    'twc0',
                    'twc1',
                    'fG',
                    'fA',
                    'fB',
                    'gammaA',
                    'gammaB',
                    'epsilon',
                    'fano',
                    'is_transmitted',
                    'is_peaks_on_sum',
                    'is_show_peaks',
                    'is_show_subfunctions'
                    ])
            writer.writerow(header)

            writer.writerow([
                    expt.is_fluospectrum00,
                    expt.is_fluospectrum01,
                    expt.is_fluospectrum02,
                    expt.is_fluospectrum03,
                    expt.is_fluospectrum04,
                    expt.ind_first_channel,
                    expt.ind_last_channel,
                    expt.ind_first_spectrum,
                    expt.ind_last_spectrum,
                    expt.gain,
                    expt.eV0,
                    expt.beam_energy,
                    expt.is_ipysheet,
                    expt.delimiter,
                    expt.fitstuck_limit,
                    expt.min_strength,
                    expt.is_fast,
                    expt.list_isfit_str,
                    expt.sl,
                    expt.ct,
                    expt.noise,
                    expt.sfa0,
                    expt.sfa1,
                    expt.tfb0,
                    expt.tfb1,
                    expt.twc0,
                    expt.twc1,
                    expt.fG,
                    expt.fA,
                    expt.fB,
                    expt.gammaA,
                    expt.gammaB,
                    expt.epsilon,
                    expt.fano,
                    expt.is_transmitted,
                    expt.is_peaks_on_sum,
                    expt.is_show_peaks,
                    expt.is_show_subfunctions
                    ])


    button_new_scan = widgets.Button(description="Analyze a new scan",layout=widgets.Layout(width='200px'))
    button_new_scan.on_click(on_button_new_scan_clicked)

    button_export = widgets.Button(description="Export to pdf", layout=widgets.Layout(width='200px'))
    button_export.on_click(on_button_export_clicked)

    button_set_params = widgets.Button(description="Set params",layout=widgets.Layout(width='200px'))
    button_set_params.on_click(on_button_set_params_clicked)

    button_save_params = widgets.Button(description="Save current params as default",layout=widgets.Layout(width='250px'))
    button_save_params.on_click(on_button_save_params_clicked)

    button_load_params = widgets.Button(description="Load params",layout=widgets.Layout(width='200px'))
    button_load_params.on_click(on_button_load_params_clicked)

    display(widgets.HBox([button_new_scan,button_export]))
    print(100*"-")
    print("Set parameters:")
    display(widgets.HBox([button_set_params, button_load_params, button_save_params]))
    print(100*"-")

    if expt.is_extract_done:

        button_plot_peaks = widgets.Button(description="Set peaks",layout=widgets.Layout(width='200px'))
        button_plot_peaks.on_click(on_button_plot_peaks_clicked)

        button_reload_peaks = widgets.Button(description="Reload default peaks",layout=widgets.Layout(width='250px'))
        button_reload_peaks.on_click(on_button_reload_peaks_clicked)

        button_save_peaks = widgets.Button(description="Save current peaks as default",layout=widgets.Layout(width='250px'))
        button_save_peaks.on_click(on_button_save_peaks_clicked)

        print("Set peaks:")

        if expt.is_fit_ready:
            display(widgets.HBox([button_plot_peaks,button_reload_peaks, button_save_peaks]))
        else:
            display(button_plot_peaks)
        print(100*"-")

    if expt.is_fit_ready:

        button_start_fit = widgets.Button(description="Start fit",layout=widgets.Layout(width='200px'))
        button_start_fit.on_click(on_button_start_fit_clicked)

        button_add_plot = widgets.Button(description="Add a plot to report",layout=widgets.Layout(width='200px'))
        button_add_plot.on_click(on_button_add_plot_clicked)

        button_extract_mean = widgets.Button(description="Extract averages",layout=widgets.Layout(width='200px'))
        button_extract_mean.on_click(on_button_extract_mean_clicked)

        print("Fit:")

        if expt.is_fit_done:
            display(widgets.HBox([button_start_fit, button_add_plot, button_extract_mean]))
        else:
            display(widgets.HBox([button_start_fit,  button_add_plot]))

        print(100*"-")

def Set_params(expt):
    """Fill the parameters and extract the scan."""

    # Clear the output
    clear_output(wait=True)

    # Quick extraction of scan info
    nexus = PN.PyNexusFile(expt.path)

    # Extract number of spectrums taken during the scan
    expt.nb_allspectrums = nexus.get_nbpts()
    print("There are %g spectrums in the scan."%(expt.nb_allspectrums))

    # Extract list of detector elements available
    stamps = nexus.extractStamps()
    fluospectrums_available = []
    for i in range(len(stamps)):
        if (stamps[i][1] != None and "fluospectrum0" in stamps[i][1].lower()):
            fluospectrums_available.append(stamps[i][1].lower()[-1])

    print("List of available elements: ", ["Element %s"%s for s in fluospectrums_available])

    def on_button_extract_clicked(b):
        """Extract the scan."""

        # Give the info that the extraction was done
        expt.is_extract_done = True

        # Update the parameters with current values
        update_params()

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_panel(expt)

        print("Extraction of:\n%s"%expt.path)

        print(PN._RED+"Wait for the extraction to finish..."+PN._RESET)

        # Load the file
        Extract_nexus(expt)

        # Set and plot the channels and spectrums subsets
        Set_subsets(expt)
        Plot_subsets(expt)



    def update_params():
        """Update the parameters with the current values"""

        expt.is_fluospectrum00 = w_is_fluospectrum00.value
        expt.is_fluospectrum01 = w_is_fluospectrum01.value
        expt.is_fluospectrum02 = w_is_fluospectrum02.value
        expt.is_fluospectrum03 = w_is_fluospectrum03.value
        expt.is_fluospectrum04 = w_is_fluospectrum04.value
        expt.ind_first_channel = w_ind_first_channel.value
        expt.ind_last_channel = w_ind_last_channel.value
        expt.ind_first_spectrum = w_ind_first_spectrum.value
        expt.ind_last_spectrum = w_ind_last_spectrum.value
        expt.gain = w_gain.value
        expt.eV0 = w_eV0.value
        expt.beam_energy = w_beam_energy.value
        expt.is_ipysheet = w_is_ipysheet.value
        expt.delimiter = w_delimiter.value
        expt.fitstuck_limit = w_fitstuck_limit.value
        expt.min_strength = w_min_strength.value
        expt.is_fast = w_is_fast.value
        expt.sl = w_sl.value
        expt.ct = w_ct.value
        expt.noise = w_noise.value
        expt.sfa0 = w_sfa0.value
        expt.sfa1 = w_sfa1.value
        expt.tfb0 = w_tfb0.value
        expt.tfb1 = w_tfb1.value
        expt.twc0 = w_twc0.value
        expt.twc1 = w_twc1.value
        expt.fG = w_fG.value
        expt.fA = w_fA.value
        expt.fB = w_fB.value
        expt.gammaA = w_gammaA.value
        expt.gammaB = w_gammaB.value
        expt.epsilon = w_epsilon.value
        expt.fano = w_fano.value
        expt.is_transmitted = w_is_transmitted.value
        expt.is_peaks_on_sum = w_is_peaks_on_sum.value
        expt.is_show_peaks = w_is_show_peaks.value
        expt.is_show_subfunctions = w_is_show_subfunctions.value

        # Particular case of list_isfit, going from str to array
        list_isfit = ['sl'*w_is_sl.value, 'ct'*w_is_ct.value, 'noise'*w_is_noise.value,
                      'sfa0'*w_is_sfa0.value, 'sfa1'*w_is_sfa1.value, 'tfb0'*w_is_tfb0.value, 'tfb1'*w_is_tfb1.value,
                      'twc0'*w_is_twc0.value, 'twc1'*w_is_twc1.value, 'fG'*w_is_fG.value,
                      'fA'*w_is_fA.value, 'fB'*w_is_fB.value, 'gammaA'*w_is_gammaA.value, 'gammaB'*w_is_gammaB.value,
                      'epsilon'*w_is_epsilon.value, 'fano'*w_is_fano.value]

        while("" in list_isfit) :
            list_isfit.remove("")

        expt.list_isfit = list_isfit
        expt.list_isfit_str = ','.join(list_isfit)

        # Prepare the header of the csv file
        with open(expt.working_dir+expt.id+'/Parameters.csv', "w", newline='') as f:
            writer = csv.writer(f,delimiter=';',dialect='excel')
            header = np.array([
                    'is_fluospectrum00',
                    'is_fluospectrum01',
                    'is_fluospectrum02',
                    'is_fluospectrum03',
                    'is_fluospectrum04',
                    'ind_first_channel',
                    'ind_last_channel',
                    'ind_first_spectrum',
                    'ind_last_spectrum',
                    'gain',
                    'eV0',
                    'beam_energy',
                    'is_ipysheet',
                    'delimiter',
                    'fitstuck_limit',
                    'min_strength',
                    'is_fast',
                    'list_isfit_str',
                    'sl',
                    'ct',
                    'noise',
                    'sfa0',
                    'sfa1',
                    'tfb0',
                    'tfb1',
                    'twc0',
                    'twc1',
                    'fG',
                    'fA',
                    'fB',
                    'gammaA',
                    'gammaB',
                    'epsilon',
                    'fano',
                    'is_transmitted',
                    'is_peaks_on_sum',
                    'is_show_peaks',
                    'is_show_subfunctions'
                    ])
            writer.writerow(header)

            writer.writerow([
                    expt.is_fluospectrum00,
                    expt.is_fluospectrum01,
                    expt.is_fluospectrum02,
                    expt.is_fluospectrum03,
                    expt.is_fluospectrum04,
                    expt.ind_first_channel,
                    expt.ind_last_channel,
                    expt.ind_first_spectrum,
                    expt.ind_last_spectrum,
                    expt.gain,
                    expt.eV0,
                    expt.beam_energy,
                    expt.is_ipysheet,
                    expt.delimiter,
                    expt.fitstuck_limit,
                    expt.min_strength,
                    expt.is_fast,
                    expt.list_isfit_str,
                    expt.sl,
                    expt.ct,
                    expt.noise,
                    expt.sfa0,
                    expt.sfa1,
                    expt.tfb0,
                    expt.tfb1,
                    expt.twc0,
                    expt.twc1,
                    expt.fG,
                    expt.fA,
                    expt.fB,
                    expt.gammaA,
                    expt.gammaB,
                    expt.epsilon,
                    expt.fano,
                    expt.is_transmitted,
                    expt.is_peaks_on_sum,
                    expt.is_show_peaks,
                    expt.is_show_subfunctions
                    ])


    # Load the scan info from file
    with open(expt.working_dir+expt.id+'/Parameters.csv', "r") as f:
        reader = csv.DictReader(f, delimiter=';',dialect='excel')
        for row in reader:
            is_fluospectrum00 = eval(row['is_fluospectrum00'])
            is_fluospectrum01 = eval(row['is_fluospectrum01'])
            is_fluospectrum02 = eval(row['is_fluospectrum02'])
            is_fluospectrum03 = eval(row['is_fluospectrum03'])
            is_fluospectrum04 = eval(row['is_fluospectrum04'])
            ind_first_channel = int(row['ind_first_channel'])
            ind_last_channel = int(row['ind_last_channel'])
            ind_first_spectrum = int(row['ind_first_spectrum'])
            ind_last_spectrum = int(row['ind_last_spectrum'])
            gain = float(row['gain'].replace(',', '.'))
            eV0 = float(row['eV0'].replace(',', '.'))
            beam_energy = float(row['beam_energy'].replace(',', '.'))
            is_ipysheet = eval(row['is_ipysheet'])
            delimiter = str(row['delimiter'])
            fitstuck_limit = int(row['fitstuck_limit'])
            min_strength = float(row['min_strength'].replace(',', '.'))
            is_fast = eval(row['is_fast'])
            list_isfit_str = str(row['list_isfit_str'])
            sl = float(row['sl'].replace(',', '.'))
            ct = float(row['ct'].replace(',', '.'))
            noise = float(row['noise'].replace(',', '.'))
            sfa0 = float(row['sfa0'].replace(',', '.'))
            sfa1 = float(row['sfa1'].replace(',', '.'))
            tfb0 = float(row['tfb0'].replace(',', '.'))
            tfb1 = float(row['tfb1'].replace(',', '.'))
            twc0 = float(row['twc0'].replace(',', '.'))
            twc1 = float(row['twc1'].replace(',', '.'))
            fG = float(row['fG'].replace(',', '.'))
            fA = float(row['fA'].replace(',', '.'))
            fB = float(row['fB'].replace(',', '.'))
            gammaA = float(row['gammaA'].replace(',', '.'))
            gammaB = float(row['gammaB'].replace(',', '.'))
            epsilon = float(row['epsilon'].replace(',', '.'))
            fano = float(row['fano'].replace(',', '.'))
            is_transmitted = eval(row['is_transmitted'])
            is_peaks_on_sum = eval(row['is_peaks_on_sum'])
            is_show_peaks = eval(row['is_show_peaks'])
            is_show_subfunctions = eval(row['is_show_subfunctions'])

    # convert list_isfit_str into a list
    list_isfit = [str(list_isfit_str.split(',')[i]) for i in range(len(list_isfit_str.split(',')))]


    w_is_fluospectrum00 = widgets.Checkbox(
        value=is_fluospectrum00,
        style=style,
        description='Element 0')

    w_is_fluospectrum01 = widgets.Checkbox(
        value=is_fluospectrum01,
        style=style,
        description='Element 1')

    w_is_fluospectrum02 = widgets.Checkbox(
        value=is_fluospectrum02,
        style=style,
        description='Element 2')

    w_is_fluospectrum03 = widgets.Checkbox(
        value=is_fluospectrum03,
        style=style,
        description='Element 3')

    w_is_fluospectrum04 = widgets.Checkbox(
        value=is_fluospectrum04,
        style=style,
        description='Element 4')

    w_ind_first_channel = widgets.BoundedIntText(
        value=ind_first_channel,
        min=0,
        max=2048,
        step=1,
        description='First channel',
        layout=widgets.Layout(width='200px'),
        style=style)

    w_ind_last_channel = widgets.BoundedIntText(
        value=ind_last_channel,
        min=0,
        max=2048,
        step=1,
        description='Last channel',
        layout=widgets.Layout(width='200px'),
        style=style)

    w_ind_first_spectrum = widgets.IntText(
        value=ind_first_spectrum,
        step=1,
        description='First spectrum',
        layout=widgets.Layout(width='200px'),
        style=style)

    w_ind_last_spectrum = widgets.IntText(
        value=ind_last_spectrum,
        step=1,
        description='Last spectrum',
        layout=widgets.Layout(width='200px'),
        style=style)

    w_gain = widgets.FloatText(
        value=gain,
        description='Gain',
        layout=widgets.Layout(width='120px'),
        style=style)

    w_eV0 = widgets.FloatText(
        value=eV0,
        description='eV0',
        layout=widgets.Layout(width='100px'),
        style=style)
    
    w_beam_energy = widgets.FloatText(
        value=beam_energy,
        description='Energy (eV)',
        layout=widgets.Layout(width='150px'),
        style=style)

    w_is_ipysheet = widgets.Checkbox(
        value=is_ipysheet,
        style=style,
        layout=widgets.Layout(width='150px'),
        description='Use ipysheet')

    w_delimiter = widgets.Text(
        value=delimiter,
        description='Delimiter',
        layout=widgets.Layout(width='100px'),
        style=style)

    w_fitstuck_limit = widgets.IntText(
        value=fitstuck_limit,
        description='Limit iter.',
        layout=widgets.Layout(width='150px'),
        style=style)

    w_min_strength = widgets.FloatText(
        value=min_strength,
        description='Strength min.',
        layout=widgets.Layout(width='180px'),
        style=style)

    # Fit params: boolean
    w_is_fast = widgets.Checkbox(
        value=is_fast,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='Fast extract')

    w_is_gammaA = widgets.Checkbox(
        value='gammaA' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='gammaA')

    w_is_gammaB = widgets.Checkbox(
        value='gammaB' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='gammaB')

    w_is_fA = widgets.Checkbox(
        value='fA' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='fA')

    w_is_fB = widgets.Checkbox(
        value='fB' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='fB')

    w_is_fG = widgets.Checkbox(
        value='fG' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='fG')

    w_is_twc0 = widgets.Checkbox(
        value='twc0' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='twc0')

    w_is_twc1 = widgets.Checkbox(
        value='twc1' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='twc1')

    w_is_tfb1 = widgets.Checkbox(
        value='tfb1' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='tfb1')

    w_is_tfb0 = widgets.Checkbox(
        value='tfb0' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='tfb0')

    w_is_sfa0 = widgets.Checkbox(
        value='sfa0' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='sfa0')

    w_is_sfa1 = widgets.Checkbox(
        value='sfa1' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='sfa1')

    w_is_sl = widgets.Checkbox(
        value='sl' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='sl')

    w_is_ct = widgets.Checkbox(
        value='ct' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='ct')

    w_is_noise = widgets.Checkbox(
        value='noise' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='noise')

    w_is_epsilon = widgets.Checkbox(
        value='epsilon' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='epsilon')

    w_is_fano = widgets.Checkbox(
        value='fano' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='fano')

    # Fit params: value

    w_gammaA = widgets.FloatText(
        value=gammaA,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='gammaA')

    w_gammaB = widgets.FloatText(
        value=gammaB,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='gammaB')

    w_fA = widgets.FloatText(
        value=fA,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='fA')

    w_fB = widgets.FloatText(
        value=fB,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='fB')


    w_fG = widgets.FloatText(
        value=fG,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='fG')

    w_twc0 = widgets.FloatText(
        value=twc0,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='twc0')

    w_twc1 = widgets.FloatText(
        value=twc1,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='twc1')

    w_tfb1 = widgets.FloatText(
        value=tfb1,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='tfb1')

    w_tfb0 = widgets.FloatText(
        value=tfb0,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='tfb0')

    w_sfa0 = widgets.FloatText(
        value=sfa0,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='sfa0')

    w_sfa1 = widgets.FloatText(
        value=sfa1,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='sfa1')

    w_sl = widgets.FloatText(
        value=sl,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='sl')

    w_ct = widgets.FloatText(
        value=ct,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='ct')

    w_noise = widgets.FloatText(
        value=noise,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='noise')

    w_epsilon = widgets.FloatText(
        value=epsilon,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='epsilon')

    w_fano = widgets.FloatText(
        value=fano,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='fano')

    w_is_transmitted = widgets.Checkbox(
        value=is_transmitted,
        style=style,
        layout=widgets.Layout(width='150px'),
        description='Transmit fit params')

    w_is_peaks_on_sum = widgets.Checkbox(
        value=is_peaks_on_sum,
        style=style,
        layout=widgets.Layout(width='150px'),
        description='Set peaks on sum')

    w_is_show_peaks = widgets.Checkbox(
        value=is_show_peaks,
        layout=widgets.Layout(width='120px'),
        style=style,
        description='Show peaks?')

    w_is_show_subfunctions = widgets.Checkbox(
        value=is_show_subfunctions,
        layout=widgets.Layout(width='170px'),
        style=style,
        description='Show sub-functions?')


    button_extract = widgets.Button(description="Extract the scan",layout=widgets.Layout(width='500px'))
    button_extract.on_click(on_button_extract_clicked)

    display(widgets.HBox([w_ind_first_channel, w_ind_last_channel, w_ind_first_spectrum, w_ind_last_spectrum]))

    w_fluospectrum = widgets.HBox([w_is_fluospectrum00, w_is_fluospectrum01,w_is_fluospectrum02,
                                   w_is_fluospectrum03,w_is_fluospectrum04])
    display(w_fluospectrum)

    print("-"*100)
    # Fit params
    display(widgets.HBox([w_is_sl, w_is_ct, w_is_noise, w_is_sfa0, w_is_sfa1, w_is_tfb0, w_is_tfb1,w_is_twc0]))
    display(widgets.HBox([w_is_twc1, w_is_fG, w_is_fA, w_is_fB, w_is_gammaA,w_is_gammaB, w_is_epsilon, w_is_fano]))

    display(widgets.HBox([w_sl, w_ct, w_noise, w_fG]))
    display(widgets.HBox([w_sfa0, w_sfa1, w_tfb0, w_tfb1]))
    display(widgets.HBox([w_twc0, w_twc1, w_epsilon, w_fano]))
    display(widgets.HBox([w_fA, w_fB, w_gammaA,w_gammaB]))

    print("-"*100)
    display(widgets.HBox([w_gain, w_eV0, w_beam_energy, w_delimiter, w_fitstuck_limit, w_min_strength]))
    display(widgets.HBox([w_is_ipysheet, w_is_fast,
                          w_is_transmitted, w_is_peaks_on_sum, w_is_show_peaks, w_is_show_subfunctions]))

    display(widgets.HBox([button_extract]))



def Set_scan_identifiers(expt):
    """
    Create a series of identifiers for the current scan.
    """
    # For example:
    # expt.nxs = 'SIRIUS_2017_12_11_08042.nxs'
    # expt.path = '/Users/arnaudhemmerle/recording/SIRIUS_2017_12_11_08042.nxs'
    # expt.id = 'SIRIUS_2017_12_11_08042'
    # expt.number = 8042

    expt.path = expt.recording_dir+expt.nxs
    expt.id = expt.nxs[:-4]
    split_name = expt.nxs.split('.')[0].split('_')
    expt.number = int(expt.nxs.split('.')[0].split('_')[-1])


def Extract_nexus(expt):
    """
    1) Extract all the chosen fluospectrums in the nexus file.
    2) Correct with ICR/OCR.
    3) Sum the fluospectrums and put them in scan.allspectrums_corr
    """

    def extract_and_correct(ind_spectrum):
        """Extract the fluospectrum of index ind_spectrum from the nexus file and correct it with ICR/OCR"""

        for i in range(len(stamps)):
            if (stamps[i][1] != None and stamps[i][1].lower() == "fluoicr0"+ind_spectrum):
                fluoicr = data[i]
            if (stamps[i][1] != None and stamps[i][1].lower() == "fluoocr0"+ind_spectrum):
                fluoocr = data[i]
            if (stamps[i][1] != None and stamps[i][1].lower() == "fluospectrum0"+ind_spectrum):
                fluospectrum = data[i]
            if (stamps[i][1] == None and stamps[i][0].lower() == "integration_time"):
                integration_time = data[i]

        ICR = fluoicr
        try:
            OCR = fluoocr
        except:
            # If OCR is not in the data, calculate it.
            print(PN._RED+"OCR not found in data. Taking OCR = spectrum_intensity/counting_time."+PN._RESET)
            OCR = np.array([np.sum(fluospectrum[n])/integration_time[n] for n in range(len(fluospectrum))])

        ratio = np.array([ICR[n]/OCR[n] if (~np.isclose(OCR[n],0.) & ~np.isnan(OCR[n]) & ~np.isnan(ICR[n]))
                         else 0. for n in range(len(ICR))])
        spectrums_corr = np.array([fluospectrum[n]*ratio[n] for n in range(len(ratio))])

        return spectrums_corr

    expt.nexus = PN.PyNexusFile(expt.path, fast=expt.is_fast)

    stamps, data= expt.nexus.extractData()

    # Extract timestamps
    for i in range(len(stamps)):
        if (stamps[i][1]== None and stamps[i][0]=='sensorsRelTimestamps'):
            expt.allsensorsRelTimestamps = data[i]

    # Get the chosen fluospectrums
    expt.fluospectrums_chosen = np.array([expt.is_fluospectrum00,expt.is_fluospectrum01,
                                     expt.is_fluospectrum02,expt.is_fluospectrum03, expt.is_fluospectrum04])


    # Correct each chosen fluospectrum with ICR/OCR and sum them
    allspectrums_corr = np.zeros((expt.nb_allspectrums, 2048))

    if expt.is_fluospectrum00:
        fluospectrum00 = extract_and_correct('0')
        allspectrums_corr  = allspectrums_corr  + fluospectrum00
    if expt.is_fluospectrum01:
        fluospectrum01 = extract_and_correct('1')
        allspectrums_corr  = allspectrums_corr  + fluospectrum01
    if expt.is_fluospectrum02:
        fluospectrum02 = extract_and_correct('2')
        allspectrums_corr  = allspectrums_corr  + fluospectrum02
    if expt.is_fluospectrum03:
        fluospectrum03 = extract_and_correct('3')
        allspectrums_corr  = allspectrums_corr  + fluospectrum03
    if expt.is_fluospectrum04:
        fluospectrum04 = extract_and_correct('4')
        allspectrums_corr  = allspectrums_corr  + fluospectrum04

    expt.allspectrums_corr = allspectrums_corr
    expt.nexus.close()

def Set_subsets(expt):
    """
    Select spectrums and channel range for the fits.
    """

    # Look for subsets of consecutive non-empty spectrums
    ind_non_zero_spectrums = np.where(np.sum(expt.allspectrums_corr, axis = 1)>10.)[0]
    list_ranges = np.split(ind_non_zero_spectrums, np.where(np.diff(ind_non_zero_spectrums) != 1)[0]+1)
    expt.last_non_zero_spectrum = ind_non_zero_spectrums[-1]

    #for ranges in list_ranges:
        #print('Recommended spectrum range: [%g:%g]'%(ranges[0],ranges[-1]))
    print('File empty after spectrum %g.'%ind_non_zero_spectrums[-1])

    # Subset of channels and spectrums defined by user
    expt.channels = np.arange(expt.ind_first_channel, expt.ind_last_channel+1)
    expt.spectrums = expt.allspectrums_corr[expt.ind_first_spectrum:expt.ind_last_spectrum+1,
                                                      expt.ind_first_channel:expt.ind_last_channel+1]

    # Subset of timestamps (for later saving of the data)
    expt.sensorsRelTimestamps = expt.allsensorsRelTimestamps[expt.ind_first_spectrum:expt.ind_last_spectrum+1]

def Plot_subsets(expt):
    """
    Plot the whole spectrum range (stopping at the last non-zero spectrum).
    Used to check which subset the user wants.
    """

    fig = plt.figure(figsize=(12,6))
    fig.suptitle(expt.nxs, fontsize=14)
    ax1 = fig.add_subplot(111)
    ax1.set_title('All the spectrums in the file (stopping at the last non-zero spectrum)')
    ax1.set(xlabel = 'spectrum index', ylabel = 'channel')
    ax1.set_xlim(left = -1, right = expt.last_non_zero_spectrum+1)
    ax1.axvline(expt.ind_first_spectrum, linestyle = '--', color = 'y', label = 'Selected spectrum range')
    ax1.axvline(expt.ind_last_spectrum, linestyle = '--', color = 'y')
    im1 = ax1.imshow(expt.allspectrums_corr.transpose(), cmap = 'viridis', aspect = 'auto', norm=mplcolors.LogNorm())
    plt.legend()

    # Plot the whole channel range
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Whole range of channels on the sum of all spectrums')
    ax1.set(xlabel = 'channel', ylabel = 'counts')
    ax1.axvline(expt.ind_first_channel, linestyle = '--', color = 'r', label = 'Selected channel range')
    ax1.axvline(expt.ind_last_channel, linestyle = '--', color = 'r')
    ax1.plot(np.arange(2048), expt.allspectrums_corr.sum(axis = 0), 'k.-')
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = fig.add_subplot(212)
    ax2.set(xlabel = 'channel', ylabel = 'counts')
    ax2.axvline(expt.ind_first_channel, linestyle = '--', color = 'r')
    ax2.axvline(expt.ind_last_channel, linestyle = '--', color = 'r')
    ax2.plot(np.arange(2048), expt.allspectrums_corr.sum(axis = 0), 'k.-')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom = 1)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.subplots_adjust(hspace=.0)

    #Plot the selected spectrum range
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('SELECTED RANGES', fontsize=14)
    ax1 = fig.add_subplot(111)
    ax1.set_title('Subset of spectrums [%g:%g]'%(expt.ind_first_spectrum,expt.ind_last_spectrum))
    ax1.set(xlabel = 'spectrum index', ylabel = 'channel')
    im1 = ax1.imshow(expt.spectrums.transpose(), cmap = 'viridis', aspect = 'auto', norm=mplcolors.LogNorm(),
                     interpolation='none',
                     extent=[expt.ind_first_spectrum,expt.ind_last_spectrum,
                             expt.ind_last_channel,expt.ind_first_channel])

    #Plot the selected channel range
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Subset of channels [%g:%g]'%(expt.ind_first_channel,expt.ind_last_channel))
    ax1.set(xlabel = 'channel', ylabel = 'counts')
    ax1.plot(expt.channels, expt.spectrums[0], 'r-', label = 'Spectrum %g'%expt.ind_first_spectrum)
    ax1.plot(expt.channels, expt.spectrums[-1], 'b-', label = 'Spectrum %g'%expt.ind_last_spectrum)
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = fig.add_subplot(212)
    ax2.set(xlabel = 'channel', ylabel = 'counts')
    ax2.plot(expt.channels, expt.spectrums[0], 'r-')
    ax2.plot(expt.channels, expt.spectrums[-1], 'b-')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom = 1)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.subplots_adjust(hspace=.0)


def Set_peaks(expt):
    """
    1) Check if the csv file Peaks.csv exists, if not copy DefaultPeaks.csv in the expt folder
    2) If ipysheet is activated, display the interactive sheet. If not, extract the peaks from Peaks.csv
    3) Save the peaks in Peaks.csv
    Update scan.arr_peaks, with the info on the peaks later used to create the Group and Peak objects.
    """

    # Array which will contain the info on peaks
    arr_peaks = np.array([])

    # Check if the csv file already exists, if not copy the DefaultPeaks.csv file
    if not os.path.isfile(expt.working_dir+expt.id+'/Peaks.csv'):
        shutil.copy('DefaultPeaks.csv', expt.working_dir+expt.id+'/Peaks.csv')

    with open(expt.working_dir+expt.id+'/Peaks.csv', "r") as f:
        csvreader = csv.reader(f, delimiter=expt.delimiter)
        # First line is the header
        expt.peaks_header = next(csvreader)
        nb_columns = len(expt.peaks_header)
        for row in csvreader:
            arr_peaks = np.append(arr_peaks, row)
    arr_peaks = np.reshape(arr_peaks, (len(arr_peaks)//nb_columns,nb_columns))

    # String to print the peaks
    prt_peaks = '\t'.join([str(cell) for cell in expt.peaks_header])+'\n'
    prt_peaks += '\n'.join(['\t \t'.join([str(cell[0:7]) for cell in row]) for row in arr_peaks if row[0]!=''])+'\n'
    prt_peaks += "Peaks saved in:\n%s"%(expt.working_dir+expt.id+'/Peaks.csv')

    expt.arr_peaks = arr_peaks
    expt.prt_peaks = prt_peaks

    # Determine the number of rows to have a fixed number of empty rows
    nb_filled_rows = len([elem for elem in arr_peaks if elem[0]!=''])
    nb_empty_rows = len([elem for elem in arr_peaks if elem[0]==''])
    if nb_empty_rows<15:
        nb_rows = nb_filled_rows+15
    else:
        nb_rows = np.shape(arr_peaks)[0]


    if expt.is_ipysheet:
        sheet = ipysheet.easy.sheet(columns=nb_columns, rows=nb_rows ,column_headers = expt.peaks_header)

        # ipysheet does not work correctly with no entries
        # it is necessary to fill first the cells with something
        nb_rows_to_fill = nb_rows - np.shape(arr_peaks)[0]
        fill = np.reshape(np.array(nb_rows_to_fill*nb_columns*['']),
                          (nb_rows_to_fill, nb_columns))
        arr_peaks = np.vstack((arr_peaks, fill))

        for i in range(nb_columns):
            ipysheet.easy.column(i,  arr_peaks[:,i])

        def on_button_update_clicked(b):
            """
            Update the peaks in the sheet by writing in Peaks.csv and re-displaying the peaks and panel
            """

            # Clear the plots and reput the boxes
            clear_output(wait=True)
            Display_panel(expt)

            # Give the info that the set peaks was done
            expt.is_set_done = True

            # Collect the info from the sheet, store them in arr_peaks, write to Peaks.csv
            arr_peaks = ipysheet.numpy_loader.to_array(ipysheet.easy.current())

            with open(expt.working_dir+expt.id+'/Peaks.csv', "w", newline='') as f:
                writer = csv.writer(f,delimiter=expt.delimiter)
                writer.writerow(expt.peaks_header)
                writer.writerows(arr_peaks)

            # Reput the sheet set peaks
            Set_peaks(expt)

            # Extract the info from the sheet
            Extract_groups(expt)

            # Display the peaks on the selected spectrum or on the sum
            if expt.is_peaks_on_sum:
                Display_peaks(expt)
            else:
                w = widgets.interact(Display_peaks,
                                     expt=widgets.fixed(expt),
                                     spectrum_index=widgets.IntText(value=0, step=1, description='Spectrum:'))

            expt.arr_peaks = arr_peaks

        def on_button_add_from_db_clicked(b):
            """
            Add a peak from the database.
            The strength of each peak is calculated via:
            - the line intensity relative to the group of peaks with the same initial level (ex. L1)
            - the jump ratio of the initial level
            - the fluo yield of the initial level
            - the probability to transition to another level
            Selection is made through nested interactive widgets.
            The selected group of peaks is then written in Peaks.csv.
            """

            # Clear the plots and reput the boxes
            clear_output(wait=True)
            Display_panel(expt)

            # Create the list of atoms based on the database
            atom_name_list = [str(xraylib.AtomicNumberToSymbol(i)) for i in range(1,99)]

            # Construct an array for extracting line names from xraylib
            line_names = []
            with open('xraylib_lines.pro', "r") as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    if row!=[]:
                        if (row[0][0]!=';' and row[0]!='end'):
                            line_names.append(row[0].split(' = ')[0].split('_')[0])
            
            def select_group(atom_chosen):
                """
                Widget to show and select the group of peaks to add (from 'K', 'L', or 'M')
                """

                tbw = []
                
                def print_group(group_chosen):
                    """
                    Print the group of peaks.
                    """
                    print('Peaks to be added:')
                    print('')
                    
                    if group_chosen == 'K':
                        ind_min = -29
                        ind_max = 0
                    if group_chosen == 'L':
                        ind_min = -113
                        ind_max = -29
                    if group_chosen == 'M':
                        ind_min = -219
                        ind_max = -113
                    for i in range(ind_min,ind_max):

                        Z = xraylib.SymbolToAtomicNumber(atom_chosen)
                        try:
                            strength =  xraylib.CS_FluorLine_Kissel(Z, i, expt.beam_energy/1000.)
                            energy = xraylib.LineEnergy(Z, i)*1000.

                            # Put an absolute cut-off on the strength
                            if strength>expt.min_strength:          
                                # Array to be written
                                tbw.append([atom_chosen,
                                            line_names[-i-1],
                                            '{:.1f}'.format(energy),
                                            '{:f}'.format(strength),'no','yes'])
                                print('Line name: %s, energy (eV): %g, strength: %g'%(line_names[-i-1], energy, strength))
                        except:
                            pass


                    def on_button_add_group_clicked(b):
                        """
                        Add the group of peaks to the file "Peaks.csv"
                        """

                        # Rewrite the previous peaks (without the empty lines)
                        with open(expt.working_dir+expt.id+'/Peaks.csv', "w", newline='') as f:
                            writer = csv.writer(f,delimiter=expt.delimiter)
                            writer.writerow(expt.peaks_header)
                            writer.writerows([elem for elem in expt.arr_peaks_all if elem[0]!=''])


                        # Add the new lines
                        with open(expt.working_dir+expt.id+'/Peaks.csv', "a", newline='') as f:
                            writer = csv.writer(f,delimiter=expt.delimiter)
                            writer.writerows(tbw)

                        # Reconstruct expt.arr_peaks with the new lines
                        arr_peaks_all = np.array([])
                        with open(expt.working_dir+expt.id+'/Peaks.csv', "r") as f:
                            csvreader = csv.reader(f, delimiter=expt.delimiter)
                            # First line is the header
                            expt.peaks_header = next(csvreader)
                            for row in csvreader:
                                arr_peaks_all = np.append(arr_peaks_all, row)

                        arr_peaks_all = np.reshape(arr_peaks_all, (len(arr_peaks_all)//nb_columns,nb_columns))
                        expt.arr_peaks_all = arr_peaks_all

                        print(PN._RED+"Done!"+PN._RESET)
                        print(PN._RED+"Click on Set peaks to check peaks."+PN._RESET)
                        print(PN._RED+"Click on Start fit to directly start the fit."+PN._RESET)


                    # Button to add the displayed group of peaks
                    button_add_group = widgets.Button(description="Add group of peaks",layout=widgets.Layout(width='300px'))
                    button_add_group.on_click(on_button_add_group_clicked)

                    display(button_add_group)

                w_print_group =  widgets.interact(print_group,
                                group_chosen = widgets.Dropdown(
                                options=['K', 'L', 'M'],
                                description='Select level:',
                                layout=widgets.Layout(width='200px'),
                                style=style)
                                )

            w_select_atom = widgets.interact(select_group,
                                atom_chosen = widgets.Dropdown(
                                options=atom_name_list,
                                value = 'Ar',
                                description='Select atom:',
                                layout=widgets.Layout(width='200px'),
                                style=style)
                                )


        button_update = widgets.Button(description="Update peaks")
        button_update.on_click(on_button_update_clicked)

        button_add_from_db = widgets.Button(description="Add peaks from database", layout=widgets.Layout(width='300px'))
        button_add_from_db.on_click(on_button_add_from_db_clicked)

        display(sheet)
        display(widgets.HBox([button_update, button_add_from_db]))

    else:
        print("Peaks imported from %s"%(expt.working_dir+expt.id+'/Peaks.csv'))
        print('\t'.join([str(cell) for cell in expt.peaks_header]))
        print('\n'.join(['\t \t'.join([str(cell) for cell in row]) for row in arr_peaks if row[0]!='']))

def Extract_groups(expt):
    """
    Create objects Group and Peak from info in scan.arr_peaks
    """

    # Create objects Group and Peak from info in arr_peaks
    # Peaks are grouped by lines belonging to the same fluo element and with the same family K L or M
    # Each "ElemName+FamilyName" gives a new Group with Group.name = "ElemName_FamilyName"
    # Each "LineName" gives a new Group.Peak with Group.Peak.name = "LineName"
    # "Position (eV)" -> Group.Peak.position_init
    # "Strength" -> Group.Peak.strength
    # "Fit position?" -> Group.Peak.is_fitpos
    # The array scan.Groups contains the list of objects Group

    # Indicator for the panel
    expt.is_fit_ready = True

    # Remove the peaks which are not fitted from scan.arr_peaks
    expt.arr_peaks_all = expt.arr_peaks
    expt.arr_peaks = expt.arr_peaks[np.where(expt.arr_peaks[:,5]!='no')]

    # List of groups of peaks (same elem, same family K L or M)
    Groups = []

    ###################################################################
    # Construct the groups and lines
    for i in range(np.shape(expt.arr_peaks)[0]):

        elem_name = expt.arr_peaks[i][0]

        if elem_name != '':

            line_name = expt.arr_peaks[i][1]

            # Determine the family name from the line name
            if line_name[0] in ['K', 'L', 'M']:
                family_name = line_name[0]
            else:
                family_name = line_name

            # Define the name of a group of peaks
            group_name = elem_name+'_'+family_name


            # Check if the group has already been created
            is_new_group = True
            for group in Groups:
                if group_name == group.name:
                    is_new_group = False

            # Create a new group
            if is_new_group:
                newGroup = Group(group_name)
                Groups = np.append(Groups, newGroup)
                Groups[-1].elem_name = elem_name
                Groups[-1].peaks = []

            # Convert yes/no in True/False
            if expt.arr_peaks[i][4] == 'yes':
                is_fitpos = True
            else:
                is_fitpos = False

            # Add the peak to the right group
            for group in Groups:
                if group_name == group.name:
                    newPeak = Peak(
                    name = str(expt.arr_peaks[i][1]),
                    position_init = float(expt.arr_peaks[i][2]),
                    strength = float(expt.arr_peaks[i][3]),
                    is_fitpos = is_fitpos)

                    group.peaks = np.append(group.peaks, newPeak)

    expt.groups = Groups

    ###################################################################
    # Compute the relative intensity (relative to the most intense peak,
    # i.e. intensity_rel = 1 for the most intense line of a given family K L or M)

    for group in expt.groups:
        max_strength = 0.

        # Extract the most intense strength of the group
        for peak in group.peaks:
            if peak.strength>max_strength:
                max_strength = peak.strength

        # Normalize the strengths wit the most intense one
        for peak in group.peaks:
            peak.intensity_rel = peak.strength/max_strength



def Validate_sheet(expt):
    """
    Validate the info in the current sheet by transferring them to scan.arr_peaks and save them in Peaks.csv
    """

    # Collect the info from the sheet, store them in arr_peaks, write to Peaks.csv
    arr_peaks = ipysheet.numpy_loader.to_array(ipysheet.easy.current())

    with open(expt.working_dir+expt.id+'/Peaks.csv', "w", newline='') as f:
        writer = csv.writer(f,delimiter=expt.delimiter)
        writer.writerow(expt.peaks_header)
        writer.writerows(arr_peaks)

    # String to print the peaks
    prt_peaks = '\t'.join([str(cell) for cell in expt.peaks_header])+'\n'
    prt_peaks += '\n'.join(['\t \t'.join([str(cell) for cell in row]) for row in arr_peaks if row[0]!=''])+'\n'
    prt_peaks += "Peaks saved in:\n%s"%(expt.working_dir+expt.id+'/Peaks.csv')

    expt.arr_peaks = arr_peaks
    expt.prt_peaks = prt_peaks


def Display_peaks(expt, spectrum_index=0):
    """
    Plot the position of each peaks on the given spectrum spectrum_index (or the sum).
    Take spectrum_index, the index of which spectrum you want to use.
    """

    # Convert channels into eV
    expt.eV = expt.channels*expt.gain + expt.eV0

    if expt.is_peaks_on_sum:
        # We work on the sum to define the peaks
        spectrum = np.sum(expt.spectrums, axis=0)
    else:
        # We work on the spectrum specified by spectrum_index
        spectrum = expt.spectrums[spectrum_index]

    # Plot the spectrum and each line given by the user
    Plot_spectrum(expt, spectrum_index=spectrum_index)

    if expt.is_ipysheet: print(expt.prt_peaks)


def Plot_spectrum(expt, spectrum_index=0, dparams_list=None):
    """
    Plot data of a specific spectrum (given by spectrum_index).
    If a dparams_list is given, redo and plot the fit with the given parameters.
    """
    n = spectrum_index
    eV = expt.eV
    groups = expt.groups

    if dparams_list != None:
        # We work on the spectrum specified by spectrum_index
        spectrum = expt.spectrums[n]

    else:
        if expt.is_peaks_on_sum:
            # We work on the sum to define the peaks
            spectrum = np.sum(expt.spectrums, axis=0)
        else:
            # We work on the spectrum specified by spectrum_index
            spectrum = expt.spectrums[n]


    if dparams_list != None:
        sl_list = dparams_list['sl_list']
        ct_list = dparams_list['ct_list']

        # Fit the spectrum with the given parameters
        for group in groups:
            group.area = group.area_list[n]
            for peak in group.peaks:
                peak.position = peak.position_list[n]
        dparams = {}
        for name in dparams_list:
            dparams[name[:-5]] = dparams_list[name][n]
        spectrum_fit, gau_tot, she_tot, tail_tot, baseline, compton = AF.Fcn_spectrum(dparams, groups, eV)

    else:
        for group in groups:
            for peak in group.peaks:
                peak.position = peak.position_init

    # Plot the whole spectrum
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(211)
    colors = iter(['#006BA4', '#FF800E', '#ABABAB', '#595959', 'k', '#C85200', 'b', '#A2C8EC', '#FFBC79']*200)
    linestyles = iter(['--', '-.', '-', ':']*400)
    ax1.set(xlabel = 'E (eV)', ylabel = 'counts')


    elem_dict = {}
    for group in expt.groups:

        # To have one color/label per elem
        if group.elem_name in elem_dict.keys():
            color = elem_dict[group.elem_name][0]
            linestyle = elem_dict[group.elem_name][1]
            is_first_group_of_elem = False

        else:
            color = next(colors)
            linestyle = next(linestyles)
            elem_dict.update({group.elem_name:(color,linestyle)})
            is_first_group_of_elem = True

        is_first_peak_of_group = True

        for peak in group.peaks:
            position = peak.position
            if (is_first_peak_of_group and is_first_group_of_elem):
                if expt.is_show_peaks:
                    # Plot the peak only if asked
                    ax1.axvline(x = position,  color = color, linestyle = linestyle, label = group.elem_name)
                    is_first_peak_of_group = False
            else:
                if expt.is_show_peaks:
                    ax1.axvline(x = position,  color = color, linestyle = linestyle, label = '')

    ax1.plot(eV, spectrum, 'k.')
    if dparams_list != None: ax1.plot(eV, spectrum_fit, 'r-', linewidth = 2)
    if expt.is_show_peaks:  ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)
    for item in ([ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(14)


    ax2 = fig.add_subplot(212)
    colors = iter(['#006BA4', '#FF800E', '#ABABAB', '#595959', 'k', '#C85200', 'b', '#A2C8EC', '#FFBC79']*200)
    linestyles = iter(['--', '-.', '-', ':']*400)
    ax2.set(xlabel = 'E (eV)', ylabel = 'counts')
    for group in groups:

        color = elem_dict[group.elem_name][0]
        linestyle = elem_dict[group.elem_name][1]

        for peak in group.peaks:
            position = peak.position
            if expt.is_show_peaks:
                # Plot the peak only if asked, and if its strength is > than a min value
                ax2.axvline(x = position,  color = color, linestyle = linestyle)

    ax2.plot(eV, spectrum, 'k.')
    if dparams_list != None:
        ax2.plot(eV, spectrum_fit, 'r-', linewidth = 2)
        if expt.is_show_subfunctions:
            ax2.plot(eV,gau_tot, 'm--', label = 'Gaussian')
            ax2.plot(eV,she_tot, 'g-',label = 'Step')
            ax2.plot(eV,tail_tot, 'b-', label = 'Low energy tail')
            ax2.plot(eV,baseline, 'k-',label = 'Continuum')
            ax2.legend(loc = 0)
    ax2.set_ylim(bottom = 1)
    ax2.set_yscale('log')
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    for item in ([ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(14)

    plt.subplots_adjust(hspace=.0)
    fig.subplots_adjust(top=0.95)
    if (expt.is_peaks_on_sum and dparams_list==None):
        fig.suptitle(expt.nxs+': Sum of spectrums', fontsize=14)
    else:
        fig.suptitle(expt.nxs+': Spectrum number %g/%g'%(n,(len(expt.spectrums)-1)), fontsize=14)


    # Plot each peak
    colors = iter(['#006BA4', '#FF800E', '#ABABAB', '#595959', '#C85200', 'b', '#A2C8EC', '#FFBC79']*200)
    linestyles = iter(['-.', '-', ':']*400)

    count = 0
    for group in groups:
        for peak in group.peaks:
            if count%2==0: fig = plt.figure(figsize=(14,4.7))
            plt.subplot(1, 2, count%2+1)

            position = peak.position
            position_init = peak.position_init

            ind_min = np.argmin(np.abs(np.array(eV)-0.9*position))
            ind_max = np.argmin(np.abs(np.array(eV)-1.1*position))

            spectrum_zoom = spectrum[ind_min:ind_max]
            eV_zoom = eV[ind_min:ind_max]

            if dparams_list != None:
                spectrum_fit_zoom = spectrum_fit[ind_min:ind_max]
                intensity_rel = peak.intensity_rel
                area = group.area_list[n]

                if peak.is_fitpos:
                    title0 = 'position(init) = %g eV, position(fit)=%g eV'%(position_init,position)
                else:
                    title0 = 'position = %g eV'%(position)

                title = group.elem_name + ' ' + peak.name + '\n' \
                        +'group area = %g, relative int = %g'%(area,intensity_rel) + '\n'\
                        + title0
            else:
                title = group.elem_name + ' ' + peak.name +'\n'+'position = %g eV'%(position)

            plt.gca().set_title(title)

            plt.plot(eV_zoom, spectrum_zoom, 'k.')
            if dparams_list != None: plt.plot(eV_zoom, spectrum_fit_zoom, 'r-', linewidth = 2)
            plt.xlabel('E (eV)')

            # Plot each line in the zoom
            for group_tmp in groups:
                for peak_tmp in group_tmp.peaks:
                    position_tmp = peak_tmp.position
                    if (eV[ind_min]<position_tmp and eV[ind_max]>position_tmp):
                        if (group_tmp.name==group.name and peak_tmp.name == peak.name):
                            color = 'k'
                            linestyle = '--'
                        else:
                            color = next(colors)
                            linestyle = next(linestyles)

                        plt.axvline(x = position_tmp , label = group_tmp.elem_name+' '+peak_tmp.name,
                                    linestyle = linestyle, color = color)
            plt.legend()

            if  count%2==1: plt.show()
            count+=1

    # If there was an odd number of plots, add a blank figure
    if count%2==1:
        plt.subplot(122).axis('off')
        plt.show()


def Plot_fit_results(expt, spectrum_index=None, dparams_list=None, is_save=False):
    """
    Plot all the params in dparams_list, as a function of the spectrum.
    If spectrum_index is given, plot its position on each plot.
    If is_save, save each plot in a png.
    """
    groups = expt.groups
    spectrums = expt.spectrums

    scans = np.arange(np.shape(spectrums)[0])


    # Plot areas & save plots
    is_title = True
    elem_already_plotted = []
    for group in groups:

        if group.elem_name not in elem_already_plotted:

            elem_already_plotted = np.append(elem_already_plotted, group.elem_name)

            fig, ax = plt.subplots(figsize=(15,4))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))


            # To group elem on the same plot
            for group_tmp in groups:
                if group_tmp.elem_name == group.elem_name:
                    list_lines_str = '['+' '.join([p.name for p in group_tmp.peaks])+']'
                    plt.plot(scans, group_tmp.area_list, '.-', label = 'Area %s %s'%(group_tmp.elem_name,list_lines_str))

            plt.legend(bbox_to_anchor=(0,1.02,1,0.2),loc = 'lower left',ncol = 5)
            plt.tight_layout()

            if is_save: plt.savefig(expt.working_dir+expt.id+'/area_'+group.elem_name+'.png')
            if spectrum_index!=None: plt.axvline(x = spectrum_index, linestyle = '--', color = 'black')
            if is_title:
                ax.set_title('AREAS\n\n')
                ax.title.set_fontsize(18)
                ax.title.set_fontweight('bold')
                is_title = False

            #ax.set_ylim(bottom=ax.get_ylim()[0]*0.7, top=ax.get_ylim()[1]*1.3)
            plt.show()

    # Plot positions & save plots
    is_title = True
    for group in groups:
        for peak in group.peaks:
            if peak.is_fitpos:
                fig, ax = plt.subplots(figsize=(15,4))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                plt.plot(scans, peak.position_list, 'b.-', label = 'Position %s '%(group.elem_name+'.'+peak.name))
                plt.legend()
                if is_save: plt.savefig(expt.working_dir+expt.id+'/position_'+group.elem_name+'_'+peak.name+'.png')
                if is_title:
                    ax.set_title('POSITIONS\n\n')
                    ax.title.set_fontsize(18)
                    ax.title.set_fontweight('bold')
                    is_title = False
                if spectrum_index!=None: plt.axvline(x = spectrum_index, linestyle = '--', color = 'black')
                plt.show()

    # Plot other params & save plots
    # Plot only the params which were fitted
    is_title = True
    for name in dparams_list:
        if name[:-5] in expt.list_isfit:
            fig, ax = plt.subplots(figsize=(15,4))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            plt.plot(scans, dparams_list[name], 'k.-', label = name[:-5])
            plt.legend()
            if is_save: plt.savefig(expt.working_dir+expt.id+'/'+str(name[:-5])+'.png')
            if is_title:
                ax.set_title('OTHER PARAMETERS\n\n')
                ax.title.set_fontsize(18)
                ax.title.set_fontweight('bold')
                is_title = False
            if spectrum_index!=None: plt.axvline(x = spectrum_index, linestyle = '--', color = 'black')
            plt.show()

def Choose_spectrum_to_plot(expt):
    """
    Select a spectrum to plot with its fit.
    """

    def on_button_add_clicked(b):
        """Add the plot to the report."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_panel(expt)

        expt.is_show_peaks = w_is_show_peaks.value
        expt.is_show_subfunctions = w_is_show_subfunctions.value

        code = 'FF.Load_results(expt, spectrum_index='+str(w_index.value)+')'
        Create_cell(code=code, position='below', celltype='code', is_print=True, is_execute=True)

    def on_button_display_clicked(b):
        """Display the selected plot to the report"""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_panel(expt)

        expt.is_show_peaks = w_is_show_peaks.value
        expt.is_show_subfunctions = w_is_show_subfunctions.value

        display(widgets.HBox([w_index, w_is_show_peaks, w_is_show_subfunctions, button_display]))
        display(button_add)

        # Plot the spectrum and fit
        Load_results(expt, w_index.value)


    w_index = widgets.IntText(description="Spectrum:",
                              style=style,
                              layout=widgets.Layout(width='200px'))

    w_is_show_peaks = widgets.Checkbox(
                              description='Show peaks?',
                              value=expt.is_show_peaks,
                              style=style,
                              layout=widgets.Layout(width='120px'))

    w_is_show_subfunctions = widgets.Checkbox(
                          description='Show sub-functions?',
                          value=expt.is_show_subfunctions,
                          style=style,
                          layout=widgets.Layout(width='170px'))

    button_display = widgets.Button(description="Preview the selected plot",layout=widgets.Layout(width='300px'))
    button_display.on_click(on_button_display_clicked)

    display(widgets.HBox([w_index, w_is_show_peaks, w_is_show_subfunctions, button_display]))

    button_add = widgets.Button(description="Add the selected plot",layout=widgets.Layout(width='300px'))
    button_add.on_click(on_button_add_clicked)
    display(button_add)

def Load_results(expt, spectrum_index=0):
    """
    Load and plot the results of a previous fit.
    Redo the fit with all the results from FitResults.csv
    """
    groups = expt.groups

    dparams_list = {'sl_list','ct_list',
                    'sfa0_list','sfa1_list','tfb0_list','tfb1_list',
                    'twc0_list','twc1_list',
                    'noise_list','fano_list', 'epsilon_list',
                    'fG_list'
                    ,'fA_list','fB_list','gammaA_list','gammaB_list'
                    }

    # Init all the lists
    dparams_list = dict.fromkeys(dparams_list, np.array([]))

    for group in groups:
        group.area_list = np.array([])
        for peak in group.peaks:
            peak.intensity_rel_list = np.array([])
            peak.position_list = np.array([])


    with open(expt.working_dir+expt.id+'/FitResults.csv', "r") as f:
        reader = csv.DictReader(f, delimiter=expt.delimiter)
        for row in reader:
            for group in groups:
                group.area_list = np.append(group.area_list, np.float(row['#'+group.name+'.area'].replace(',','.')))

                for peak in group.peaks:
                    peak.position_list = np.append(peak.position_list,
                                       np.float(row['#'+group.elem_name+'_'+peak.name+'.position'].replace(',','.')))

            for name in dparams_list:
                    dparams_list[name] = np.append(dparams_list[name], np.float(row['#'+name[:-5]].replace(',','.')))


    print("Fit results for %s"%expt.nxs)
    print("Spectrum interval = [%g,%g]"%(expt.ind_first_spectrum,expt.ind_last_spectrum))
    print("Channel interval = [%g,%g]"%(expt.ind_first_channel,expt.ind_last_channel))
    tmp = np.array([0,1,2,3,4])
    print("List of chosen elements: ", ["Element %g"%g for g in tmp[expt.fluospectrums_chosen]])
    print("")
    print("Parameters used:")
    print("gain = %g"%expt.gain +"; eV0 = %g"%expt.eV0)
    print("beam energy = %g"%expt.beam_energy)
    print("List of fitted parameters: "+str(expt.list_isfit))
    print("")
    print("Initial fit parameters:")
    print("epsilon = %g"%expt.epsilon+"; fano = %g"%expt.fano+
          "; noise = %g"%expt.noise)
    print("sl = %g"%expt.sl+"; ct = %g"%expt.ct)
    print("sfa0 = %g"%expt.sfa0+"; sfa1 = %g"%expt.sfa1+
          "; tfb0 = %g"%expt.tfb0+"; tfb1 = %g"%expt.tfb1)
    print("twc0 = %g"%expt.twc0+"; twc1 = %g"%expt.twc1)
    print("fG = %g"%expt.fG)
    print("fA = %g"%expt.fA+"; fB = %g"%expt.fB+"; gammaA = %g"%expt.gammaA+"; gammaB = %g"%expt.gammaB)
    print("")


    # To generate pdf plots for the PDF rendering
    set_matplotlib_formats('png', 'pdf')

    Plot_spectrum(expt, spectrum_index, dparams_list)
    Plot_fit_results(expt, spectrum_index, dparams_list, is_save=False)

    # Restore it to png only to avoid large file size
    set_matplotlib_formats('png')

class Group:
    """
    Class for group of peaks from a same element and with the same family.
    """
    def __init__(self, name):

        # Name of the group elem_familyName (Au_L1M2, Cl_KL3, ...)
        self.name = name

        # Array for peaks
        self.peaks = []


class Peak:
    """
    Class for fluo peak belongin to a Group.
    """

    def __init__(self, name, position_init, strength, is_fitpos = False):

        # Name after the corresponding transition (KL3, L1M3, X, ...)
        self.name = name

        # Position before fit, as given by the user
        self.position_init = position_init

        # Strength of the line (Fluo production cross section)
        self.strength = strength

        # Set if the position of the line is fitted
        self.is_fitpos = is_fitpos




