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
from psutil import virtual_memory as vm

import base64
from IPython.display import clear_output, Javascript, display, HTML
import subprocess

try:
    import ipysheet
except:
    print('Careful: the module ipysheet is not installed!')


    
__version__="0.5"

"""
-Here are defined all the functions relevant to the front end of JupyFluo,
i.e. the widgets (buttons, interactive plots), the self-generation of cells, conversion and saving of files ...
-Arguments are passed from the Notebook through objects belonging to the classes Experiment and Scan only.
"""


# General parameters for layout and style of widgets
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
        
def Generate_cells_on_click(expt):
    """
    Generate the cells for the next analysis.
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
    ['# New scan: replace by name', position, 'markdown', True, True],
    ['Describe the scan here.', position, 'markdown', True, True],
    ['scan = FF.Define_scan(expt)', position, 'code', False, True],
    ['## Peak definition', position, 'markdown', False, True],
    ['# Run this cell\n'+\
     'FF.Define_peaks(expt)', position, 'code', False, False],
    ['# Run this cell\n'+\
     'FF.Extract_elems(expt)\n'+\
     'w = widgets.interact(FF.Display_peaks, expt=widgets.fixed(expt), spectrum_index=widgets.IntText(value=0, step=1, description=\'Spectrum:\'))'\
     , position,'code', False, False],
    ['## Fit the spectrums', position, 'markdown', False, True],
    ['# Run this cell\n'+\
     'AF.Fit_spectrums(expt)', position, 'code', False, False],
    ['## Show the results on a given spectrum', position, 'markdown', False, True],
    ['# Run this cell\n'+\
     'FF.Choose_spectrum_to_plot()', position,'code', False, False],
    ['FF.Generate_cells_on_click(expt)', position, 'code', False, True]
    ]

    def on_button_generate_clicked(b):
        """Generate the cells when clicked"""
        
        for cell in cells:
            Create_cell(code=cell[0], position=cell[1], celltype=cell[2], is_print=cell[3], is_execute = cell[4])

    def on_button_export_clicked(b):
        """
        Export the notebook to PDF.
        """
        
        print('Export in progress...')
        
        export_done = Export_nb_to_pdf(expt.notebook_name)
        
        if export_done:
            print('Notebook exported to %s.pdf'%expt.notebook_name.split('.')[0])
        else:
            print("There was something wrong with the export to pdf. Please try again.")
    
    button_generate = widgets.Button(description="Start analysis", layout=widgets.Layout(width='200px'))
    button_export = widgets.Button(description="Export to pdf", layout=widgets.Layout(width='200px'))
    
    display(widgets.HBox([button_generate, button_export]))
    button_generate.on_click(on_button_generate_clicked)
    button_export.on_click(on_button_export_clicked)

    
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
        if (time.time()-t0) > 100:
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



def Define_scan(expt):
    """
    1) Define the object Scan.
    2) Create widgets to allow for user to enter required params.
    3) Execute the spectrum extraction when clicking on the button.
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
        Define_scan_identifiers(expt)

        # Create a folder for saving params and results, if it does not already exist.
        if not os.path.exists(expt.working_dir+expt.id):
            os.mkdir(expt.working_dir+expt.id)   
            
        # Check if the csv file for parameters already exists, if not copy the DefaultParameters.csv file
        if not os.path.isfile(expt.working_dir+expt.id+'/Parameters.csv'):
            shutil.copy('DefaultParameters.csv', expt.working_dir+expt.id+'/Parameters.csv')    
            
        Display_widgets(expt)

    w_select_scan = widgets.Dropdown(
        options=expt.list_nxs_files,
        description='Select scan:',
        layout=widgets.Layout(width='400px'),
        style=style)
        
        
    button_choose_scan = widgets.Button(description="OK",layout=widgets.Layout(width='100px'))
    button_choose_scan.on_click(on_button_choose_scan_clicked)
    
    display(widgets.HBox([w_select_scan, button_choose_scan]))


def Display_widgets(expt):    

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
    

    def on_button_extract_clicked(b):
        """Extract the scan."""

        # Update the parameters with current values
        update_params()
        
        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Display_widgets(expt)
        
        # Load the file
        Extract_nexus(expt)

        print("Extraction of:\n%s"%expt.path)
        
        # Define and plot the channels and spectrums subsets
        Define_subsets(expt)
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
        expt.is_ipysheet = w_is_ipysheet.value
        expt.delimiter = w_delimiter.value
        expt.fitstuck_limit = w_fitstuck_limit.value
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
                    'is_ipysheet',
                    'delimiter',
                    'fitstuck_limit',
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
                    'fano'
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
                    expt.is_ipysheet,
                    expt.delimiter,
                    expt.fitstuck_limit,
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
                    expt.fano
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
            is_ipysheet = eval(row['is_ipysheet'])
            delimiter = str(row['delimiter'])
            fitstuck_limit = int(row['fitstuck_limit'])
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
        layout=widgets.Layout(width='150px'),
        style=style)
    
    w_eV0 = widgets.FloatText(
        value=eV0,
        description='eV0',
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
    
    button_extract = widgets.Button(description="Extract the scan",layout=widgets.Layout(width='500px'))
    button_extract.on_click(on_button_extract_clicked)

    display(widgets.HBox([w_ind_first_channel, w_ind_last_channel, w_ind_first_spectrum, w_ind_last_spectrum]))
    print("-"*100)    
    print("List of available elements: ", ["Element %s"%s for s in fluospectrums_available])

    w_fluospectrum = widgets.HBox([w_is_fluospectrum00, w_is_fluospectrum01,w_is_fluospectrum02,
                                   w_is_fluospectrum03,w_is_fluospectrum04])  
    display(w_fluospectrum)
   
    display(widgets.HBox([w_is_sl, w_is_ct, w_is_noise, w_is_sfa0, w_is_sfa1, w_is_tfb0, w_is_tfb1,w_is_twc0]))
    display(widgets.HBox([w_is_twc1, w_is_fG, w_is_fA, w_is_fB, w_is_gammaA,w_is_gammaB, w_is_epsilon, w_is_fano]))   
    
    display(widgets.HBox([w_sl, w_ct, w_noise, w_fG]))
    display(widgets.HBox([w_sfa0, w_sfa1, w_tfb0, w_tfb1]))
    display(widgets.HBox([w_twc0, w_twc1, w_epsilon, w_fano]))
    display(widgets.HBox([w_fA, w_fB, w_gammaA,w_gammaB]))  

    print("-"*100)
    display(widgets.HBox([w_gain, w_eV0, w_delimiter, w_fitstuck_limit, w_is_ipysheet, w_is_fast]))

    display(widgets.HBox([button_extract]))
    

        

def Define_scan_identifiers(expt):
    """
    Create a series of identifiers for the current scan.
    """
    # For example:
    # scan.nxs = 'SIRIUS_2017_12_11_08042.nxs'
    # scan.path = '/Users/arnaudhemmerle/recording/SIRIUS_2017_12_11_08042.nxs'
    # scan.id = 'SIRIUS_2017_12_11_08042'
    # scan.number = 8042
    
    expt.path = expt.recording_dir+expt.nxs
    expt.id = expt.nxs[:-4]
    split_name = expt.nxs.split('.')[0].split('_')
    expt.number = int(expt.nxs.split('.')[0].split('_')[-1])


def Extract_nexus(expt):
    """
    1) Extract all the requested fluospectrums in the nexus file.
    2) Correct with ICR/OCR.
    3) Sum the fluospectrums and put them in scan.allspectrums_corr
    """
    
    expt.nexus = PN.PyNexusFile(expt.path, fast=expt.is_fast)

    stamps, data= expt.nexus.extractData()
    
    # Extract timestamps
    for i in range(len(stamps)):
        if (stamps[i][1]== None and stamps[i][0]=='sensorsRelTimestamps'):
            expt.allsensorsRelTimestamps = data[i]

            
    def extract_and_correct(ind_spectrum):
        """Extract the requested fluospectrum from the nexus file and correct it with ICR/OCR"""

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
            print(PN._RED+"OCR not found in data. Taking OCR = spectrum_intensity/counting_time."+PN._RESET)
            OCR = np.array([np.sum(fluospectrum[n])/integration_time[n] for n in range(len(fluospectrum))])
               
        ratio = np.array([ICR[n]/OCR[n] if (~np.isclose(OCR[n],0.) & ~np.isnan(OCR[n]) & ~np.isnan(ICR[n]))
                         else 0. for n in range(len(ICR))])            
        spectrums_corr = np.array([fluospectrum[n]*ratio[n] for n in range(len(ratio))])

        return spectrums_corr

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

def Define_subsets(expt):
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

    if np.sum(expt.spectrums[0])<10.:
        CBOLD = '\033[1m'
        CEND = '\033[0m'
        print(CBOLD + 'ERROR: The first spectrum cannot be empty!!!' + CEND)


def Define_peaks(expt):
    """
    1) Check if the csv file Peaks.csv exists, if not create it with an example.
    2) If ipysheet is activated, display the interactive sheet. If not, extract the peaks from Peaks.csv
    3) Save the peaks in Peaks.csv
    Update scan.arr_peaks, with the info on the peaks later used to create the Elem and Lines objects.
    """

    nb_rows = 20

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


    if expt.is_ipysheet:
        sheet = ipysheet.easy.sheet(columns=nb_columns, rows=nb_rows ,column_headers = expt.peaks_header)

        # ipysheet does not work correctly with None entries
        # It is necessary to fill first the cells with something
        nb_rows_to_fill = nb_rows - np.shape(arr_peaks)[0]
        fill = np.reshape(np.array(nb_rows_to_fill*nb_columns*['']),
                          (nb_rows_to_fill, nb_columns))
        arr_peaks = np.vstack((arr_peaks, fill))

        for i in range(nb_columns):
            ipysheet.easy.column(i,  arr_peaks[:,i])
        display(sheet)

    else:
        print("Peaks imported from %s"%(expt.working_dir+expt.id+'/Peaks.csv'))
        print('\t'.join([str(cell) for cell in expt.peaks_header]))
        print('\n'.join(['\t \t'.join([str(cell) for cell in row]) for row in arr_peaks if row[0]!='']))


    expt.arr_peaks = arr_peaks

def Extract_elems(expt):
    """
    1) Validate the current sheet if using ipysheet
    2) Create objects Elem and Lines from info in scan.arr_peaks
    """
    # 1) Validate the current sheet if using ipysheet
    if expt.is_ipysheet:
        Validate_sheet(expt)

    # 2) Create objects Elem and Lines from info in arr_peaks
    # Each "Peak name" gives a new Elem with Elem.name = "Peak name"
    # Each "Line name" gives a new Elem.Line with Elem.Line.name = "Line name"
    # "Position (eV)" -> Elem.Line.position_i
    # "Fit position?" -> Elem.Line.is_fitpos
    # The array scan.elems contains the list of objects Elem

    # Remove the peaks which are not fitted from scan.arr_peaks
    expt.arr_peaks = expt.arr_peaks[np.where(expt.arr_peaks[:,4]!='no')]

    elems = []
    for i in range(np.shape(expt.arr_peaks)[0]):
        elem_name = expt.arr_peaks[i][0]
        if (elem_name != '' and elem_name != expt.arr_peaks[i-1][0]):
            newElem = Elem(elem_name)
            elems = np.append(elems, newElem)
            elems[-1].lines = []

    for i in range(np.shape(expt.arr_peaks)[0]):
        elem_name = expt.arr_peaks[i][0]
        for elem in elems:
            if elem.name == elem_name:
                if expt.arr_peaks[i][3] == 'yes':
                    is_fitpos = True
                else:
                    is_fitpos = False
                elem.newLine = Line(name = str(expt.arr_peaks[i][1]),
                                    position_i = float(expt.arr_peaks[i][2]),
                                    is_fitpos = is_fitpos)
                elem.lines = np.append(elem.lines, elem.newLine)

    expt.elems = elems


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
    1) Define initial guesses for the relative amplitudes of each lines from each elem, used later in the fit.
    2) Plot the position of each peaks on the given spectrum spectrum_index.
    Take spectrum_index, the index of which spectrum you want to use.
    """

    # Convert channels into eV
    expt.eV = expt.channels*expt.gain + expt.eV0

    eV = expt.eV
    elems = expt.elems

    # We work on the spectrum specified by spectrum_index
    spectrum = expt.spectrums[spectrum_index]

    # 1) Define initial guesses for the relative amplitudes
    # Get each peak approx. intensity (intensity at the given peak position)
    for elem in elems:
        for line in elem.lines:
                position = line.position_i
                ind_position = np.argmin(np.abs(np.array(eV)-position))
                intensity = spectrum[ind_position]
                line.intensity_0 = float(intensity)

    # If the relative intensity of each line is not known a priori and therefore fitted:
    # An initial guess is extracted from the intensity of the spectrum at the given line position
    for elem in elems:
        # Get the intensity of the most intense line
        temp_list = []
        for line in elem.lines:
            temp_list.append(line.intensity_0)
        highest = np.max(temp_list)

        # Normalise all the lines relative to the most intense one
        if elem.isfit_intRel:
            for line in elem.lines:
                line.intRel_i = line.intensity_0/highest


    # 2) Plot the spectrum and each line given by the user
    Plot_spectrum(expt, spectrum_index=spectrum_index)

    if expt.is_ipysheet: print(expt.prt_peaks)


def Plot_spectrum(expt, spectrum_index=0, dparams_list=None, is_save=False):
    """
    Plot data of a specific spectrum (given by spectrum_index).
    If a dparams_list is given, redo and plot the fit with the given parameters.
    If is_save, save the plots in png and the fitting curve in spectrum_X_fit.csv
    """
    n = spectrum_index
    eV = expt.eV
    elems = expt.elems

    spectrum = expt.spectrums[n]

    if dparams_list != None:
        sl_list = dparams_list['sl_list']
        ct_list = dparams_list['ct_list']

        # Fit the spectrum with the given parameters
        for elem in elems:
            elem.area = elem.area_list[n]
            for line in elem.lines:
                line.position = line.position_list[n]
                line.intRel = line.intRel_list[n]
        dparams = {}
        for name in dparams_list:
            dparams[name[:-5]] = dparams_list[name][n]
        spectrum_fit = AF.Fcn_spectrum(dparams, elems, eV)

    else:
        for elem in elems:
            for line in elem.lines:
                line.position = line.position_i

    # Plot the whole spectrum
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(211)
    colors = iter(['r', 'b', 'y', 'c', 'm', 'g', 'orange', 'brown']*20)
    ax1.set(xlabel = 'E (eV)', ylabel = 'counts')
    for elem in elems:
        for line in elem.lines:
            position = line.position
            ax1.axvline(x = position,  color = next(colors), linestyle = '--', label = elem.name+' '+line.name)
    ax1.plot(eV, spectrum, 'k.')
    if dparams_list != None: ax1.plot(eV, spectrum_fit, 'r-')
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)
    for item in ([ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(14)


    ax2 = fig.add_subplot(212)
    colors = iter(['r', 'b', 'y', 'c', 'm', 'g', 'orange', 'brown']*20)
    ax2.set(xlabel = 'E (eV)', ylabel = 'counts')
    for elem in elems:
        for line in elem.lines:
            position = line.position
            ax2.axvline(x = position,  color = next(colors), linestyle = '--')
    ax2.plot(eV, spectrum, 'k.')
    if dparams_list != None:
        ax2.plot(eV, spectrum_fit, 'r-')
        #ax2.plot(eV, sl_list[n]*eV+ct_list[n], 'b-', label = 'Linear background')
        #ax2.legend(loc = 1)
    ax2.set_ylim(bottom = 1)
    ax2.set_yscale('log')
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    for item in ([ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(14)
    
    plt.subplots_adjust(hspace=.0)
    fig.subplots_adjust(top=0.95)
    fig.suptitle(expt.nxs+': Spectrum number %g/%g'%(n,(len(expt.spectrums)-1)), fontsize=14)

    if is_save:
        delimiter = expt.delimiter
        plt.savefig(expt.working_dir+expt.id+'/spectrum_'+str(spectrum_index)+'_fit.png')
        np.savetxt(str(expt.working_dir+expt.id+'/spectrum_'+str(spectrum_index)+'_fit.csv'),
                   np.transpose([eV, spectrum, spectrum_fit]),
                   header = '#eV'+delimiter+'#spectrum'+delimiter+'#spectrum_fit',
                   delimiter = delimiter,
                   comments = ''
                  )
    plt.show()
    plt.close()

    # Plot each peak
    colors = iter(['r', 'b', 'y', 'c', 'm', 'g', 'orange', 'brown']*20)

    count = 0
    for elem in elems:
        for line in elem.lines:
            if count%2==0: fig = plt.figure(figsize=(14,4.7))
            plt.subplot(1, 2, count%2+1)
                
            position = line.position
            position_i = line.position_i

            ind_min = np.argmin(np.abs(np.array(eV)-0.9*position))
            ind_max = np.argmin(np.abs(np.array(eV)-1.1*position))

            spectrum_zoom = spectrum[ind_min:ind_max]
            eV_zoom = eV[ind_min:ind_max]

            if dparams_list != None:
                spectrum_fit_zoom = spectrum_fit[ind_min:ind_max]
                intRel = line.intRel_list[n]
                area = elem.area_list[n]

                if line.is_fitpos:
                    title0 = 'position(init) = %g eV, position(fit)=%g eV'%(position_i,position)
                else:
                    title0 = 'position = %g eV'%(position)

                title = elem.name + ' ' + line.name + '\n' \
                        +'elem area = %g, relative int = %g'%(area,intRel) + '\n'\
                        + title0
            else:
                title = elem.name + ' ' + line.name +'\n'+'position = %g eV'%(position)

            plt.gca().set_title(title)

            plt.plot(eV_zoom, spectrum_zoom, 'k.')
            if dparams_list != None: plt.plot(eV_zoom, spectrum_fit_zoom, 'r-')
            plt.xlabel('E (eV)')

            # Plot each line in the zoom
            for elem_tmp in elems:
                for line_tmp in elem_tmp.lines:
                    position_tmp = line_tmp.position
                    if (eV[ind_min]<position_tmp and eV[ind_max]>position_tmp):
                        plt.axvline(x = position_tmp , label = elem_tmp.name+' '+line_tmp.name,
                                    linestyle = '--', color = next(colors))
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
    elems = expt.elems
    spectrums = expt.spectrums

    scans = np.arange(np.shape(spectrums)[0])
    
    # Plot areas & save plots
    is_title = True
    for elem in elems:
        fig, ax = plt.subplots(figsize=(15,4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.plot(scans, elem.area_list, 'r.-', label = 'Area %s'%elem.name)
        plt.legend()
        if is_save: plt.savefig(expt.working_dir+expt.id+'/area_'+elem.name+'.png')
        if spectrum_index!=None: plt.axvline(x = spectrum_index, linestyle = '--', color = 'black')
        if is_title: 
            ax.set_title('PEAK AREA\n')
            ax.title.set_fontsize(18)
            ax.title.set_fontweight('bold')            
            is_title = False
        plt.show()

    # Plot positions & save plots
    is_title = True
    for elem in elems:
        for line in elem.lines:
            if line.is_fitpos:
                fig, ax = plt.subplots(figsize=(15,4))                  
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                plt.plot(scans, line.position_list, 'b.-', label = 'Position %s '%(elem.name+'.'+line.name))
                plt.legend()
                if is_save: plt.savefig(expt.working_dir+expt.id+'/position_'+elem.name+line.name+'.png')
                if is_title:
                    ax.set_title('PEAK POSITION\n')
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
                ax.set_title('OTHER PARAMETERS\n')
                ax.title.set_fontsize(18)
                ax.title.set_fontweight('bold')                
                is_title = False
            if spectrum_index!=None: plt.axvline(x = spectrum_index, linestyle = '--', color = 'black')
            plt.show()

def Choose_spectrum_to_plot():
    """
    Select a spectrum to plot with its fit.
    """
    w_spectrum_index = widgets.IntText(value=0, step=1, description='Spectrum:')
    w_is_save = widgets.Checkbox(value=False, description='Save Fit?')

    display(widgets.HBox([w_spectrum_index,w_is_save]))

    def on_button_clicked(b):
        """Validate the values when the button is clicked."""
        code = 'FF.Load_results(expt, spectrum_index='+str(w_spectrum_index.value)+', is_save='+str(w_is_save.value)+')'
        Create_cell(code=code, position='below', celltype='code', is_print=True, is_execute=True)

    button = widgets.Button(description="Click to plot",layout=widgets.Layout(width='300px', height='40px'))
    display(button)
    button.on_click(on_button_clicked)
            
def Load_results(expt, spectrum_index=0, is_save=False):
    """
    Load and plot the results of a previous fit.
    If is_save, save the fitting curve as a csv and as a png.
    Read the result from FitResults.csv
    """
    elems = expt.elems

    if is_save:
        print('Saved: spectrum_'+str(spectrum_index)+'_fig.csv')
        print('Saved: spectrum_'+str(spectrum_index)+'_fit.png')

    dparams_list = {'sl_list','ct_list',
                    'sfa0_list','sfa1_list','tfb0_list','tfb1_list',
                    'twc0_list','twc1_list',
                    'noise_list','fano_list', 'epsilon_list',
                    'fG_list','fA_list','fB_list','gammaA_list','gammaB_list'}

    # Init all the lists
    dparams_list = dict.fromkeys(dparams_list, np.array([]))

    for elem in elems:
        elem.area_list = np.array([])
        for line in elem.lines:
            line.intRel_list = np.array([])
            line.position_list = np.array([])


    with open(expt.working_dir+expt.id+'/FitResults.csv', "r") as f:
        reader = csv.DictReader(f, delimiter=expt.delimiter)
        for row in reader:
            for elem in elems:
                elem.area_list = np.append(elem.area_list, np.float(row['#'+elem.name+'.area'].replace(',','.')))

                for line in elem.lines:
                    line.intRel_list = np.append(line.intRel_list,
                                                 np.float(row['#'+elem.name+'.'+line.name+'.intRel'].replace(',','.')))
                    line.position_list = np.append(line.position_list,
                                                   np.float(row['#'+elem.name+'.'+line.name+'.position'].replace(',','.')))

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
    print("fA = %g"%expt.fA+"; fB = %g"%expt.fB+
          "; gammaA = %g"%expt.gammaA+"; gammaB = %g"%expt.gammaB)
    print("")
    
        
    # To generate pdf plots for the PDF rendering
    set_matplotlib_formats('png', 'pdf') 
    
    Plot_spectrum(expt, spectrum_index, dparams_list, is_save)
    Plot_fit_results(expt, spectrum_index, dparams_list, is_save)
    
    # Restore it to png only to avoid large file size
    set_matplotlib_formats('png') 

class Elem:
    """
    Class for fluo elements (ex: Au, Cl, Ar, ...).
    """
    def __init__(self, name, isfit_intRel = True):
        self.name = name
        self.lines = []

        # Set if the relative intensity is fitted
        self.isfit_intRel = isfit_intRel

class Line:
    """
    Class for fluo lines (ex: La1, Lb3, ...) of a given element.
    """
    
    def __init__(self, name, position_i, intRel_i = 1., is_fitpos = False):
        self.name = name

        # Position before any fit
        self.position_i = position_i

        # Set if the position of the line is fitted
        self.is_fitpos = is_fitpos

        # The relative intensity (relative to the most intense one,
        # i.e. intRel = 1 for the most intense line of an elem)
        # If isfit_intRel = True, it will be fitted, else it should be given
        self.intRel_i = intRel_i



