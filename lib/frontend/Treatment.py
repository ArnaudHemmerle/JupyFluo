from . import Utils
from . import Action

from lib.extraction.common import PyNexus as PN
from lib.extraction import XRF as XRF

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


"""Frontend library for all the widgets concerning the Treatments in the notebook."""

def Choose(expt):
    '''
    Choose the Treatment to be applied to the selected scan.

    Parameters
    ----------
    expt : object
        object from the class Experiment
    '''
    
    # Styling options for widgets
    style = {'description_width': 'initial'}
    tiny_layout = widgets.Layout(width='150px', height='40px')
    short_layout = widgets.Layout(width='200px', height='40px')
    medium_layout = widgets.Layout(width='250px', height='40px')
    large_layout = widgets.Layout(width='300px', height='40px')
    
    
    def on_button_set_params_clicked(b):
        """Set the parameters and extract the scan."""

        # Reset the buttons
        expt.scan.is_extract_done = False
        expt.scan.is_fit_ready = False
        expt.scan.is_fit_done = False

        # Set the parameters
        Set_params(expt)
    
    
    def on_button_plot_peaks_clicked(b):
        """Plot the peaks."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        # Reput the sheet set peaks
        Set_peaks(expt)

        # Extract the info from the sheet
        Extract_groups(expt)

        # Display the peaks on the selected spectrum or on the sum
        if expt.scan.is_peaks_on_sum:
            Display_peaks(expt)
        else:
            w = widgets.interact(Display_peaks,
                                 expt=widgets.fixed(expt),
                                 spectrum_index=widgets.IntText(value=0, step=1, description='Spectrum:'))

    def on_button_save_params_clicked(b):
        """Save the current params as default ones."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        # Copy the current params as the defaut file
        shutil.copy(expt.working_dir+expt.scan.id+'/Parameters.csv','DefaultParameters.csv')

        print("Current set of parameters saved as default.")
    
    def on_button_load_params_clicked(b):
        """Load a set of params as the current one."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        list_params_files = [file for file in sorted(os.listdir('parameters/'))][::-1]

        w_select_files = widgets.Dropdown(options=list_params_files)
    
        def on_button_import_clicked(b):
            "Make the copy."

            # Copy the selected params as the current params file
            shutil.copy('parameters/'+w_select_files.value, expt.working_dir+expt.scan.id+'/Parameters.csv')

            print(str(w_select_files.value)+" imported as current set of parameters.")

        button_import = widgets.Button(description="OK",layout=widgets.Layout(width='100px'))
        button_import.on_click(on_button_import_clicked)

        display(widgets.HBox([w_select_files, button_import]))

    def on_button_save_peaks_clicked(b):
        """Save the current peaks as default ones."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        # Copy the current params as the defaut file
        shutil.copy(expt.working_dir+expt.scan.id+'/Peaks.csv','DefaultPeaks.csv')

        print("Current set of peaks saved as the default one.")

    def on_button_reload_peaks_clicked(b):
        """Reload the default peaks as current ones."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        # Copy the current params as the defaut file
        shutil.copy('DefaultPeaks.csv', expt.working_dir+expt.scan.id+'/Peaks.csv')

        print("Default set of peaks saved as the current one.")
    
    def on_button_export_clicked(b):
        """Export the notebook to PDF."""
        
        print('Export in progress...')
        
        export_done = Action.Export_nb_to_pdf(expt.notebook_name)
        
        if export_done:
            print('Notebook exported to %s.pdf'%expt.notebook_name.split('.')[0])
        else:
            print("There was something wrong with the export to pdf.")
            print("Did you rename the Notebook? If yes:")
            print("1) Change the value of expt.notebook_name in the first cell (top of the Notebook).")
            print("2) Re-execute the first cell.")
            print("3) Try to export the pdf again in the last cell (bottom of the Notebook).")
    
    

    # Next action   
    def on_button_next_clicked(b):
        #clear_output(wait=False)
        
        Utils.Delete_current_cell()
        
        Utils.Create_cell(code='FE.Action.Choose(expt)',
                    position ='at_bottom', celltype='code', is_print=False)        
        
    def on_button_markdown_clicked(b):
        """
        Insert a markdown cell below the current cell.
        """ 
        Utils.Delete_current_cell()
        
        Utils.Create_cell(code='', position ='below', celltype='markdown', is_print=True, is_execute=False)
    
        Utils.Create_cell(code='FE.Treatment.Choose(expt)', position ='at_bottom', celltype='code', is_print=False)
       
    # Display the widgets    
  
    button_export = widgets.Button(description="Export to pdf", layout=widgets.Layout(width='200px'))
    button_export.on_click(on_button_export_clicked)

    button_set_params = widgets.Button(description="Set params",layout=widgets.Layout(width='200px'))
    button_set_params.on_click(on_button_set_params_clicked)

    button_save_params = widgets.Button(description="Save current params as default",layout=widgets.Layout(width='250px'))
    button_save_params.on_click(on_button_save_params_clicked)

    button_load_params = widgets.Button(description="Load params",layout=widgets.Layout(width='200px'))
    button_load_params.on_click(on_button_load_params_clicked)
        
    button_next = widgets.Button(description="Analyze a new scan")
    button_next.on_click(on_button_next_clicked)
    
    button_markdown = widgets.Button(description="Insert comment")
    button_markdown.on_click(on_button_markdown_clicked)

    display(widgets.HBox([button_next,button_markdown, button_export]))
    print(100*"-")
    print("Set parameters:")
    display(widgets.HBox([button_set_params, button_load_params, button_save_params]))
    print(100*"-")

        
    # Buttons for specific Treatment
    #buttons2 = widgets.HBox([button_GIXD, button_XRF, button_isotherm, button_pilatus, button_GIXS])
    #display(buttons2)
    
    
    
