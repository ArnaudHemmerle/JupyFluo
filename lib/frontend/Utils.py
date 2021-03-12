import os
import base64
from IPython.display import clear_output, Javascript, display, HTML
import time
import subprocess
from lib.extraction.common import PyNexus as PN


"""
Frontend functions which require the use of Javascript, and are therefore not usable in Jupyter Lab.
"""


   
def Delete_current_cell():
    '''Delete the cell which called this function in the IPython Notebook.
       The code requires direct use of Javascript and is therefore not usable in Jupyter Lab.'''
    
    display(Javascript(
        """
        var index = IPython.notebook.get_selected_cells_indices();
        IPython.notebook.delete_cell(index);
        """
    ))

def Create_cell(code, position='below', celltype='markdown', is_print = False, is_execute = True):
    '''
    Create a cell in the IPython Notebook.
    The code requires direct use of Javascript and is therefore not usable in Jupyter Lab.    

    Parameters
    ----------
    code : str
        code to fill the new cell with
    position: str, optional
        where to put the cell: "below" or "at_bottom"    
    celltype: str, optional
        type of cell: "code" or "markdown"
    is_print: bool, optional
        print the cell in the pdf report
    '''

    # Delay to ensure unique id
    time.sleep(0.1)
    
    encoded_code = (base64.b64encode(code.encode())).decode()

    # Create a unique id based on epoch time
    display_id = int(time.time()*1e9)

    js_code = """var cell = IPython.notebook.insert_cell_{0}("{1}");
              cell.set_text(atob("{2}"));
              """
    if not is_print: js_code += """cell.metadata.tags = ['notPrint']
                                """
        
    if is_execute: js_code += """cell.execute();
                                """
    display(Javascript(js_code.format(position, celltype, encoded_code)),display_id=display_id)

    # Necessary hack to avoid self-execution of cells at notebook re-opening
    # See http://tiny.cc/fnf3nz
    display(Javascript(""" """), display_id=display_id, update=True)
    

def Refresh_current_cell():
    '''Refresh the current cell.
       The code requires direct use of Javascript and is therefore not usable in Jupyter Lab.'''
    
    # Create a unique id based on epoch time
    display_id = int(time.time()*1e9)

    display(Javascript("""IPython.notebook.execute_selected_cells();"""),display_id=display_id)

    # Necessary hack to avoid self-execution of cells at notebook re-opening
    # See http://tiny.cc/fnf3nz
    display(Javascript(""" """), display_id=display_id, update=True)
    
    
def Save_nb():
    '''Save the current state of the notebook (including the widgets).
       The code requires direct use of Javascript and is therefore not usable in Jupyter Lab.'''
    
    display(HTML('<script>Jupyter.menubar.actions._actions["widgets:save-with-widgets"].handler()</script>') )
    
    
    