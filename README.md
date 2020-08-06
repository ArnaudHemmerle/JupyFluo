# JupyFluo

JupyFluo is a Jupyter Notebook to analyze X-Ray Fluorescence (XRF) experiments on the beamline SIRIUS at the synchrotron SOLEIL.  
The notebook should be first set up by an Expert following the instructions in the "Expert" section. User can then follow the guidelines in the "User" section to start using the notebook. Please note that the notebook is currently in development. As such, be skeptical about any unexpected results.  Any feedback on the notebook or the code is welcome.

## Last versions of modules:
FrontendFunctions.py: 0.9  
AnalysisFunctions.py: 0.11

## User
<!-- [![image](https://imgur.com/NeXYpj8.png)](https://www.youtube.com/watch?v=d7EmOnYjqbk) -->

### Getting Started
1. Once the notebook is open, check that the following parameters in the first cell are correct:
```
# Name of the notebook: expt.notebook_name
# Directory where the data will be saved: expt.working_dir
# Directory where the nexus files are: expt.recording_dir
```  


2. Run the first cell, check that no missing file is reported.

3. Use the dropdown list to choose the scan. Click on ```OK```.  

4. Enter the information on the scan in the corresponding markdown cell (double-click on the text).


5. Click on ```Set params```

6. The panel is used to set the parameters (most of them were already determined by the beamline staff). You should only worry about the subset of spectrums you want to extract, i.e. changing only the parameters ```First Spectrum``` and ```Last Spectrum```.


7. Put first, for example, ```First Spectrum=0``` and ```Last Spectrum=1```. Click on ```Extract the scan``` at the bottom of the panel.


8. Use the top figure to choose your subset of spectrums. 

9. Click again on ```Set params``` to update the parameters ```First Spectrum``` and ```Last Spectrum``` with your choice. Click again on ```Extract the scan```.

10. A csv file ```Parameters.csv``` with all the parameters is created in the folder ```working_directory/filename/```.

### Choose the peaks
1. Click on ```Set peaks```.  

2. A few peaks are already displayed (typically the Rayleigh and Compton peaks). Check their energies.

3. Use the tool ```Add peaks from database``` below the sheet to import new peaks.

4. Modify the sheet to remove peaks or change their parameters. You can leave a peak in the list and do not include it in the analysis by writting ```no``` in the column ```#Fit Peak?```. 

5. **The strength of a line is its X-ray fluorescence production cross section with full cascade (in cm^2/g).** It is extracted from the database but can be obtained here as well http://lvserver.ugent.be/xraylib-web/.

6. Keep the peak/line names ```Elastic/El``` and ```Compton/Co```for the elastic (Rayleigh) and Compton peaks. You can add a Compton peak for an element by naming it ```Compton``` as well, and adding the element in the line column. For example, if you want to add a Compton peak for the Au La1 line, name the peak ```Compton``` and the line ```CoAuLa1```.

7. If you add an escape peak, do not name it with the name of its corresponding element! For example, if you add the escape peak of Au La1, name it 'EscAuLa1', name the line 'Esc', and put a strength of 1.

8. You can use the plot below the sheet to find where the peaks are. The plots are not updated in real time, you need to follow the next points to update the plots.

9. When you think you are done with the peaks, validate the sheet by clicking on ```Update Peaks``` below it.

10. A summary of your peaks appear. Modify them or click on ```Start Fit``` to start the fit.

11. Note: You can also directly edit the Excel file ```Peaks.csv``` in your folder ```working_directory/filename/``` if you prefer.
 

### Fit the spectrums

When you click on ```Start Fit``` , you can follow the fit in real time. The results are continuously updated in your folder ```working_directory/filename/```. Check the files ```FitResult.csv``` and ```FitSpectrums.csv``` that you can open with you prefered software (Excel, Origin, Matlab, vim ...).


Once the fit is done, the fitted parameters will be displayed and saved as png in your folder ```working_directory/filename/```.
 
**The control panel will appear at the bottom of the plots when the fit is done.** 

### Add a plot to the PDF report

1. Until now, nothing will be rendered in the final PDF. Click on the button ```Add a plot to report```.

2. Choose a spectrum (if you were not working on the spectrums sum). Click on ```Preview the selected plot``` to preview it.

3. Choose the spectrum that you want to add to the PDF. Click on ```Add the selected plot```.

4. The output of the cell ```FF.Load_results(expt, spectrum_index=XXX)```, which is automatically generated and executed, will appear in the PDF.

5. Click on ```Export to pdf``` in the panel to generate the PDF, or ```Analyze a new scan``` to continue with the next scan. 

### Tips

To save the widget state (everything written in the panels), click in the menu Widgets/Save Notebook Widget State. Careful, the size of the notebook may increase dramatically when saving the widgets. **Careful, if you do not save the widget state and close the notebook, the panels will not be back when you open it again.**  

## Expert
<!-- [![image](https://imgur.com/a7eXXXk.png)](https://www.youtube.com/watch?v=O-ULCnkTFYU)
[![image](https://imgur.com/ypOYCvq.png)](https://www.youtube.com/watch?v=KkACazH16nw)

-->
### Getting Started

Start with a fresh download from the last main version on GitHub. If you rename the notebook, do not forget to change the corresponding parameter inside its first cell. Note that the notebook does not work with JupyterLab in its current version.

The aim of the Expert part is to determine the general and fitting parameters. It can be quite painful, but those parameters should not vary during an experiment. 

**You should always start with a set of parameters already completed for the same energy as yours.** To do so, click on ```Load params``` after extracting the spectrums.  
Do not start from zero! Taking a previous set of parameters should directly give you reasonnable results.

### General parameters
1. Click on the right fluo elements.

2. Fill in ```Gain```, ```eV0```, ```Delimiter```, ```Limit iter.```, ```Energy```, and check the checkboxes.

3. Choose a range of channels covering from the lowest peak you want to fit to the end of the elastic peak (if included).

4. Extract about 10 spectrums from the file, and try to choose good ones (without any weird behaviour). 


### Peak definition

Determine the peaks to be used (see guidelines in the User section if needed).


### Determine the fit parameters
See inside ```AnalysisFunctions.py```  and the section ```Quick description of the parameters``` of this file to have an explanation on the peak fitting. Here we detail a procedure which seems to be robust for determining the fit parameters, **when starting from a previous configuration.**

1. Some parameters do not have to be fitted, and can be kept constant in our experimental configuration:

```
fano = 0.115
epsilon = 0.0036
tfb0 = 1e-5
tfb1 = 1e-5
twc0 = 1e-5
twc1 = 1e-5
fB = 1.0e-10
gammaB = 1.0e10
```
 
2. Linear background:  
Put ```sl=0.```. Start by finding manually a reasonnable value for ```ct```, which should be close to the value of the background in a region without peaks.

3. Step function at low energy (<8 keV):  
Adjust ```sfa0``` manually (or you can try to fit it). It will change the step function at low energy.

4. Compton peaks:  
- Fit only ```gammaA```, which sets the slope of the foot of the Compton peaks at low energy. When the fit is done, get the average value using the button ```Extract averages```.
- Fit only ```fA```, which sets the transition between a linear slope to a Gaussian in the Compton foot, at low energy.

5. Gaussians width:  
Fit only ```noise``` and ```fG```.

6. Other parameters:  
If the fit is still not good enough, you can try to fit independantly ```sl``` and ```sfa1```.

### Finishing
1. Try to keep only ```ct``` as a fitting parameters for the User (or no fit parameters at all). If the Compton peak varies a lot with time, add ```gammaA```.

2. **Make the list of peaks the default one by clicking on ```Save current peaks as default``` in the control panel.**

3. **Make the list of parameters the default one by clicking on ```Save current params as default``` in the control panel.**

4. Delete the cells you have generated (keep only the first one), and the file is ready for the User.


## Description of the parameters

### References
- The spectrum function and peak definition are adapted from results in the following papers (please cite these references accordingly):
    -  M. Van Gysel, P. Lemberge & P. Van Espen,
    “Description of Compton peaks in energy-dispersive  x-ray fluorescence spectra”,
    X-Ray Spectrometry 32 (2003), 139–147
    -   M. Van Gysel, P. Lemberge & P. Van Espen, “Implementation of a spectrum fitting
    procedure using a robust peak model”, X-Ray Spectrometry 32 (2003), 434–441
    - J. Campbell & J.-X. Wang, “Improved model for the intensity of low-energy tailing in
    Si (Li) x-ray spectra”, X-Ray Spectrometry 20 (1991), 191–197  
    

- See this reference as well for results using these peak/spectrum definitions:
    - F. Malloggi, S. Ben Jabrallah, L. Girard, B. Siboulet, K. Wang, P. Fontaine, and J. Daillant, "X-ray Standing Waves and Molecular Dynamics Studies of Ion Surface Interactions in Water at a Charged Silica Interface", J. Phys. Chem. C, 123 (2019), 30294–30304  
    

- The  X-ray fluorescence production cross section are extracted from the database xraylib (https://github.com/tschoonj/xraylib/wiki):
    - T. Schoonjans et al. "The xraylib library for X-ray-matter interactions. Recent developments" Spectrochimica Acta Part B: Atomic Spectroscopy 66 (2011), pp. 776-784
(https://github.com/xraypy/XrayDB).

### Quick description of the parameters
Here we give a quick description of each parameters and typical values. For more details, see corresponding publications and the code itself.  

Note that for synchrotron-based experiments some parameters can be significantly different than parameters used with lab X-ray sources (most of them can actually be cancelled). 

- ```First/last channel```: the subset of channels to be extracted. Between 0 and 2047.
- ```First/last spectrum```: the subset of spectrums to be extracted. The first spectrum in the file has the index 0.


- ```Elements XXX```: check the box corresponding the detector elements. Typical value: ```Element 4``` for the single-element detector, ```Element 0,1,2,3``` for the four-elements detector.

- ```Energy```: the beamline energy in eV, for calculating the fluorescence cross sections. 

- ```sl, ct```: linear baseline ```ct+sl*eV```.
- ```noise```: electronic noise (FWHM in keV). Typical value: ```noise=0.1```.
- ```fG```: broadening factor of the gaussian width for the Compton peak. Typical value:  ```fG=1.-1.5```.


- ```sfa0, sfa1```: a0 (intercept) and a1 (slope) for shelf fractions. Define the step function at low energy (~<8 keV). Most of the time the slope can be cancelled. Typical values: ```sfa0=[1e-4,5e-4], sfa1=1e-5```.

- ```tfb0, tfb1```: b0 (intercept) and b1 (slope) for tail fractions. Can be cancelled. Typical values: ```tb0=1e-5, tfb1=1e-5```.  

- ```twc0, twc1```: c0 (intercept) and c1 (slope) for tail widths. Can be cancelled. Typical values: ```twc0=1e-5, twc1=1e-5```.

- ```epsilon```: energy to create a charge carrier pair in the detector (in keV). Typical value: ```epsilon=0.0036```. Keep it fixed for Si detector.

- ```fano```: fano factor. Typical value: ```fano=0.115```. Keep it fixed for Si detector.


- ```fA, fB```: Tail fractions fA (low energy) and fB (high energy) for the Compton peak.  Set the transition between a linear slope to a Gaussian in the Compton feet. Typical values: ```fA=[0.05,0.3], fB=1e-10```.

- ```gammaA, gammaB```: Tail gammas gammaA (low energy) and gammaB (high energy). Set the slope of the feet of Compton peaks. Typical values: ```gammaA=[1-5], gammaB=1e+10```.  


- ```gain, ev0```: linear conversion channels/eVs through ```eVs = gain*channels + eV0```. Typical values: ```gain=9.9, ev0=0```.


- ```Delimiter```: the column delimiter used in csv files. Typical value: ```;```.
- ```Limit iter.```: number of iterations at which a fit is considered as stuck and returns NaN values. Typical value: ```1000``` (but can be increased if needed).


- ```Use ipysheet```: use ipysheet or only Peak.csv to define the peaks. Typical value: ```True```.
- ```Fast extract```: use the fast extract option of PyNexus. Typical value: ```True```.
- ```Transmit fit params```: when ```True```, the results of a fit are the initial guess of the next fit. Can trigger a bad behaviour when there are sudden jumps in the spectrum.

- ```Set peaks on sum```: use the integration of all the sprectrums to define the peaks.

- ```Show peaks?```: show the peaks or not in the plots.  

- ```Strength min```: minimal strength to appear in the peak database (in cm^2/g). 0.05 seems to be a reasonnable value. Put to zero to see all the peaks.

- The peak strengths are the X-ray fluorescence production cross section with full cascade (in cm^2/g). They are used to fix the relative intensity of each peak from the same atom.   

## Contributions
Contributions are more than welcome. Please report any bug or submit any new ideas to arnaud.hemmerle@synchrotron-soleil.fr or directly in the Issues section of the GitHub.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
