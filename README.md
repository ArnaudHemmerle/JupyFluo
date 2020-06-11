# JupyFluo

JupyFluo is a Jupyter Notebook to analyze X-Ray Fluorescence (XRF) experiments on the beamline SIRIUS at the synchrotron SOLEIL.  
The notebook should be first set up by an Expert following the instructions in the "Expert" section. User can then follow the guidelines in the "User" section to start using the notebook. Please note that the notebook is currently in development. As such, be skeptical about any unexpected results.  Any feedback on the notebook or the code is welcome.

## Last versions of modules:
FrontendFunctions.py: 0.5  
AnalysisFunctions.py: 0.3

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

3. Click on the button ```Start new analysis```. It will create all the cells needed for the analysis of one scan.

4. Enter the information on the scan in the corresponding markdown cells (double-click on the text).

5. The code cell ```scan = FF.Define_scan(expt)``` is automatically executed. Use the dropdown list to choose the scan. Click on ```OK```.


6. The panel is used to set the parameters (most of them were already determined by the expert). You should only worry about the subset of spectrums you want to extract, i.e. changing only the parameters ```First Spectrum``` and ```Last Spectrum```.


7. Put first, for example, ```First Spectrum=0``` and ```Last Spectrum=1```. Click on ```Extract the scan```.


8. Use the top figure to choose your subset of spectrums. Do not choose an empty spectrum for the first spectrum (and try to have at least the first 10 spectrums not empty). Update the parameters ```First Spectrum``` and ```Last Spectrum``` with your choice.

9. Click on ```Extract the scan```. 

10. A csv file ```Parameters.csv``` with all the parameters is created in the folder ```working_directory/filename/```.

### Choose the peaks
1. Run the next cell:  

```
# Run this cell
FF.Define_peaks(expt)
```


2. Modify the table to add/remove peaks and fit their position or not. You can also leave a peak in the list and do not include it in the analysis by writting ```no``` in the column ```#Fit Peak?```. 

3. Peaks from the same elements should be grouped together (i.e. follow each others). 

4. Keep the peak/line names ```Elastic/El``` and ```Compton/Co```for the elastic (Rayleigh) and Compton peaks. 

5. **Validate the sheet** by clicking outside of it, and **running the cell** ```FF.Extract_elems(expt)```.

6. You can also directly edit the Excel file ```Peaks.csv``` in your folder ```working_directory/filename/```.
 
7. Run the cell:
```
# Run this cell
FF.Extract_elems(expt)
````


8. Check the position of each peak, and whether or not you are missing some peaks.


9. Adjust you peak choice by modifying the above sheet. Do not forget to **validate the sheet** by clicking outside of it, and **running the cell** ```FF.Extract_elems(expt)```.


### Fit the spectrums
When you are done with the peak definition, run the cell:
```
# Run this cell
AF.Fit_spectrums(expt)
```


You can follow the fit in real time. The results are continuously updated in your folder ```working_directory/filename/```. Check the files ```FitResult.csv``` and ```FitSpectrums.csv``` that you can open with you prefered software (Excel, Origin, Matlab, vim ...).


When the fit is done, the fitted parameters will be displayed and saved as png in your folder ```working_directory/filename/```.

### Add a plot to the PDF report

1. Until now, nothing will be rendered in the final PDF. Run the cell: 

```
# Run this cell
FF.Choose_spectrum_to_plot(expt)
````



2. Choose the spectrum that you want to add to the PDF. Click on ```Add plot to report```.


3. The output of the cell ```FF.Load_results(expt, spectrum_index=XXX)``` automatically generated and executed will appear in the PDF.

4. Finally, go to the bottom of the notebook and click on ```Export to pdf``` to generate the PDF, or ```Start new analysis``` to continue with the next scan. 

### Tips
1. If a peak position or area seems to be noisy, try switching off/on fitting its peak position.

2. To save the widget state (everything written in the panels), click in the menu Widgets/Save Notebook Widget State. Careful, the size of the notebook may increase dramatically when saving the widgets. **Carefull, if you do not save the widget state and close the notebook, the panels will not be back when you open it again.**  

## Expert
<!-- [![image](https://imgur.com/a7eXXXk.png)](https://www.youtube.com/watch?v=O-ULCnkTFYU)
[![image](https://imgur.com/ypOYCvq.png)](https://www.youtube.com/watch?v=KkACazH16nw)

-->
### Getting Started

Start with a fresh download from the last main version on GitHub. If you rename the notebook, do not forget to change the corresponding parameter inside its first cell. Note that the notebook does not work with JupyterLab in its current version.

The aim of the Expert part is to determine the general and fit parameters. It can be quite painful, but those parameters should not vary during an experiment. It is also possible to copy directly the parameters from a previous experiment (or from the examples provided here), and test if they are good for your current experimental setup.

### General parameters
1. Click on the right fluo elements.

2. Fill in ```Gain```, ```eV0```, ```Delimiter```, ```Limit iter.```, ```Nb curves intRel```, ```Use ipysheet```, ```Fast extract```, ```Transmit fit params```.

3. Choose a range of channels covering from the lowest peak you want to fit to the end of the elastic peak (if included).

4. Keep the default values in all the other boxes.

5. Extract about 10 spectrums from the file, and try to choose good ones (without any weird behaviour). 



### Peak definition
1. Determine the peaks to be used (see guidelines in the User section if needed).

2. **Warning:** For determining the fit parameters, you should fit the peak position of the most intense peaks.


### Determine the fit parameters
See inside ```AnalysisFunctions.py``` to have an explanation on the peak fitting. Here we detail a procedure which seems to be robust for determining the fit parameters.

1. Some parameters do not have to be fitted, and can be kept constant in our experimental configuration:

```
fano = 0.115
epsilon = 0.0036
tfb1 = 1e-10
fA = 1.0e-10
fB = 1.0e-10
gammaA = 1.0e10
gammaB = 1.0e10
```
 
2. For the other parameters, start with these values :

```
noise = 0.1
sl = 0.
ct = 0.
tfb0 = 0.1
twc0 = 1.
twc1 = 0.1
fG = 1.5
sfa0 = 1e-10
sfa1 = 1e-5
```

3. **Through all the procedure, keep fitting ```sl``` and ```ct```.**

4. Add ```noise``` and ```fG``` . Remove all other fitted parameters (except ```sl``` and ```ct```). When the fit is done, get the average results for ```noise``` and ```fG``` using the command (in a new cell):

```
print(np.mean(expt.dparams_list['noise_list']))
print(np.mean(expt.dparams_list['fG_list']))
```

5. Update the parameters ```noise``` and ```fG``` in the first cell, giving here for example ```noise = 0.111``` and ```fG = 1.432```.

6. Repeat step 4-5 for ```tfb0```, giving for example ```tfb0 = 0.080845```.

7. Repeat step 4-5 for ```twc0```, giving for example ```twc0 = 0.5164```.

8. Repeat step 4-5 for ```twc1```, giving for example ```twc1 = 0.1003```.

9. Repeat step 4-5 for ```sfa0``` and ```sfa1``` fitted together, giving for example ```sfa0 = -0.0002114``` and ```sfa1 = 0.0001089```. 

10. Repeat step 4-5 for ```noise``` and ```fG``` fitted together, giving for example ```noise = 0.113``` and ```fG = 1.479```.

### Finishing
1. Keep only ```sl``` and ```ct``` as fitting parameters for the User.

2. **Replace the file DefaultPeaks.csv by the file Peaks.csv**

3. **Replace the file DefaultParameters.csv by the file Parameters.csv**

4. Delete the cells you have generated (keep only the first one), and the file is ready for the User.


## Description of the parameters

### References
The spectrum function and peak definition are adapted from results in the following papers (please cite these references accordingly):
-  M. Van Gysel, P. Lemberge & P. Van Espen,
    “Description of Compton peaks in energy-dispersive  x-ray fluorescence spectra”,
    X-Ray Spectrometry 32 (2003), 139–147
-   M. Van Gysel, P. Lemberge & P. Van Espen, “Implementation of a spectrum fitting
    procedure using a robust peak model”, X-Ray Spectrometry 32 (2003), 434–441
- J. Campbell & J.-X. Wang, “Improved model for the intensity of low-energy tailing in
    Si (Li) x-ray spectra”, X-Ray Spectrometry 20 (1991), 191–197

See this reference as well for results using these peak/spectrum definitions:
- F. Malloggi, S. Ben Jabrallah, L. Girard, B. Siboulet, K. Wang, P. Fontaine, and J. Daillant, "X-ray Standing Waves and Molecular Dynamics Studies of Ion Surface Interactions in Water at a Charged Silica Interface", J. Phys. Chem. C, 123 (2019), 30294–30304

### Quick description of the parameters
Here we give a quick description of each parameters and typical values. For more details, see corresponding publications and the code itself.  

Note that for synchrotron-based experiments some parameters can be significantly different than parameters used with lab X-ray sources. For example, the Compton peak associated to the elastic peak can be described by a simple Gaussian with a broader width than the elastic peak, and does not need a tail description. Parameters for the Compton tail (fA, fB, gammaA and gammaB) are kept here, but with extreme values to cancel them. 

- ```First/last channel```: the subset of channels to be extracted. Between 0 and 2047.
- ```First/last spectrum```: the subset of spectrums to be extracted. The first spectrum in the file has the index 0.


- ```Elements XXX```: check the box corresponding the detector elements. Typical value: ```Element 4``` for the single-element detector, ```Element 0,1,2,3``` for the four-elements detector.


- ```sl, ct```: linear baseline ```ct+sl*eV```.
- ```noise```: electronic noise (FWHM in keV). Typical value: ```noise=0.1```.
- ```fG```: broadening factor of the gaussian width for the Compton peak. Typical value:  ```fG=1.5```.


- ```sfa0, sfa1```: a0 (intercept) and a1 (slope) for shelf fractions. Typical values: ```sfa0=1e-10, sfa1=1e-4```.
- ```tfb0, tfb1```: b0 (intercept) and b1 (slope) for tail fractions. Typical values: ```tb0=0.1, tfb1=1e-10```. Keep ```tfb1``` fixed.
- ```twc0, twc1```: c0 (intercept) and c1 (slope) for tail widths. Typical values: ```twc0=0.1-1, twc1=0.1```.


- ```epsilon```: energy to create a charge carrier pair in the detector (in keV). Typical value: ```epsilon=0.0036```. Keep it fixed for Si detector.
- ```fano```: fano factor. Typical value: ```fano=0.115```. Keep it fixed for Si detector.


- ```fA, fB```: Tail fractions fA (low energy) and fB (high energy) for the Compton peak. Typical values: ```fA=1e-10, fB=1e-10```. Keep fixed.
- ```gammaA, gammaB```: Tail gammas gammaA (low energy) and gammaB (high energy). Typical values: ```gammaA=1e+10, gammaB=1e+10```. Keep fixed.


- ```gain, ev0```: linear conversion channels/eVs through ```eVs = gain*channels + eV0```. Typical values: ```gain=9.9, ev0=0```.
- ```Delimiter```: the column delimiter used in csv files. Typical value: ```;```.
- ```Limit iter.```: number of iterations at which a fit is considered as stuck and returns NaN values. Typical value: ```1000``` (but can be increased if needed).
- ```Nb curves intRel```: number of curves used at the beginning of the fit to determine the relative intensities of fluorescence lines belonging to a same element. Typical value: ```5```.

- ```Use ipysheet```: use ipysheet or only Peak.csv to define the peaks. Typical value: ```True```.
- ```Fast extract```: use the fast extract option of PyNexus. Typical value: ```True```.
- ```Transmit fit params```: when ```True```, the results of a fit are the initial guess of the next fit. Can trigger a bad behaviour when there are sudden jumps in the spectrum. Typical value: ```False```.

## Contributions
Contributions are more than welcome. Please report any bug or submit any new ideas to arnaud.hemmerle@synchrotron-soleil.fr or directly in the Issues section of the GitHub.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
