import PyNexus as PN 
import FrontendFunctions as FF

import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import cm
import matplotlib.colors as mplcolors
from matplotlib.ticker import FormatStrFormatter

import numpy as np
from numpy import linalg
from math import isclose
from scipy.special import erfc
from lmfit import Minimizer, Parameters, fit_report, conf_interval, printfuncs

import os
import shutil
import csv

from IPython.display import clear_output



__version__="0.3"

"""
Here are defined the functions for analysis.

Description of the different dictionaries of parameters:
- dparams_fit contains the fitting parameters common to the whole spectrum, such as fano, fG, etc ...
- dparams_0 contains the initial guess for the LM fit
- dparams_lm contains the current parameter set during the LM fit
- dparams_list contains lists of results from the fits (one list per fit parameter)
"""


def Fit_spectrums(expt, is_save=True):
    """
    Fit procedure. Will fit each spectrum in expt.spectrums
    If is_save=True, results are saved in FitResults.csv and FitParameters.csv
    """

    #####################################################
    ################   PREPARE FIT   ####################
    #####################################################

    eV = expt.eV
    elems = expt.elems
    dparams_fit = {  'sl':expt.sl,
                     'ct':expt.ct,
                     'sfa0':expt.sfa0,
                     'sfa1':expt.sfa1,
                     'tfb0':expt.tfb0,
                     'tfb1':expt.tfb1,
                     'twc0':expt.twc0,
                     'twc1':expt.twc1,
                     'noise':expt.noise,
                     'fano':expt.fano,
                     'epsilon':expt.epsilon,
                     'fG':expt.fG,
                     'fA':expt.fA,
                     'fB':expt.fB,
                     'gammaA':expt.gammaA,
                     'gammaB':expt.gammaB}
    expt.dparams_fit=dparams_fit

    # Dictionnary with a list for each parameter updated at each fit
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
            line.intRel_tmp = np.array([])
            line.intRel_list = np.array([])
            line.position_list = np.array([])

    #####################################################
    ##############   PREPARE SAVE   #####################
    #####################################################
    if is_save:
        # Save the results (parameters)       
        # Extract the 0D stamps and data
        nexus = PN.PyNexusFile(expt.path)
        stamps0D, data0D = nexus.extractData('0D')
        nexus.close()
        
        # Prepare the header of the csv file
        header = np.array([])
        
        # Data stamps
        for i in range(len(stamps0D)):
             if (stamps0D[i][1]==None):
                    header =np.append(header, '#'+stamps0D[i][0])
             else:
                    header =np.append(header, '#'+stamps0D[i][1])
        
        # Stamps from the fit            
        for elem in elems:
            header = np.append(header, '#'+elem.name+'.area')
            for line in elem.lines:
                header = np.append(header,'#'+elem.name+'.'+line.name+'.intRel')
                header = np.append(header,'#'+elem.name+'.'+line.name+'.position')

        for name in dparams_list:
            header = np.append(header, '#'+name[:-5])                    
                    
        with open(expt.working_dir+expt.id+'/FitResults.csv', "w", newline='') as f:
            writer = csv.writer(f,delimiter=expt.delimiter)
            writer.writerow(header)

        # Save the results (fits)
        # Prepare the header of the csv file
        header = np.array(['#sensorsRelTimestamps', '#eV', '#data', '#fit'])
        with open(expt.working_dir+expt.id+'/FitSpectrums.csv', "w", newline='') as f:
            writer = csv.writer(f,delimiter=expt.delimiter)       
            writer.writerow(header)
            
    count=0
    for spectrum in expt.spectrums:

        #####################################################
        #####################   FIT   #######################
        #####################################################

        # Allow or not transmission of current fit params as init params for the next fit 
        is_transmitted = expt.is_transmitted
        if is_transmitted:
            # Initial guess for peak params
            if count==0:
                # First loop : initial guess is given by the expert fit parameters
                dparams_0 = dparams_fit.copy()
            else:
                # For the next loops, initial guess is given by the results of the previous fit
                dparams_0 = {}
                for name in dparams_list:
                    dparams_0[name[:-5]] = dparams_list[name][-1]
        else:
            # Initial guess is always given by the expert fit parameters
            dparams_0 = dparams_fit.copy()

        # Least squares fit
        # Find the least squares solution to the equation ax=b, used as initial guess for the LM fit.
        # p contains the best fit for the area of each elem (elem.area)
        # Results with the subscript ls
        a = []
        for elem in elems:
            spectrum_elem = 0.
            for line in elem.lines:
                    position = line.position_i
                    intRel = line.intRel_i
                    if elem.name == 'Compton':
                        spectrum_elem += Fcn_compton_peak(position,intRel,eV,dparams_0)
                    else:
                        spectrum_elem += Fcn_peak(position,intRel,eV,dparams_0)
            a.append(spectrum_elem)
        a = np.transpose(a)
        b = spectrum

        area_ls, residues, rank, sv = linalg.lstsq(a,b,1.e-10)

        # Store the elem.area of each elem
        i=0
        for elem in elems:
            elem.area_ls = area_ls[i]
            i+=1

        ###################################
        # LMFIT
        dparams_lm = Parameters()

        for elem in elems:
            dparams_lm.add('area_'+elem.name, value=elem.area_ls, vary=True, min = 0.)
            for line in elem.lines:
                # Decides whether the relative intensity of the line should be fitted
                if isclose(line.intRel_i,1.):
                    # Keeps the most intense peak at a relative intensity of 1 (by definition)
                    dparams_lm.add('intRel_'+elem.name+'_'+line.name, value=1., vary=False)
                else:
                    # The first curves are used to set the relative intensities
                    if count < expt.nb_curves_intrel:
                        dparams_lm.add('intRel_'+elem.name+'_'+line.name, value=line.intRel_i,
                                       vary=elem.isfit_intRel, min = 0., max = 1.)
                    else:
                        dparams_lm.add('intRel_'+elem.name+'_'+line.name, value=line.intRel,
                                       vary=False)

                # Check whether the position of the line should be fitted
                if line.is_fitpos:
                    dparams_lm.add('pos_'+elem.name+'_'+line.name, value=line.position_i,
                                   vary = True, min = line.position_i-100, max = line.position_i+100)
                else:
                    dparams_lm.add('pos_'+elem.name+'_'+line.name, value=line.position_i,
                                   vary = False)


        dparams_lm.add('sl', value=dparams_0['sl'])
        dparams_lm.add('ct', value=dparams_0['ct'])
        dparams_lm.add('sfa0', value=dparams_0['sfa0'])
        dparams_lm.add('sfa1', value=dparams_0['sfa1'], min = 0.)
        dparams_lm.add('tfb0', value=dparams_0['tfb0'])
        dparams_lm.add('tfb1', value=dparams_0['tfb1'], min = 0.)
        dparams_lm.add('twc0', value=dparams_0['twc0'])
        dparams_lm.add('twc1', value=dparams_0['twc1'], min = 0.)
        dparams_lm.add('noise', value=dparams_0['noise'])
        dparams_lm.add('fano', value=dparams_0['fano'])
        dparams_lm.add('epsilon', value=dparams_0['epsilon'])
        dparams_lm.add('fG', value=dparams_0['fG'], min = 0.)
        dparams_lm.add('fA', value=dparams_0['fA'], min = 0.)
        dparams_lm.add('fB', value=dparams_0['fB'], min = 0.)
        dparams_lm.add('gammaA', value=dparams_0['gammaA'], min = 0.)
        dparams_lm.add('gammaB', value=dparams_0['gammaB'], min = 0.)

        # Check in list_isfit which peak params should be fitted
        # By default vary = True in dparams_lm.add
        for name in expt.dparams_fit:
            dparams_lm[name].vary = False
            dparams_lm[name].vary = name in expt.list_isfit

        expt.is_fitstuck = False   
        def iter_cb(params, nb_iter, resid, *args, **kws):

            # Stop the current fit if it is stuck or if the spectrum is empty
            if nb_iter > expt.fitstuck_limit:
                expt.is_fitstuck = True

            if np.sum(spectrum)<10.:
                expt.is_fitstuck = True

            return expt.is_fitstuck

        # Do the fit, here with leastsq model
        minner = Minimizer(Fcn2min, dparams_lm, fcn_args=(elems, eV, spectrum), iter_cb=iter_cb, xtol = 1e-6, ftol = 1e-6)

        result = minner.minimize(method = 'leastsq')

        # Calculate final result with the residuals
        #final = spectrum + result.residual

        # Extract the results of the fit and put them in lists

        if expt.is_fitstuck:

            # If the fit was stuck we put NaN
            for elem in elems:
                elem.area_list = np.append(elem.area_list, np.nan)

                for line in elem.lines:
                    line.intRel_list = np.append(line.intRel_list , np.nan)
                    line.position_list = np.append(line.position_list, np.nan)

            for name in dparams_list:
                dparams_list[name] =  np.append(dparams_list[name], np.nan)

        else:
            for elem in elems:
                elem.area_list = np.append(elem.area_list, result.params['area_'+elem.name].value)

                for line in elem.lines:
                    if count<expt.nb_curves_intrel:
                        # The first fits are used to set the relative intensities
                        line.intRel_tmp = np.append(line.intRel_tmp, result.params['intRel_'+elem.name+'_'+line.name].value)
                        line.intRel = np.mean(line.intRel_tmp)

                    line.intRel_list = np.append(line.intRel_list , result.params['intRel_'+elem.name+'_'+line.name].value)
                    line.position_list = np.append(line.position_list, result.params['pos_'+elem.name+'_'+line.name].value)

            for name in dparams_list:
                dparams_list[name] =  np.append(dparams_list[name], result.params[name[:-5]].value)

        # Update the dparams_list in expt
        expt.dparams_list = dparams_list
                       
        #####################################################
        ###############   PLOT CURRENT FIT  #################
        #####################################################

        # Plot the spectrum and the fit, updated continuously

        # Dictionnary with the results of the last fit iteration
        dparams = {}
        for name in dparams_list:
            dparams[name[:-5]] = dparams_list[name][-1]
        spectrum_fit = Fcn_spectrum(dparams, elems, eV)

        clear_output(wait=True) # This line sets the refreshing
        fig = plt.figure(figsize=(15,10))
        fig.suptitle('Fit of spectrum %g/%g'%(count,(len(expt.spectrums)-1)), fontsize=14)
        fig.subplots_adjust(top=0.95)
        ax1 = fig.add_subplot(211)
        colors = iter(['r', 'b', 'y', 'c', 'm', 'g', 'orange', 'brown']*20)
        ax1.set(xlabel = 'E (eV)', ylabel = 'counts')
        for elem in elems:
            for line in elem.lines:
                position = line.position_list[-1]
                ax1.axvline(x = position,  color = next(colors) , label = elem.name+' '+line.name)
        ax1.plot(eV, spectrum, 'k.')
        ax1.plot(eV, spectrum_fit, 'r-')
        ax1.legend()
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = fig.add_subplot(212)
        colors = iter(['r', 'b', 'y', 'c', 'm', 'g', 'orange', 'brown']*20)
        ax2.set(xlabel = 'E (eV)', ylabel = 'counts')
        for elem in elems:
            for line in elem.lines:
                position = line.position_list[-1]
                ax2.axvline(x = position,  color = next(colors))
        ax2.plot(eV, spectrum, 'k.')
        ax2.plot(eV, spectrum_fit, 'r-')
        #ax2.plot(eV, dparams['sl']*eV+dparams['ct'], 'b-', label = 'Linear background')
        #ax2.legend(loc = 1)
        ax2.set_ylim(1,1e6)
        ax2.set_yscale('log')
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        plt.subplots_adjust(hspace=.0)
        plt.show()

        
        
        #####################################################
        #####################   SAVE   ######################
        #####################################################

        # Save the results from the fit
        if is_save:
            
            # Saving FitResults
            # Array to be written
            tbw = np.array([], dtype='float')

            # Put the data0D
            for i in range(len(data0D)):
                 tbw = np.append(tbw, data0D[i][expt.ind_first_spectrum+count])

            # Put the results from the fit
            for elem in elems:
                tbw = np.append(tbw,elem.area_list[-1])
                for line in elem.lines:
                    tbw = np.append(tbw,line.intRel_list[-1])
                    tbw = np.append(tbw,line.position_list[-1])
            for name in dparams_list:
                tbw = np.append(tbw,dparams_list[name][-1])    

            with open(expt.working_dir+expt.id+'/FitResults.csv', 'a+', newline='') as f:
                writer = csv.writer(f,delimiter=expt.delimiter)
                writer.writerow(tbw)

            # Saving FitSpectrums
            with open(expt.working_dir+expt.id+'/FitSpectrums.csv', 'a+', newline='') as f:
                for i in range(len(eV)):
                    writer = csv.writer(f,delimiter=expt.delimiter)
                    tbw = [expt.sensorsRelTimestamps[count],
                           np.round(eV[i],2),
                           np.round(spectrum[i],2),
                           np.round(spectrum_fit[i],2)]
                    writer.writerow(tbw)                

        count+=1
    #####################################################
    ##################   PLOT PARAMS  ###################
    #####################################################

    # At the end of the fit, plot the evolution of each param as a function of the scan
    print('####################################################')
    print('Fits are done. Results shown below.')
    print('####################################################')
    print('')

    # Results after fits
    expt.dparams_list = dparams_list

    FF.Plot_fit_results(expt, spectrum_index=None, dparams_list=dparams_list, is_save=is_save)

    print('#####################################################')
    print('Results are saved in:\n%s'%(expt.working_dir+expt.id+'/FitResults.csv'))

        
def Fcn2min(dparams, elems, eV, data):
    """
    Define objective function: returns the array to be minimized in the lmfit.
    """
    for elem in elems:
        elem.area = dparams['area_'+elem.name]

        for line in elem.lines:
            line.intRel = float(dparams['intRel_'+elem.name+'_'+line.name])
            line.position = float(dparams['pos_'+elem.name+'_'+line.name])

    model = Fcn_spectrum(dparams, elems, eV)

    return model - data


def Fcn_spectrum(dparams, elems, eV):
    """
    Definition of the spectrum as the sum of the peaks + the background.
    Takes a dictionnary with the params values, the list of Elem, the array of eV.
    """

    ct = dparams['ct']
    sl = dparams['sl']
    noise = dparams['noise']

    spectrum_tot = 0.
    for elem in elems:
        spectrum_elem = 0.
        area = elem.area
        for line in elem.lines:
            position = line.position
            intRel = line.intRel
            if elem.name == 'Compton':
                spectrum_elem += area*Fcn_compton_peak(position,intRel,eV,dparams)
            else:
                spectrum_elem += area*Fcn_peak(position,intRel,eV,dparams)
        spectrum_tot += spectrum_elem

    # We add a linear baseline, which cannot be < 0, and stops after the elastic peak (if there is one)
    limit_baseline = eV[-1]
    for elem in elems:
        if elem.name == 'Elastic':
            for line in elem.lines:
                if line.name == 'El':
                    limit_baseline = line.position

    eV_tmp = np.where(eV<limit_baseline+noise, eV, 0.)
    baseline = ct+sl*eV_tmp
    baseline = np.where(baseline>0.,baseline,0.)
    spectrum_tot+= baseline

    return spectrum_tot

def Interpolate_scf(atom, energy):
    """
    Interpolation for the scattering factors.
    Used here to take into account absorption from Si within the detector.
    Requires the file f-si in the same folder as the notebook.
    """
    en2=0.
    f2=0.
    f2p=0.
    for line in open('f-'+str(atom),'r'):
        en1=en2
        f1=f2
        f1p=f2p
        try:
            en2=float(line.split()[0])
            f2=float(line.split()[1])
            f2p=float(line.split()[2])
            if en1<=energy and en2>energy:
                scf=f1+(energy-en1)/(en2-en1)*(f2-f1)
                scfp=f1p+(energy-en1)/(en2-en1)*(f2p-f1p)
            else:
                pass
        except:
            pass
    return scf,scfp

def Fcn_peak(pos, amp, eV, dparams):
    """
    Definition of a peak (area normalised to 1).
    Following:
    - M. Van Gysel, P. Lemberge & P. Van Espen, “Implementation of a spectrum fitting
    procedure using a robust peak model”, X-Ray Spectrometry 32 (2003), 434–441
    - J. Campbell & J.-X. Wang, “Improved model for the intensity of low-energy tailing in
    Si (Li) x-ray spectra”, X-Ray Spectrometry 20 (1991), 191–197
    The params for peak definition should be passed as a dictionnary :
    dparams = {'sl': 0.01, 'ct':-23., 'sfa0':1.3 ... }
    """

    sfa0 = dparams['sfa0']
    sfa1 = dparams['sfa1']
    tfb0 = dparams['tfb0']
    tfb1 = dparams['tfb1']
    twc0 = dparams['twc0']
    twc1 = dparams['twc1']
    noise = dparams['noise']
    fano = dparams['fano']
    epsilon = dparams['epsilon']

    # We work in keV for the peak definition
    pos_keV = pos/1000.
    keV = eV/1000.

    # Peak width after correction from detector resolution (sigmajk)
    wid = np.sqrt((noise/2.3548)**2.+epsilon*fano*pos_keV)

    # Tail width (cannot be <0)
    TW = twc0 + twc1*pos_keV
    TW = np.where(TW>0.,TW,0.)

    # Energy dependent attenuation by Si in the detector
    atwe_Si = 28.086 #atomic weight in g/mol
    rad_el = 2.815e-15 #radius electron in m
    Na = 6.02e23 # in mol-1
    llambda = 12398./pos*1e-10 # in m
    # mass attenuation coefficient of Si in cm^2/g
    musi = 2.*llambda*rad_el*Na*1e4*float(Interpolate_scf('si',pos)[1])/atwe_Si

    # Shelf fraction SF (cannot be <0)
    SF = (sfa0 + sfa1*pos_keV)*musi
    SF = np.where(SF>0.,SF,0.)
    
    # Tail fraction TF (cannot be <0)
    TF = tfb0 + tfb1*musi
    TF = np.where(TF>0.,TF,0.)

    # Definition of gaussian
    arg = (keV-pos_keV)**2./(2.*wid**2.)
    farg = (keV-pos_keV)/wid
    gau = amp/(np.sqrt(2.*np.pi)*wid)*np.exp(-arg)

    # Function shelf S(i, Ejk)
    she = amp/(2.*pos_keV)*erfc(farg/np.sqrt(2.))

    # Function tail T(i, Ejk)
    tail = amp/(2.*wid*TW)*np.exp(farg/TW+1/(2*TW**2))*erfc(farg/np.sqrt(2.)+1./(np.sqrt(2.)*TW))

    # Function Peak
    ppic = np.array(gau+SF*she+TF*tail)

    return ppic

def Fcn_compton_peak(pos, amp, eV, dparams):
    """
    The function used to fit the compton peak, inspired by  M. Van Gysel, P. Lemberge & P. Van Espen,
    “Description of Compton peaks in energy-dispersive  x-ray fluorescence spectra”,
    X-Ray Spectrometry 32 (2003), 139–147
    The params for peak definition should be passed as a dictionnary :
    dparams = {'fG': 0.01, 'fA':2., ... }
    """

    fG = dparams['fG']
    fA = dparams['fA']
    fB = dparams['fB']
    gammaA = dparams['gammaA']
    gammaB = dparams['gammaB']
    noise = dparams['noise']
    fano = dparams['fano']
    epsilon = dparams['epsilon']

    # We work in keV for the peak definition
    pos_keV = pos/1000.
    keV = eV/1000.

    # Peak width after correction from detector resolution (sigmajk)
    wid = np.sqrt((noise/2.3548)**2.+epsilon*fano*pos_keV)

    # Definition of gaussian
    arg = (keV-pos_keV)**2./(2.*(fG*wid)**2.)
    gau = amp/(np.sqrt(2.*np.pi)*fG*wid)*np.exp(-arg)

    #Low energy tail TA
    farg = (keV-pos_keV)/wid
    TA = amp/(2.*wid*gammaA)*np.exp(farg/gammaA+1/(2*gammaA**2))*erfc(farg/np.sqrt(2.)+1./(np.sqrt(2.)*gammaA))

    #High energy tail TB
    TB = amp/(2.*wid*gammaB)*np.exp(-farg/gammaB+1/(2*gammaB**2))*erfc(-farg/np.sqrt(2.)+1./(np.sqrt(2.)*gammaB))

    ppic = np.array(gau+fA*TA+fB*TB)

    return ppic
