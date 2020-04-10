# PyNexus
# P. Fontaine, Synchrotron SOLEIL
# v2.2, 04/11/2013  -  Correction bug spectrum
# v2.3, 02/12/2013  -  Ajout de la posibilite de traiter un ensemble de fichier
# v2.4, 27/04/2014  -  Modification de la maniere de traiter les images
# v2.5, 28/07/2014  -  Correction bug traiment 2D et liste de fichier
# v2.6, 21/11/2014  -  Possibilite de charger les donnees spectrum en un seul coup
#                       et non plus en point par point avec l'option -fast
# v3.0, 07/01/2015 -   Compatibilite avec Python 3.0, cf JIRA SIRIUSSUIV-36
# v4.0, 05/12/2017 -    Modification pour compatibilite avec PyTables 3.0
# v4.1, 26/09/2019 - Remplacement des / par des // pour obtenir des entiers (AH)

__version__="v4.1"

import tables as T
import numpy as N
import pylab as P
import sys
import types
import os.path as pth
import os
import glob

_RED='\x1b[31;01m'
_BLUE='\x1b[34;01m'
_GREEN='\x1b[32;01m'
_TURQUOISE='\x1b[36;01m'
_RESET="\x1b[0m"
_BOLD="\x1b[01;06m"

def get_aliases(filename):
        """ build a dictionnary of the aliases and tango adresses from the elements
                in file filename
                Input :
                        filename : the config file
                Output:
                        a dictionnary which keywords are the aliases
        """
        dict=None
        try:
                fichier=open(filename, 'r')
                dict={}
        except:
                print(_RED+'Trouble opening %s config file for aliases determination'%(filename)+_RESET)
        if dict!=None:
                ligne=fichier.readline()
                while ligne!='':
                        if ligne[1]!='#':
                                tmp=ligne.expandtabs()
                                liste=ligne.split()
                                if len(liste)==2:
                                    dict[liste[0]]=liste[1]
                        ligne=fichier.readline()
        return dict

class PyNexusFile():
        """ Class for a SOLEIL Nexus file """
        def __init__(self, filename, aliases=None, **keywords):
                self.__verbose__=False                                  # Verbose mode yes or no
                if 'fast' in list(keywords.keys()):                             # If True will load 1D,spectrum data in
                        self.fast=keywords['fast']                      # in one shot. If false, will load it
                else:                                                   # point by point.
                        self.fast=False
                if 'verbose' in list(keywords.keys()):
                        self.__verbose__=keywords['verbose']
                self.filename=filename                                  # Nexus File Name
                if self.__verbose__==True: print(_BOLD+(" - Opening Nexus File %s"%(filename))+_RESET)
                try:
                        self.fileproxy=T.open_file(filename)             # Open the Nexus files
                except:
                        print(_RED+("Did not succeed to open Nexus File %s"%(filename))+_RESET)
                        raise
                self.aliases=aliases


        def extract_scan_data(self):
                """ return all the scan data extracted from the nexus file
                        Input : None
                        Output : a tuple with 2 elements
                                - stamps : the name of the registered quantities
                                - the data itself as a numpy array
                """
                return self.extract('scan_data')

        def _get_node(self, nom):
                """ return the node of name nom """
                node_to_extract=None
                for node in self.fileproxy.walk_nodes():
                        if node._v_name==nom:
                                node_to_extract=node
                return node_to_extract


        def extract_one_data_point(self, dataset, num, verbose = True):
                """ extract one data point from scan_data
                        input : number of the point
                        output:
                                the stamp
                                the data point (could be an array)
                """
                stamp=None
                data=None
                # Looking after the node
                if verbose:
                    sys.stdout.write(' Extraction from scan_data of the point %d from leaf %s\r'%(num, dataset))
                    sys.stdout.flush()

                node_to_extract=self._get_node("scan_data")
                if node_to_extract!=None:
                        for leaf in node_to_extract:
                                if str(leaf.name).find(dataset)>-1 or str(leaf.get_attr('long_name')).find(dataset)>-1:
                                        leaf_to_use=leaf
                        # Extraction of the data
                        stamp=leaf_to_use.get_attr('long_name')
                        if stamp=='':
                                stamp=leaf_to_use.name
                        try:
                                alias=leaf_to_use.getAtrr('alias')
                                if alias=='':
                                        alias=None
                        except:
                                alias='None'
                        stamp=(stamp,)
                        d=leaf_to_use.read(num, num+1, 1)
                        if len(d.shape)==3:
                                data=N.resize(d, (d.shape[1], d.shape[2]))
                        elif len(d.shape)==2:
                                data=N.resize(d, (d.shape[1]))
                        elif len(d.shape)==1:
                                data=d[0]
                if verbose:
                    sys.stdout.write('                                                                      \r')
                    sys.stdout.flush()
                return (stamp, data)

        def extractStamps(self):
                """ extract only the stamps from scan_data
                        Input :
                        output:
                                stamps : list of the longname of the data
                """
                stamps=[]
                if self.__verbose__==True: print(_BOLD+(" - Extracting scan data stamps from file %s"%(self.filename))+_RESET)
                if self.__verbose__==True: print(_BLUE+(" \t. Exploring Nexus File to find Node scan_data")+_RESET)
                # Looking after the node
                node_to_extract=self._get_node('scan_data')
                if node_to_extract!=None:
                        # Counting the number of leaves
                        nb=0
                        for leaf in node_to_extract:
                                nb=nb+1
                        if self.__verbose__==True: print(_BLUE+(" \t. Node scan_data found with %i leaves"%(nb))+_RESET)
                        # Extraction of the data
                        for leaf in node_to_extract:
                                if 'long_name' in leaf.attrs:
                                        stamp=leaf.get_attr('long_name')
                                        if sys.version_info.major==3:
                                            try:
                                                stamp=stamp.decode()
                                            except:
                                                pass
                                if stamp=='':
                                        stamp=leaf.name
                                try:
                                        alias=leaf.get_attr('alias')
                                        if sys.version_info.major==3:
                                            alias=alias.decode()
                                        if alias=='':
                                            alias=None
                                except:
                                        alias=None
                                stamp=(stamp, alias)
                                stamps.append(stamp)
                        if self.__verbose__==True: print(_BLUE+(" \t. %d stamps returned "%(len(stamps)))+_RESET)
                return stamps

        def extractDataStamp(self, which=None):
                """ extract the the data from scan_data
                        Input :
                                which : the stamp of the data to extract
                        output:
                                data : data itself
                """
                data=[]
                if self.__verbose__==True: print(_BOLD+(" - Extracting scan data %s from file %s"%(which, self.filename))+_RESET)
                if self.__verbose__==True: print(_BLUE+(" \t. Exploring Nexus File to find Node scan_data")+_RESET)
                # Looking after the node
                node_to_extract=self._get_node('scan_data')
                if node_to_extract!=None:
                        # Counting the number of leaves
                        nb=0
                        for leaf in node_to_extract:
                                nb=nb+1
                        if self.__verbose__==True: print(_BLUE+(" \t. Node scan_data found with %i leaves"%(nb))+_RESET)
                        # Extraction of the data
                        leaf_to_extract=None
                        for leaf in node_to_extract:
                                d=None
                                if 'long_name' in leaf.attrs:
                                        stamp=leaf.get_attr('long_name')
                                        if sys.version_info.major==3:
                                            try:
                                                stamp=stamp.decode()
                                            except:
                                                pass
                                        if stamp=='':
                                                stamp=leaf.name
                                else:
                                        stamp=leaf.name
                                if stamp==which:
                                        leaf_to_extract=leaf
                        if leaf_to_extract!=None:
                                if self.__verbose__==True: print(_BLUE+("\t. Leaf %s"%(leaf_to_extract.name))+"found, reading the data "+which+_RESET)
                                leaf=leaf_to_extract
                                # read the data using a method depending of size.
                                if len(leaf.shape)==1 :
                                        d=leaf.read()
                                        stamps.append(stamp)
                                        data.append(d)
                                elif len(leaf.shape)==2 :
                                        d=N.zeros(leaf.shape)
                                        for i in range(leaf.shape[0]):
                                                d[i,:]=self.extract_one_data_point(leaf.name, i)[1]
                                        stamps.append(stamp)
                                        data.append(d)
                                elif len(leaf.shape)==3 :
                                        d=N.zeros(leaf.shape)
                                        for i in range(leaf.shape[0]):
                                                d[i,:,:]=self.extract_one_data_point(leaf.name, i)[1]
                                        stamps.append(stamp)
                                        data.append(d)
                                if self.__verbose__==True:
                                        if d is None:
                                                print("\t\t No %s data in this leaf"%(genre[which]))
                                        elif type(leaf)==T.array.Array:
                                                print("\t\t "+str(d.shape)+" data found")
                                        else:
                                                print("\t\t Strange data found !! ")
                return d

        def extractData(self, which='all'):
                """ extract the the data from scan_data
                        Input :
                                which : value = 'all', '0D', '1D', '2D'
                                SPECIFY IF ALL data, or only respectively point (0D), spectrum (1D)
                                or images (2D) have to be extracted.
                        output:
                                tuple with 2 elements :
                                        stamps : longname of the data
                                        data : data itself
                """
                genre={'all':'all', '0D':'point', '1D':'Spectrum', '2D':'Image'}
                stamps=[]
                data=[]
                if self.__verbose__==True:
                    if which=='all': print(_BOLD+(" - Extracting scan data from file %s"%(self.filename))+_RESET)
                    if which=='0D': print(_BOLD+(" - Extracting point scan data from file %s"%(self.filename))+_RESET)
                    if which=='1D': print(_BOLD+(" - Extracting 1D scan data from file %s"%(self.filename))+_RESET)
                    if which=='2D': print(_BOLD+(" - Extracting 2D scan data from file %s"%(self.filename))+_RESET)
                if self.__verbose__==True: print(_BLUE+(" \t. Exploring Nexus File to find Node scan_data")+_RESET)
                # Looking after the node
                node_to_extract=self._get_node('scan_data')
                if node_to_extract!=None:
                        # Counting the number of leaves
                        nb=0
                        for leaf in node_to_extract:
                                nb=nb+1
                        if self.__verbose__==True: print(_BLUE+(" \t. Node scan_data found with %i leaves"%(nb))+_RESET)
                        # Extraction of the data
                        for leaf in node_to_extract:
                                d=None
                                if 'long_name' in leaf.attrs:
                                        stamp=leaf.get_attr('long_name')
                                        if sys.version_info.major==3:
                                            try:
                                                stamp=stamp.decode()
                                            except:
                                                pass
                                        if stamp=='':
                                                stamp=leaf.name
                                else:
                                        stamp=leaf.name
                                try:
                                        alias=leaf.get_attr('alias')
                                        if sys.version_info.major==3:
                                            alias=alias.decode()
                                        if alias=='':
                                                alias=None
                                except:
                                        alias=None
                                try:
                                        unit=leaf.get_attr('units')
                                        if sys.version_info.major==3:
                                            unit=unit.decode()
                                        if unit=='':
                                                unit =None
                                except:
                                        unit =None
                                stamp=(stamp, alias, unit)
                                if stamp[1]==None:
                                        if self.__verbose__==True:
                                            print(_BLUE+("\t. From leave %s"%(str(leaf.name)))+(", reading the data %s"%(str(stamp[0])))+_RESET)
                                else:
                                        if self.__verbose__==True:
                                            print(_BLUE+("\t. From leave %s"%(str(leaf.name)))+(", reading the data %s"%(stamp[1]))+_RESET)
                                # read the data using a method depending of size.
                                if len(leaf.shape)==1 and (which in ('all', '0D')):
                                        d=leaf.read()
                                        stamps.append(stamp)
                                        data.append(d)
                                elif len(leaf.shape)==2 and (which in ('all', '1D')):
                                        d=N.zeros(leaf.shape)
                                        if self.fast:
                                                d=leaf.read()
                                        else:
                                                for i in range(leaf.shape[0]):
                                                        d[i,:]=self.extract_one_data_point(leaf.name, i)[1]
                                        stamps.append(stamp)
                                        data.append(d)
                                elif len(leaf.shape)==3 and (which in ('all', '2D')):
                                        d=N.zeros(leaf.shape)
                                        for i in range(leaf.shape[0]):
                                                d[i,:,:]=self.extract_one_data_point(leaf.name, i)[1]
                                        stamps.append(stamp)
                                        data.append(d)
                                if self.__verbose__==True:
                                        if d is None:
                                                print("\t\t No %s data in this leaf"%(genre[which]))
                                        elif type(leaf)==T.array.Array:
                                                print("\t\t "+str(d.shape)+" data found")
                                        else:
                                                print("\t\t Strange data found !! ")
                return (stamps, data)

        def extractAndSave2DData(self):
                """ extract and save the 2D data from scan_data
                        output:
                            nb_data_saved : numer of data point saved
                """
                if self.__verbose__==True:
                    print(_BOLD+(" - Extracting 2D scan data from file %s"%(self.filename))+_RESET)
                if self.__verbose__==True: print(_BLUE+(" \t. Exploring Nexus File to find Node scan_data")+_RESET)
                # Looking after the node
                node_to_extract=self._get_node('scan_data')
                if node_to_extract!=None:
                        # Counting the number of leaves
                        nb=0
                        for leaf in node_to_extract:
                                nb=nb+1
                        if self.__verbose__==True: print(_BLUE+(" \t. Node scan_data found with %i leaves"%(nb))+_RESET)
                        nb_2D_saved=0
                        # Extraction of the data
                        for leaf in node_to_extract:
                                d=None
                                if 'long_name' in leaf.attrs:
                                        stamp=leaf.get_attr('long_name')
                                        if stamp=='':
                                                stamp=leaf.name
                                else:
                                        stamp=leaf.name
                                if self.aliases!= None:
                                        if stamp in list(self.aliases.keys()):
                                                stamp=(stamp, self.aliases[stamp])
                                        else:
                                                stamp=(stamp, None)
                                else:
                                        stamp=(stamp, None)
                                if stamp[1]==None:
                                        if self.__verbose__==True: print(_BLUE+("\t. From leave %s"%(leaf.name))+", reading the data "+stamp[0]+_RESET)
                                        stamp=str(stamp[0])
                                else:
                                        if self.__verbose__==True: print(_BLUE+("\t. From leave %s"%(leaf.name))+", reading the data "+stamp[1]+_RESET)
                                        stamp=str(stamp[1])
                                # read the data using a method depending of size.
                                if len(leaf.shape)==3:
                                        dir=self.filename[:self.filename.rfind('.')]+"_2D"
                                        try:
                                            if not os.path.isdir(dir) : os.mkdir(dir)
                                        except:
                                            if self.__verbose__==True: print(_RED+("Impossible to create a directory to save 2D data, exiting")+_RESET)
                                            raise
                                        dir=dir+'/'
                                        print("\t\t %d points to extract and save"%(leaf.shape[0]))
                                        print("\t\t Images files will be saved in %s"%(dir))
                                        for i in range(leaf.shape[0]):
                                                stamp, mat=self.extract_one_data_point(leaf.name, i)
                                                if stamp[1]==None:
                                                    stamp=stamp[0]
                                                else:
                                                    stamp==stamp[1]
                                                self._saveSingleTwoDExtractedData(dir, stamp, i, mat)
                                                nb_2D_saved=nb_2D_saved+1
                                        print("\t\t %d points to extracted and saved"%(nb_2D_saved))


        def get_nbpts(self):
                """ return the number of points recorded in the nexus file """
                node=self._get_node("scan_data")
                retour=0
                cpt=0
                for l in node:
                        cpt=cpt+1
                        retour=retour+l.nrows
                retour=retour//cpt
                if retour!=l.nrows:
                        print(_RED+'Not the same number of points within all the data set. Stange!!'+_RESET)
                return retour

        def _savePointExtractedData(self, result):
                """ Generic function to save 0D point data """
                stamps=result[0]
                dat=result[1]
                nbpts=dat[0].shape[0]
                # Preparation of filenames
                i=self.filename.rfind('.')
                racine=self.filename[:i]
                # Analysis of data
                point_data=[]
                OneD_data=[]
                TwoD_data=[]
                for i in range(len(dat)):
                        if len(dat[i].shape)==1:
                                point_data.append(i)
                # Treatement of scalar (0D) data
                try:
                        if len(point_data)>0:
                                fichier=open(racine+'.dat', 'w')
                                ligne=""
                                for i in point_data:
                                        if stamps[i][1]==None:
                                                ligne=ligne+str(stamps[i][0])+'\t'
                                        else:
                                                ligne=ligne+str(stamps[i][1])+'\t'
                                ligne=ligne+'\n'
                                print(ligne)
                                fichier.write(ligne)
                                for pt in range(nbpts):
                                        ligne=""
                                        for i in point_data:
                                                ligne=ligne+str(dat[i][pt])+'\t'
                                        ligne=ligne+('\n')
                                        fichier.write(ligne)
                                fichier.close()
                                if self.__verbose__:
                                        print(_BLUE+"\t. Scalar (0D) data saved in file %s : "%(racine+".dat")+_RESET)
                                        for i in point_data:
                                                if stamps[i][1]==None:
                                                        print("\t\t-> "+stamps[i][0])
                                                else:
                                                        print("\t\t-> "+stamps[i][1])
                        else:
                                if self.__verbose__:
                                        print(_BLUE+"\t. No Scalar (0D) data to save "+_RESET)

                except:
                        print(_RED+'Unable to save scalar (0D) data'+_RESET)
                        raise

        def _saveOneDExtractedData(self, result):
                """ Generic function to save 1D spectrum data"""
                stamps=result[0]
                dat=result[1]
                nbpts=dat[0].shape[0]
                # Preparation of filenames
                i=self.filename.rfind('.')
                racine=self.filename[:i]
                # Analysis of data
                OneD_data=[]
                for i in range(len(dat)):
                        if len(dat[i].shape)==2:
                                OneD_data.append(i)
                # Treatment of spectrum (1D) data
                if self.__verbose__ and len(OneD_data)==0:
                        print(_BLUE+"\t. No Spectrum (1D) data to save"+_RESET)
                if self.__verbose__ and len(OneD_data)>0:
                        print(_BLUE+"\t. Spectrum (1D) data :"+_RESET)
                for i in OneD_data:
                        if stamps[i][1]==None:
                                stamp=stamps[i][0]
                        else:
                                stamp=stamps[i][1]
                        filename=racine+'_'+stamp.replace( '/', '_')+'.mat'
                        try:
                                P.savetxt(filename, N.array(dat[i]))
                                if self.__verbose__:
                                        print("\t\t-> %s Spectrum (1D) data saved in %s"%(stamp, filename))
                        except:
                                print(_RED+'Unable to save Spectrum (1D) data '+stamp)
                                raise
        def _saveSingleTwoDExtractedData(self, dir, stamp, num, mat):
                """ Generic function to save a single 2D image data
                     Input :    dir : directory where image have to be stored
                                stamp : sensor attached to the image
                                num : point (image) number
                                mat : the matrix of the image itself
                """
                # Preparation of filenames
                i=self.filename.rfind('.')
                racine=self.filename[self.filename.rfind('/')+1:self.filename.rfind('.')]
                filename=dir+racine+'_'+stamp.replace('/', '_')
                try:
                    sys.stdout.write('save  image %d    \r'%(num))
                    sys.stdout.flush()
                    P.savetxt(filename+'_'+str(num)+'.imag', mat)
                    sys.stdout.write('               \r')
                    sys.stdout.flush()
                except:
                        print(_RED+'Unable to save image (2D) data '+stamp)
                        raise

        def _saveTwoDExtractedData(self, result):
                """ Generic function to save 2D image data"""
                stamps=result[0]
                dat=result[1]
                nbpts=dat[0].shape[0]
                # Preparation of filenames
                i=self.filename.rfind('.')
                racine=self.filename[:i]
                # Analysis of data
                TwoD_data=[]
                for i in range(len(dat)):
                        if len(dat[i].shape)==3:
                                TwoD_data.append(i)
                # Treatment of image (2D) data
                if self.__verbose__ and len(TwoD_data)==0:
                        print(_BLUE+"\t. No Image (2D) data to save"+_RESET)
                for i in TwoD_data:
                        if self.__verbose__:
                                print(_BLUE+"\t. Images (2D) data :"+_RESET)
                        if stamps[i][1]==None:
                                stamp=stamps[i][0]
                        else:
                                stamp=stamps[i][1]
                        filename=racine+'_'+stamp.replace('/', '_')
                        try:
                                for num in range(nbpts):
                                        sys.stdout.write('image %d/%d    \r'%(num, nbpts))
                                        sys.stdout.flush()
                                        P.savetxt(filename+'.'+str(num)+'.imag', dat[i][num, :,:])
                                sys.stdout.write('               \r')
                                sys.stdout.flush()
                                if self.__verbose__:
                                        print("\t\t-> %s Images (2D) data saved in %s from 0 to %d"%(stamps, filename+""".##.imag""", nbpts-1))
                        except:
                                print(_RED+'Unable to save image (2D) data '+stamps)
                                raise

        def savePointExtractedData(self, result):
                """ Save only the 0D, point, scalar, previously extracted data to a .dat file
                        Input : result : a tuple with stamp (name) and data coming from the extract functions
                        Output : the file on the disk
                """
                if self.__verbose__==True :
                    print(_BOLD+" - Save point (0D) extracted data from Nexus File : "+self.filename +_RESET)
                dat=result[1]
                if len(dat)==0:
                    if self.__verbose__: print("\t. No data point to save !!")
                else:
                    nbpts=dat[0].shape[0]
                    if self.__verbose__==True :
                        print(_BLUE+"\t. "+str(nbpts)+" data points to save"+_RESET)
                    # Treatement of scalar (0D) data
                    self._savePointExtractedData(result)

        def saveOneDExtractedData(self, result):
                """ Save only the 1D spectrum previously extracted data to a .mat file
                        Input : result : a tuple with stamp (name) and data coming from the extract functions
                        Output : the file on the disk
                """
                dat=result[1]
                if self.__verbose__==True :
                    print(_BOLD+" - Save Spectrum (1D) extracted data from Nexus File : "+self.filename +_RESET)
                if len(dat)==0:
                    if self.__verbose__: print("\t. No data point to save !!")
                else:
                    nbpts=dat[0].shape[0]
                    if self.__verbose__==True :
                        print(_BLUE+"\t. "+str(nbpts)+" data points to save"+_RESET)
                    # Treatment of spectrum (1D) data
                    self._saveOneDExtractedData(result)

        def saveTwoDExtractedData(self, result):
                """ Save only the 2D Image previously extracted data to a .imag file
                        Input : result : a tuple with stamp (name) and data coming from the extract functions
                        Output : the file on the disk
                """
                dat=result[1]
                nbpts=dat[0].shape[0]
                if self.__verbose__==True :
                        print(_BOLD+" - Save image (2D) extracted data from Nexus File : "+self.filename +_RESET)
                        print(_BLUE+"\t. "+str(nbpts)+" data points to save"+_RESET)
                # Treatment of image (2D) data
                self._saveTwoDExtractedData(result)

        def saveExtractedData(self, result):
                """ Save all the previously extracted data to files
                        Input : result : a tuple with stamp (name) and data coming from the extract functions
                        Output : the file on the disk, .dat for 0D point data, .mat for spectrum data, and .imag for image data
                """
                dat=result[1]
                if self.__verbose__==True :
                        print(_BOLD+" - Save extracted data from Nexus File : "+self.filename +_RESET)
                if len(dat)==0:
                        if self.__verbose__: print("\t. No data point to save !!")
                else:
                        nbpts=dat[0].shape[0]
                        if self.__verbose__: print(_BLUE+"\t. "+str(nbpts)+" data points to save"+_RESET)
                        # Treatement of scalar (0D) data
                        self._savePointExtractedData(result)
                        # Treatment of spectrum (1D) data
                        self._saveOneDExtractedData(result)
                        # Treatment of image (2D) data
                        self._saveTwoDExtractedData(result)

        def close(self):
                """ Close the Nexus File """
                if self.__verbose__==True: print(_BOLD+(" - Closing Nexus File %s"%(self.filename))+_RESET)
                self.fileproxy.close()

if __name__=="__main__":
        stamps=None
        data=None
        save=False
        save0D=False
        save1D=False
        save2D=False
        print_info=False
        plot=False
        local=False
        x=None
        y=None
        log=None
        fast=False
        verbeux=False
        config=None
        print('')
        print(_BOLD+" - ---------------------PyNexus "+__version__+"---------------------------"+_RESET)

        if len(sys.argv)>=2:
                for i in range(1, len(sys.argv), 1):
                        if sys.argv[i]=='-v':
                                verbeux=True
                        if sys.argv[i]=='-fast':
                                fast=True
                        if sys.argv[i]=='-x':
                                x=int(sys.argv[i+1])
                        if sys.argv[i]=='-y':
                                y=int(sys.argv[i+1])
                        if sys.argv[i]=='-s':
                                if len(sys.argv)>i+1:
                                        if sys.argv[i+1]=='0D':
                                                save0D=True
                                                save=False
                                        elif sys.argv[i+1]=='1D':
                                                save1D=True
                                                save=False
                                        elif sys.argv[i+1]=='2D':
                                                save2D=True
                                                save=False
                                        else:
                                                save0D=True
                                                save1D=True
                                                save2D=False
                        if sys.argv[i].find('-l')>-1:
                                local=True
                        if sys.argv[i].find("-i")>-1:
                                print_info=True
                        if sys.argv[i].find("-p")>-1:
                                plot=True
                                if len(sys.argv)>i+1:
                                        if sys.argv[i+1]=='log':
                                                log='log'
                                        elif sys.argv[i+1]=='logx':
                                                log='logx'
                                        elif sys.argv[i+1]=='logy':
                                                log='logy'
        else:
                print("PyNexus function "+__version__)
                print("")
                print("Usage : python PyNexus.py [options] my_nexus_file.nxs")
                print("")
                print("Where options include:")
                print("\t\t-c config.txt : use config.txt as source file for aliases conversion")
                print("\t\t-s : to save all the 0D and 1D extracted data to ASCII files")
                print("\t\t-s 0D : to save only 0D, point extracted data to ASCII files")
                print("\t\t-s 1D : to save only 1D, spectrum extracted data to ASCII files")
                print("\t\t-s 2D : to save only 2D, image extracted data to ASCII files")
          #     print "\t\t-l : when to save option active, save the ASCII file in the working directory"
          #     print "\t\t     rather than in the original directory"
                print("\t\t-p : plot the extracted 2D data")
                print("\t\t-p logx : plot the extracted 2D data in logx scale")
                print("\t\t-p logy : plot the extracted 2D data in logy scale")
                print("\t\t-p log : plot the extracted 2D data in loglog scale")
                print("\t\t-x : when plot option selected : specify the x axis")
                print("\t\t-y : when plot option selected, specify the y axis")
                print("\t\t-v : Verbose mode on, off if not specified")
                print("\t\t-fast : enable to load 1D, spectrum data in one shot from the disk ")
                print("\t\t        instead of the default point to point mode. Can be used when done")
                print("\t\t        done on fast computer with fast connection to the data file ")
                print("")
                print(_BOLD+" - -----------------------------------------------------------"+_RESET)
                quit()



        if config!=None:
                aliases=get_aliases(config)
                if aliases!=None:
                        print(_BOLD+(" - %d aliases found in conf file %s"%(len(aliases), config))+_RESET)
        else:
                aliases=None
        strt=1
        while strt<len(sys.argv) and sys.argv[strt][0]=='-':
            strt=strt+1
        filelist=[]
        for filename in sys.argv[strt:]:
            if filename.find('.nxs')>-1:
                filelist.append(filename)
        print(_RED+(' - %d Nexus file(s) to treat'%(len(filelist)))+_RESET)
        for filename in filelist:
            print(_RED+" Treat : "+filename+_RESET)
            ok=False
            if filename.rfind(".nxs")>-1:
                try:
                        nexus=PyNexusFile(filename, verbose=verbeux, fast=fast)
                        ok=True
                except:
                        print(_RED+'Trouble with presumed nexus file '+filename+_RESET)
                        ok=False
            else:
                print(_RED+filename+"is not a Nexus files (.nxs)"+_RESET)
            if ok : # The Nexus file is indeed opened
                #try:
                if 1==1:
                    if save0D:
                        stamps, data = nexus.extractData('0D')
                        nexus.savePointExtractedData((stamps, data))
                    if save1D:
                        stamps, data = nexus.extractData('1D')
                        nexus.saveOneDExtractedData((stamps, data))
                    if save2D:
                        nexus.extractAndSave2DData()
                    if plot:
                        stamps=nexus.extractStamps()
                        if x==None or y==None:
                            print(_BLUE+"\t. Available counter to plot : "+_RESET)
                            for i in range(len(stamps)):
                                if stamps[i][1] is None:
                                        stamp=stamps[i][0]
                                else:
                                        stamp=stamps[i][1]
                                print("\t\t%d\t--> %s"%(i, stamp))
                            print("\t\tUse option -x i -y j to specify the axis of the plot")
                        else:
                            for i in range(len(stamps)):
                                if stamps[i][1] is None:
                                        stamp=stamps[i][0]
                                else:
                                        stamp=stamps[i][1]
                                print("\t\t%d\t--> %s"%(i, stamp))
                            datax=nexus.extractDataStamp(stamps[x][0])
                            datay=nexus.extractDataStamp(stamps[y][0])
                            if len(datay.shape)==2:
                                datay=datay.sum(axis=1)
                            if len(datay.shape)==3:
                                datay=datay.sum(axis=2)
                                datay=datay.sum(axis=1)
                            if len(datax.shape)==2:
                                datax=datax.sum(axis=1)
                            if len(datax.shape)==3:
                                datax=datax.sum(axis=2)
                                datax=datax.sum(axis=1)
                            if log==None:
                                P.plot(datax,datay, 'ro-')
                            elif log=='log':
                                P.loglog(datax,datay, 'ro-')
                            elif log=='logx':
                                P.semilogx(datax,datay, 'ro-')
                            elif log=='logy':
                                P.semilogy(datax,datay, 'ro-')
                            if stamps[x][1]==None:
                                P.xlabel(stamps[x][0])
                            else:
                                P.xlabel(stamps[x][1])
                            if stamps[y][1]==None:
                                P.ylabel(stamps[y][0])
                            else:
                                P.ylabel(stamps[y][1])
                            P.title(nexus.filename)
                try:
                    pass
                except:
                    print(_RED+'Trouble Reading or handling data'+_RESET)
            nexus.close()
        print(_BOLD+" - ---------------------PyNexus "+__version__+"---------------------------"+_RESET)
        print("")
        if plot:
                print("")
                sys.stdout.write("Hit Enter to quit\r")
                sys.stdout.flush()
                if sys.version_info.major==2:
                    a=raw_input()
                elif sys.version_info.major==3:
                    a=input()
                print("                              ")

