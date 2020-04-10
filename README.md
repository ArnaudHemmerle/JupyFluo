# FluoJupyter

FluoJupyter is a Jupyter Notebook to analyze X-Ray Fluorescence (XRF) experiments on the beamline SIRIUS at the synchrotron SOLEIL. The notebook should be first set up by an Expert following the instructions in the "Expert" section. Users can then follow the guidelines in the "User" section to start using the notebook.

## User

## Expert

### Getting Started

Copy the notebook FluoJupyter.ipynb on the local folder where the nxs files are. The files should contain 'Fluo' in their name. Note that the notebook does not work with JupyterLab in its current version.

The aim of the Expert part is to determine the parameters in the first cell. It can be quite painful, but those parameters should not vary during an experiment. It is also possible to copy directly the parameters from a previous experiment (or from the examples provided here), and test if they are good for your current experimental setup.

### Conversion channels <-> eVs
First update the parameters 'gain' and 'eV0':
```
dparams_general['gain'] = 10.
dparams_general['eV0'] = 0.
```
used to convert the channels in eVs through the relation:
```
eVs = gain*channels + eV0
```


dparams_general['gain'] = 1.

# Energy of the channel 0
dparams_general['eV0'] = 0.


### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
