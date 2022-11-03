c-VEP based BCI using CNN for bitwise stimulation decoding
====

Python scripts of a Brain Computer Interface (BCI) using c-VEP stimuli to operate a T9 (11 classes). The offline but self-paced classification relies on a CNN to decode the stimulation pattern and then template matching to identify the target. It follows [EEG2Code](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221909) line of work. The GUI is using Psychopy<sup>3</sup>. The code is not specific to any device as it relies on [LSL](https://github.com/sccn/labstreaminglayer). Only some adjustements should be done regarding the EEG device used, you can find some details in [Example usage](#example-usage) section.
It was developped in the [Human-Factors department](https://personnel.isae-supaero.fr/neuroergonomie-et-facteurs-humains-dcas?lang=en) of ISAE-Supaero (France) by the team under the supervision of [Frédéric Dehais](https://personnel.isae-supaero.fr/frederic-dehais/).  

The code was used for an experiment in an under-review paper. The associated data are available [here](https://zenodo.org/).

## Contents

[Dependencies](#dependencies)  
[Installation](#installation)  
[Example usage](#example-usage)  
[Help](#help)

## Dependencies

* [Psychopy<sup>3</sup>](https://www.psychopy.org/download.html)
* [MNE](https://mne.tools/stable/install/mne_python.html)
* [pylsl](https://github.com/chkothe/pylsl)
* [Sklearn](https://scikit-learn.org/stable/install.html)
* [Keras](https://keras.io/)
* [imblearn](https://imbalanced-learn.org/stable/)
* Pickle

## Installation

Clone the repo:

```bash
git clone https://github.com/ludovicdmt/offline_cVEP_bitwise.git
cd ${INSTALL_PATH}
```

Install conda dependencies and the project with

```bash
conda env create -f environment.yml
```

The `pyRiemann` package has to be installed separately using `pip`:
```bash
conda activate psychopy
pip install pyriemann
```

If the dependencies in `environment.yml` change, update dependencies with

```bash
conda env update --file environment.yml
```

We used an EEG BrainProduct system to collect data with a native sampling frequency F<sub>s</sub> of 500Hz.

## Example Usage

To collect data, run the presentation part:

```bash
cd ${INSTALL_PATH}/presentation
python cVEP_offline.py
```

> A PyLSL stream for the markers EEG is created so it can be synchronized with the LSL EEG stream for recording.

To change stimulation, inter-trial, cue times or amplitude of the stimuli please go to the [config file](https://github.com/ludovicdmt/offline_cVEP_bitwise/blob/main/presentation/T9_config_cVEpoffline.json).  


After data collection you can run the classification. This part can also be used to reproduce results from the paper using our data.

To run the classifcation without golden subjects pre-training:
```bash
cd ${INSTALL_PATH}/classification
python run_async_comp.py
```

With the golden subjects pre-training:
```bash
cd ${INSTALL_PATH}/classification
python run_pretrain_comp.py
```

Classification performance are outputted in a `csv` file in the `results` directory.

## Help

You will probably need to do some adjustement to collect EEG stream if you are not using a BrainProduct EEG.  
If you experience issues during  use of this code, you can post a new issue on the [issues webpage](https://github.com/ludovicdmt/offline_cVEP_bitwise/issues).  
I will reply to you as soon as possible and I'm very interested in to improve it.


