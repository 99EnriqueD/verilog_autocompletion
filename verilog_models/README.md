# Verilog Autocompletion Models
Code implementation for training and evaluating deep learning models from "[A Deep Learning Framework for Verilog Autocompletion Towards Design and Verification Automation](https://arxiv.org/abs/2304.13840)" which is to be presented at DAC 2023. 

## Dataset
The final dataset csvs can be accessed in [this Google Drive folder](https://drive.google.com/drive/folders/1J0Y8u3u1mGJ-NflPtd9AdmTJR7ylTFRM?usp=sharing). The *licenses* subfolder provides repository and file indexing as well as license information.

To use the datasets with the code here, download them to the data folder.

## Installation

A conda environment with all neccessary packages can be made using the following command:

`conda env create -f conda_env.yml`

However, **you may need to change the pytorch and cuda versions** depending on your setup.

## Scripts
Example scripts for running training and evaluation of various model types (and datasets as documented in the scripts themselves) on a PBS system are provided in the scripts subfolder. 

## Citation
If the datasets and/or code in this repository helped you with your work, please cite our work as:

`@misc{dehaerne2023deep,
      title={A Deep Learning Framework for Verilog Autocompletion Towards Design and Verification Automation}, 
      author={Enrique Dehaerne and Bappaditya Dey and Sandip Halder and Stefan De Gendt},
      year={2023},
      eprint={2304.13840},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}`