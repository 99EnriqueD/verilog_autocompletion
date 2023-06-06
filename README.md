# Verilog Autocompletion
Code for "[A Deep Learning Framework for Verilog Autocompletion Towards Design and Verification Automation](https://arxiv.org/abs/2304.13840)" which is to be presented at DAC 2023.

The code is split into two parts with corresponding directories in this repository:
- **verilog_datasets_creation**: code used for creating and curating our Verilog code datasets.
- **verilog_autocompletion**: code used to train and evaluate Verilog autocompletion models.
Each part has its own README file with installation instructions and an overview of the code inside. Note that the code has not been tested since refactoring.

## Dataset
The final dataset csvs can be accessed in [this Google Drive folder](https://drive.google.com/drive/folders/1J0Y8u3u1mGJ-NflPtd9AdmTJR7ylTFRM?usp=sharing). The *licenses* subfolder provides repository and file indexing as well as license information.  

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
