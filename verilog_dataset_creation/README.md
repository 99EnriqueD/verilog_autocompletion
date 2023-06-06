# Open Source Verilog Code Dataset Collection and Curation
This code was used to collect the Verilog dataset for "[A Deep Learning Framework for Verilog Autocompletion Towards Design and Verification Automation](https://arxiv.org/abs/2304.13840)" which is to be presented at DAC 2023. Easily adaptable to creating a dataset for other languages (e.g., VHDL) as well.

## Dataset
The final dataset csvs can be accessed in [this Google Drive folder](https://drive.google.com/drive/folders/1J0Y8u3u1mGJ-NflPtd9AdmTJR7ylTFRM?usp=sharing). The *licenses* subfolder provides repository and file indexing as well as license information.  

## Installation
A conda environment with all neccessary packages can be made using the following command:

`conda env create -f conda_env.yml`

Note that the environment can contain extra dependencies not required for running all code in this project.

## Overview of code
### github_download.ipynb
Notebook used to search and download code from GitHub.
### verilog_analysis_and_filtering.ipynb
Notebook used to analyze, filter, and remove exact copies of files
### verilog_parsing.ipynb
For parsing verilog files using icarus ([pyverilog](https://pypi.org/project/pyverilog/)) and [verilator](https://www.veripool.org/verilator/). Big thank you to both for making their work open-source.
### split_dataset.ipynb
Splitting dataset into train/val/test splits. Also some extra, split-specific processing code.
### mine_labeled_data.ipynb
Mining *module* and *function* snippets from unlabeled, full-file data.
### sourcerercc_tokenizing.py
Tokenizer code for identifying near-duplicate files. Based on code from [SourcererCC](https://github.com/Mondego/SourcererCC/blob/88314f51fdf3cd89103eb2c19fb56464308846c8/tokenizers/file-level/tokenizer.py)
### utils.py
Supplementary code used in *verilog_analysis_and_filtering.ipynb* and *verilog_parsing.ipynb*.

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