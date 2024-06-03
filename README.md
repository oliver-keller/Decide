# DECIDE (**D**ecoding **E**nergy scenarios **C**omplexity **I**nto **DE**cisions)

This software package translates many energy scenarios into storylines represented by decision trees. 

## Referencing

If you use our software or any part of it, please cite the following publication:

F. Baader, S. Moret, W. Wiesemann, I. Staffell, A. Bardow. (2023): "Streamlining Energy Transition Scenarios to Key Policy Decisions", 
https://doi.org/10.48550/arXiv.2311.06625


## Content

The repository contains the following folders:

- utilities: core scripts common to all models and case studies
- european\_case\_study: scripts and data to reproduce the European case study in the paper
- global\_case\_study: scripts and data to reproduce the Global case study in the paper

## Getting Started

The code was developed and tested with Python 3.9. Additional package requirements are indicated in the requirements.txt file.

First, clone a copy of the whole repository to your machine. 
Navigate to your local folder, open a command prompt and execute: 

```
git clone https://gitlab.ethz.ch/epse/systems-design-public/decide.git
```

To reproduce the results of our case studies, navigate to the corresponding folders and run the python scripts. 
For example, for the Global case study, run the following:

```
cd global_case_study
python pre_processing.py
python run_tree.py
python explain_tree.py
```

The results and figures will be available in the corresponding folders.

To run the code for other case studies, add your input data in the raw\_data folder and run the above scripts.

## License

The software package is available open-source under the [Apache 2.0 licence](https://opensource.org/license/apache-2-0/)
