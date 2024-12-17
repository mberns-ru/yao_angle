# Yao Lab Angle Analysis

Takes SLEAP-tracked videos and calculates angle speed and turn duration.

Angle calculations: Xingeng Zhang, Anindita Chavan
Analysis: Madeline Berns

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)


## Installation

To install, first download the files and place them in a project folder.
Then, open Anaconda Prompt and type the following to install the dependencies in a new Conda environment:

```bash
cd project_folder
conda env create --file requirements.yml
conda activate angle_env
```

## Usage

To run the model, type the following command into the active Conda environment (angle_env):

```bash
python get_angles.py
```

This will open a GUI with the following inputs:
- Select Data Path: Path to folder of animal data (i.e. 'data/F38_HL/'). This should contain multiple sessions, including a TEMPLATE one.

Then click "Run Script" and you're good to go!

Once all data has been processed, you can run the following code on the overall data folder to analyze all data pre- and post-HL:
```bash
python analyze_angles.py
```

This will open a GUI with the following inputs:
- Select Data Folder: Path to folder of ALL data (i.e. '/data'). Should contain multiple animals pre- and post- HL conditions.

Then click "Run Analysis." All results will be saved in /angle_results.