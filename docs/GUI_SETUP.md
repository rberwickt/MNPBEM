# MNPBEM GUI Environment Setup Guide

## Download GitHub Repos

Start by downloading this repo (https://github.com/rberwickt/MNPBEM.git) and the simulation pipeline repo (https://github.com/Yoo-JK/pymnpbem_simulation.git) onto your computer. Don't place them in the same folder, as it may conflict when they are installed as pip packages.

## Conda Environment Setup

Create the environment using the setup file (environment.yml) by running `conda env create -f environment.yml --solver=libmamba` in the same folder using anaconda/miniconda. The solver argument fixes some issues with the packages taking a very long time to be resolved. This new environment should be called "mnpbem". Activate the environment with `conda activate mnpbem` and then navigate to where the repos from earlier are downloaded. 

In the simulation repo, run `pip install -e .`, this will install the repo as a package with the editable tag. The editable tag allows for the package to automatically update when the folder is changed (such as by pulling from the GitHub repo).

In the main MNPBEM repo. run `pip install -e ".[GPU]"` (or whatever installation you need as outlined in the INSTALL.md doc). After this, your environment should be setup for running the GUI. If you would like to use GPU acceleration, you must run `set MNPBEM_GPU=1`. 

## Running the GUI

Navigate to the folder of the MNPBEM repo, and run `python -m GUI.gui_main`. This should open up the GUI. 

This can also be substituted with a batch file (or the non-windows equivalent) to make things one click. An example is shown here:

**`MNPBEM_GUI.bat`**
```bat
@echo off
call "C:\Users\USER\Anaconda3\condabin\conda.bat" activate mnpbem
set MNPBEM_GUI=1
cd "WHEREVER MNPBEM REPO IS INSTALLED"

python -m GUI.gui_main
pause
```

## Updating the GUI

Since the two dependencies are installed as editable modules, the GUI can be updated simply by pulling new changes from the GitHub repo. This can be done by running `git pull` (or `git pull origin (whatever branch)` if you need a specific branch) in each folder.

### Haven't written the guide on how to use it yet. :/
