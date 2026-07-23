#  <img src="../GUI/images/Landes_group_logo_cropped.png" alt="Landes Group Logo" width="80" height="80" style="vertical-align: middle;"> Python MNPBEM GUI Guide 

### Setup
Detailed setup instructions can be found in the [setup guide](./GUI_SETUP.md).

## How to use the GUI
The GUI is divided into 3 main screens: __Setup__, __Simulation__, and __Post-Processing__. 
>At this time, there is no way to move to the previous screen, so any mistakes made on the previous screen will have to be corrected by running the GUI again.

### Setup

Launching the GUI will bring you to the setup screen, where you can configure the compute settings (number of workers, threads, and GPUs per worker) as well as load material files for use in the simulation.

<img src="./gui_images/new_setup.png" alt="Landes Group Logo" style="vertical-align: middle;">

#### Environment Setup

The main driver of the simulation is the number of workers, which each run the specified amount of threads. It is generally recommended to not have the total number of threads (`workers * threads`) exceed the amount of physical cores in your CPU when running in CPU mode (GPUs per worker set as 0).

The Setup screen will automatically detect the amount of CPU logical processors (not the amount of physical cores) and the amount of GPUs. For most CPUs, the amount of physical cores is half the amount of logical processors, but verify by checking for your machine first. When running the simulation in GPU mode, do not exceed the number of connected GPUs (threads can still be increased).

There will be a recommended configuration, but it does not guarantee the best performace. Benchamarking different configurations can help find the best configuration for your system.

#### Loading Material Files

The user defined content folder can be opened by pressing the button at the bottom of the screen labeled "Open User-Defined Content Folder". In the folder labeled "materials", all of the material files for the simulation are placed. 

For a material file to be recognized as valid, it must be either a `.dat` table file with 3 columns: Energy (in eV), n, k where n is the real part of the refractive index and k is the imaginary part (often called the extinction coefficient), or a `.py` code file with a specific function header. An commented example of the function header can be found in the `vacuum.py` material file, and can serve as a reference for creating your own. 

While on the setup screen, any new files added to the material folder will automatically be loaded, and should appear in the "Loaded Materials" section if they are successfully loaded into the GUI. If a file is added to the folder after continuing to the simulation screen, then it will only be loaded on the next time the setup screen is shown.

#### Continuing to Simulation
Once the environment has been configured, and all desired materials have been loaded into the simulation, pressing the "Continue to Simulation" button at the bottom of the screen will progress to the simulation configuration.

### Simulation

This screen is where all of the settings for the metal nanoparticle BEM simulations are configured, and several properties can be examined. At this time, the simulation will solve for cross section spectra, and near field enhancement.

Once all settings have been configured, the simulation can be started with the button labeled "Run Simulation" on the right side.

<img src="./gui_images/new_overview.png" alt="Landes Group Logo" style="vertical-align: middle;">

#### Solver Selection

This widget determines which solver should be used for the simulation, whether to solve for fields, the range of wavelengths/energy to solve over, and GPU precision (if applicable). 

The two available solvers are Retarded and Quasistatic, with Retarded generally being more accurate. Both will give results, with Quasistatic being the faster of the two.

The wavelength range uses spline interpolation with the number of steps to genereate the wavelengths to calculate for, with more steps giving a more accurate simulation. 

#### Field Grid Settings



#### Refractive Index Graph



#### Environment Settings



#### Structure Settings


##### Mesh Preview



#### Excitation Settings



