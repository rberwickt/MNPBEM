#  <img src="../GUI/images/Landes_group_logo_cropped.png" alt="Landes Group Logo" width="80" height="80" style="vertical-align: middle;"> Python MNPBEM GUI Guide 

### Setup
Detailed setup instructions can be found in the [setup guide](./GUI_SETUP.md).

## How to use the GUI
The GUI is divided into 3 main screens: __Setup__, __Simulation__, and __Post-Processing__. 
>At this time, there is no way to move to the previous screen, so any mistakes made on the previous screen will have to be corrected by running the GUI again.

#### Using graph settings

All matplotlib graphs in the GUI share similar tools which allow for changing what is shown as well as saving as an image/pdf. These tools can be found in a bar on top of the graph as shown below.

<img src="./gui_images/graph_bar.png" alt="Graph Settings Bar" style="vertical-align: middle;">

The 3 most important settings are the 3 rightmost buttons. The floppy disk is used to save the graph, and is straightforward.

##### Configure Subplots
<img src="./gui_images/edit_bars_highlighted.png" alt="Configure Subplots Settings Icon" style="vertical-align: middle;">

The "Configure Subplots" button, which is highlighted above (the button with sliders) is used to change the area on the Figure window that the graph is allowed to be in. 

<img src="./gui_images/edit_bars.png" alt="Subplot Settings Menu" style="vertical-align: middle;">

The borders and spacings of the graph area are able to be changed freely, and will update live as they are edited. The most useful setting here is the "Tight Layout" button, as it will attempt to fit the graph nicely onto your screen.

##### Edit Axis and Curves

<img src="./gui_images/axes_curves_highlighted.png" alt="Edit axis and curve button icon" style="vertical-align: middle;">

The "Edit axis, curve, and image parameters" button, which is highligthed above (the button with an arrow on a plot) is used to change a variety of options above the axes and curves, including color, bounds, labels, and style. 

<img src="./gui_images/axes.png" alt="Axes Edit Menu" style="vertical-align: middle;"> <img src="./gui_images/curves.png" alt="Curves Edit Menu" style="vertical-align: middle;">

If you would like to remove a curve, simply change the line style to "None" and click the "Apply" button to remove it. To show the individual points on a plot, change the marker from "nothing" to your desired style.

### Setup

Launching the GUI will bring you to the setup screen, where you can configure the compute settings (number of workers, threads, and GPUs per worker) as well as load material files for use in the simulation.

<img src="./gui_images/new_setup.png" alt="GUI Setup Screen" style="vertical-align: middle;">

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

<img src="./gui_images/new_overview.png" alt="GUI Simulation Dashboard" style="vertical-align: middle;">

#### Solver Selection

This widget determines which solver should be used for the simulation, whether to solve for fields, the range of wavelengths/energy to solve over, and GPU precision (if applicable). 

The wavelength range uses spline interpolation with the number of steps to genereate the wavelengths to calculate for, with more steps giving a more accurate simulation. 

#### Field Grid Settings

If field calculations are enabled, the field grid is the area and amount of points the field is calculated at. It is recommended to have at least double the diameter of the structure on each side. An example is a 50nm diameter sphere would be recommended to have the x and y set at at least 100nm in each direction.

Increasing nx and ny increases the number of points caluclated, which corresponds to how accurate the resulting grid is. More points means a less blurry-looking interpolated graph in your results.

The z range and steps can remain at 0, as unless a 3D graph of points is required, one slice generally illustrates the structure quite well. Increasing the amount of slices will have a large impact on run time for field calculations, so only change it when needed.

#### Refractive Index Graph

This graph is purely for inspecting the dielectric function of the materials, and does not effect the simulation at all. It allows for viewing 

#### Environment Settings



#### Structure Settings


##### Mesh Preview



#### Excitation Settings



