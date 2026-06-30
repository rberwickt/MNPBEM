## PyMNPBEM GUI Correspondance Mapping
##### Section-by-section mapping of MATLAB GUI functionality to PyMNPBEM Functions

### BEM Solvers
###### Relatively straigtforward mapping, selects what solver class to use for the simulation
##### Retarded
* `BEMRet, BEMRetLayer, BEMRetIter(?)`
##### Quasistatic
* `BEMStat, BEMStatlayer`
##### Iterative
* Unclear, has `BEMRetIter`but also other functions shown below
* `op = dict(sim='ret', interp='curv', RelCutoff=2)`
* "pass `iter={'tol':1e-6,'restart':30}` to solver"
* "pass `aca={'htol':1e-6,'kmax':100}` to solver"

### Excitation Sources
Python Implementation has 2 different classes for each excitation: Retarded vs. Static (Iterative falls under Retarded?)

Oddly the MATLAB GUI has all the parameters under Source + Detector Input, including options for the excitation source you aren't using(?). This is a good candidate for consolidation into one more comprehensive tab since this is a bit confusing.
##### Plane Wave
* Static: `PlaneWaveStat([[1, 0, 0]])`, input is wave vector?
* Standard Planewave?: `planewave(pol, dir, op)`, unsure of parameters
    * MATLAB equivalent only says `exc = planewaveret(...)`
* Retarded: `PlaneWaveRet([[1,0,0]], [[0,0,1]])`
* MATLAB GUI parameters: 
    * Polarization (s and p)
    * Jones Vectors (Ex, Ey, Ez)
    * Direction (Dir_x, Dir_y, Dir_z)
    * Polarization Angle (in degrees)
##### Dipole
* Retarded and Quasistatic: `DipoleRet(pt)` (or `DipoleStat`)
* MATLAB GUI Parameters
    * Oscillation Direction (x, y, or z)
    * Dipole Position (in nm) for x, y, z
        > Notes that "Range for spatial scans in nm is defined from the Beam tab" (e-beam settings)
    * Option to replace dipole with QD 
        > Notes that "ε of QD defined by
ε(3). Disable for mapping."
##### Electron Beam
MATLAB manual advises keeping beam width smaller than mesh size
* Only Retarded? `EELSRet(p, impact, w, vel)`
* MATLAB GUI Options
    * Kinetic Energy (eV)
    * Width (nm)
    * Beam Range for X direction Farfield scan (nm)
        * Min and Max X 
            > "Only used when steps > 1"
        * Steps X
    * Beam Position (nm)
        * Position X and Y
    * Resolution for maps (what are the maps)
        * Steps XY (one parameter)
### Permitivity Options
I am just using the MATLAB GUI sections here for organization. The manual states the entires here are what is graphed in the **refractive index graphs**(structures 1,2,3)
##### Structure
The structures should all fall under `mnpbem.materials`, including anything user-defined which can use `EpsFun` as a wrapper (check this)
* MATLAB GUI Options
    > Manual: "The material and Drude models work by instantly exporting tabulated data which can then be selected 
from the dropdown list for each structure". This means that it is tables like our formatting(?)
    * ε of Environment
    * ε of Structure 1 (Main Structure)
    * ε of Structure 2
        * Corresponds to either: Second Structure (for dual structures), Substrate, or Cover Layer
    * ε of Structure 3 (Second layer of substrate)
        > For the structures, there is also option to select a defined material (including user defined) and a checkbox presumably enable that material?
##### Graphene / MoS2
I can't find an equivalent at a glance in the python docs, more info needed
##### Drude

* MATLAB GUI Options (and their mappings to what I've seen in the equation before)
    * Plasma Frequency (eV) -> $\omega_p$
    * Epsilon Infinity -> $\varepsilon_\infty$
    * Γ (eV) -> sometimes is $\gamma$
##### Lorentz
Presumably Drude-Lorentz model
* MATLAB GUI Options
    * $\omega_p$ (eV)
    * $\omega_{eg}$ (eV)
    * $\gamma_{eg}$ (eV)
### Structure Options
##### Structure
Dropdown of different structure, including user defined
* Defined in the python code with `mnpbem.geometry`
* User defined structure parameters detailed in MATLAB GUI manual


### Detector Input and Energy Range 
Unsure what category to really fit energy range under, it seems the GUI maker had the same issue
##### Detector Input
Using MATLAB GUI settings:
* Detector Angle Settings (degrees)
    * XY plane
    * XZ plane
    * Detector Range (degrees)
    * Radius (nm)
        > "For accurate CL probability set Radius = 0.5"
    * Mesh (no units? <- default is 1000)
* Map Spatial Range
    > "0 is auto"
    * XY $(\text{nm})^2$
* Colormap range for Field 
    > "0 value for both is auto" (no units provided)
    * Start Value
    * End Value

##### Energy Range (eV)
Not entirely sure what this is (spectrum range?)
* Min
* Max
* Steps
## PyMNPBEM GUI Pages Outline
Might not wireframe this out since I can refer to the existing GUI for structure
### Dashboard
###### Main hub of the GUI, where the settings are configured before running the simulation, where data is loaded and results are saved 
**See correspondance map above for detail**
#### Components
##### Current Compute Mode
* CPU only
* GPU accelerated
* multi-node MPI (mpi4py)
* FMM acceleration (need more info on this one)

(From below this, sectioned the same as the MATLAB GUI)
##### File Options
* Quick Load (unsure of function)
* Quick Save (unsure of function)
* Load
* Save As
    * Saves the current settings to a file, we could use a json?
* Save Results
    > MATLAB Manual notes results are saved automatically, but are overwritten at the end of simulation unless specifically saved with this option

My best guess at the quick save/load (without experimenting on my own) is that they save and load the settings from an internal file to quickly test and modify simulation settings.
##### Solver Selection
Choice between Retarded, Quasistatic, and Iterative 
* Stretch Goal: Define briefly the different solvers and their strengths/why you would choose each one 
##### Calculation Toggles
This is a lot of options, but they are all pretty small so it can remain like MATLAB GUI.

Includes user defined calculations
##### Excitation Source
See mapping
##### Permitivity Options
Includes user defined materials(?)
##### Refractive Index Graphs
Shows for all present materials, including user-defined (see the MATLAB GUI manual on the definitions for this)
##### Structure Options
Seems to determine the mesh shape, resolution, and position.

Includes user defined structures
##### Source and Detector Input
Should probably be separated and the source options can be merged into the Excitation Source section
##### 3D Mesh View 
MATLAB GUI has both 3D and Detector View (unsure what detector is)

Quick searching shows a few different ways it can be done (Qt3D, q3dviewer, and pyqtgraph.opengl). The opengl option seems too intensive for the short timeline (although it is the most powerful)
### Output (haven't run software yet!)
###### Include an option to view the previous results / view resulsts from a file again? (may be out of scope)
Judging from the GUI github images, it displays pretty standard MATLAB graph views, so we will likely do something similar
* Show the graph
* Allow the graph to be saved
    * Allow the raw data to be saved presumably
### User Data Uploading
##### Unsure if this needs to be it's own page, but may make uploading more straightforward (DISCUSS THIS)
### (Stretch Goal) Tutorial Page
Presumably installation instructions are bundled in the readme on github, but a quick tutorial page of how to use the GUI could be useful, but is definitely not needed for a complete and functional GUI 

 
## Temp Notes
Implementing everything using Qt for Python (pyqt), 3D rendering can be handled by PyVista (example found here https://docs.pyvista.org/examples/00-load/create_tri_surface)