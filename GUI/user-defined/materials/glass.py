from mnpbem.materials.eps_const import EpsConst
from typing import Callable # typing isn't strictly necessary (handled by the internal state class)

# example of how to create your own material dielectric function, as long as it returns some dielectric function defined by:
#       EpsConst, EpsTable (can also just leave a .dat file), EpsDrude, EpsFun, EpsNonlocal
#  Any more specific function can be wrapped in EpsFun as documented in the api reference (provided it has the same arguments and return values) 
#
#  IMPORTANT NOTES: imported materials CANNOT share the same name, regardless of file type (ex. gold.dat and gold.py will conflict and only one will be loaded)
#                   the function that returns the dielectric function MUST be named generate_eps_func for the GUI to import it properly
def generate_eps_func() -> Callable[[float], tuple[complex, float]]:
    return EpsConst(1.5 ** 2)