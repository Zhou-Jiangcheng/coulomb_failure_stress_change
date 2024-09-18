# Introduction

This Python package serves as the frontend for building Green's function libraries for static and dynamic stress, and for calculating Coulomb failure stress. The backend uses Wang Rongjiang's stress calculation programs, including EDGRN/EDCMP and QSSP (Wang, 1999; Wang, 2003; Wang et al., 2017). The code for building the static stress Green's function library uses the `multiprocessing` library (single-node multi-process). The code for building the dynamic stress Green's function library uses the MPI (multi-node) parallel mode.

Wang, Rongjiang. (1999). A simple orthonormalization method for stable and efficient computation of Green’s functions.  *Bulletin of the Seismological Society of America* ,  *89* (3), 733–741. [https://doi.org/10.1785/BSSA0890030733](https://doi.org/10.1785/BSSA0890030733)

  Wang, R. (2003). Computation of deformation induced by earthquakes in a multi-layered elastic crust—FORTRAN programs EDGRN/EDCMP.  *Computers & Geosciences* ,  *29* (2), 195–207. [https://doi.org/10.1016/S0098-3004(02)00111-5](https://doi.org/10.1016/S0098-3004(02)00111-5)

  Wang, Rongjiang, Heimann, S., Zhang, Y., Wang, H., & Dahm, T. (2017). Complete synthetic seismograms based on a spherical self-gravitating Earth model with an atmosphere–ocean–mantle–core structure.  *Geophysical Journal International* ,  *210* (3), 1739–1764.

# Installation

1. Install the requirments. (Debian 12, Python 3.11)

```
sudo apt install gfortran
conda install numpy scipy pandas mpi4py tqdm -c conda-forge
```

Install `pygrnwang`. https://github.com/Zhou-Jiangcheng/pygrnwang

2. Compile the corresponding Fortran source files.
3. Copy the exec files to `output` folder.
