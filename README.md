# Code for the paper "Residual deep Gaussian processes on manifolds". 

## Overview of directories
- `experiments/` contains executable files and jupyter notebooks, which can be ran to reproduce results used in Section 4 of the manuscript. Due to refactoring changes, the results will not be precisely the same; however, they should certainly stay within a margin of error due to randomness in sampling of the values reported in the paper. 

- `data/` stores the UCI and wind interpolation datasets. It also contains the files, which can be used to download this data. 

- `plots/` contains the notebooks, which can be ran to produce the files which were used for Blender renders in Figures 1, 2, 4, 5 in the manuscript. 

- `fundamental_system/` contains precomputed files needed to evaluate spherical harmonics quickly. 


### Detailed instructions 
#### `experiments/`
Each experiment reported in the paper (temporarily except for Bayesian optimisation) has three corresponding files in this directory: `run_{experiment_name}.py`, `{experiment_name}_commands.ipynb`, and `plot_{experiment_name}.ipynb`. To run an experiment, you should:
1. Run the code in `{experiment_name}_commands.ipynb` to generate an executable files for running the chosen experiment for a desired set of parameters.
2. Execute the produced `run_{experiment_name}.sh` to run the experiment and save its result.
To visualise the results as it is done in the paper, one can execute the code `plot_{experiment_name}.ipynb`.
The exception here is the geometry-aware Bayesian optimisation experiment, which can currently be performed by running the `bo.ipynb` notebook. This will be changed to match the format of the other experiments soon. There is a known issue with NaNs sometimes being encountered in this experiment. This will be fixed, but for now it is recommended to increase `raw_samples` or change the random seed. 

<br>

Additionally, the `experiments/` directory also contains refactored code for the models used in these experiments. These are:
- `kernels.py`: all kernel objects
- `models.py`: all models objects, including priors, variational posteriors, and deep models 
- `spherical_harmonics.py`: spherical harmonics and vector spherical harmonics (called there spherical harmonic fields)
- `utils.py`: utility functions


#### data
The key file in this directory is `download_uci_data.sh`. This file should be executed to download the Yacht, Concrete, Energy, Kin8mn, and Power UCI datasets.

The ERA5 dataset should be downloaded manually, moved to the data folder, and renamed to `era5.nc`. 
To download the data, please visit https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means?tab=download and select the following options in the data request form
- Product type: Reanalysis
- Variable: U-component of wind, V-component of wind
- Pressure level: 500 hPa, 800 hPa, 1000 hPa
- Year: 2010
- Month: all available
- Time: 00:00
- Format: NetCDF.


### Requirements 
Create a fresh Python 3.10.14 virtual environment (e.g. using conda or venv) and run the following command to install the necessary dependencies:
`pip install jax==0.4.30 gpjax matplotlib plotly numpy==1.26.4 pandas xlrd netCDF4 openpyxl skyfield cdsapi seaborn cartopy`
