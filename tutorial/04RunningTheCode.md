## DGApy:

In this tutorial we will learn how to install the code (if you did not read the README.md), how a config file looks, how to run
 the code and how the output is structured.

#### Installation:

To install the code you will have to clone the repository using:


```
git clone git@github.com:PaulWorm/DGApy.git
```

Then install [anaconda](https://www.anaconda.com/download) and set up a new python environment:

```
conda create --name dga_py python=3.9
```

Activate the environment:

```
conda activate dga_py 
```

go into the dga folder and install the code:

```
cd ./dga
. install.sh
```

This should provide you with the packages:

- dga (main routines)
- postproc (postprocessing routines)
- ana_cont (source of ana_cont)
- test_util (testing utility; not needed for production)

Furthermore, console scripts are provided:

- dga_main (run the main dga script) 
- dga_max_ent (perform analytic continuation for a completed dga run)
- dga_dc (create default config file in the current folder)
- dga_test (run the test suite)

### Config file: 

A template config file is located in 'config_templates/dga_config.yaml'. Navigate into a folder with the necessary input files:

```
cd  ./tests/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/
```

Run the dga_dc cli to create a default config file:

```
dga_dc
```

In this case the config file is already mostly suited for the input we are using. To use the minimal dataset that is included 
in this git repo change: 

```
dmft_input:
  type: "test"     # Input format; available: w2dyn, EDFermion
  input_path: "./"        # Location of the files; default: "./"
  fname_1p: 'minimal_dataset.npy' # Name of the one-particle DMFT calculation; default: 1p-data.hdf5
  fname_2p: 'g4iw_sym.hdf5' # not used for this input type
```

Now the code can be run by executing:

```
dga_main 
```

This will create an output folder that starts with LDGA_{...}. Within that folder several plots and the output files are 
dumped. The output should (for the most part) be self-explanatory (hopefully).

To run the code with mpi support use: 

```
mpiexec -np <number of tasks> dga_main 
```

where <number of tasks> is the number of mpi tasks you want to use.
In the next tutorial (PostProcessing.ipynb) we will take a look at the output files. 
