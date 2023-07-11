## Description
DGA is a python package for calculations using the lambda corrected dynamical vertex correction. 

## Installation
Run "pip install -e ." in the root directory. This will provide the package dga and the console script dga_main.


## Usage
To run a dga calculation prepare a config file, like the template in config_templates. Furthermore, two files containing 
the results of a converged DMFT calculation must be prepared as starting point. 

## About

This package has been developed by Paul Worm. 
In case you use the code, please consider citing the following works: 

My [thesis](https://repositum.tuwien.at/handle/20.500.12708/176739) which covers the theoretical background. 
[This paper](https://iopscience.iop.org/article/10.1088/2515-7639/ac7e6d), on which the asymptotic correctons are based. 
And finally the original [paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.75.045118), where the dynamical vertex approximation was first derived. 

If you have any questions feel free to reach out to me via [e-mail](mailto:pworm42@gmail.com).

## Notes on usage: 
niw_core should NOT be larger than niv_core. Otherwise, finite size artifacts show up in the bare bubble contribution.  

## Roadmap
ToDo

## Profiling
Use mprof run -M mpiexec -np <#processes> python <script.py> to profile mpi memory consumption and mprof plot -o <file.png> to plot the results of the profiling.
