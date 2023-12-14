
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://gitlab.com/PWorm/dga/-/graphs/main)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit/)

[//]: # ([![coverage]&#40;./coverage.svg&#41;]&#40;&#41;)
# DGApy

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [About](#about)
- [Testing](#testing)


## Description
DGA is a python package for calculations using the lambda corrected dynamical vertex correction. 

## Installation
Run the script
```
. install.sh
```
in the root directory. This provides the python package dga and several command line interfaces. 

## Usage

### Python package

To use the dga python package, import it via

```
import dga.<submodule name>
```

The package is split into several modules. For a more detailed tutorial see the tutorial folder.


### Command line interfaces

#### dga_dc

Create the default config file in the current folder.

#### dga_main

To run a dga calculation perform the following steps:

- Prepare a config file
- Prepare the DMFT input files
- Run the dga_main cli

prepare a config file, like the template in config_templates. Additionally, two files containing 
the results of a converged DMFT calculation must be prepared as starting point. Currently, loading input is primarily supported 
for [w2dynamics](https://github.com/w2dynamics/w2dynamics). In case you want to use another impurity solver for the DMFT input, 
feel free to contact me via [e-mail](mailto:pworm42@gmail.com) or open an issue.

#### dga_test

Run the test suite. This will run the unit tests and linting. 

## References

This package has been developed by [Paul Worm](https://www.linkedin.com/in/pworm/). 
In case you use the code, please consider citing the following works: 

My [thesis](https://repositum.tuwien.at/handle/20.500.12708/176739) which covers the theoretical background. 
[This paper](https://iopscience.iop.org/article/10.1088/2515-7639/ac7e6d), on which the asymptotic correctons are based. 
And finally the original [paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.75.045118), where the dynamical vertex approximation was first derived. 

If you have any questions feel free to reach out to me via [e-mail](mailto:pworm42@gmail.com).

For analytic continuation the [ana_cont](https://github.com/josefkaufmann/ana_cont) package is used and directly integrated as 
source. In case you use the analytic continuation, please cite this 
[paper](https://www.sciencedirect.com/science/article/pii/S0010465522002387).

Routines for finding the chemical potential are based on those found in the 
[w2dynamics](https://github.com/w2dynamics/w2dynamics) code. 

A bibtex file (dga_bib.md) containing all the references is included in the root directory.


## Publications using DGApy

- S. Di Cataldo, P. Worm <em> et al.</em> 
  - [Unconventional superconductivity without doping: infinite-layer nickelates under 
    pressure](https://arxiv.org/abs/2311.06195)

## Testing: 

#### Unit testing

To run the unit tests, run 

```
. dga_test.sh 
```

in the root directory. This will run unit tests and linting which are also included in the gitlab CI.

#### End-to-end testing 
Several tests use datasets generated with w2dynamics or EDFermion. If you want to run tests that use those datasets, please 
contact me, so I can share them with you.


