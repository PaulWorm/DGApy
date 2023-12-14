# Frequency ranges and asymptotics

## Frequency grids

Niv should always >= niw. Otherwise, weirds kinks appear at the boundary. Furthermore, niv_core + niv_shell should be larger
than niw_core. Otherwise, the asymptotic behavior of the bubble displays artifacts at the largest fermionic Matsubara
frequencies. This could (in principle) be avoided by using 'center' for the frequency summation convention. However, this
would then not be consistent with the current frequency layout of the four-point functions.

We adopt the following notation for the different boxes:

- core: inner part with Gamma = Gamma_DMFT
- shell: outer part with Gamma = U
- asympt: asymptotic part with Gamma = 0 and chi_r = chi_0 and f_r = u_r
- tilde: corrected quantity with Gamma = 0 and chi = chi_0
  

Convergence with number of frequencies is generally quite slow.

## Momentum Grids

Using non-uniform grids for the k-mesh does not yield reasonable results. Similarly, using different meshes for k and q
is not advised.

# mf.vn() and mv.wn()

The functions for mf.vn() and mf.wn() are overloaded. The distinction between beta and n is done via type checking (float vs.
int). This can lead to issues if beta is defined as an integer.

# Heuristics:

Emperically, if t=1, niv_core = niw_core = 2 \beta + 10 seems to be a good tradeoff between frequency convergence, qmc error and
computational cost for sampling G2.

# Eliashberg and Superconductivity

## Heuristics

It seems that using 'spch' for the lambda correction yields larger Tc's compared to using 'sp' only. This is presumably
because 'sp' tends to dampen and screen the magnetic susceptibility more compared to using 'spch'.

# Performance:

The following parameter set took 4 minutes on the VSC5 run on 5 devel nodes with 96 cores per node (without Eliashberg):

nk = nq = 100
niw_core = niv_core = 40
niv_shell = 200
symmetry = two_dimensional_square

For the same dataset the complet MaxEnt of Siwk, Giwk, Chi-dens and Chi-magn took also about 4 minutes.

nwr = 501
beta = 12.5

Using also the eliashberg part of the code, it takes 8 minutes to complete the same dataset, but with 80x80x1 k-points instead.

# Memory limit:

Currently on the VSC5 the memory limit is reached for core = 80 / shell = 500 and nk = nq = 100x100x1 on the normal 512 GiB
nodes.

## Profiling

Use mprof run -M mpiexec -np <#processes> python <script.py> to profile mpi memory consumption and mprof plot -o <file.png> to
plot the results of the profiling.

# For Developers:

## Code style and structure:

This code follows the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide.
For linting, use
[pylint](https://pypi.org/project/pylint/#:~:text=Pylint%20is%20a%20static%20codethe%20code%20could%20be%20refactored.)
with the [Google style guide](https://google.github.io/styleguide/pyguide.html).

### Strings:

Within this code ' ' is used for strings.

### Line length:

Contrary to most python code the maximum line length is 130 characters.

### Redundant key word:

redundant-keyword-arg checking is disabled in the .pylintrc file. This is because otherwise the multimethod of mf.vn() and mf.
wn() are not recognized correctly.

## Technical details:

### vrg:

vrg is defined by two equations:

vrg(w,v) = 1/gchi0(w,v) * sum_vp gchi_aux(w,v,vp)

while 

vrg(w,vp) = 1/gchi0(w,vp) * sum_v gchi_aux(w,v,vp).

At first glance one would assume that this is the same. However, because of monta carlo error these may differ. To avoid this 
the method LocalFourPoint.symmetrize_v_vp() exists. 



