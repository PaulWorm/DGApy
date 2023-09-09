# Frequency ranges and asymptotics

Niv should always >= niw. Otherwise, weirds kinks appear at the boundary.

I switched back to a non-shifted grid for chi0 

We adopt the following notation for the different boxes: 

-) core: inner part with Gamma = Gamma_DMFT 
-) shell: outer part with Gamma = U
-) tilde: corrected quantity with Gamma = 0 and chi = chi_0

Convergence with number of frequencies is generally quite slow.

# Heuristics: 

Emperically, if t=1,  niv_core = niw_core = 2 \beta + 10 seems to be a good tradeoff between frequency convergence, qmc error and 
computational cost for sampling G2. 

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