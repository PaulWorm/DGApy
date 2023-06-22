# Naming conventions:

niw/niv : number of frequencies
wn/vn : array of frequency indices
iwn/ivn : single frequency index
w/v : array of frequencies (2n + 1) * pi/beta
iw/iv : array of imaginary frequencies 1j * (2n + 1) * pi/beta

# Frequency ranges and asymptotics

Niv should always >= niw. Otherwise, weirds kinks appear at the boundary.

I switched back to a non-shifted grid for chi0 

We adopt the following notation for the different boxes: 

-) core: inner part 
-) shell: outer part 
-) tilde: corrected quantity

Convergence with number of asymptotic frequencies is quite slow. 
