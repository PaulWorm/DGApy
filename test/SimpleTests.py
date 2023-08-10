import numpy as np


bandshift_dmft = 0.05
gamma_dmft = 0.01
Z_dmft = 0.8
np.savetxt('./TestPlots/' + 'Siw_DMFT_polyfit.txt', np.array([[bandshift_dmft, ], [gamma_dmft, ], [Z_dmft, ]]).T,
           header='bandshift gamma Z', fmt='%.6f')