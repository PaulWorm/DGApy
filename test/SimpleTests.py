import numpy as np

import dga.analytic_continuation as dga_cont

# dga_cont.MAX_ENT_DEFAULT_SETTINGS['bw'] = 1

print(dga_cont.MAX_ENT_DEFAULT_SETTINGS)


test = {
    'cut': 0.04, # only relevant for the Lorentzian mesh,
    'n_fit': 20,
    'bw': 1,
    'nwr': 1001,  # Number of frequencies on the real axis
    'wmax': 15,  # frequency range on the real axis

}

dga_cont.MAX_ENT_DEFAULT_SETTINGS.update(test)

print(dga_cont.MAX_ENT_DEFAULT_SETTINGS)
