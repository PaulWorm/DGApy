import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI as mpi
import Config as conf
import Output as output
import Hk as hamk

# Define MPI communicator:
comm = mpi.COMM_WORLD

# Create the DGA Config object:
# dga_conf = conf.DgaConfig(BoxSizes=box_sizes, Options=options, SystemParameter=sys_param, Names=names,
#                           ek_funk=hamk.ek_3d)

base = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta50_n0.95/'
path = base + 'LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60/'
dga_conf_dict = np.load(path + 'config.npy', allow_pickle=True).item()
dmft_sde = np.load(path + 'dmft_sde.npy', allow_pickle=True).item()
dga_sde = np.load(path + 'dga_sde.npy', allow_pickle=True).item()
sloc = dga_conf_dict['dmft1p']['sloc']
sigma = dga_sde['sigma_nc']
beta = dga_conf_dict['system']['beta']

# Create the DGA Config object:
dga_conf = conf.DgaConfig(BoxSizes=dga_conf_dict['box_sizes'], Options=dga_conf_dict['options'], SystemParameter=dga_conf_dict['system']
                          , Names=dga_conf_dict['names'],ek_funk=hamk.ek_3d)
# ------------------------------------------ ANALYTIC CONTINUATION SET-UP ----------------------------------------------
t = 1
nwr = 501
me_conf = conf.MaxEntConfig(t=t, beta=beta, mesh_type='linear')
me_conf.nwr = nwr
n_fit = me_conf.get_n_fit_opt(n_fit_min=dga_conf.box.niv_core, n_fit_max=dga_conf.box.niv_urange)
bw = me_conf.get_bw_opt()

#%%
# Broadcast bw_opt_dga
comm.Barrier()
if comm.rank == 0:
    bw_opt_dga = np.empty((1,))
    bw_opt_vrg_re = np.empty((1,))
else:
    bw_opt_dga = np.empty((1,))
    bw_opt_vrg_re = np.empty((1,))

if dga_conf.opt.do_max_ent_loc:
    if comm.rank == 0:
        bw_range_loc = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]) * bw
        output.max_ent_loc_bw_range(dga_conf=dga_conf, me_conf=me_conf, bw_range=bw_range_loc, sigma=sloc,
                                    n_fit=n_fit, adjust_mu=True, name='dmft')
        bw_opt_dga[0] = output.max_ent_loc_bw_range(dga_conf=dga_conf, me_conf=me_conf, bw_range=bw_range_loc,
                                                    sigma=sigma,
                                                    n_fit=n_fit, adjust_mu=True, name='dga')
    comm.Bcast(bw_opt_dga, root=0)