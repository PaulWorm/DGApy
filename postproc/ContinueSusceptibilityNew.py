# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import MatsubaraFrequencies as mf
import continuation as cont
import BrillouinZone as bz
import Output as output

def get_peak_locations(data,w):
    peak = []
    for p in data:
        peak.append(w[np.argmax(p)])
    return np.array(peak)

def get_peak_index(data):
    peak = []
    for p in data:
        peak.append(np.argmax(p))
    return np.array(peak)

def get_peak(data):
    peak = []
    for p in data:
        peak.append(np.max(p))
    return np.array(peak)


def get_model(w, model_type='flat', **kwargs):
    model = None
    if (model_type == 'flat'):
        model = np.ones_like(w)
    elif (model_type == 'gauss'):
        assert 'sigma' in kwargs, 'sigma not found in input parameters.'
        model = np.exp(-w ** 2 / sigma ** 2)
    model /= np.trapz(model, w)
    return model


def get_w(w_type='lin', nw=101, **kwargs):
    if (w_type == 'lin'):
        return np.linspace(0, w_max, num=nw)
    elif (w_type == 'tan'):
        assert 'w_max' in kwargs, 'w_max not found in input parameters.'
        return w_max * np.tan(np.linspace(0., np.pi / 2.1, num=nw)) / np.tan(np.pi / 2.1)
    else:
        print('No valid w_type supplied.')
        return None


def create_output_folder(input_path='./'):
    output_path = input_path
    if(do_ladder):
        output_folder = f'ChiCont_nw_{nw}_err_{err}_sigma_{sigma}_ladder'
    else:
        output_folder = f'ChiCont_nw_{nw}_err_{err}_sigma_{sigma}_lambda'
    output_path = output.uniquify(output_path + output_folder) + '/'
    out_dir = output_path
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return out_dir

def load_data_adga(path,fname,ncut,**kwargs):
    # Load data:
    f = h5py.File(input_path + fname, 'r')
    chi_m = f['susceptibility/nonloc/magn'][0, 0, :, :, :].real
    beta = f['input/beta'][()]
    f.close()
    niw = chi_m.shape[-1] // 2
    if(ncut == -1):
        ncut = niw-1
    chi_m = chi_m[..., niw:niw + ncut]
    iw = mf.w(beta, niw)

    iw = iw[niw:niw + ncut]
    nk = chi_m.shape[:3]
    k_path = bz.KPath(nk=nk, path=bz_k_path)
    ddict = {
        'chi':chi_m,
        'iw': iw,
        'k_path':k_path

    }
    return ddict


def load_data_dga(path, fname,ncut,**kwargs):
    if(do_ladder):
        chi_data = np.load(path + 'chi_ladder.npy', allow_pickle=True).item()
    else:
        chi_data = np.load(path + 'chi_lambda.npy', allow_pickle=True).item()
    config = np.load(path + 'config.npy', allow_pickle=True).item()

    chi = chi_data['magn'].mat.real

    beta = config.sys.beta
    niw = config.box.niw_core
    nk = config.k_grid.nk
    iw = mf.w(beta, niw)
    if(ncut == -1):
        ncut = niw-1
    iw = iw[niw:niw + ncut]
    chi = chi[..., niw:niw + ncut]
    k_path = bz.KPath(nk=nk, path=bz_k_path)
    ddict = {
        'chi': chi,
        'iw': iw,
        'k_path': k_path

    }
    return ddict

def load_data_dga_old(path, fname,ncut,do_ladder=False):
    config = np.load(path + 'config.npy', allow_pickle=True).item()
    if(do_ladder):
        chi_data = np.load(path + 'chi_ladder.npy', allow_pickle=True).item()
        chi = chi_data['chi_magn_ladder'].mat.real
    else:
        chi_data = np.load(path + 'chi_lambda.npy', allow_pickle=True).item()
        chi = chi_data['chi_magn_lambda'].mat.real

    beta = config['system']['beta']
    niw = config['box_sizes']['niw_core']
    nk = config['box_sizes']['nk']
    if(ncut == -1):
        ncut = niw-1
    iw = mf.w(beta, niw)
    iw = iw[niw:niw + ncut]
    chi = chi[..., niw:niw + ncut]
    k_path = bz.KPath(nk=nk, path=bz_k_path)
    ddict = {
        'chi': chi,
        'iw': iw,
        'k_path': k_path

    }
    return ddict

def load_data(path,fname,ncut,data_type,**kwargs):
    if(data_type == 'adga'):
        return load_data_adga(path,fname,ncut,**kwargs)
    if(data_type == 'dga'):
        return load_data_dga(path,fname,ncut,**kwargs)
    if(data_type == 'dga_old'):
        return load_data_dga_old(path,fname,ncut,**kwargs)
    else:
        raise ValueError('No known data-type supplied')


# Input parameters:
input_path = '/mnt/d/Research/HubbardModel_tp-0.0_tpp0.0/2DSquare_U8_tp-0.0_tpp0.0_beta4_n0.85/scdga_kaufmann/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.0_tpp0.0/2DSquare_U8_tp-0.0_tpp0.0_beta20_n0.85/scdga_kaufmann/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.0_tpp0.0/2DSquare_U10_tp-0.0_tpp0.0_beta1_n1.00/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.0_tpp0.0/2DSquare_U16_tp-0.0_tpp0.0_beta3_n1.00/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.95/LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60/'
input_path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.90/LambdaDgaPython/LambdaDga_lc_sp_Nk6400_Nq6400_core59_invbse60_vurange500_wurange59/'
input_path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.90/LambdaDgaPython/LambdaDga_lc_sp_Nk10000_Nq10000_core59_invbse60_vurange500_wurange59_1/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.80/LambdaDga_lc_sp_Nk14400_Nq14400_core100_invbse100_vurange500_wurange100/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.80/LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange200_wurange60/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.85/LambdaDga_lc_sp_Nk10000_Nq10000_core80_invbse80_vurange500_wurange80/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.90/LambdaDga_lc_sp_Nk14400_Nq14400_core100_invbse100_vurange500_wurange100/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.925/LambdaDga_lc_sp_Nk10000_Nq10000_core80_invbse80_vurange250_wurange80/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.95/LambdaDga_lc_spch_Nk14400_Nq14400_core80_invbse80_vurange500_wurange80/'
input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta50_n0.95/LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange150_wurange60_1/'
input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta50_n0.90/LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60/'
# input_path = '/mnt/d/Research/HubbardModel_tp-0.0_tpp0.0/2DSquare_U8_tp-0.0_tpp0.0_beta10_n0.85/scdga_kaufmann/'
input_path = '/mnt/d/Research/superconductingDome/U8t_tp-0.2t_tpp0.1t_n0.90_b060/DGA/'
# input_path = '/mnt/d/Research/superconductingDome/U8t_tp-0.2t_tpp0.1t_n0.85_b060/DGA/'
# input_path = '/mnt/d/Research/superconductingDome/U8t_tp-0.2t_tpp0.1t_n0.80_b060/DGA/'
# input_path = '/mnt/d/Research/superconductingDome/U8t_tp-0.2t_tpp0.1t_n0.75_b060/DGA/'
# input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta30_n0.95/SCDGA/'
# input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta30_n0.95/SCDGA/'
# input_path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.80/LambdaDgaPython/LambdaDga_lc_sp_Nk6400_Nq6400_core59_invbse60_vurange500_wurange59/'
# input_path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.80/LambdaDgaPython/LambdaDga_lc_sp_Nk6400_Nq6400_core59_invbse60_vurange500_wurange59/'
# input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.80/scdga_invert/'
# input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.80/LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange150_wurange60/'
# input_path = '/mnt/d/Research/HubbardModel_tp-0.3_tpp0.12/2DSquare_U8_tp-0.3_tpp0.12_beta25_n0.80/LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60/'
# input_path = '/mnt/d/Research/HubbardModel_tp-0.0_tpp0.0/2DSquare_U10_tp-0.0_tpp0.0_beta25_n1.00/LambdaDga_lc_sp_Nk6400_Nq6400_core60_invbse60_vurange200_wurange60/'
# input_path = '/mnt/d/Research/HubbardModel_tp-0.0_tpp0.0/2DSquare_U8_tp-0.0_tpp0.0_beta25_n1.00/LambdaDga_lc_sp_Nk4096_Nq4096_core50_invbse50_vurange200_wurange50/'
# input_path = '/mnt/d/Research/U10_Nickelates/2DSquare_U10_tp-0.25_tpp0.12_beta75_n0.95/LambdaDga_lc_sp_Nk19600_Nq19600_core75_invbse75_vurange250_wurange75/'
# input_path = '/mnt/d/Research/U10_Nickelates/2DSquare_U10_tp-0.25_tpp0.12_beta75_n0.95/LambdaDga_lc_sp_Nk19600_Nq19600_core75_invbse75_vurange250_wurange75/'
# input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.95/LambdaDga_lc_sp_Nk10000_Nq10000_core80_invbse80_vurange500_wurange80/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U9_tp-0.25_tpp0.12_beta75_n0.95/LambdaDga_lc_sp_Nk14400_Nq14400_core100_invbse100_vurange500_wurange100/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U9_tp-0.25_tpp0.12_beta75_n0.90/LambdaDga_lc_sp_Nk14400_Nq14400_core100_invbse100_vurange500_wurange100/'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U9_tp-0.25_tpp0.12_beta75_n0.80/LambdaDga_lc_sp_Nk14400_Nq14400_core100_invbse100_vurange500_wurange100/'


input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.95/LambdaDga_lc_sp_Nk10000_Nq10000_core80_invbse80_vurange500_wurange80/'
input_path = 'D:/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.80/LambdaDga_lc_sp_Nk14400_Nq14400_core100_invbse100_vurange500_wurange100/'
# input_path = 'D:/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.875/LambdaDga_lc_sp_Nk14400_Nq14400_core100_invbse100_vurange500_wurange100/'

# input_path = 'D:/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.95/LambdaDgaPython/LambdaDga_lc_sp_Nk6400_Nq6400_core59_invbse60_vurange500_wurange59/'


fname = 'adga_sc_3.hdf5'
fname = 'adga-022.hdf5'
# fname = 'adga-019.hdf5'
# fname = 'adga-018.hdf5'
# fname = 'adga-017.hdf5'
# fname = 'adga-000.hdf5'
# fname = 'adga_sc_3.hdf5'
# fname = 'adga_sc.hdf5'
# fname = 'adga-000.hdf5'
# fname = 'adga_sc_4.hdf5'


# Which code to load data from:
input_type = 'dga'
do_ladder = False
# max-ent specifications:
err = 0.005
w_max = 10
nw = 501
use_preblur = False
bw = 0.1
sigma = 4.0
ncut = 29
nmin = 0
alpha_det = 'chi2kink'
channel = 'magn'
name = '_pipi'
# bz_k_path = 'Gamma-X-M-Gamma'
bz_k_path = 'Gamma-X-M2-Gamma'
model_type = 'flat'
w_type = 'tan'

w = get_w(w_type=w_type, w_max=w_max, nw=nw)
model = get_model(w=w, model_type=model_type, sigma=sigma)
output_folder = create_output_folder(input_path=input_path)

data = load_data(input_path,fname,ncut,input_type,do_ladder=do_ladder)
iw = data['iw']
k_path = data['k_path']
chi_m = data['chi']
err = err * np.ones_like(data['iw'])



sol_list = []
for i in range(k_path.nk_tot):
    print(i)
    chi = chi_m[k_path.ikx[i], k_path.iky[i],k_path.ikz[i],:]
    probl = cont.AnalyticContinuationProblem(im_axis=iw, re_axis=w, im_data=chi, kernel_mode='freq_bosonic')
    sol, _ = probl.solve(method='maxent_svd', optimizer='newton', alpha_determination=alpha_det,
                         model=model, stdev=err,verbose=False)
    if i % 10 == 5:
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(18,5))
        ax[0].plot(w, sol.A_opt)
        ax[1].plot(iw, chi - sol.backtransform)
        #ax[2].errorbar(x=iw, y=giw.real, yerror=err)
        #ax[2].errorbar(x=iw, y=giw.imag, yerror=err)
        ax[2].plot(sol.backtransform)
        ax[2].plot(chi, marker='x')
        plt.savefig(output_folder + f'sanity_check_{i}.png')
        plt.show()
    sol_list.append(sol)

spec_arr = np.asarray([sol.A_opt for sol in sol_list])
np.save(output_folder + 'chi.npy', spec_arr)
np.save(output_folder + 'w.npy', w)
np.save(output_folder + 'q_path_obj.npy', k_path)
np.save(output_folder + 'q_path_dict.npy', k_path.__dict__)
q_dict ={
    'kpts':k_path.kpts,
    'cind':k_path.cind,
    'k_axis':k_path.k_axis,
    'k_val': k_path.k_val,
    'path': k_path.path,
    'path_split': k_path.ckps
}
np.save(output_folder + 'q_path.npy',q_dict)

chi_w = w[:, None] * (spec_arr.T)
peak = get_peak_locations(chi_w.T,w)
fig = plt.figure(figsize=[6, 3])
plt.pcolormesh(k_path.k_axis, w, chi_w, cmap='terrain')
plt.colorbar()
plt.plot(k_path.k_axis, peak, 'o', color='cornflowerblue', mec='k', mew=0.5)
plt.ylim(0, 1.0)
plt.ylabel('Energy [eV]')
plt.xlabel('q')
plt.xticks(k_path.k_axis[k_path.cind], labels=['$\Gamma$', 'X', 'M/2', '$\Gamma$'])
plt.tight_layout()
plt.title('Paramagnon dispersion')
plt.savefig(output_folder + 'chi_m_cont_q_path.png')
plt.show()