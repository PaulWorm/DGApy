import numpy as np
import matplotlib.pyplot as plt

def geometric_summation(mat0,factor, n_sum=1):
    result = mat0
    term = mat0
    for i in range(n_sum):
        term = -(term @ factor @ mat0)
        result = result + term

    return result

def geometric_inversion(mat0,factor):
    return np.linalg.inv(np.linalg.inv(mat0) + factor)

def geometric_inversion_v2(mat0,factor):
    return mat0 @ np.linalg.inv(np.eye(np.shape(mat0)[0]) + factor @ mat0)


if __name__ == '__main__':
    input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/LambdaDga_lc_sp_Nk576_Nq576_core40_invbse40_vurange40_wurange40/'

    config = np.load(input_path+'config.npy', allow_pickle=True).item()
    dmft1p = np.load(input_path+'dmft1p.npy', allow_pickle=True).item()
    gamma = np.load(input_path+'gamma_dmft.npy', allow_pickle=True).item()
    import FourPoint as fp
    chi0 = fp.BubbleGenerator(giw=dmft1p['gloc'], beta=config.sys.beta, niv_sum=config.box.niv_core, iw=(0,))
    niv_core = config.box.niv_core
    chi0.set_gchi0()
    u = config.sys.u
    beta = config.sys.beta
    mat0 = np.diag(chi0.gchi0[0,:])

    factor = (gamma['dens'].mat[config.box.niw_core,:,:] - u/beta**2)
    gsum_inv = geometric_inversion(mat0,factor)[niv_core,niv_core]
    gsum_inv_v2 = geometric_inversion_v2(mat0,factor)[niv_core,niv_core]
    gsum_sum = []
    gsum_sum2 = []
    n_arr = np.arange(5)
    for i in n_arr:
        gsum_sum.append(geometric_summation(mat0,factor,n_sum=i)[niv_core,niv_core])
    gsum_sum = np.array(gsum_sum)
    gsum_sum2 = np.array(gsum_sum2)


    plt.plot(n_arr,gsum_inv.real-gsum_sum.real, '-o')
    plt.xlim(1,None)
    #plt.ylim(-0.5,0.5)
    plt.show()
    #
    # mat0 = np.random.rand(2,2)
    # factor = np.random.rand(2,2)
    # gsum_inv = geometric_inversion(mat0, factor)
    # gsum_sum = geometric_summation(mat0, factor, n_sum=1000)
    #
    # print(gsum_inv-gsum_sum)
