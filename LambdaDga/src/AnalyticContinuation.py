import numpy as np
import matplotlib.pyplot as plt

def extract_coefficient_imaxis(siwk=None, iw=None, N=4, order=3):
    xnew = np.linspace(-0.0, iw[N - 1] + 0.01, num=50)
    coeff_imag = np.polyfit(iw[0:N], siwk[0:N].imag, order)
    coeff_real = np.polyfit(iw[0:N], siwk[0:N].real, order)
    poly_imag = np.polyval(coeff_imag,xnew)
    poly_real = np.polyval(coeff_real,xnew)
    coeff_imag_der = np.polyder(poly_imag)
    poly_imag_der = np.polyval(coeff_imag_der,xnew)
    Z = 1. / (1. - poly_imag_der[0])

    return poly_real[0], poly_imag[0], Z

if __name__=='__main__':
    input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/LambdaDga_Nk6400_Nq6400_core32_urange32/'