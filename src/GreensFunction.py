import numpy as np
import matplotlib.pyplot as plt
import BrillouinZone as bz
import MatsubaraFrequencies as mf
import SelfEnergy as se

class GreensFunction():
    '''Object to build the Green's function from hr and sigma'''
    def __init__(self, beta,sigma,k_grid,hr):
        self.beta = beta
        self.k_grid = k_grid
        self.hr = hr
        self.sigma = sigma


