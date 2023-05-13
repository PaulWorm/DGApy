'''
    Specify the style of matplotlib plots here
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler


FIGSIZE = (7, 3)
ALPHA = 0.7
BASECOLOR = 'cornflowerblue'
DEFAULTCOLOURS = ['cornflowerblue','firebrick','seagreen','goldenrod','indigo']
DIVERGINGCOLOURS = ['navy','indigo','cornflowerblue','goldenrod','firebrick']
DEFAULTMARKERS = ['o','^','h','d','<']
DEFAULTMARKERSIZE = [6,5,4,3,2]
DEFAULTALPHA = [1,0.9,0.8,0.7,0.6]
mpl.rcParams["savefig.dpi"] = 500
mpl.rcParams["figure.dpi"] = 500

mpl.rcParams['axes.prop_cycle'] = cycler.cycler(
    color=DEFAULTCOLOURS,marker=DEFAULTMARKERS,markersize=DEFAULTMARKERSIZE,alpha=DEFAULTALPHA)

mpl.rcParams["lines.markeredgecolor"] = 'k'
# mpl.rcParams["lines.alpha"] = ALPHA
mpl.rcParams["lines.markersize"] = 3
# mpl.rcParams["lines.prop_cycle"] = cycler.cycler(marker=DEFAULTMARKERS)