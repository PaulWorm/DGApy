# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Contains output related routines.


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys,os

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def uniquify(path=None):
    '''

    path: path to be checked for uniqueness
    return: updated unique path
    '''
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path

if __name__=="__main__":
    path = 'C:/Users/pworm/Research/FiniteLayerNickelates/N=5/GGA+U/dx2-y2_modified/n0.82/Continuation'
    path_unique = uniquify(path)
    #os.mkdir(path_unique)