# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Handels the logging of program progress.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import logging
import RealTime as realt
# ----------------------------------------------- CLASSES --------------------------------------------------------------

class MpiLogger():
    ''' Logs the progress of the code '''

    def __init__(self, logfile = 'dga.log', comm=None):
        self.logfile = logfile
        self.comm = comm
        self.rt = realt.real_time()

        logging.basicConfig(filename=self.logfile, encoding='utf-8', level=logging.INFO)



    @property
    def is_root(self):
        return self.comm.rank == 0



    def log_event(self, message):
        if(self.is_root):
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
            logging.info(':' + message)
        else:
            pass

    def log_cpu_time(self, task):
        if(self.is_root):
            message = self.rt.string_time(string=task)
            self.log_event(message=message)
        else:
            pass





# ---------------------------------------------- FUNCTIONS -------------------------------------------------------------