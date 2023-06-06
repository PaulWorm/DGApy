# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Handels the logging of program progress.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import RealTime as realt
from datetime import datetime
# ----------------------------------------------- CLASSES --------------------------------------------------------------

class MpiLogger():
    ''' Logs the progress of the code '''

    def __init__(self, logfile = 'dga.log', comm=None, output_path='./'):
        self.logfile = logfile
        self.comm = comm
        self.out_dir = output_path
        self.rt = realt.real_time()
        # create file
        if(self.is_root):
            f = open(self.logfile, 'w')
            f.write(' Local-T : Total-T : Ctask-T :  Comment \n')
            f.close()

    @property
    def is_root(self):
        return self.comm.rank == 0



    def log_event(self, message):
        if(self.is_root):
            f = open(self.logfile,'a')
            f.write(self.local_time() + ' : ' + self.rt.tot_time() + ' : ' + self.rt.task_time() + ' : ' + message + '\n')
            f.close()
        else:
            pass

    def log_cpu_time(self, task):
        if(self.is_root):
            self.log_event(message=task)
        else:
            pass

    def local_time(self):
        now = datetime.now()
        return now.strftime("%H:%M:%S")






# ---------------------------------------------- FUNCTIONS -------------------------------------------------------------

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    logger = MpiLogger(logfile='test.log', comm=comm)

    logger.log_cpu_time(task='test logger')
    logger.log_cpu_time(task='test logger 2')
