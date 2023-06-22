# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Handels the logging of program progress.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import os
import time
import datetime
import psutil
import resource
# ----------------------------------------------- CLASSES --------------------------------------------------------------

class real_time():
    ''' simple class to keep track of real time '''

    def __init__(self):
        self._ts = time.time()
        self._tl = time.time()
        self._tm = []
        self._tm.append(0.)

    def create_file(self, fname='cpu_time.txt'):
        self.fname = fname
        self.file = open(self.fname,'a+')
        self.file.close()

    def open_file(self):
        self.file = open(self.fname,'a')

    def close_file(self):
        self.file.close()

    def measure_time(self):
        self._tm.append(time.time() - self._tl)
        self._tl = time.time()

    def print_time(self, string=''):
        self.measure_time()
        print(string + ' took {} seconds'.format(self._tm[-1]))

    def string_time(self, string=''):
        self.measure_time()
        return string + ' took {} seconds'.format(self._tm[-1])

    def tot_time(self):
        return str(datetime.timedelta(seconds=round(time.time() - self._ts)))

    def task_time(self):
        self.measure_time()
        return str(datetime.timedelta(seconds=round(self._tm[-1])))

    def write_time_to_file(self, string='',rank=0):
        if(rank==0):
            string_time = self.string_time(string=string)
            self.open_file()
            self.file.write(string_time + '\n')
            self.close_file()
        else:
            pass

class MpiLogger():
    ''' Logs the progress of the code '''

    def __init__(self, logfile = 'dga.log', comm=None, output_path='./'):
        self.logfile = logfile
        self.comm = comm
        self.out_dir = output_path
        self.rt = real_time()
        # create file
        if(self.is_root):
            f = open(self.logfile, 'w')
            f.write(' Local-T : Total-T : Ctask-T :  Comment \n')
            f.close()

    @property
    def is_root(self):
        return self.comm.rank == 0


    def log_memory_usage(self):
        if(self.is_root):
            f = open(self.logfile,'a')
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1e-6
            message = f'Estimated RAM used by root (GB): {mem}'
            f.write(self.local_time() + ' : ' + message + '\n')
            f.close()
        # elif(self.comm.size > 1 and self.comm.rank == 1):
        #     f = open(self.logfile,'a')
        #     mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1e-6
        #     message = f'Estimated RAM used by other ranks (GB): {mem}'
        #     f.write(self.local_time() + ' : ' + message + '\n')
        #     f.close()
        else:
            pass

    def log_event(self, message):
        if(self.is_root):
            f = open(self.logfile,'a')
            f.write(self.local_time() + ' : ' + self.rt.tot_time() + ' : ' + self.rt.task_time() + ' : ' + message + '\n')
            f.close()
        else:
            pass

    def log_message(self, message):
        if(self.is_root):
            f = open(self.logfile,'a')
            f.write(self.local_time() + ' : ' + message + '\n')
            f.close()
        else:
            pass

    def log_cpu_time(self, task):
        if(self.is_root):
            self.log_event(message=task)
        else:
            pass

    def local_time(self):
        now = datetime.datetime.now()
        return now.strftime("%H:%M:%S")






# ---------------------------------------------- FUNCTIONS -------------------------------------------------------------

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    logger = MpiLogger(logfile='test.log', comm=comm)

    logger.log_cpu_time(task='test logger')
    logger.log_cpu_time(task='test logger 2')
