# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Handels the logging of program progress.


'''
    Module for logging the progress of the program
'''

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import time
import datetime


# ----------------------------------------------- CLASSES --------------------------------------------------------------


class RealTime():
    ''' simple class to keep track of real time '''

    def __init__(self):
        self._ts = time.time()
        self._tl = time.time()
        self._tm = []
        self._tm.append(0.)

    def create_file(self, fname='cpu_time.txt'):
        self.fname = fname
        self.file = open(self.fname, 'a+', encoding='utf-8')
        self.file.close()

    def open_file(self):
        self.file = open(self.fname, 'a', encoding='utf-8')

    def close_file(self):
        self.file.close()

    def measure_time(self):
        self._tm.append(time.time() - self._tl)
        self._tl = time.time()

    def print_time(self, string=''):
        self.measure_time()
        print(string + f' took {self._tm[-1]} seconds')

    def string_time(self, string=''):
        self.measure_time()
        return string + f' took {self._tm[-1]} seconds'

    def tot_time(self):
        return str(datetime.timedelta(seconds=round(time.time() - self._ts)))

    def task_time(self):
        self.measure_time()
        return str(datetime.timedelta(seconds=round(self._tm[-1])))

    def write_time_to_file(self, string='', rank=0):
        if rank == 0:
            string_time = self.string_time(string=string)
            self.open_file()
            self.file.write(string_time + '\n')
            self.close_file()
        else:
            pass


class MpiLogger():
    ''' Logs the progress of the code '''

    def __init__(self, logfile='dga.log', comm=None, output_path='./'):
        self.logfile = logfile
        self.comm = comm
        self.out_dir = output_path
        self.rt = RealTime()
        # create file
        if self.is_root:
            with open(self.logfile, 'w', encoding='utf-8') as file:
                file.write(' Local-T : Total-T : Ctask-T :  Comment \n')

    @property
    def is_root(self):
        return self.comm.rank == 0

    def log_memory_usage(self, obj_name, obj, n_exists=1):
        if self.is_root:
            with open(self.logfile, 'a', encoding='utf-8') as file:
                mem = obj.size * obj.itemsize * 1e-9 * n_exists
                message = f'{obj_name} uses (GB): {mem:.6f}'
                file.write(f'{self.local_time()} : {message}\n')
        else:
            pass

    def log_event(self, message):
        if self.is_root:
            with open(self.logfile, 'a', encoding='utf-8') as file:
                file.write(self.local_time() + ' : ' + self.rt.tot_time() + ' : ' + self.rt.task_time() + ' : ' + message + '\n')
        else:
            pass

    def log_message(self, message):
        if self.is_root:
            with open(self.logfile, 'a', encoding='utf-8') as file:
                file.write(self.local_time() + ' : ' + message + '\n')
        else:
            pass

    def log_cpu_time(self, task):
        if self.is_root:
            self.log_event(message=task)
        else:
            pass

    def local_time(self):
        now = datetime.datetime.now()
        return now.strftime('%H:%M:%S')


if __name__ == '__main__':
    pass
