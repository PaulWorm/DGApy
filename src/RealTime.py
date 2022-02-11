import time


class real_time():
    ''' simple class to keep track of real time '''

    def __init__(self):
        self._ts = time.time()
        self._tl = time.time()
        self._tm = []
        self._tm.append(0)

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
        print(string + 'took {} seconds'.format(self._tm[-1]))

    def string_time(self, string=''):
        self.measure_time()
        return string + 'took {} seconds'.format(self._tm[-1])

    def write_time_to_file(self, string='',rank=0):
        if(rank==0):
            string_time = self.string_time(string=string)
            self.open_file()
            self.file.write(string_time + '\n')
            self.close_file()
        else:
            pass