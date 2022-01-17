import time


class real_time():
    ''' simple class to keep track of real time '''

    def __init__(self):
        self._ts = time.time()
        self._tl = time.time()
        self._tm = []
        self._tm.append(0)

    def measure_time(self):
        self._tm.append(time.time() - self._tl)
        self._tl = time.time()

    def print_time(self, string=''):
        self.measure_time()
        print(string + 'took {} seconds'.format(self._tm[-1]))

    def string_time(self, string=''):
        self.measure_time()
        return string + 'took {} seconds'.format(self._tm[-1])
