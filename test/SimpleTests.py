import numpy as np


list = ['1', '2', '3']

value = '1'

assert value in list, "value not in list"


print(f"{list}")

class TestClass():

    def __init__(self,a,b,c):

        self._a = a
        self._b = b
        self._c = c

    @property
    def a(self):
        return self._a

    def method(self):
        return self.a+self._b+self._c

tc = TestClass(1,2,3)

dict = tc.__dict__

# tc2 = TestClass(**dict)

print(dict)


