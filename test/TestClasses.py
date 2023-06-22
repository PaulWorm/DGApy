''' Simple script to test options for classes'''

class MyClass():

    att1 = 10
    att2 = 20
    att3 = None

    def __init__(self, att3=None):
        self.att3 = att3

if __name__ == '__main__':

    my_class = MyClass(att3=15)
    print(f'{my_class.att1}')
    my_class2 = MyClass(att3=20)
    my_class2.att1 = 30
    print(f'{my_class.att1}')
    print(f'{my_class2.att1}')

