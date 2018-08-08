import pathos.pools as pp

class someClass(object):
    def __init__(self):
        pass

    def f(self, x):
        return x*x

    def go(self):
        pool = pp.ProcessPool(4)
        print pool.map(self.f, range(10))

if __name__ == '__main__':

    obj = someClass()
    obj.go()
