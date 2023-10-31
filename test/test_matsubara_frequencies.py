import numpy as np
import dga.matsubara_frequencies as mf



def test_vn():
    vn = mf.vn(10)
    assert np.equal(vn,np.arange(-10,10)).all()

    vn = mf.vn(10,pos=True)
    assert np.equal(vn,np.arange(0,10)).all()

    v = mf.vn(10.,10,pos=True)
    assert np.equal(v,np.pi/10. * (2*mf.vn(10,pos=True)+1)).all()

    print('---------------')
    print('Passed vn test ')
    print('---------------')

if __name__ == '__main__':
    test_vn()
