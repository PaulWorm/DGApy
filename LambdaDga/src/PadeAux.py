# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Pade solver from (https://github.com/JohanSchott/Pade_approximants/blob/master/python/Pade.ipynb)


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import sys
import matplotlib.pylab as plt


# -------------------------------------------- PADE KAUFMANN ----------------------------------------------------------
# Josefs pade solver is basically Thiele's recersive solver, but rather slow.

class OptimizationResult(object):
    """Object for holding the result of an optimization.

    This class has no methods except the constructor,
    it is thus essentially a collection of output numbers.

    All member variables have None as default value, different solvers
    override different variables. A_opt is always set, since it
    is the main result of analytic continuation.
    """

    def __init__(self,
                 u_opt=None,
                 A_opt=None,
                 chi2=None,
                 backtransform=None,
                 entropy=None,
                 n_good=None,
                 probability=None,
                 alpha=None,
                 convergence=None,
                 trace=None,
                 Q=None,
                 norm=None,
                 blur_width=None,
                 numerator=None,
                 denominator=None,
                 numerator_function=None,
                 denominator_function=None,
                 check=None,
                 ivcheck=None,
                 g_ret=None):
        self.u_opt = u_opt
        self.A_opt = A_opt
        self.chi2 = chi2
        self.backtransform = backtransform
        self.entropy = entropy
        self.n_good = n_good
        self.probability = probability
        self.alpha = alpha
        self.convergence = convergence
        self.trace = trace
        self.Q = Q
        self.norm = norm
        self.blur_width = blur_width
        self.numerator=numerator
        self.denominator=denominator
        self.numerator_function=numerator_function
        self.denominator_function=denominator_function
        self.check = check
        self.ivcheck = ivcheck
        self.g_ret = g_ret

class PadeSolver():
    """Pade solver"""
    def __init__(self, im_axis, re_axis, im_data):
        """
        Parameters
        ----------
        im_axis : numpy.ndarray
                  Matsubara frequencies which are used for the continuation
        re_axis : numpy.ndarray
                  Real-frequency points at which the Pade interpolant is evaluated
        im_data : Green's function values at the given points `im_axis`
        """
        self.im_axis = im_axis
        self.re_axis = re_axis
        self.im_data = im_data

        # Compute the Pade coefficients
        self.a = compute_coefficients(1j * self.im_axis, self.im_data)

    def check(self, im_axis_fine=None):
        """Sanity check for Pade approximant

        Evaluate the Pade approximant on the imaginary axis,
        however not only at Matsubara frequencies, but on a
        dense grid. If the approximant is good, then this
        should yield a smooth interpolating curve on the Matsubara
        axis. On the other hand, little 'spikes' are a hint
        for pole-zero pairs close to the imaginary axis. This
        is usually a result of noise and a different choice of
        Matsubara frequencies may help.

        Parameters
        ----------
        im_axis_fine : numpy.ndarray, default=None
                  Imaginary-axis points where the approximant is
                  evaluated for checking. If not specified,
                  an array of length 500 is generated, which goes
                  up to twice the maximum frequency that was used
                  for constructing the approximant.

        Returns
        -------
        numpy.ndarray
                  Values of the Pade approximant at the points
                  of `im_axis_fine`.
        """

        if im_axis_fine is None:
            self.ivcheck = np.linspace(0, 2 * np.max(self.im_axis), num=500)
        else:
            self.ivcheck = im_axis_fine
        check = C(self.ivcheck * 1j, 1j * self.im_axis,
                            self.im_data, self.a)
        return check

    def solve(self, show_plot=False):
        """Compute the Pade approximation on the real axis.

        The main numerically heavy computation is done in the Cython module
        `pade.pyx`. Here we just call the functions.
        In the Pade method, the numerator and denominator approximants
        are generated separately, and then the division is done.
        As an additional feature we add the callable `numerator_function`
        and `denominator_function` to the OptimizationResult object.
        """

        def numerator_function(z):
            return A(z, self.im_axis.shape[0],
                           1j * self.im_axis, self.im_data, self.a)

        def denominator_function(z):
            return B(z, self.im_axis.shape[0],
                           1j * self.im_axis, self.im_data, self.a)

        numerator = A(self.re_axis, self.im_axis.shape[0],
                           1j * self.im_axis, self.im_data, self.a)
        denominator = B(self.re_axis, self.im_axis.shape[0],
                           1j * self.im_axis, self.im_data, self.a)

        result = numerator / denominator

        result_dict = {}
        result_dict.update({"numerator": numerator,
                            "denominator": denominator,
                            "numerator_function": numerator_function,
                            "denominator_function": denominator_function,
                            "check": self.check(),
                            "ivcheck": self.ivcheck,
                            "A_opt": -result.imag / np.pi,
                            "g_ret": result})

        return OptimizationResult(**result_dict)

def compute_coefficients(zi,ui):
  n=len(zi)
  g=np.zeros((n,n),dtype=np.complex)
  g[:,0]=ui
  for i in range(n):
    for j in range(1,i+1):
      g[i,j]=(g[j-1,j-1]-g[i,j-1])/((zi[i]-zi[j-1])*g[i,j-1])
  return np.diag(g)


def a(zi,ui):
  n=len(zi)
  alist=np.zeros((n-1),dtype=np.double)
  for i in range(1,n):
    alist[i-1]=g(i,i,zi,ui)
  return alist

def g(p,i,zi,ui):
  if p==1:
    return ui[i-1]
  elif p>1:
    return (g(p-1,p-1,zi,ui)-g(p-1,i,zi,ui))/((zi[i-1]-zi[p-2])*g(p-1,i,zi,ui))

def A(zr,n,zi,ui,a):
  if n==0:
    return 0.
  elif n==1:
    return a[0]
  else:
    return A(zr,n-1,zi,ui,a) + (zr-zi[n-2])*a[n-1]*A(zr,n-2,zi,ui,a)

def B(zr,n,zi,ui,a):
  if n==0:
    return 1.
  elif n==1:
    return 1.
  else:
    return B(zr,n-1,zi,ui,a) + (zr-zi[n-2])*a[n-1]*B(zr,n-2,zi,ui,a)

def C(zr,zi,ui,a):
  n=len(zi)
  return A(zr,n,zi,ui,a)/B(zr,n,zi,ui,a)

# -------------------------------------------- PADE JOHANSCHOTT ----------------------------------------------------------


def padeThiele(z, f, zut):
    '''
    Thiele's reciprocal difference method.

    Input variables:
    z    - complex. points in the complex plane.
    f    - complex. Corresponding (Green's) function in the complex plane.
    zut    - complex. points in the complex plane.

    Returns the obtained Padé approximant at the points zut.
    '''
    N = len(z)
    r = N // 2
    # Build up upper diagonal of matrix g_{i,j} = g_{i}(z_{j}) from recursive algorithm:
    # g_{i,j} = (g_{i-1,i-1}-g_{i-1,j})/((z_j-z_{i-1})*g_{i-1,j}) for i=1,2,3,...,N-1
    # with
    # g_{0,j} = f_j
    # Note difference to litterature due to start with index 0.
    g = np.zeros((N, N), dtype=np.complex256)  # quadruple precision
    g[0, :] = np.squeeze(f)
    for i in range(1, N):
        g[i, :] = (g[i - 1, i - 1] - g[i - 1, :]) / ((z - z[i - 1]) * g[i - 1, :])
    c = g.diagonal()
    # The Padé approximant, represented as a terminated continued fraction,
    # has coefficients which are given by the diagonal: c_j = g_{j,j}.
    # Once these coefficients are determined,
    # calculate approximant by recursive algorithm:
    # A_{n+1}(z) = A_n(z)+(z-z_{n-1})*c_n*A_{n-1}(z)
    # B_{n+1}(z) = B_n(z)+(z-z_{n-1})*c_n*B_{n-1}(z)
    # for n=1,2,3,...,N-1
    # with
    # A_0 = 0
    # A_1 = c[0]
    # B_0 = B_1 = 1.
    A = np.zeros((N + 1, len(zut)), dtype=np.complex256)  # quadruple precision
    B = np.zeros((N + 1, len(zut)), dtype=np.complex256)  # quadruple precision
    A[0, :] = 0
    A[1, :] = c[0]
    B[0, :] = 1
    B[1, :] = 1
    for n in range(1, N):
        A[n + 1, :] = A[n, :] + (zut - z[n - 1]) * c[n] * A[n - 1, :]
        B[n + 1, :] = B[n, :] + (zut - z[n - 1]) * c[n] * B[n - 1, :]
    return A[-1, :] / B[-1, :]


def padeMatrix(z, f, N, verbose=False):
    '''
    Input variables:
    z       - complex. points in the complex plane.
    f       - complex. Corresponding (Green's) function in the complex plane.
    N       - int. Number of Padé coefficients to use
    verbose - boolean. Determine if to print solution information

    Returns the obtained Padé coefficients.
    '''
    # number of input points
    M = len(z)
    r = N // 2
    y = f * z ** r
    A = np.ones((M, N), dtype=np.complex)
    for i in range(M):
        A[i, :r] = z[i] ** (np.arange(r))
        A[i, r:] = -f[i] * z[i] ** (np.arange(r))
        # Calculated Padé coefficients
    # rcond=-1 means all singular values will be used in the solution.
    sol = np.linalg.lstsq(A, y, rcond=-1)
    # Padé coefficents
    x = sol[0]
    if verbose:
        print('error_2= ', np.linalg.norm(np.dot(A, x) - y))
        print('residuals = ', sol[1])
        print('rank = ', sol[2])
        print('singular values / highest_singlular_value= ', sol[3] / sol[3][0])
    return x


def epade(z, x):
    '''
    Input variables:
    z - complex. Points where continuation is evaluated.
    x - complex. Padé approximant coefficient.

    Returns the value of the Padé approximant at the points z.
    '''
    r = len(x) // 2
    numerator = np.zeros(len(z), dtype=np.complex256)
    denomerator = np.zeros(len(z), dtype=np.complex256)
    for i in range(r):
        numerator += x[i] * z ** i
        denomerator += x[r + i] * z ** i
    denomerator += z ** r
    return numerator / denomerator


def padeNonlinear(z, f, N, verbose=False):
    '''
    Input variables:
    z      - complex. points in the complex plane.
    f      - complex. Corresponding (Green's) function in the complex plane.
    N      - integer. Number of complex Padé coefficients to use
    solver - integer. Either 2 or 3. Determines which type of non-linear LS routine to use.

    Returns the obtained Padé coefficients.
    '''

    def epader(Zin, Xin):
        '''
        Input variables:
        Zin - float64 array. The input points, seperated in real and imaginary part.
        Xin - float64 array. The Padè coefficients, seperated in real and imaginary part.

        Returns: a float64 array. The Pade approximant, separated in real and imaginary part.
        '''
        m = len(Zin)
        n = len(Xin)
        zin = Zin[:m // 2] + Zin[m // 2:] * 1j
        xin = Xin[:n // 2] + Xin[n // 2:] * 1j
        p = epade(zin, xin)
        P = np.zeros_like(Zin)
        P[:m // 2] = p.real
        P[m // 2:] = p.imag
        return P

    def pade_err(Xin, Zin, Fin):
        '''
        Input variables:
        Xin - float64 array. The Padè coefficients, seperated in real and imaginary part.
        Zin - float64 array. The input points, seperated in real and imaginary part.
        Fin - float64 array. The function, seperated in real and imaginary part.

        Returns: a float64 array. The Padé error, seperated in real and imaginary part.
        '''
        Diff = epader(Zin, Xin) - Fin
        return Diff

    def jac(Xin, Zin, Fin):
        '''
        Input variables:
        Xin - float64 array. The Padè coefficients, seperated in real and imaginary part.
        Zin - float64 array. The input points, seperated in real and imaginary part.
        Fin - float64 array. The function, seperated in real and imaginary part.

        Returns: a float64 matrix. The derivative of the Padé error w.r.t. the Xin variables, seperated in real and imaginary part.
        '''
        m = len(Zin)
        n = len(Xin)
        r = n // 4
        zin = Zin[:m // 2] + Zin[m // 2:] * 1j
        xin = Xin[:n // 2] + Xin[n // 2:] * 1j
        nu = np.zeros(len(zin), dtype=np.complex256)
        de = np.zeros(len(zin), dtype=np.complex256)
        for i in range(r):
            nu += xin[i] * zin ** i
            de += xin[r + i] * zin ** i
        de += zin ** r
        a = np.zeros((m // 2, r), dtype=np.complex128)
        b = np.zeros((m // 2, r), dtype=np.complex128)
        c = np.zeros((m // 2, r), dtype=np.complex128)
        d = np.zeros((m // 2, r), dtype=np.complex128)
        for i in range(r):
            a[:, i] = zin ** i / de
            b[:, i] = -nu * zin ** i / de ** 2
            c[:, i] = 1j * zin ** i / de
            d[:, i] = -nu * 1j * z ** i / de ** 2
        e = np.concatenate((a, b, c, d), axis=1)
        J = np.zeros((m, n), dtype=np.float64)  # the Jacobian, which will be returned
        J[:m // 2, :] = e.real
        J[m // 2:, :] = e.imag
        return J

    # convert complex input arrays to real-valued arrays which are twice as long.
    M = len(z)  # number of input points
    Z = np.zeros(2 * M, dtype=np.float64)
    F = np.zeros(2 * M, dtype=np.float64)
    Z[:M] = z.real
    Z[M:] = z.imag
    F[:M] = f.real
    F[M:] = f.imag

    from scipy.optimize import leastsq
    X0 = np.zeros(2 * N, dtype=np.float64)  # starting Pade coefficients
    X0[N // 2 - 1] = 1
    if(verbose):
        print('number of variables:', 2 * N, '. maxiteration:', 100 * (2 * N + 1))
    res = leastsq(pade_err, X0, args=(Z, F), Dfun=jac, ftol=10 ** (-14), xtol=10 ** (-14), full_output=1)
    X = res[0]
    infodict = res[2]
    mesg = res[3]
    ret_val = res[4]
    if(verbose):
        print('--------output from leastsq-----------------')
        print(mesg)
        print('number of function calls:', infodict['nfev'])
        print('max-deviation value:', np.max(infodict['fvec']))
        if ret_val in [1, 2, 3, 4]:
            print('The non-linear LS Padé method was sucessful.')
        else:
            print('The non-linear LS Padé method was NOT sucessful. Check solution...')
            print('return value =', ret_val)
    x = X[:N] + X[N:] * 1j
    return x


def acPade(z, fs, N, zout, solver):
    '''
    Input variables:
    z      - complex. points in the complex plane.
    fs     - complex. Corresponding (Green's) function in the complex plane.
                      Columns represent different functions.
    N      - int. Number of Padé coefficients to use
    zout   - complex. Points where continuation is evaluated
    solver - integer. How to evaluate the Pade approximant at zout.
                      Either using Beach's matrix formulation, Thiele's algorithm
                      or some kind of non-linear method.

    Assumes the asymptote of fs is 0+a/z+b/z^2.
    '''
    if not N % 2 == 0:
        sys.exit("Error: The number of Padé coefficients in acPade function should be even.")
    fout = []
    for f in fs.T:  # column by column
        if solver == 0:  # Beach's matrix formulation
            x = padeMatrix(z, f,
                           N)  # Construct matrix and rhs in linear system of equations and seek a LS solution.
            fout.append(epade(zout, x).T)  # Evaluate Padé approximant at points zout
        elif solver == 1:  # Thiele's algorithm
            if N == len(z):
                fout.append(padeThiele(z, f, zout).T)
            else:
                sys.exit("Error: For Thiele's algorithm, equal number of coefficients as input points are expected.")
        elif solver == 2:
            x = padeNonlinear(z, f, N)  # Seek a non-linear LS solution of: P(z)-f
            fout.append(epade(zout, x).T)  # Evaluate Padé approximant at points zout
        else:
            print("solver = ", solver)
            sys.exit("Error: solver value not valid.")

    fout = np.array(fout).T
    return fout


def pick_points(z, f, nmin, M, cols):
    '''
    z      - complex. Points in the complex plane.
    f      - complex. function values for different orbitals at the points 'z'.
    nmin   - integer. minimum index of the points to pick.
    M      - integer. Number of points to pick.
    cols   - integer. Columns to pick.

    Pick out some of the input points from z and f.
    Pick out columns from f given by variable cols.
    '''
    if nmin >= 0:
        zp = z[nmin:nmin + M]
        fp = f[nmin:nmin + M, cols]
    else:
        # use mirror symmetry of (Green's) function: f(z^*)_{i,i} = f(z)_{i,i}^*
        # Extension to multi-orbital case is: f(z^*)_{i,j} = f(z)_{j,i}^*
        nadd = -nmin
        zp = np.hstack([np.conj(z[nadd - 1::-1]), z[:M - nadd]])
        fp = np.vstack([np.conj(f[nadd - 1::-1, cols]), f[:M - nadd, cols]])
    return zp, fp


def pade(zin, fin, zout):
    '''
    Input variables:
    zin  - complex. Points in complex plane.
    fin  - complex. With several columns for different orbitals. Expects a 2d array
    zout - complex. Points where continuation is evaluated.
    '''
    # ---------------------------------------------------------------------------------------
    # Pade parameter settings
    cols = [0]  # which columns in fin to continue. E.g. range(17), [0,1], [0]
    nmins = [-3, -2, -1]  # minimum index of the input points to include. E.g. [-5,-1], [0]
    # For values < 0 mirror values are added to
    # input points before continuations.
    Mmin = 50  # minimum number of input points to use
    Mmax = 100  # maximum number of input points to use
    Nmin = 40  # minimum number of Padé coefficients to use
    Nmax = 80  # maximum number of Padé coefficients to use
    Mstep = 4  # step size in M
    Nstep = 4  # step size in N
    diagonalPade = False  # To perform continuations with only N==M or with N<=M
    solver = 0  # How to evaluate the Pade approximant at zout.
    # Either using Beach's matrix formulation, Thiele's algorithm
    # or some kind of non-linear method.
    # solver=0 (Beach's method)
    # solver=1 (Thiele's algorithm). Requires diagonalPade=True
    # solver=2 (non-linear LS)
    c1v = 0.99  # value for criterion 1
    c2v = 0.51  # value for criterion 2

    # Computational time for Beach's method:
    # O((Nmax-Nmin)*(Mmax-Mmin)*Nmax^3) for doing the continuations
    # O(((Nmax-Nmin)*(Mmax-Mmin))^2*Mmax) for calculating the devations between the continuations
    # ---------------------------------------------------------------------------------------

    Ms = np.arange(Mmin, Mmax + 1, Mstep)
    Ns = np.arange(Nmin, Nmax + 1, Nstep)

    # 2Do: find constant asymptote term from input data to
    # make the data to continue optimal for Pade
    # It works quite well without this acually...

    # Loop over all the continuations to perform
    fs = []
    counter = 0
    for nmin in nmins:
        for M in Ms:
            zin_p, fin_p = pick_points(zin, fin, nmin, M, cols)
            if not diagonalPade:
                for N in Ns[Ns <= M]:
                    counter += 1
                    f = acPade(zin_p, fin_p, N, zout, solver)
                    if np.all(np.imag(f) <= 0):
                        fs.append(f)
            else:
                counter += 1
                N = M
                f = acPade(zin_p, fin_p, N, zout, solver)
                if np.all(np.imag(f) <= 0):
                    fs.append(f)

    print(counter, ' continuations performed per orbital.')
    if not fs:
        sys.exit("Error: No physical continuations found.")
    fs = np.array(fs)
    masks = []
    fmeans = []
    # Loop over the orbitals (they are independent from now on)
    for a in range(np.shape(fs)[2]):
        f = fs[:, :, a]
        if np.size(f, 0) == 1:  # only one physical continuation
            mask = [True]
            fmean = f[0, :]
        else:
            # loop over all physical continuations,
            # calculating distance between imaginary part of the continuations
            delta = np.zeros((len(f), len(f)))  # store distances between all the continuations
            for i, fi in enumerate(f):
                for j in range(i + 1, len(f)):
                    delta[i, j] = np.linalg.norm(fi.imag - f[j].imag, ord=1)
            d = np.zeros(len(f))  # store accumulative distances to the other continuations
            for i, fi in enumerate(f):
                d[i] = np.sum(delta[:i, i]) + np.sum(delta[i, i + 1:])
            # calculate if to include or reject continuations based on two criteria:
            c1 = d <= c1v * np.mean(d)
            c2 = np.zeros(len(d), dtype=np.bool)
            ind = np.argsort(d)[:int(c2v * len(d))]  # indices to 51% lowest distances
            c2[ind] = True
            mask = c1 & c2  # continuations to include
            fmean = np.mean(f[mask], axis=0)  # average selected continuations
        masks.append(mask)
        fmeans.append(fmean)
    masks = np.array(masks).T
    fmeans = np.array(fmeans).T
    # return the average continuations, all physical continuations and masks for averaging
    return fmeans, fs, masks

def load_and_prepare_input_data(folder,inputfile='pade.in',realaxisfile='exact.dat',delta=0.01):
    '''
    folder - directory to test model to analytically continue
    inputfile - input data
    realaxisfile - exact values on the real-axis
    delta - distance above the real axis where to evaluete the continuations
    '''
    # Matsubara input data
    d = np.loadtxt(folder + 'pade.in')
    zin = d[:,0] + d[:,1]*1j
    fin = d[:,2]+d[:,3]*1j
    # (Fortran ordering)
    fin = np.atleast_2d(fin).T
    # exact function
    exact = np.loadtxt(folder + 'exact.dat')
    # real axis energy
    w = exact[:,0]
    zout = w + delta*1j
    # the exact Green's function
    exact = exact[:,1] + exact[:,2]*1j
    return zin,fin,w,zout,exact

if __name__ == '__main__':
    #zin, fin, w, zout, exact = load_and_prepare_input_data('../../../PadeApproximants/tests/betheU0/')
    zin, fin, w, zout, exact = load_and_prepare_input_data('../../../PadeApproximants/tests/two-poles_wc1_dw0.5/')
    #zin, fin, w, zout, exact = load_and_prepare_input_data('../../../PadeApproximants/tests/haldane_model/')
    #zin, fin, w, zout, exact = load_and_prepare_input_data('../../../PadeApproximants/tests/Sm7/')

    fmeans, fs, masks = pade(zin, fin, zout)
    print('-----------  After pade routine  ----------')
    print('(#physical continuation, #E, #orbitals) = ', np.shape(fs))
    print('#picked continuations =', np.sum(masks))
    print('(#E, #orbitals) = ', np.shape(fmeans))

    cont = padeThiele(zin,fin,zout)
    contMatrix = padeMatrix(zin,fin,10,True)

    plt.figure()
    plt.plot(zin.imag,fin.imag)
    plt.show()

    plt.figure()
    # pick one orbital to plot
    col = 0
    # plot all physical continuations
    plt.plot(w, -1. / np.pi * np.imag(fs[:, :, col].T), '-g')
    plt.plot(w, -1. / np.pi * np.imag(fs[0, :, col].T), '-g', label='all physical continuations')
    # plot all picked continuations
    plt.plot(w, -1. / np.pi * np.imag(fs[masks[:, col], :, col].T), '-b')
    plt.plot(w, -1. / np.pi * np.imag((fs[masks[:, col], :, col].T)[:, 0]), '-b', label='all picked continuations')
    # plot the average of the picked continuations
    plt.plot(w, -1. / np.pi * np.imag(fmeans[:, col]), '-r', linewidth=2, label='average')
    # plot exact spectral function
    plt.plot(w, -1. / np.pi * np.imag(exact), '-k', linewidth=1.5, label='exact')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\rho$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    plt.plot(w, -1. / np.pi * np.imag(fmeans[:, col]), '-r', linewidth=2, label='average')
    plt.plot(w, -1. / np.pi * np.imag(exact), '-k', linewidth=1.5, label='exact')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\rho$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    plt.plot(w, -1. / np.pi * np.imag(exact), '-k', linewidth=1.5, label='exact')
    plt.plot(w, -1. / np.pi * np.imag(cont), '-r', linewidth=1.5, label='PadeThiel')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\rho$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # fig = plt.figure()
    # plt.plot(w, -1. / np.pi * np.imag(exact), '-k', linewidth=1.5, label='exact')
    # plt.plot(w, -1. / np.pi * np.imag(contMatrix), '-r', linewidth=1.5, label='contMatrix')
    # plt.xlabel(r'$\omega$')
    # plt.ylabel(r'$\rho$')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()