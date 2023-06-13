# ======================================================================================================================
# ----------------------------------------------- FOUR-POINT-FACTORY ---------------------------------------------------
# ======================================================================================================================


class local_four_point_factory():
    ''' class which contains operations on four-point objects. '''

    def __init__(self, u: float = 1.0, chi0_core: LocalBubble = None, chi0_urange: LocalBubble = None,
                 chi0_asympt: LocalBubble = None):
        self._u = u
        self._chi0_core = chi0_core
        self._chi0_urange = chi0_urange
        self._chi0_asympt = chi0_asympt

        assert (
                chi0_urange.niv_sum == chi0_asympt.niv_sum), 'Niv_sum i inconsistent between chi0_urange and chi0_asympt'

    @property
    def u(self) -> float:
        return self._u

    @property
    def niv_core(self) -> int:
        return self._chi0_core.niv_sum

    @property
    def niv_urange(self) -> int:
        return self._chi0_urange.niv_sum

    @property
    def niv_asympt(self) -> int:
        return self._chi0_asympt.niv_asympt

    @property
    def beta(self) -> float:
        return self._chi0_core.beta

    @property
    def gchi0_core(self):
        return self._chi0_core.gchi0

    @property
    def gchi0_urange(self):
        return self._chi0_urange.gchi0

    def get_ur(self, channel='dens'):
        if (channel == 'magn'):
            sign = 1
        else:
            sign = -1
        return self.u * sign

    # ==================================================================================================================
    def gammar_from_gchir(self, gchir: LocalFourPoint = None):
        u_r = self.get_ur(gchir.channel)
        gammar = np.array(
            [self.gammar_from_gchir_wn(gchir=gchir.mat[wn], gchi0_urange=self.gchi0_urange[wn], niv_core=self.niv_core,
                                       beta=gchir.beta, u=u_r) for wn in gchir.wn_lin])
        return LocalFourPoint(matrix=gammar, giw=gchir.giw, channel=gchir.channel, beta=gchir.beta, iw=gchir.iw_core)

    @classmethod
    def gammar_from_gchir_wn(self, gchir=None, gchi0_urange=None, niv_core=10, beta=1.0, u=1.0):
        full = u / (beta * beta) + np.diag(1. / gchi0_urange)
        inv_full = np.linalg.inv(full)
        inv_core = cut_iv(inv_full, niv_core)
        core = np.linalg.inv(inv_core)
        chigr_inv = np.linalg.inv(gchir)
        return -(core - chigr_inv - u / (beta * beta))

    # ==================================================================================================================

    # ==================================================================================================================
    def gchi_aux_from_gammar(self, gammar: LocalFourPoint = None):
        u_r = self.get_ur(gammar.channel)
        gchi_aux = np.array(
            [self.gchi_aux_from_gammar_wn(gammar=gammar.mat[wn], gchi0=self.gchi0_core[wn], beta=gammar.beta,
                                          u=u_r) for wn in gammar.wn_lin])
        return LocalFourPoint(matrix=gchi_aux, giw=gammar.giw, channel=gammar.channel, beta=gammar.beta, iw=gammar.iw_core)

    @classmethod
    def gchi_aux_from_gammar_wn(self, gammar=None, gchi0=None, beta=1.0, u=1.0):
        gchi0_inv = np.diag(1. / gchi0)
        chi_aux_inv = gchi0_inv + gammar - u / (beta * beta)
        return np.linalg.inv(chi_aux_inv)
    # ==================================================================================================================