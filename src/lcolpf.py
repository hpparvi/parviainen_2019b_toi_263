from pathlib import Path
from numpy import where, inf
from pytransit import PhysContLPF, LinearModelBaseline, NormalPrior as NP, UniformPrior as UP
from pytransit.contamination import Instrument, SMContamination, sdss_g, sdss_r, sdss_i

from core import read_lco_data, zero_epoch, period, star_teff, star_logg, star_z


class LCOLPF(LinearModelBaseline, PhysContLPF):
    def __init__(self, name: str, use_ldtk: bool = False):
        times, fluxes, pbs, wns, covs = read_lco_data()
        PhysContLPF.__init__(self, name, ["g'", "r'", "i'"], times, fluxes, pbids=[0, 1, 2], wnids=[0, 1, 2],
                             covariates=covs)
        self.result_dir = Path('results')

        self.set_prior('zero_epoch', NP(zero_epoch.n, zero_epoch.s))
        self.set_prior('period', NP(period.n, period.s))
        self.set_prior('k2_app', UP(0.12 ** 2, 0.30 ** 2))
        self.set_prior('teff_h', NP(3250, 140))

        if use_ldtk:
            self.add_ldtk_prior(star_teff, star_logg, star_z, (sdss_g, sdss_r, sdss_i))

    def _init_instrument(self):
        self.instrument = Instrument('example', [sdss_g, sdss_r, sdss_i])
        self.cm = SMContamination(self.instrument, "i'")
        self.lnpriors.append(lambda pv: where(pv[:, 4] < pv[:, 5], 0, -inf))
