import pandas as pd

from pathlib import Path
from numpy import arange, where, inf
from pytransit import PhysContLPF, LinearModelBaseline, NormalPrior as NP, UniformPrior as UP
from pytransit.contamination import Instrument, SMContamination, sdss_r, sdss_i, sdss_z

from core import read_m2, reduced_m2_files, zero_epoch, period, star_teff, star_logg, star_z


class M2LPF(LinearModelBaseline, PhysContLPF):
    def __init__(self, name: str, use_ldtk: bool = False):
        times, fluxes, pbs, wns, covs = read_m2(reduced_m2_files)
        pbnames = 'r i z_s'.split()
        pbids = pd.Categorical(pbs, categories=pbnames).codes
        PhysContLPF.__init__(self, name, pbnames, times, fluxes, pbids=pbids, wnids=arange(len(pbs)),
                             covariates=covs)

        self.result_dir = Path('results')
        self.set_prior('zero_epoch', NP(zero_epoch.n, zero_epoch.s))
        self.set_prior('period', NP(period.n, period.s))
        self.set_prior('k2_app', UP(0.12 ** 2, 0.30 ** 2))
        self.set_prior('teff_h', NP(3250, 140))

        if use_ldtk:
            self.add_ldtk_prior(star_teff, star_logg, star_z, passbands=(sdss_r, sdss_i, sdss_z))

    def _init_instrument(self):
        """Set up the instrument and contamination model."""
        self.instrument = Instrument('example', [sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")
        self.lnpriors.append(lambda pv: where(pv[:, 4] < pv[:, 5], 0, -inf))
