"""Contaminated TESS LPF

This module contains the log posterior function (LPF) for a TESS light curve with possible third-light
contamination from an unresolved source inside the photometry aperture.
"""
from pathlib import Path

from numpy import repeat, inf, where, newaxis, squeeze, atleast_2d, isfinite, concatenate, zeros
from numpy.random.mtrand import uniform
from pytransit.lpf.tesslpf import TESSLPF
from pytransit.param import UniformPrior as UP, NormalPrior as NP, PParameter
from ldtk import tess

from core import zero_epoch, period, star_logg, star_teff, star_z, tess_file

class CTESSLPF(TESSLPF):
    """Contaminated TESS LPF

    This class implements a log posterior function for a TESS light curve that allows for unknown flux contamination.
    The amount of flux contamination is not constrained.
    """
    def __init__(self, name: str, flux_type: str, use_ldtk: bool = False, nsamples: int = 2,
                 nlegendre: int = 3, bldur: float = 0.1, trdur: float = 0.04):

        assert flux_type.lower() in ('sap', 'pdc'), "Flux type needs to be either 'sap' or 'pdc'"
        super().__init__(name, tess_file, zero_epoch.n, period.n, nsamples=nsamples, nlegendre=nlegendre,
                         bldur=bldur, trdur=trdur, use_pdc=(flux_type.lower() == 'pdc'))

        self.result_dir = Path('results')
        self.set_prior('zero_epoch', NP(zero_epoch.n - self.bjdrefi, 3 * zero_epoch.s))
        self.set_prior('period', NP(period.n, 3 * period.s))
        self.set_prior('k2_true', UP(0.1 ** 2, 0.75 ** 2))
        if use_ldtk:
            self.add_ldtk_prior(star_teff, star_logg, star_z, passbands=(tess,))

    def _init_p_planet(self):
        ps = self.ps
        pk2 = [PParameter('k2_true', 'true_area_ratio', 'A_s', UP(0.10 ** 2, 0.75 ** 2), (0.10 ** 2, 0.75 ** 2)),
               PParameter('k2_app', 'apparent_area_ratio', 'A_s', UP(0.10 ** 2, 0.30 ** 2), (0.10 ** 2, 0.30 ** 2))]
        ps.add_passband_block('k2', 1, 2, pk2)
        self._pid_k2 = repeat(ps.blocks[-1].start, self.npb)
        self._start_k2 = ps.blocks[-1].start
        self._sl_k2 = ps.blocks[-1].slice
        self.lnpriors.append(lambda pv: where(pv[:, 5] < pv[:, 4], 0, -inf))

    def transit_model(self, pv):
        pv = atleast_2d(pv)
        flux = super().transit_model(pv)
        cnt = 1. - pv[:, 5] / pv[:, 4]
        return squeeze(cnt[:, newaxis] + (1. - cnt[:, newaxis]) * flux)

    def create_pv_population(self, npop=50):
        pvp = zeros((0, len(self.ps)))
        npv, i = 0, 0
        while npv < npop and i < 10:
            pvp_trial = self.ps.sample_from_prior(npop)
            pvp_trial[:, 5] = pvp_trial[:, 4]
            cref = uniform(0, 0.99, size=npop)
            pvp_trial[:, 4] = pvp_trial[:, 5] / (1. - cref)
            lnl = self.lnposterior(pvp_trial)
            ids = where(isfinite(lnl))
            pvp = concatenate([pvp, pvp_trial[ids]])
            npv = pvp.shape[0]
            i += 1
        pvp = pvp[:npop]
        return pvp
