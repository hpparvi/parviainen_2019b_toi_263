from pathlib import Path
from time import strftime
from numba import njit, prange
import pandas as pd
import xarray as xa
from matplotlib.pyplot import subplots, setp
from numpy import arange, concatenate, zeros, inf, atleast_2d, where, repeat, squeeze, argsort, isfinite, ones, c_, \
    array
from numpy.polynomial.legendre import legvander
from numpy.random import normal, uniform

from pytransit.contamination import SMContamination, Instrument
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit import LinearModelBaseline
from pytransit.lpf.cntlpf import PhysContLPF
from pytransit.lpf.cntlpf import map_pv_pclpf, map_ldc, contaminate
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits import epoch
from pytransit.param.parameter import NormalPrior as NP, UniformPrior as UP, GParameter, PParameter, LParameter
from pytransit.utils.misc import fold
from ldtk import tess

from core import read_tess, read_m2, read_lco_data, zero_epoch, period, reduced_m2_files, tess_file, star_teff, star_logg, star_z


class TMLPF(LinearModelBaseline, PhysContLPF):
    def __init__(self, name: str, tess_baseline_duration: float = 0.1, tess_transit_duration: float = 0.04,
                 use_ldtk: bool = False):

        times_t, fluxes_t, pbs_t, wns_t = read_tess(tess_file, zero_epoch.n, period.n,
                                                    baseline_duration_d=tess_baseline_duration,
                                                    transit_duration_d=tess_transit_duration)
        times_m, fluxes_m, pbs_m, wns_m, covs_m = read_m2(reduced_m2_files)
        times_l, fluxes_l, pbs_l, wns_l, covs_l = read_lco_data()

        times = times_t + times_m + times_l
        fluxes= fluxes_t + fluxes_m + fluxes_l
        pbs = pbs_t + pbs_m + pbs_l
        wns = wns_t + wns_m + wns_l
        covs = len(times_t)*[array([[]])] + covs_m + covs_l

        self._stess = len(times_t)
        self._ntess = sum([t.size for t in times_t])

        pbnames = 'tess g r i z_s'.split()
        pbids = pd.Categorical(pbs, categories=pbnames).codes
        wnids = concatenate([zeros(len(times_t), 'int'), arange(1, 13)])

        PhysContLPF.__init__(self, name, passbands=pbnames, times=times, fluxes=fluxes, pbids=pbids, wnids=wnids,
                             covariates=covs)
        self.result_dir = Path('results')

        self.set_prior('zero_epoch', NP(zero_epoch.n, zero_epoch.s))
        self.set_prior('period', NP(period.n, period.s))
        self.set_prior('k2_app', UP(0.12 ** 2, 0.30 ** 2))
        self.set_prior('teff_h', NP(3250, 140))

        if use_ldtk:
            self.add_ldtk_prior(star_teff, star_logg, star_z, (tess, sdss_g, sdss_r, sdss_i, sdss_z))


    def _init_p_planet(self):
        ps = self.ps
        pk2 = [PParameter('k2_app', 'apparent_area_ratio', 'A_s', UP(0.1**2, 0.30**2), (0.1**2, 0.30**2))]
        pcn = [GParameter('k2_true', 'true_area_ratio', 'As', UP(0.1**2, 0.75**2), bounds=(1e-8, inf)),
               GParameter('teff_h', 'host_teff', 'K', UP(2500, 12000), bounds=(2500, 12000)),
               GParameter('teff_c', 'contaminant_teff', 'K', UP(2500, 12000), bounds=(2500, 12000)),
               GParameter('k2_app_tess', 'tess_apparent_area_ratio', 'A_s', UP(0.1**2, 0.30**2), (0.1**2, 0.3**2))]
        ps.add_passband_block('k2', 1, 1, pk2)
        self._pid_k2 = repeat(ps.blocks[-1].start, self.npb)
        self._start_k2 = ps.blocks[-1].start
        self._sl_k2 = ps.blocks[-1].slice
        ps.add_global_block('contamination', pcn)
        self._pid_cn = arange(ps.blocks[-1].start, ps.blocks[-1].stop)
        self._sl_cn = ps.blocks[-1].slice

    def create_pv_population(self, npop=50):
        pvp = zeros((0, len(self.ps)))
        npv, i = 0, 0
        while npv < npop and i < 10:
            pvp_trial = self.ps.sample_from_prior(npop)
            pvp_trial[:, 5] = pvp_trial[:, 4]
            cref = uniform(0, 0.99, size=npop)
            pvp_trial[:, 5] = pvp_trial[:, 4] / (1. - cref)
            lnl = self.lnposterior(pvp_trial)
            ids = where(isfinite(lnl))
            pvp = concatenate([pvp, pvp_trial[ids]])
            npv = pvp.shape[0]
            i += 1
        pvp = pvp[:npop]
        return pvp

    def additional_priors(self, pv):
        """Additional priors."""
        pv = atleast_2d(pv)
        return sum([f(pv) for f in self.lnpriors], 0)

    def _init_instrument(self):
        """Set up the instrument and contamination model."""
        self.instrument = Instrument('example', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")
        self.lnpriors.append(lambda pv: where(pv[:, 4] < pv[:, 5], 0, -inf))
        self.lnpriors.append(lambda pv: where(pv[:, 8] < pv[:, 5], 0, -inf))

    def transit_model(self, pvp):
        pvp = atleast_2d(pvp)
        cnt = zeros((pvp.shape[0], self.npb))
        pvt = map_pv_pclpf(pvp)
        ldc = map_ldc(pvp[:, self._sl_ld])
        flux = self.tm.evaluate_pv(pvt, ldc)
        cnt[:, 0] = 1 - pvp[:, 8] / pvp[:, 5]
        for i, pv in enumerate(pvp):
            if (2500 < pv[6] < 12000) and (2500 < pv[7] < 12000):
                cnref = 1. - pv[4] / pv[5]
                cnt[i, 1:] = self.cm.contamination(cnref, pv[6], pv[7])
            else:
                cnt[i, 1:] = -inf
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def plot_folded_tess_transit(self, method='de', pv=None, figsize=None, ylim=None):
        assert method in ('de', 'mc')
        if pv is None:
            if method == 'de':
                pv = self.de.minimum_location
            else:
                df = self.posterior_samples(derived_parameters=False)
                pv = df.median.values

        etess = self._ntess
        t = self.timea[:etess]
        fo = self.ofluxa[:etess]
        fm = squeeze(self.transit_model(self.de.population))[self.de.minimum_index, :etess]
        bl = squeeze(self.baseline(self.de.population))[self.de.minimum_index, :etess]

        fig, ax = subplots(figsize=figsize)
        phase = pv[1] * (fold(t, pv[1], pv[0], 0.5) - 0.5)
        sids = argsort(phase)
        phase = phase[sids]
        bp, bf, be = downsample_time(phase, (fo / bl)[sids], 4 / 24 / 60)
        ax.plot(phase, (fo / bl)[sids], 'k.', alpha=0.2)
        ax.errorbar(bp, bf, be, fmt='ko')
        ax.plot(phase, fm[sids], 'k')
        setp(ax, ylim=ylim, xlabel='Time', ylabel='Normalised flux')
        return fig

    def plot_m2_transits(self, figsize=(14, 5)):
        fig, axs = subplots(3, 4, figsize=figsize, constrained_layout=True, sharex='all', sharey='all')
        fmodel = squeeze(self.flux_model(self.de.population))[self.de.minimum_index]
        etess = self._stess
        t0, p = self.de.minimum_location[[0,1]]

        for i, ax in enumerate(axs.T.flat):
            t = self.times[etess + i]
            e = epoch(t.mean(), t0, p)
            tc = t0 + e * p
            ax.plot(t - tc, self.fluxes[etess + i], 'k.', alpha=0.2)
            ax.plot(t - tc, fmodel[self.lcslices[etess + i]], 'k')
            setp(ax, xlim=(-0.045, 0.045))


        setp(axs, ylim=(0.92, 1.05))
        setp(axs[-1, :], xlabel='Time [BJD]')
        setp(axs[:, 0], ylabel='Normalised flux')
        return fig
