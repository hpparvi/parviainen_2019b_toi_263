import pandas as pd

from copy import copy
from pathlib import Path

from astropy.io import fits as pf
from astropy.stats import sigma_clip
from astropy.table import Table
from numpy import zeros, diff, concatenate, sqrt
from uncertainties import ufloat, nominal_value

from pytransit.utils.keplerlc import KeplerLC

N = lambda a: a/a.median()

zero_epoch = ufloat(2458386.1723494, 1.4703e-03)
period     = ufloat(      0.5567365, 7.3461e-05)

star_teff = (3250, 140)
star_logg = (4.9, 0.1)
star_z    = (0.0, 0.1)

root = Path(__file__).parent.parent.resolve()
tess_file = root.joinpath('photometry/tess/tess2018263035959-s0003-0000000120916706-0123-s_lc.fits').resolve()
reduced_m2_files = sorted(root.joinpath('results').glob('*fits'))

def normalize(a):
    if isinstance(a, pd.DataFrame):
        return (a - a.mean()) / a.std()

# TESS routines
# -------------

def read_tess(dfile: Path, zero_epoch: float, period: float, use_pdc: bool = False,
              transit_duration_d: float = 0.1, baseline_duration_d: float = 0.3):
    fcol = 'PDCSAP_FLUX' if use_pdc else 'SAP_FLUX'
    tb = Table.read(dfile)
    bjdrefi = tb.meta['BJDREFI']
    df = tb.to_pandas().dropna(subset=['TIME', 'SAP_FLUX', 'PDCSAP_FLUX'])
    lc = KeplerLC(df.TIME.values + bjdrefi, df[fcol].values, zeros(df.shape[0]),
                  nominal_value(zero_epoch), nominal_value(period), transit_duration_d, baseline_duration_d)
    times, fluxes = copy(lc.time_per_transit), copy(lc.normalized_flux_per_transit)
    return times, fluxes, len(times)*['tess'], [diff(concatenate(fluxes)).std() / sqrt(2)]

# MuSCAT2 routines
# ----------------

def read_m2(files: list):
    times, fluxes, pbs, wns, covs = [], [], [], [], []
    for f in files:
        with pf.open(f) as hdul:
            npb = (len(hdul)-1)//2
            for hdu in hdul[1:1+npb]:
                fobs = hdu.data['flux'].astype('d').copy()
                fmod = hdu.data['model'].astype('d').copy()
                time = hdu.data['time_bjd'].astype('d').copy()
                mask = ~sigma_clip(fobs-fmod, sigma=5).mask
                times.append(time[mask])
                fluxes.append(fobs[mask])
                pbs.append(hdu.header['filter'])
                wns.append(hdu.header['wn'])
            for i in range(npb):
                covs.append(Table.read(f, 1+npb+i).to_pandas().values[:,1:])
    return times, fluxes, pbs, wns, covs


# LCO routines
# ------------

def read_lco_table(fname: Path):
    dff = pd.read_csv(fname, sep='\t', index_col=0)
    refcols = [c for c in dff.columns if 'rel_flux_C' in c]
    df = dff['BJD_TDB rel_flux_T1'.split() + refcols]
    covariates = normalize(dff[['AIRMASS', 'FWHM_Mean', 'X(IJ)_T1', 'Y(IJ)_T1']]).values
    return df, covariates

def read_lco_data():
    dfg, cg = read_lco_table('photometry/lco/TIC120916706-01_20190121_LCO-SAAO-1m_gp_measurements.tbl')
    dfr, cr = read_lco_table('photometry/lco/TIC120916706.01_20181226_LCO-CTIO-1m_rp_measurements.tbl')
    dfi, ci = read_lco_table('photometry/lco/TIC120916706-01_20181213_LCO-SAAO-1m_ip_measurements.tbl')
    dfs = dfg, dfr, dfi
    covs = [cg, cr, ci]

    times = [df.BJD_TDB.values for df in dfs]
    fluxes = [N(df.rel_flux_T1).values for df in dfs]
    return times, fluxes, 'g r i'.split(), [diff(f).std() / sqrt(2) for f in fluxes], covs