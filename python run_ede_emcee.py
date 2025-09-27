#professor_Ahmed_Ali_EQST-GP_modal*/1.0_GPL.V3
"""
run_ede_emcee.py
Affine-invariant MCMC (emcee) inference for EQST-GP / EDE toy model using CC+BAO+H0 data.

Outputs:
 - emcee_ede_samples.txt  (flattened samples)
 - emcee_ede_corner.png   (corner plot)
 - emcee_ede_autocorr.txt (estimated autocorrelation times)
 - emcee_ede_summary.txt  (best-fit + posterior means/medians)
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee, corner
from scipy.linalg import inv, eigh
import time

# -------------------------
# 1) Data (example assembled used in notebook)
# -------------------------
# Cosmic chronometers (representative compilation; if you have a precise table replace it)
cc_data = np.array([
    [0.09, 69, 12],
    [0.17, 83, 8],
    [0.27, 77, 14],
    [0.40, 95, 17],
    [0.90, 117, 23],
    [1.30, 168, 17],
    [1.43, 177, 18],
    [1.53, 140, 14],
    [1.75, 202, 40],
    [0.48, 97, 62],
    [0.88, 90, 40],
    [0.1791, 75, 4],
    [0.1993, 75, 5],
    [0.3519, 83, 14],
    [0.5929, 104, 13],
    [0.6797, 92, 8],
    [0.7812, 105, 12],
    [0.8754, 125, 17],
    [1.037, 154, 20],
    [0.07, 69, 19.6],
    [0.12, 68.6, 26.2],
    [0.20, 72.9, 29.6],
    [0.28, 88.8, 36.6],
    [1.363, 160, 33.6],
    [1.965, 186.5, 50.4],
    [0.3802, 83, 13.5],
    [0.4004, 77, 10.2],
    [0.4247, 87.1, 11.2],
    [0.44497, 92.8, 12.9],
    [0.4783, 80.9, 9.0],
    [0.47, 89, 49.6]
])
bao_data = np.array([
    [0.38, 81.5, 1.9],
    [0.51, 90.5, 1.97],
    [0.61, 97.3, 2.1]
])
planck = np.array([0.0, 67.4, 0.5])
shoes  = np.array([0.0, 73.2, 1.3])

obs_z = np.concatenate([cc_data[:,0], bao_data[:,0], [planck[0], shoes[0]]])
obs_H = np.concatenate([cc_data[:,1], bao_data[:,1], [planck[1], shoes[1]]])
obs_sigma = np.concatenate([cc_data[:,2], bao_data[:,2], [planck[2], shoes[2]]])

# Build covariance: diagonal for CC and H0, correlated block for BAO (approx r=0.3)
n_cc = len(cc_data)
n_bao = len(bao_data)
bao_start = n_cc
cov = np.diag(obs_sigma**2)
r = 0.3
for i in range(n_bao):
    for j in range(n_bao):
        cov[bao_start + i, bao_start + j] = (r if i!=j else 1.0) * bao_data[i,2] * bao_data[j,2]

# Precompute inverse for likelihood
cov_inv = np.linalg.inv(cov)

# -------------------------
# 2) Model definitions
# -------------------------
Om = 0.315
Or = 9.2e-5
Ok = 0.0
Omega_L0 = 0.685

def omega_ede(z, A, ln1pz_c, sigma):
    """Gaussian bump in ln(1+z)"""
    x = np.log(1.0 + z)
    return A * np.exp(-0.5 * ((x - ln1pz_c)/sigma)**2)

def friedmann_ede(z, H0, A, ln1pz_c, sigma, Oneg, Oneg2):
    Ode = omega_ede(z, A, ln1pz_c, sigma)
    Omega_L_z = Omega_L0 + Ode + Oneg/(1+z) + Oneg2/(1+z)**2
    val = Om*(1+z)**3 + Or*(1+z)**4 + Ok*(1+z)**2 + Omega_L_z
    # safe sqrt (small negative numerical noise clipped to 0)
    return H0 * np.sqrt(np.maximum(val, 0.0))

# -------------------------
# 3) Priors, likelihood, posterior
# -------------------------
def log_prior(theta):
    H0, A, ln1pz_c, sigma, Oneg, Oneg2 = theta
    if not (50.0 < H0 < 80.0): return -np.inf
    if not (0.0 <= A < 0.3): return -np.inf
    if not (0.0 < ln1pz_c < 8.0): return -np.inf
    if not (0.05 <= sigma <= 2.0): return -np.inf
    if not (-0.05 <= Oneg <= 0.05): return -np.inf
    if not (0.0 <= Oneg2 < 0.01): return -np.inf
    return 0.0

def log_likelihood(theta):
    H0, A, ln1pz_c, sigma, Oneg, Oneg2 = theta
    model = np.array([friedmann_ede(z, H0, A, ln1pz_c, sigma, Oneg, Oneg2) for z in obs_z])
    diff = obs_H - model
    chi2 = float(diff @ cov_inv @ diff)
    return -0.5 * chi2

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta)

# -------------------------
# 4) emcee sampler setup
# -------------------------
ndim = 6
nwalkers = 128
# Initial guess: H0~68, A~0.12, ln1pz_c ~ ln(1+300) ~ 5.71, sigma~0.6, small Oneg, Oneg2
p0 = np.array([68.0, 0.12, np.log(1+300.0), 0.6, -0.002, 0.0005])
pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

print("Starting emcee with ndim={}, nwalkers={}".format(ndim, nwalkers))
start_time = time.time()

sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda th: log_posterior(th))
# run: you can increase steps for better sampling; here we do 5000 steps (tunable)
nsteps = 5000
sampler.run_mcmc(pos, nsteps, progress=True)

elapsed = time.time() - start_time
print("emcee finished in {:.1f} s".format(elapsed))

# -------------------------
# 5) Postprocessing and diagnostics
# -------------------------
# Flatten chain, discard burn-in
burn = int(0.2 * nsteps)
flat_samples = sampler.get_chain(discard=burn, thin=1, flat=True)
print("Flat samples shape:", flat_samples.shape)

# Estimate autocorrelation time (may warn if chain too short)
try:
    tau = sampler.get_autocorr_time(tol=0)
    np.savetxt('emcee_ede_autocorr.txt', tau)
    print("Autocorr times:", tau)
except Exception as e:
    tau = None
    print("Warning: could not estimate autocorr reliably (chain maybe short):", e)

# Save samples
np.savetxt('emcee_ede_samples.txt', flat_samples)
print("Saved samples to emcee_ede_samples.txt")

# Corner plot
labels = ['H0','A','ln1pz_c','sigma','Oneg','Oneg2']
fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16,0.5,0.84], show_titles=True)
fig.savefig('emcee_ede_corner.png', dpi=300)
print("Saved corner plot to emcee_ede_corner.png")

# Posterior summary
means = np.mean(flat_samples, axis=0)
meds = np.median(flat_samples, axis=0)
best_idx = np.argmax(sampler.get_log_probability(discard=burn, flat=True))
best_params = flat_samples[best_idx]

with open('emcee_ede_summary.txt', 'w') as f:
    f.write("emcee EDE summary\n")
    f.write(f"Elapsed (s): {elapsed:.1f}\n")
    f.write("Means:\n")
    for name, val in zip(labels, means):
        f.write(f"  {name}: {val:.6e}\n")
    f.write("Medians:\n")
    for name, val in zip(labels, meds):
        f.write(f"  {name}: {val:.6e}\n")
    f.write("Best-fit (max posterior from samples):\n")
    f.write("  " + " ".join([f"{p:.6e}" for p in best_params]) + "\n")
    if tau is not None:
        f.write("Autocorr times:\n")
        for name, t in zip(labels, tau):
            f.write(f"  {name}: {t:.3f}\n")

print("Saved summary to emcee_ede_summary.txt")

# Optionally: quick visual of H(z) for best-fit
H0_b, A_b, ln1pz_c_b, sigma_b, Oneg_b, Oneg2_b = best_params
zplot = np.logspace(-3, np.log10(2.0), 300)
H_best = np.array([friedmann_ede(z, H0_b, A_b, ln1pz_c_b, sigma_b, Oneg_b, Oneg2_b) for z in zplot])
H_lcdm_example = np.array([friedmann_ede(z, 67.4, 0.0, 1.0, 1.0, 0.0, 0.0) for z in zplot])  # for reference

plt.figure(figsize=(8,5))
plt.semilogx(zplot, H_best, label='EDE best-fit')
plt.semilogx(zplot, H_lcdm_example, linestyle='--', label='ΛCDM ref (H0=67.4)')
plt.errorbar(obs_z, obs_H, yerr=obs_sigma, fmt='o', color='k', markersize=3, alpha=0.7)
plt.xlabel('z'); plt.ylabel('H(z) [km/s/Mpc]'); plt.legend(); plt.grid(True, alpha=0.3)
plt.title('H(z) best-fit vs data')
plt.savefig('emcee_ede_Hz_bestfit.png', dpi=300)
print("Saved H(z) best-fit plot to emcee_ede_Hz_bestfit.png")

print("All done. Inspect emcee_ede_corner.png and emcee_ede_summary.txt for results.")
