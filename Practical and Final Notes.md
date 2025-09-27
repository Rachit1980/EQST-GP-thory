Practical and Final Notes

The script uses representative H(z) data—for numerical accuracy, when you want final results, you need to replace the dataset with official sources (Pantheon+, BAO official covariance matrices, and/or compressed Planck likelihood or full Planck via Cobaya).

The number of steps (nsteps = 5000) and nwalkers = 128 are considered a practical equilibrium; for more robust results, increase nsteps to 15000–30000 (depending on your hardware capabilities (3.2 GHz / 4 cores / 32 GB RAM minm)) and check autocorr (printed in emcee_ede_autocorr.txt).