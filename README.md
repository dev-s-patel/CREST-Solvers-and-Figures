# Cosmological Regularized Effective Spectral Transfer (CREST) — Numerical Pipeline & Verification Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23864786.svg)](https://doi.org/10.5281/zenodo.23864786)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official computational reproduction suite, Einstein–Boltzmann transfer pipeline, and Bayesian Markov Chain Monte Carlo (MCMC) verification codebase for the preprint:

> **Cosmological Regularized Effective Spectral Transfer (CREST): An Effective Field Theory of UV-Regularized Cosmological Perturbations, CAMB Implementation, and Planck Likelihood Bounds**  
> **Author:** Dev S. Patel (Version 2.0, November 2026)  
> **DOI:** [10.5281/zenodo.23864786](https://doi.org/10.5281/zenodo.23864786)

---

## 🔭 Overview & Key Physical Results

This repository contains the numerical verification algorithms, modified CAMB Boltzmann transfer components, and Bayesian likelihood sampling pipelines establishing the observational constraints of **Cosmological Regularized Effective Spectral Transfer (CREST)**—a phenomenological effective field theory (EFT) parameterizing ultraviolet (UV) dispersion cutoffs in the primordial scalar sector while strictly recovering continuous General Relativity and the standard $\Lambda\mathrm{CDM}$ expansion history in the infrared regime ($k \ll k_c$).

* **UV Regularization Scale & Primordial Transfer Kernel:** Characteristic lattice cutoff scale $\ell_{\mathrm{cell}} \equiv 2\sqrt{\ln 2}\,\ell_P \approx 2.6914 \times 10^{-35}\text{ m}$. Orientation-averaged tight-binding cubic lattice dispersion coupled to Mukhanov–Sasaki Riccati–Bogoliubov mode evolution yields an isotropic effective transfer kernel $\mathcal{T}(u) \approx 1 - (\pi^2/40)u^2$ ($u \equiv k/k_c$) and a modulated primordial curvature power spectrum:
  $$\mathcal{P}_{\mathcal{R}}(k) = \mathcal{P}_0(k)\left[1 - \epsilon_{\mathrm{reg}}\left(1 - \mathcal{T}^2(k/k_c)\right)\right]$$
* **Empirical Planck 2018 Fit:** Minimum-variance best-fit log-likelihood $-\ln \mathcal{L}_{\min} = 483.187585$ ($\chi^2_{\mathrm{total}} = 1002.8555$), yielding an absolute global improvement of $\Delta\chi^2 \approx -13.025$ over baseline $\Lambda\mathrm{CDM}$ ($\chi^2_{\Lambda\mathrm{CDM}} = 1015.88$).
* **Ab-Initio Concordance:** Posterior coupling $\epsilon_{\mathrm{reg}} = 0.05498$ ($\epsilon_{\mathrm{reg}} = 0.03666$) confirms the theoretical boundary prediction ($\epsilon_{\mathrm{theory}} \approx 0.0508$) within a $0.15\sigma$ concordance core without post-hoc tuning.
* **Bayesian Model Selection:** Physical single-parameter extension ($N_{\mathrm{param}} = 7$, spatial cutoff $k_c = 0.370 \pm 0.136\text{ Mpc}^{-1}$) achieves decisive information-theoretic support of $\Delta\mathrm{AIC} \approx -10.805$ and $\Delta\mathrm{BIC} \approx -5.0$ relative to continuous $\Lambda\mathrm{CDM}$.
* **Acoustic Tail Damping:** Finite recombination visibility ($\Delta\eta_{\mathrm{rec}} \simeq 150\text{–}250\text{ Mpc}$) smears out microscopic mode ringing, isolating a monotonic power suppression ($-1.48\%$ core window, $-2.34\%$ deep tail) projecting to multipoles $\ell \sim 1800$ directly accessible to CMB-S4, Simons Observatory, and SPT-3G.

---

## 💻 Terminal Execution Commands

### 0. Clone and Install Dependencies

```bash
git clone [https://github.com/dev-s-patel/CAST-Solvers-and-Figures.git](https://github.com/dev-s-patel/CAST-Solvers-and-Figures.git)
cd CAST-Solvers-and-Figures
pip install -r requirements.txt
```

### 1. Appendix A.1: Fourth-Order Runge–Kutta Riccati–Bogoliubov Solver

```bash
python run_cast_rk4_riccati_solver.py
```

* **Script:** `run_cast_rk4_riccati_solver.py`
* **Function:** Numerically integrates the coupled first-order Mukhanov–Sasaki Riccati–Bogoliubov system across $50{,}000$ conformal de Sitter time slices with sub-cycle resolution ($30$ steps/oscillation).
* **Verification:** Validates cycle-averaged adiabatic projection onto the theoretical asymptotic envelope ($|\beta_k^{\mathrm{asymp}}| = 2.28 \times 10^{-4}$) while monitoring symplectic unitarity violation ($||\alpha_k|^2 - |\beta_k|^2 - 1|$) at machine precision ceiling ($\mathcal{O}(10^{-14})$).

### 2. Appendix A.2: 3D Brillouin Zone Density of States Simpson Integrator

```bash
python run_cast_simpson_integrator.py
```

* **Script:** `run_cast_simpson_integrator.py`
* **Function:** Computes the orientation-averaged 3D lattice acoustic dispersion relation over $\mathbb{S}^2$ via 2D composite Simpson quadrature across the first Brillouin zone.
* **Verification:** Validates strict infrared continuum recovery ($\mathcal{T}(u \to 0) = 1.000000026$, absolute error $\le 2.6 \times 10^{-8}$), leading-order Taylor series consistency ($1 - (\pi^2/40)u^2$ with residuals $\mathcal{O}(10^{-6})$), and boundary impedance cutoff damping ($\mathcal{T}_{\mathrm{BZ}} \approx 0.7774$).

### 3. Appendix A.3: Standalone Production CAMB Boltzmann Pipeline

```bash
python run_cast_camb_production.py
```

* **Script:** `run_cast_camb_production.py`
* **Function:** Injects the 3D Brillouin cellular regularizer directly into the CAMB linear Einstein–Boltzmann differential hierarchy via `SplinedInitialPower`, solving full line-of-sight photon transfer functions up to $\ell_{\max} = 5000$.
* **Outputs:**
  * `cast_camb_deficit.pdf` (Publication-grade vector plot)
  * `cast_camb_deficit.png` (High-resolution 300 DPI preview)

### 4. Appendix A.4: Vectorized Cobaya Theory Wrapper

```bash
# Verify component import inside Cobaya ecosystem
python -c "import cast_theory; print('[OK] CREST Cobaya wrapper loaded')"
```

* **Script:** `cast_theory.py`
* **Function:** Subclasses `cobaya.theories.camb.CAMB` to intercept the parameter stream ($k_c, \epsilon_{\mathrm{reg}}$), constructs the regularized primordial spectrum $\mathcal{P}_{\mathcal{R}}(k)$ across a dense $3000$-node wavenumber grid ($k \in [10^{-4}, 5.0]\text{ Mpc}^{-1}$), and overrides CAMB's internal power table while locking non-linear matter corrections to background $n_s$.

### 5. Appendix A.5: Production Cobaya MCMC Sampling & Posterior Contours

```bash
# Execute production MCMC sampling (MPI multi-core ready)
cobaya-run cast_mcmc_run.yaml

# Generate 1D posterior distributions and 2D credible region contours
python plot_contours.py
```

* **Configuration:** `cast_mcmc_run.yaml`
* **Dataset:** Co-added, foreground-marginalized Planck 2018 high-$\ell$ likelihood (`plik_lite_TTTEEE`, $\ell \in [30, 2508]$) + low-$\ell$ Commander temperature ($\ell \in [2, 29]$) + low-$\ell$ SimAll EE polarization ($\ell \in [2, 29]$).
* **Sampling Statistics:** Fast-slow Metropolis–Hastings sampling across $4$ independent chains reaching $37{,}091$ accepted steps ($N_{\mathrm{eff}} = 25{,}347$, Gelman–Rubin convergence $R - 1 = 0.0291 < 0.035$).
* **Outputs:**
  * `cast_planck_contours.pdf` (Vector contour triangle plot)
  * `cast_planck_contours.png` (High-resolution raster plot)

---

## 📦 Requirements

* **Python:** $\ge 3.8$
* `numpy >= 1.24.0`
* `scipy >= 1.10.0`
* `matplotlib >= 3.7.0`
* `camb >= 1.4.0`
* `cobaya >= 3.3.0`
* `getdist >= 1.4.0`
* `pyyaml >= 6.0`

Install all dependencies via `pip`:

```bash
pip install numpy scipy matplotlib camb cobaya getdist pyyaml
```

---

## 📜 Citation

If you use or verify this computational pipeline in your research, please cite the CREST manuscript:

```bibtex
@article{patel2026crest,
  title   = {Cosmological Regularized Effective Spectral Transfer (CREST): An Effective Field Theory of UV-Regularized Cosmological Perturbations, CAMB Implementation, and Planck Likelihood Bounds},
  author  = {Patel, Dev S.},
  year    = {2026},
  month   = {nov},
  doi     = {10.5281/zenodo.23864786},
  url     = {[https://doi.org/10.5281/zenodo.23864786](https://doi.org/10.5281/zenodo.23864786)}
}
```

---

## ⚖️ License

This project is open-source and distributed under the [MIT License](LICENSE).