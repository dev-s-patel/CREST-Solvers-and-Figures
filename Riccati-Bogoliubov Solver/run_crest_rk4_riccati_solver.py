import numpy as np
import matplotlib.pyplot as plt


def generate_riccati_emergence_plot(
    eta_h_arr,
    beta_arr,
    err_arr,
    analytic_beta,
    filename_prefix="crest_riccati_emergence",
):
    """Generates the two-panel Figure 1 publication plot from RK4 trajectory data."""
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.5, 6.8),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.14},
    )

    # --- Top Panel: Bogoliubov Excitation Trajectory ---
    scale = 1e4
    ax1.plot(
        eta_h_arr,
        beta_arr * scale,
        color="#2b6cb0",
        linewidth=0.85,
        alpha=0.92,
        label=r"Numerical $|\beta_k(\eta)|$ (RK4 Explicit Trajectory)",
    )
    ax1.axhline(
        analytic_beta * scale,
        color="#b83227",
        linestyle="--",
        linewidth=1.8,
        label=rf"Analytic Adiabatic Envelope: $|\beta_k^{{\mathrm{{asymp}}}}| = {analytic_beta:.2e}$",
    )
    ax1.axvline(
        -1.0,
        color="#4a5568",
        linestyle=":",
        linewidth=1.4,
        label=r"UV Horizon Threshold ($\eta H_* = -1.0$)",
    )

    # Callout Annotations
    ax1.annotate(
        "UV Emergence\nHigh-Frequency Oscillations",
        xy=(-1.05, 1.55),
        xytext=(-1.18, 2.55),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#e6e6fa", edgecolor="#4a5568", lw=1.0
        ),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )
    ax1.annotate(
        "Adiabatic Settling Window",
        xy=(-0.35, analytic_beta * scale),
        xytext=(-0.55, 0.85),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#fdf2f2", edgecolor="#4a5568", lw=1.0
        ),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )

    ax1.set_ylim(-0.2, 3.25)
    ax1.set_ylabel(
        r"Excitation $|\beta_k(\eta)| \times 10^4$", fontsize=11.5, fontweight="bold"
    )
    ax1.set_title(
        "Mukhanov-Sasaki Riccati-Bogoliubov Mode Emergence across UV Boundary",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=9.2)
    ax1.grid(True, linestyle=":", alpha=0.65)

    # --- Bottom Panel: Symplectic Unitarity Conservation ---
    err_safe = np.maximum(err_arr, 1e-16)
    ax2.semilogy(
        eta_h_arr,
        err_safe,
        color="#2f855a",
        linewidth=0.9,
        label=r"Symplectic Unitarity Violation $\left|\,|\alpha_k|^2 - |\beta_k|^2 - 1\,\right|$",
    )
    ax2.axhline(
        1e-14,
        color="#4a5568",
        linestyle="--",
        linewidth=1.1,
        label=r"Machine Epsilon Ceiling $\mathcal{O}(10^{-14})$",
    )

    ax2.set_xlim(-1.2, -0.2)
    ax2.set_ylim(1e-16, 1e-12)
    ax2.set_ylabel("Unitary Error", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel(
        r"Conformal Inflationary Time $\eta \cdot H_*$", fontsize=11.5, fontweight="bold"
    )
    ax2.legend(loc="lower left", frameon=True, framealpha=0.94, fontsize=9.0)
    ax2.grid(True, which="major", linestyle=":", alpha=0.65)

    plt.tight_layout()
    fig.savefig(f"{filename_prefix}.png", dpi=300)
    fig.savefig(f"{filename_prefix}.pdf")
    plt.close(fig)
    print(f"Saved publication figures: {filename_prefix}.pdf & {filename_prefix}.png")


def run_crest_rk4_riccati_solver():
    print("=== CREST PRODUCTION RK4 RICCATI-BOGOLIUBOV SOLVER ===")
    print("Resolving UV Regularizer Mode Emergence with Sub-Cycle Stepping...")

    # Scaled Benchmark Parameters for Exact Numerical Resolution
    H_star = 1.0e-3
    M_cell = 1.0
    u_val = 0.50  # Normalized wavenumber u = k / k_c

    # Boundary crossing at a_0 = 1.0; eta_0 = -1 / H_star
    eta_in = -1.2 / H_star   # Prior to emergence threshold
    eta_out = -0.2 / H_star  # Asymptotic post-emergence regime

    # Explicit sub-cycle RK4 stepping (>= 30 steps per oscillation)
    N_steps = 50000
    d_eta = (eta_out - eta_in) / N_steps

    def dispersion(eta):
        arg = -0.5 * H_star * u_val * eta
        arg = np.clip(arg, 0.01, np.pi / 2 - 0.01)
        omega = 2.0 * M_cell * np.sin(arg)
        d_omega = -H_star * u_val * M_cell * np.cos(arg)
        return omega, d_omega

    Y = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

    def derivatives(eta, state):
        alpha, beta, theta = state[0], state[1], state[2].real
        omega, d_omega = dispersion(eta)
        g = d_omega / (2.0 * omega)
        d_alpha = g * beta * np.exp(2.0j * theta)
        d_beta = g * alpha * np.exp(-2.0j * theta)
        d_theta = omega + 0.0j
        return np.array([d_alpha, d_beta, d_theta], dtype=complex)

    # Pre-allocate trajectory arrays for plotting
    eta_hist = np.empty(N_steps)
    beta_hist = np.empty(N_steps)
    unitary_err_hist = np.empty(N_steps)

    print(f"Executing RK4 across {N_steps} slices (Step size d_eta = {d_eta:.4f})...")
    eta = eta_in
    for step in range(N_steps):
        k1 = derivatives(eta, Y)
        k2 = derivatives(eta + 0.5 * d_eta, Y + 0.5 * d_eta * k1)
        k3 = derivatives(eta + 0.5 * d_eta, Y + 0.5 * d_eta * k2)
        k4 = derivatives(eta + d_eta, Y + d_eta * k3)
        Y += (d_eta / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        eta += d_eta

        # Record step diagnostics
        eta_hist[step] = eta * H_star
        beta_hist[step] = np.abs(Y[1])
        unitary_err_hist[step] = np.abs((np.abs(Y[0]) ** 2 - np.abs(Y[1]) ** 2) - 1.0)

    alpha_final = Y[0]
    beta_final = Y[1]
    norm_check = np.abs(alpha_final)**2 - np.abs(beta_final)**2
    norm_error = np.abs(norm_check - 1.0)

    sinc_u = np.sinc(u_val)
    analytic_beta = (H_star / M_cell) * np.abs(1.0 - sinc_u) / (2.0 * np.sqrt(sinc_u))

    print("\n================ RK4 CONVERGENCE REPORT ================")
    print(f"Final |alpha_k|             : {np.abs(alpha_final):.8f}")
    print(f"Final |beta_k| (Numerical)  : {np.abs(beta_final):.6e}")
    print(f"Asymptotic Modulus (Theory) : {analytic_beta:.6e}")
    print(f"Symplectic Invariance Norm  : {norm_check.real:.12f} (Target: 1.000000000000)")
    print(f"Unitary Violation Error     : {norm_error:.2e}")
    print("========================================================\n")

    # Link Plot Generator
    generate_riccati_emergence_plot(
        eta_hist, beta_hist, unitary_err_hist, analytic_beta
    )


if __name__ == '__main__':
    run_crest_rk4_riccati_solver()