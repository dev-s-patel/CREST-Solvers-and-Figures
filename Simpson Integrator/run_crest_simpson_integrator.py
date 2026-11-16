import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson


def compute_T_kernel(u_array, nx, ny, nz, sin_theta, theta, phi):
    """Evaluates the 2D spherical Simpson integral of the 3D cubic dispersion kernel."""
    results = np.zeros_like(u_array, dtype=float)
    for idx, u in enumerate(u_array):
        arg_x = 0.5 * np.pi * u * nx
        arg_y = 0.5 * np.pi * u * ny
        arg_z = 0.5 * np.pi * u * nz

        dispersion_sum = np.sin(arg_x) ** 2 + np.sin(arg_y) ** 2 + np.sin(arg_z) ** 2
        v_phase = (2.0 / (np.pi * u)) * np.sqrt(dispersion_sum)
        integrand = v_phase * sin_theta

        integral_phi = simpson(integrand, x=phi, axis=1)
        integral_2D = simpson(integral_phi, x=theta, axis=0)
        results[idx] = integral_2D / (4.0 * np.pi)
    return results


def run_crest_simpson_integrator():
    print("=== CREST PRODUCTION 3D BRILLOUIN ZONE SIMPSON INTEGRATOR ===")
    print("Evaluating Orientation-Averaged Lattice Transfer Kernel...")

    N_theta = 120  # Even intervals for composite Simpson rule
    N_phi = 240
    theta = np.linspace(1e-6, np.pi - 1e-6, N_theta + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, N_phi + 1)

    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    sin_theta = np.sin(THETA)
    nx = np.sin(THETA) * np.cos(PHI)
    ny = np.sin(THETA) * np.sin(PHI)
    nz = np.cos(THETA)

    # 1. Discrete Benchmark Verification Table
    u_benchmarks = np.array([1e-5, 0.10, 0.25, 0.50, 0.75, 1.00])
    T_benchmarks = compute_T_kernel(
        u_benchmarks, nx, ny, nz, sin_theta, theta, phi
    )

    print(
        f"\nIntegrating across {N_theta} x {N_phi} grid points per wavenumber..."
    )
    print("-" * 65)
    print(f"{'Normalized u (k/k_c)':<22} | {'Numerical T(u)':<20} | {'Status'}")
    print("-" * 65)

    for u, T_val in zip(u_benchmarks, T_benchmarks):
        status = (
            "IR Continuum Limit" if u < 1e-4 else "Lattice Dispersion Damping"
        )
        print(f"{u:<22.5f} | {T_val:<20.8f} | {status}")

    print("-" * 65)
    u_test = 0.10
    analytic_u01 = 1.0 - (np.pi**2 / 40.0) * (u_test**2)
    numerical_u01 = T_benchmarks[1]
    discrepancy = np.abs(numerical_u01 - analytic_u01)

    print("\n================ SIMPSON CONVERGENCE REPORT ================")
    print(
        f"Infrared Limit T(u -> 0)     : {T_benchmarks[0]:.10f} (Target: 1.0000000000)"
    )
    print(f"IR Absolute Error            : {np.abs(T_benchmarks[0] - 1.0):.2e}")
    print(f"Order u^2 Analytic Benchmark : {analytic_u01:.8f}")
    print(f"Order u^2 Numerical Result   : {numerical_u01:.8f}")
    print(f"Taylor Consistency Residual  : {discrepancy:.2e}")
    print("============================================================\n")

    # 2. High-Resolution Curve & Residual Evaluation for Figure 2
    print("Generating publication-grade Figure 2 (crest_brillouin_kernel)...")
    u_dense = np.linspace(1e-4, 1.0, 150)
    T_dense = compute_T_kernel(u_dense, nx, ny, nz, sin_theta, theta, phi)

    # Analytic Taylor expansion valid in the IR regime (u <= 0.45)
    u_taylor = np.linspace(1e-4, 0.45, 80)
    T_taylor = 1.0 - (np.pi**2 / 40.0) * (u_taylor**2)

    # Residual evaluation up to u = 0.40
    u_res = np.linspace(0.02, 0.40, 20)
    T_num_res = compute_T_kernel(u_res, nx, ny, nz, sin_theta, theta, phi)
    T_ana_res = 1.0 - (np.pi**2 / 40.0) * (u_res**2)
    residuals = np.abs(T_num_res - T_ana_res)

    # 3. Two-Panel Publication Plot
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.5),
        dpi=300,
        gridspec_kw={"height_ratios": [2.2, 1]},
        sharex=True,
    )

    # Top Panel: Transfer Kernel T(u)
    ax1.plot(
        u_dense,
        T_dense,
        color="#08519c",
        linewidth=2.3,
        label=r"3D Brillouin Simpson Quadrature $\mathcal{T}(u)$",
    )
    ax1.plot(
        u_taylor,
        T_taylor,
        color="#cb181d",
        linestyle="--",
        linewidth=2.0,
        label=r"Analytic Taylor Series: $1 - \frac{\pi^2}{40}u^2$",
    )
    ax1.scatter(
        u_benchmarks,
        T_benchmarks,
        color="#ff7f00",
        edgecolor="black",
        zorder=5,
        s=55,
        label="Verified Numerical Benchmarks",
    )

    # Clean EFT Annotations
    ax1.annotate(
        f"IR Continuum Limit\n{T_benchmarks[0]:.10f}",
        xy=(0.0, T_benchmarks[0]),
        xytext=(0.12, 0.962),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="#eef2ff", edgecolor="#4f46e5"
        ),
    )
    ax1.annotate(
        f"UV Brillouin Boundary\nImpedance Damping: {T_benchmarks[-1]:.4f}",
        xy=(1.0, T_benchmarks[-1]),
        xytext=(0.56, 0.805),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="#fff1f2", edgecolor="#e11d48"
        ),
    )

    ax1.set_ylabel(
        r"Effective Transfer Kernel $\mathcal{T}(u)$",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_title(
        "Isotropic 3D Cubic Lattice Dispersion & UV Boundary Damping",
        fontsize=13,
        pad=10,
        fontweight="bold",
    )
    ax1.set_ylim(0.73, 1.025)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        fontsize=9.5,
    )

    # Bottom Panel: Taylor Series Truncation Residual
    ax2.semilogy(
        u_res,
        residuals,
        color="#238b45",
        marker="s",
        markersize=4.5,
        linewidth=1.6,
        label=r"Taylor Residual $|\mathcal{T}_{\mathrm{num}} - \mathcal{T}_{\mathrm{analytic}}|$",
    )
    ax2.set_xlabel(
        r"Normalized Wavenumber $u = k/k_c$", fontsize=12, fontweight="bold"
    )
    ax2.set_ylabel("Residual", fontsize=11, fontweight="bold")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(1e-7, 2e-3)
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        fontsize=9.5,
    )

    plt.tight_layout()
    fig.savefig("crest_brillouin_kernel.png", dpi=300)
    fig.savefig("crest_brillouin_kernel.pdf")
    plt.close(fig)
    print(
        "Saved 'crest_brillouin_kernel.png' and 'crest_brillouin_kernel.pdf' successfully."
    )


if __name__ == "__main__":
    run_crest_simpson_integrator()