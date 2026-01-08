# =========================================================
# FULL WORKING CODE (Google Colab ready)
# Generates ONLY the 5 requested figures, with NO TITLES
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION BLOCK
# =========================================================

CONFIG = {
    # Geometry / grid
    "L": 1.0,          # cell side length (arbitrary units)
    "M": 200,          # grid resolution (M x M pixels)

    # Base optical gain parameters (reference wavelength)
    "a": 0.5,
    "b": 0.07,
    "lambda_ref": 550e-9,  # reference wavelength (m), ~green

    # Saturation (nonlinear PV response)
    "use_saturation": True,
    "alpha_sat": 0.2,      # saturation strength; 0 = linear

    # Spectral model
    "use_spectral": True,
    "lambdas": np.array([450e-9, 550e-9, 700e-9]),  # blue, green, red
    "w_sun":   np.array([0.30,   0.40,   0.30]),    # solar spectrum weights
    "S_lambda":np.array([0.90,   1.00,   0.80]),    # cell QE at each ?

    # Radii for different geometries
    "R_mono":    0.08,  # monodisperse random circular lenses
    "R_mean":    0.08,  # mean radius for polydisperse lognormal
    "sigma_R":   0.30,  # lognormal sigma for polydisperse
    "R_flake":   0.08,  # equivalent radius for flakes
    "R_ordered": 0.08,  # radius for ordered lattice

    # Monte Carlo control
    "n_realizations": 8,    # per N when sweeping
    "base_seed": 42,
}

# =========================================================
# GRID INITIALIZATION
# =========================================================

L = CONFIG["L"]
M = CONFIG["M"]
dx = L / M

x_coords = np.linspace(dx/2, L - dx/2, M)
y_coords = np.linspace(dx/2, L - dx/2, M)
XX, YY = np.meshgrid(x_coords, y_coords, indexing="ij")

SHOW_FIGS = True
OUTPUT_FORMAT = "png"   # you can change to "jpg" if you want

def savefig(basename):
    """Save current figure and optionally show."""
    filename = f"{basename}.{OUTPUT_FORMAT}"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    if SHOW_FIGS:
        plt.show()
    plt.close()
    print(f"Saved: {filename}")
    return filename

# =========================================================
# OPTICAL MODEL: f(k, ?) AND g(f)
# =========================================================

def a_lambda(lam, config=CONFIG):
    return config["a"] * (lam / config["lambda_ref"])**0.1

def b_lambda(lam, config=CONFIG):
    return config["b"] * (lam / config["lambda_ref"])**0.1

def f_k_lambda(k_values, lam, config=CONFIG):
    a_l = a_lambda(lam, config)
    b_l = b_lambda(lam, config)
    return 1.0 + a_l * k_values - b_l * k_values**2

def f_k_ref(k_values, config=CONFIG):
    return 1.0 + config["a"] * k_values - config["b"] * k_values**2

def g_cell(f_loc, config=CONFIG):
    """
    Nonlinear PV response g(f).
    First clip negative gains: f_eff = max(f, 0)
    """
    f_eff = np.maximum(f_loc, 0.0)
    if not config["use_saturation"]:
        return f_eff
    alpha = config["alpha_sat"]
    return f_eff / (1.0 + alpha * f_eff)

def compute_I_over_I0_from_Pk(k_values, Pk, config=CONFIG):
    """
    Global enhancement I/I0 from Pk using spectral + saturation
    (or scalar model if spectral disabled).
    """
    if config["use_spectral"]:
        lambdas = config["lambdas"]
        w_sun = config["w_sun"]
        S_lambda = config["S_lambda"]

        I_num = 0.0
        W_den = 0.0
        for lam, w, S in zip(lambdas, w_sun, S_lambda):
            fk = f_k_lambda(k_values, lam, config)
            fk_eff = g_cell(fk, config)
            I_num += w * S * np.sum(Pk * fk_eff)
            W_den += w * S
        return I_num / W_den
    else:
        fk = f_k_ref(k_values, config)
        fk_eff = g_cell(fk, config)
        return np.sum(Pk * fk_eff)

# =========================================================
# STATISTICS FROM k(x,y)
# =========================================================

def compute_stats(k_grid, config=CONFIG):
    """
    From overlap map k_grid, compute:
      - k_values, Pk
      - I_over_I0
      - mean_k
      - phi (coverage fraction)
      - phi_ben (beneficial fraction using f_ref(k)>1)
    """
    k_flat = k_grid.ravel()
    k_max = int(k_flat.max())

    if k_max == 0:
        return {
            "k_values": np.array([0], dtype=int),
            "Pk": np.array([1.0]),
            "I_over_I0": 1.0,
            "mean_k": 0.0,
            "phi": 0.0,
            "phi_ben": 0.0,
        }

    k_values = np.arange(k_max + 1, dtype=int)
    Pk = np.zeros_like(k_values, dtype=float)

    for i, kv in enumerate(k_values):
        Pk[i] = np.mean(k_flat == kv)

    I_over_I0 = compute_I_over_I0_from_Pk(k_values, Pk, config)
    mean_k = np.sum(Pk * k_values)

    phi = np.mean(k_grid > 0)

    fk_ref_vals = f_k_ref(k_values, config)
    beneficial_mask = fk_ref_vals > 1.0
    phi_ben = np.sum(Pk[beneficial_mask])

    return {
        "k_values": k_values,
        "Pk": Pk,
        "I_over_I0": I_over_I0,
        "mean_k": mean_k,
        "phi": phi,
        "phi_ben": phi_ben,
    }

# =========================================================
# DEPOSITION GEOMETRIES
# =========================================================

def deposit_circular_mono(N, config=CONFIG, seed=None):
    rng = np.random.default_rng(seed)
    k_grid = np.zeros((M, M), dtype=int)
    R = config["R_mono"]

    for _ in range(N):
        x0 = rng.uniform(0, L)
        y0 = rng.uniform(0, L)
        mask = (XX - x0)**2 + (YY - y0)**2 <= R**2
        k_grid[mask] += 1

    return k_grid

def deposit_circular_poly(N, config=CONFIG, seed=None):
    rng = np.random.default_rng(seed)
    k_grid = np.zeros((M, M), dtype=int)
    R_mean = config["R_mean"]
    sigma_R = config["sigma_R"]

    for _ in range(N):
        R_i = rng.lognormal(mean=np.log(R_mean), sigma=sigma_R)
        x0 = rng.uniform(0, L)
        y0 = rng.uniform(0, L)
        mask = (XX - x0)**2 + (YY - y0)**2 <= R_i**2
        k_grid[mask] += 1

    return k_grid

def deposit_flakes(N, config=CONFIG, seed=None):
    rng = np.random.default_rng(seed)
    k_grid = np.zeros((M, M), dtype=int)
    R_equiv = config["R_flake"]

    for _ in range(N):
        x0 = rng.uniform(0, L)
        y0 = rng.uniform(0, L)
        theta = rng.uniform(0, 2*np.pi)

        a_e = rng.uniform(0.7, 1.3) * R_equiv
        area_target = np.pi * R_equiv**2
        b_e = area_target / (np.pi * a_e)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        X_shift = XX - x0
        Y_shift = YY - y0
        u = X_shift * cos_t + Y_shift * sin_t
        v = -X_shift * sin_t + Y_shift * cos_t

        mask = (u / a_e)**2 + (v / b_e)**2 <= 1.0
        k_grid[mask] += 1

    return k_grid

def deposit_circular_ordered(N, config=CONFIG):
    k_grid = np.zeros((M, M), dtype=int)
    if N <= 0:
        return k_grid

    R = config["R_ordered"]

    n_side = int(np.sqrt(N))
    if n_side * n_side < N:
        n_side += 1

    xs = np.linspace(0, L, n_side, endpoint=False) + L / (2 * n_side)
    ys = np.linspace(0, L, n_side, endpoint=False) + L / (2 * n_side)

    centers = [(x, y) for x in xs for y in ys]

    for (x0, y0) in centers[:N]:
        mask = (XX - x0)**2 + (YY - y0)**2 <= R**2
        k_grid[mask] += 1

    return k_grid

def deposit_lenses(N, geom="random_mono", config=CONFIG, seed=None):
    if N <= 0:
        return np.zeros((M, M), dtype=int)

    if geom == "random_mono":
        return deposit_circular_mono(N, config, seed)
    elif geom == "random_poly":
        return deposit_circular_poly(N, config, seed)
    elif geom == "flakes":
        return deposit_flakes(N, config, seed)
    elif geom == "ordered":
        return deposit_circular_ordered(N, config)
    else:
        raise ValueError(f"Unknown geometry: {geom}")

# =========================================================
# SWEEP OVER N
# =========================================================

def sweep_N_values(N_values, geom="random_mono", config=CONFIG):
    N_values = np.array(N_values, dtype=int)
    n_real = config["n_realizations"]
    base_seed = config["base_seed"]

    I_mean_list = []
    I_std_list = []
    mean_k_list = []
    phi_list = []
    phi_ben_list = []

    for idx, N in enumerate(N_values):
        I_reps = []
        mean_k_reps = []
        phi_reps = []
        phi_ben_reps = []

        for r in range(n_real):
            seed = base_seed + 1000 * idx + r
            k_grid = deposit_lenses(N, geom=geom, config=config, seed=seed)
            stats = compute_stats(k_grid, config)

            I_reps.append(stats["I_over_I0"])
            mean_k_reps.append(stats["mean_k"])
            phi_reps.append(stats["phi"])
            phi_ben_reps.append(stats["phi_ben"])

        I_reps = np.array(I_reps)
        mean_k_reps = np.array(mean_k_reps)
        phi_reps = np.array(phi_reps)
        phi_ben_reps = np.array(phi_ben_reps)

        I_mean_list.append(I_reps.mean())
        I_std_list.append(I_reps.std(ddof=1) if len(I_reps) > 1 else 0.0)
        mean_k_list.append(mean_k_reps.mean())
        phi_list.append(phi_reps.mean())
        phi_ben_list.append(phi_ben_reps.mean())

    return {
        "N_values": N_values,
        "I_over_I0_mean": np.array(I_mean_list),
        "I_over_I0_std": np.array(I_std_list),
        "mean_k": np.array(mean_k_list),
        "phi": np.array(phi_list),
        "phi_ben": np.array(phi_ben_list),
    }

# =========================================================
# ONLY THE 5 REQUESTED PLOTS (NO TITLES)
# =========================================================

def plot_I_vs_phi(res, basename="fig_enhancement_vs_phi"):
    plt.figure(figsize=(6, 4))
    plt.plot(res["phi"], res["I_over_I0_mean"], marker="o")

    idx_opt = np.argmax(res["I_over_I0_mean"])
    phi_opt = res["phi"][idx_opt]
    I_opt = res["I_over_I0_mean"][idx_opt]
    plt.axvline(phi_opt, linestyle=":", linewidth=1)
    plt.scatter([phi_opt], [I_opt])

    plt.xlabel("Coverage fraction $\\phi$")
    plt.ylabel("Enhancement factor $I/I_0$")
    savefig(basename)

def plot_k_map(N, geom, basename="fig_k_map_N160_random_mono"):
    k_grid = deposit_lenses(N, geom=geom, config=CONFIG, seed=5678)

    plt.figure(figsize=(5, 4))
    im = plt.imshow(
        k_grid.T, origin="lower", extent=[0, L, 0, L],
        interpolation="nearest", aspect="equal"
    )
    plt.colorbar(im, label="Overlap depth $k$")
    plt.xlabel("x")
    plt.ylabel("y")
    savefig(basename)

def plot_gain_function(basename="fig_local_gain_clipped"):
    k = np.arange(0, 15)
    f_raw = f_k_ref(k, CONFIG)
    f_clip = np.maximum(f_raw, 0.0)
    gf = g_cell(f_raw, CONFIG)

    plt.figure(figsize=(6, 4))
    plt.plot(k, f_raw, marker="o", label="$f_{\\rm raw}(k)$")
    plt.plot(k, f_clip, marker="s", label="$f_{\\rm clip}(k)=\\max(f,0)$")
    plt.plot(k, gf, marker="^", label="$g(f_{\\rm clip}(k))$")
    plt.axhline(1.0, linestyle="--", linewidth=1, label="=1")
    plt.xlabel("Overlap depth $k$")
    plt.ylabel("Gain")
    plt.legend()
    savefig(basename)

def plot_Pk_evolution(N_list, geom="random_mono",
                      basename="fig_Pk_vs_N_random_mono"):
    plt.figure(figsize=(6, 4))
    for N in N_list:
        k_grid = deposit_lenses(N, geom=geom, config=CONFIG, seed=100 + N)
        stats = compute_stats(k_grid, CONFIG)
        k = stats["k_values"]
        Pk = stats["Pk"]
        mask = Pk > 1e-4
        plt.plot(k[mask], Pk[mask], marker="o", linestyle="-", label=f"N={N}")

    plt.xlabel("Overlap depth $k$")
    plt.ylabel("$P_k$")
    plt.legend()
    savefig(basename)

def plot_spectral_local_gain_only(N_example=160, geom="random_mono",
                                  basename="spectral_optimum_fkl"):
    k_grid = deposit_lenses(N_example, geom=geom, config=CONFIG, seed=10101)
    stats = compute_stats(k_grid, CONFIG)
    k_values = stats["k_values"]

    lambdas = CONFIG["lambdas"]
    lam_nm = lambdas * 1e9

    plt.figure(figsize=(6, 4))
    for lam, lam_label in zip(lambdas, lam_nm):
        fk = f_k_lambda(k_values, lam, CONFIG)
        fk_clip = np.maximum(fk, 0.0)
        plt.plot(k_values, fk_clip, marker="o", linestyle="-",
                 label=f"? = {lam_label:.0f} nm")

    plt.axhline(1.0, linestyle="--", linewidth=1, label="=1")
    plt.xlabel("Overlap depth $k$")
    plt.ylabel("Local gain $f_{\\rm clip}(k,\\lambda)$")
    plt.legend()
    savefig(basename)

# =========================================================
# MAIN EXECUTION (Colab cell)
# =========================================================

# 1) Sweep and plot Enhancement vs coverage phi
N_values = np.arange(0, 341, 20)
res_random = sweep_N_values(N_values, geom="random_mono", config=CONFIG)
plot_I_vs_phi(res_random, basename="fig_enhancement_vs_phi")

# 2) k-map for N = 160
N_show = 160
plot_k_map(N_show, "random_mono", basename="fig_k_map_N160_random_mono")

# 3) Local gain and nonlinear response (clipped)
plot_gain_function(basename="fig_local_gain_clipped")

# 4) Evolution of Pk with N: N=40,160,200
plot_Pk_evolution([40, 160, 200], geom="random_mono",
                  basename="fig_Pk_vs_N_random_mono")

# 5) Spectral local gain (clipped) (ONLY f(k,?) figure)
plot_spectral_local_gain_only(N_example=N_show, geom="random_mono",
                              basename="spectral_optimum_fkl")

print("Done: generated only the 5 requested figures (no titles).")
