"""
Solution to the H1B visa misallocation model based on Hopenhayn (1992).

Converted from Julia to Python. Solves a dynamic firm problem with heterogeneous
productivity, CES labor aggregation over domestic/foreign workers, and an H1B
visa lottery mechanism. General equilibrium clears the domestic labor market,
pins down the lottery success rate via the visa cap, and satisfies free entry.
"""
#%%
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d
from scipy.stats import beta as beta_dist
from scipy.sparse import csr_matrix
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


#%%─────────────────────────────────────────────────────────────────────
# Model parameters & containers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Primitives:
    """Structural model parameters."""
    beta: float = 0.96          # Discount factor
    phi: float = 0.5            # Returns to scale
    theta: float = 0.3          # CES parameter
    zeta: float = 1.0           # Foreign labor productivity scale
    psi: float = 0.5            # Domestic labor supply elasticity
    psi_bar: float = 1.0        # Domestic labor supply scale
    eta: float = 2.0            # Lottery variance shifter

    kappa_o: float = 0.5        # Fixed cost of operation
    kappa_e: float = 1.0        # Fixed cost of entry
    kappa_f: float = 0.15       # Fixed cost of hiring foreign labor
    delta: float = 0.16         # Visa expiration rate
    F_bar: float = 0.25         # Visa hiring cap


@dataclass
class Simulations:
    """Grid / numerical parameters."""
    z_grid: np.ndarray = field(default_factory=lambda: np.array([0.25, 1.0]))
    P_z: np.ndarray = field(default_factory=lambda: np.array([[0.9, 0.1],
                                                               [0.1, 0.9]]))
    nu_z: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))

    f_max: float = 2.0
    n_f: int = 50
    f_grid: np.ndarray = field(default=None)
    n_nodes: int = 15           # Quadrature nodes for Beta integration
    d_high: float = 100.0       # Upper bound for domestic labor search
    a_tol: float = 1e-4         # Minimum application tolerance

    max_iter: int = 10_000      # VFI / distribution max iterations
    tol: float = 1e-8

    ge_max_iter: int = 50       # GE outer‑loop iterations
    ge_tol: float = 1e-5
    ge_damping: float = 0.382
    W_lower: float = 0.75       # Wage search bounds
    W_upper: float = 2.0
    M_lower: float = 0.01       # Entry mass search bounds
    M_upper: float = 100.0

    def __post_init__(self):
        self.n_z = len(self.z_grid)
        if self.f_grid is None:
            self.f_grid = np.linspace(0.0, self.f_max, self.n_f)


@dataclass
class Results:
    """Mutable container for all solution objects."""
    # Firm problem
    d_policy: np.ndarray        # Static domestic labor policy  (n_f, n_z)
    profit: np.ndarray          # Static profit                 (n_f, n_z)
    x_policy: np.ndarray        # Exit policy (bool)            (n_f, n_z)
    V: np.ndarray               # Value function                (n_f, n_z)
    V_stay: np.ndarray          # Value of staying              (n_f, n_z)
    f_policy: np.ndarray        # Foreign labor target policy   (n_f, n_z)
    a_policy: np.ndarray        # Application policy            (n_f, n_z)
    g: np.ndarray               # Distribution over (f, z)      (n_f, n_z)
    T_star: Optional[csr_matrix] = None  # Transition matrix

    # Quadrature cache: list of (nodes, weights) or None per (i_f, i_f_next)
    E_cache: Optional[list] = None

    # General equilibrium aggregates
    M: float = 1.0              # Mass of entrants
    N: float = 0.5              # Total mass of firms
    D: float = 1.0              # Total domestic labor stock
    F: float = 1.0              # Total foreign labor stock
    A: float = 1.0              # Total applications
    F_minus: float = 1.0        # Foreign labor firings
    mu: float = 0.5             # Lottery mean success rate
    W: float = 1.5              # Wage rate


def initialize(*, beta=0.96, phi=0.5, theta=0.3, zeta=1.0, psi=0.5,
               psi_bar=1.0, eta=1.5, kappa_o=0.5, kappa_e=1.0, kappa_f=0.12,
               delta=0.16, F_bar=0.25,
               z_grid=np.array([0.25, 1.0]),
               P_z=np.array([[0.9, 0.1], [0.1, 0.9]]),
               nu_z=np.array([0.5, 0.5]),
               f_max=2.0, n_f=50, n_nodes=15,
               d_high=100.0, a_tol=1e-4,
               max_iter=10_000, tol=1e-7,
               ge_max_iter=50, ge_tol=1e-5, ge_damping=0.618,
               W_lower=0.75, W_upper=2.0, M_lower=0.01, M_upper=100.0,
               mu_init=0.5):
    """Build and return (Primitives, Simulations, Results) with given params."""

    prim = Primitives(beta=beta, phi=phi, theta=theta, zeta=zeta,
                      psi=psi, psi_bar=psi_bar, eta=eta,
                      kappa_o=kappa_o, kappa_e=kappa_e, kappa_f=kappa_f,
                      delta=delta, F_bar=F_bar)

    sim = Simulations(z_grid=z_grid, P_z=P_z, nu_z=nu_z,
                      f_max=f_max, n_f=n_f, n_nodes=n_nodes,
                      d_high=d_high, a_tol=a_tol,
                      max_iter=max_iter, tol=tol,
                      ge_max_iter=ge_max_iter, ge_tol=ge_tol,
                      ge_damping=ge_damping,
                      W_lower=W_lower, W_upper=W_upper,
                      M_lower=M_lower, M_upper=M_upper)

    n_z = sim.n_z
    res = Results(
        d_policy=np.zeros((n_f, n_z)),
        profit=np.zeros((n_f, n_z)),
        x_policy=np.zeros((n_f, n_z), dtype=bool),
        V=np.zeros((n_f, n_z)),
        V_stay=np.zeros((n_f, n_z)),
        f_policy=np.zeros((n_f, n_z)),
        a_policy=np.zeros((n_f, n_z)),
        g=np.ones((n_f, n_z)) / (n_f * n_z),
        mu=mu_init,
        W=1.5,
    )
    return prim, sim, res


#%%─────────────────────────────────────────────────────────────────────
# Firm's static problem
# ──────────────────────────────────────────────────────────────────────

def labor_index(d, f, prim):
    """CES labor aggregator."""
    return (d ** prim.theta
            + prim.zeta ** (1 - prim.theta) * f ** prim.theta) ** (1.0 / prim.theta)


def compute_profit(z, d, f, W, prim):
    """Revenue minus costs for a single (z, d, f) triple."""
    ell = labor_index(d, f, prim)
    revenue = z * ell ** prim.phi
    cost = W * (d + f) + prim.kappa_o
    return revenue - cost


def solve_static_profit(W, prim, sim):
    """
    For every (f, z) gridpoint, solve for optimal domestic labor d* and
    record the resulting profit π(z, f).
    """
    profit = np.zeros((sim.n_f, sim.n_z))
    d_policy = np.zeros((sim.n_f, sim.n_z))

    for i_z in range(sim.n_z):
        z = sim.z_grid[i_z]
        for i_f in range(sim.n_f):
            f = sim.f_grid[i_f]
            opt = minimize_scalar(
                lambda d: -compute_profit(z, d, f, W, prim),
                bounds=(1e-8, sim.d_high),
                method="bounded",
            )
            if not opt.success and not np.isfinite(opt.fun):
                raise RuntimeError(f"Static optimization failed for z={z}, f={f}")
            d_policy[i_f, i_z] = opt.x
            profit[i_f, i_z] = -opt.fun

    return profit, d_policy


# ──────────────────────────────────────────────────────────────────────
# Gauss–Jacobi quadrature for Beta expectations
# ──────────────────────────────────────────────────────────────────────

def _beta_quadrature(alpha_a, beta_a, n_nodes):
    """
    Return (nodes, weights) on [0, 1] for the Beta(alpha_a, beta_a)
    distribution, using Gauss–Legendre quadrature mapped through the
    Beta CDF (quantile‑based approach matching Julia Expectations.jl).
    """
    # Use Gauss-Legendre nodes on [0,1] mapped through Beta quantile
    # This matches the Julia Expectations.jl approach
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_nodes)
    # Map from [-1, 1] to [0, 1]
    u_nodes = 0.5 * (gl_nodes + 1.0)
    u_weights = 0.5 * gl_weights

    dist = beta_dist(alpha_a, beta_a)
    nodes = dist.ppf(u_nodes)
    # The weights stay as the Gauss-Legendre weights on [0,1],
    # since ∫ h(x) dF(x) = ∫_0^1 h(F^{-1}(u)) du
    weights = u_weights

    return nodes, weights


def build_expectation(prim, sim, res):
    """
    Pre-compute quadrature (nodes, weights) for every feasible
    (f_current → f_next) transition that involves a lottery.
    Returns a list-of-lists; cache[i_f][i_next] is either
    (nodes, weights) or None.
    """
    cache = [[None] * sim.n_f for _ in range(sim.n_f)]

    for i_f in range(sim.n_f):
        f = sim.f_grid[i_f]
        f_decay = (1.0 - prim.delta) * f
        for i_next in range(sim.n_f):
            f_next = sim.f_grid[i_next]
            if f_next > f_decay + sim.a_tol:
                a = f_next - f_decay
                alpha_a = prim.eta * res.mu * a
                beta_a = prim.eta * (1.0 - res.mu) * a
                if alpha_a > 0 and beta_a > 0:
                    nodes, weights = _beta_quadrature(alpha_a, beta_a, sim.n_nodes)
                    cache[i_f][i_next] = (nodes, weights)

    return cache


#%%─────────────────────────────────────────────────────────────────────
# Bellman operator & VFI
# ──────────────────────────────────────────────────────────────────────

def firm_bellman(prim, sim, res):
    """
    One iteration of the Bellman operator.  Returns updated
    (V_next, x_next, f_tilde_next, V_stay_next, a_policy).
    """
    n_f, n_z = sim.n_f, sim.n_z
    f_grid = sim.f_grid
    P_z = sim.P_z

    f_tilde_next = np.zeros((n_f, n_z))
    continuation = np.full((n_f, n_z), -np.inf)

    # Build interpolants for V(·, z') for each z'
    V_interps = [
        interp1d(f_grid, res.V[:, i_z], kind="linear",
                 fill_value="extrapolate")
        for i_z in range(n_z)
    ]

    for i_f in range(n_f):
        f = f_grid[i_f]
        f_decay = (1.0 - prim.delta) * f

        for i_z in range(n_z):
            for i_f_next in range(n_f):
                f_tilde = f_grid[i_f_next]

                # ── Case A: deterministic (firing or natural attrition) ──
                if f_tilde <= f_decay + sim.a_tol:
                    candidate = P_z[i_z, :] @ res.V[i_f_next, :]
                    if candidate > continuation[i_f, i_z]:
                        continuation[i_f, i_z] = candidate
                        f_tilde_next[i_f, i_z] = f_tilde

                # ── Case B: lottery (hiring) ──
                else:
                    a = f_tilde - f_decay
                    quad = res.E_cache[i_f][i_f_next]
                    if quad is None:
                        continue
                    nodes, weights = quad

                    EV = 0.0
                    for i_z_next in range(n_z):
                        # ∫ V(z', f_decay + p·a) dBeta(p)
                        f_realized = f_decay + nodes * a
                        v_vals = V_interps[i_z_next](f_realized)
                        beta_integral = weights @ v_vals
                        EV += P_z[i_z, i_z_next] * beta_integral

                    candidate = EV - prim.kappa_f * a * res.mu
                    if candidate > continuation[i_f, i_z]:
                        continuation[i_f, i_z] = candidate
                        f_tilde_next[i_f, i_z] = f_tilde

    # Exit when continuation value is negative
    x_next = continuation < 0.0
    V_next = res.profit + prim.beta * (~x_next).astype(float) * continuation
    V_stay_next = res.profit + prim.beta * continuation

    # Applications = max(f_tilde - (1-δ)f, 0), zero if exiting
    a_policy = (f_tilde_next
                - (1.0 - prim.delta) * f_grid[:, None]) * (~x_next).astype(float)
    a_policy = np.maximum(a_policy, 0.0)

    return V_next, x_next, f_tilde_next, V_stay_next, a_policy


def VFI(prim, sim, res):
    """Value‑function iteration until convergence."""
    res.profit, res.d_policy = solve_static_profit(res.W, prim, sim)
    res.E_cache = build_expectation(prim, sim, res)

    for it in range(1, sim.max_iter + 1):
        V_next, x_next, f_tilde_next, V_stay_next, a_policy = \
            firm_bellman(prim, sim, res)

        diff = np.max(np.abs(V_next - res.V))
        res.V[:] = V_next
        res.V_stay[:] = V_stay_next
        res.x_policy[:] = x_next
        res.f_policy[:] = f_tilde_next
        res.a_policy[:] = a_policy

        if diff < sim.tol:
            print(f"VFI converged in {it} iterations (diff = {diff:.2e})")
            return

    raise RuntimeError("VFI did not converge within the maximum iterations")


#%%─────────────────────────────────────────────────────────────────────
# Stationary distribution
# ──────────────────────────────────────────────────────────────────────

def build_T_star(prim, sim, res):
    """
    Build the sparse transition matrix T* over the joint state (f, z),
    accounting for exits, deterministic transitions, and lottery draws.
    """
    rows, cols, vals = [], [], []
    f_midpoints = 0.5 * (sim.f_grid[:-1] + sim.f_grid[1:])
    total_states = sim.n_f * sim.n_z

    for i_z in range(sim.n_z):
        for i_f in range(sim.n_f):
            if res.x_policy[i_f, i_z]:
                continue  # exiting firms leave no mass

            source = i_f + i_z * sim.n_f
            f_curr = sim.f_grid[i_f]
            f_decay = (1.0 - prim.delta) * f_curr
            f_target = res.f_policy[i_f, i_z]

            # ── Case A: deterministic transition ──
            if f_target <= f_decay + sim.a_tol:
                idx_f_next = np.searchsorted(sim.f_grid, f_target - 1e-10)
                idx_f_next = min(idx_f_next, sim.n_f - 1)
                for i_z_next in range(sim.n_z):
                    prob_z = sim.P_z[i_z, i_z_next]
                    if prob_z > 0.0:
                        dest = idx_f_next + i_z_next * sim.n_f
                        rows.append(source)
                        cols.append(dest)
                        vals.append(prob_z)

            # ── Case B: lottery transition ──
            else:
                a = f_target - f_decay
                alpha_d = prim.eta * res.mu * a
                beta_d = prim.eta * (1.0 - res.mu) * a
                if alpha_d <= 0 or beta_d <= 0:
                    continue
                dist = beta_dist(alpha_d, beta_d)

                for j_f in range(sim.n_f):
                    bin_lo = -np.inf if j_f == 0 else f_midpoints[j_f - 1]
                    bin_hi = np.inf if j_f == sim.n_f - 1 else f_midpoints[j_f]

                    p_lo = np.clip((bin_lo - f_decay) / a, 0.0, 1.0)
                    p_hi = np.clip((bin_hi - f_decay) / a, 0.0, 1.0)

                    mass_f = 0.0
                    if p_hi > p_lo:
                        mass_f = dist.cdf(p_hi) - dist.cdf(p_lo)

                    if mass_f > 1e-12:
                        for i_z_next in range(sim.n_z):
                            prob_z = sim.P_z[i_z, i_z_next]
                            total_prob = mass_f * prob_z
                            if total_prob > 0.0:
                                dest = j_f + i_z_next * sim.n_f
                                rows.append(source)
                                cols.append(dest)
                                vals.append(total_prob)

    T = csr_matrix((vals, (rows, cols)), shape=(total_states, total_states))
    return T


def solve_distribution(prim, sim, res):
    """
    Iterate g_{t+1} = T*' g_t  +  M · entry  until convergence.
    """
    res.T_star = build_T_star(prim, sim, res)

    # Entry vector: f = 0 (index 0), z drawn from nu_z
    entry_vec = np.zeros(sim.n_f * sim.n_z)
    for i_z in range(sim.n_z):
        entry_vec[0 + i_z * sim.n_f] = sim.nu_z[i_z]

    g_vec = res.g.ravel(order="F").copy()  # column-major to match (f, z) indexing

    for it in range(1, sim.max_iter + 1):
        # T_star is (source, dest); we need T_star^T @ g  (i.e. propagate mass)
        g_next_vec = res.T_star.T @ g_vec + res.M * entry_vec

        diff = np.max(np.abs(g_next_vec - g_vec))
        g_vec[:] = g_next_vec

        if diff < sim.tol:
            res.g[:] = g_vec.reshape((sim.n_f, sim.n_z), order="F")
            res.N = res.g.sum()
            print(f"Distribution converged in {it} iters "
                  f"(diff = {diff:.2e}, M = {res.M:.4f})")
            return

    res.g[:] = g_vec.reshape((sim.n_f, sim.n_z), order="F")
    res.N = res.g.sum()
    print("Distribution did NOT converge within maximum iterations")


#%%─────────────────────────────────────────────────────────────────────
# General equilibrium
# ──────────────────────────────────────────────────────────────────────

def _dot_matrices(A, B):
    """Element-wise product summed (Frobenius inner product), matching Julia's ⋅."""
    return np.sum(A * B)


def solve_GE(prim, sim, res):
    """
    Outer loop: bisect on μ.
    For each μ candidate:
      1. Find W satisfying free entry  E_ν[V(z, 0)] = κ_e
      2. Find M clearing the domestic labor market
      3. Compute implied μ from visa cap / total applications
    """
    mu_lower = 0.0
    mu_upper = 1.0

    for it_ge in range(1, sim.ge_max_iter + 1):
        res.mu = 0.5 * (mu_lower + mu_upper)
        print(f"\nGE iter {it_ge}: μ = {res.mu:.4f}, "
              f"bounds = [{mu_lower:.4f}, {mu_upper:.4f}]")

        # ── Step 1: find W via free entry ──
        def entry_residual(log_W):
            res.W = np.exp(log_W)
            VFI(prim, sim, res)
            ev = res.V[0, :] @ sim.nu_z
            return (ev - prim.kappa_e) ** 2

        opt_W = minimize_scalar(entry_residual,
                                bounds=(np.log(sim.W_lower), np.log(sim.W_upper)),
                                method="bounded")
        if not opt_W.success and not np.isfinite(opt_W.fun):
            raise RuntimeError(f"Wage optimisation failed at GE iter {it_ge}")
        res.W = np.exp(opt_W.x)
        VFI(prim, sim, res)  # re‑solve at optimal W

        # ── Step 2: find M clearing domestic labor market ──
        def labor_residual(log_M):
            res.M = np.exp(log_M)
            solve_distribution(prim, sim, res)
            D_demand = _dot_matrices(res.d_policy, res.g)
            D_supply = prim.psi_bar * res.W ** prim.psi
            return (D_demand - D_supply) ** 2

        opt_M = minimize_scalar(labor_residual,
                                bounds=(np.log(sim.M_lower), np.log(sim.M_upper)),
                                method="bounded")
        if not opt_M.success and not np.isfinite(opt_M.fun):
            raise RuntimeError(f"Labor market opt failed at GE iter {it_ge}")
        res.M = np.exp(opt_M.x)
        solve_distribution(prim, sim, res)

        # ── Step 3: compute aggregates & implied μ ──
        res.A = _dot_matrices(res.a_policy, res.g)
        res.D = _dot_matrices(res.d_policy, res.g)
        res.F = sim.f_grid @ res.g.sum(axis=1)

        mu_implied = min(prim.F_bar / res.A, 1.0) if res.A > 0 else 1.0
        mu_residual = mu_implied - res.mu
        mu_error = abs(mu_residual)

        print(f"  W={res.W:.3f}, M={res.M:.3f}, A={res.A:.2f}, "
              f"μ_impl={mu_implied:.4f}, residual={mu_residual:.6f}")

        if mu_error < sim.ge_tol:
            print(f"✓ GE converged in {it_ge} iterations")
            return

        # ── Step 4: bisection update ──
        if mu_residual > 0:
            mu_lower = res.mu
        else:
            mu_upper = res.mu

        if (mu_upper - mu_lower) < sim.ge_tol:
            print(f"✓ GE converged (bounds collapsed) in {it_ge} iterations")
            return

    raise RuntimeError(f"GE did not converge within {sim.ge_max_iter} iterations")


#%%─────────────────────────────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    prim, sim, res = initialize(
        phi=0.6, theta=0.4, kappa_o=0.5, kappa_f=0.4, F_bar=0.15,
        n_f=50, f_max=5.0, ge_tol=1e-5, n_nodes=10, mu_init=0.5,
        max_iter=20_000, ge_max_iter=100,
        W_lower=0.1, W_upper=2.0, M_lower=0.01, M_upper=10.0,
    )

    # Inner loop
    t0 = time.perf_counter()
    VFI(prim, sim, res)
    print(f"VFI time: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    solve_distribution(prim, sim, res)
    print(f"Distribution time: {time.perf_counter() - t0:.2f}s")

    # Outer loop – general equilibrium
    t0 = time.perf_counter()
    solve_GE(prim, sim, res)
    print(f"GE time: {time.perf_counter() - t0:.2f}s")
# %%
# plot the value function, domestic labor policy, foreign labor policy, and application policy
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    f_grid = sim.f_grid
    z_vals = sim.z_grid

    # Value Function
    plt.figure(figsize=(8, 6))
    for i_z, z in enumerate(z_vals):
        plt.plot(f_grid, res.V[:, i_z], label=f'z={z}')
    plt.title('Value Function V(f, z)')
    plt.xlabel('Foreign Labor f')
    plt.ylabel('Value V')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Domestic Labor Policy
    plt.figure(figsize=(8, 6))
    for i_z, z in enumerate(z_vals):
        plt.plot(f_grid, res.d_policy[:, i_z], label=f'z={z}')
    plt.title('Domestic Labor Policy d*(f, z)')
    plt.xlabel('Foreign Labor f')
    plt.ylabel('Domestic Labor d*')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Foreign Labor Target Policy
    plt.figure(figsize=(8, 6))
    for i_z, z in enumerate(z_vals):
        plt.plot(f_grid, res.f_policy[:, i_z], label=f'z={z}')
    plt.title('Foreign Labor Target Policy f~(f, z)')
    plt.xlabel('Foreign Labor f')
    plt.ylabel('Target Foreign Labor f~')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Application Policy
    plt.figure(figsize=(8, 6))
    for i_z, z in enumerate(z_vals):
        plt.plot(f_grid, res.a_policy[:, i_z], label=f'z={z}')
    plt.title('Application Policy a*(f, z)')
    plt.xlabel('Foreign Labor f')
    plt.ylabel('Applications a*')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Exit Policy
    plt.figure(figsize=(8, 6))
    for i_z, z in enumerate(z_vals):
        plt.plot(f_grid, res.x_policy[:, i_z], label=f'z={z}')
    plt.title('Exit Policy x*(f, z)')
    plt.xlabel('Foreign Labor f')
    plt.ylabel('Exit Decision x* (1=exit, 0=stay)')
    plt.legend()
    plt.tight_layout()
    plt.show()
# %%
