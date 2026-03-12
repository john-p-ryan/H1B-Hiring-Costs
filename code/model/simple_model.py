#%%
import numpy as np
from scipy.optimize import root, least_squares
from scipy.integrate import quad
from scipy.stats import expon


#%%
################################################
# Model parameters
################################################
psi = 0.5  # labor supply elasticity
psi_bar = 1.0  # weight on labor disutility
phi = 0.5  # output elasticity of labor
mf = 0.3  # foreign labor supply
kappa_o = 0.2  # fixed cost of operation
kappa_e = 0.3  # fixed cost of entry
kappa_f = 0.2  # fixed cost of foreign labor market access
theta = 0.6  # CES parameter
xi = 0.7  # productivity weight on foreign labor

# Distribution of firm productivities (Exponential with scale=2.0)
dist_scale = 2.0

#%%
################################################
# Define supply and demand functions
################################################

def N(w):
    """Labor supply function"""
    return psi_bar * w**psi

def l(z, w):
    """Labor demand for domestic-only firms"""
    return (phi * z / w)**(1 / (1 - phi))

def pi_d(z, w_d):
    """Profit for domestic-only firms"""
    return z * l(z, w_d)**phi - w_d * l(z, w_d) - kappa_o

def W(w_d, w_f):
    """Composite wage index"""
    return (w_d**(-theta/(1-theta)) + xi * w_f**(-theta/(1-theta)))**(-(1-theta)/theta)

def d(z, w_d, w_f):
    """Domestic labor demand for firms hiring both"""
    return (w_d / W(w_d, w_f))**(-1/(1-theta)) * l(z, W(w_d, w_f))

def f(z, w_d, w_f):
    """Foreign labor demand for firms hiring both"""
    return (w_f / W(w_d, w_f))**(-1/(1-theta)) * xi * l(z, W(w_d, w_f))

def pi_f(z, w_d, w_f):
    """Profit for firms hiring both domestic and foreign"""
    return (z * l(z, W(w_d, w_f))**phi - 
            w_d * d(z, w_d, w_f) - 
            w_f * f(z, w_d, w_f) - 
            kappa_o - kappa_f)

def V(z, w_d, w_f):
    """Value function: max of 0, domestic profit, foreign profit"""
    return max(0.0, pi_d(z, w_d), pi_f(z, w_d, w_f))

def z_lower(w_d):
    """Exit cutoff for domestic firms (where pi_d = 0)"""
    return (kappa_o / (1-phi))**(1-phi) * (w_d/phi)**phi

def z_upper(w_d, w_f):
    """Foreign hiring cutoff (where pi_d = pi_f)"""
    from scipy.optimize import minimize_scalar
    def obj(z):
        return (pi_f(z, w_d, w_f) - pi_d(z, w_d))**2
    result = minimize_scalar(obj, bounds=(1e-6, 20.0), method='bounded')
    return result.x

def d_demand(z, w_d, w_f):
    """Domestic labor demand by productivity level"""
    z_low = z_lower(w_d)
    z_high = z_upper(w_d, w_f)
    
    if z < z_low:
        return 0.0
    elif z < z_high:
        return l(z, w_d)
    else:
        return d(z, w_d, w_f)

def f_demand(z, w_d, w_f):
    """Foreign labor demand by productivity level"""
    z_high = z_upper(w_d, w_f)
    
    if z < z_high:
        return 0.0
    else:
        return f(z, w_d, w_f)

def profit(z, w_d, w_f):
    """Profit by productivity level"""
    z_low = z_lower(w_d)
    z_high = z_upper(w_d, w_f)
    
    if z < z_low:
        return 0.0
    elif z < z_high:
        return pi_d(z, w_d)
    else:
        return pi_f(z, w_d, w_f)


#%%
################################################
# Expectation operator using numerical integration
################################################

def expectation(func, w_d, w_f):
    """Compute expectation over exponential distribution"""
    def integrand(z):
        return func(z, w_d, w_f) * expon.pdf(z, scale=dist_scale)
    
    # Integrate from 0 to a large value (99.9th percentile of distribution)
    upper_limit = expon.ppf(0.999, scale=dist_scale)
    result, _ = quad(integrand, 0, upper_limit, limit=100)
    return result


#%%
################################################
# Equilibrium system of equations
################################################

def equilibrium_equations(x):
    """System of equations that should equal zero at equilibrium"""
    w_d, w_f, M = x
    
    # Ensure positive values
    if w_d <= 0 or w_f <= 0 or M <= 0:
        return np.array([1e10, 1e10, 1e10])
    
    try:
        # Compute expected values using numerical integration
        expected_profit = expectation(profit, w_d, w_f)
        expected_d_demand = expectation(d_demand, w_d, w_f)
        expected_f_demand = expectation(f_demand, w_d, w_f)
        
        # Three equilibrium conditions (should all equal zero)
        eq1 = expected_profit - kappa_e  # Free entry condition
        eq2 = N(w_d) - M * expected_d_demand  # Domestic labor market clearing
        eq3 = N(w_f) * mf - M * expected_f_demand  # Foreign labor market clearing
        
        return np.array([eq1, eq2, eq3])
    
    except:
        return np.array([1e10, 1e10, 1e10])


def equilibrium_equations_scaled(x):
    """Scaled version for better numerical performance"""
    w_d, w_f, M = x
    
    if w_d <= 0 or w_f <= 0 or M <= 0:
        return np.array([1e10, 1e10, 1e10])
    
    try:
        expected_profit = expectation(profit, w_d, w_f)
        expected_d_demand = expectation(d_demand, w_d, w_f)
        expected_f_demand = expectation(f_demand, w_d, w_f)
        
        # Scale by typical magnitudes to improve conditioning
        eq1 = (expected_profit - kappa_e) / kappa_e
        
        supply_d = N(w_d)
        if supply_d > 1e-8:
            eq2 = (supply_d - M * expected_d_demand) / supply_d
        else:
            eq2 = supply_d - M * expected_d_demand
        
        supply_f = N(w_f) * mf
        if supply_f > 1e-8:
            eq3 = (supply_f - M * expected_f_demand) / supply_f
        else:
            eq3 = supply_f - M * expected_f_demand
        
        return np.array([eq1, eq2, eq3])
    
    except:
        return np.array([1e10, 1e10, 1e10])

#%%
################################################
# Solve for general equilibrium with early stopping
################################################

def solve_equilibrium(use_scaled=False, convergence_tol=1e-6):
    """
    Solve equilibrium using multiple methods and initial guesses.
    Stops as soon as a successfully converged solution is found.
    
    Parameters:
    -----------
    use_scaled : bool
        Whether to use scaled equations
    convergence_tol : float
        Tolerance for considering solution converged (max absolute residual)
    
    Returns:
    --------
    best : dict
        The converged solution
    """
    
    # Choose equation system
    eq_func = equilibrium_equations_scaled if use_scaled else equilibrium_equations
    
    # Multiple initial guesses to avoid local solutions
    initial_guesses = [
        np.array([15.0, 15.0, 150.0]),
        np.array([10.0, 10.0, 100.0]),
        np.array([5.0, 8.0, 50.0]),
        np.array([2.0, 3.0, 10.0]),
    ]
    
    # Methods to try
    methods = ['hybr', 'lm']
    
    print("Searching for converged solution...")
    print("=" * 70)
    
    # Try root finders
    for method in methods:
        for i, x0 in enumerate(initial_guesses):
            try:
                result = root(
                    eq_func, 
                    x0, 
                    method=method,
                    options={'xtol': 1e-10, 'ftol': 1e-10} if method == 'lm' else {'xtol': 1e-10}
                )
                
                # Evaluate residuals with unscaled equations for comparison
                residuals = equilibrium_equations(result.x)
                max_residual = np.max(np.abs(residuals))
                
                print(f"root-{method}, guess {i+1}: success={result.success}, "
                      f"max_residual={max_residual:.2e}")
                
                # Check if converged
                if result.success and max_residual < convergence_tol:
                    print(f"\n✓ CONVERGED SOLUTION FOUND!")
                    print("=" * 70)
                    return {
                        'method': f'root ({method})',
                        'initial_guess': i+1,
                        'x': result.x,
                        'success': True,
                        'max_residual': max_residual,
                        'residuals': residuals
                    }
                
            except Exception as e:
                print(f"root-{method}, guess {i+1}: FAILED ({str(e)[:50]})")
    
    # Try least_squares with bounds
    for i, x0 in enumerate(initial_guesses):
        try:
            result = least_squares(
                eq_func,
                x0,
                bounds=([0.1, 0.1, 1], [np.inf, np.inf, np.inf]),
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10
            )
            
            # Evaluate residuals with unscaled equations
            residuals = equilibrium_equations(result.x)
            max_residual = np.max(np.abs(residuals))
            
            print(f"least_squares, guess {i+1}: success={result.success}, "
                  f"max_residual={max_residual:.2e}")
            
            # Check if converged
            if result.success and max_residual < convergence_tol:
                print(f"\n✓ CONVERGED SOLUTION FOUND!")
                print("=" * 70)
                return {
                    'method': 'least_squares',
                    'initial_guess': i+1,
                    'x': result.x,
                    'success': True,
                    'max_residual': max_residual,
                    'residuals': residuals
                }
            
        except Exception as e:
            print(f"least_squares, guess {i+1}: FAILED ({str(e)[:50]})")
    
    # If we get here, no solution converged
    print("\n✗ WARNING: No converged solution found with residuals < {:.0e}".format(convergence_tol))
    print("=" * 70)
    return None


# Solve the equilibrium - try unscaled first
print("\nSOLVING WITH UNSCALED EQUATIONS")
print("=" * 70)
best = solve_equilibrium(use_scaled=False)

# If unscaled didn't work, try scaled
if best is None:
    print("\n\nSOLVING WITH SCALED EQUATIONS")
    print("=" * 70)
    best = solve_equilibrium(use_scaled=True)

# Display results if solution found
if best is not None:
    w_d_eq, w_f_eq, M_eq = best['x']

    print("\n" + "=" * 70)
    print("EQUILIBRIUM SOLUTION")
    print("=" * 70)
    print(f"Method: {best['method']} (initial guess {best['initial_guess']})")
    print(f"Domestic wage (w_d):  {w_d_eq:.8f}")
    print(f"Foreign wage (w_f):   {w_f_eq:.8f}")
    print(f"Mass of firms (M):    {M_eq:.8f}")
    print(f"Max residual:         {best['max_residual']:.2e}")
    print("\nProductivity cutoffs:")
    print(f"Exit cutoff (z_lower):     {z_lower(w_d_eq):.8f}")
    print(f"Foreign hiring (z_upper):  {z_upper(w_d_eq, w_f_eq):.8f}")

    # Verify equilibrium conditions
    expected_profit_eq = expectation(profit, w_d_eq, w_f_eq)
    expected_d_demand_eq = expectation(d_demand, w_d_eq, w_f_eq)
    expected_f_demand_eq = expectation(f_demand, w_d_eq, w_f_eq)

    print("\n" + "=" * 70)
    print("EQUILIBRIUM VERIFICATION")
    print("=" * 70)
    print("Free entry condition:")
    print(f"  E[profit] = {expected_profit_eq:.8f} (should equal kappa_e = {kappa_e})")
    print(f"  Residual = {expected_profit_eq - kappa_e:.10f}")

    print("\nDomestic labor market:")
    print(f"  Supply: {N(w_d_eq):.8f}")
    print(f"  Demand: {M_eq * expected_d_demand_eq:.8f}")
    print(f"  Residual = {N(w_d_eq) - M_eq * expected_d_demand_eq:.10f}")

    print("\nForeign labor market:")
    print(f"  Supply: {N(w_f_eq) * mf:.8f}")
    print(f"  Demand: {M_eq * expected_f_demand_eq:.8f}")
    print(f"  Residual = {N(w_f_eq) * mf - M_eq * expected_f_demand_eq:.10f}")

    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("NO CONVERGED SOLUTION FOUND")
    print("=" * 70)
    print("Consider adjusting:")
    print("- Initial guesses")
    print("- Convergence tolerance")
    print("- Model parameters")
    print("=" * 70)


#%%
################################################
# Plotting code for visualization
################################################
import matplotlib.pyplot as plt

# Only proceed with plotting if we found a solution
if best is not None:
    w_d_eq, w_f_eq, M_eq = best['x']
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))
    
    # =====================================================
    # Panel 1: Profit functions across productivity z
    # =====================================================
    ax1 = plt.subplot(2, 3, 1)
    
    # Create range of z values
    z_low = z_lower(w_d_eq)
    z_high = z_upper(w_d_eq, w_f_eq)
    z_range = np.linspace(0.01, min(z_high * 3, expon.ppf(0.95, scale=dist_scale)), 500)
    
    # Calculate profits
    profits_d = [pi_d(z, w_d_eq) for z in z_range]
    profits_f = [pi_f(z, w_d_eq, w_f_eq) for z in z_range]
    profits_actual = [profit(z, w_d_eq, w_f_eq) for z in z_range]
    
    ax1.plot(z_range, profits_d, label='Domestic only π_d', linewidth=2, alpha=0.7)
    ax1.plot(z_range, profits_f, label='Hire foreign π_f', linewidth=2, alpha=0.7)
    ax1.plot(z_range, profits_actual, label='Actual profit (max)', linewidth=3, 
             color='black', linestyle='--')
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax1.axvline(x=z_low, color='red', linestyle='--', alpha=0.5, 
                label=f'Exit cutoff: {z_low:.3f}')
    ax1.axvline(x=z_high, color='green', linestyle='--', alpha=0.5, 
                label=f'Foreign hiring: {z_high:.3f}')
    
    ax1.set_xlabel('Productivity (z)', fontsize=11)
    ax1.set_ylabel('Profit', fontsize=11)
    ax1.set_title('Profit Functions by Productivity Level', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # =====================================================
    # Panel 2: Domestic labor demand across productivity z
    # =====================================================
    ax2 = plt.subplot(2, 3, 2)
    
    labor_d_demand = [d_demand(z, w_d_eq, w_f_eq) for z in z_range]
    
    ax2.plot(z_range, labor_d_demand, linewidth=2, color='steelblue')
    ax2.axvline(x=z_low, color='red', linestyle='--', alpha=0.5, 
                label=f'Exit: {z_low:.3f}')
    ax2.axvline(x=z_high, color='green', linestyle='--', alpha=0.5, 
                label=f'Foreign hiring: {z_high:.3f}')
    ax2.fill_between(z_range, 0, labor_d_demand, alpha=0.3, color='steelblue')
    
    ax2.set_xlabel('Productivity (z)', fontsize=11)
    ax2.set_ylabel('Domestic Labor Demand', fontsize=11)
    ax2.set_title('Domestic Labor Demand by Productivity', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # =====================================================
    # Panel 3: Foreign labor demand across productivity z
    # =====================================================
    ax3 = plt.subplot(2, 3, 3)
    
    labor_f_demand = [f_demand(z, w_d_eq, w_f_eq) for z in z_range]
    
    ax3.plot(z_range, labor_f_demand, linewidth=2, color='darkorange')
    ax3.axvline(x=z_high, color='green', linestyle='--', alpha=0.5, 
                label=f'Foreign hiring: {z_high:.3f}')
    ax3.fill_between(z_range, 0, labor_f_demand, alpha=0.3, color='darkorange')
    
    ax3.set_xlabel('Productivity (z)', fontsize=11)
    ax3.set_ylabel('Foreign Labor Demand', fontsize=11)
    ax3.set_title('Foreign Labor Demand by Productivity', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # =====================================================
    # Panel 4: Domestic labor market (quantity vs wage)
    # =====================================================
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate equilibrium quantity for scaling
    eq_quantity_d = N(w_d_eq)
    
    # Create wage range for domestic market
    w_range = np.linspace(0.1, w_d_eq * 2, 200)
    
    # Supply curve
    supply_d = [N(w) for w in w_range]
    
    # For demand, we need to compute aggregate demand at each wage
    # holding w_f constant at equilibrium and adjusting M to maintain foreign market clearing
    demand_d = []
    for w in w_range:
        # At equilibrium, M adjusts to clear markets
        # For visualization, we can use the equilibrium M and expected demand at this wage
        exp_d = expectation(lambda z, wd, wf: d_demand(z, wd, wf), w, w_f_eq)
        demand_d.append(M_eq * exp_d)
    
    ax4.plot(supply_d, w_range, linewidth=2.5, label='Supply N(w)', color='blue')
    ax4.plot(demand_d, w_range, linewidth=2.5, label='Demand M·E[d]', color='red')
    ax4.axhline(y=w_d_eq, color='green', linestyle='--', linewidth=2, 
                label=f'Equilibrium w_d = {w_d_eq:.3f}')
    ax4.scatter([eq_quantity_d], [w_d_eq], s=200, c='green', marker='*', 
                zorder=5, edgecolors='black', linewidths=1.5,
                label='Equilibrium')
    
    # Set reasonable x-axis limits based on equilibrium quantity
    ax4.set_xlim(0, eq_quantity_d * 2)
    
    ax4.set_xlabel('Quantity of Domestic Labor', fontsize=11)
    ax4.set_ylabel('Domestic Wage (w_d)', fontsize=11)
    ax4.set_title('Domestic Labor Market Equilibrium', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # =====================================================
    # Panel 5: Foreign labor market (quantity vs wage)
    # =====================================================
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate equilibrium quantity for scaling
    eq_quantity_f = N(w_f_eq) * mf
    
    # Create wage range for foreign market
    w_f_range = np.linspace(0.1, w_f_eq * 2, 200)
    
    # Supply curve (scaled by mf)
    supply_f = [N(w) * mf for w in w_f_range]
    
    # Demand curve
    demand_f = []
    for w in w_f_range:
        exp_f = expectation(lambda z, wd, wf: f_demand(z, wd, wf), w_d_eq, w)
        demand_f.append(M_eq * exp_f)
    
    ax5.plot(supply_f, w_f_range, linewidth=2.5, label='Supply N(w)·mf', color='blue')
    ax5.plot(demand_f, w_f_range, linewidth=2.5, label='Demand M·E[f]', color='red')
    ax5.axhline(y=w_f_eq, color='green', linestyle='--', linewidth=2, 
                label=f'Equilibrium w_f = {w_f_eq:.3f}')
    ax5.scatter([eq_quantity_f], [w_f_eq], s=200, c='green', marker='*', 
                zorder=5, edgecolors='black', linewidths=1.5,
                label='Equilibrium')
    
    # Set reasonable x-axis limits based on equilibrium quantity
    ax5.set_xlim(0, eq_quantity_f * 2)
    
    ax5.set_xlabel('Quantity of Foreign Labor', fontsize=11)
    ax5.set_ylabel('Foreign Wage (w_f)', fontsize=11)
    ax5.set_title('Foreign Labor Market Equilibrium', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # =====================================================
    # Panel 6: Productivity distribution with cutoffs
    # =====================================================
    ax6 = plt.subplot(2, 3, 6)
    
    z_pdf_range = np.linspace(0, expon.ppf(0.99, scale=dist_scale), 300)
    pdf_values = expon.pdf(z_pdf_range, scale=dist_scale)
    
    # Color regions differently
    exit_region = z_pdf_range < z_low
    domestic_region = (z_pdf_range >= z_low) & (z_pdf_range < z_high)
    foreign_region = z_pdf_range >= z_high
    
    ax6.fill_between(z_pdf_range[exit_region], 0, pdf_values[exit_region], 
                     alpha=0.4, color='red', label='Exit (no hiring)')
    ax6.fill_between(z_pdf_range[domestic_region], 0, pdf_values[domestic_region], 
                     alpha=0.4, color='steelblue', label='Domestic only')
    ax6.fill_between(z_pdf_range[foreign_region], 0, pdf_values[foreign_region], 
                     alpha=0.4, color='darkorange', label='Hire foreign')
    
    ax6.plot(z_pdf_range, pdf_values, linewidth=2, color='black', alpha=0.7)
    ax6.axvline(x=z_low, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax6.axvline(x=z_high, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax6.set_xlabel('Productivity (z)', fontsize=11)
    ax6.set_ylabel('Probability Density', fontsize=11)
    ax6.set_title('Productivity Distribution and Firm Decisions', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # =====================================================
    # Final adjustments
    # =====================================================
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("PLOTS GENERATED SUCCESSFULLY")
    print("=" * 70)

else:
    print("\n" + "=" * 70)
    print("NO PLOTS GENERATED - No equilibrium solution found")
    print("=" * 70)

#%%