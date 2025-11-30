import os
import asyncio
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

async def n_proof(c, lambda_, n, n_a):
    L = n_a / c
    if L <= n_a:  # enforce domain
        return np.inf
    # Compute logs
    log_c_Ln = np.log(L / n) / np.log(c)  # log_c(L/n) = log_c((n_a/n)/c)
    if log_c_Ln <= 1:  # domain restriction
        return np.inf
    inner = 1 - 1 / log_c_Ln
    if inner <= 0:
        return np.inf
    log_base_0_5 = np.log(inner) / np.log(0.5)
    return L + lambda_ / log_base_0_5

# Define the equation as a function of c
def opt_eq(c, lambda_, n, n_a):
    if c <= 0 or c > 1 or lambda_ <= 0 or n <= 0 or n_a < 0:
        return np.nan

    ln = np.log  # natural logarithm
    
    try:
        log_c = ln(c)
        log_na_c_n = ln(n_a / (c * n))
        
        if log_na_c_n == 0:
            return np.nan
        
        left_term = c**2 * ln(2) * lambda_ * (
            (-log_c) / (c * (log_na_c_n**2)) - 1 / (c * log_na_c_n)
        )
        
        inner_ratio = (-log_c) / log_na_c_n + 1
        if inner_ratio <= 0:
            return np.nan
        
        right_term = n_a * inner_ratio * (np.log(inner_ratio)**2)
        
        return left_term - right_term
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.nan

async def n_proof_ni(c, lambda_, n, n_a):
    L = n_a / c
    if L <= n_a:  # enforce domain
        return np.inf
    # Compute logs
    log_c_Ln = np.log(L / n) / np.log(c)  # log_c(L/n) = log_c((n_a/n)/c)
    if log_c_Ln <= 1:  # domain restriction
        return np.inf
    inner = 1 - 1 / log_c_Ln
    if inner <= 0:
        return np.inf
    log_base_0_5 = np.log(inner) / np.log(0.5)
    return L + (lambda_ - np.log(n*c)/np.log(0.5)) / log_base_0_5

def opt_eq_ni(c, lambda_, n, n_a):
    if c <= 0 or c > 1 or lambda_ <= 0 or n <= 0 or n_a < 0:
        return np.nan

    ln = np.log  # natural logarithm
    
    try:
        # Compute components
        log_na_c_n = ln(n_a / (c * n))
        log_na_c2_n = ln(n_a / (c**2 * n))
        log_log_log = ln(log_na_c2_n / log_na_c_n)

        if log_na_c_n == 0 or log_na_c2_n == 0: # or log_log_log == 0: # in c=0 log_log_log becomes 0 and the derivative is an indefinite form 0/0. However, everything works also if I do not check this case.
            return np.nan
        
        term12 = (log_log_log*n_a + c) * log_log_log * log_na_c_n * log_na_c2_n
        term3 = c * ln(n_a/n) * (lambda_*ln(2) + ln(c*n))
    
        return term12 + term3
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.nan

# Function to find optimal c for given parameters
async def find_opt_c(eq, lambda_, n, n_a):
    def f(c):
        return eq(c, lambda_, n, n_a)
    
    try:
        result = root_scalar(f, bracket=[1e-8, 1], method='brentq')
        return result.root if result.converged else None
    except ValueError:
        return None

async def verify_opt(n_proof_func, lambda_, n, n_a, c_opt, radius=0.1, num_points=50):
    """
    Evaluates a given function on a neighborhood around a point.
    
    Parameters:
    func (callable): The function to evaluate. Must accept a NumPy array or scalar.
    point (float): The center point of the neighborhood.
    radius (float): The radius around the point to evaluate.
    num_points (int): Number of points in the neighborhood.
    
    Returns:
    tuple: (x_values, y_values) where x_values are the points and y_values are the function evaluations.
    """
    # Create neighborhood points
    c_values = np.linspace(c_opt - radius, c_opt + radius, num_points)
    
    # Evaluate the function on these points
    n_proof_opt = await n_proof_func(c_opt, lambda_, n, n_a)
    
    for c in c_values:
        L = n_a/c
        if await n_proof_func(L, lambda_, n, n_a) < n_proof_opt:
            return False
    
    return True

lambda_ = 50.0
START_HEIGHT = 904000
CHAINTIP = 3154000
STEP = 25000
n_a_values = [50.0 * (i+1) for i in range(5)]
file_path = "experiments/na_values.csv"

async def main():
    df_list : list[pd.DataFrame] = list()
    heights = [i for i in range(START_HEIGHT, CHAINTIP, STEP)]
    for n_a in n_a_values:
        c_values = await asyncio.gather(
            *[find_opt_c(opt_eq, lambda_, n, n_a) for n in heights]
        )
        c_values_ni = await asyncio.gather(
            *[find_opt_c(opt_eq_ni, lambda_, n, n_a) for n in heights]
        )
        L_values = [n_a / c for c in c_values]
        L_values_ni = [n_a / c for c in c_values_ni]
        n_samples = await asyncio.gather(
            *[n_proof(c_opt, lambda_, n, n_a) for n, c_opt in zip(heights, c_values, strict=True)]
        )
        n_samples_ni = await asyncio.gather(
            *[n_proof_ni(c_opt, lambda_, n, n_a) for n, c_opt in zip(heights, c_values_ni, strict=True)]
        )
        m_values = [tot - L for tot, L in zip(n_samples, L_values, strict=True)]
        m_values_ni = [tot - L for tot, L in zip(n_samples_ni, L_values_ni, strict=True)]
        
        for n, c_opt, L_opt, tot_opt, m_opt in zip(heights, c_values, L_values, n_samples, m_values, strict=True):
            print(f"lambda={lambda_}, n={n}, n_a={n_a}, interactive:")
            if not await verify_opt(n_proof, lambda_, n, n_a, c_opt):
                print("OPTIMUM VERIFICATION FAILED!")
            else:
                print(f"c_opt={c_opt}, L_opt={L_opt}, m_opt={m_opt}, tot_opt={tot_opt}")
        for n, c_opt, L_opt, tot_opt, m_opt in zip(heights, c_values_ni, L_values_ni, n_samples_ni, m_values_ni, strict=True):
            print(f"lambda={lambda_}, n={n}, n_a={n_a}, non-interactive:")
            if not await verify_opt(n_proof_ni, lambda_, n, n_a, c_opt):
                print("OPTIMUM VERIFICATION FAILED!")
            else:
                print(f"c_opt={c_opt}, L_opt={L_opt}, m_opt={m_opt}, tot_opt={tot_opt}")
        
        new_rows = [
            {'chaintip': h, 'c': c, 'L': l, 'm': m, 'n_a': n_a, 'non_interactive': False} 
            for h, c, l, m in zip(heights, c_values, L_values, m_values, strict=True)
        ]
        new_rows_ni = [
            {'chaintip': h, 'c': c, 'L': l, 'm': m, 'n_a': n_a, 'non_interactive': True} 
            for h, c, l, m in zip(heights, c_values_ni, L_values_ni, m_values_ni, strict=True)
        ]
        df_list.append(pd.DataFrame(new_rows))
        df_list.append(pd.DataFrame(new_rows_ni))
    df = pd.concat([elem for elem in df_list], ignore_index=True)
    df.sort_index()
    if not os.path.isfile(file_path):
        df.to_csv(file_path)

if __name__ == '__main__':
    asyncio.run(main())