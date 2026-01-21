"""
Experiment 72: Inverse Generator Search

Objective:
Find the mathematical generator of the singular value sequence.
We stripped away the 'magic number' physics. Now we look for the 
dynamical system.

Hypothesis:
The sequence of singular values S_n is not random, but generated 
by a recurrence relation: S_{n+1} = f(S_n).

Method:
1. Load the singular values S (unnormalized, raw).
2. Plot S_{n+1} vs S_n (Return Map).
3. Test standard recurrence classes:
   - Geometric: S_{n+1} = r * S_n
   - Power: S_{n+1} = a * (S_n)^k
   - Logistic: x_{n+1} = r * x_n * (1 - x_n) (requires normalization)
   - Exponential: S_{n+1} = exp(a * S_n + b)
4. Measure the fit quality (R^2).
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import load_wow_signal

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Define candidate functions
def func_linear(x, a, b):
    return a * x + b

def func_power(x, a, k):
    return a * np.power(x, k)

def func_logistic(x, r):
    # Logistic map usually on [0,1]
    return r * x * (1 - x)

def func_exp(x, a, b):
    return a * np.exp(b * x)

def analyze_recurrence(S):
    """Analyze the recurrence S_{n+1} = f(S_n)."""
    # Prepare X (S_n) and Y (S_{n+1})
    X = S[:-1]
    Y = S[1:]
    
    results = {
        "data_points": list(zip(X.tolist(), Y.tolist())),
        "models": {}
    }
    
    # 1. Linear (Geometric Decay check)
    try:
        popt, pcov = curve_fit(func_linear, X, Y)
        residuals = Y - func_linear(X, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2 = 1 - (ss_res / ss_tot)
        results["models"]["linear"] = {
            "params": [float(p) for p in popt],
            "r2": float(r2),
            "equation": f"S(n+1) = {popt[0]:.4f} * S(n) + {popt[1]:.4f}"
        }
    except:
        pass

    # 2. Power Law
    try:
        popt, pcov = curve_fit(func_power, X, Y, maxfev=5000)
        residuals = Y - func_power(X, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2 = 1 - (ss_res / ss_tot)
        results["models"]["power"] = {
            "params": [float(p) for p in popt],
            "r2": float(r2),
            "equation": f"S(n+1) = {popt[0]:.4f} * S(n)^{popt[1]:.4f}"
        }
    except:
        pass
        
    return results

def main():
    print("=" * 60)
    print("Experiment 72: Inverse Generator Search")
    print("=" * 60)
    
    # Load Signal
    wow = load_wow_signal()
    # Normalized for stability (max = 1)
    wow = wow / np.max(wow)
    S = linalg.svd(wow, compute_uv=False)
    
    # We only care about the signal modes, not the noise tail
    # Identify 'elbow' or use first k modes
    # exp69 said d=1.72, let's look at top 10
    S_signal = S[:10]
    
    print(f"\nAnalyzing Top 10 Singular Values:")
    print(f"   {[f'{s:.4f}' for s in S_signal]}")
    
    # 1. Check Recurrence
    print(f"\n1. Fitting Recurrence Relations (S_n -> S_n+1)...")
    results = analyze_recurrence(S_signal)
    
    best_model = None
    best_r2 = -np.inf
    
    for name, model in results["models"].items():
        print(f"   Model: {name:10s} R2: {model['r2']:.4f}  [{model['equation']}]")
        if model['r2'] > best_r2:
            best_r2 = model['r2']
            best_model = model
            
    # 2. Check Specific Map: Logistic
    # Logistic map requires mapping S to [0,1]. S is already max=1.
    # Check if x_n+1 = 4 x_n (1 - x_n) (Chaos limit)
    
    # 3. Check for Feigenbaum Constants
    # If it's a period-doubling cascade, ratio of gaps should be ~4.669
    # Gaps: (S0-S1), (S1-S2), (S2-S3)...
    
    gaps = S_signal[:-1] - S_signal[1:]
    gap_ratios = gaps[:-1] / gaps[1:]
    
    print(f"\n2. Feigenbaum Delta Check (Target ~4.669):")
    print(f"   Gap Ratios: {[f'{g:.4f}' for g in gap_ratios[:5]]}")
    
    # Save
    with open(RESULTS_DIR / "exp72_results.json", "w") as f:
        json.dump({
            "experiment": "exp72_inverse_generator",
            "timestamp": datetime.now().isoformat(),
            "singular_values": [float(s) for s in S_signal],
            "recurrence_models": results["models"],
            "gap_ratios": [float(g) for g in gap_ratios]
        }, f, indent=2)

if __name__ == "__main__":
    main()
