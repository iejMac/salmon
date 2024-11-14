import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from matplotlib.patches import Patch

@dataclass
class ConstraintSystem:
    L: int
    alpha: List[float]
    u: List[float]
    omega: List[float]
    
    def g_l(self, a: List[float], b: List[float], c: List[float], i: int) -> float:
        """Calculate g_i value"""
        return max(a[self.L] + b[self.L], 2*a[self.L] + c[self.L]) + a[i-1]
    
    def r_l(self, a: List[float], b: List[float], c: List[float], l: int) -> float:
        """Calculate r_l value recursively"""
        if l == 1:
            return self.g_l(a, b, c, 1) + a[0] + c[0]
        
        gl = self.g_l(a, b, c, l)
        r_prev = self.r_l(a, b, c, l-1)
        
        return min(
            gl + a[l-1] + c[l-1] - self.alpha[l-1],
            gl + a[l-1] + c[l-1] + r_prev - self.u[l-1],
            0.5 + r_prev - self.omega[l-1]
        )
    
    def check_constraints(self, a: List[float], b: List[float], c: List[float]) -> bool:
        """Check if all constraints are satisfied"""
        eps = 1e-10
        
        # Basic constraints
        if abs(a[0] + b[0]) > eps: return False
        if any(abs(a[l] + b[l] - 0.5) > eps for l in range(1, self.L)): return False
        if a[self.L] + b[self.L] < 0.5 - eps: return False
        if any(self.r_l(a, b, c, l) < -eps for l in range(1, self.L + 1)): return False
        
        # Final constraint
        r_L = self.r_l(a, b, c, self.L)
        return min(
            a[self.L] + b[self.L] + r_L - self.omega[self.L],
            2*a[self.L] + c[self.L] - self.alpha[self.L],
            2*a[self.L] + c[self.L] + r_L - self.u[self.L]
        ) >= -eps

    def visualize_2d_vars(self, var1_info: Tuple[str, int], var2_info: Tuple[str, int], 
                         base_point: dict, radius: float = 1.0, resolution: int = 100):
        """2D visualization with r_L=0 highlight"""
        # Setup base arrays and grid
        var_types = {'a': [], 'b': [], 'c': []}
        for v in var_types:
            var_types[v] = [base_point[f'{v}_{i+1}'] for i in range(self.L + 1)]
        
        var1_type, var1_idx = var1_info
        var2_type, var2_idx = var2_info
        base_x = var_types[var1_type][var1_idx-1]
        base_y = var_types[var2_type][var2_idx-1]
        
        x = np.linspace(base_x - radius, base_x + radius, resolution)
        y = np.linspace(base_y - radius, base_y + radius, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=bool)
        R_L = np.zeros_like(X, dtype=float)
        
        # Evaluate points
        for i in range(resolution):
            for j in range(resolution):
                a, b, c = [lst.copy() for lst in [var_types['a'], var_types['b'], var_types['c']]]
                
                for var_type, var_idx, val in [(var1_type, var1_idx, X[i,j]), 
                                             (var2_type, var2_idx, Y[i,j])]:
                    if var_type == 'a': a[var_idx-1] = val
                    elif var_type == 'b': b[var_idx-1] = val
                    else: c[var_idx-1] = val
                
                Z[i,j] = self.check_constraints(a, b, c)
                R_L[i,j] = self.r_l(a, b, c, self.L)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=[0, 0.5, 1], colors=['white', 'lightblue'], alpha=0.5)
        plt.contour(X, Y, R_L, levels=[0], colors=['red'], linewidths=2, linestyles='--')
        plt.plot([base_x], [base_y], 'k*', markersize=10)
        
        plt.legend(handles=[
            Patch(facecolor='lightblue', alpha=0.5, label='Feasible Region'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='r_L = 0'),
            plt.Line2D([0], [0], marker='*', color='k', label='Base point', 
                      markersize=10, linestyle='None')
        ])
        
        plt.xlabel(f'{var1_type}_{var1_idx}')
        plt.ylabel(f'{var2_type}_{var2_idx}')
        plt.title(f'Feasible Region: {var1_type}_{var1_idx} vs {var2_type}_{var2_idx}')
        plt.grid(True)
        plt.show()

# Create test case with given parameters
def create_test_system():
    L = 2
    alpha = [1.0] * (L + 1)  # alpha_l = 1.0 for all l
    u = [1.0] * (L + 1)      # u_l = 1.0 for all l
    omega = [0.5] * (L + 1)  # omega_l = 0.5 for all l
    
    system = ConstraintSystem(L, alpha, u, omega)
    
    # Create base point
    base_point = {
        'a_1': -0.5,     # a_1 = -0.5
        'a_2': 0.0,      # a_l = 0.0 for l in [2,L]
        'a_3': 0.5,      # a_{L+1} = 0.5
        
        'b_1': 0.5,      # All b_l = 0.5
        'b_2': 0.5,
        'b_3': 0.5,
        
        'c_1': 0.0,      # All c_l = 0.0
        'c_2': 0.0,
        'c_3': 0.0
    }
    
    return system, base_point

# Example usage
if __name__ == "__main__":
		# Create system with your parameters
		system, base_point = create_test_system()

		# Verify the base point is feasible
		is_feasible = system.check_constraints(
				[base_point[f'a_{i+1}'] for i in range(system.L + 1)],
				[base_point[f'b_{i+1}'] for i in range(system.L + 1)],
				[base_point[f'c_{i+1}'] for i in range(system.L + 1)]
		)
		print(f"Base point is feasible: {is_feasible}")

		radius = 1.0
		resolution = 128

		system.visualize_2d_vars(('a', 1), ('b', 1), base_point, radius=radius, resolution=resolution)
		system.visualize_2d_vars(('a', 3), ('b', 3), base_point, radius=radius, resolution=resolution)
		system.visualize_2d_vars(('c', 1), ('c', 2), base_point, radius=radius, resolution=resolution)
		system.visualize_2d_vars(('c', 1), ('c', 3), base_point, radius=radius, resolution=resolution)
		system.visualize_2d_vars(('c', 2), ('c', 3), base_point, radius=radius, resolution=resolution)
