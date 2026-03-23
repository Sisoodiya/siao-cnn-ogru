"""
Aquila Optimizer (AO) for Weighted Kurtosis-Skewness (WKS) Feature Optimization

Implements the Aquila Optimizer meta-heuristic algorithm to find optimal
weights for combining Kurtosis and Skewness features for reactor fault classification.

WKS = ω * Kurtosis + Skewness

Aquila Optimizer Phases:
1. Expanded Exploration (High soar with vertical stoop)
2. Narrowed Exploration (Contour flight with short glide)
3. Expanded Exploitation (Low flight with slow descent)
4. Narrowed Exploitation (Walk and grab prey)

Objective: Maximize inter-class separability, minimize intra-class variance

Author: Meta-Heuristic Optimization Researcher
"""

import logging
from typing import Tuple, Optional, List, Callable
import logging
from typing import Tuple, Optional, List, Callable

# ── Always use NumPy — CuPy's memory pool steals GPU from PyTorch ─────────
# The kurtosis/skewness/Fisher computations here are small (~1200 samples)
# and run fine on CPU. CuPy's memory pool doesn't release back to PyTorch,
# causing OOM when both coexist.
import numpy as xp
import numpy as np
_USE_GPU = False

def _to_xp(arr):
    """Move a NumPy array to the active backend (GPU or CPU)."""
    if _USE_GPU:
        return xp.asarray(arr)
    return xp.asarray(arr) if not isinstance(arr, xp.ndarray) else arr

def _to_np(arr):
    """Return a plain NumPy array regardless of backend."""
    if _USE_GPU and isinstance(arr, xp.ndarray):
        return xp.asnumpy(arr)
    return np.asarray(arr)
# ───────────────────────────────────────────────────────────────────────────

from scipy.special import gamma as scipy_gamma
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# =============================================================================
# Kurtosis and Skewness Computation
# =============================================================================

def compute_kurtosis(X, axis: int = 1):
    """
    Compute kurtosis for each window. GPU-accelerated via CuPy if available.
    X: [num_windows, window_size, num_signals]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_g = _to_xp(X)
        X_clean = xp.where(xp.isnan(X_g), xp.zeros_like(X_g), X_g)
        mean = xp.mean(X_clean, axis=axis, keepdims=True)
        std  = xp.std (X_clean, axis=axis, keepdims=True) + 1e-10
        z    = (X_clean - mean) / std
        kurt = xp.mean(z**4, axis=axis) - 3.0
        return _to_np(kurt)


def compute_skewness(X, axis: int = 1):
    """
    Compute skewness for each window. GPU-accelerated via CuPy if available.
    X: [num_windows, window_size, num_signals]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_g = _to_xp(X)
        X_clean = xp.where(xp.isnan(X_g), xp.zeros_like(X_g), X_g)
        mean = xp.mean(X_clean, axis=axis, keepdims=True)
        std  = xp.std (X_clean, axis=axis, keepdims=True) + 1e-10
        z    = (X_clean - mean) / std
        skew = xp.mean(z**3, axis=axis)
        return _to_np(skew)


def compute_wks(X, omega: float, axis: int = 1):
    """Compute WKS = omega * Kurtosis + Skewness. GPU-accelerated."""
    kt = compute_kurtosis(X, axis=axis)
    sk = compute_skewness(X, axis=axis)
    kt = np.nan_to_num(kt, nan=0.0)
    sk = np.nan_to_num(sk, nan=0.0)
    return omega * kt + sk


# =============================================================================
# Fisher Criterion (Class Separability)
# =============================================================================

def fisher_criterion(features, labels) -> float:
    """
    Compute Fisher's Discriminant Ratio. GPU-accelerated via CuPy.
    features: [num_samples, num_features], labels: [num_samples]
    """
    f = _to_xp(features)
    l = _to_xp(labels)
    unique_classes = np.unique(_to_np(labels))   # np.unique works on CPU
    num_classes = len(unique_classes)
    if num_classes < 2:
        return 0.0
    overall_mean = xp.mean(f, axis=0)
    inter_class_var = xp.zeros(1, dtype=f.dtype)
    intra_class_var = xp.zeros(1, dtype=f.dtype)
    for c in unique_classes:
        class_mask   = l == c
        class_samples = f[class_mask]
        n_c = int(class_samples.shape[0])
        if n_c == 0:
            continue
        class_mean = xp.mean(class_samples, axis=0)
        inter_class_var += n_c * xp.sum((class_mean - overall_mean) ** 2)
        intra_class_var += xp.sum((class_samples - class_mean) ** 2)
    if float(_to_np(intra_class_var)[0]) < 1e-10:
        intra_class_var = xp.ones_like(intra_class_var) * 1e-10
    return float(_to_np(inter_class_var / intra_class_var)[0])


# =============================================================================
# Aquila Optimizer
# =============================================================================

class AquilaOptimizer:
    """
    Aquila Optimizer (AO) - Nature-inspired meta-heuristic algorithm.
    
    Based on the hunting behavior of Aquila eagles:
    1. Expanded Exploration: High soar with vertical stoop
    2. Narrowed Exploration: Contour flight with short glide (Levy flight)
    3. Expanded Exploitation: Low flight with slow descent
    4. Narrowed Exploitation: Walk and grab prey
    """
    
    def __init__(
        self,
        objective_func: Callable,
        dim: int = 1,
        lb: float = 0.0,
        ub: float = 2.0,
        pop_size: int = 30,
        max_iter: int = 100,
        alpha: float = 0.1,
        delta: float = 0.1
    ):
        """
        Initialize Aquila Optimizer.
        
        Args:
            objective_func: Function to maximize (higher = better)
            dim: Number of dimensions (1 for omega only)
            lb: Lower bound
            ub: Upper bound
            pop_size: Population size
            max_iter: Maximum iterations
            alpha: Control parameter for exploitation
            delta: Control parameter for exploration
        """
        self.objective_func = objective_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.delta = delta
        
        # Population
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = -np.inf
        
        # Convergence history
        self.history = []
        
        logger.info(f"AquilaOptimizer: dim={dim}, bounds=[{lb}, {ub}], pop={pop_size}, iter={max_iter}")
    
    def _initialize_population(self) -> None:
        """Initialize random population within bounds."""
        self.population = np.random.uniform(
            self.lb, self.ub, 
            size=(self.pop_size, self.dim)
        )
        self.fitness = np.zeros(self.pop_size)
        
        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = self.objective_func(self.population[i])
            
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()
    
    def _levy_flight(self, dim: int) -> np.ndarray:
        """
        Generate Levy flight step.
        
        Returns:
            Levy flight step vector
        """
        beta = 1.5
        sigma_u = (
            scipy_gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (scipy_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, 1, dim)
        
        step = u / (np.abs(v) ** (1 / beta))
        return step
    
    def _expanded_exploration(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        X_mean: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Expanded Exploration: High soar with vertical stoop.
        
        Equation (3) in AO paper.
        """
        X_new = X_best * (1 - t / self.max_iter) + (X_mean - X_best) * np.random.random()
        return np.clip(X_new, self.lb, self.ub)
    
    def _narrowed_exploration(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Narrowed Exploration: Contour flight with short glide (Levy flight).
        
        Equation (5) in AO paper.
        """
        levy = self._levy_flight(self.dim)
        y = X_best - X + levy
        X_new = X_best + np.random.random() * y
        return np.clip(X_new, self.lb, self.ub)
    
    def _expanded_exploitation(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Expanded Exploitation: Low flight with slow descent.
        
        Equation (13) in AO paper.
        """
        alpha = self.alpha
        delta = self.delta
        
        # Spiral shape parameters
        theta = np.random.uniform(-np.pi, np.pi)
        r = np.random.random() * (self.ub - self.lb) * (1 - t / self.max_iter)
        
        # Quality function
        QF = t ** (2 * np.random.random() - 1) / (1 - self.max_iter) ** 2
        
        X_new = X_best - (X - X_best) * alpha * QF - r * np.cos(theta)
        return np.clip(X_new, self.lb, self.ub)
    
    def _narrowed_exploitation(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Narrowed Exploitation: Walk and grab prey.
        
        Equation (14) in AO paper.
        """
        QF = t ** (2 * np.random.random() - 1) / (1 - self.max_iter) ** 2
        
        G1 = 2 * np.random.random() - 1
        G2 = 2 * (1 - t / self.max_iter)
        
        X_new = QF * X_best - (G1 * X * np.random.random()) - G2 * np.random.random() * (X - X_best)
        return np.clip(X_new, self.lb, self.ub)
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the Aquila Optimizer.
        
        Returns:
            Tuple of (best_solution, best_fitness)
        """
        logger.info("Starting Aquila Optimizer...")
        
        # Initialize population
        self._initialize_population()
        self.history.append(self.best_fitness)
        
        for t in range(1, self.max_iter + 1):
            X_mean = np.mean(self.population, axis=0)
            
            for i in range(self.pop_size):
                X = self.population[i]
                
                # Choose strategy based on iteration and random factor
                r = np.random.random()
                t_ratio = t / self.max_iter
                
                if t_ratio <= 2/3:
                    # Exploration phase
                    if r < 0.5:
                        X_new = self._expanded_exploration(X, self.best_solution, X_mean, t)
                    else:
                        X_new = self._narrowed_exploration(X, self.best_solution, t)
                else:
                    # Exploitation phase
                    if r < 0.5:
                        X_new = self._expanded_exploitation(X, self.best_solution, t)
                    else:
                        X_new = self._narrowed_exploitation(X, self.best_solution, t)
                
                # Evaluate new solution
                fitness_new = self.objective_func(X_new)
                
                # Update if better
                if fitness_new > self.fitness[i]:
                    self.population[i] = X_new
                    self.fitness[i] = fitness_new
                    
                    if fitness_new > self.best_fitness:
                        self.best_fitness = fitness_new
                        self.best_solution = X_new.copy()
            
            self.history.append(self.best_fitness)
            
            if t % 10 == 0 or t == 1:
                logger.info(f"Iteration {t}/{self.max_iter}: Best fitness = {self.best_fitness:.6f}")
        
        logger.info(f"Optimization complete. Best ω = {self.best_solution[0]:.4f}, Fitness = {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness


# =============================================================================
# WKS Optimizer (High-level API)
# =============================================================================

class WKSOptimizer:
    """
    Weighted Kurtosis-Skewness Optimizer using Aquila Optimizer.
    
    Finds optimal ω for WKS = ω * Kurtosis + Skewness
    to maximize class separability.
    """
    
    def __init__(
        self,
        omega_bounds: Tuple[float, float] = (0.0, 2.0),
        pop_size: int = 30,
        max_iter: int = 50
    ):
        """
        Initialize WKS Optimizer.
        
        Args:
            omega_bounds: (lower, upper) bounds for ω
            pop_size: Population size for AO
            max_iter: Maximum iterations
        """
        self.omega_bounds = omega_bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.optimal_omega = None
        self.optimal_fitness = None
        self.optimizer = None
    
    def _objective(
        self,
        omega: np.ndarray,
        X_windows: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Objective function: Fisher criterion on WKS features.
        
        Args:
            omega: Weight value [1]
            X_windows: Window data [num_windows, window_size, num_signals]
            y: Labels [num_windows]
        
        Returns:
            Fisher criterion value (higher = better)
        """
        omega_val = omega[0] if isinstance(omega, np.ndarray) else omega
        
        # Compute WKS features
        wks_features = compute_wks(X_windows, omega_val, axis=1)
        
        # Compute Fisher criterion
        return fisher_criterion(wks_features, y)
    
    def optimize(
        self,
        X_windows: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Find optimal ω using Aquila Optimizer.
        
        Args:
            X_windows: Window data [num_windows, window_size, num_signals]
            y: Labels [num_windows]
        
        Returns:
            Tuple of (optimal_omega, optimal_fitness, convergence_history)
        """
        logger.info("=" * 60)
        logger.info("WKS Optimization using Aquila Optimizer")
        logger.info("=" * 60)
        logger.info(f"Input shape: {X_windows.shape}")
        logger.info(f"Omega bounds: {self.omega_bounds}")
        
        # Create objective function closure
        def objective(omega):
            return self._objective(omega, X_windows, y)
        
        # Initialize and run Aquila Optimizer
        self.optimizer = AquilaOptimizer(
            objective_func=objective,
            dim=1,
            lb=self.omega_bounds[0],
            ub=self.omega_bounds[1],
            pop_size=self.pop_size,
            max_iter=self.max_iter
        )
        
        best_solution, best_fitness = self.optimizer.optimize()
        
        self.optimal_omega = best_solution[0]
        self.optimal_fitness = best_fitness
        
        logger.info("=" * 60)
        logger.info("WKS Optimization Complete!")
        logger.info(f"  Optimal ω: {self.optimal_omega:.4f}")
        logger.info(f"  Fisher Criterion: {self.optimal_fitness:.6f}")
        logger.info("=" * 60)
        
        return self.optimal_omega, self.optimal_fitness, np.array(self.optimizer.history)
    
    def extract_wks_features(
        self,
        X_windows: np.ndarray,
        omega: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract WKS features using optimal or specified ω.
        
        Args:
            X_windows: Window data [num_windows, window_size, num_signals]
            omega: Weight value (uses optimal if None)
        
        Returns:
            WKS features [num_windows, num_signals]
        """
        if omega is None:
            if self.optimal_omega is None:
                raise ValueError("Run optimize() first or provide omega")
            omega = self.optimal_omega
        
        return compute_wks(X_windows, omega, axis=1)


# =============================================================================
# Visualization
# =============================================================================

def plot_convergence(history: np.ndarray, save_path: Optional[str] = None):
    """
    Plot optimization convergence history.
    
    Args:
        history: Fitness values per iteration
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fisher Criterion', fontsize=12)
        plt.title('Aquila Optimizer Convergence', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


def plot_class_separability(
    features: np.ndarray,
    labels: np.ndarray,
    feature_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize class separability for a specific feature.
    
    Args:
        features: Feature matrix [num_samples, num_features]
        labels: Class labels
        feature_idx: Which feature to visualize
        save_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        
        unique_classes = np.unique(labels)
        
        plt.figure(figsize=(12, 6))
        
        for c in unique_classes:
            class_data = features[labels == c, feature_idx]
            plt.hist(class_data, alpha=0.5, label=f'Class {c}', bins=30)
        
        plt.xlabel(f'Feature {feature_idx} Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Class Separability - WKS Feature Distribution', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Separability plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Aquila Optimizer for WKS Feature Optimization - Demo")
    print("=" * 60)
    
    # Create dummy data
    num_windows = 200
    window_size = 50
    num_signals = 10
    num_classes = 6
    
    X_windows = np.random.randn(num_windows, window_size, num_signals).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_windows)
    
    print(f"Input X_windows shape: {X_windows.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Optimize WKS
    wks_optimizer = WKSOptimizer(
        omega_bounds=(0.0, 2.0),
        pop_size=20,
        max_iter=30
    )
    
    optimal_omega, fitness, history = wks_optimizer.optimize(X_windows, y)
    
    print(f"\nOptimal ω: {optimal_omega:.4f}")
    print(f"Fisher Criterion: {fitness:.6f}")
    
    # Extract features with optimal omega
    wks_features = wks_optimizer.extract_wks_features(X_windows)
    print(f"WKS features shape: {wks_features.shape}")
