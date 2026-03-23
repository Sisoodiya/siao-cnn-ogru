"""
Self-Improved Aquila Optimizer (SIAO)

An enhanced version of the Aquila Optimizer with chaotic maps
for improved exploration-exploitation balance.

Improvements over standard AO:
1. Chaotic Quality Function using:
   - Logistic chaos map
   - Exponential decay
   - Gaussian-chaotic map
2. Self-improved search mechanism
3. RMSE-based objective function
4. Convergence tracking and stability analysis

Author: Optimization Algorithm Expert
"""

import logging
from typing import Tuple, Optional, List, Callable, Dict

# ── Always use NumPy for SIAO population arrays ──────────────────────────────
# CuPy wastes GPU memory on population vectors (50 agents × millions of weights)
# that competes with PyTorch. The population math is trivial CPU work;
# only the objective function (PyTorch forward pass) should use GPU.
import numpy as xp
import numpy as np
_USE_GPU = False

def _to_xp(arr):
    """Move a NumPy/list array to the active backend (GPU or CPU)."""
    if _USE_GPU:
        return xp.asarray(arr)
    if not isinstance(arr, xp.ndarray):
        return xp.asarray(arr)
    return arr

def _to_np(arr):
    """Return a plain NumPy array regardless of backend."""
    if _USE_GPU and isinstance(arr, xp.ndarray):
        return xp.asnumpy(arr)
    return np.asarray(arr)
# ───────────────────────────────────────────────────────────────────────────

from scipy.special import gamma as scipy_gamma
from rich.console import Console
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()

warnings.filterwarnings('ignore')


# =============================================================================
# Chaotic Maps
# =============================================================================

class ChaoticMaps:
    """
    Collection of chaotic maps for Quality Function enhancement.
    """
    
    @staticmethod
    def logistic_map(x: float, r: float = 4.0) -> float:
        """
        Logistic chaos map.
        
        x_{n+1} = r * x_n * (1 - x_n)
        
        Args:
            x: Current value in [0, 1]
            r: Control parameter (typically 4 for full chaos)
        
        Returns:
            Next chaotic value
        """
        return r * x * (1 - x)
    
    @staticmethod
    def logistic_sequence(length: int, x0: float = 0.7, r: float = 4.0) -> np.ndarray:
        """Generate sequence of logistic map values."""
        seq = np.zeros(length)
        seq[0] = x0
        for i in range(1, length):
            seq[i] = ChaoticMaps.logistic_map(seq[i-1], r)
        return seq
    
    @staticmethod
    def exponential_decay(t: int, t_max: int, alpha: float = 2.0) -> float:
        """
        Exponential decay function.
        
        D(t) = exp(-alpha * t / t_max)
        
        Args:
            t: Current iteration
            t_max: Maximum iterations
            alpha: Decay rate
        
        Returns:
            Decay factor
        """
        return np.exp(-alpha * t / t_max)
    
    @staticmethod
    def gaussian_chaotic(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        Gaussian-chaotic map (Eq. 17-19).
        
        Combines Gaussian distribution with chaotic perturbation.
        
        Args:
            x: Input value
            mu: Mean
            sigma: Standard deviation
        
        Returns:
            Gaussian-chaotic value
        """
        # Gaussian component
        gaussian = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        
        # Chaotic perturbation
        chaos = ChaoticMaps.logistic_map(np.abs(x) % 1)
        
        # Combined (Eq. 19)
        return gaussian * chaos + (1 - gaussian) * x
    
    @staticmethod
    def chaotic_quality_function(
        t: int,
        t_max: int,
        chaos_value: float,
        method: str = 'combined'
    ) -> float:
        """
        Enhanced Quality Function with chaotic maps.
        
        Args:
            t: Current iteration
            t_max: Maximum iterations
            chaos_value: Current chaotic sequence value
            method: 'logistic', 'exponential', 'gaussian', or 'combined'
        
        Returns:
            Quality function value
        """
        if method == 'logistic':
            # Logistic chaos-based QF
            return t ** (2 * chaos_value - 1) / (1 - t_max) ** 2
        
        elif method == 'exponential':
            # Exponential decay QF
            decay = ChaoticMaps.exponential_decay(t, t_max)
            return decay * (2 * np.random.random() - 1)
        
        elif method == 'gaussian':
            # Gaussian-chaotic QF
            gc = ChaoticMaps.gaussian_chaotic(chaos_value)
            return gc * t ** (2 * gc - 1)
        
        else:  # 'combined'
            # Combined strategy (Eq. 17-19)
            logistic = ChaoticMaps.logistic_map(chaos_value)
            decay = ChaoticMaps.exponential_decay(t, t_max)
            gaussian = ChaoticMaps.gaussian_chaotic(logistic)
            
            # Weighted combination
            w1, w2, w3 = 0.4, 0.3, 0.3
            combined = w1 * logistic + w2 * decay + w3 * gaussian
            
            return t ** (2 * combined - 1) / max((1 - t_max) ** 2, 1e-10)


# =============================================================================
# Self-Improved Aquila Optimizer (SIAO)
# =============================================================================

class SelfImprovedAquilaOptimizer:
    """
    Self-Improved Aquila Optimizer (SIAO).
    
    Enhanced AO with chaotic quality functions for:
    - Better exploration-exploitation balance
    - Faster convergence
    - Improved stability
    
    Hunting Strategies:
    1. Expanded Exploration (High soar with vertical stoop)
    2. Narrowed Exploration (Contour flight with Levy)
    3. Expanded Exploitation (Low flight with slow descent)
    4. Narrowed Exploitation (Walk and grab prey)
    """
    
    def __init__(
        self,
        objective_func: Callable,
        dim: int,
        lb: np.ndarray,
        ub: np.ndarray,
        pop_size: int = 30,
        max_iter: int = 100,
        chaos_method: str = 'combined',
        minimize: bool = True,
        batch_objective_func: Optional[Callable] = None,
        convergence_patience: int = 0
    ):
        """
        Initialize SIAO.

        Args:
            objective_func: Function to optimize (single weight vector)
            dim: Number of dimensions
            lb: Lower bounds array
            ub: Upper bounds array
            pop_size: Population size
            max_iter: Maximum iterations
            chaos_method: Chaotic map method
            minimize: True for minimization, False for maximization
            batch_objective_func: Optional function that evaluates a list of
                weight vectors at once and returns a list of losses
            convergence_patience: Stop early if best fitness doesn't improve
                for this many iterations. 0 = disabled.
        """
        self.objective_func = objective_func
        self.batch_objective_func = batch_objective_func
        self.dim = dim
        # Store bounds on GPU so all bound-clipping stays on the same device
        self.lb = _to_xp(np.array(lb) if not isinstance(lb, np.ndarray) else lb)
        self.ub = _to_xp(np.array(ub) if not isinstance(ub, np.ndarray) else ub)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.chaos_method = chaos_method
        self.minimize = minimize
        self.convergence_patience = convergence_patience

        # Population
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = np.inf if minimize else -np.inf

        # Chaotic sequence
        self.chaos_seq = ChaoticMaps.logistic_sequence(max_iter + 1)

        # Convergence tracking
        self.history = []
        self.stability_history = []

        logger.info(f"SIAO: dim={dim}, pop={pop_size}, iter={max_iter}, chaos={chaos_method}"
                     f", batch_eval={'ON' if batch_objective_func else 'OFF'}"
                     f", conv_patience={convergence_patience}")
    
    def _initialize_population(self) -> None:
        """Initialize population with chaos-enhanced distribution (GPU arrays)."""
        # Allocate population on active device (GPU if CuPy, else CPU)
        self.population = xp.zeros((self.pop_size, self.dim), dtype=xp.float32)

        for i in range(self.pop_size):
            # Chaotic init on CPU, then move to device
            chaos_val = _to_xp(
                ChaoticMaps.logistic_sequence(self.dim, x0=0.1 + 0.8 * i / self.pop_size)
            ).astype(xp.float32)
            self.population[i] = self.lb + chaos_val * (self.ub - self.lb)

        self.fitness = xp.zeros(self.pop_size, dtype=xp.float64)

        # Batch evaluate initial population
        if self.batch_objective_func is not None:
            weight_list = [_to_np(self.population[i]) for i in range(self.pop_size)]
            losses = self.batch_objective_func(weight_list)
            for i, loss in enumerate(losses):
                self.fitness[i] = loss
                if self._is_better(loss, self.best_fitness):
                    self.best_fitness = loss
                    self.best_solution = self.population[i].copy()
        else:
            for i in range(self.pop_size):
                self.fitness[i] = self._evaluate(self.population[i])
                if self._is_better(float(self.fitness[i]), self.best_fitness):
                    self.best_fitness = float(self.fitness[i])
                    self.best_solution = self.population[i].copy()
    
    def _evaluate(self, x) -> float:
        """Evaluate objective function. Converts GPU array to NumPy for PyTorch."""
        return float(self.objective_func(_to_np(x)))
    
    def _is_better(self, new_fitness, old_fitness) -> bool:
        """Check if new fitness is better (always uses scalar floats)."""
        nf = float(new_fitness)
        of = float(old_fitness)
        if self.minimize:
            return nf < of
        return nf > of
    
    def _levy_flight(self):
        """Generate Levy flight step (on GPU if CuPy active)."""
        beta = 1.5
        sigma_u = (
            scipy_gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (scipy_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        # Generate on CPU first, then move to active device
        u    = np.random.normal(0, sigma_u, self.dim)
        v    = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return _to_xp(step.astype(np.float32))   # keep on same device as population
    
    def _get_quality_function(self, t: int) -> float:
        """Get chaotic quality function value."""
        chaos_val = self.chaos_seq[t % len(self.chaos_seq)]
        return ChaoticMaps.chaotic_quality_function(
            t, self.max_iter, chaos_val, self.chaos_method
        )
    
    def _expanded_exploration(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        X_mean: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Strategy 1: Expanded Exploration - High soar with vertical stoop.
        
        X_new = X_best * (1 - t/T) + (X_mean - X_best) * rand * chaos
        """
        chaos_val = self.chaos_seq[t % len(self.chaos_seq)]
        X_new = X_best * (1 - t / self.max_iter) + \
                (X_mean - X_best) * np.random.random() * chaos_val
        return self._bound_check(X_new)
    
    def _narrowed_exploration(self, X, X_best, t):
        """
        Strategy 2: Narrowed Exploration - Contour flight with Levy.
        Uses chaotic Levy flight for fine exploration.
        """
        levy      = self._levy_flight()                             # xp array
        chaos_val = float(self.chaos_seq[t % len(self.chaos_seq)]) # Python float
        y         = X_best - X + levy * chaos_val
        X_new     = X_best + float(np.random.random()) * y
        return self._bound_check(X_new)
    
    def _expanded_exploitation(self, X, X_best, t):
        """
        Strategy 3: Expanded Exploitation - Low flight with slow descent.
        Uses chaotic spiral for exploitation.
        """
        QF        = float(self._get_quality_function(t))
        theta     = float(np.random.uniform(-np.pi, np.pi))
        chaos_val = float(self.chaos_seq[t % len(self.chaos_seq)])
        r         = chaos_val * (self.ub - self.lb) * (1 - t / self.max_iter)  # xp array
        X_new     = X_best - (X - X_best) * 0.1 * QF - r * float(np.cos(theta))
        return self._bound_check(X_new)
    
    def _narrowed_exploitation(self, X, X_best, t):
        """
        Strategy 4: Narrowed Exploitation - Walk and grab prey.
        Fine-tuned search around best solution.
        """
        QF        = float(self._get_quality_function(t))
        chaos_val = float(self.chaos_seq[t % len(self.chaos_seq)])
        G1        = float(2 * chaos_val - 1)
        G2        = float(2 * (1 - t / self.max_iter))
        r1        = float(np.random.random())
        r2        = float(np.random.random())
        X_new     = QF * X_best - (G1 * X * r1) - G2 * r2 * (X - X_best)
        return self._bound_check(X_new)
    
    def _self_improvement(self, X, X_new, X_best, t):
        """
        Self-improvement mechanism.
        Combines current and new solutions with adaptive weights.
        """
        w         = float(1 - t / self.max_iter)
        chaos_val = float(self.chaos_seq[t % len(self.chaos_seq)])
        X_improved = w * X + (1 - w) * X_new + 0.1 * chaos_val * (X_best - X_new)
        return self._bound_check(X_improved)
    
    def _bound_check(self, X) -> xp.ndarray:
        """Ensure solution is within bounds (GPU clip)."""
        return xp.clip(_to_xp(X), self.lb, self.ub)
    
    def _compute_stability(self) -> float:
        """Compute population stability metric (scalar, always CPU)."""
        if len(self.history) < 5:
            return 1.0
        recent = np.array(self.history[-5:], dtype=np.float64)
        std  = float(np.std(recent))
        mean = float(np.abs(np.mean(recent)))
        return std / (mean + 1e-10)
    
    def optimize(self) -> Tuple[np.ndarray, float, Dict]:
        """
        Run SIAO optimization.

        Returns:
            Tuple of (best_solution, best_fitness, info_dict)
        """
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn, MofNCompleteColumn

        console.print(f"[bold cyan]SIAO Optimization: dim={self.dim}, pop={self.pop_size}, max_iter={self.max_iter}[/bold cyan]")

        # Initialize
        self._initialize_population()
        self.history.append(self.best_fitness)

        console.print(f"[bold]Initial best loss: {self.best_fitness:.6f}[/bold]")

        no_improve_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="cyan", finished_style="bold cyan"),
            MofNCompleteColumn(),
            TextColumn("[yellow]{task.fields[metrics]}"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=5,
        ) as progress:

            siao_task = progress.add_task("SIAO", total=self.max_iter, metrics=f"best={self.best_fitness:.6f}")

            for t in range(1, self.max_iter + 1):
                # Mean over population stays on GPU
                X_mean = xp.mean(self.population, axis=0)

                # --- Generate all candidate solutions first ---
                candidates_improved = []
                candidates_new = []

                for i in range(self.pop_size):
                    X = self.population[i]

                    # Strategy selection based on progress and randomness
                    r = np.random.random()
                    t_ratio = t / self.max_iter

                    if t_ratio <= 2/3:
                        if r < 0.5:
                            X_new = self._expanded_exploration(X, self.best_solution, X_mean, t)
                        else:
                            X_new = self._narrowed_exploration(X, self.best_solution, t)
                    else:
                        if r < 0.5:
                            X_new = self._expanded_exploitation(X, self.best_solution, t)
                        else:
                            X_new = self._narrowed_exploitation(X, self.best_solution, t)

                    X_improved = self._self_improvement(X, X_new, self.best_solution, t)

                    candidates_new.append(X_new)
                    candidates_improved.append(X_improved)

                # --- Batch evaluate all candidates at once ---
                if self.batch_objective_func is not None:
                    all_candidates = []
                    for i in range(self.pop_size):
                        all_candidates.append(_to_np(candidates_improved[i]))
                        all_candidates.append(_to_np(candidates_new[i]))

                    all_losses = self.batch_objective_func(all_candidates)

                    for i in range(self.pop_size):
                        fitness_improved = all_losses[2 * i]
                        fitness_new = all_losses[2 * i + 1]

                        if self._is_better(fitness_improved, fitness_new):
                            final_X = candidates_improved[i]
                            final_fitness = fitness_improved
                        else:
                            final_X = candidates_new[i]
                            final_fitness = fitness_new

                        if self._is_better(final_fitness, float(self.fitness[i])):
                            self.population[i] = _to_xp(final_X)
                            self.fitness[i] = final_fitness

                            if self._is_better(final_fitness, self.best_fitness):
                                self.best_fitness = final_fitness
                                self.best_solution = _to_xp(final_X).copy()
                else:
                    # Fallback: sequential evaluation
                    for i in range(self.pop_size):
                        fitness_improved = self._evaluate(candidates_improved[i])
                        fitness_new = self._evaluate(candidates_new[i])

                        if self._is_better(fitness_improved, fitness_new):
                            final_X = candidates_improved[i]
                            final_fitness = fitness_improved
                        else:
                            final_X = candidates_new[i]
                            final_fitness = fitness_new

                        if self._is_better(final_fitness, float(self.fitness[i])):
                            self.population[i] = _to_xp(final_X)
                            self.fitness[i] = final_fitness

                            if self._is_better(final_fitness, self.best_fitness):
                                self.best_fitness = final_fitness
                                self.best_solution = _to_xp(final_X).copy()

                # Track convergence
                prev_best = self.history[-1] if self.history else self.best_fitness
                self.history.append(self.best_fitness)
                self.stability_history.append(self._compute_stability())

                # Update progress bar
                progress.update(siao_task, advance=1,
                                metrics=f"best={self.best_fitness:.6f}")

                # Convergence early stopping
                if self.convergence_patience > 0:
                    if abs(self.best_fitness - prev_best) < 1e-6:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0

                    if no_improve_count >= self.convergence_patience:
                        console.print(f"[bold yellow]SIAO converged at iter {t}/{self.max_iter} "
                                     f"(no improvement for {self.convergence_patience} iters)[/bold yellow]")
                        break

        console.print(f"[bold green]SIAO complete: best loss = {self.best_fitness:.6f}[/bold green]")
        info = {
            'history':          np.array(self.history),
            'stability':        np.array(self.stability_history),
            'final_population': _to_np(self.population),
            'final_fitness':    _to_np(self.fitness)
        }

        best_np = _to_np(self.best_solution)

        return best_np, self.best_fitness, info


# =============================================================================
# RMSE Objective Function
# =============================================================================

def create_rmse_objective(y_true: np.ndarray, model_func: Callable) -> Callable:
    """
    Create RMSE objective function for optimization.
    
    Args:
        y_true: Ground truth values
        model_func: Function that takes parameters and returns predictions
    
    Returns:
        Objective function that computes RMSE
    """
    def rmse_objective(params: np.ndarray) -> float:
        y_pred = model_func(params)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse
    
    return rmse_objective


# =============================================================================
# Visualization
# =============================================================================

def plot_siao_convergence(
    history: np.ndarray,
    stability: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """Plot SIAO convergence and stability."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2 if stability is not None else 1, figsize=(14, 5))
        
        if stability is None:
            axes = [axes]
        
        # Convergence plot
        axes[0].plot(history, 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('RMSE / Fitness', fontsize=12)
        axes[0].set_title('SIAO Convergence', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Stability plot
        if stability is not None:
            axes[1].plot(stability, 'r-', linewidth=2)
            axes[1].set_xlabel('Iteration', fontsize=12)
            axes[1].set_ylabel('Stability Index', fontsize=12)
            axes[1].set_title('Population Stability', fontsize=14)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


# =============================================================================
# Benchmark Functions
# =============================================================================

class BenchmarkFunctions:
    """
    Standard benchmark functions for optimization testing.
    """
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere function (unimodal): f(x) = sum(x^2)"""
        return np.sum(x ** 2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin function (multimodal)"""
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock function (valley-shaped)"""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley function (multimodal)"""
        d = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Griewank function (multimodal)"""
        sum_sq = np.sum(x ** 2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_sq - prod_cos + 1


def run_siao_demo(
    func_name: str = 'sphere',
    dim: int = 10,
    pop_size: int = 30,
    max_iter: int = 100,
    show_plot: bool = True
) -> Tuple[np.ndarray, float, Dict]:
    """
    Run SIAO optimization demo with a benchmark function.
    
    Args:
        func_name: 'sphere', 'rastrigin', 'rosenbrock', 'ackley', 'griewank'
        dim: Number of dimensions
        pop_size: Population size
        max_iter: Maximum iterations
        show_plot: Whether to show convergence plot
    
    Returns:
        Tuple of (best_solution, best_fitness, info_dict)
    """
    # Get benchmark function
    functions = {
        'sphere': (BenchmarkFunctions.sphere, -10, 10),
        'rastrigin': (BenchmarkFunctions.rastrigin, -5.12, 5.12),
        'rosenbrock': (BenchmarkFunctions.rosenbrock, -5, 10),
        'ackley': (BenchmarkFunctions.ackley, -32.768, 32.768),
        'griewank': (BenchmarkFunctions.griewank, -600, 600),
    }
    
    if func_name not in functions:
        raise ValueError(f"Unknown function: {func_name}. Choose from {list(functions.keys())}")
    
    func, lb_val, ub_val = functions[func_name]
    lb = lb_val * np.ones(dim)
    ub = ub_val * np.ones(dim)
    
    print(f"Running SIAO on {func_name} function (dim={dim})")
    print("=" * 60)
    
    # Create and run SIAO
    siao = SelfImprovedAquilaOptimizer(
        objective_func=func,
        dim=dim,
        lb=lb,
        ub=ub,
        pop_size=pop_size,
        max_iter=max_iter,
        chaos_method='combined',
        minimize=True
    )
    
    best_solution, best_fitness, info = siao.optimize()
    
    print(f"\nResults:")
    print(f"  Best Fitness: {best_fitness:.6e}")
    print(f"  Best Solution (first 5): {best_solution[:min(5, dim)]}")
    
    if show_plot:
        plot_siao_convergence(info['history'], info['stability'])
    
    return best_solution, best_fitness, info


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Self-Improved Aquila Optimizer (SIAO) - Demo")
    print("=" * 60)
    
    # Run demo with Sphere function
    best_solution, best_fitness, info = run_siao_demo(
        func_name='sphere',
        dim=10,
        pop_size=30,
        max_iter=100,
        show_plot=True
    )
