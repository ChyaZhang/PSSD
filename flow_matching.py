import torch
import torch.nn.functional as F
from typing import Optional, Literal

class FlowMatcher:
    def __init__(self, sigma: float = 0.1,
                 path_type: Literal["linear", "optimal_transport", "variance_preserving"] = "linear"):
        """
        Args:
            sigma: Standard deviation for regularization noise
            path_type: Type of interpolation path to use
        """
        self.sigma = sigma
        self.path_type = path_type

    def get_train_tuple(self, x1: torch.Tensor, x0: torch.Tensor = None):
        """
        Samples a point on the path and returns the perturbed data, the time step,
        and the target vector field.

        Args:
            x1: The target data (real gene expression data), shape (N, C, ...).
            x0: The source data (noise), shape (N, C, ...). If None, it's sampled internally.

        Returns:
            xt: The perturbed data at time t, shape (N, C, ...).
            t: The time steps, shape (N,).
            ut: The target vector field, shape (N, C, ...).
        """
        # Ensure x1 is float
        x1 = x1.to(torch.float32)

        # Sample x0 from a standard normal distribution if not provided
        if x0 is None:
            x0 = torch.randn_like(x1)

        # Sample time t from a uniform distribution [0, 1]
        t = torch.rand(x1.size(0), device=x1.device)

        # Get interpolation and vector field based on path type
        if self.path_type == "linear":
            xt, ut = self._linear_path(x0, x1, t)
        elif self.path_type == "optimal_transport":
            xt, ut = self._optimal_transport_path(x0, x1, t)
        elif self.path_type == "variance_preserving":
            xt, ut = self._variance_preserving_path(x0, x1, t)
        else:
            raise ValueError(f"Unknown path type: {self.path_type}")

        return xt, t, ut
    
    def _linear_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        """
        Linear interpolation path (simplest and most common)
        xt = (1-t)*x0 + t*x1
        ut = x1 - x0
        """
        t_broadcast = t.view(-1, *([1] * (x1.dim() - 1)))
        xt = (1 - t_broadcast) * x0 + t_broadcast * x1
        ut = x1 - x0
        return xt, ut
    
    def _optimal_transport_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        """
        Optimal transport path with optional Gaussian regularization
        xt = (1-t)*x0 + t*x1 + sigma*sqrt(t(1-t))*epsilon
        ut = x1 - x0
        """
        t_broadcast = t.view(-1, *([1] * (x1.dim() - 1)))
        
        # Linear interpolation
        xt = (1 - t_broadcast) * x0 + t_broadcast * x1
        
        # Add regularization noise if sigma > 0
        if self.sigma > 0:
            noise_scale = self.sigma * torch.sqrt(t_broadcast * (1 - t_broadcast))
            epsilon = torch.randn_like(x0)
            xt = xt + noise_scale * epsilon
        
        ut = x1 - x0
        return xt, ut
    
    def _variance_preserving_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        """
        Variance preserving path (similar to VP-SDE)
        Maintains variance throughout the trajectory
        """
        t_broadcast = t.view(-1, *([1] * (x1.dim() - 1)))
        
        # VP schedule parameters
        alpha_t = torch.cos(0.5 * torch.pi * t_broadcast)
        sigma_t = torch.sin(0.5 * torch.pi * t_broadcast)
        
        # Interpolation with variance preservation
        xt = alpha_t * x0 + sigma_t * x1
        
        # Compute vector field (derivative of the path)
        # For VP path: ut = -0.5*pi*sin(0.5*pi*t)*x0 + 0.5*pi*cos(0.5*pi*t)*x1
        dalpha_dt = -0.5 * torch.pi * torch.sin(0.5 * torch.pi * t_broadcast)
        dsigma_dt = 0.5 * torch.pi * torch.cos(0.5 * torch.pi * t_broadcast)
        ut = dalpha_dt * x0 + dsigma_dt * x1
        
        return xt, ut

class ODESolver:
    """
    ODE Solver for sampling with different numerical methods
    """
    def __init__(self, 
                 method: Literal["euler", "midpoint", "rk4", "heun"] = "euler"):
        """
        Args:
            method: Numerical integration method to use
        """
        self.method = method
    
    @torch.no_grad()
    def sample(self, 
               model: torch.nn.Module,
               z: torch.Tensor,
               y: torch.Tensor,
               num_steps: int = 100,
               device: str = "cuda", 
               coordinates=None):
        """
        Generate samples by solving ODE: dx/dt = v(x, t, y) from t=0 to t=1
        
        Args:
            model: Trained flow matching model
            z: Initial noise, shape (N, 1, NumGene)
            y: Condition, shape (N, CondSize)
            num_steps: Number of ODE steps
            device: Computing device
            
        Returns:
            x: Generated samples
        """
        model.eval()
        x = z.to(device)
        y = y.to(device)
        if coordinates is not None:
            coordinates = coordinates.to(device)
        
        # Time discretization
        dt = 1.0 / num_steps
        
        if self.method == "euler":
            return self._euler_step(model, x, y, num_steps, dt, coordinates)
        elif self.method == "midpoint":
            return self._midpoint_step(model, x, y, num_steps, dt, coordinates)
        elif self.method == "rk4":
            return self._rk4_step(model, x, y, num_steps, dt, coordinates)
        elif self.method == "heun":
            return self._heun_step(model, x, y, num_steps, dt, coordinates)
        else:
            raise ValueError(f"Unknown ODE solver method: {self.method}")
    
    def _euler_step(self, model, x, y, num_steps, dt, coordinates=None):
        """
        Euler method (first-order)
        x_{n+1} = x_n + dt * v(x_n, t_n)
        """
        from tqdm import tqdm
        
        for i in tqdm(range(num_steps), desc="Euler sampling"):
            t = i * dt
            t_batch = torch.full((x.shape[0],), t, device=x.device)
            t_discrete = t_batch * 999  # Scale for model's time embedding
            
            # v = model(x, t_discrete, y=y)
            if coordinates is not None:
                v = model(x, t_discrete, y=y, coordinates=coordinates)
            else:
                v = model(x, t_discrete, y=y)
            x = x + dt * v
        
        return x
    
    def _midpoint_step(self, model, x, y, num_steps, dt, coordinates=None):
        """
        Midpoint method (second-order)
        k1 = v(x_n, t_n)
        k2 = v(x_n + dt/2 * k1, t_n + dt/2)
        x_{n+1} = x_n + dt * k2
        """
        from tqdm import tqdm
        
        for i in tqdm(range(num_steps), desc="Midpoint sampling"):
            t = i * dt
            t_batch = torch.full((x.shape[0],), t, device=x.device)
            t_discrete = t_batch * 999
            
            # First evaluation
            # k1 = model(x, t_discrete, y=y)
            if coordinates is not None:
                k1 = model(x, t_discrete, y=y, coordinates=coordinates)
            else:
                k1 = model(x, t_discrete, y=y)
            
            # Midpoint evaluation
            x_mid = x + 0.5 * dt * k1
            t_mid = t + 0.5 * dt
            t_mid_batch = torch.full((x.shape[0],), t_mid, device=x.device)
            t_mid_discrete = t_mid_batch * 999
            # k2 = model(x_mid, t_mid_discrete, y=y)
            if coordinates is not None:
                k2 = model(x_mid, t_mid_discrete, y=y, coordinates=coordinates)
            else:
                k2 = model(x_mid, t_mid_discrete, y=y)
            
            # Update
            x = x + dt * k2
        
        return x
    
    def _rk4_step(self, model, x, y, num_steps, dt, coordinates=None):
        """
        Runge-Kutta 4th order method
        Most accurate but computationally expensive
        """
        from tqdm import tqdm
        
        for i in tqdm(range(num_steps), desc="RK4 sampling"):
            t = i * dt
            if coordinates is not None:
                # k1
                t_batch = torch.full((x.shape[0],), t, device=x.device)
                k1 = model(x, t_batch * 999, y=y, coordinates=coordinates)
                
                # k2
                x2 = x + 0.5 * dt * k1
                t2 = t + 0.5 * dt
                t2_batch = torch.full((x.shape[0],), t2, device=x.device)
                k2 = model(x2, t2_batch * 999, y=y, coordinates=coordinates)
                
                # k3
                x3 = x + 0.5 * dt * k2
                k3 = model(x3, t2_batch * 999, y=y, coordinates=coordinates)

                # k4
                x4 = x + dt * k3
                t4 = t + dt
                t4_batch = torch.full((x.shape[0],), t4, device=x.device)
                k4 = model(x4, t4_batch * 999, y=y, coordinates=coordinates)
            else:
                # k1
                t_batch = torch.full((x.shape[0],), t, device=x.device)
                k1 = model(x, t_batch * 999, y=y)
                
                # k2
                x2 = x + 0.5 * dt * k1
                t2 = t + 0.5 * dt
                t2_batch = torch.full((x.shape[0],), t2, device=x.device)
                k2 = model(x2, t2_batch * 999, y=y)
                
                # k3
                x3 = x + 0.5 * dt * k2
                k3 = model(x3, t2_batch * 999, y=y)  # Same time as k2
                
                # k4
                x4 = x + dt * k3
                t4 = t + dt
                t4_batch = torch.full((x.shape[0],), t4, device=x.device)
                k4 = model(x4, t4_batch * 999, y=y)
                
            # Weighted average
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return x
    
    def _heun_step(self, model, x, y, num_steps, dt, coordinates=None):
        """
        Heun's method (improved Euler, second-order)
        Predictor-corrector approach
        """
        from tqdm import tqdm
        
        for i in tqdm(range(num_steps), desc="Heun sampling"):
            t = i * dt
            t_batch = torch.full((x.shape[0],), t, device=x.device)
            t_discrete = t_batch * 999
            
            # Predictor step (Euler)
            v1 = model(x, t_discrete, y=y, coordinates=coordinates) if coordinates is not None else model(x, t_discrete, y=y)
            x_pred = x + dt * v1
            
            # Corrector step
            t_next = t + dt
            t_next_batch = torch.full((x.shape[0],), t_next, device=x.device)
            t_next_discrete = t_next_batch * 999
            v2 = model(x_pred, t_next_discrete, y=y, coordinates=coordinates)
            
            # Average the slopes
            x = x + 0.5 * dt * (v1 + v2)
        
        return x


# Convenience function for backward compatibility
def ode_sampler(model, z, y, num_steps=100, device="cuda", method="euler"):
    """
    Wrapper function for ODE sampling
    
    Args:
        model: Trained flow matching model
        z: Initial noise
        y: Condition
        num_steps: Number of ODE steps
        device: Computing device
        method: ODE solver method
    
    Returns:
        Generated samples
    """
    solver = ODESolver(method=method)
    return solver.sample(model, z, y, num_steps, device)