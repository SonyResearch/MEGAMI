
import torch
import os

def extend_grid(grid):
    N = grid.shape[0]
    n = 2 / N
    x = torch.concatenate((torch.tensor([-1-n]), grid, torch.tensor([1+n])))
    pts = x.clone()
    return pts

def calculate_mu_grid(mu, G, range=(-1, 1)):
    grid = torch.linspace(-1, 1, steps=G, dtype=torch.float32)
    grid = torch.sign(grid) * (((1 + mu) ** (grid.abs()) - 1) / mu)
    grid = extend_grid(grid)[None, :]
    return grid


def CCR_Basis(t, alpha):
    """"
    Quartic catmull rom basis
    """
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t

    f_0 = 0.5 * (-t + 2 * (1 + alpha) * t2 - (1 + 4 * alpha) * t3 + 2 * alpha * t4)
    f_1 = 0.5 * (2 - (5 + 2 * alpha) * t2 + (3 + 4 * alpha) * t3 - 2 * alpha * t4)
    f_2 = 0.5 * (t + 2 * (2 - alpha) * t2 - (3 - 4 * alpha) * t3 - 2 * alpha * t4)
    f_3 = 0.5 * (-1 * (1 - 2 * alpha) * t2 + (1 - 4 * alpha) * t3 + 2 * alpha * t4)
    f = torch.stack((f_0, f_1, f_2, f_3), dim=1)
    return f


def calculate_CCR(x_eval, grid, alphas, coefs, device='cpu'):
    """
    Calculate the CCR curve
    :param x_eval: torch.tensor, shape (N, 1)
    :param grid: torch.tensor, shape (1, N)
    :param alphas: torch.tensor, shape (N,)
    :param coefs: torch.tensor, shape (N,)
    """

    x_eval = x_eval.squeeze(0).permute(1, 0)

    x_in_grid = (x_eval >= grid[:, :-1]) * (x_eval < grid[:, 1:]).to(device)
    interval_differences = torch.diff(grid).to(device)
    t_for_curve_segments = x_eval * x_in_grid.to(torch.int).to(device)  # Change boolean to sample values on positions
    # recalculate t for segment to be between 0-1
    t_for_curve_segments = (t_for_curve_segments - grid[:, :-1]) / interval_differences
    t_for_curve_segments = t_for_curve_segments * x_in_grid  # this is maybe not needed

    y = torch.zeros_like(x_eval).to(device)
    basis_b = torch.zeros((x_eval.shape[0], 4)).to(device)
    pts_temp = torch.zeros((x_eval.shape[0], 4)).to(device)
    for col in range(1, coefs.shape[0]-2):
        t = t_for_curve_segments[(x_in_grid[:, col] == True),col]
        t_idx = torch.nonzero(x_in_grid[:, col]).squeeze()
        if t.shape[-1] != 0:
            # calculate mapping
            basis = CCR_Basis(t, alphas[col-1])
            basis_b[t_idx] = basis.clone()
            pts_temp[t_idx] = coefs[col-1:col+3].clone()

    y = torch.sum(basis_b * pts_temp, dim=1)
    return y



class CubicCatmullRomSpline(torch.nn.Module):
    def __init__(
        self,
        mu: float = 20,
        G: int = 41,
        grid_range=(-5, 5),
        optimize_alphas: bool = False,
        fix_zero: bool = True,
    ):
        super().__init__()
        self.mu = mu
        self.G = G
        grid = calculate_mu_grid(self.mu, self.G, grid_range)
        self.register_buffer("grid", grid, persistent=False)
        
        #make sure that G is odd
        assert self.G % 2 == 1, "G must be an odd number for symmetric grid."

        self.fix_zero = fix_zero

        self.grid_size= self.grid.shape[-1]
        if self.fix_zero:
            grid_to_copy= self.grid.clone()
            grid_to_copy=torch.cat((grid_to_copy[:,0:self.grid_size//2], grid_to_copy[:,-self.grid_size//2 +1:]), dim=-1)
            self.coefs_optimizable = torch.nn.Parameter(grid_to_copy.view(-1), requires_grad=True)
        else:
            self.coefs_optimizable = torch.nn.Parameter(self.grid.clone().view(-1), requires_grad=True)

        alphas = torch.zeros(self.G - 1, dtype=torch.float32, requires_grad=optimize_alphas)

        self.register_buffer("alphas", alphas, persistent=False)

        self.grid_range = grid_range
        


    def forward(self, x: torch.Tensor ):

        orig_shape=x.shape

        #Since the CCR is a sample-wise independent operation, we can flatten the input tensor
        x=x.view(1,-1)

        if (x >= self.grid_range[1]).any() or (x <= self.grid_range[0]).any():
            print(f"Input {x} is out of bounds for the grid range {self.grid_range}. Clamping to range.")
            x = torch.clamp(x, self.grid_range[0]+1e-4, self.grid_range[1]-1e-4)

        if self.fix_zero:
            coefs= torch.cat((self.coefs_optimizable[0:self.grid_size//2], torch.zeros((1), device=x.device), self.coefs_optimizable[-self.grid_size//2 +1:]), dim=-1)
        else:
            coefs = self.coefs_optimizable

        y= calculate_CCR(x_eval=x[:, None], grid=self.grid, alphas=self.alphas, coefs=coefs, device=x.device)

        y=y.view(orig_shape)

        return y

    
if __name__ == "__main__":
    # Example usage
    mu = 20
    G = 41

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ccr = CubicCatmullRomSpline(mu=mu, G=G).to(device)

    # Create a sample input tensor
    x = torch.linspace(-1, 1, steps=100).view(1, -1).to(device)

    # Forward pass through the CCR module
    output = ccr(x)
