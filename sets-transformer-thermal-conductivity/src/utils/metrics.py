def mean_absolute_error(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def mean_squared_error(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def r_squared(predictions, targets):
    ss_total = torch.sum((targets - torch.mean(targets)) ** 2)
    ss_residual = torch.sum((targets - predictions) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0

def calculate_metrics(predictions, targets):
    mae = mean_absolute_error(predictions, targets)
    mse = mean_squared_error(predictions, targets)
    r2 = r_squared(predictions, targets)
    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'R^2': r2.item()
    }