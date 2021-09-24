import torch
import numpy as np

class MSEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, target, samples=None):
        if samples is None:
            samples = prediction
        delta = samples - target
        output = torch.pow(delta, 2).mean()
        for d in delta.shape:
            delta /= d
        ctx.save_for_backward(delta)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        delta, = ctx.saved_tensors
        grad_prediction = grad_target = grad_samples = None
        if ctx.needs_input_grad[0]:
            grad_prediction = grad_output * 2 * delta
        if ctx.needs_input_grad[1]:
            grad_target = -grad_output * 2 * delta

        return grad_prediction, grad_target, grad_samples


class MAEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, target, samples=None):
        if samples is None:
            samples = prediction
        delta = samples - target
        output = torch.abs(delta).mean()
        delta = torch.sign(delta)
        for d in delta.shape:
            delta /= d
        ctx.save_for_backward(delta)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        delta, = ctx.saved_tensors
        grad_prediction = grad_target = grad_samples = None
        if ctx.needs_input_grad[0]:
            grad_prediction = grad_output * delta
        if ctx.needs_input_grad[1]:
            grad_target = -grad_output * delta

        return grad_prediction, grad_target, grad_samples


class IV_MSELoss(torch.nn.Module):
    def __init__(self):
        super(IV_MSELoss, self).__init__()

    def forward(self, prediction, target, samples=None):
        if prediction.shape[0] != target.shape[0]:
            raise RuntimeError(
                "Size mismatch between prediction (%s) and target (%s)"
                % (prediction.shape, target.shape)
            )
        if samples is not None and prediction.shape[0] != samples.shape[0]:
            raise RuntimeError(
                "Size mismatch between prediction (%s) and samples (%s)"
                % (prediction.shape, samples.shape)
            )

        return MSEFunction.apply(prediction, target, samples)


class IV_MAELoss(torch.nn.Module):
    def __init__(self):
        super(IV_MAELoss, self).__init__()

    def forward(self, prediction, target, samples=None):
        if prediction.shape[0] != target.shape[0]:
            raise RuntimeError(
                "Size mismatch between prediction (%s) and target (%s)"
                % (prediction.shape, target.shape)
            )
        if samples is not None and prediction.shape[0] != samples.shape[0]:
            raise RuntimeError(
                "Size mismatch between prediction (%s) and samples (%s)"
                % (prediction.shape, samples.shape)
            )

        return MAEFunction.apply(prediction, target, samples)

class MixtureGaussianLoss(torch.nn.Module):
    def __init__(self):
        super(MixtureGaussianLoss, self).__init__()
        self.LOG_ROOT_2PI = torch.log(torch.sqrt(torch.from_numpy(np.array(2. * np.pi)))) 
        self.softmax = torch.nn.Softmax(dim=1)
        self.eps = 1e-16

    def forward(self, prediction, target):
        mu, log_sigma, log_pi = prediction
        y = target.view(target.shape[0], -1, 1).expand_as(mu)
        z = (y - mu) / (torch.exp(log_sigma) + self.eps)
        log_normals = -0.5 * (torch.pow(z, 2)) - self.LOG_ROOT_2PI - log_sigma
        out = (-torch.logsumexp(log_normals + log_pi, dim=-1) + torch.logsumexp(log_pi, dim=-1)).mean()
        if torch.isnan(out).any():
            print("Hit NaN")
            import pdb; pdb.set_trace()
        return out
