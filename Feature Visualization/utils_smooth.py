import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        self.model = model
        self.target_layers = target_layers
        self.cuda = use_cuda
        self.reshape_transform = reshape_transform

        if self.cuda:
            self.model = model.cuda()

        self.activations = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output)

        def backward_hook(module, grad_out, grad_in):
            self.gradients.append(grad_out[0])

        for layer in self.target_layers:
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_category is None:
            target_category = np.argmax(output.cpu().detach().numpy())

        loss = output[:, target_category]
        loss.backward(retain_graph=True)

        activations = self.activations[-1].cpu().data.numpy()
        gradients = self.gradients[-1].cpu().data.numpy()

        if self.reshape_transform is not None:
            activations = self.reshape_transform(torch.from_numpy(activations)).numpy()

        weights = np.mean(gradients, axis=2, keepdims=True)
        cam = np.sum(weights * activations, axis=1)

        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class SmoothGradCAM(GradCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None, noise_level=0.1, num_samples=50):
        super(SmoothGradCAM, self).__init__(model, target_layers, use_cuda, reshape_transform)
        self.noise_level = noise_level
        self.num_samples = num_samples

    def __call__(self, input_tensor, target_category=None):
        smoothed_cam = None

        for i in range(self.num_samples):
            noisy_input = input_tensor + self.noise_level * torch.randn_like(input_tensor)
            cam = super(SmoothGradCAM, self).__call__(noisy_input, target_category)

            if smoothed_cam is None:
                smoothed_cam = cam
            else:
                smoothed_cam += cam

        smoothed_cam = smoothed_cam / self.num_samples
        return smoothed_cam

