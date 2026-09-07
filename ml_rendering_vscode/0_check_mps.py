import platform
import torch

print('Python/Platform:', platform.platform())
print('PyTorch:', torch.__version__)
print('MPS built:', torch.backends.mps.is_built())
print('MPS available:', torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(256, 256, device=device)
    y = x @ x.T
    print('MPS test OK:', y.shape, y.device)
else:
    print('MPS is not available. Training will fall back to CPU and be very slow.')
