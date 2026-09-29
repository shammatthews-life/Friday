import torch
import torchvision

print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_version', torch.version.cuda)
print('device_count', torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print('device_name', torch.cuda.get_device_name(0))
else:
    print('device_name', 'N/A')

x = torch.randn(2, 3, device='cuda')
y = torch.randn(2, 3, device='cuda')
z = x + y
print('cuda_tensor_ok', z.shape, z.device, z.sum().item())
