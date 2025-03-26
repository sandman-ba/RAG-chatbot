import torch

print("CUDA version: ", torch.cuda.is_available())
print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
