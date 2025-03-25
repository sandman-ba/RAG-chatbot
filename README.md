# LLMs for Sentiment and Behaviour Analysis

## PyTorch

If you want to use your GPU for `torch`, you must install the appropriate version for your GPU model. See the [PyTorch website](https://pytorch.org/get-started/locally/ 'PyTorch website') to find the correct version. For example, for an AMD graphics card run

    pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.2.4

inside your virtual environment. On the other hand, for an Nvidia card with CUDA 12.6 run instead

    pip3 install torch --index-url https://download.pytorch.org/whl/cu126
