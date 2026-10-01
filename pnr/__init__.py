"""Course-compatible Pick–Nav–Reach integration and experimental controllers."""
import os

# Headless GPU rendering for every pnr module: the NVIDIA EGL device (the default device
# probe picks one that fails on this machine). Must be set before mujoco is imported.
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')

__version__ = "0.1.0"
