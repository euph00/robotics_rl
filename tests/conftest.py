"""The regression suite pins the course physics (unlimited motors, mu 0.25 grip), which its
expected trajectories encode; tests/test_physics.py covers the realistic settings explicitly."""
import os

os.environ.setdefault('PNR_PHYSICS', 'course')
