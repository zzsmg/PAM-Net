import sys, os

dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)
from PAM_FFA import PAM_FFA
from PerceptualLoss import LossNetwork as PerLoss
from TotalLoss import CustomLoss_function as Total