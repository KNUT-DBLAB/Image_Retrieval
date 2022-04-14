import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import shutil
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split

# Our Modules
from utils import compute_nystrom, create_train_val_test_loaders, save_checkpoint, AverageMeter
from model import CNN



