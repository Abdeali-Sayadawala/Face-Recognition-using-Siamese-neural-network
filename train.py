# -*- coding: utf-8 -*-

import os 
import cv2
import numpy as np
from reconet import Reconet

dim = 200

x1 = np.load('training data/x1.npy')
x2 = np.load('training data/x2.npy')
y = np.load('training data/y.npy')

"""x1 = x1.reshape((x1.shape[0], dim**2*3)).astype(np.float32)
x2 = x2.reshape((x2.shape[0], dim**2*3)).astype(np.float32)"""

rec = Reconet()

parameters = {
        'batch_size' : 2,
        'epochs' : 7,
        'callbacks' : None,
        'val_data' : None
        }

rec.fit([x1, x2], y, hyperparameters=parameters)
rec.save_model('model45.h5')

