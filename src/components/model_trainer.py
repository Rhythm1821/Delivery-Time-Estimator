import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

