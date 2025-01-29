from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from importlib.resources import files
import pytest
from tests.helpers.reference_solutions import fourier_analysis

print(files("biotsavart_modes"))