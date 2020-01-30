# Package info

import os
import sys

idx = os.getcwd().rfind('quantum_HEOM')
ROOT_DIR = os.getcwd()[:idx] + 'quantum_HEOM'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                                 '..')))
