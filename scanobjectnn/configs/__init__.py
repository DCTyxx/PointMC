import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from scanobjectnn.configs import scanobjectnn_l


model_configs = {
    'l': scanobjectnn_l.ModelConfig(),
}
