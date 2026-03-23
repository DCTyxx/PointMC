import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from shapenetpart.configs import shapenetpart_l

model_configs = {
    'l': shapenetpart_l.ModelConfig(),
}
