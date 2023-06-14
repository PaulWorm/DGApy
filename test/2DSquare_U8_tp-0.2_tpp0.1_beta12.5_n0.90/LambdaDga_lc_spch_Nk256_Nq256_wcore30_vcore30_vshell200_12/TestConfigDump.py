import numpy as np
from ruamel.yaml import YAML
import dga.config as config


conf_file = YAML().load(open('./config.yaml'))

dga_config = config.DgaConfig(conf_file)

