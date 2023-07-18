import logging
import os, sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

base_path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path="{dir}/.env".format(dir=base_path))

logger = logging.getLogger('AB Pipeline')
logger.setLevel(os.getenv("LOG_LEVEL"))
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

PATH_METRIC_CONFIGS = "params/metrics/"
DEFAULT_ESTIMATOR = "t_test_linearization"
DEFAULT_METRIC_TYPE = "ratio"
DEFAULT_UNIT_LEVEL = "client_id"
DEFAULT_VALUE = "Unknown"
VARIANT_COL = "experiment_variant"
USER_ID_COL = "client_id"


