import os
import random
import numpy as np
import tensorflow as tf


def seed_everything(seed: int = 0) -> None:
    """
    Seed random everything
    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
