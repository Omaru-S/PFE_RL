# policies.py
import numpy as np
from matrix_utils import nrows_communication_matrix, nrows_vision_matrix

def naive_policy(step: int):
    """
    Naive policy: send all messages unconditionally.

    Parameters
    ----------
    step : int
        The time step index to determine the number of agents.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        is_send_cam: shape (m,)
        is_send_cpm: shape (m,)
    """
    m = nrows_communication_matrix(step)  # total agents for CAM
    is_send_cam = np.ones(m, dtype=int)

    # CPM agents are a subset (or equal in size) of CAM agents
    p = nrows_vision_matrix(step)  # total CPM-capable agents
    is_send_cpm = np.zeros(m, dtype=int)
    is_send_cpm[:p] = 1

    return is_send_cam, is_send_cpm
