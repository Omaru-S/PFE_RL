import numpy as np
from matrix_utils import read_communication_matrix, read_vision_matrix, read_capabilities
from policies import naive_policy


def com_bernoulli_step(step: int) -> np.ndarray:
    """
    One‐step communication simulation.

    Parameters
    ----------
    step : int
        Index of the communication-probability file (…/step_{step}.npz).

    Returns
    -------
    np.ndarray
        Binary matrix of the same shape as the probability matrix:
        1  → message was successfully delivered
        0  → delivery failed or was never attempted (probability was 0)
    """
    # Load P(success) for every sender → receiver pair
    prob_matrix = read_communication_matrix(step)      # shape (N,N)

    # Uniform[0,1) draw for every entry
    trials = np.random.random(prob_matrix.shape)

    # A delivery succeeds when U < p; zeros stay zero automatically
    success_matrix = (trials < prob_matrix).astype(np.uint8)

    return success_matrix

def max_communication_step(step : int) ->  np.ndarray:
    # Load P(success) for every sender → receiver pair
    prob_matrix = read_communication_matrix(step)      # shape (N,N)


    # A delivery succeeds when U < p; zeros stay zero automatically
    success_matrix = (0.0 < prob_matrix).astype(np.uint8)

    return success_matrix


def vision_bernoulli_step(step: int) -> np.ndarray:
    """
    One‐step communication simulation.

    Parameters
    ----------
    step : int
        Index of the communication-probability file (…/step_{step}.npz).

    Returns
    -------
    np.ndarray
        Binary matrix of the same shape as the probability matrix:
        1  → message was successfully delivered
        0  → delivery failed or was never attempted (probability was 0)
    """
    # Load P(success) for every sender → receiver pair
    prob_matrix = read_vision_matrix(step)      # shape (N,N)

    # Uniform[0,1) draw for every entry
    trials = np.random.random(prob_matrix.shape)

    # A delivery succeeds when U < p; zeros stay zero automatically
    success_matrix = (trials < prob_matrix).astype(np.uint8)

    return success_matrix

def max_vision_step(step : int) ->  np.ndarray:
    # Load P(success) for every sender → receiver pair
    prob_matrix = read_vision_matrix(step)      # shape (N,N)

    # A delivery succeeds when U < p; zeros stay zero automatically
    success_matrix = (0.0 < prob_matrix).astype(np.uint8)

    return success_matrix

# ────────────────────────────────────────────────────────────────
# 1)  Knowledge + objects_in_vision
# ────────────────────────────────────────────────────────────────
def compute_knowledge(
    step: int,
    is_send_cam,               # shape (m,)
    is_send_cpm,               # shape (m,)
):
    """
    Returns
    -------
    tuple[list[float], list[int]]
        (knowledge, objects_in_vision)
          • knowledge ........ length-m   (CAM vehicles)
          • objects_in_vision  length-p   (CPM vehicles)
    """
    # --- cast control vectors --------------------------------------------------
    is_send_cam = np.asarray(is_send_cam, dtype=int).reshape(-1)
    is_send_cpm = np.asarray(is_send_cpm, dtype=int).reshape(-1)

    # --- Bernoulli draws -------------------------------------------------------
    com_success    = com_bernoulli_step(step)      # (m, m)
    vision_success = vision_bernoulli_step(step)   # (p, n)

    m = com_success.shape[0]          # CAM count
    p = vision_success.shape[0]       # CPM count (p ≤ m)

    # --- mute CAM rows ---------------------------------------------------------
    com_success *= is_send_cam[:, None].astype(np.uint8)
    np.fill_diagonal(com_success, 1)

    # --- CPM-directed sub-matrix & mute rows -----------------------------------
    cpm_com_success = com_success[:, :p]
    cpm_com_success *= is_send_cpm[:, None].astype(np.uint8)
    diag_idx = np.arange(p)
    cpm_com_success[diag_idx, diag_idx] = np.where(
        is_send_cpm[diag_idx] == 0, 1, cpm_com_success[diag_idx, diag_idx]
    )
    # --- finish pipeline -------------------------------------------------------
    objects_in_vision = vision_success.sum(axis=1, keepdims=True)   # (p, 1)
    cam_success       = com_success.sum(axis=1, keepdims=True)      # (m, 1)

    knowledge = cam_success + cpm_com_success.dot(objects_in_vision)

    # Return both pieces, already flattened
    return (
        knowledge.flatten().tolist(),           # length-m
        objects_in_vision.flatten().tolist(),   # length-p
    )

def compute_max_knowledge(    step: int,
    is_send_cam,               # shape (m,)
    is_send_cpm,               # shape (m,)
):
    """
    Returns
    -------
    tuple[list[float], list[int]]
        (knowledge, objects_in_vision)
          • knowledge ........ length-m   (CAM vehicles)
          • objects_in_vision  length-p   (CPM vehicles)
    """
    # --- cast control vectors --------------------------------------------------
    is_send_cam = np.asarray(is_send_cam, dtype=int).reshape(-1)
    is_send_cpm = np.asarray(is_send_cpm, dtype=int).reshape(-1)

    # --- Bernoulli draws -------------------------------------------------------
    com_success    = max_communication_step(step)    # (m, m)
    vision_success = max_vision_step(step)   # (p, n)

    m = com_success.shape[0]          # CAM count
    p = vision_success.shape[0]       # CPM count (p ≤ m)

    # --- mute CAM rows ---------------------------------------------------------
    com_success *= is_send_cam[:, None].astype(np.uint8)
    np.fill_diagonal(com_success, 1)

    # --- CPM-directed sub-matrix & mute rows -----------------------------------
    cpm_com_success = com_success[:, :p]
    cpm_com_success *= is_send_cpm[:, None].astype(np.uint8)
    diag_idx = np.arange(p)
    cpm_com_success[diag_idx, diag_idx] = np.where(
        is_send_cpm[diag_idx] == 0, 1, cpm_com_success[diag_idx, diag_idx]
    )
    # --- finish pipeline -------------------------------------------------------
    objects_in_vision = vision_success.sum(axis=1, keepdims=True)   # (p, 1)
    cam_success       = com_success.sum(axis=1, keepdims=True)      # (m, 1)

    knowledge = cam_success + cpm_com_success.dot(objects_in_vision)

    # Return both pieces, already flattened
    return (
        knowledge.flatten().tolist(),           # length-m
        objects_in_vision.flatten().tolist(),   # length-p
    )

# ────────────────────────────────────────────────────────────────
# 2)  Bytes transmitted this step
# ────────────────────────────────────────────────────────────────
def compute_bytes(
    is_send_cam,              # length-m, 0/1
    is_send_cpm,              # length-m, 0/1
    objects_in_vision,        # length-p, ints
) -> list[int]:
    """
    Compute bytes sent by each agent this step.

    - CAM: 190 bytes if is_send_cam[i] == 1
    - CPM: 121 + 35 * objects_in_vision[i], only if is_send_cpm[i] == 1

    Returns
    -------
    list[int]
        Concatenated list of CAM and CPM bytes: [cam_0, ..., cam_m-1, cpm_0, ..., cpm_p-1]
    """
    # Cast and mask
    is_send_cam       = np.asarray(is_send_cam, dtype=int).reshape(-1)      # (m,)
    is_send_cpm       = np.asarray(is_send_cpm, dtype=int).reshape(-1)      # (m,)
    objects_in_vision = np.asarray(objects_in_vision, dtype=int).reshape(-1)  # (p,)

    # Compute CAM and CPM bytes
    cam_bytes = (is_send_cam * 190).tolist()

    p = len(objects_in_vision)
    cpm_mask = is_send_cpm[:p]                  # CPM mask only applies to first p vehicles
    visible_objects = objects_in_vision * cpm_mask
    cpm_bytes = np.zeros(is_send_cpm.shape[0], dtype=int)
    cpm_bytes[:p] = (121*is_send_cpm[:p] + 35 * visible_objects)

    return (cam_bytes + cpm_bytes).flatten().tolist()


def sum_bytes(bytes_list: list[int]) -> int:
    """
    Sum the total number of bytes from the per-agent list.
    """
    return sum(bytes_list)

def sum_knowledge(knowledge: list[float]) -> float:
    """
    Sum the total knowledge across all agents.
    """
    return sum(knowledge)

# is_send_cam, is_send_cpm = naive_policy(1)
# knowledge, objects_in_vision = compute_max_knowledge(1,is_send_cam,is_send_cpm)
# print(knowledge)
# print(objects_in_vision)
# print(sum_bytes(compute_bytes(is_send_cam,is_send_cpm,objects_in_vision)))
# print(sum_knowledge(knowledge))