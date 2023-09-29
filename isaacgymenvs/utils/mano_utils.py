import numpy as np
import argparse
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F


def rotate_pcd_with_quaternion(verts, quaternion):
    norm = torch.norm(quaternion, dim=1, keepdim=True)
    quaternion = quaternion / norm

    rot_matrix = quaternion_to_matrix(quaternion)

    # Rotate vertices
    verts_rot = torch.einsum('bij,bkj->bki', rot_matrix, verts)

    return verts_rot

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
def quaternion_multiply(q, r):
    """
    Multiply two sets of quaternions.

    Parameters:
    q (torch.Tensor): A tensor of shape (n, 4) representing n quaternions.
    r (torch.Tensor): A tensor of shape (n, 4) representing n quaternions.

    Returns:
    torch.Tensor: A tensor of shape (n, 4) representing the product of the input quaternions.
    """
    # Ensure the input tensors have the correct shape
    assert q.shape[1] == 4 and r.shape[1] == 4, "Input tensors must have shape (n, 4)"

    # Extract the components of the input quaternions
    x1, y1, z1, w1 = q.split(1, dim=1)
    x2, y2, z2, w2 = r.split(1, dim=1)

    # Compute the product of the quaternions
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.cat((x3, y3, z3, w3), dim=1)

def dgrasp_to_mano_np(param):
    bs = param.shape[0]
    eulers = param[:,6:].reshape(bs,-1, 3).copy()

    # exchange ring finger and little finger's sequence
    temp = eulers[:,6:9].copy()
    eulers[:,6:9] = eulers[:,9:12]
    eulers[:,9:12] = temp

    eulers = eulers.reshape(-1,3)
    # change euler angle to axis angle
    rotvec = R.from_euler('XYZ', eulers, degrees=False)
    rotvec = rotvec.as_rotvec().reshape(bs,-1)
    global_orient = R.from_euler('XYZ', param[:,3:6], degrees=False)
    global_orient = global_orient.as_rotvec()

    # translation minus a offset
    offset = np.array([[0.09566993, 0.00638343, 0.00618631]])
    mano_param = np.concatenate([global_orient, rotvec, param[:,:3] - offset],axis=1)

    return mano_param
def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_ijkr = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_ijkr / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    ret = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))
    # exchange rijk -> ijkr
    return ret[..., [1, 2, 3, 0]]

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def euler_angles_to_quaternion(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to quaternions.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return matrix_to_quaternion(euler_angles_to_matrix(euler_angles, convention))

def euler_angles_to_axis_angle(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to axis/angle.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return matrix_to_axis_angle(euler_angles_to_matrix(euler_angles, convention))

def dgrasp_to_mano(param):
    bs = param.shape[0]

    eulers = param[:,6:].reshape(bs,-1, 3).clone()
    # exchange ring finger and little finger's sequence
    temp = eulers[:,6:9].clone()
    eulers[:,6:9] = eulers[:,9:12]
    eulers[:,9:12] = temp

    # change euler angle to axis angle
    rotvec = euler_angles_to_axis_angle(eulers, 'XYZ')
    rotvec = rotvec.reshape(bs,-1)
    global_orient = euler_angles_to_axis_angle(param[:,3:6], 'XYZ')

    # translation minus a offset
    offset = torch.tensor([[0.09566993, 0.00638343, 0.00618631]],device=param.device)
    mano_param = torch.cat([global_orient, rotvec, param[:,:3] - offset],axis=1)
    return mano_param



def show_pointcloud_objhand(hand, obj):
    '''
    Draw hand and obj xyz at the same time
    :param hand: [778, 3]
    :param obj: [3000, 3]
    '''


    hand_dim = hand.shape[0]
    obj_dim = obj.shape[0]
    handObj = np.vstack((hand, obj))
    c_hand, c_obj = np.array([[1, 0, 0]]), np.array([[0, 0, 1]]) # RGB
    c_hand = np.repeat(c_hand, repeats=hand_dim, axis=0) # [778,3]
    c_obj = np.repeat(c_obj, repeats=obj_dim, axis=0) # [3000,3]
    c_hanObj = np.vstack((c_hand, c_obj)) # [778+3000, 3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(handObj)
    pc.colors = o3d.utility.Vector3dVector(c_hanObj)
    o3d.visualization.draw_geometries([pc])


