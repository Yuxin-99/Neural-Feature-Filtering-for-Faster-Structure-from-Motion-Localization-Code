import cv2
import pycolmap
from database import COLMAPDatabase
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import os

from read_model import read_images_binary

eps = 1e-15


def evaluate_est_pose(poses_est, params):
    print("start to compute errors")
    # read the ground truth poses
    poses_gt = {}

    # read the corresponding camera data from images.bin
    images = read_images_binary(params.query_gt_img_bin_path)
    for img in images.values():
        rt = np.r_[img.qvec, img.tvec]
        poses_gt[img.name] = rt

    # if gt_from_model:
    #
    # else:
    #     # read from the provided ground truth text files
    #     poses_gt_path = params.query_camera_poses_folder
    #     poses_gt_files = os.listdir(poses_gt_path)
    #     for pose_gt_file in poses_gt_files:
    #         path = poses_gt_path + "/" + pose_gt_file
    #         with open(path) as f:
    #             for line in f:
    #                 pose_gt = line.split(" ")
    #                 poses_gt[pose_gt[0]] = np.array(pose_gt[1:8], dtype=np.float64)

    # loop over the estimated poses to compute the error
    r_errors = {}
    t_errors = {}
    session_id = params.dataset_id
    for img_nm in poses_est.keys():
        print("evaluate the query image: " + img_nm, end="\r")
        gt_img_nm = "session_" + session_id + "/" + img_nm
        rvec_est = poses_est[img_nm][0:3]
        r_est_matrix = cv2.Rodrigues(rvec_est)[0]
        rq_est = matrix_to_quaternion(r_est_matrix)
        t_est = poses_est[img_nm][3:]

        pose_gt = poses_gt.get(gt_img_nm)
        if pose_gt is None:
            print("Couldn't find ground truth of " + gt_img_nm + " in gt/model/images.bin")
            continue
        rq_gt = poses_gt[gt_img_nm][0:4]
        rq_gt_matrix = quaternion_to_matrix(rq_gt)
        t_gt = poses_gt[gt_img_nm][4:]
        # if the ground truth is read from the provided img text file provided by the dataset,
        # then need to convert it from a camera center to a translation vector
        # if not gt_from_model:
        #     camera_center = np.array([t_gt]).transpose()
        #     t_gt = - rq_gt_matrix.dot(camera_center)

        camera_center_est = -(r_est_matrix.transpose()).dot(t_est)
        t_gt = -(rq_gt_matrix.transpose()).dot(t_gt)
        # if gt_from_model:
        #     t_gt = -(rq_gt_matrix.transpose()).dot(t_gt)

        r_err = compute_rot_quat_err(rq_gt, rq_est)
        # r_err = compute_rot_mx_err(rq_gt_matrix, r_est_matrix)
        # t_err = compute_trans_error(t_gt, t_est, 1)
        t_err = compute_trans_error(t_gt, camera_center_est[:, 0], 1)
        r_errors[img_nm] = r_err
        t_errors[img_nm] = t_err

    print()
    print("finish computing errors")

    r_errs = np.array([list(r_errors.values())]).transpose()
    r_err_avg = np.sum(r_errs) / len(r_errors)
    print("Average rotation error: " + str(r_err_avg) + ", no of images " + str(len(r_errors)))

    t_errs = np.array([list(t_errors.values())]).transpose()
    t_err_avg = np.sum(t_errs) / len(t_errs)
    print("Average translation error: " + str(t_err_avg) + ", no of images " + str(len(r_errors)))

    maa = ComputeMaa(r_errs, t_errs)
    print(f'Mean average Accuracy: ": {maa[0]:.05f}')

    # draw a bar plot for the rotation error
    draw_error_plt(r_errors.keys(), list(r_errors.values()), "Rotation", "degrees", r_err_avg, params.results_path)
    # draw a bar plot for the translation error
    draw_error_plt(t_errors.keys(), list(t_errors.values()), "Translation", "meters", t_err_avg, params.results_path)

    # save the errors
    np.save(params.pose_rot_err_save_path, r_errors)
    np.save(params.pose_translation_err_save_path, t_errors)


def get_image_pose(db, query_image):
    qx = db.execute("SELECT prior_qw FROM images WHERE name = " + "'" + query_image + "'")
    qx = str(qx.fetchone()[0])
    return qx


# compute_error and matrix_to_quaternion from
# https://www.kaggle.com/code/eduardtrulls/imc2022-training-data?scriptVersionId=92062607
def compute_rot_quat_err(q_gt, q):
    """Compute the error metric for a single example.
    The function returns two errors, over rotation (as quaternion) and translation.
    These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy."""

    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    # q_gt_norm = q_gt / np.linalg.norm(q_gt)
    # q_norm = q / np.linalg.norm(q)
    # err_q = 2 * np.arccos(np.sum(q_norm * q_gt_norm))

    return err_q * 180 / np.pi


def compute_trans_error(T_gt, T, scale):
    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_t


def compute_rot_mx_err(rot_gt, rot):
    """The function returns the error over rotation (as matrix) """

    m = np.transpose(rot_gt).dot(rot)
    err_r = np.arccos((np.trace(m) - 1) / 2)

    return err_r * 180 / np.pi


def ComputeMaa(err_q, err_t):
    """Compute the mean Average Accuracy at different tresholds, for one scene."""
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10)

    assert len(err_q) == len(err_t)

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)


def matrix_to_quaternion(matrix):
    """Transform a rotation matrix into a quaternion."""
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # The quaternion is the eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q


def quaternion_to_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def draw_error_plt(img_names, errors, err_name, unit, err_avg, save_path):
    x_name = range(len(img_names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_name, errors, width=3, log=True)
    ax.set_xlim(xmin=-0, xmax=len(img_names))
    ax.set_title(err_name + "Error (" + unit + ') Per Query Image')
    ax.set_xlabel('Image')
    # ax.set_ylabel(err_name + ' Error')

    ax.axhline(err_avg, color='red', linewidth=1, label="Average: " + "{0:.3f} ".format(err_avg) + unit)
    ax.legend(loc='upper left')
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, err_avg, "{0:.3f}".format(err_avg), color="red", transform=trans,
            ha="right", va="center")

    plt.savefig(save_path + err_name + 'Error.png', dpi=300, bbox_inches='tight')
    plt.show()
