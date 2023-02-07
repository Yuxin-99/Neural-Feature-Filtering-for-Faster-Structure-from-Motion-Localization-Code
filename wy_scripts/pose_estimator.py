import numpy as np
import cv2

# undistorted_img: Total number of images that have enough inliers: 1459
#                  Total inliers: 257691, no of images 1459
#                  Average matches per image: 176.62165867032215, no of images 1459


def do_pose_estimation(matches, query_images_names, query_img_path, est_save_path):
    c0_intrinsics = np.array([[868.993378, 0, 525.942323],
                              [0, 866.063001, 420.042529],
                              [0, 0, 1]])
    c1_intrinsics = np.array([[873.382641, 0, 529.324138],
                              [0, 876.489513, 397.272397],
                              [0, 0, 1]])
    c0_dist_coeff = np.array([-0.399431, 0.188924, 0.000153, 0.000571])
    c1_dist_coeff = np.array([-0.397066, 0.181925, 0.000176, -0.000579])
    # loop over each query image
    good_img_num = 0
    inliers_sum = []
    poses = {}
    for i in range(len(query_images_names)):
        query_img_nm = query_images_names[i]
        query_matches = matches.get(query_img_nm)
        if (query_matches is None) or (len(query_matches) < 4):
            print("No enough matches for the query image: " + query_img_nm, end="\r")
            continue

        # solve the camera pose matrix
        img_pnts = query_matches[:, 0:2]
        obj_pnts = query_matches[:, 2:5]
        # read the camera intrinsics from query.db
        # matrix = get_camera_matrix(query_db, )
        if "c0" in query_img_nm:
            camera_matrix = c0_intrinsics
        else:
            camera_matrix = c1_intrinsics
        ret_val, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=obj_pnts, imagePoints=img_pnts, cameraMatrix=camera_matrix,
                                                          distCoeffs=None, iterationsCount=1000, flags=cv2.SOLVEPNP_EPNP)   # SOLVEPNP_ITERATIVE
        if not ret_val:
            print("solvePnPRansac failed on the query image: " + query_img_nm, end="\r")
            continue

        if inliers is None:
            print("No inliers found when solving the query image: " + query_img_nm, end="\r")
            continue

        # project the 3d keypoint onto the query image frame
        inlier_matches = query_matches[inliers[:, 0]]
        inliers_sum.append(len(inlier_matches))
        inlier_img_pnts = inlier_matches[:, 0:2]
        inlier_obj_pnts = inlier_matches[:, 2:5]
        # dist_coeff = np.zeros()
        if rvec is None:
            rvec = np.zeros([3, 1])
        if tvec is None:
            tvec = np.zeros([3, 1])
        # shape(num_inlier, 1, 2)
        projected_pnts, _ = cv2.projectPoints(np.array(inlier_obj_pnts, dtype=np.float64), rvec, tvec, camera_matrix, distCoeffs=None)
        # points_est = project_obj_pnts(inlier_obj_pnts, rvec, tvec, camera_matrix)
        rt = np.r_[rvec, tvec]
        poses[query_img_nm] = rt

        # visualize the estimated 3d projected points
        query_img = cv2.imread(query_img_path + "/" + query_img_nm)
        query_img_with_pnts = draw_pnts_on_img(query_img, inlier_img_pnts, projected_pnts)
        # cv2.imshow("estimation of " + query_img_nm, query_img)
        # cv2.waitKey()
        save_name = str(good_img_num) + "_" + query_img_nm
        cv2.imwrite(est_save_path + save_name, query_img_with_pnts)
        good_img_num += 1

    print()

    print("Total number of images that have enough inliers: " + str(good_img_num))
    degenerate_pose_perc = (len(query_images_names) - good_img_num) / len(query_images_names)
    print("Percentage of degenerate poses: " + str(degenerate_pose_perc))
    total_all_images = np.sum(inliers_sum)
    print("Total inliers: " + str(total_all_images) + ", no of images " + str(good_img_num))
    matches_all_avg = total_all_images / len(inliers_sum)
    print("Average inliers per image: " + str(matches_all_avg) + ", no of images " + str(good_img_num))
    return poses, degenerate_pose_perc


def draw_pnts_on_img(query_img, img_pnts, projected_pnts):
    res_img = query_img
    for m in range(len(img_pnts)):
        # draw the original 2d points on the query image
        img_pnt = img_pnts[m]
        query_color = (255, 0, 0)
        res_img = cv2.circle(res_img, (int(img_pnt[0]), int(img_pnt[1])), 12, query_color, -1)

        # draw all the projected 3d points on the query image
        projected_pnt = projected_pnts[m][0]
        if np.isnan(projected_pnt[0]) or np.isnan(projected_pnt[1]):
            continue
        if (projected_pnt[0] >= 1024) or (projected_pnt[1] >= 768):
            continue
        projected_color = (0, 0, 255)
        res_img = cv2.circle(res_img, (int(projected_pnt[0]), int(projected_pnt[1])), 8, projected_color, -1)

    return res_img


def project_obj_pnts(obj_pnts, rvec, tvec, K):
    rotm = cv2.Rodrigues(rvec)[0]
    Rt = np.r_[(np.c_[rotm, tvec]), [np.array([0, 0, 0, 1])]]

    obj_points = np.hstack((obj_pnts, np.ones((obj_pnts.shape[0], 1))))  # make homogeneous
    img_point = K.dot(Rt.dot(obj_points.transpose())[0:3])
    img_point = img_point / img_point[2]
    img_point = img_point.transpose()
    return img_point[:, 0:2]