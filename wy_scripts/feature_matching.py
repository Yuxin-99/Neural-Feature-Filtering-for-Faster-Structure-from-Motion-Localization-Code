import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import time

from itertools import chain

from read_model import get_image_id


def get_keypoints_xy(db, image_id):
    query_image_keypoints_data = db.execute("SELECT data FROM keypoints WHERE image_id = " + "'" + image_id + "'")
    query_image_keypoints_data = query_image_keypoints_data.fetchone()[0]
    query_image_keypoints_data_cols = db.execute("SELECT cols FROM keypoints WHERE image_id = " + "'" + image_id + "'")
    query_image_keypoints_data_cols = int(query_image_keypoints_data_cols.fetchone()[0])
    query_image_keypoints_data = db.blob_to_array(query_image_keypoints_data, np.float32)
    query_image_keypoints_data_rows = int(np.shape(query_image_keypoints_data)[0] / query_image_keypoints_data_cols)
    query_image_keypoints_data = query_image_keypoints_data.reshape(query_image_keypoints_data_rows, query_image_keypoints_data_cols)
    query_image_keypoints_data_xy = query_image_keypoints_data[:, 0:2]
    return query_image_keypoints_data_xy


# indexing is the same as points3D indexing for trainDescriptors
def get_queryDescriptors(db, image_id):
    query_image_descriptors_data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + image_id + "'")
    query_image_descriptors_data = query_image_descriptors_data.fetchone()[0]
    query_image_descriptors_data = db.blob_to_array(query_image_descriptors_data, np.uint8)
    descs_rows = int(np.shape(query_image_descriptors_data)[0] / 128)
    query_image_descriptors_data = query_image_descriptors_data.reshape([descs_rows, 128])

    row_sums = query_image_descriptors_data.sum(axis=1)
    # query_image_descriptors_data = query_image_descriptors_data / row_sums[:, np.newaxis]
    queryDescriptors = query_image_descriptors_data.astype(np.float32)
    return queryDescriptors


def get_camera_matrix(db, query_img):
    camera_id = db.execute("SELECT camera_id FROM images WHERE name = " + "'" + query_img + "'")
    camera_id = str(camera_id.fetchone()[0])
    camera_params = db.execute("SELECT params FROM cameras WHERE camera_id = " + "'" + camera_id + "'")
    camera_params = camera_params.fetchone()[0]
    camera_data = db.blob_to_array(camera_params, np.float32)
    return camera_data


def feature_matcher_wrapper(db, query_images, trainDescriptors, points3D_xyz, params, clf_model, verbose=False, points_scores_array=None):
    ratio_test_val = params.ratio_test_val
    session_id = params.dataset_id
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    matching_time = {}
    matchable_threshold = 0.5

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        query_image_name = "session_" + session_id + "/" + query_image
        image_id = get_image_id(db, query_image_name)
        # keypoints data
        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)
        running_time = 0

        if clf_model is not None:
            start_time = time.perf_counter()
            pred_matchable = clf_model.predict(queryDescriptors)
            elapsed_time = time.perf_counter() - start_time
            running_time += elapsed_time

            matchable_desc_indices = np.where(pred_matchable > matchable_threshold)[0]
            keypoints_xy = keypoints_xy[matchable_desc_indices]
            queryDescriptors = queryDescriptors[matchable_desc_indices]

            # matrix = get_camera_matrix(db, query_image)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=0)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
        start_time = time.perf_counter()
        matcher = cv2.BFMatcher()
        # Matching on trainDescriptors (remember these are the means of the 3D points)
        temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # m the closest, n is the second closest
        good_matches = []
        for m, n in temp_matches:       # TODO: maybe consider what you have at this point? and add it to the if condition?
            assert(m.distance <= n.distance)
            # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
            if m.distance < (ratio_test_val * n.distance):          # and (score_m > score_n):
                if m.queryIdx >= keypoints_xy.shape[0]:
                    # keypoints_xy.shape[0] always same as queryDescriptors.shape[0]
                    raise Exception("m.queryIdx error!")
                if m.trainIdx >= points3D_xyz.shape[0]:
                    raise Exception("m.trainIdx error!")
                # idx1.append(m.queryIdx)
                # idx2.append(m.trainIdx)
                scores = []
                xy2D = keypoints_xy[m.queryIdx, :].tolist()
                xyz3D = points3D_xyz[m.trainIdx, :].tolist()

                if points_scores_array is not None:
                    for points_scores in points_scores_array:
                        scores.append(points_scores[0, m.trainIdx])
                        scores.append(points_scores[0, n.trainIdx])

                match_data = [xy2D, xyz3D, [m.distance, n.distance], scores]
                match_data = list(chain(*match_data))
                good_matches.append(match_data)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        running_time += elapsed_time
        matching_time[query_image] = running_time
        matches[query_image] = np.array(good_matches)
        matches_sum.append(len(good_matches))
        if verbose:
            print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image +
                  ", good match: " + str(len(good_matches)) + ", time: " + str(elapsed_time), end="\r")

    if verbose:
        print()
        total_all_images = np.sum(matches_sum)
        print("Total matches: " + str(total_all_images) + ", no of images " + str(len(query_images)))
        matches_all_avg = total_all_images / len(matches_sum)
        print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(query_images)))

    draw_time_plt(matching_time, params.results_path)
    return matches, matching_time


def draw_time_plt(img_times, save_path):
    img_names = list(img_times.keys())
    times = list(img_times.values())
    x_name = range(len(img_names))
    avg_time = sum(times) / len(times)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_name, times, width=3, log=False)
    ax.set_xlim(xmin=0, xmax=len(img_times))
    ax.set_title('Feature Matching Time (seconds) Per Query Image')
    ax.set_xlabel('Image')
    # plt.ylabel('Feature Matching Time (seconds)')

    ax.axhline(avg_time, color='red', linewidth=1, label="Average: " + "{0:.3f} ".format(avg_time) + "seconds")
    ax.legend(loc='upper left')
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, avg_time, "{0:.3f}".format(avg_time), color="red", transform=trans,
            ha="right", va="center")

    plt.savefig(save_path + 'Feature_Matching_Time.png', dpi=300, bbox_inches='tight')
    plt.show()
