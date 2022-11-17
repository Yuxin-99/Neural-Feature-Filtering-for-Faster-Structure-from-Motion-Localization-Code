import re
import sys

import colmap
from parameters import Parameters


def main():
    # construct the parameters
    base_path = sys.argv[1]  # e.g.: ../Dataset/slice7
    # assume the dataset number is included in the directory name
    dataset_id = int(re.search(r'\d+', base_path).group())
    dataset_id = str(dataset_id)
    params = Parameters(base_path, dataset_id)
    colmap.feature_extractor(params.query_db_path, params.query_img_folder, query=True)


if __name__ == "__main__":
    main()