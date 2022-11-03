import os

# read the names of the query images
query_images_path = "../slice25/query"
query_images_names = sorted(os.listdir(query_images_path))

# write the query image name into the image-list file one by one
img_list_text_path = "../slice25/test-image-list.txt"
with open(img_list_text_path, "w") as f:
    for query_img_name in query_images_names:
        f.write(query_img_name)
        f.write("\n")

