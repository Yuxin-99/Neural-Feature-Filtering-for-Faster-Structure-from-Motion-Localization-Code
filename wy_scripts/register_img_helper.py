import glob


def write_query_img_txt(img_list_text_path, query_images_names):
    with open(img_list_text_path, "w") as f:
        for query_img_name in query_images_names:
            if "img" in query_img_name:
                f.write(query_img_name)
                f.write("\n")


# this will generate the text file needed for model aligner
# the resulting file will contain base images names with their corresponding reference model centers
def gen_base_cam_centers_txt(base_images_dir, reference_model_images_path):
    images = []
    base_images = []

    for file in glob.glob(base_images_dir+"/*.jpg"):
        name = file.split("/")[-1]
        base_images.append(name)

    with open(reference_model_images_path) as f:
        lines = f.readlines()

    for line in lines:
        image_name = line.split(" ")[0]
        if image_name in base_images:
            x = line.split(" ")[-4]
            y = line.split(" ")[-3]
            z = line.split(" ")[-2]
            images.append(image_name + " " + x + " " + y + " " + z)

    with open(base_images_dir+'/../base_images_cam_centers.txt', 'w') as f:
        for image in images:
            f.write("%s\n" % image)
