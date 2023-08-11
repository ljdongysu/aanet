import cv2
import numpy as np
import argparse
import os

def GetDepthImgPSL(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 0
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 0
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)
    return depth_img_rgb.astype(np.uint8)

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def Walk(path, suffix:list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path,]

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    try:
        file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))
    except:
        pass
    return file_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default=None, required=True, type=str, help='Depth image from slam')

    parser.add_argument("--output", default=None, type=str, help="dir to save slam depth image with same type of tof")

    args = parser.parse_args()

    root_len = len(args.input)
    depth_yaml_files = []

    if os.path.isdir(args.input):
        paths = Walk(args.input, ['yaml'])
        for image_name in paths:
            if "img/depth" in image_name:
                depth_yaml_files.append(image_name)

    else:
        print("need --images for input images' dir")
        assert 0
    for yaml_file in depth_yaml_files:
        write_file = os.path.join(args.output, yaml_file[root_len+1:])
        print(type(write_file), write_file)
        write_file = write_file.replace(".yaml", ".png").replace("img/depth", "img/depth_psl")
        write_file_show = write_file.replace(".yaml", ".png").replace("img/depth", "img/disp")

        cv_file = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
        matrix = cv_file.getNode("depth").mat()

        matrix = matrix[:,:,2]
        matrix = np.nan_to_num(matrix)

        matrix *= 100
        print(np.max(matrix), np.min(matrix))
        disp_img = 3423/matrix
        image_write = GetDepthImgPSL(matrix)
        MkdirSimple(write_file)
        print(write_file)
        cv2.imwrite(write_file, image_write)

        MkdirSimple(write_file_show)
        print(write_file_show)
        disp_img = cv2.applyColorMap((disp_img).astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imwrite(write_file_show, disp_img)

