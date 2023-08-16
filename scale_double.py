import cv2
import argparse
import os

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

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default=None, required=True, type=str, help='Depth image from slam')

    parser.add_argument("--output", default=None, type=str, help="dir to save slam depth image with same type of tof")

    args = parser.parse_args()

    root_len = len(args.input)
    aanet_images = []

    if os.path.isdir(args.input):
        paths = Walk(args.input, ['png'])
        for image_name in paths:
            aanet_images.append(image_name)

    for image_name in aanet_images:
        image = cv2.imread(image_name)
        image_double_write = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
        image_write_file = os.path.join(args.output, "double",image_name[root_len+1:])
        print(image_name, image_write_file)
        MkdirSimple(image_write_file)
        cv2.imwrite(image_write_file, image_double_write)

