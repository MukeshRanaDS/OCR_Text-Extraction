import os
import cv2
import argparse
from path import Path
from typing import List
import matplotlib.pyplot as plt
from word_detector import detect, prepare_img, sort_multiline
from config import page_path

list_img_names_serial = []
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=Path(page_path))
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--sigma', type=float, default=11)
parser.add_argument('--theta', type=float, default=7)
parser.add_argument('--min_area', type=int, default=100)
parser.add_argument('--img_height', type=int, default=1000)
parsed = parser.parse_args()

print("File path: ", parsed.data)


def get_img_files(data_dir: Path) -> List[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res


def save_image_names_to_text_files():
    for fn_img in get_img_files(parsed.data):
        print(f'Processing file {fn_img}')

        # load image and process it
        img = prepare_img(cv2.imread(fn_img), parsed.img_height)
        detections = detect(img,
                            kernel_size=parsed.kernel_size,
                            sigma=parsed.sigma,
                            theta=parsed.theta,
                            min_area=parsed.min_area)

        # sort detections: cluster into lines, then sort each line
        lines = sort_multiline(detections)

        # plot results
        plt.imshow(img, cmap='gray')
        num_colors = 7
        colors = plt.cm.get_cmap('rainbow', num_colors)
        for line_idx, line in enumerate(lines):
            for word_idx, det in enumerate(line):
                xs = [det.bbox.x, det.bbox.x, det.bbox.x +
                      det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h,
                      det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                plt.plot(xs, ys, c=colors(line_idx % num_colors))
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
                print(det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h)
                crop_img = img[det.bbox.y:det.bbox.y +
                                          det.bbox.h, det.bbox.x:det.bbox.x + det.bbox.w]

                path = r"D:\iam\extracted_word_imgs_1"
                # Check whether the specified
                # path exists or not
                isExist = os.path.exists(path)
                if isExist == False:
                    os.mkdir(path)
                    print("Directory Created")

                cv2.imwrite(f"{path}/L" + str(line_idx) + "W" +
                            str(word_idx) + ".jpg", crop_img)
                full_img_path = "L" + \
                                str(line_idx) + "W" + str(word_idx) + ".jpg"
                list_img_names_serial.append(full_img_path)
                print(list_img_names_serial)
                list_img_names_serial_set = set(list_img_names_serial)

                textfile = open(r"D:\iam\extracted_dataset_sequence\img_names_sequence.txt", "w")
                for element in list_img_names_serial:
                    textfile.write(element + "\n")
                textfile.close()
        plt.show()


save_image_names_to_text_files()