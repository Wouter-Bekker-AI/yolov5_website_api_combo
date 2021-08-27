# Plotting utils
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=2):
    see_through = False
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # x comes in as top left x,y and bottom left x,y
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # Object Box
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 2, fontScale=tl / 3, thickness=tf)[0]
        c3 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # c1 becomes bottom left x,y and c3 top right of x,y of label_box

        if see_through:
            # First we crop the label area from the image
            label_area = im[c3[1]:c1[1], c1[0]:c3[0]]  # y is first section x is second section

            for i in range(3):  # Then we merge the bbox color with label area using a weighted sum
                label_area[:, :, i] = label_area[:, :, i] * 0.5 + color[i] * 0.2

            im[c3[1]:c1[1], c1[0]:c3[0]] = label_area  # Insert the label area back into the image
            label_frame_color = np.array(color) / 2  # To give the frame a light border

        if see_through:  # Plot  see_through label or normal on image
            cv2.rectangle(im, c1, c3, label_frame_color, 1, cv2.LINE_AA)  # Label Box See_Through
        else:
            cv2.rectangle(im, c1, c3, color, -1, cv2.LINE_AA)  # Label Box Filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 2, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return np.asarray(im)


def plot_one_box_PIL(box, im, color=(128, 128, 128), label=None, line_thickness=None):
    # Plots one bounding box on image 'im' using PIL
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    line_thickness = line_thickness or max(int(min(im.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=color)  # plot
    if label:
        font = ImageFont.truetype("Arial.ttf", size=max(round(max(im.size) / 40), 12))
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=color)
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(im)
