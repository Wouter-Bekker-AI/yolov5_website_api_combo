"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import requests
import argparse
import numpy as np
import tinify
import cv2

from pandas import DataFrame
from flask import Flask, render_template, request, redirect

application = Flask(__name__)


def comp_save_img(image):
    """Try and use the tinify api to compress image that is sent back to user.
    Only if the correct tinify.key is passed in the --key arg.
    If no or incorrect key is passed then just send back full image
    compressions limited to 500 per month then it will revert to sending back normal image"""
    global try_comp
    if try_comp:
        try:
            tinify.validate()
            compression_count = tinify.compression_count
            if compression_count < 500:
                compress = True
            else:
                compress = False
        except tinify.Error:
            compress = False
            try_comp = False
            cv2.imwrite("static/image_OUT.jpg", image)
        if compress:
            source = tinify.from_file(image)  # Tinify the result for faster transfer
            source.to_file("static/image_OUT.jpg")
    else:
        cv2.imwrite("static/image_OUT.jpg", image)


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=2):
    """The API sends back the coordinates of the boxes in the form top left xy bottom right xy.
    This function plots the boxes on the image that will be send back."""
    see_through = True  # Create see through labels
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # x comes in as top left x,y and bottom left x,y
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # Object Box

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 2, fontScale=tl / 3, thickness=tf)[0]
        c3 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # c1 becomes bottom left x,y and c3 top right of x,y of label_box

        if see_through:  # First we crop the label area from the image
            label_area = im[c3[1]:c1[1], c1[0]:c3[0]]  # y is first section x is second section

            for i in range(3):  # Then we merge the bbox color with label area using a weighted sum
                label_area[:, :, i] = label_area[:, :, i] * 0.5 + color[i] * 0.3

            im[c3[1]:c1[1], c1[0]:c3[0]] = label_area  # Insert the label area back into the image
            label_frame_color = np.array(color) / 2  # To give the frame a light border

            cv2.rectangle(im, c1, c3, label_frame_color, 1, cv2.LINE_AA)  # Label Box See_Through
        else:
            cv2.rectangle(im, c1, c3, color, -1, cv2.LINE_AA)  # Label Box Filled

        cv2.putText(im, label, (c1[0], c1[1] - 2), 2, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return np.asarray(im)


@application.route("/tinify", methods=["PUT"])
def change_tinify():
    """To remotely change the try_comp variable"""
    global try_comp
    if request.method == "PUT":
        try_comp = not try_comp  # Flip boolean value
        return f'try_comp value changed to {try_comp}'


@application.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        image_name = file.filename
        if not file.content_type == 'image/jpeg':
            return 'That is not a valid image. Please click the back button to return to the upload screen.'
        to_panda = True
        visulize = True
        #api_url = "http://objectdetect.ddns.net:5000/api"
        api_url = "http://102.141.179.196:5000/api"
        file.save('./static/image_IN.jpg')

        subject_image = './static/image_IN.jpg'
        image_data = open(subject_image, "rb").read()
        image_payload = {"image": image_data}
        data_payload = {"image_name": image_name}
        try:
            response = requests.post(api_url, files=image_payload, data=data_payload).json()
        except Exception:
            return "Object Detection API is currently down for maintenance. Please check back later."
        if type(response) is dict:
            if to_panda:
                data = DataFrame(response).round(2)
                names = data['name'].tolist()
                if 'Weapon' in names:
                    data.iloc[data[data['name'] == 'Weapon'].index.astype('int64'), 5] = 81
                    # ^^^This just replaces the id of weapons from 0 to 81
                if visulize:
                    img = cv2.imread('./static/image_IN.jpg')
                    for i in range(len(data)):
                        box = [data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], data.iloc[i][3]]
                        lbl = f"{data.iloc[i][6]} {round(data.iloc[i][4], 2)}%"
                        clr = [255, 255, 255]  # BGR
                        clr[0] = int(255 - data.iloc[i][5] * 2)
                        clr[1] = int(255 - data.iloc[i][5])
                        clr[2] = int(data.iloc[i][5] * 3)
                        plot_one_box(box, img, clr, lbl)
                    comp_save_img(img)

        else:
            img = cv2.imread('./static/image_IN.jpg')
            comp_save_img(img)
        return render_template("return.html")

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask web_page for object detection")
    parser.add_argument("--port", default=80, type=int, help="port number")
    parser.add_argument("--key", type=str, help="tinify api key")
    parser.add_argument("--comp", type=bool, default=True, help="tinify api key")
    args = parser.parse_args()
    tinify.key = args.key
    try_comp = args.comp

    application.run(host="0.0.0.0", port=args.port, debug=False)  # debug=True causes Restarting with stat
