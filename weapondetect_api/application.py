"""
Run a rest API exposing the weapon detection model
"""
import argparse
import io
import pandas as pd
from PIL import Image
import torch
from flask import Flask, request
from flask_restful import Resource, Api
from utils.general import check_requirements
import os
application = app = Flask(__name__)
api = Api(app)


class weapondetect(Resource):
    def __init__(self):
        pass

    def get(self):
        return 200

    def post(self):
        '''if not request.method == "POST":
            return "ERROR"'''

        if request.values:  # todo make validation is a image was sent not the name
            image_name = request.values['image_name']
            image_file = request.files["image"]
            image_bytes = image_file.read()
            filename, file_extension = os.path.splitext(image_name)
            #print(f"Name is {filename} and ext is {file_extension}")
            if file_extension in ['.jpg','.jpeg']:
                img = Image.open(io.BytesIO(image_bytes))
                #img.show()
                data = pd.DataFrame()
                response = pd.DataFrame()

                results = model2(img, size=640)
                if results.pandas().xyxy[0].size > 0:
                    response = results.pandas().xyxy[0]
                    data = results.pandas().xywhn[0]
                results.render()


                results = model(img, size=640)
                if results.pandas().xyxy[0].size > 0:
                    response = response.append(results.pandas().xyxy[0], ignore_index=True)
                    data = data.append(results.pandas().xywhn[0], ignore_index=True)
                results.render()

                #for pic in results.imgs:
                    #img_base64 = Image.fromarray(pic)  todo get both model outputs on same image
                    #img_base64.save(f"./data/labeled_images/{name}", format="JPEG")  todo get both model outputs on same image

                if not data.empty:
                    names = data['name'].tolist()
                    if 'Weapon' in names:
                        data.iloc[data[data['name'] == 'Weapon'].index, 5] = 81  # just replaces the id of weapons from 0 to 81
                    #print(data)
                    data_list = data.values.tolist()

                    with open('./data/ID.txt', 'r') as r:
                        id = int(r.read()) + 1
                    with open('./data/ID.txt', 'w') as w:
                        w.write(str(id))

                    name = f"{id}_{image_name}"  # assign a name to the image
                    #print(f"NAME IS {name}")
                    img.save(f"./data/images/{name}")  # save a copy of the image
                    with open(f"./data/labels/{name[:-4]}.txt", 'w') as f:  # write the detections to a label in yolov5 format
                        index = 0
                        for line in data_list:
                            p_string = ''
                            for i in [5,0,1,2,3]:
                                if i == 3:
                                    p_string += f"{line[i]}"
                                else:
                                    p_string += f"{line[i]} "

                            '''listToStr = ' '.join(map(str, line))'''
                            f.write(p_string)
                            index += 1
                            if index < len(data_list):
                                f.write('\n')

                    response = response.to_dict()
                    return response
                else:
                    return "No objects found."
            else:
                return "Not a valid image format"
        else:
            return "NO NAME RECEIVED"


api.add_resource(weapondetect, '/api')


if __name__ == "__main__":
    check_requirements()
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load(".", "custom", path='./weapon_detector.pt', source='local').autoshape()
    model.eval()
    model2 = torch.hub.load(".", "custom", path='./yolov5x.pt', source='local').autoshape()
    model2.eval()

    app.run(host="0.0.0.0", port=args.port, debug=False)  # debug=True causes Restarting with stat
