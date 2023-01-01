from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from keras.utils.layer_utils import get_source_inputs
from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from flask import jsonify
import keras.engine.topology as KE
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/', methods = ['GET','POST'])
def home():
    return render_template("upload.html")

@app.route('/compare', methods = ['POST'])
def compare():
    file1 = request.files.getlist("file")
    print(file1[0])
    file2 = request.files.getlist("file2")
    print(file2[0])
    filename1 = file1[0].filename
    filename2 = file2[0].filename
    target = os.path.join(APP_ROOT, 'images')
    destination1 = "/".join([target, filename1])
    destination2 = "/".join([target, filename2])
    file1[0].save(destination1)
    file2[0].save(destination2)
    detector = MTCNN()
    faces = [extract_face(image, detector) for image in [destination1,destination2]]
    match_str, similarity = get_similarity(faces)
    data_json = {
        'match_str' : match_str,
        'similarity_value' : similarity
    }
    return jsonify(data_json)


def extract_face(imagePath, detector, resize=(224,224)):
  image = cv2.imread(imagePath)
  faces = detector.detect_faces(image)
  x1,y1,width,height = faces[0]['box']
  x2,y2 = x1+width , y1+height

  face_boundary = image[y1:y2,x1:x2]

  face_image = cv2.resize(face_boundary,resize)

  return face_image



def get_embeddings(faces):
  face = np.asarray(faces,'float32')

  face = preprocess_input(face,version=2)

  model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

  return model.predict(face)


def get_similarity(faces):
  embeddings = get_embeddings(faces)
  score = cosine(embeddings[0], embeddings[1])
  print(embeddings[0])
  if score <= 0.5:
    return "Face Matched",score
  return "Face Not Matched", score



# driver function
if __name__ == '__main__':
  
    app.run(debug = True)