import os
from flask.helpers import flash
from tensorflow.keras.models import model_from_json
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import PIL
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])




# opening and store file in a variable
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model

loaded_model.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

images = {
    'Daisy' : 'https://images.unsplash.com/photo-1559406994-913e66cdc5ad?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80',
    'Dandelion': 'https://images.unsplash.com/photo-1589912593528-1106d19aa17e?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=379&q=80',
    'Roses': 'https://images.unsplash.com/photo-1560939674-b87318b31767?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80',
    'Sunflowers' : 'https://images.unsplash.com/photo-1597848212624-a19eb35e2651?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=435&q=80',
    'Tulips': 'https://images.unsplash.com/photo-1614791199038-6869a104fe5f?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80',
}

def allowed_file(filename):    
    return ('.' in filename) and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_image(image_path):
    
    im = Image.open(image_path).convert('RGB')
    im = im.resize((180,180))
    image_np = np.array(im)
    print(image_np)
    # print(image_np)
    print(image_np.shape)
    img_4d=image_np.reshape(-1,180,180,3)
    prediction=loaded_model.predict(img_4d)[0]
    
    
    dict1 =  {class_names[i]: float(prediction[i]) for i in range(5)}
    return dict1


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"FYIOz\n\xec]/'



# !Routing #

@app.route('/')
def home():
    return render_template('index.html', count=0)


@app.route('/predict', methods=['POST'])
def predict():

    print("Sunil is a great developer")


    # Form handling part
    if 'file' not in request.files:
        flash('No file part')
        return 'No file uploaded' 
        # redirect(request.url)
    
    file = request.files['file']
        # if user does not select file or submit a empty part without filename

    if file.filename == '':
            flash('No selected file')
            return 'Input cannnot be empty!!Please upload some image' 
            # redirect(request.url)
    if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dict1 = convert_image(UPLOAD_FOLDER+"/"+filename)
            print(dict1)

    print('*'*20)
   


    return render_template('index.html',count = len(list(images.keys())),flowers = list(images.keys()), links = list(images.values()), results = [np.round(x*100) for x in list(dict1.values())])


if __name__ == "__main__":
    app.run(debug=True)
