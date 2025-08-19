# from flask import Flask, render_template, request, send_from_directory
# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import load_model
# import os
# import numpy as np

# app = Flask(__name__)
# # Load the pre-trained model
# model = load_model('models/model.h5')

# class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# up_folder = './uploads'
# if not os.path.exists(up_folder):
#     os.makedirs(up_folder)

# app.config['up_folder'] = up_folder

# def predicter(img_path):
#     image_size = 128
#     img=load_img(img_path,target_size=(image_size,image_size))
#     img_array=img_to_array(img)/255.0
#     img_array=np.expand_dims(img_array,axis=0)

#     pred=model.predict(img_array)
#     predicted_class_index = np.argmax(pred, axis=1)[0]
#     confidence_score = np.max(pred, axis=1)[0]

#     if class_labels[predicted_class_index] == 'notumor':
#         return "No Tumor", confidence_score
#     else:
#         return f"Tumor: {class_labels[predicted_class_index]}", confidence_score


# @app.route('/',methods=['GET', 'POST'])

# def index():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filepath = os.path.join(app.config['up_folder'], file.filename)
#             file.save(filepath)
#             result, confidence = predicter(filepath)

#             return render_template('index.html',result=result, confidence=f'{confidence*100:.2f}%', file_path=f'/uploads/{file.filename}')
#     return render_template('index.html',result=None)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['up_folder'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('models/model.h5')

# Class labels
class_labels = ['notumor', 'meningioma', 'glioma', 'pituitary']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)
            print(result)
            print(confidence)
            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)