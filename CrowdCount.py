from flask import Flask, request, send_file, redirect, url_for, render_template, jsonify, session
from io import BytesIO
import os
import cv2
import tensorflow
import base64
import shutil
import numpy as np
import time
from PIL import Image
from skimage.feature import peak_local_max
import keras
from keras import backend as K


@keras.saving.register_keras_serializable()
class CSRNet:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        kernel = (3, 3)
        init = keras.initializers.RandomNormal(stddev=0.01)
        model = keras.models.Sequential()

        # Frontend (VGG-16)
        model.add(keras.layers.Conv2D(64, kernel_size=kernel, activation='relu', padding='same', input_shape=(None, None, 3), kernel_initializer=init))
        model.add(keras.layers.Conv2D(64, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(keras.layers.MaxPooling2D(strides=2))

        model.add(keras.layers.Conv2D(128, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(128, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(keras.layers.MaxPooling2D(strides=2))

        model.add(keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(keras.layers.MaxPooling2D(strides=2))

        model.add(keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))

        # Backend (Dilated Convolutions)
        model.add(keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', dilation_rate=2, padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', dilation_rate=2, padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', dilation_rate=2, padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', dilation_rate=2, padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(128, kernel_size=kernel, activation='relu', dilation_rate=2, padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(64, kernel_size=kernel, activation='relu', dilation_rate=2, padding='same', kernel_initializer=init))
        model.add(keras.layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer=init))

        # Compile the model
        sgd = keras.optimizers.SGD(learning_rate=1e-7, decay=5e-4, momentum=0.95)
        model.compile(optimizer=sgd, loss=self.euclidean_distance_loss, metrics=['mse'])

        return model

    def euclidean_distance_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def predict(self, image):
        return self.model.predict(image)

    def get_config(self):
        return {"name": self.__class__.__name__}


crowdcount = CSRNet()
crowdcount.load_weights("model_A_weights.h5")


# Preprocess the input image
def preprocess_image(image_path):
    print('Geneal preprocessing imge started')
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, Image.Image):
        image= image_path
    image = np.array(image)
    image = image / 255.0
    image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
    image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
    image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print('genral preprocess image is concluded')
    return image

# Detect peaks in the density map
def detect_peaks(density_map, distance, threshold):
    print('detection of peaks in frame stated')
    peaks = peak_local_max(density_map, min_distance=distance, threshold_rel=threshold)
    print('detection of peaks in frame is concluded')
    return peaks, len(peaks)

# Overlay numbers on the image
def overlay_numbers(image, peaks, original_shape, density_map_shape):
    # Calculate the scaling factors
    print('overlaying the frame with number has stated')
    scale_x = original_shape[1] / density_map_shape[1]
    scale_y = original_shape[0] / density_map_shape[0]

    for i, peak in enumerate(peaks):
        x, y = peak
        # Scale the coordinates
        x = int(x * scale_y)
        y = int(y * scale_x)
        # Add text with a black outline for better readability
        cv2.putText(image, str(i+1), (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline
        cv2.putText(image, str(i+1), (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Red text
    print('overlaying the frame with number is complete')
    return image

# Process an image
def process_image(model, image_path, distance, threshold):
    preprocessed_image = preprocess_image(image_path)
    density_map = model.predict(preprocessed_image).squeeze()
    peaks, num_peaks = detect_peaks(density_map, distance, threshold)
    original_image = cv2.imread(image_path)
    numbered_image = overlay_numbers(original_image, peaks, original_image.shape, density_map.shape)
    return numbered_image, num_peaks


# Process a video
def process_video(model, video_path, distance, threshold):
    print('processing the picture has started')
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    peaks_track = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print('frame captured')
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        preprocessed_image = preprocess_image(image)
        print('proceeding to predict with the model')
        density_map = model.predict(preprocessed_image).squeeze()
        print('predicting with the model is completed ')
        peaks, num_peaks = detect_peaks(density_map, distance, threshold)
        original_image = frame
        numbered_frame = overlay_numbers(original_image, peaks, original_image.shape, density_map.shape)
        print('frame outcome is ready')
        processed_frames.append(numbered_frame)
        peaks_track.append(num_peaks)
        
    cap.release()
    height, width, layers = processed_frames[0].shape
    video_path = 'static/processed_video.mp4'
    print('saving the output video to the upload directory')
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in processed_frames:
        video.write(frame)

    print('saving the output video to the upload directory')   
    video.release()
    return video_path, np.max(peaks_track)


decode={'low':1, 'medium':2, 'high':3, 'white':0.019, 'black':0.004}
encode={1:'Low', 2:'Medium', 3:'High', 0.019:'White', 0.004:'Black'}


app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'

@app.route('/') 
def note(): 
    return render_template('note.html')

@app.route('/cc_home', methods=['GET']) 
def cc_home(): 
    return render_template('cc_home.html')

# Main function to determine input type and process accordingly
@app.route('/process_picture', methods=['POST']) 
def process_picture(model=crowdcount):
    distance = decode[request.form.get('densityLevel')]
    threshold = decode[request.form.get('ethnicity')]
    input_type = request.form.get('inputType')
    
    if input_type == 'live':
        session['distance'] = distance
        session['threshold'] = threshold
        return render_template('scan1.html')
    
    file = request.files['file']
    folder = 'static'
    input_path =os.path.join(folder, file.filename)
    print('process has started')
    
    if input_path.endswith(('.jpg', '.jpeg', '.png')):
        # Process the image
        numbered_image , num_peaks= process_image(model, input_path, distance, threshold)
        final_image = cv2.cvtColor(numbered_image, cv2.COLOR_BGR2RGB)
        # Define the output path 
        output_path = 'static/processed_image.jpg'

        # Save the processed image 
        pil_image = Image.fromarray(final_image)
        pil_image.save(output_path)
        
        time.sleep(1)
        

        return render_template('result3.html', resultUrl=output_path, resultType='image', num_peaks=num_peaks, distance=encode[distance], threshold=encode[threshold])
    
    elif input_path.endswith(('.mp4', '.avi', '.mov')):
        # Process the video
        print('path has branched out')
        video_path, max_peak = process_video(model, input_path, distance, threshold)
        time.sleep(1)
        print('all done')
        return render_template('result2.html', resultUrl=video_path, resultType='video', max_peak=max_peak, distance=encode[distance], threshold=encode[threshold])
    

@app.route('/live_stream', methods=['GET'])
def live_stream(model=crowdcount):
    input_path="static/captured-image.png"
    distance = session.get('distance', [])
    threshold = session.get('threshold', [])

    numbered_image , num_peaks= process_image(model, input_path, distance, threshold)
    final_image = cv2.cvtColor(numbered_image, cv2.COLOR_BGR2RGB)
    # Define the output path 
    output_path = 'static/processed_image.jpg'

    # Save the processed image 
    pil_image = Image.fromarray(final_image)
    pil_image.save(output_path)
        
    time.sleep(1)
        

    return render_template('result3.html', resultUrl=output_path, resultType='image', num_peaks=num_peaks, distance=encode[distance], threshold=encode[threshold])
    




@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    folder = 'static'
    if os.path.exists(folder):
        shutil.rmtree(folder)  # Remove the existing folder and its contents
    os.makedirs(folder)  # Create a new, empty folder
    if file:
        # Save the uploaded file to a temporary location
        file_path = os.path.join(folder, file.filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path})
    else: 
        return jsonify({'error': 'No file uploaded'}), 400
    


@app.route('/upload_snap', methods=['POST'])
def upload_snap():
    data_url = request.data.decode('utf-8')
    # Decode base64 image
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    print('Save the image')
    folder = 'static'
    if os.path.exists(folder):
        shutil.rmtree(folder)
    
    os.makedirs(folder)  # Create a new, empty folder
    print('creatin static')
    filepath = os.path.join(folder, 'captured-image.png')
    with open(filepath, 'wb') as f:
        f.write(data)
    
    return 'Image uploaded successfully', 200

        

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)

