from flask import Flask, request, render_template, jsonify
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import CognitiveServicesCredentials, ApiKeyCredentials
import os

# Initialize Flask app
app = Flask(__name__)

# Azure credentials and endpoints
computer_vision_key = 'f2871c5b5a044d3cb3997e643b48b4de'
computer_vision_endpoint = 'https://classdemo-cylim.cognitiveservices.azure.com/'
custom_vision_key = '6cd85f8875484ea184b0f7c4f35f2f55'
custom_vision_endpoint = 'https://classdemocylim1-prediction.cognitiveservices.azure.com/'
project_id = '51e16077-19aa-4801-b978-01541c75a2fb'
published_name = 'Iteration1'

# Initialize the Computer Vision and Custom Vision clients
computervision_client = ComputerVisionClient(computer_vision_endpoint, CognitiveServicesCredentials(computer_vision_key))
custom_vision_credentials = ApiKeyCredentials(in_headers={"Prediction-key": custom_vision_key})
custom_vision_client = CustomVisionPredictionClient(custom_vision_endpoint, custom_vision_credentials)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['file']
    if image_file:
        # Read the image and analyze it using Computer Vision
        results = computervision_client.describe_image_in_stream(image_file)

        # Extracting description from the received results
        description = 'No description detected.'
        if results.captions:
            description = results.captions[0].text + ' with confidence ' + str(results.captions[0].confidence)

        return jsonify(description=description)
    return 'No file uploaded', 400

@app.route('/customvision', methods=['POST'])
def custom_vision():
    image_file = request.files['file']
    if image_file:
        # Process the image using Custom Vision
        results = custom_vision_client.classify_image(project_id, published_name, image_file.read())

        # Extracting prediction from the received results
        if results.predictions:
            # Get the prediction with the highest probability
            top_prediction = max(results.predictions, key=lambda p: p.probability)
            prediction_text = f"{top_prediction.tag_name} with probability {top_prediction.probability:.2f}"
            return jsonify(prediction=prediction_text)
        else:
            return jsonify(prediction="No prediction available")
    return 'No file uploaded', 400

if __name__ == '__main__':
    app.run(debug=True)

