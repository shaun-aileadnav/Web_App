import os
import wandb
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms

from main import MyNeuralNet

CLASSES = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

# Initialize Flask app
app = Flask(__name__)

# Set up image transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to the input size of your model (28x28)
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0),  # Normalize by dividing by 255
])

# Load the model from W&B Model Registry
def load_model():
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    run = wandb.init(project="your_project_name", job_type="load_model")
    model_artifact = run.use_artifact('ai-leadnav-org/wandb-registry-model/<Shauns_models>:latest', type='model')
    model_dir = model_artifact.download()
    
    model = MyNeuralNet()  # Replace with your model class
    model.load_state_dict(torch.load(os.path.join(model_dir, "my_model.pth")))
    return model

# Load the model
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template for the home page

@app.route('/predict/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400  # Return a 400 error if no file is provided
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400  # Return a 400 error if no file is selected
    
    img = Image.open(file.stream).convert("L")  # Convert image to grayscale
    img = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        model.eval()
        output = model(img)
        prediction = output.argmax().item()  # Get the predicted class index

    return jsonify({"predicted_class": CLASSES[prediction]})  # Return the prediction as a JSON response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
