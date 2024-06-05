import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from services.client import Image2ImageClient

app = Flask(__name__)

UPLOAD_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/stable_diffusion/generate', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    if 'prompt' not in request.json:
        return jsonify({'status': 'error', 'message': 'Prompt not provided'})

    file = request.files['file']

    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # where the magic happens
        client = Image2ImageClient()
        client.load_image(file_path)
        # TODO: remove file from temp folder

        prompt = request.json['prompt']
        images = client.generate(prompt)

        for image in images:
            image.save(f"output/{filename}.jpeg")

        return jsonify({'status': 'success', 'message': 'Images generated' })

    return jsonify({'status': 'error', 'message': 'Invalid file extension'})


if __name__ == '__main__':
    app.run(debug=True)