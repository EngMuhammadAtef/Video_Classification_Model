# import libraries
from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from classification_model.kinetics_model import load_video, predict
from classification_model.OCR_model import image2text, video2text
from classification_model.image_object_detect import extract_objectNames
from speech_model.speech2text import transcribe_video
from googletrans import Translator

# Initial Flask object
app = Flask(__name__)

# Authentication setup
auth = HTTPBasicAuth()

# Initial Translator model
translator = Translator()

VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'WebM'}
IMAGE_ALLOWED_EXTENSIONS = {'jpg', 'jpeg','png', 'gif', 'tiff', 'WebM'}

# In-memory user storage
users = {
    "user1": generate_password_hash("msso3ks24as7fh48fasdakms"),
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

@app.route('/classify_media', methods=['POST'])
@auth.login_required
def classify_media():
    # Validate request for media file using Flask-Requests
    if 'media' not in request.files:
        return jsonify({'error': 'Missing List of media file'}), 400
    
    media_files = request.files.getlist('media')
    results = []

    for media in media_files:
        # get the path of the media
        media_path = media.filename
        if not media_path:
            results.append({'error': 'Empty media file'})
            continue
        
        extension = media_path.rsplit('.', 1)[1].lower()
        
        # if the media is a video
        if extension in VIDEO_ALLOWED_EXTENSIONS:
            language = request.args.get('language', 'English')
            # convert the video to the text
            text = transcribe_video(media_path, language)
            # Process the video file, convert to frames or batches
            frames = load_video(media_path)
            result = predict(frames)
            result['text_on_frames'] = video2text(frames)
            result['Subtitle'] = str(translator.translate(text).text)
            # Return the classification result
            results.append(result)
        
        # if the media is a image
        elif extension in IMAGE_ALLOWED_EXTENSIONS:
            # Classify The Image by extracting object names
            object_names = extract_objectNames(media_path)
            
            # Make predictions
            text = image2text(media_path)
            
            # Return the classification result
            results.append({'text_on_image': str(text), 'objectInImage': dict(object_names)})
        else:
            results.append({'error': 'Unsupported media format'})

    return jsonify(results)


if __name__ == '__main__':
    app.run()
