import speech_recognition as sr
from moviepy.editor import VideoFileClip

# Dictionary of supported languages and their codes
lang_codes = {
    'English': 'en-US',
    'Spanish': 'es-ES',
    'French': 'fr-FR',
    'German': 'de-DE',
    'Italian': 'it-IT',
    'Japanese': 'ja-JP',
    'Korean': 'ko-KR',
    'Chinese': 'zh-CN',
    'Portuguese': 'pt-BR',
    'Russian': 'ru-RU',
    'Hindi': 'hi-IN',
    'Arabic': 'ar-EG',
    'Dutch': 'nl-NL',
    'Swedish': 'sv-SE'
}

def transcribe_audio(file_path, lang='English'):
    recognizer = sr.Recognizer()
    text = ''
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language=lang_codes[lang])
    except sr.UnknownValueError:
        print("LanguageError: The model could not understand the audio")
    return text

def transcribe_video(file_path, lang='English'):
    # Load the video file
    video = VideoFileClip(file_path)
    # Extract audio from the video
    audio = video.audio

    # Save audio to a temporary file
    audio_file_path = "temp_audio.wav"
    audio.write_audiofile(audio_file_path)

    # recognizer
    text = transcribe_audio(audio_file_path, lang)
    
    import os
    os.remove(audio_file_path)
    return text

# Example usage
# transcribe_audio("1.wav", 'English')
# transcribe_audio("2.wav", 'Arabic')