import os
import requests
import wave
import numpy as np
import os
import uuid

from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)


def get_audio_response(text):
    url = "https://api.cartesia.ai/tts/bytes"
    payload = {
        # "model_id": "sonic-english",
        "model_id": "sonic-multilingual",
        "transcript": text,
        "duration": 123,
        "voice": {
            "mode": "id",
            # Jarvis model id
            # "id": "1d92e61c-e8a2-4544-9d17-b6dfb38e212a"
            "id": "0c687b45-575e-455c-ba57-401a484cd7f7" # Sherry
        },
        "output_format": {
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        }
    }
    headers = {
        "X-API-Key": os.environ["CARTESIA_API_KEY"],
        "Cartesia-Version": "2024-06-10",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the raw bytes to numpy array of float32
        audio_data = np.frombuffer(response.content, dtype=np.float32)
        
        # Normalize the float32 data to int16 range
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        
        # Open a new .wav file in write mode
        with wave.open('audio/response.wav', 'wb') as wav_file:
            # Set the parameters for the .wav file
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 2 bytes per sample for 16-bit audio
            wav_file.setframerate(payload['output_format']['sample_rate'])
            
            # Write the audio data
            wav_file.writeframes(audio_data_int16.tobytes())

        print("Audio saved as response.wav")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    return


# get_audio_response("Hello, brandon.") 
def elevenlabs_tts(text: str, save_path: str, language: str) -> str:
    """
    Converts text to speech and saves the output as an MP3 file.

    This function uses a specific client for text-to-speech conversion. It configures
    various parameters for the voice output and saves the resulting audio stream to an
    MP3 file with a unique name.

    Args:
        text (str): The text content to convert to speech.

    Returns:
        str: The file path where the audio file has been saved.
    """
    language_to_voice_id = {
        "ENGLISH": "mbL34QDB5FptPamlgvX5",  # Jay - Asian, Singapore
        # "INDONESIAN": "1k39YpzqXZn52BgyLyGO",  # Bee Ard - Indonesian
        "INDONESIAN": "v70fYBHUOrHA3AKIBjPq",  # Mahaputra - Indonesian
    }
    # Calling the text_to_speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id=language_to_voice_id[language],  # Jay - Asian, Singapore
        optimize_streaming_latency="0",
        output_format="mp3_44100_128",
        text=text,
        # model_id="eleven_multilingual_v2",
        model_id="eleven_turbo_v2",
        voice_settings=VoiceSettings(
            stability=0.7,
            similarity_boost=0.6,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Generating a unique file name for the output MP3 file
    save_file_path = f"{save_path}"
    # Writing the audio stream to the file

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"A new audio file was saved successfully at {save_file_path}")

    # Return the path of the saved audio file
    return save_file_path


if __name__ == "__main__":
    elevenlabs_tts("你可以 帮我 Search Google SEA-LION in Singapore Ape ini?", "audio/test_elevenlabs.mp3")