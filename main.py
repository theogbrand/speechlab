import os
from time import time
import asyncio

import requests
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import pygame
from pygame import mixer
from tts import elevenlabs_tts, get_audio_response

from record import speech_to_text
import json
from datetime import datetime
from local_file_stt import transcribe_audio_file
from local_whisper_stt import local_whisper_transcribe

# Load API keys
load_dotenv()

# mixer is a pygame module for playing audio
mixer.init()

# Change the context if you want to change Jarvis' personality
system_prompt = "You are Jarvis, Brandon's human assistant. You are witty and full of personality. Your answers should usually be crisp in 1-2 short sentences, unless a discussion is rightfully necessary."
LOCAL_RECORDING_PATH = "audio/recording.wav"
SAVE_CONVERSATION_DIR = "entries"

def ask_ai(messages: list[dict]) -> str:
    """
    Send a prompt to the GPT-3 API and return the response.

    Args:
        - state: The current state of the app.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "system", "content": system_prompt}] + messages,
    )
    return response.choices[0].message.content


def ask_sealion(messages: List[Dict[str, str]]) -> str:
    """
    Send a prompt to the SEA-LION API and return the response.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content'.

    Returns:
        The complete response as a string.

    Raises:
        requests.RequestException: If there's an error with the API request.
        ValueError: If the API response is not in the expected format.
    """
    url = "https://api.sea-lion.ai/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['AISG_INFERENCE_API_KEY']}"
    }
    data = {
        "messages": messages,
        "model": "aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct",
        "stream": False,
        "max_tokens": 128
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if "choices" not in result or not result["choices"]:
            raise ValueError("Invalid response format from API")
            
        return result["choices"][0]["message"]["content"]

    except requests.RequestException as e:
        raise requests.RequestException(f"Error making request to SEA-LION API: {e}")


def log(log: str):
    """
    Print and write to status.txt
    """
    print(log)
    with open("status.txt", "w") as f:
        f.write(log)


def to_epoch(dt):
    return int(dt.timestamp())


def add_conversation_data(conversations_arr):
    if not os.path.exists(SAVE_CONVERSATION_DIR):
        os.makedirs(SAVE_CONVERSATION_DIR)

    now = datetime.now()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    current_day_epoch = to_epoch(start_of_day)

    matching_files = []
    for filename in os.listdir(SAVE_CONVERSATION_DIR):
        if filename.endswith(".json") and str(current_day_epoch) in filename:
            matching_files.append(filename)

    if len(matching_files) == 0:
        new_filename = f"{current_day_epoch}.json"
        file_path = os.path.join(SAVE_CONVERSATION_DIR, new_filename)
        with open(file_path, "w") as f:
            json.dump(
                {"date": current_day_epoch, "conversations": conversations_arr}, f
            )
    
    for file in matching_files:
        with open(os.path.join(SAVE_CONVERSATION_DIR, file), "r+") as f:
            data = json.load(f)
            data["conversations"].extend(conversations_arr)
            f.seek(0)
            f.truncate()
            json.dump(data, f)


if __name__ == "__main__":
    while True:
        # Record audio
        log("Listening...")
        speech_to_text()
        log("Done listening")
        llm_conversation = []
        conversation = []
        # Transcribe audio
        current_time = time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        human_reply = transcribe_audio_file(LOCAL_RECORDING_PATH, "54.255.127.241")
        # human_reply = local_whisper_transcribe(LOCAL_RECORDING_PATH)["text"]
        transcription_time = time() - current_time
        log(f"Finished transcribing in {transcription_time:.2f} seconds.")

        current_time = time()
        conversation.append({"role": "user", "content": human_reply})
        llm_conversation.append(
            {
                "timestamp": to_epoch(datetime.now()),
                "role": "user",
                "content": human_reply,
            }
        )
        ai_response = ask_sealion(messages=conversation)
        conversation.append({"role": "assistant", "content": ai_response})
        llm_conversation.append(
            {
                "timestamp": to_epoch(datetime.now()),
                "role": "assistant",
                "content": ai_response,
            }
        )
        gpt_time = time() - current_time
        log(f"Finished generating response in {gpt_time:.2f} seconds.")

        current_time = time()
        # get_audio_response(ai_response)
        elevenlabs_tts(ai_response, "audio/response.mp3")
        audio_time = time() - current_time
        log(f"Finished generating audio in {audio_time:.2f} seconds.")

        log("Speaking...")
        sound = mixer.Sound("audio/response.mp3")
        add_conversation_data(llm_conversation)
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        print(f"\n --- USER: {human_reply}\n --- JARVIS: {ai_response}\n")
