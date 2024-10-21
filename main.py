import os
from time import time
import asyncio

from dotenv import load_dotenv
from openai import OpenAI
import pygame
from pygame import mixer
from tts import get_audio_response

from record import speech_to_text
import json
from datetime import datetime
from local_file_stt import transcribe_audio_file

# Load API keys
load_dotenv()

# mixer is a pygame module for playing audio
mixer.init()

# Change the context if you want to change Jarvis' personality
system_prompt = "You are Jarvis, Brandon's human assistant. You are witty and full of personality. Your answers should usually be crisp in 1-2 short sentences, unless a discussion is rightfully necessary."
RECORDING_PATH = "audio/recording.wav"


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
    directory = "entries"

    now = datetime.now()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    current_day_epoch = to_epoch(start_of_day)

    matching_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json") and str(current_day_epoch) in filename:
            matching_files.append(filename)

    if len(matching_files) == 0:
        new_filename = f"{current_day_epoch}.json"
        file_path = os.path.join(directory, new_filename)
        with open(file_path, "w") as f:
            json.dump(
                {"date": current_day_epoch, "conversations": conversations_arr}, f
            )
    
    for file in matching_files:
        with open(os.path.join(directory, file), "r+") as f:
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
        human_reply = transcribe_audio_file(RECORDING_PATH, "54.255.127.241")
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
        ai_response = ask_ai(messages=conversation)
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
        get_audio_response(ai_response)
        audio_time = time() - current_time
        log(f"Finished generating audio in {audio_time:.2f} seconds.")

        log("Speaking...")
        sound = mixer.Sound("audio/response.wav")
        add_conversation_data(llm_conversation)
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        print(f"\n --- USER: {human_reply}\n --- JARVIS: {ai_response}\n")
