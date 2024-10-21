"""Main file for the Jarvis project"""

import os
from os import PathLike
from time import time
import asyncio
from typing import Union

from dotenv import load_dotenv
from openai import OpenAI
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import pygame
from pygame import mixer
from tts import get_audio_response

from record import speech_to_text

# from groq_stt import groq_transcribe
from llm import LLM
import json
from datetime import datetime
from rag_tools import reflect_tool
import stat


# Load API keys
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Initialize APIs
# deepgram = Deepgram(DEEPGRAM_API_KEY)
deepgram = DeepgramClient(DEEPGRAM_API_KEY)
# mixer is a pygame module for playing audio
mixer.init()

# Change the context if you want to change Jarvis' personality
system_prompt = "You are Jarvis, Brandon's human assistant. You are witty and full of personality. Your answers should usually be crisp in 1-2 short sentences, unless a discussion is rightfully necessary."
# conversation = {"Conversation": []}
conversation = []
RECORDING_PATH = "audio/recording.wav"


def request_gpt(prompt: str) -> str:
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
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def deepgram_transcribe(
    file_name: Union[Union[str, bytes, PathLike[str], PathLike[bytes]], int]
):
    """
    Transcribe audio using Deepgram API.

    Args:
        - file_name: The name of the file to transcribe.

    Returns:
        The response from the API.
    """
    # with open(file_name, "rb") as audio:
    #     source = {"buffer": audio, "mimetype": "audio/wav"}
    #     response = await deepgram.transcription.prerecorded(source)
    #     return response["results"]["channels"][0]["alternatives"][0]["words"]
    with open(file_name, "rb") as audio:
        try:
            # STEP 1 Create a Deepgram client using the DEEPGRAM_API_KEY from environment variables
            buffer_data = audio.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }

            # STEP 2 Call the transcribe_url method on the prerecorded class
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                summarize="v2",
                # detect_language=True,
            )
            file_response = deepgram.listen.prerecorded.v("1").transcribe_file(
                payload, options
            )

            json_data = file_response.to_json()
            data = json.loads(json_data)

            # return data["results"]["summary"]["short"]
            return data["results"]["channels"][0]["alternatives"][0]["transcript"]

        except Exception as e:
            print(f"Exception: {e}")


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

    # Get the current date at the start of the day (midnight)
    now = datetime.now()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    current_day_epoch = to_epoch(start_of_day)

    matching_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json") and str(current_day_epoch) in filename:
            matching_files.append(filename)

    # for file in matching_files:
    #     print("matched file: ", file)
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
        # Transcribe audio
        current_time = time()
        # human_reply = groq_transcribe(RECORDING_PATH)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        human_reply = loop.run_until_complete(deepgram_transcribe(RECORDING_PATH))
        # human_reply = " ".join(
        #     word_dict.get("word") for word_dict in words if "word" in word_dict
        # )
        # conversation_data = {"human_reply": human_reply}
        # with open("conv.json", "a") as f:
        #     json.dump(conversation_data, f)
        #     f.write("\n")
        # add_conversation_data("user", human_reply)
        transcription_time = time() - current_time
        log(f"Finished transcribing in {transcription_time:.2f} seconds.")

        # Get response from AI
        current_time = time()
        conversation.append({"role": "user", "content": human_reply})
        llm_conversation.append(
            {
                "timestamp": to_epoch(datetime.now()),
                "role": "user",
                "content": human_reply,
            }
        )
        ai_response = LLM(system_message=system_prompt).generate_response(
            messages=conversation
        )
        # ai_response = reflect_tool(human_reply, conversation)
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

        # Convert response to audio
        current_time = time()
        get_audio_response(ai_response)
        # audio = elevenlabs.generate(
        #     text=response, voice="Adam", model="eleven_monolingual_v1"
        # )
        # elevenlabs.save(audio, "audio/response.wav")
        audio_time = time() - current_time
        log(f"Finished generating audio in {audio_time:.2f} seconds.")

        # Play response
        log("Speaking...")
        sound = mixer.Sound("audio/response.wav")
        # Add response as a new line to conv.txt
        # with open("conv.txt", "a") as f:
        #     f.write(f"{ai_response}\n")
        # conversation_data = {"ai_response": ai_response}
        # with open("conv.json", "a") as f:
        #     json.dump(conversation_data, f)
        #     f.write("\n")
        # add_conversation_data("assistant", ai_response)
        add_conversation_data(llm_conversation)
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        print(f"\n --- USER: {human_reply}\n --- JARVIS: {ai_response}\n")
