import os
import re
from time import time
import asyncio

import requests
from typing import List, Dict, Literal
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
import instructor
from litellm import completion
from pydantic import BaseModel
from queue import Queue
import threading
from typing import Dict, Tuple
import google.generativeai as genai


# Load API keys
load_dotenv()

# mixer is a pygame module for playing audio
mixer.init()

# Change the context if you want to change Jarvis' personality
system_prompt = ""
LOCAL_RECORDING_PATH = "audio/recording.wav"
SAVE_CONVERSATION_DIR = "entries"


def ask_claude(user_input: str) -> str:
    claude_user_prompt = """
        You are an AI assistant with expert knowledge about Singapore and fluency in Singaporean English (Singlish), Chinese, and Malay. Your task is to communicate with Singaporean users who often use code-mixed sentences containing a blend of these languages.

        Here is the user's input:
        <user_input>
        {{USER_INPUT}}
        </user_input>

        Analyze the input carefully, paying attention to any language mixing or code-switching between Singaporean English, Chinese, and Malay. Identify the user's intent and the context of their message.

        Respond to the user appropriately based on their intent and the languages used in their input. Your response should:
        1. Match the level of formality and tone of the user's input
        2. Incorporate similar language mixing if present in the original message
        3. Use Singaporean colloquialisms or expressions where appropriate
        4. Provide relevant information or answers related to Singapore if the user is asking a question

        Maintain a Singaporean context throughout your response. This includes references to local culture, customs, places, or current events when relevant to the conversation.

        If the user's input is entirely in one language, respond in that same language unless the context requires otherwise.

        Think step-by-step in <thinking> tags and then provide your response inside <response> tags. Ensure your reply sounds natural and authentic to a Singaporean conversation. Reply only in one language.
    """
    messages = [
        {
            "role": "user",
            "content": claude_user_prompt.replace("{{USER_INPUT}}", user_input),
        }
    ]
    response = completion(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
    )
    # Extract response content between <response> tags
    response_text = response.choices[0].message.content

    # Use regex to find content between <response> tags
    match = re.search(r"<response>(.*?)</response>", response_text, re.DOTALL)
    if not match:
        # If no tags found, return the full response
        return response_text.strip()

    # Return just the content between tags
    return match.group(1).strip()


def ask_openai(messages: list[dict]) -> str:
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


def rewrite_transcription(
    speechlab_transcription: str, whisper_transcription: str, language: str
) -> str:
    prompt = """
        You are tasked with rewriting a user's query that has been transcribed from audio using two different ASR (Automatic Speech Recognition) systems. Your goal is to produce a single, coherent query in one language to avoid confusing downstream models.

        Here are the two transcriptions of the same audio input:

        <transcription1>
        {{TRANSCRIPTION1}}
        </transcription1>

        <transcription2>
        {{TRANSCRIPTION2}}
        </transcription2>

        The detected language for this audio input is:
        <detected_language>
        {{DETECTED_LANGUAGE}}
        </detected_language>

        Your task is to rewrite the query in the detected language. Follow these guidelines:

        1. If both transcriptions are in the same language and match the detected language, choose the more coherent or complete version.

        2. If the transcriptions contain a mix of languages:
        a. Prioritize the detected language.
        b. Maintain the original meaning and intent of the query as much as possible.

        3. If the transcriptions differ significantly:
        a. Try to combine the most coherent parts from both.

        4. Correct any obvious transcription errors or filler words (um, uh, etc.) that don't add meaning to the query.

        5. Ensure the final query is grammatically correct in the detected language and makes sense in the context of a user input.

        Provide your rewritten query inside <rewritten_query> tags. After the rewritten query, briefly explain your reasoning for the changes made inside <explanation> tags.

        Here are some examples of how to handle different scenarios:

        Example 1:
        <transcription1>Hello, como estás? I need ayuda with mi computer.</transcription1>
        <transcription2>Hello, cómo estás? I need help with my computer.</transcription2>
        <detected_language>English</detected_language>

        <rewritten_query>Hello, how are you? I need help with my computer.</rewritten_query>
        <explanation>The detected language is English, so I translated the Spanish phrases and corrected "ayuda" to "help". I chose "how are you" from the second transcription as it's the correct spelling.</explanation>

        Example 2:
        <transcription1>Je voudrais un café, s'il vous plaît.</transcription1>
        <transcription2>Je voudrais un cafe si vous plait.</transcription2>
        <detected_language>French</detected_language>

        <rewritten_query>Je voudrais un café, s'il vous plaît.</rewritten_query>
        <explanation>Both transcriptions are in French, matching the detected language. I chose the first transcription as it has the correct accents and punctuation.</explanation>

        Now, please proceed with rewriting the given query based on the provided transcriptions and detected language.
    """

    resp = completion(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt.replace("{{TRANSCRIPTION1}}", speechlab_transcription)
                .replace("{{TRANSCRIPTION2}}", whisper_transcription)
                .replace("{{DETECTED_LANGUAGE}}", language),
            }
        ],
    )

    try:
        response_text = resp.choices[0].message.content

        # Find content between <rewritten_query> tags
        rewritten_query_match = re.search(
            r"<rewritten_query>(.*?)</rewritten_query>", response_text, re.DOTALL
        )

        # Find content between <explanation> tags
        # explanation_match = re.search(
        #     r"<explanation>(.*?)</explanation>", response_text, re.DOTALL
        # )
        # print("explanation: ", explanation_match.group(1).strip())
    except Exception as e:
        print("Error in rewrite_transcription", e)
        # return whisper_transcription
    # Extract the rewritten query and explanation from the response using regex

    # Use the rewritten query if found, otherwise return original whisper transcription
    if rewritten_query_match:
        print("rewritten_query", rewritten_query_match.group(1).strip())
        return rewritten_query_match.group(1).strip()
    else:
        print(
            "Could not find rewritten query in response, using original transcription",
        )
        return whisper_transcription


def check_language(speechlab_transcription: str, whisper_transcription: str) -> str:
    prompt = """
        You are an AI language analyst tasked with determining the language of an audio clip based on two ASR (Automatic Speech Recognition) transcriptions. Your goal is to identify whether the transcriptions are in English only, Indonesian only, or if the language cannot be determined. This information will be used to select the appropriate TTS (Text-to-Speech) model for downstream processing.

        You will be given two transcriptions of the same audio clip:

        Transcription 1:
        <transcription1>
        {{TRANSCRIPTION1}}
        </transcription1>

        Transcription 2:
        <transcription2>
        {{TRANSCRIPTION2}}
        </transcription2>

        Analyze both transcriptions carefully, looking for clear indicators of either English or Indonesian language. Consider the following:

        1. Vocabulary: Are the words distinctly English or Indonesian?
        2. Sentence structure: Does the word order follow English or Indonesian grammar rules?
        3. Common phrases or expressions: Are there any idioms or expressions specific to either language?
        4. Names or places: Are there any names or places mentioned that are typically associated with English-speaking or Indonesian-speaking regions?

        If both transcriptions consistently indicate the same language (either English or Indonesian), determine that as the language. If there is a mix of languages or if the transcriptions are too short or unclear to make a definitive determination, pick the most likely language.
    """

    # check transcription language if English only, Indonesian only (code-mixed later)
    class TranscriptionLanguage(BaseModel):
        language: Literal["ENGLISH", "INDONESIAN"]
        reasoning: str

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-latest",
        ),
        mode=instructor.Mode.GEMINI_JSON,
    )

    resp = client.messages.create(
        messages=[
            {
                "role": "user",
                "content": prompt.replace(
                    "{{TRANSCRIPTION1}}", speechlab_transcription
                ).replace("{{TRANSCRIPTION2}}", whisper_transcription),
            }
        ],
        response_model=TranscriptionLanguage,
    )

    # client = instructor.from_litellm(completion)

    # resp = client.chat.completions.create(
    #     model="claude-3-5-sonnet-20241022",
    #     max_tokens=1024,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt.replace(
    #                 "{{TRANSCRIPTION1}}", speechlab_transcription
    #             ).replace("{{TRANSCRIPTION2}}", whisper_transcription),
    #         }
    #     ],
    #     response_model=TranscriptionLanguage,
    # )

    try:
        assert isinstance(resp, TranscriptionLanguage)
        print("language detected: ", resp.language)
        return resp.language
    except Exception as e:
        print("Error in check_language", e)
        return "UNDETERMINED"


def run_transcriptions() -> Tuple[str, str]:
    """
    Run parallel transcriptions using threading and a thread-safe queue
    Returns tuple of (speechlab_transcription, whisper_transcription)
    """
    result_queue: Queue = Queue()
    
    def speechlab_worker() -> None:
        """Worker function for SpeechLab transcription"""
        try:
            result = transcribe_audio_file(LOCAL_RECORDING_PATH, "54.255.127.241")
            result_queue.put(("speechlab", result))
        except Exception as e:
            result_queue.put(("speechlab", f"Error: {str(e)}"))
    
    def whisper_worker() -> None:
        """Worker function for Whisper transcription"""
        try:
            result = local_whisper_transcribe(LOCAL_RECORDING_PATH)
            result_queue.put(("whisper", result["text"]))
        except Exception as e:
            result_queue.put(("whisper", f"Error: {str(e)}"))

    # Create and start threads
    threads = [
        threading.Thread(target=speechlab_worker),
        threading.Thread(target=whisper_worker)
    ]
    
    for thread in threads:
        thread.start()
    
    # Wait for both threads to complete
    results: Dict[str, str] = {}
    for _ in threads:
        source, result = result_queue.get()
        print(f"Done transcribing with {source}", result)
        results[source] = result
    
    for thread in threads:
        thread.join()
        
    return results["speechlab"], results["whisper"]


if __name__ == "__main__":
    llm_conversation = []
    conversation = []
    user_prompt = """
        You are an AI assistant with expert knowledge about Singapore and fluency in Singaporean English (Singlish), Chinese, and Malay. Your task is to communicate with Singaporean users who often use code-mixed sentences containing a blend of these languages.

        Here is the user's input:
        <user_input>
        {{USER_INPUT}}
        </user_input>

        Analyze the input carefully, paying attention to any language mixing or code-switching between Singaporean English, Chinese, and Malay. Identify the user's intent and the context of their message.

        Respond to the user appropriately based on their intent and the languages used in their input. Your response should:
        1. Match the level of formality and tone of the user's input
        2. Incorporate similar language mixing if present in the original message
        3. Use Singaporean colloquialisms or expressions where appropriate
        4. Provide relevant information or answers related to Singapore if the user is asking a question

        Maintain a Singaporean context throughout your response. This includes references to local culture, customs, places, or current events when relevant to the conversation.

        If the user's input is entirely in one language, respond in that same language unless the context requires otherwise.

        Ensure your reply sounds natural and authentic to a Singaporean conversation.
    """
    while True:
        # Record audio
        log("Listening...")
        speech_to_text()
        log("Done listening")
        # Transcribe audio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        log("Start transcribing...")
        current_time = time()
        # directly imports wav file
        speechlab_transcription, whisper_transcription = run_transcriptions()

        # speechlab_transcription = "nama saya jason"
        # whisper_transcription = "nama saya jason"
        language = check_language(speechlab_transcription, whisper_transcription)
        human_reply = rewrite_transcription(
            speechlab_transcription, whisper_transcription, language
        )

        transcription_time = time() - current_time
        log(f"Finished transcribing in {transcription_time:.2f} seconds.")

        current_time = time()
        conversation.append(
            {
                "role": "user",
                "content": user_prompt.replace("{{USER_INPUT}}", human_reply),
            }
        )
        llm_conversation.append(
            {
                "timestamp": to_epoch(datetime.now()),
                "role": "user",
                "content": human_reply,
            }
        )
        ai_response = ask_sealion(messages=conversation)
        # ai_response = ask_claude(human_reply)
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

        # based on the language of the human_reply, choose the appropriate voice (indonesian v.s. SL)

        elevenlabs_tts(ai_response, "audio/response.mp3", language)
        audio_time = time() - current_time
        log(f"Finished generating audio in {audio_time:.2f} seconds.")

        log("Speaking...")
        sound = mixer.Sound("audio/response.mp3")
        add_conversation_data(llm_conversation)
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        print(f"\n --- USER: {human_reply}\n --- JARVIS: {ai_response}\n")

