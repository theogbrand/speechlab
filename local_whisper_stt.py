import logging
import os
import time
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf

load_dotenv()
MODEL_ID = os.getenv("MODEL_ID", default="openai/whisper-large-v3-turbo")


if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        logging.warning(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        logging.warning(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

else:
    device = torch.device("mps")
    torch_dtype = torch.float16

logging.info(device)
model_id = MODEL_ID
logging.info("Instantiating model...")
ins_time = time.time()
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)
logging.info(f"{MODEL_ID} loaded: --- {time.time() - ins_time} seconds ---")


def local_whisper_transcribe(audio_file_path: str) -> str:
    """
    Transcribe an audio file using the local Whisper models.

    Args:
        audio_file_path (str): Path to the audio file to transcribe.

    Returns:
        str: The transcribed text.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio file is not a .wav file.
    """
    # Determine if MPS/CUDA is available
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    if not audio_file_path.lower().endswith(".wav"):
        raise ValueError("The audio file must be a .wav file")

    try:
        logging.info("load file")
        audio_array, sample_rate = sf.read(audio_file_path)

        logging.info("transcribe")
        start_time = time.time()
        result = pipe(audio_array)
        logging.info(result)
        transcribe_time = time.time() - start_time
        logging.info("Time taken: --- %s seconds ---" % (transcribe_time))
        result["time_taken"] = transcribe_time
        return result
    except Exception as e:
        logging.error("error:", str(e))
        return {"error": str(e)}


if __name__ == "__main__":
    print(
        local_whisper_transcribe(
            audio_file_path="audio/recording.wav",
        )
    )
