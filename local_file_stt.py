import os
from scripts.abax_live_transcribe import AbaxStreamingClient


def transcribe_audio_file(
    audio_file_path: str, server_uri: str, rate: int = 32000
) -> str:
    """
    Transcribe an audio file using the AbaxStreamingClient.

    Args:
        audio_file_path (str): Path to the audio file to transcribe.
        server_uri (str): URI of the transcription server.
        rate (int, optional): Rate in bytes/sec at which audio should be sent to the server. Defaults to 32000.

    Returns:
        str: The transcribed text.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio file is not a .wav file.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    if not audio_file_path.lower().endswith(".wav"):
        raise ValueError("The audio file must be a .wav file")

    with open(audio_file_path, "rb") as audio_file:
        client = AbaxStreamingClient(
            mode="file",
            audiofile=audio_file,
            url=f"ws://{server_uri}:8080/client/ws/speech?content-type=&accessToken=&token=&model=28122023-onnx",
            byterate=rate,
        )

        client.connect()
        result = client.get_full_hyp()
        print(f"Transcription: {result}")

    return result


if __name__ == "__main__":
    transcribe_audio_file(
        audio_file_path="audio/recording.wav",
        server_uri="54.255.127.241",
    )
