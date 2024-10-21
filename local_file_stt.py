import os
import sys
import argparse
from scripts.abax_live_transcribe import AbaxStreamingClient


def transcribe_audio_file(
    audio_file_path: str, server_uri: str, token: str, model: str, rate: int = 32000
) -> str:
    """
    Transcribe an audio file using the AbaxStreamingClient.

    Args:
        audio_file_path (str): Path to the audio file to transcribe.
        server_uri (str): URI of the transcription server.
        token (str): User token for authentication.
        model (str): Model to use for transcription.
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

    content_type = f"audio/x-raw, layout=(string)interleaved, rate=(int){rate // 2}, format=(string)S16LE, channels=(int)1"

    uri_params = {
        "content-type": content_type,
        "accessToken": token,
        "token": token,
        "model": model,
    }
    full_uri = f"{server_uri}?{'&'.join([f'{k}={v}' for k, v in uri_params.items()])}"

    with open(audio_file_path, "rb") as audio_file:
        client = AbaxStreamingClient(
            mode="file", audiofile=audio_file, url=full_uri, byterate=rate
        )

        client.connect()
        result = client.get_full_hyp()

    return result


def main() -> None:
    """
    Main function to handle command-line arguments and run the transcription.
    """
    # parser = argparse.ArgumentParser(
    #     description="Transcribe an audio file using AbaxStreamingClient"
    # )
    # parser.add_argument(
    #     "--audio", default="audio/recording.wav", help="Path to the audio file"
    # )
    # parser.add_argument("--uri", required=True, help="Server websocket URI")
    # parser.add_argument("--token", required=True, help="User token")
    # parser.add_argument("--model", required=True, help="Model to use for transcription")
    # parser.add_argument("--rate", type=int, default=32000, help="Rate in bytes/sec")

    # args = parser.parse_args()

    try:
        transcription = AbaxStreamingClient(
            "file",
            argparse.FileType('rb')("/Users/ob1/projects/speechlab/audio/recording.wav"),
            "ws://54.255.127.241:8080/client/ws/speech?content-type=&accessToken=&token=&model=28122023-onnx",
            byterate=32000,
            save_adaptation_state_filename=None,
            send_adaptation_state_filename=None,
        )
        transcription.connect()
        result = transcription.get_full_hyp()
        print(f"Transcription: {result}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
