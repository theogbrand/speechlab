"""Function for recording audio from a microphone."""
import io
import typing
import time
import wave
from pathlib import Path

from rhasspysilence import WebRtcVadRecorder, VoiceCommand, VoiceCommandResult
import pyaudio
import keyboard

pa = pyaudio.PyAudio()


def speech_to_text() -> None:
    """
    Records audio until silence is detected
    Saves audio to audio/recording.wav
    """
    recorder = WebRtcVadRecorder(
        vad_mode=3,
        silence_seconds=60,
    )
    print("Press Enter to start recording...")
    keyboard.wait('enter')  # Wait for Enter key to start recording
    print("Recording started. Press Enter to stop.")
    recorder.start()
    # file directory
    wav_sink = "audio/"
    # file name
    wav_filename = "recording"
    if wav_sink:
        wav_sink_path = Path(wav_sink)
        if wav_sink_path.is_dir():
            # Directory to write WAV files
            wav_dir = wav_sink_path
        else:
            # Single WAV file to write
            wav_sink = open(wav_sink, "wb")
    voice_command: typing.Optional[VoiceCommand] = None
    audio_source = pa.open(
        rate=16000,
        format=pyaudio.paInt16,
        channels=1,
        input=True,
        frames_per_buffer=960,
    )
    audio_source.start_stream()

    def buffer_to_wav(buffer: bytes) -> bytes:
        """Wraps a buffer of raw audio data in a WAV"""
        rate = int(16000)
        width = int(2)
        channels = int(1)

        with io.BytesIO() as wav_buffer:
            wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
            with wav_file:
                wav_file.setframerate(rate)
                wav_file.setsampwidth(width)
                wav_file.setnchannels(channels)
                wav_file.writeframesraw(buffer)

            return wav_buffer.getvalue()

    try:
        chunk = audio_source.read(960)
        while chunk:
            # Look for speech/silence
            voice_command = recorder.process_chunk(chunk)

            if keyboard.is_pressed('enter'):
                print("Recording stopped.")
                # Reset
                audio_data = recorder.stop()
                if audio_data is None:
                    print("No audio data recorded.")
                    break
                if wav_dir:
                    # Write WAV to directory
                    wav_path = (wav_dir / time.strftime(wav_filename)).with_suffix(
                        ".wav"
                    )
                    wav_bytes = buffer_to_wav(audio_data)
                    wav_path.write_bytes(wav_bytes)
                    print(f"Audio saved to {wav_path}")
                    break
                elif wav_sink:
                    # Write to WAV file
                    wav_bytes = buffer_to_wav(audio_data)
                    wav_sink.write(wav_bytes)
                    print("Audio saved to WAV file")
                    break
            # Next audio chunk
            chunk = audio_source.read(960)

    finally:
        try:
            audio_source.close_stream()
        except Exception:
            pass


if __name__ == "__main__":
    speech_to_text()
