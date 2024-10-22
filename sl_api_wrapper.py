import logging
import time
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from scripts.abax_live_transcribe import AbaxStreamingClient

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Configure CORS
origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPEECHLAB_PUBLIC_SERVER_URI = "54.255.127.241"

@app.get("/ping")
def ping():
    return {"response": "PONG"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)): # remind end users that monochannel (WAV) audio must be sent, stereochannel MP3 data type is not supported; 
    try:
        audio_content = await file.read()
        logging.info("loaded file")
        audio_file = io.BytesIO(audio_content)

        logging.info("transcribing")
        start_time = time.time()

        client = AbaxStreamingClient(
            mode="file",
            audiofile=audio_file,
            url=f"ws://{SPEECHLAB_PUBLIC_SERVER_URI}:8080/client/ws/speech?content-type=&accessToken=&token=&model=28122023-onnx",
            byterate=32000,
        )

        client.connect()
        transcription = client.get_full_hyp()
        print(f"Transcription: {transcription}")

        result = {}
        result["transcription"] = transcription
        logging.info(transcription)
        transcribe_time = time.time() - start_time
        logging.info("Time taken: --- %s seconds ---" % (transcribe_time))
        result["time_taken"] = transcribe_time
        return JSONResponse(content=result)
    except Exception as e:
        logging.error("error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        await file.close()
