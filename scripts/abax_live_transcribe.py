import argparse
from ws4py.client.threadedclient import WebSocketClient
import time
import threading
import sys
import urllib.parse
import queue
import json
import os
import pyaudio
import ssl
import logging


import contextlib
@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

from datetime import datetime
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
# 640 bytes, 1280 bytes
# 1second of audio data = sampling rate (byte_rate) * 2 bytes

# their model is trained with 16khz. 44.1/48 khz is the norm for airpods, but code handles downsampling
# sampling rate = 16000 (16 kHz)
# modern devices try to use higher sampling rates to generate more info but require more bandwidth to transfer the data
# check that monochannel (WAV) audio is sent, stereochannel MP3 data type is not supported; 

def rate_limited(maxPerSecond):
    minInterval = 1.0 / float(maxPerSecond)
    def decorate(func):
        lastTimeCalled = [time.perf_counter(),0]
        def rate_limited_function(*args,**kargs):
            if lastTimeCalled[1]==0:
                lastTimeCalled[0]=time.perf_counter()
            elapsed = time.perf_counter() - lastTimeCalled[0]
            leftToWait =  minInterval*lastTimeCalled[1] - elapsed
            lastTimeCalled[1] += 1
            if leftToWait>0:
                time.sleep(leftToWait)
            ret = func(*args,**kargs)
            return ret
        return rate_limited_function
    return decorate


class AbaxStreamingClient(WebSocketClient):
    dt1 = datetime.now()
    
    def __init__(self, mode, audiofile, url, keywordfile=None, protocols=None, extensions=None, heartbeat_freq=None, byterate=32000,
                 save_adaptation_state_filename=None, ssl_options=None, send_adaptation_state_filename=None):
        super(AbaxStreamingClient, self).__init__(url, protocols, extensions, heartbeat_freq)
        self.final_hyps = []
        self.audiofile = audiofile
        self.keywordfile = keywordfile
        self.byterate = byterate
        self.final_hyp_queue = queue.Queue()
        self.save_adaptation_state_filename = save_adaptation_state_filename
        self.send_adaptation_state_filename = send_adaptation_state_filename

        self.ssl_options = ssl_options or {}

        if self.scheme == "wss":
            # Prevent check_hostname requires server_hostname (ref #187)
            if "cert_reqs" not in self.ssl_options:
                self.ssl_options["cert_reqs"] = ssl.CERT_NONE

        self.mode = mode
        with ignoreStderr():
            self.audio = pyaudio.PyAudio()
        self.isStop = False
    
    @rate_limited(25)
    def send_data(self, data):
        self.send(data, binary=True)
    
    def opened(self):
        def send_data_to_ws():
            if self.send_adaptation_state_filename is not None:
                try:
                    adaptation_state_props = json.load(open(self.send_adaptation_state_filename, "r"))
                    self.send(json.dumps(dict(adaptation_state=adaptation_state_props)))
                except:
                    e = sys.exc_info()[0]

            if self.mode == 'stream':
                stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
                while not self.isStop:
                    data = stream.read(int(self.byterate / 8), exception_on_overflow=False)
                    self.send_data(data) # send data

                stream.stop_stream()
                stream.close()
                self.audio.terminate()
                
            elif self.mode == 'file':
                with self.audiofile as audiostream:
                    for block in iter(lambda: audiostream.read(int(self.byterate/25)), ""):
                        self.send_data(block)
                        if (len(block) == 0):
                          break
            self.send("EOS")
            
        t = threading.Thread(target=send_data_to_ws)
        t.start()


    def received_message(self, m):
        response = json.loads(str(m))
        if response['status'] == 0:
            if 'result' in response:
                trans = response['result']['hypotheses'][0]['transcript']
                if response['result']['final']:
                    print(response)
                    dt2 = datetime.now()
                    delta = (dt2 - self.dt1).total_seconds()
                    trans = trans.replace("<unk>","").lower()

                    #print("\033[H\033[J") # clear console for better output
                    # if (trans != "<blank>") and (trans != "") and (trans != "ya") and (trans != "i") and (trans != "a") and (trans != "okay") and (trans != "嗯"):
                    self.final_hyps.append(trans)
                    print ("+" + str(delta) + ": " + trans)
        else:
            if 'message' in response:
                print("Server message: %s" %  response['message'])


    def get_full_hyp(self, timeout=60):
        return self.final_hyp_queue.get(timeout)

    def closed(self, code, reason=None):
        self.final_hyp_queue.put(" ".join(self.final_hyps))


def main():

    parser = argparse.ArgumentParser(description='Command line client for kaldigstserver')
    parser.add_argument('-o', '--option', default="file", dest="mode", help="Mode of transcribing: audio file or streaming")
    parser.add_argument('-u', '--uri', default="ws://localhost:8888/client/ws/speech", dest="uri", help="Server websocket URI")
    parser.add_argument('-r', '--rate', default=32000, dest="rate", type=int, help="Rate in bytes/sec at which audio should be sent to the server. NB! For raw 16-bit audio it must be 2*samplerate!")
    parser.add_argument('-t', '--token', default="", dest="token", help="User token")
    parser.add_argument('-m', '--model', default=None, dest="model", help="model in azure container")
    parser.add_argument('-l', '--log', default="", dest="log", help="save the text result")
    parser.add_argument('--save-adaptation-state', help="Save adaptation state to file")
    parser.add_argument('--send-adaptation-state', help="Send adaptation state from file")
    parser.add_argument('--content-type', default='', help="Use the specified content type (empty by default, for raw files the default is  audio/x-raw, layout=(string)interleaved, rate=(int)<rate>, format=(string)S16LE, channels=(int)1")
    parser.add_argument('audiofile', nargs='?', help="Audio file to be sent to the server", type=argparse.FileType('rb'), default=sys.stdin)

    args = parser.parse_args()

    if args.mode == 'file' or args.mode == 'stream':
        content_type = args.content_type
        if content_type == '' and args.audiofile.name.endswith(".raw") or args.mode == 'stream':
            content_type = "audio/x-raw, layout=(string)interleaved, rate=(int)%d, format=(string)S16LE, channels=(int)1" %(args.rate/2)

        ws = AbaxStreamingClient(args.mode, args.audiofile, 
                    args.uri + '?%s' % (urllib.parse.urlencode([("content-type", content_type)])) + '&%s' % (urllib.parse.urlencode([("accessToken", args.token)])) + '&%s' % (urllib.parse.urlencode([("token", args.token)])) + '&%s' % (urllib.parse.urlencode([("model", args.model)])), 
                    byterate=args.rate,
                    save_adaptation_state_filename=args.save_adaptation_state, send_adaptation_state_filename=args.send_adaptation_state)

        ws.connect()
        result = ws.get_full_hyp()

        if (args.log != ""):
            with open(args.log, 'a', encoding='utf-8') as logfile:
                logfile.write(str(args.audiofile.name.split("/")[-1].replace(".wav","")) + " " + result + '\n')
                logfile.close()
            
    else:
        print('\nTranscribe mode must be file or stream!\n')

if __name__ == "__main__":
    main()
