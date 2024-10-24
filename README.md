# UPDATES on Speechlab ASR engine

# Installation

# Server setup ONLY
* use system python, do not use venv, install packages directly using apt-install
* install requirements ONLY for ```scripts/abax_live_transcribe.py``` and ```sl_api_wrapper.py```
```bash
sudo apt update
sudo apt install python3-pip

sudo apt install python3-ws4py python3-pyaudio
```

* Run server
```bash

```

Updated on: October 16, 2024
--------------------------------------------------------------------------------

# Overview
   
This is the guideline to setup and running demo with the latest Speechlab ASR models
This guide includes guideline for the Online ASR system


Load the docker application

```
docker load -i images/abx-sdk-31122024.tar
```

### Environment and setup 

- Project folder structure

		├── docker-compose.yml
		├── images
		│   └── abx-sdk-31122024.tar
		├── models
		│   └── 28122023-onnx
		│       ├── app.conf
		│       ├── bpe.model
		│       ├── bpe.model.mal
		│       ├── ctc.onnx
		│       ├── decoder.onnx
		│       ├── encoder.onnx
		│       ├── units.txt
		│       ├── vad
		│       └── words.txt
		├── README.md
		└── scripts
			└── abax_live_transcribe.py


- Libraries/tools needed

        Docker, docker-compose
        ws4py, urllib, pyaudio

- Input data
	
	- Monochannel, mono track
	- Preferred sampling rate 16khz
	- Preferred .wav format

- System requirement
	
	- Preferred Ubuntu 24.04 onwards, or Amazon Linux 2
	- vCPU cores, 16GB RAM, 50Gb storage
	

## Online ASR system

### Start/Stop the system

- **Start the system**
```
  $ docker-compose -f docker-compose.yml up -d
```

- **Stop the system**
```
  $ docker-compose -f docker-compose.yml down
```

- **Test with a sample audio**

```
  $ python scripts/abax_live_transcribe.py -u ws://localhost:8080/client/ws/speech -m 28122023-onnx <path/to/audio.wav>
```

### Customisation and scaling

- **Vertical scaling**

	Increase the number of threadings to handle request via this environment variable in docker-compose file, before you start the docker-compose

	```
	  INSTANCE_NUM=2
	```

- **Horizontal scaling**

	Increase the number of workers to handle request while starting the docker-compose

	```
	  $ docker-compose -f docker-compose.yml up -d --scale decoding-sdk-worker=2
	```
	
	With the number of workers = 2, and number of threads = 2, the system can handle 4 concurrent requests.
	

- **Change the port**

```
  $ python scripts/abax_live_transcribe.py -u ws://localhost:8080/client/ws/speech -m 28122023-onnx <path/to/audio.wav>
```


### Understand the output

- Partial results (the decoding is happening and the result is not final yet)
  + JSON Object

```
    { 'status': 0,
      'segment': 1,
      'result': {
        'hypotheses': [{
          'transcript': "<transcript>"
        }]
        'final': True/False
      },
      'id': '<unique-id>'
    }
```

  + Example:
  
```
    INFO 2021-12-16 16:14:27,490 ...  of course it's mainly strong health systems in countries where uh mother's. 
    INFO 2021-12-16 16:14:28,201 {'status': 0, 'segment': 55, 'result': {'hypotheses': [{'transcript': "or whatever pieces of course it's mainly strong health systems in countries i'll just wear a mother's."}], 'final': False}, 'id': '74f72e7a-06b6-48de-97ac-bb421c90ba63'} 
    INFO 2021-12-16 16:14:28,201 ... se it's mainly strong health systems in countries i'll just wear a mother's. 
    INFO 2021-12-16 16:14:29,426 {'status': 0, 'segment': 55, 'result': {'hypotheses': [{'transcript': "or whatever pieces of course it's mainly strong health systems in countries i'll just wear a mother's birthday."}], 'final': False}, 'id': '74f72e7a-06b6-48de-97ac-bb421c90ba63'} 
    INFO 2021-12-16 16:14:29,427 ... ainly strong health systems in countries i'll just wear a mother's birthday. 
    INFO 2021-12-16 16:14:31,035 {'status': 0, 'segment': 55, 'result': {'hypotheses': [{'transcript': "or whatever pieces of course it's mainly strong health systems in countries i'll just wear a mother's safely."}], 'final': False}, 'id': '74f72e7a-06b6-48de-97ac-bb421c90ba63'} 
    INFO 2021-12-16 16:14:31,035 ...  mainly strong health systems in countries i'll just wear a mother's safely. 
    INFO 2021-12-16 16:14:34,708 {'status': 0, 'result': {'final': True, 'hypotheses': [{'transcript': "or whatever pieces of course it's mainly strong health systems in countries i'll just wear a mother's safely.", 'likelihood': 345.01, 'confidence': 0.1529537658105976}]}, 'segment-length': 11.34, 'segment-start': 356.79, 'total-length': 368.5, 'segment': 55, 'id': '74f72e7a-06b6-48de-97ac-bb421c90ba63'} 
```
  
- Final results (The decoding for the utterance is completed, the transcripts are finalized).
  + Final=True in the JSON object.
  

  
  
  
  
  
  
# End.
