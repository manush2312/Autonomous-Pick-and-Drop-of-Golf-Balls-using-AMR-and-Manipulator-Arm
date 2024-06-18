import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16

import os
import pyaudio
#from /home/user/anaconda3/envs/spt/lib/python3.9/site-packages/pyaudio/
import wave
from pydub import AudioSegment
import cv2
import threading
import numpy as np
import assemblyai as aai
import sys

#sys.path.append('./src/speech_to_text/speech_to_text/')
sys.path.append('./')
from lstm import forward
from temp import record_audio
# Global flag to control the recording loop
stop_recording = False
class_id = None

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Int16, '/class_id', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.publisher_count = 0

    def timer_callback(self):
        global class_id
        msg = Int16()
        if class_id is None:
            print('calling record function')
            class_id = self.record("recorded_audio.wav") 
        msg.data = class_id
        print('ros msg data: ', msg.data)
        self.publisher_.publish(msg)

    def record(self, filename, duration=5, sample_rate=16000, channels=2, chunk_size=1024):
        global stop_recording
        aai.settings.api_key = "a4333434ff7d4bdbbe7721d596706667"
        transcriber = aai.Transcriber()

        #transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/news.mp4")

        # Initialize PyAudio
        audio = pyaudio.PyAudio()

        # Open a new stream to record audio
        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk_size)

        print("Recording...")

        frames = []
        # Record audio until stop_recording flag is set orduration is reached
        background = np.zeros((200, 400, 3))
        text = 'Recording Audio, Press q to stop'
        cv2.putText(background, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA) 
        cv2.imshow('test', np.ones((640, 480)))
        while not stop_recording:
        # for _ in range(int(sample_rate / chunk_size * duration)):
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                stop_recording = True
                cv2.destroyAllWindows()
            data = stream.read(chunk_size)
            frames.append(data)
        
        print("Finished recording.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded audio to a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        print('#' * 100)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Convert the WAV file to MP3
        audio_wav = AudioSegment.from_wav(filename)
        audio_wav.export(filename.replace('.wav', '.mp3'), format="mp3")

        print("Audio saved as:", filename.replace('.wav', '.mp3'))

        transcript = transcriber.transcribe("./recorded_audio.mp3")
        print(transcript.text)
        return forward(transcript.text)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
