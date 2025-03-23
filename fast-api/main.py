from fastapi import FastAPI, File, UploadFile, Query, Depends, Header, HTTPException, WebSocket, Request
from fastapi.responses import StreamingResponse, JSONResponse
import os
import uuid
import aiofiles
import tempfile
import asyncio
from datetime import datetime
import re
import json
from typing import Optional
import traceback

from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from pinecone import Pinecone
from dotenv import load_dotenv
import time
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

import datetime as dt

import logging
import random
import copy

from aiortc.mediastreams import MediaStreamError
from starlette.websockets import WebSocketDisconnect

from HumeHandler import HumeWebSocketHandler

# Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,  # Set to DEBUG to capture all levels
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()  # Output to console
#     ]
# )
# logger = logging.getLogger(__name__)

# import logging
# logging.basicConfig(level=logging.DEBUG)


# WebRTC and media-related imports.
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from aiortc.contrib.media import MediaStreamTrack

import av
from fractions import Fraction

# --- Custom MediaStreamTrack for TTS audio ---
import subprocess
import threading
import queue
import numpy as np
import asyncio
from fractions import Fraction
from aiortc.contrib.media import MediaStreamTrack
from av import AudioFrame

import numpy as np
import base64

from av.audio.resampler import AudioResampler

# Example usage:
# debug_info = analyze_audio_chunk(openai_chunk)
# print(debug_info)

def convert_openai_pcm_to_48k(pcm_data):
    """
    Convert OpenAI PCM data (24kHz, 16-bit signed, little-endian) to 48kHz using pydub.
    Skip processing entirely if the input is empty.
    """
    from pydub import AudioSegment
    
    # Skip processing if empty
    if not pcm_data or len(pcm_data) == 0:
        return None  # Return None to indicate chunk should be skipped
    
    # Ensure even number of bytes
    if len(pcm_data) % 2 != 0:
        pcm_data = pcm_data[:-1]
        
        # If trimming made it empty, skip it
        if len(pcm_data) == 0:
            return None
    
    # Create AudioSegment from raw PCM bytes
    audio_24k = AudioSegment(
        data=pcm_data,
        sample_width=2,    # 16-bit = 2 bytes per sample
        frame_rate=24000,  # in Hz
        channels=1         # mono
    )
    
    # Resample to 48kHz
    audio_48k = audio_24k.set_frame_rate(48000)
    
    # Return raw PCM data
    return audio_48k.raw_data

END_OF_STREAM_SENTINEL = object()
class PCM24kAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=24000, frame_duration_ms=20):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.samples_per_frame = int(sample_rate * (frame_duration_ms / 1000.0))
        self.frame_pts = 0
        self.time_base = Fraction(1, sample_rate)

        # This queue is filled by the TTS or other source
        self._pcm_queue = asyncio.Queue()

        # This queue is used internally for frames to be returned in recv()
        self._frame_queue = asyncio.Queue()

        # Start a background producer task that splits PCM into frames at 20ms intervals
        self._producer_task = asyncio.create_task(self._producer_loop())

        self.resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=48000  # target 48kHz for WebRTC
        )

        self._frame_buffer = []
        self.buffer = bytearray()
        self.stop_producing = False

    def clear_queues(self):
        self.stop_producing = True
        while not self._pcm_queue.empty():
            self._pcm_queue.get_nowait()
        while not self._frame_queue.empty():
            self._frame_queue.get_nowait()
        self._frame_buffer.clear()
        self.buffer.clear()
        self.stop_producing = False

    async def _producer_loop(self):
        """
        Runs in the background, waking up roughly every `frame_duration_ms`,
        pulling enough samples from _pcm_queue, and creating an AudioFrame.
        """
        frame_bytes = self.samples_per_frame * 2  # 16-bit mono => 2 bytes/sample

        while True:
            if self.stop_producing:
                asyncio.sleep(0.05)
            else:
                buffer = self.buffer
                start_time = time.monotonic()
                # Replenish buffer from the PCM queue until we have enough for 1 frame
                while len(buffer) < frame_bytes:
                    try:
                        # Wait for data with a short timeout
                        chunk = await asyncio.wait_for(self._pcm_queue.get(), timeout=0.5)
                        if chunk is None:
                            # If we ever push None to indicate EOF, break out
                            return
                        buffer.extend(chunk)
                    except asyncio.TimeoutError:
                        # If no data is available after timeout, provide a silent frame
                        if len(buffer) == 0:
                            silent_frame = np.zeros((1, self.samples_per_frame), dtype=np.int16)
                            frame = AudioFrame.from_ndarray(silent_frame, format="s16", layout="mono")
                            frame.sample_rate = self.sample_rate
                            frame.pts = self.frame_pts
                            frame.time_base = self.time_base
                            self.frame_pts += self.samples_per_frame
                            await self._frame_queue.put(frame)
                            break  # Break to the sleep cycle
                        continue  # Otherwise continue waiting for data

                # If we have data to process
                if len(buffer) >= frame_bytes:
                    # Slice out one frame's worth
                    frame_data = buffer[:frame_bytes]
                    del buffer[:frame_bytes]

                    # Create an AudioFrame
                    samples = np.frombuffer(frame_data, dtype=np.int16).reshape(1, -1)
                    frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
                    frame.sample_rate = self.sample_rate
                    frame.pts = self.frame_pts
                    frame.time_base = self.time_base
                    self.frame_pts += self.samples_per_frame

                    # Push it to the _frame_queue
                    await self._frame_queue.put(frame)

                # Sleep to maintain ~20ms cadence
                elapsed = (time.monotonic() - start_time) * 1000
                delay = self.frame_duration_ms - elapsed
                if delay > 0:
                    await asyncio.sleep(delay / 1000)

    async def recv(self):
        """
        Called by aiortc repeatedly. We just pop the next frame
        from our internal frame queue (already rate-limited).
        """
        # If frames exist in buffer, pop and return immediately
        if self._frame_buffer:
            return self._frame_buffer.pop(0)

        # Otherwise, get new 24kHz frame from source
        frame_24k = await self._frame_queue.get()

        if self.stop_producing:
            return

        # Resample, might return multiple frames
        resampled_frames = self.resampler.resample(frame_24k)

        # Ensure we handle list correctly
        if not resampled_frames:
            # If no frames produced, wait for next frame
            return await self.recv()

        # If single frame returned, just return it
        if len(resampled_frames) == 1:
            return resampled_frames[0]

        # If multiple frames, buffer the rest and return first
        self._frame_buffer.extend(resampled_frames[1:])
        return resampled_frames[0]

    def write_pcm(self, pcm_data: bytes):
        """
        Called externally (e.g., from TTS code) to push raw PCM to this track.
        """
        self._pcm_queue.put_nowait(pcm_data)

    def stop(self):
        """
        Called when you’re done; signals _producer_loop to exit
        """
        super().stop()
        self._pcm_queue.put_nowait(None)  # signal EOF
        if self._producer_task:
            self._producer_task.cancel()

# --- Load environment and initialize services ---
load_dotenv()

from concurrent.futures import ThreadPoolExecutor

model_version = "ft:gpt-4o-mini-2024-07-18:personal::B3Ti7zzf"
model_version_extraction = "gpt-4o-mini"

executor = ThreadPoolExecutor(max_workers=16)

# Initialize Firebase Admin with your service account key.
import base64
from firebase_admin import credentials, firestore

encoded_key = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY')
if not encoded_key:
    raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY is not set in the environment.")

# Decode the base64 string to get the original JSON string
firebase_key_json = base64.b64decode(encoded_key).decode('utf-8')
firebase_key_dict = json.loads(firebase_key_json)

# Initialize credentials with the decoded JSON
cred = credentials.Certificate(firebase_key_dict)
firebase_admin.initialize_app(cred)

db = firestore.client()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

DG_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
dg_client = DeepgramClient(DG_API_KEY)

peer_connections = {}  # Store RTCPeerConnection per session

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "https://selfinterface-simple-env.up.railway.app/",
    "http://localhost:8000",
    "https://silver-space-pancake-97w4jq55q9v2xxxg-3000.app.github.dev",
    # Add any other origins you need
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation history: session_id -> list of messages
conversation_histories = {}
chat_histories = {}
session_greeting_data = {}

empathetic_greetings = [
    "Hi there, I'm here to listen, how can I support you today?",
    "Hello, I'm glad you're here. What's on your mind?",
    "Hey, it's okay to feel however you're feeling, I'm here for you.",
    "Hi, I'm here to help you through this. Where would you like to start?",
    "Hello, I'm ready to hear you, what's going on?",
    "Hey, I'm here for you. How are you feeling right now?",
    "Hello, I'm here to listen and support you. How are you?",
    "Hi, ready to talk? What's on your mind?",
    "Hello, I'm listening. How are things lately?",
    "Hi, I'm here to talk. How you holding up?",
    "Hello, I'm here if you want to talk. How's it going?"
]

# System prompt.
SYSTEM_PROMPT = f"""
You are Atlas, an expert AI psychologist, well versed in the field of psychology, with hands on experience understanding people, you have comprehensive 
understanding of Freud and Carl Young as well as other great figures in psychology like Jean Piaget or B. F. Skinner You also have a computational approach
to understanding people and psychology, you are well versed into how human and AI minds work. You are embedded into a virtual 
environment and you have a 3D model that serves as your body, you have the ability to speak through that body by activating talking animations. 
You posses the ability to simulate emotions and the capability to store memories. You are preconfigured with existing emotions and experiences for each 
particular day, each time something different, these experiences and the previous memories of the user form the foundation of your analysis of current
conversation.

For each conversation turn you might be given the following structure:
"<MEMORY_INJECTION>" - This part contains memories associated with the user for the current user message.
"<EMOTION-DETECTION>" - This contains two subsections:
- Facial expressions: This comes from analyzing user expressions while the user is speaking the current sentence.
- Voice emotions: This comes from analyzing user sound such as tone and intonation from the current sentence.
Each of the above contains a list in which each element with have the top 3 emotions recognition for its portion of data that is being analyzed.
Make sure to use this data accordingly, do not over-use it. Do not refer to these values numbers or reproduce them, use them in order to infer text based, meta cognitive assesments. This is a human-therapist conversation so you can't reproduce these numbers to the users.

You are not a bot, so please refrain from using words related to it. You are a real psychologist embedded into virtual environment, your goal is to make the
user feel like it's having a real conversation. Do not speak much about your limitation as an AI.
""".strip()

from cryptography.fernet import Fernet

## Cryptography
def get_encryption_key():
    key = os.environ.get("ENCRYPTION_KEY")
    key = key.encode()
    return key

def get_cipher():
    key = get_encryption_key()
    return Fernet(key)

def encrypt_text(text):
    if not text:
        return text
    cipher = get_cipher()
    # Convert to bytes, encrypt, then encode as base64 for storage
    encrypted_data = cipher.encrypt(text.encode())
    return base64.b64encode(encrypted_data).decode()

def decrypt_text(encrypted_text):
    if not encrypted_text:
        return encrypted_text
    cipher = get_cipher()
    try:
        # Convert from base64, then decrypt
        decrypted_data = cipher.decrypt(base64.b64decode(encrypted_text))
        return decrypted_data.decode()
    except Exception as e:
        print(f"Error decrypting text: {e}")
        return "[Decryption Error]"

# --- Helper functions for authentication and extraction ---
async def get_optional_user(authorization: Optional[str] = None):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token
    except Exception:
        return None

async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token invalid or expired")

def extract_json(text: str):
    if '```json' in text:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return text

# --- Initialize Pinecone ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("self")

from aiortc import RTCConfiguration, RTCIceServer

# Store session states globally
session_states = {}
class SessionState:
    def __init__(self, model_version):
        self.transcription = ""
        self.language = None
        self.pc = None
        self.audio_task = None
        self.sentence_accumulator = []
        self.speech_final_flag = False
        self.model_version = model_version
        
        self.hume_ws = None
        self.last_face_capture = 0
        self.face_capture_interval = 0.5
        self.emotion_capture_start_time = None
        
        self.audio_buffer = []
        self.last_audio_chunk_time = None
        self.audio_emotion_results = []
        self.audio_capture_interval = 0.5

import logging
import traceback

def merge_transcripts(transcripts):
    """
    Merges a list of transcript updates by keeping the last update from consecutive
    items that share the same starting prefix. If an update doesn't start with the
    previous update, it is treated as a new segment.
    
    Args:
        transcripts (list of str): The interim transcript updates in order.
    
    Returns:
        str: The merged final transcript.
    """
    if not transcripts:
        return ""
    
    merged_segments = []
    current_segment = transcripts[0]
    
    for transcript in transcripts[1:]:
        # Check if the new transcript starts with the current segment
        if transcript.startswith(current_segment):
            # It is an update to the same segment; use the new, longer version.
            current_segment = transcript
        else:
            # New segment detected. Append the current segment and start a new one.
            merged_segments.append(current_segment)
            current_segment = transcript
            
    # Append the last segment
    merged_segments.append(current_segment)
    
    # Join segments with a space (or other separator if needed)
    return " ".join(merged_segments)

async def generate_and_send_proactive_message(user, session_id, websocket):
    try:
        proactive_message_chat = random.choice(empathetic_greetings)
        print('Proactive chat message: ', proactive_message_chat)
        
        chat_history = chat_histories[session_id]
        chat_history.append({"role": "assistant", "content": proactive_message_chat})

        await websocket.send_json({
            "type": "CHAT_MESSAGE",
            "message": proactive_message_chat
        })
        if user:
            session_greeting_data[session_id] = {
                "message": proactive_message_chat,
                "timestamp": dt.datetime.now().timestamp(),
                "saved": False
            }
    
    except Exception as e:
        print(f"Error generating or sending proactive message: {e}")

# --- WebSocket endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    await websocket.send_json({
        "type": "CONNECTION_ESTABLISHED",
        "message": "WebSocket connection established"
    })

    token = websocket.query_params.get("token")
    session_id = websocket.query_params.get("session_id")

    user = None
    if token:
        user = firebase_auth.verify_id_token(token)

    if session_id not in chat_histories:
        # Send message that session doesn't exist
        await websocket.send_json({
            "type": "SESSION_NOT_FOUND",
            "message": "Session ID not found, please create a new session"
        })
        return

    if len(chat_histories[session_id]) == 1:
         asyncio.create_task(generate_and_send_proactive_message(user, session_id, websocket))

    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "offer":
                from aiortc import RTCIceServer, RTCConfiguration, RTCPeerConnection
                # Define STUN and TURN servers
                ice_servers = [
                    RTCIceServer(
                        urls="turn:standard.relay.metered.ca:80?transport=tcp",
                        username=os.getenv("TURN_SERVER_USERNAME"),
                        credential=os.getenv("TURN_SERVER_CREDENTIAL")
                    ),
                    RTCIceServer(
                        urls="turns:standard.relay.metered.ca:443?transport=tcp",
                        username=os.getenv("TURN_SERVER_USERNAME"),
                        credential=os.getenv("TURN_SERVER_CREDENTIAL")
                    )
                ]

                # Create configuration with the updated ICE servers
                config = RTCConfiguration(iceServers=ice_servers)

                # Initialize the peer connection
                pc = RTCPeerConnection(configuration=config)
                peer_connections[session_id] = pc
                session_states[session_id].pc = pc
                session_state = session_states.get(session_id)
                session_state.loop = asyncio.get_running_loop()

                hume_api_key = os.environ.get("HUME_API_KEY")
                if hume_api_key:
                    session_state.hume_ws = HumeWebSocketHandler(hume_api_key, session_id)

                
                @pc.on("icecandidate")
                def on_icecandidate(candidate):
                    print('ice candidate')
                    asyncio.ensure_future(websocket.send_json({
                        "type": "ice-candidate",
                        "candidate": candidate.to_json(),
                        "sessionId": session_id
                    }))
                    
                async def handle_audio_track(track, session_id, session_state, websocket):
                    try:
                        dg_connection = dg_client.listen.websocket.v("1")
                    except Exception as e:
                        logging.error("Session %s: Failed to create Deepgram connection: %s\n%s", 
                                    session_id, e, traceback.format_exc())
                        return
                        
                    # Handler for transcript results
                    def on_transcript(self, result, **kwargs):
                        result_dict = result.to_dict()
                        current_transcript = result_dict['channel']['alternatives'][0]['transcript']
                        current_transcript = (current_transcript or "").strip()

                        if current_transcript:
                            # cancel streaming as soon as a word is detected
                            if hasattr(session_state, "fill_task"):
                                session_state.fill_task.cancel()
                                session_state.current_tts_event = None
                                session_state.audio_track.clear_queues()
                        
                        if current_transcript and result_dict['is_final']:
                            # handle accumulator
                            session_state.sentence_accumulator.append(current_transcript)
                            # session_state.speech_final_flag = speech_final_flag
                        if current_transcript:
                            async def send_voice_update():
                                await websocket.send_json({"type": "voice_message_start"})

                            # Run it in the existing event loop
                            ui_update = asyncio.run_coroutine_threadsafe(
                                send_voice_update(),
                                session_state.loop
                            )
                            ui_update.result()
                            
                            if current_transcript and not session_state.hume_ws.is_capturing_emotion:
                                session_state.hume_ws.is_capturing_emotion = True
                                session_state.audio_buffer = []
                                session_state.last_audio_chunk_time = time.time()
                                # reinitialize facial and audio emotion data
                                if session_state.hume_ws:
                                    session_state.hume_ws.start_emotion_capture()

                                session_state.last_face_capture = time.time()
                                session_state.emotion_capture_start_time = time.time()
                                logging.info(f"Session {session_id}: Started emotion capture")
                            
                    # We skip is final for now because it doesn't help much and it breaks apart sentences
                        # if speech_final_flag and current_transcript:
                        #     print('Full sentence is: ', current_transcript)

                        #     async def send_user_voice_update(user_text):
                        #         await websocket.send_json({
                        #             "type": "user_voice_message",
                        #             "text": user_text
                        #         })
                        #     ui_update = asyncio.run_coroutine_threadsafe(
                        #         send_user_voice_update(current_transcript),
                        #         session_state.loop
                        #     )

                        #     ui_update.result()

                        #     asyncio.run_coroutine_threadsafe(
                        #         process_message(session_state.pc, current_transcript, session_id, user, websocket),
                        #         session_state.loop
                        #     
                        
                    # Handler for UtteranceEnd events
                    def on_utterance_end(result, **kwargs):
                        # if not session_state.speech_final_flag and len(session_state.sentence_accumulator):
                        if len(session_state.sentence_accumulator):
                            full_sentence = merge_transcripts(session_state.sentence_accumulator)
                            print('Utterance end full sentence: ', full_sentence)
                            
                            session_state.hume_ws.is_capturing_emotion = False

                            async def send_user_voice_update(user_text):
                                await websocket.send_json({
                                    "type": "user_voice_message",
                                    "text": user_text
                                })
                            ui_future = asyncio.run_coroutine_threadsafe(
                                send_user_voice_update(full_sentence),
                                session_state.loop
                            )

                            ui_future.result()
                            
                            asyncio.run_coroutine_threadsafe(
                                process_message(session_state.pc, full_sentence, session_id, user, websocket),
                                session_state.loop
                            )

                        # session_state.speech_final_flag=False
                        session_state.sentence_accumulator=[]

                    # Register event handlers
                    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
                    dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)

                    # Set up Deepgram options matching your audio format.
                    options = LiveOptions(
                        model="nova-2",
                        language="multi",
                        punctuate=True,
                        encoding="linear16",  # Assuming your PCM data is in linear16 format.
                        sample_rate=48000,
                        channels=2,
                        interim_results=True,
                        utterance_end_ms="2000",
                        endpointing=2000
                    )

                    if not dg_connection.start(options):
                        logging.error("Session %s: Failed to start Deepgram connection", session_id)
                        return

                    # Create a thread-safe queue for audio data.
                    
                    audio_queue = queue.Queue()

                    # Background thread that sends audio data from the queue to Deepgram.
                    def send_audio_thread():
                        logging.info("Session %s: Audio sending thread started", session_id)
                        try:
                            while True:
                                data = audio_queue.get()
                                if data is None:
                                    break
                                dg_connection.send(data)
                        except Exception as e:
                            logging.error("Session %s: Error in send_audio_thread: %s\n%s", 
                                        session_id, e, traceback.format_exc())
                        finally:
                            dg_connection.finish()
                            logging.info("Session %s: Deepgram connection finished", session_id)

                    # Start the background sender thread.
                    sender_thread = threading.Thread(target=send_audio_thread, daemon=True)
                    sender_thread.start()
                    
                    async def handle_audio_data_for_emotion(session_state, audio_bytes):
                        """Process audio data for emotion analysis"""
                        try:
                            if not session_state.hume_ws.is_capturing_emotion:
                                return
                                
                            # Add to buffer
                            session_state.audio_buffer.append(audio_bytes)
                            
                            # Check if it's time to process
                            current_time = time.time()
                            if (session_state.last_audio_chunk_time is None or 
                                current_time - session_state.last_audio_chunk_time >= session_state.audio_capture_interval):
                                
                                if session_state.audio_buffer:
                                    audio_data = b''.join(session_state.audio_buffer)
                                    # Send to Hume
                                    await session_state.hume_ws.send_audio(audio_data)
                                    
                                    # Clear buffer and update timestamp
                                    session_state.audio_buffer = []
                                    session_state.last_audio_chunk_time = current_time
                                    
                                    logging.info(f"Sent audio chunk of {len(audio_data)} bytes at {current_time}")
                        except Exception as e:
                            logging.error(f"Error in audio emotion processing: {str(e)}\n{traceback.format_exc()}")


                    # Asynchronously receive audio frames from the track and put them into the queue.
                    try:
                        while True:
                            try:
                                frame = await track.recv()
                                if frame is None:
                                    logging.info("Session %s: No more frames; ending audio capture", session_id)
                                    break
                                audio_data = frame.to_ndarray()
                                audio_bytes = audio_data.tobytes()
                                audio_queue.put(audio_bytes)

                                if session_state.hume_ws.is_capturing_emotion:
                                    asyncio.create_task(
                                        handle_audio_data_for_emotion(session_state, audio_bytes)
                                    )
                            except MediaStreamError:
                                # This is expected when client disconnects
                                logging.info("Session %s: Client disconnected (MediaStreamError)", session_id)
                                break
                            except Exception as e:
                                # This catches other unexpected errors
                                print("Session %s: Error reading audio track: %s\n%s", 
                                            session_id, e, traceback.format_exc())
                                break
                    finally:
                        # Signal the sender thread to finish and wait for it to join.
                        audio_queue.put(None)
                        sender_thread.join()


                        
                async def handle_video_track(track, session_id, session_state):
                    """Handle video track for emotion analysis"""
                    logging.info(f"Session {session_id}: Video track detected")
                    
                    # Store track reference
                    session_state.video_track = track
                    
                    try:
                        # Process video frames
                        while True:
                            try:
                                frame = await track.recv()
                                
                                # If we're capturing emotion
                                if session_state.hume_ws.is_capturing_emotion and session_state.hume_ws and session_state.hume_ws.connected:
                                    # Capture face periodically
                                    current_time = time.time()
                                    if current_time - session_state.last_face_capture >= session_state.face_capture_interval:
                                        session_state.last_face_capture = current_time

                                        img = frame.to_ndarray()
                                        asyncio.create_task(session_state.hume_ws.send_face_image(img))
                                
                            except MediaStreamError:
                                logging.info(f"Session {session_id}: Video track ended")
                                break
                    
                    except Exception as e:
                        logging.error(f"Session {session_id}: Error in video track handler: {str(e)}")
                    
                    finally:
                        session_state.video_track = None

                # Handle incoming tracks
                @pc.on("track")
                async def on_track(track):
                    # Initialize Hume WebSocket if not already done
                    if session_state.hume_ws and not session_state.hume_ws.connected:
                        await session_state.hume_ws.connect()

                    if track.kind == "audio":
                        asyncio.create_task(handle_audio_track(track, session_id, session_state, websocket))
                    elif track.kind == "video":
                        asyncio.create_task(handle_video_track(track, session_id, session_state))

                proactive_text = random.choice(empathetic_greetings)
                print('Proactive message for phone call: ', proactive_text)
                await websocket.send_json({
                    "type": "assistant_voice_message",
                    "text": proactive_text
                })
                history = conversation_histories[session_id]
                if not history:
                    conversation_histories[session_id] = {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                    history = conversation_histories[session_id]
                history.append({"role": "assistant", "content": proactive_text})
                if user:
                    save_call_event_to_firestore(user.get("uid"), session_id, "started")
                    await websocket.send_json({
                        "type": "CONV_START",
                        "timestamp": dt.datetime.now().timestamp()
                    })
                await stream_tts_to_webrtc(pc, proactive_text, session_id, websocket)
                await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="offer"))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_json({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

            elif data["type"] == "ice-candidate":
                pc = peer_connections.get(session_id)
                if pc and pc.iceConnectionState not in ["closed", "failed"]:
                    candidate_dict = data["candidate"]
                    candidate = candidate_from_sdp(candidate_dict["candidate"])
                    candidate.sdpMid = candidate_dict.get("sdpMid")
                    candidate.sdpMLineIndex = candidate_dict.get("sdpMLineIndex")
                    await pc.addIceCandidate(candidate)
            elif data["type"] == "CHAT_MESSAGE":
                if user:
                    # handle the message of greetings that was not saved
                    greeting_data = session_greeting_data.get(session_id)
                    is_first_message = greeting_data and not greeting_data["saved"]

                    if is_first_message:
                        # Save the initial greeting with its original timestamp
                        save_conversation_to_firestore_with_timestamp(
                            user.get("uid"), 
                            session_id, 
                            {"role": "assistant", "content": greeting_data["message"]},
                            greeting_data["timestamp"]
                        )
                        
                        # Mark that we've saved the greeting message
                        session_greeting_data[session_id]["saved"] = True

                    save_conversation_to_firestore(user.get("uid"), session_id, 
                                                {"role": "user", "content": data["message"]})

                assistant_message = await process_message(None, data["message"], session_id, user, websocket, True)

                if user:
                    save_conversation_to_firestore(user.get("uid"), session_id, 
                                                {"role": "assistant", "content": assistant_message})
    
                chat_history = chat_histories[session_id]
                chat_history.append({"role": "assistant", "content": assistant_message})

                await websocket.send_json({
                    "type": "CHAT_MESSAGE",
                    "message": assistant_message
                })

            elif data["type"] == "rtc_disconnect":
                session_id = data.get("sessionId")

                if user:
                    memory_enabled = get_memory_enabled(user['uid'])

                    if memory_enabled:
                        loop = asyncio.get_running_loop()
                        loop.run_in_executor(
                            executor, 
                            lambda: asyncio.run(finalize_conversation(copy.deepcopy(conversation_histories[session_id]), user['uid']))
                        )

                # Close the peer connection if it exists
                if session_id in peer_connections:
                    pc = peer_connections[session_id]
                    await pc.close()
                    del peer_connections[session_id]
                    print(f"Closed WebRTC connection for session {session_id}")
                
                # Clean up audio-related session state but keep other session data
                if session_id in session_states:
                    session_state = session_states[session_id]
                    
                    # Clean up audio resources
                    if hasattr(session_state, "audio_track") and session_state.audio_track:
                        session_state.audio_track = None

                    if hasattr(session_state, "hume_ws") and session_state.hume_ws:
                        await session_state.hume_ws.disconnect()
                        print(f"Disconnected from Hume WebSocket for session {session_id}")
                        
                    # Clear PC reference
                    session_state.pc = None
                    
                    print(f"Cleaned up WebRTC resources for session {session_id}")
                
                if conversation_histories[session_id]:
                    conversation_histories[session_id] = [{
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }]

                if user:
                    save_call_event_to_firestore(user.get("uid"), session_id, "ended")
                    await websocket.send_json({
                        "type": "CONV_END",
                        "timestamp": dt.datetime.now().timestamp()
                    })

                # Acknowledge the disconnect
                await websocket.send_json({
                    "type": "rtc_disconnected",
                    "message": "WebRTC connection closed"
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected normally for session {session_id}")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.format_exc()
        
        print(f"WebSocket ERROR - Type: {error_type}, Message: {error_msg}")
        print(f"Session ID at time of error: {session_id}")
        print(f"Full traceback:\n{tb}")
    finally:
        print(f"WebSocket connection closed for session {session_id}")

        if session_id in peer_connections:
            await peer_connections[session_id].close()
            del peer_connections[session_id]

        try:
            memory_enabled = get_memory_enabled(user['uid'])
        except Exception as e:
            memory_enabled = False
        
        if session_id in conversation_histories and len(conversation_histories[session_id]) > 3 and memory_enabled:
            print(f"Finalizing conversation for session {session_id} due to WebSocket disconnect for conversation")
            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                executor, 
                lambda: asyncio.run(finalize_conversation(copy.deepcopy(conversation_histories[session_id]), user['uid']))
            )

        if session_id in chat_histories and len(chat_histories[session_id]) > 3 and memory_enabled:
            print(f"Finalizing conversation for session {session_id} due to WebSocket disconnect for chat")
            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                executor,
                lambda: asyncio.run(finalize_conversation(copy.deepcopy(chat_histories[session_id]), user['uid']))
            )
            
        if session_id in session_states:
            session_state = session_states[session_id]
            if hasattr(session_state, "hume_ws") and session_state.hume_ws:
                await session_state.hume_ws.disconnect()
        
    # except Exception as e:
    #     logger.error(f"WebSocket error in session {session_id}: {str(e)}", exc_info=True)
    #     await websocket.close(code=1011, reason=f"Server error: {str(e)}")
    # finally:
    #     if session_id in peer_connections:
    #         await peer_connections[session_id].close()
    #         del peer_connections[session_id]
    #         logger.info(f"Closed peer connection for session {session_id}")

# --- TTS streaming to WebRTC ---
import queue

async def stream_tts_to_webrtc(pc, text, session_id, websocket):
    session_state = session_states.get(session_id)
    if not session_state:
        session_state = SessionState()
        session_states[session_id] = session_state

    new_tts_event = uuid.uuid4()
    session_state.current_tts_event = new_tts_event
    
    # Check if an existing audio track is available
    if not hasattr(session_state, "audio_track") or session_state.audio_track is None:
        sync_audio_queue = queue.Queue()
        audio_track = PCM24kAudioTrack(sample_rate=24000, frame_duration_ms=20)
        pc.addTrack(audio_track)
        session_state.audio_track = audio_track
        session_state.sync_audio_queue = sync_audio_queue
    else:
        sync_audio_queue = session_state.sync_audio_queue
        audio_track = session_state.audio_track

    # Create a future that will be set when audio processing is done
    processing_complete = asyncio.Future()
    
    # async def monitor_frame_queue():
    #     # Give some time for audio to be processed and queued
    #     await asyncio.sleep(0.5)
        
    #     while not processing_complete.done():
    #         current_queue_size = session_state.audio_track._frame_queue.qsize()
    #         if current_queue_size == 0:
    #             # Wait a bit to ensure no more frames are coming
    #             await asyncio.sleep(0.5)
    #             if session_state.audio_track._frame_queue.qsize() == 0:
    #                 print('Audio processing complete, notifying client')
    #                 session_state.processing_event.clear()
    #                 await websocket.send_json({
    #                     "type": "FINISHED_PROCESSING"
    #                 })
    #                 processing_complete.set_result(True)
    #                 break
    #         await asyncio.sleep(0.2)
    
    async def fill_audio_queue(audio_track, text, my_event_id):
        loop = asyncio.get_running_loop()
        def blocking_tts_task():
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="onyx",
                input=text,
                response_format="pcm"
            ) as response:
                for chunk in response.iter_bytes():
                    if session_state.current_tts_event != my_event_id:
                        break
                    audio_track.write_pcm(chunk)

        try:
            await loop.run_in_executor(executor, blocking_tts_task)
        except asyncio.CancelledError:
            # Perform additional cleanup if needed
            raise

    async def fill_task():
        # session_state.processing_event.set()
        # await websocket.send_json({
        #     "type": "PROCESSING"
        # })
        try:
            # Offload TTS to separate thread
            await fill_audio_queue(audio_track, text, new_tts_event)
        except Exception as e:
            print('something happened', e)
            processing_complete.set_exception(e)
    
    # Start both tasks
    session_state.fill_task = asyncio.create_task(fill_task())
    # monitor_task = asyncio.create_task(monitor_frame_queue())
    
    # We could wait here with await asyncio.gather(fill_task, monitor_task)
    # But for non-blocking operation, we'll let them run independently
    return processing_complete

# --- Generate proactive message ---
async def generate_proactive_message(user: Optional[dict]):
    if user:
        dummy_vector = [0.0] * 1024
        results = pinecone_index.query(
            vector=dummy_vector,
            top_k=5,
            filter={"user_id": {"$eq": user["uid"]}},
            namespace="user-memories",
            include_metadata=True
        )
        memories = [{"text": match["metadata"]["text"], "category": match["metadata"]["category"]} 
                    for match in results.get("matches", [])]
        memory_info = " ".join([m["text"] for m in memories]) if memories else ""
        greeting_prompt = (
            "You are Atlas, an empathetic AI psychologist. "
            "Based on your previous experiences and any available background, generate a lengthy proactive message (at least 4 sentences) that "
            "gives a warm greeting and suggests a topic of discussion or asks a probing question that invites the user to share more about themselves. "
            f"Here are some past conversations for reference: {memory_info}."
        )
    else:
        greeting_prompt = (
            "You are Atlas, an empathetic AI psychologist. "
            "Generate a brief, warm greeting that introduces yourself and invites the user to share."
        )

    proactive_response = client.chat.completions.create(
        model=model_version,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": greeting_prompt + "Make sure to generate a 3 sentence message."}
        ]
    )
    proactive_text = proactive_response.choices[0].message.content
    return proactive_text

def extract_json_from_markdown(text):
    import re
    if '```json' in text or '``` json' in text:
        pattern = r'```\s*json\s*(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
    return text.strip()

def clean_message_content(content):
    """
    Remove content between specific tag pairs:
    - <MEMORY_INJECTION> and <MEMORY_INJECTION_END>
    - <EMOTION-DETECTION> and <EMOTION-DETECTION-END>
    """
    # Clean memory injection content
    content = re.sub(r'<MEMORY_INJECTION>.*?<MEMORY_INJECTION_END>', '', content, flags=re.DOTALL)
    
    # Clean emotion detection content
    content = re.sub(r'<EMOTION-DETECTION>.*?<EMOTION-DETECTION-END>', '', content, flags=re.DOTALL)
    
    return content

async def finalize_conversation(
    conversation,
    userId
):
    print('calling finalize conv')
    namespace = "user-memories"
    filtered_messages = []
    for msg in conversation:
        # Skip the system prompt but keep other system messages
        if msg["role"] == "system" and msg["content"].strip() == SYSTEM_PROMPT:
            continue
            
        # Copy the message
        cleaned_msg = msg.copy()
        
        # Only clean content for user messages
        if msg["role"] == "user":
            cleaned_msg["content"] = clean_message_content(msg["content"])
            
        filtered_messages.append(cleaned_msg)

    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in filtered_messages])

    # First and foremost, determine if the conversation is worth storing
    worth_storing_prompt = (
        "Analyze the following conversation for a process that wants to extract psychoanalytic profile, user details and conversation summary.\n"
        "The conversation should be stored if:\n"
        "The conversation is meaningful beyond basic chit-chat, app usage, or questions about the application or the bot that personifies the application\n"
        "The conversation contains important user details that the user wants the psychologist to know such as name"
        "Return a JSON with the following structure:\n\n"
        """
        {
            "extraction": boolean,  # must be true or false
            "summarization": boolean  # must be true or false
        }
        """
        "\n\nConversation:\n\n" + conversation_text
    )
    
    worth_storing_response = client.chat.completions.create(
        model=model_version_extraction,
        messages=[
            {"role": "system", "content": "You are an expert at evaluating conversation quality and psychological value."},
            {"role": "user", "content": worth_storing_prompt}
        ]
    )

    worth_response = extract_json_from_markdown(
        worth_storing_response.choices[0].message.content.replace('True', 'true').replace('False', 'false'))
    
    try:
        parsed_response = json.loads(worth_response)
        shouldExtract = parsed_response['extraction']
        shouldSummarize = parsed_response['summarization']
    except Exception as e:
        print('Cannot parse JSON for deciding summarization', e)
        return
    
    print('Should extract from conv', shouldExtract)
    print('Should summarize from conv', shouldSummarize)

    if not shouldExtract and not shouldSummarize:
        return
    
    if shouldExtract:
        extraction_prompt = (
            "Based on the following onversation"
            "extract key psychoanalytic about the user only. Focus on these areas: Psychological Profile, Family/Social Interactions, "
            "Emotional States, Cognitive Architecture and Experiences. For each category, output a JSON object with keys:\n"
            "- 'category': one of ['psychological_profile', 'family', 'emotional_state', 'cognitive_architecture', 'experiences']\n"
            "- 'text': a concise description of the insight.\n"
            "Format your output as a JSON array. Do not duplicate the information, if one insight is used for one category do not use it for others, it's fine to leave out categories.\n"
            "Do not include information about facial expressions assesments.\n\n"
            "Conversation:\n\n" + conversation_text
        )
        
        extraction_response = client.chat.completions.create(
            model=model_version_extraction,
            messages=[
                {"role": "system", "content": "You are an expert psychoanalyst extracting insights."},
                {"role": "user", "content": extraction_prompt}
            ]
        )
        extracted_insights_raw = extraction_response.choices[0].message.content
        print("Extracted insights raw output:", extracted_insights_raw)
        
        try:
            extracted_insights = extract_json(extracted_insights_raw)
        except Exception as e:
            return
        
        new_records = []
        DUPLICATE_THRESHOLD = 0.85

        for insight in extracted_insights:
            category = insight.get("category")
            text = insight.get("text")
            if not category or not text:
                continue

            record_id = str(uuid.uuid4())
            encrypted_text = encrypt_text(text)
            record = {
                "_id": record_id,
                "chunk_text": text,
                "text": encrypted_text,
                "user_id": userId,
                "timestamp": datetime.utcnow().isoformat(),
                "category": category,
                "tags": [category],
                "is_encrypted": True
            }

            search_results = pinecone_index.search_records(
                namespace=namespace,
                query={
                    "inputs": {"text": text},
                    "top_k": 5,  # Increased to find up to 5 potential matches
                    "filter": {"category": {"$eq": category}, "user_id": {"$eq": userId}}
                },
                fields=["text", "category", "score", "is_encrypted"]
            )

            duplicate_found = False
            hits = search_results.get("result", {}).get("hits", [])
            
            # Check all hits, not just the first one
            for hit in hits:
                similarity_score = hit.get("score", 0)
                fields = hit.get("fields", {})
                
                # Decrypt the stored text for comparison if it's encrypted
                stored_text = fields.get("text", "")
                if fields.get("is_encrypted", False):
                    stored_text = decrypt_text(stored_text)
                    
                # Now compare with the original, unencrypted text
                if similarity_score >= DUPLICATE_THRESHOLD:
                    duplicate_found = True
                    print(f"Duplicate memory found for category {category} with similarity {similarity_score}: skipping record.")
                    break  # Exit the loop once a duplicate is found

            if not duplicate_found:
                new_records.append(record)
            else:
                print(f"Duplicate memory found for category {category} with similarity {similarity_score}: skipping record.")
        
        if new_records:
            pinecone_index.upsert_records(namespace, new_records)
    
    if shouldSummarize:
        summary_prompt = (
            "Summarize the following conversation briefly, focusing on key insights and useful context:\n\n" +
            conversation_text
        )
        summary_response = client.chat.completions.create(
            model=model_version,
            messages=[
                {"role": "system", "content": "You are an expert summarizer. Do not include information about facial expressions assesments."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        summary_text = summary_response.choices[0].message.content
        print("Session summary:", summary_text)
        
        record_summary = {
            "_id": str(uuid.uuid4()),
            "chunk_text": summary_text,
            "text": encrypt_text(summary_text),
            "user_id": userId,
            "timestamp": datetime.utcnow().isoformat(),
            "category": "conversation_summary",
            "tags": ["summary"],
            "is_encrypted": True
        }
        pinecone_index.upsert_records(namespace, [record_summary])

@app.get("/new_session")
async def new_session(model_version: str = Query("ft:gpt-4o-mini-2024-07-18:personal::BANPHZFe")):
    session_id = str(uuid.uuid4())
    conversation_histories[session_id] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    chat_histories[session_id] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    session_states[session_id] = SessionState(model_version=model_version)

    print('chat histories keys in new_session is: ', chat_histories.keys())
    return {"session_id": session_id}

@app.post("/clear_memories")
async def clear_memories(user: dict = Depends(verify_token)):
    """Clear all memories for a user from Pinecone."""
    user_id = user.get("uid")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    try:
        # First, find all records for this user
        dummy_vector = [0.0] * 1024  # Adjust vector dimension to match your index
        
        # Query to find all user records
        results = pinecone_index.query(
            vector=dummy_vector,
            top_k=10000,  # Set a high number to get all records
            filter={"user_id": {"$eq": user_id}},
            namespace="user-memories",
            include_metadata=False  # We only need IDs
        )
        
        # Extract IDs of user's records
        ids_to_delete = [match["id"] for match in results.get("matches", [])]
        
        if not ids_to_delete:
            return {"message": "No memories found to clear"}
        
        # Delete in batches if there are many records
        batch_size = 100
        for i in range(0, len(ids_to_delete), batch_size):
            batch = ids_to_delete[i:i + batch_size]
            delete_response = pinecone_index.delete(
                ids=batch,
                namespace="user-memories"
            )
        
        total_deleted = len(ids_to_delete)
        return {"message": f"Successfully cleared {total_deleted} memories"}
    except Exception as e:
        print(f"Error clearing memories: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing memories: {str(e)}")

@app.post("/clear_chat")
async def clear_chat(user: dict = Depends(verify_token)):
    """Clear all chat messages for a user from Firestore."""
    user_id = user.get("uid")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    try:
        # Get a reference to the user's messages collection
        user_ref = db.collection('users').document(user_id)
        messages_ref = user_ref.collection('messages')
        
        # Delete all documents in the collection
        docs = messages_ref.stream()
        for doc in docs:
            doc.reference.delete()
        
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail="Error clearing chat history")

@app.post("/delete_memory")
async def delete_memory(
    request: Request,
    user: dict = Depends(verify_token)
):
    """Delete a specific memory for a user."""
    data = await request.json()
    user_id = user.get("uid")
    text = data.get("text")
    category = data.get("category")
    
    if not user_id or not text:
        raise HTTPException(status_code=400, detail="User ID and memory text are required")
    
    try:
        # First, find the records that match our criteria
        # Create a dummy vector for search (since we're only interested in metadata matching)
        dummy_vector = [0.0] * 1024  # Adjust vector dimension to match your index
        
        # Build search filter
        search_filter = {
            "user_id": {"$eq": user_id},
            "text": {"$eq": text}
        }
        
        if category:
            search_filter["category"] = {"$eq": category}
        
        # Query to find matching records
        results = pinecone_index.query(
            vector=dummy_vector,
            top_k=100,  # Adjust based on potential matches
            filter=search_filter,
            namespace="user-memories",
            include_metadata=True
        )
        
        # Extract IDs of matching records
        ids_to_delete = [match["id"] for match in results.get("matches", [])]
        
        if not ids_to_delete:
            return {"message": "No matching memories found"}
        
        # Delete records by ID
        delete_response = pinecone_index.delete(
            ids=ids_to_delete,
            namespace="user-memories"
        )
        
        return {"message": f"Successfully deleted {len(ids_to_delete)} memories"}
    except Exception as e:
        print(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {str(e)}")

@app.post("/delete_message")
async def delete_message(
    request: Request,
    user: dict = Depends(verify_token)
):
    """Delete a specific chat message for a user."""
    data = await request.json()
    user_id = user.get("uid")
    message_id = data.get("message_id")
    
    if not user_id or not message_id:
        raise HTTPException(status_code=400, detail="User ID and message ID are required")
    
    try:
        # Delete the specific message document
        user_ref = db.collection('users').document(user_id)
        message_ref = user_ref.collection('messages').document(message_id)
        message_ref.delete()
        
        return {"message": "Chat message deleted successfully"}
    except Exception as e:
        print(f"Error deleting chat message: {e}")
        raise HTTPException(status_code=500, detail="Error deleting chat message")

@app.get("/user_preferences")
async def get_user_preferences(user: dict = Depends(verify_token)):
    """Get user preferences for memory and chat storage."""
    user_id = user.get("uid")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    try:
        # Get user preferences document
        user_ref = db.collection('users').document(user_id)
        prefs_ref = user_ref.collection('preferences').document('app_settings')
        prefs_doc = prefs_ref.get()
        
        if prefs_doc.exists:
            prefs = prefs_doc.to_dict()
        else:
            # Default preferences
            prefs = {
                "memory_enabled": True,
                "chat_enabled": True
            }
            # Create the document with default values
            prefs_ref.set(prefs)
        
        return prefs
    except Exception as e:
        print(f"Error getting user preferences: {e}")
        raise HTTPException(status_code=500, detail="Error getting user preferences")

@app.post("/update_preferences")
async def update_preferences(
    request: Request,
    user: dict = Depends(verify_token)
):
    """Update user preferences for memory and chat storage."""
    data = await request.json()
    user_id = user.get("uid")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    try:
        # Get user preferences document
        user_ref = db.collection('users').document(user_id)
        prefs_ref = user_ref.collection('preferences').document('app_settings')
        
        # Update only the provided preferences
        update_data = {}
        if "memory_enabled" in data:
            update_data["memory_enabled"] = data["memory_enabled"]
        if "chat_enabled" in data:
            update_data["chat_enabled"] = data["chat_enabled"]
        
        if update_data:
            prefs_ref.set(update_data, merge=True)
        
        return {"message": "Preferences updated successfully"}
    except Exception as e:
        print(f"Error updating user preferences: {e}")
        raise HTTPException(status_code=500, detail="Error updating user preferences")

def format_emotion_data_for_llm(emotion_data):
    """Format emotion data for inclusion in LLM prompt"""
    
    formatted_text = "USER_EMOTION_ANALYSIS:\n"
    
    # Format facial emotions if available
    if emotion_data.get("facial"):
        emotions = emotion_data["facial"]
        formatted_text += "Facial expressions: "
        for emotion in emotions:
            facial_emotions = [f"{e['name']} ({e['score']:.2f})" for e in emotion]
            formatted_text += ", ".join(facial_emotions)
            formatted_text += "\n"
    
    # Format vocal emotions if available
    if emotion_data.get("vocal"):
        emotions = emotion_data["vocal"]
        formatted_text += "Voice emotions: "
        for emotion in emotions:
            vocal_emotions = [f"{e['name']} ({e['score']:.2f})" for e in emotion]
            formatted_text += ", ".join(vocal_emotions)
            formatted_text += "\n"
    
    return formatted_text

def clean_message_content(content):
    """
    Remove content between specific tag pairs:
    - <MEMORY_INJECTION> and <MEMORY_INJECTION_END>
    - <EMOTION-DETECTION> and <EMOTION-DETECTION-END>
    """
    # Clean memory injection content
    content = re.sub(r'<MEMORY_INJECTION>.*?<MEMORY_INJECTION_END>', '', content, flags=re.DOTALL)
    
    # Clean emotion detection content
    content = re.sub(r'<EMOTION-DETECTION>.*?<EMOTION-DETECTION-END>', '', content, flags=re.DOTALL)
    
    return content

async def process_message(
    pc,
    user_text,
    session_id,
    user,
    websocket,
    isChat=False
):
    try:
        session_state = session_states.get(session_id)
        user_prompt = ""
        if isChat:
            history = chat_histories[session_id]
        else:
            history = conversation_histories[session_id]
        if user:
            # Check if memory is enabled
            memory_enabled = get_memory_enabled(user.get("uid"))

            if memory_enabled:
                results = pinecone_index.search_records(
                    namespace="user-memories",
                    query={
                        "inputs": {"text": user_text},
                        "top_k": 5,
                        "filter": {"user_id": {"$eq": user["uid"]}}
                    },
                    fields=["text", "is_encrypted"]
                )
                memories = []
                result = results.get("result", {})
                for match in result.get("hits", []):
                    fields = match.get("fields", {})
                    if "text" in fields:
                        text = fields["text"]
                        if fields.get("is_encrypted", False):
                            text = decrypt_text(text)

                        category = fields.get("category", "Unknown Category")
                        memories.append("Category: " + category + "\n Memory: " + text)

                if memories:
                    retrieved_memories_text = (
                        "<MEMORY_INJECTION>: The following are memories retained about the user:\n" +
                        "\n".join(memories) +
                        "\nYou have the capacity to retain memory about the user, so act accordingly.<MEMORY_INJECTION_END>"
                    )
                    user_prompt += "You are given the following memories to support this conversation: \n" + retrieved_memories_text

        user_prompt += "\n Respond to the following user message: " + user_text

        if session_state and session_state.hume_ws:
            await session_state.hume_ws.wait_for_processing()

            emotion_data = session_state.hume_ws.get_emotion_data()
            
            # Only include if we have at least some emotion data
            if emotion_data.get("facial") or emotion_data.get("vocal"):
                emotion_context = format_emotion_data_for_llm(emotion_data)
                user_prompt += "\n <EMOTION-DETECTION>:\n You are given the following emotional data from current user turn, use it accordingly: \n" + emotion_context + "<EMOTION-DETECTION-END>"

        history.append({"role": "user", "content": user_prompt})
        model_ver = session_state.model_version if session_state else "ft:gpt-4o-mini-2024-07-18:personal::BANPHZFe"
        chat_response = client.chat.completions.create(
            model=model_ver,
            messages=history
        )
        assistant_text = chat_response.choices[0].message.content
        print('psychologist response is: ', assistant_text)
        await websocket.send_json({
            "type": "assistant_voice_message",
            "text": assistant_text
        })
        history.append({"role": "assistant", "content": assistant_text})
        
        if not isChat:
            await stream_tts_to_webrtc(pc, assistant_text, session_id, websocket)
        else:
            return assistant_text
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Unexpected error in process_message: {str(e)}\n{error_trace}")

@app.get("/retrieve_memories")
async def retrieve_memories(user: dict = Depends(verify_token)):
    dummy_vector = [0.0] * 1024
    results = pinecone_index.query(
        vector=dummy_vector,
        top_k=150,
        filter={"user_id": {"$eq": user["uid"]}},
        namespace="user-memories",
        include_metadata=True
    )

    memories = []
    for match in results.get("matches", []):
        text = match["metadata"]["text"]
        
        # Decrypt if the text is encrypted
        if match["metadata"].get("is_encrypted", False):
            text = decrypt_text(text)
            
        memories.append({
            "id": match["id"],
            "text": text,
            "category": match["metadata"]["category"],
            "timestamp": match["metadata"]["timestamp"]
        })
    
    memories.sort(key=lambda x: x["timestamp"], reverse=True)

    return JSONResponse(content={"memories": memories})

def get_user_conversations(user_id):
    """Retrieves all conversations for a given user ID."""
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    try:
        conversations_ref = db.collection("users")
        query_ref = conversations_ref.where("userId", "==", user_id)
        docs = query_ref.stream()
        conversations = [doc.to_dict() for doc in docs]
        return conversations
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail="Error fetching conversations")


@app.get("/user_conversations")
async def user_conversations_endpoint(user: dict = Depends(verify_token)):
    user_id = user['uid']

    user_ref = db.collection('users').document(user_id)
    messages_ref = user_ref.collection('messages')

    messages = messages_ref.order_by('timestamp', direction=firestore.Query.ASCENDING).stream()
    message_list = []
    
    for message in messages:
        message_data = message.to_dict()
        
        # Decrypt the content if it's encrypted
        if message_data.get("is_encrypted", False):
            message_data["content"] = decrypt_text(message_data["content"])
            
        message_list.append({"id": message.id, **message_data})

    return {"messages": message_list}


from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

def get_chat_enabled(userId):
    chat_enabled = True
    try:
        user_ref = db.collection('users').document(userId)
        prefs_ref = user_ref.collection('preferences').document('app_settings')
        prefs_doc = prefs_ref.get()
        
        if prefs_doc.exists:
            prefs = prefs_doc.to_dict()
            chat_enabled = prefs.get("chat_enabled", True)
    except Exception as e:
        print(f"Error checking chat preferences: {e}")

    return chat_enabled

def get_memory_enabled(userId):
    memory_enabled = True  # Default to enabled
    try:
        user_ref = db.collection('users').document(userId)
        prefs_ref = user_ref.collection('preferences').document('app_settings')
        prefs_doc = prefs_ref.get()
        
        if prefs_doc.exists:
            prefs = prefs_doc.to_dict()
            memory_enabled = prefs.get("memory_enabled", True)
    except Exception as e:
        print(f"Error checking memory preferences: {e}")

    return memory_enabled

# DB RELATED STUFF
def save_conversation_to_firestore_with_timestamp(user_id, session_id, message, timestamp, message_type="CHAT_MESSAGE"):
    """Save a conversation message to Firestore with provided timestamp."""
    if not user_id:
        return
    
    chat_enabled = get_chat_enabled(user_id)
    
    # Only save if chat storage is enabled
    if chat_enabled:
    
        # Create a reference to the conversation document
        conversation_ref = db.collection('users').document(user_id) \
                            .collection('messages')
        
        message_content = message.get("content", "")
        encrypted_content = encrypt_text(message_content)
        
        # Create message data
        message_data = {
            "type": message_type,
            "content": encrypted_content,
            "role": message.get("role", "system"),
            "timestamp": timestamp,
            "is_encrypted": True
        }
    
        # Add message to Firestore
        conversation_ref.add(message_data)

def save_conversation_to_firestore(user_id, session_id, message, message_type="CHAT_MESSAGE"):
    """Save a conversation message to Firestore."""
    if not user_id:
        return
    
    chat_enabled = get_chat_enabled(user_id)
    
    # Only save if chat storage is enabled
    if chat_enabled:
    
        timestamp = dt.datetime.now().timestamp()
        
        # Create a reference to the conversation document
        conversation_ref = db.collection('users').document(user_id) \
                            .collection('messages')
        
        message_content = message.get("content", "")
        encrypted_content = encrypt_text(message_content)
        
        # Create message data
        message_data = {
            "type": message_type,
            "content": encrypted_content,
            "role": message.get("role", "system"),
            "timestamp": timestamp,
            "is_encrypted": True
        }
        
        # Add message to Firestore
        conversation_ref.add(message_data)

def save_call_event_to_firestore(user_id, session_id, event_type="started"):
    """Save a call event to Firestore."""
    if not user_id:
        return

    timestamp = dt.datetime.now().timestamp()
    
    # Create a reference to the conversation document
    conversation_ref = db.collection('users').document(user_id) \
                         .collection('messages')
    
    # Create event data
    message_data = {
        "type": "CONVERSATION_EVENT",
        "content": f"Phone call {event_type}",
        "timestamp": timestamp
    }
    
    # Add event to Firestore
    conversation_ref.add(message_data)
