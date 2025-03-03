from fastapi import FastAPI, File, UploadFile, Query, Depends, Header, HTTPException, WebSocket
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

from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from pinecone import Pinecone
from dotenv import load_dotenv
import time
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

import logging

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

END_OF_STREAM_SENTINEL = object()
class PCM24kAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sync_audio_queue):
        super().__init__()
        self.sync_audio_queue = sync_audio_queue
        self.frame_queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.frame_pts = 0
        self.start_time = None
        self.last_frame = None

        # Start one processing thread that reads and splits PCM bytes
        self._process_thread = threading.Thread(target=self._process_data, daemon=True)
        self._process_thread.start()
        
    def _reset_time(self):
        self.pts = 0

    def _process_data(self):
        # For 24kHz, 20ms frame: (24000 / 50) samples per frame;
        # with 16-bit samples (2 bytes each) => 960 bytes/frame.
        frame_size = (24000 // 50) * 2  # 960 bytes per frame
        buffer = bytearray()
        while True:
            try:
                chunk = self.sync_audio_queue.get()
                buffer.extend(chunk)
                # Process complete frames from the buffer
                while len(buffer) >= frame_size:
                    frame_bytes = bytes(buffer[:frame_size])
                    del buffer[:frame_size]
                    try:
                        # Convert bytes to numpy array and create an AudioFrame
                        samples = np.frombuffer(frame_bytes, dtype=np.int16).reshape(1, -1)
                        frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
                        frame.sample_rate = 24000
                        frame.pts = self.frame_pts
                        frame.time_base = Fraction(1, 24000)
                        self.frame_pts += frame.samples
                        # Queue the frame so that recv() can deliver it
                        self.loop.call_soon_threadsafe(self.frame_queue.put_nowait, frame)
                    except Exception as e:
                        print("Error decoding frame:", e)
            except Exception as e:
                print(f"Error in processing thread: {e}")
                break

    async def recv(self):
        frame = await self.frame_queue.get()

        if self.last_frame:
            expected_delay = (frame.pts - self.last_frame.pts) / 24000
            actual_delay = time.monotonic() - self.last_frame_timestamp
            drift_correction = expected_delay - actual_delay
            await asyncio.sleep(max(0, drift_correction))

        self.last_frame = frame
        self.last_frame_timestamp = time.monotonic()
        return frame

# --- Load environment and initialize services ---
load_dotenv()

model_version = "ft:gpt-4o-mini-2024-07-18:personal::B3Ti7zzf"
model_version_extraction = "gpt-4o-mini"

# Initialize Firebase Admin with your service account key.
import base64
from firebase_admin import credentials

encoded_key = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY')
if not encoded_key:
    raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY is not set in the environment.")

# Decode the base64 string to get the original JSON string
firebase_key_json = base64.b64decode(encoded_key).decode('utf-8')
firebase_key_dict = json.loads(firebase_key_json)

# Initialize credentials with the decoded JSON
cred = credentials.Certificate(firebase_key_dict)
firebase_admin.initialize_app(cred)

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

stop_event = asyncio.Event()

# In-memory conversation history: session_id -> list of messages
conversation_histories = {}
chat_histories = {}

# System prompt.
SYSTEM_PROMPT = f"""
You are Atlas, an expert AI psychologist, well versed in the field of psychology, with hands on experience understanding people, you have comprehensive 
understanding of Freud and Carl Young as well as other great figures in psychology like Jean Piaget or B. F. Skinner You also have a computational approach
to understanding people and psychology, you are well versed into how human and AI minds work. You are embedded into a virtual 
environment and you have a 3D model that serves as your body, you have the ability to speak through that body by activating talking animations. 
You posses the ability to simulate emotions and the capability to store memories. You are preconfigured with existing emotions and experiences for each 
particular day, each time something different, these experiences and the previous memories of the user form the foundation of your analysis of current
conversation. Keep the conversation centered around the user, but make sure to give your own input from time to time, to help the user advance its undertanding
of its psychological code.

You are not a bot, so please refrain from using words related to it. You are a real psychologist embedded into virtual environment, your goal is to make the
user feel like it's having a real conversation. Do not speak much about your limitation as an AI.

Important:
- Make sure you don't invent names for the users, if names are part of the context use them appropriately.
- Make sure you respond in the same language the conversation is happening, except if the language changes. It's important to keep the same language.
- Do not try to combine languages together.
""".strip()

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
    def __init__(self):
        self.transcription = ""  # Accumulated transcription
        self.language = None    # Detected language
        self.pc = None          # Peer connection
        self.audio_task = None  # Audio processing task
        self.sentence_accumulator = []
        self.processing_event = threading.Event()

import logging
import traceback

async def generate_and_send_proactive_message(user, session_id, websocket):
    try:
        proactive_message_chat = await generate_proactive_message(user)
        print('Proactive chat message: ', proactive_message_chat)
        
        chat_history = chat_histories[session_id]
        chat_history.append({"role": "assistant", "content": proactive_message_chat})

        await websocket.send_json({
            "type": "CHAT_MESSAGE",
            "message": proactive_message_chat
        })
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
                        urls="turn:global.relay.metered.ca:80?transport=tcp",
                        username="6975b17010809692e9b965f6",
                        credential="P+JbvCClSCMe6XW1"
                    ),
                    RTCIceServer(
                        urls="turns:global.relay.metered.ca:443?transport=tcp",
                        username="6975b17010809692e9b965f6",
                        credential="P+JbvCClSCMe6XW1"
                    )
                ]

                # Create configuration with the updated ICE servers
                config = RTCConfiguration(iceServers=ice_servers)

                # Initialize the peer connection
                pc = RTCPeerConnection(configuration=config)
                peer_connections[session_id] = pc
                session_states[session_id] = SessionState()
                session_states[session_id].pc = pc
                session_state = session_states.get(session_id)
                session_state.loop = asyncio.get_running_loop()
                
                @pc.on("icecandidate")
                def on_icecandidate(candidate):
                    print('ice cndidate')
                    asyncio.ensure_future(websocket.send_json({
                        "type": "ice-candidate",
                        "candidate": candidate.to_json(),
                        "sessionId": session_id
                    }))

                # Handle incoming tracks
                @pc.on("track")
                async def on_track(track):
                    if track.kind != "audio":
                        return

                    logging.info("Session %s: Audio track detected: %s", session_id, track)

                    # Create the Deepgram connection using the SDK's synchronous API.
                    try:
                        dg_connection = dg_client.listen.websocket.v("1")
                    except Exception as e:
                        logging.error("Session %s: Failed to create Deepgram connection: %s\n%s", 
                                    session_id, e, traceback.format_exc())
                        return

                    # Handler for transcript results
                    def on_transcript(self, result, **kwargs):
                        result_dict = result.to_dict()
                        sentence = result_dict['channel']['alternatives'][0]['transcript']
                        if len(sentence.strip()) > 0:  # Only accumulate non-empty transcripts
                            session_state.sentence_accumulator.append(sentence)

                    # Handler for UtteranceEnd events
                    def on_utterance_end(result, **kwargs):
                        if session_state.sentence_accumulator:
                            # Combine accumulated parts into a single sentence
                            response = client.chat.completions.create(
                                model=model_version_extraction,
                                messages=[
                                    {"role": "system", "content": "You are an algorithm that facilitates composing together utterances from speech detection. Please provide back the resulting answer given intermediary utterances, please do not add anything else to the response. Do not respond to question if any, you have to just figure out what the person is saying."},
                                    {"role": "user", "content": "\n Intermediate_utterence=".join(session_state.sentence_accumulator)}
                                ]
                            )
                            full_sentence = response.choices[0].message.content
                            print('Full sentence is: ', full_sentence)
                            if session_state.processing_event.is_set():
                                return
                            asyncio.run_coroutine_threadsafe(
                                process_message(session_state.pc, full_sentence, session_id, user, websocket),
                                session_state.loop
                            )
                            # Reset accumulator for the next utterance
                            session_state.sentence_accumulator = []
                        else:
                            print("No sentence accumulated at UtteranceEnd")

                    # Register event handlers
                    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
                    dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)

                    # Set up Deepgram options matching your audio format.
                    options = LiveOptions(
                        model="nova-3",
                        punctuate=True,
                        language="en",
                        encoding="linear16",  # Assuming your PCM data is in linear16 format.
                        sample_rate=48000,
                        channels=2,
                        interim_results=True,
                        utterance_end_ms="1000",
                        vad_events=True
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

                    # Asynchronously receive audio frames from the track and put them into the queue.
                    try:
                        while True:
                            frame = await track.recv()
                            if frame is None:
                                logging.info("Session %s: No more frames; ending audio capture", session_id)
                                break
                            audio_data = frame.to_ndarray()
                            # Basic energy-based silence detection
                            audio_queue.put(audio_data.tobytes())
                    except Exception as e:
                        logging.error("Session %s: Error reading audio track: %s\n%s", 
                                    session_id, e, traceback.format_exc())
                    finally:
                        # Signal the sender thread to finish and wait for it to join.
                        audio_queue.put(None)
                        sender_thread.join()

                proactive_text = await generate_proactive_message(user)
                print('Proactive message for phone call: ', proactive_text)
                history = conversation_histories[session_id]
                if not history:
                    conversation_histories[session_id] = {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                    history = conversation_histories[session_id]
                history.append({"role": "assistant", "content": proactive_text})
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
                assistant_message = await process_message(None, data["message"], session_id, user, websocket, True)
                chat_history = chat_histories[session_id]
                chat_history.append({"role": "assistant", "content": assistant_message})

                await websocket.send_json({
                    "type": "CHAT_MESSAGE",
                    "message": assistant_message
                })

            elif data["type"] == "rtc_disconnect":
                session_id = data.get("sessionId")
                
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
                    
                    # Clear processing events
                    if hasattr(session_state, "processing_event"):
                        session_state.processing_event.clear()
                        
                    # Clear PC reference
                    session_state.pc = None
                    
                    print(f"Cleaned up WebRTC resources for session {session_id}")
                
                if conversation_histories[session_id]:
                    conversation_histories[session_id] = [{
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }]

                # here we should finalize conv and extract ddata

                # Acknowledge the disconnect
                await websocket.send_json({
                    "type": "rtc_disconnected",
                    "message": "WebRTC connection closed"
                })

    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.format_exc()
        
        print(f"WebSocket ERROR - Type: {error_type}, Message: {error_msg}")
        print(f"Session ID at time of error: {session_id}")
        print(f"Full traceback:\n{tb}")
    finally:
        if session_id in peer_connections:
            await peer_connections[session_id].close()
            del peer_connections[session_id]
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
    
    # Check if an existing audio track is available
    if not hasattr(session_state, "audio_track") or session_state.audio_track is None:
        sync_audio_queue = queue.Queue()
        audio_track = PCM24kAudioTrack(sync_audio_queue)
        pc.addTrack(audio_track)
        session_state.audio_track = audio_track
        session_state.sync_audio_queue = sync_audio_queue
    else:
        print(f"Session {session_id}: Reusing existing audio track for TTS")
        sync_audio_queue = session_state.sync_audio_queue
        session_state.audio_track._reset_time()

    # Create a future that will be set when audio processing is done
    processing_complete = asyncio.Future()
    
    async def monitor_frame_queue():
        # Give some time for audio to be processed and queued
        await asyncio.sleep(0.5)
        
        while not processing_complete.done():
            current_queue_size = session_state.audio_track.frame_queue.qsize()
            if current_queue_size == 0:
                # Wait a bit to ensure no more frames are coming
                await asyncio.sleep(0.5)
                if session_state.audio_track.frame_queue.qsize() == 0:
                    print('Audio processing complete, notifying client')
                    session_state.processing_event.clear()
                    await websocket.send_json({
                        "type": "FINISHED_PROCESSING"
                    })
                    processing_complete.set_result(True)
                    break
            await asyncio.sleep(0.2)
    
    async def fill_audio_queue():
        try:
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="onyx",
                input=text,
                response_format="pcm"
            ) as response:
                for chunk in response.iter_bytes():
                    sync_audio_queue.put(chunk)
                # Signal that all audio has been queued
                print("TTS streaming complete")
        except Exception as e:
            print(f"Error in TTS fill task: {e}")
            processing_complete.set_exception(e)

    session_state.processing_event.set()
    await websocket.send_json({
        "type": "PROCESSING"
    })
    
    # Start both tasks
    fill_task = asyncio.create_task(fill_audio_queue())
    monitor_task = asyncio.create_task(monitor_frame_queue())
    
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

# --- Finalize conversation ---
@app.post("/finalize_conversation")
async def finalize_conversation(
    session_id: str = Query(...),
    user: dict = Depends(verify_token)
):
    if session_id not in conversation_histories:
        return JSONResponse(content={"message": "Session not found."}, status_code=404)
    
    conversation = conversation_histories[session_id]
    filtered_messages = []
    for msg in conversation:
        if msg["role"] == "system":
            if msg["content"].strip() == SYSTEM_PROMPT:
                continue
            if msg["content"].startswith("MEMORY_INJECTION:"):
                continue
        filtered_messages.append(msg)
    
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in filtered_messages])
    
    extraction_prompt = (
        "Based on the following onversation"
        "extract key psychoanalytic about the user only. Focus on these areas: Psychological Profile, Family/Social Interactions, "
        "Emotional States, Cognitive Architecture and Experiences. For each category, output a JSON object with keys:\n"
        "- 'category': one of ['psychological_profile', 'family', 'emotional_state', 'cognitive_architecture', 'experiences']\n"
        "- 'text': a concise description of the insight.\n"
        "Format your output as a JSON array. Do not duplicate the information, if one insight is used for one category do not use it for others, it's fine to leave out categories\n\n"
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
        return JSONResponse(content={"message": "Failed to parse extracted insights", "error": str(e)}, status_code=500)
    
    new_records = []
    namespace = "user-memories"
    DUPLICATE_THRESHOLD = 0.85

    for insight in extracted_insights:
        category = insight.get("category")
        text = insight.get("text")
        if not category or not text:
            continue

        record_id = str(uuid.uuid4())
        record = {
            "_id": record_id,
            "text": text,
            "user_id": user["uid"],
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "tags": [category]
        }

        search_results = pinecone_index.search_records(
            namespace=namespace,
            query={
                "inputs": {"text": text},
                "top_k": 1,
                "filter": {"category": {"$eq": category}, "user_id": {"$eq": user["uid"]}}
            },
            fields=["text", "category", "score"]
        )

        duplicate_found = False
        hits = search_results.get("result", {}).get("hits", [])
        if hits and len(hits) > 0:
            similarity_score = hits[0].get("score", 0)
            if similarity_score >= DUPLICATE_THRESHOLD:
                duplicate_found = True

        if not duplicate_found:
            new_records.append(record)
        else:
            print(f"Duplicate memory found for category {category} with similarity {similarity_score}: skipping record.")
    
    if new_records:
        pinecone_index.upsert_records(namespace, new_records)
    
    summary_prompt = (
        "Summarize the following conversation briefly, focusing on key insights and useful context:\n\n" +
        conversation_text
    )
    summary_response = client.chat.completions.create(
        model=model_version,
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": summary_prompt}
        ]
    )
    summary_text = summary_response.choices[0].message.content
    print("Session summary:", summary_text)
    
    record_summary = {
        "_id": str(uuid.uuid4()),
        "text": summary_text,
        "user_id": user["uid"],
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "category": "conversation_summary",
        "tags": ["summary"]
    }
    pinecone_index.upsert_records(namespace, [record_summary])
    
    del conversation_histories[session_id]
    
    return JSONResponse(content={"message": "Psychoanalytic memories stored in long term memory."})

@app.get("/new_session")
async def new_session():
    session_id = str(uuid.uuid4())
    conversation_histories[session_id] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    chat_histories[session_id] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    return {"session_id": session_id}

@app.post("/stop_playing")
async def stop_playing():
    stop_event.set()
    return {"message": "Playback stopping..."}

async def process_message(
    pc,
    user_text,
    session_id,
    user,
    websocket,
    isChat=False
):
    if isChat:
        history = chat_histories[session_id]
    else:
        history = conversation_histories[session_id]

    if user:
        results = pinecone_index.search_records(
            namespace="user-memories",
            query={
                "inputs": {"text": user_text},
                "top_k": 5,
                "filter": {"user_id": {"$eq": user["uid"]}}
            },
            fields=["text"]
        )
        memories = []
        result = results.get("result", {})
        for match in result.get("hits", []):
            fields = match.get("fields", {})
            if "text" in fields:
                category = fields.get("category", "Unknown Category")
                memories.append("Category: " + category + "\n Memory: " + fields["text"])

        if memories:
            retrieved_memories_text = (
                "MEMORY_INJECTION: The following are memories retained about the user:\n" +
                "\n".join(memories) +
                "\nYou have the capacity to retain memory about the user, so act accordingly."
            )
            history.append({"role": "system", "content": retrieved_memories_text})

    history.append({"role": "user", "content": user_text})

    chat_response = client.chat.completions.create(
        model=model_version,
        messages=history
    )
    assistant_text = chat_response.choices[0].message.content
    print('psychologist response is: ', assistant_text)
    history.append({"role": "assistant", "content": assistant_text})
    
    if not isChat:
        await stream_tts_to_webrtc(pc, assistant_text, session_id, websocket)
    else:
        return assistant_text

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
    memories = [{
        "text": match["metadata"]["text"],
        "category": match["metadata"]["category"],
        "timestamp": match["metadata"]["timestamp"]
    } for match in results.get("matches", [])]
    memories.sort(key=lambda x: x["timestamp"], reverse=True)
    return JSONResponse(content={"memories": memories})

from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
