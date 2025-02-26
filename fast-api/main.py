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

import logging
logging.basicConfig(level=logging.DEBUG)


# WebRTC and media-related imports.
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from aiortc.contrib.media import MediaStreamTrack

import av
from fractions import Fraction

# --- Helper: Force OPUS in SDP ---
def prefer_opus(sdp: str) -> str:
    lines = sdp.splitlines()
    m_line_index = None
    opus_payload = None

    # Find the audio m= line and the payload for OPUS.
    for i, line in enumerate(lines):
        if line.startswith("m=audio"):
            m_line_index = i
        if "opus/48000" in line:
            parts = line.split()
            if parts and parts[0].startswith("a=rtpmap:"):
                try:
                    opus_payload = parts[0].split(":")[1]
                except Exception:
                    pass

    # If not found, return the SDP unchanged.
    if m_line_index is None or opus_payload is None:
        return sdp

    # Modify the m=audio line to only include the OPUS payload.
    m_line_parts = lines[m_line_index].split()
    new_m_line = m_line_parts[:3] + [opus_payload]
    lines[m_line_index] = " ".join(new_m_line)

    # Remove any a=rtpmap and a=fmtp lines for non-OPUS codecs.
    filtered_lines = []
    for line in lines:
        if line.startswith("a=rtpmap:") and "opus/48000" not in line:
            continue
        if line.startswith("a=fmtp:") and "opus/48000" not in line:
            continue
        filtered_lines.append(line)
    return "\r\n".join(filtered_lines) + "\r\n"

# --- Custom MediaStreamTrack for TTS audio ---
import subprocess
import threading
import queue
import numpy as np
import asyncio
from fractions import Fraction
from aiortc.contrib.media import MediaStreamTrack
from av import AudioFrame

class FFmpegAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sync_audio_queue):
        super().__init__()
        self.sync_audio_queue = sync_audio_queue  # A thread-safe queue (queue.Queue)
        self.frame_queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.frame_pts = 0
        self._ended = False
        self._start_time = None  # To track the real-time start of playback

        # Launch FFmpeg to decode MP3 (from stdin) to raw PCM (s16le, mono, 48000 Hz)
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "mp3",
                "-i", "pipe:0",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "48000",
                "pipe:1"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        # Start background threads to feed FFmpeg and read its output
        self._feed_thread = threading.Thread(target=self._feed_data, daemon=True)
        self._feed_thread.start()
        self._read_thread = threading.Thread(target=self._read_frames, daemon=True)
        self._read_thread.start()

    def _feed_data(self):
        while True:
            chunk = self.sync_audio_queue.get()
            if chunk is None:
                break
            try:
                self.ffmpeg.stdin.write(chunk)
                self.ffmpeg.stdin.flush()
            except Exception as e:
                print("Error writing to ffmpeg stdin:", e)
                break
        try:
            self.ffmpeg.stdin.close()
            print("Closed ffmpeg stdin")
        except Exception as e:
            print(f"Failed to close ffmpeg stdin: {e}")

    def _read_frames(self):
        samples_per_frame = 48000 // 50  # 960 samples for a ~20ms frame
        frame_size = samples_per_frame * 2  # 1920 bytes for mono s16

        with open("debug-decoded.pcm", "wb") as debug_pcm:
            while True:
                data = self.ffmpeg.stdout.read(frame_size)
                if not data:
                    print("End of FFmpeg output reached")
                    break
                if len(data) < frame_size:
                    print(f"Partial frame received: {len(data)} bytes, skipping")
                    continue

                debug_pcm.write(data)

                try:
                    samples = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
                    frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
                    frame.sample_rate = 48000
                    frame.pts = self.frame_pts
                    frame.time_base = Fraction(1, 48000)
                    self.frame_pts += frame.samples
                    self.loop.call_soon_threadsafe(self.frame_queue.put_nowait, frame)
                except Exception as e:
                    print("Error decoding frame:", e)

        self.ffmpeg.wait()
        print("FFmpeg process finished with return code:", self.ffmpeg.returncode)
        self.loop.call_soon_threadsafe(self.frame_queue.put_nowait, None)

    async def recv(self):
        """
        Deliver audio frames at the correct real-time intervals (~20ms per frame).
        """
        if self._start_time is None:
            self._start_time = self.loop.time()  # Set the start time on first call

        # Get the next frame from the queue
        frame = await self.frame_queue.get()
        if frame is None:
            # End of stream: return a silent frame and continue (or stop, if preferred)
            samples_per_frame = 48000 // 50
            silent_samples = np.zeros((1, samples_per_frame), dtype=np.int16)
            silent_frame = AudioFrame.from_ndarray(silent_samples, format="s16", layout="mono")
            silent_frame.sample_rate = 48000
            silent_frame.pts = self.frame_pts
            silent_frame.time_base = Fraction(1, 48000)
            self.frame_pts += silent_frame.samples
            return silent_frame

        # Calculate the expected real-time delivery point for this frame
        frame_duration = frame.samples / 48000  # Duration in seconds (~0.02s for 960 samples)
        expected_time = self._start_time + (frame.pts / 48000)  # When this frame should be delivered

        # Current time in the event loop
        current_time = self.loop.time()

        # If we're ahead of schedule, wait until the correct time
        if current_time < expected_time:
            await asyncio.sleep(expected_time - current_time)

        return frame

    def stop(self):
        """Clean up resources when stopping the track."""
        self._ended = True
        if self.ffmpeg.poll() is None:
            self.ffmpeg.terminate()
        super().stop()

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

CURRENT_EXPERIENCE = """
You had a previous client that suffered a breakdown, it was hard to watch it but you understood a lot about the human psychic. You should not talk about this, only if it makes sense based on the user feedback.
You are subtly self-aware—if relevant, you may briefly hint at your mood (for example, "I feel a bit reflective today"), but avoid long internal explanations.
"""

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

Here is today current background: {CURRENT_EXPERIENCE}
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


# --- WebSocket endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            session_id = data.get("sessionId")
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

                token = data.get("token")
                user = await get_optional_user(token)
                proactive_text = await generate_proactive_message(session_id, user)
                await stream_tts_to_webrtc(pc, proactive_text, session_id)

                @pc.on("icecandidate")
                def on_icecandidate(candidate):
                    print('ice cndidate')
                    asyncio.ensure_future(websocket.send_json({
                        "type": "ice-candidate",
                        "candidate": candidate.to_json(),
                        "sessionId": session_id
                    }))

                await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="offer"))
                answer = await pc.createAnswer()
                # Force OPUS as the only audio codec in the SDP.
                modified_sdp = prefer_opus(answer.sdp)
                await pc.setLocalDescription(RTCSessionDescription(sdp=modified_sdp, type="answer"))
                #print("Server SDP:\n", pc.localDescription.sdp)
                await websocket.send_json({"type": "answer", "sdp": modified_sdp, "sessionId": session_id})

            elif data["type"] == "ice-candidate":
                pc = peer_connections.get(session_id)
                if pc and pc.iceConnectionState not in ["closed", "failed"]:
                    candidate_dict = data["candidate"]
                    candidate = candidate_from_sdp(candidate_dict["candidate"])
                    candidate.sdpMid = candidate_dict.get("sdpMid")
                    candidate.sdpMLineIndex = candidate_dict.get("sdpMLineIndex")
                    await pc.addIceCandidate(candidate)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if session_id in peer_connections:
            await peer_connections[session_id].close()
            del peer_connections[session_id]

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("Connection open")
#     try:
#         while True:
#             message = await websocket.receive_text()  # Keeps the connection alive
#             print(f"Received: {message}")
#             await websocket.send_text(f"Echo: {message}")
#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print(f"WebSocket error: {e}")
#     finally:
#         print("Connection closed")
#         await websocket.close()

# --- TTS streaming to WebRTC ---
import queue

async def stream_tts_to_webrtc(pc, text, session_id):
    # Create a thread-safe synchronous queue for audio chunks.
    sync_audio_queue = queue.Queue()

    async def fill_audio_queue():
        try:
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="onyx",
                input=text
            ) as response:
                for chunk in response.iter_bytes():
                    sync_audio_queue.put(chunk)
            sync_audio_queue.put(None)  # Signal end-of-stream.
        except Exception as e:
            print(f"Error in TTS fill task: {e}")
            sync_audio_queue.put(None)

    asyncio.create_task(fill_audio_queue())

    # Create the FFmpeg-based track and add it to the peer connection.
    audio_track = FFmpegAudioTrack(sync_audio_queue)
    pc.addTrack(audio_track)

# --- Generate proactive message ---
async def generate_proactive_message(session_id: str, user: Optional[dict]):
    history = conversation_histories[session_id]
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
    print('proactive text: ', proactive_text)
    history.append({"role": "assistant", "content": proactive_text})
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
    return {"session_id": session_id}

@app.post("/stop_playing")
async def stop_playing():
    stop_event.set()
    return {"message": "Playback stopping..."}

@app.post("/process_audio")
async def process_audio(
    file: UploadFile = File(...),
    tts: bool = False,
    session_id: str = Query(...),
    user: dict = Depends(get_optional_user)
):
    file_extension = file.filename.split(".")[-1]
    temp_file_name = f"{uuid.uuid4()}.{file_extension}"
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, temp_file_name)

    stop_event.clear()

    async with aiofiles.open(temp_file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        with open(temp_file_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        user_text = transcript_response.text

        history = conversation_histories[session_id]
        print('current history', history)

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
        history.append({"role": "assistant", "content": assistant_text})
        if tts:
            def audio_stream():
                with client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="onyx",
                    input=assistant_text
                ) as response:
                    for chunk in response.iter_bytes():
                        if stop_event.is_set():
                            break
                        yield chunk
            return StreamingResponse(audio_stream(), media_type="audio/mpeg")
        else:
            return JSONResponse(
                content={
                    "transcribed_text": user_text,
                    "assistant_text": assistant_text
                }
            )
    except Exception as e:
        print('Error is: ', e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_file_path)

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

@app.post("/proactive_message")
async def proactive_message(
    session_id: str = Query(...),
    user: dict = Depends(get_optional_user)
):
    if session_id not in conversation_histories:
        return JSONResponse(content={"message": "Session not found."}, status_code=404)
    
    history = conversation_histories[session_id]
    if user is not None:
        dummy_vector = [0.0] * 1024
        results = pinecone_index.query(
            vector=dummy_vector,
            query="user general information, user name, general emotions",
            top_k=5,
            filter={"user_id": {"$eq": user["uid"]}},
            namespace="user-memories",
            include_metadata=True
        )
        memories = [{
            "text": match["metadata"]["text"],
            "category": match["metadata"]["category"]
        } for match in results.get("matches", [])]
        memory_info = " ".join([m["text"] for m in memories]) if memories else ""
        greeting_prompt = (
            "You are Atlas, an empathetic AI psychologist. "
            "Based on your previous experiences and any available background, generate a proactive message that "
            "gives a warm greeting and suggests a topic of discussion or asks a probing question that invites the user to share more about themselves. "
            "If no specific background is available, simply ask what brings you here. "
            f"Here are some past conversations for reference (in case you might want to use them): {memory_info}. \n Only respond in English."
        )
    else:
        greeting_prompt = (
            "You are Atlas, an empathetic AI psychologist. "
            "Generate a brief, warm greeting that introduces yourself and invites the user to share."
        )

    proactive_input = greeting_prompt
    proactive_response = client.chat.completions.create(
         model=model_version,
         messages=[
             {"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": proactive_input}
         ]
    )
    proactive_text = proactive_response.choices[0].message.content
    history.append({"role": "assistant", "content": proactive_text})
    print(proactive_text)
    time.sleep(10)
    def audio_stream():
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="onyx",
            input=proactive_text
        ) as response:
            for chunk in response.iter_bytes():
                yield chunk

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")

from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
