from fastapi import FastAPI, File, UploadFile, Query, Depends, Header, HTTPException
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

load_dotenv()

model_version = "ft:gpt-4o-mini-2024-07-18:personal::B3Ti7zzf"
model_version_extraction = "gpt-4o-mini"

# Initialize Firebase Admin with your service account key.
cred = credentials.Certificate("../serviceAccountKey.json")
firebase_admin.initialize_app(cred)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
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

# A less verbose, conversational system prompt.
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

async def get_optional_user(authorization: Optional[str] = Header(None)):
    """
    Attempt to decode a Firebase token if present. 
    If valid, return the decoded token; if invalid or missing, return None.
    """
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

# Initialize Pinecone.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("self")

@app.post("/finalize_conversation")
async def finalize_conversation(
    session_id: str = Query(...),
    user: dict = Depends(verify_token)
):
    # Ensure the session exists.
    if session_id not in conversation_histories:
        return JSONResponse(content={"message": "Session not found."}, status_code=404)
    
    conversation = conversation_histories[session_id]
    
    # --- Filter out messages that are not part of the "new session conversation" ---
    # Exclude the base SYSTEM_PROMPT and any memory injection messages.
    filtered_messages = []
    for msg in conversation:
        if msg["role"] == "system":
            # Skip the base system prompt.
            if msg["content"].strip() == SYSTEM_PROMPT:
                continue
            # Skip messages injected as long-term memory.
            if msg["content"].startswith("MEMORY_INJECTION:"):
                continue
        filtered_messages.append(msg)
    
    # Build the conversation text from filtered messages.
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in filtered_messages])
    
    # --- STEP 1: Extraction ---
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
    
    # --- STEP 2: Deduplication and Upsertion ---
    new_records = []
    namespace = "user-memories"
    DUPLICATE_THRESHOLD = 0.85  # Define an appropriate threshold

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

        # Deduplication: Query Pinecone for a similar memory in the same category.
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
            # Assume the hit includes a "score" field indicating similarity (higher is more similar)
            similarity_score = hits[0].get("score", 0)
            if similarity_score >= DUPLICATE_THRESHOLD:
                duplicate_found = True

        if not duplicate_found:
            new_records.append(record)
        else:
            print(f"Duplicate memory found for category {category} with similarity {similarity_score}: skipping record.")
    
    if new_records:
        pinecone_index.upsert_records(namespace, new_records)
    
    # Optionally, store a brief summary of the session.
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
    
    # Clear the in-memory conversation history after finalization.
    del conversation_histories[session_id]
    
    return JSONResponse(content={"message": "Psychoanalytic memories stored in long term memory."})

@app.get("/new_session")
async def new_session():
    """Generate a new session ID and initialize the conversation history."""
    session_id = str(uuid.uuid4())
    # Initialize with the base system prompt.
    conversation_histories[session_id] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    return {"session_id": session_id}

@app.post("/stop_playing")
async def stop_playing():
    """Signal to stop the currently playing TTS stream."""
    stop_event.set()
    return {"message": "Playback stopping..."}

@app.post("/process_audio")
async def process_audio(
    file: UploadFile = File(...),
    tts: bool = False,
    session_id: str = Query(...),
    user: dict = Depends(get_optional_user)  # may be None for unauthenticated users.
):
    """
    Receives an audio file, transcribes it, retrieves relevant long-term memories (if any),
    appends them to the conversation history, and generates a response.
    """
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

        # If the user is authenticated, retrieve long-term memories.
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
                # Mark injected memories with a special prefix so they can be filtered out later.
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
    """
    Retrieve all long-term memories for the current user using a dummy vector for metadata filtering.
    """
    dummy_vector = [0.0] * 1024
    results = pinecone_index.query(
        vector=dummy_vector,
        top_k=150,
        filter={"user_id": {"$eq": user["uid"]}},
        namespace="user-memories",
        include_metadata=True
    )

    # Build a list including the timestamp
    memories = [{
        "text": match["metadata"]["text"],
        "category": match["metadata"]["category"],
        "timestamp": match["metadata"]["timestamp"]  # ensure timestamp is in a sortable format
    } for match in results.get("matches", [])]

    # Sort memories from most recent to oldest
    memories.sort(key=lambda x: x["timestamp"], reverse=True)

    return JSONResponse(content={"memories": memories})

@app.post("/proactive_message")
async def proactive_message(
    session_id: str = Query(...),
    user: dict = Depends(get_optional_user)  # user may be None if not logged in
):
    # Ensure the session exists.
    if session_id not in conversation_histories:
        return JSONResponse(content={"message": "Session not found."}, status_code=404)
    
    history = conversation_histories[session_id]
    
    # Build a greeting prompt based on whether a user is logged in.
    if user is not None:
        # Use a dummy vector for metadata filtering as in retrieve_memories.
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

    # Combine the system prompt with the proactive greeting instruction.
    proactive_input = greeting_prompt
    proactive_response = client.chat.completions.create(
         model=model_version,
         messages=[
             {"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": proactive_input}
         ]
    )
    proactive_text = proactive_response.choices[0].message.content
    # Append the proactive text to the conversation history.
    history.append({"role": "assistant", "content": proactive_text})
    print(proactive_text)
    time.sleep(10)
    # Now stream TTS audio for the proactive message.
    def audio_stream():
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="onyx",
            input=proactive_text
        ) as response:
            for chunk in response.iter_bytes():
                yield chunk

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")
