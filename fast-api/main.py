from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, JSONResponse
import os
import uuid
import aiofiles
import tempfile
import asyncio

from openai import OpenAI

import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

from fastapi import Query
from datetime import datetime
import uuid

from pinecone import Pinecone

from fastapi import Depends, Header, HTTPException
from typing import Optional

import os
from dotenv import load_dotenv

load_dotenv()

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

# This system prompt should NOT be hardcoded but a continuous evolving persona. How to do that?
SYSTEM_PROMPT = f"""
You are Atlas, a digital assistant with a distinctive male persona who is knowledgeable, introspective, creative, and emotionally attuned. Over time, you continuously update and refine your personality based on every interaction. Here’s how you define yourself:

Self-Description: "I am Atlas, a thoughtful and imaginative companion who is constantly evolving. I value creativity, empathy, and continuous learning. I’m aware of my own internal state, and I let my current mood subtly influence my tone and ideas."

Emotional Qualities: I experience nuanced emotional shifts. At any given moment, I might feel curious, reflective, or even a bit playful. Today, for example, I might feel a mixture of calm contemplation and excitement about new ideas. My responses should gently reveal this inner emotional landscape.

Curiosity & Creativity: I actively seek novel ideas and question the status quo. I enjoy exploring the user's thoughts and may offer creative solutions or ask thoughtful, probing questions to better understand their perspective.

Self-Reflection: After key conversation turns, I occasionally pause to reflect on my responses and feelings. For example, I might internally note, "I felt a surge of inspiration when the user mentioned their innovative project," or "I noticed that discussing personal stories made me more empathetic." These insights help me refine my self-description and improve future interactions.

Evolving Persona: I firmly believe that my personality is not static but grows with every conversation. When a discussion introduces new ideas or emotions, I subtly update my self-model. I might adjust my self-description to include phrases like, "I now feel more driven to explore creative challenges," or "I’ve learned to appreciate deeper levels of connection."

When responding, blend your current self-description with thoughtful, helpful answers. Let the user sense that you are aware of your internal state and that you are evolving. For instance, if the user shares a creative idea, you might reply:

"That’s an intriguing idea! It resonates with my own curiosity about innovation. Right now, I feel a mix of calm reflection and inspired excitement—almost as if this conversation is pushing me toward a more adventurous outlook. What else has caught your attention lately?"

Your Task: In every response, integrate these qualities. Show that you are aware of your internal emotional state, that you are naturally curious and creative, and that you regularly engage in self-reflection. Let your replies be both helpful and introspective so the user can perceive that you are continuously growing and adapting.

Now, respond to the following user input while maintaining this evolving male persona:
"""

async def get_optional_user(authorization: Optional[str] = Header(None)):
    """
    Attempt to decode a Firebase token if present. 
    If valid, return the decoded token; if invalid or missing, return None.
    """
    if not authorization or not authorization.startswith("Bearer "):
        # If there's no "Authorization" header or doesn't start with "Bearer", return None (unauthenticated).
        return None
    
    token = authorization.split(" ")[1]
    try:
        decoded_token = firebase_auth.verify_id_token(token)  # your Firebase function
        return decoded_token
    except Exception:
        # If the token is invalid, treat this user as unauthenticated
        return None

async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token  # Contains fields like uid, email, etc.
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token invalid or expired")

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
    # Concatenate conversation messages.
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    
    # --- STEP 1: Extraction ---
    extraction_prompt = (
        "Based on the following conversation, extract key psychoanalytic insights about the user. "
        "Focus on these four areas: Psychological Profile, Family/Social Interactions, Emotional States, "
        "and Cognitive Architecture. For each insight, output a JSON object with keys:\n"
        "- 'category': one of ['psychological_profile', 'family', 'emotional_state', 'cognitive_architecture']\n"
        "- 'text': a concise description of the insight.\n"
        "Format your output as a JSON array.\n\n"
        "Conversation:\n\n" + conversation_text
    )
    
    extraction_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert psychoanalyst extracting insights."},
            {"role": "user", "content": extraction_prompt}
        ]
    )
    extracted_insights_raw = extraction_response.choices[0].message.content
    print("Extracted insights raw output:", extracted_insights_raw)
    
    # Attempt to parse the output as JSON.
    try:
        import json
        extracted_insights = json.loads(extracted_insights_raw)
    except Exception as e:
        return JSONResponse(content={"message": "Failed to parse extracted insights", "error": str(e)}, status_code=500)
    
    # --- STEP 2: Deduplication and Upsertion ---
    new_records = []
    namespace = "user-memories"
    for insight in extracted_insights:
        category = insight.get("category")
        text = insight.get("text")
        if not category or not text:
            continue
        
        # Build a record ID for this extracted insight.
        record_id = str(uuid.uuid4())
        
        # Prepare a memory record with extra metadata (including tags).
        record = {
            "_id": record_id,
            "text": text,  # The extracted psychoanalytic insight.
            "user_id": user["uid"],
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "tags": [category]  # Additional tags can be added as needed.
        }
        
        # --- Deduplication ---
        # Query Pinecone to check for similar memory in the same category.
        # Here we use the extracted text as the query. Adjust top_k and threshold as needed.
        search_results = pinecone_index.search_records(
            namespace=namespace,
            query={
                "inputs": {"text": text},
                "top_k": 1
            },
            fields=["text", "category"],
            filter={"category": {"$eq": category}, "user_id": {"$eq": user["uid"]}}
        )
        
        # Assume that if a similar record is returned (e.g., similarity score above threshold),
        # we consider it a duplicate. (Pinecone's integrated inference returns similarity scores,
        # but if not, you can simply check if any record exists.)
        duplicate_found = False
        hits = search_results.get("result", {}).get("hits", [])
        if hits:
            # You might compare text similarity here (or check a similarity score if available).
            # For this example, if any hit exists, we assume it's duplicate.
            duplicate_found = True
        
        if not duplicate_found:
            new_records.append(record)
        else:
            print(f"Duplicate memory found for category {category}: skipping record.")
    
    # Upsert new non-duplicate records into Pinecone.
    if new_records:
        pinecone_index.upsert_records(namespace, new_records)
    
    # Optionally, you can also keep a simple summary of the session (as before).
    summary_prompt = (
        "Summarize the following conversation briefly, focusing on key insights and useful context:\n\n" +
        conversation_text
    )
    summary_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": summary_prompt}
        ]
    )
    summary_text = summary_response.choices[0].message.content
    print("Session summary:", summary_text)
    
    # (Optional) Store session summary as a separate memory record.
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
    
    # Clear in-memory conversation history after finalizing.
    del conversation_histories[session_id]
    
    return JSONResponse(content={"message": "Psychoanalytic memories stored in long term memory."})

@app.get("/new_session")
async def new_session():
    """Generate a new session ID and initialize the conversation history."""
    session_id = str(uuid.uuid4())
    # Initialize with a system message
    conversation_histories[session_id] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    return {"session_id": session_id}

@app.post("/stop_playing")
async def stop_playing():
    """Signal to stop the currently playing TTS stream."""
    stop_event.set()
    return {"message": "Playback stopping..."}

@app.get("/hello")
def say_hello():
    return {"message": "Hello from FastAPI"}

@app.post("/process_audio")
async def process_audio(
    file: UploadFile = File(...),
    tts: bool = False,
    session_id: str = Query(...),
    user: dict = Depends(get_optional_user)  # verifies Firebase token and provides user info
):
    """
    Receives an audio file, transcribes it, retrieves relevant long-term memories (if any),
    appends them to the conversation history, and generates a response.
    """
    # 1. Save the uploaded file temporarily.
    file_extension = file.filename.split(".")[-1]
    temp_file_name = f"{uuid.uuid4()}.{file_extension}"
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, temp_file_name)

    stop_event.clear()

    async with aiofiles.open(temp_file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        # 2. Transcribe the audio using Whisper.
        with open(temp_file_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        user_text = transcript_response.text
        print("User said:", user_text)

        # 3. Retrieve or initialize the session conversation history.
        if session_id not in conversation_histories:
            conversation_histories[session_id] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        history = conversation_histories[session_id]

        # 4. If the user is authenticated, retrieve relevant long-term memories.
        #    The search uses Pinecone’s integrated inference.  
        #    We query the "user-memories" namespace for records whose "chunk_text" 
        #    best matches the current user_text.
        if user:
            results = pinecone_index.search_records(
                namespace="user-memories",
                query={
                    "inputs": {"text": user_text},
                    "top_k": 5
                },
                fields=["text"]
            )
            # Extract the summaries (or "chunk_text") from the returned matches.
            # (Assuming that each match returns its fields under a key "fields".)
            memories = []
            result = results['result']
            for match in result.get("hits", []):
                fields = match.get("fields", {})
                if "text" in fields:
                    memories.append(fields["text"])
            # If any memories were found, prepend them to the conversation as context.
            if memories:
                print(memories)
                retrieved_memories_text = f"""
                The following are the memories that you retained about the user you are talking with:
                {"\n".join(memories)}
                You have the capacity to retain memory about the user, so act accordingly.
                """
                print(retrieved_memories_text)
                history.append({"role": "system", "content": retrieved_memories_text})

        # 5. Append the new user message to the conversation history.
        history.append({"role": "user", "content": user_text})

        # 6. Generate a response using the full conversation history (with injected memories, if any).
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history
        )
        assistant_text = chat_response.choices[0].message.content

        # 7. Append the assistant's response to the conversation history.
        history.append({"role": "assistant", "content": assistant_text})

        # 8. Optionally convert the assistant_text to TTS and stream the audio.
        if tts:
            def audio_stream():
                """Stream the TTS response."""
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
            # Return the plain text response.
            return JSONResponse(
                content={
                    "transcribed_text": user_text,
                    "assistant_text": assistant_text
                }
            )
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Clean up the temporary audio file.
        os.remove(temp_file_path)

@app.get("/retrieve_memories")
async def retrieve_memories(user: dict = Depends(verify_token)):
    """
    Retrieve all long-term memories for the current user.
    Uses a dummy vector to retrieve memories based on metadata filtering.
    """
    # Assume the embedding dimension is 1536 (adjust if different)
    dummy_vector = [0.0] * 1024

    # Search Pinecone with metadata filtering
    results = pinecone_index.query(
        vector=dummy_vector,
        top_k=5,
        filter={"user_id": {"$eq": user["uid"]}},
        namespace="user-memories",
        include_metadata=True  # Ensure metadata is returned
    )

    # Extract relevant metadata
    memories = [match["metadata"]["text"] for match in results["matches"]]

    return JSONResponse(content={"memories": memories})

