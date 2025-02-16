import React, { useState, useEffect, useRef, Suspense } from 'react';
import './App.css';
import LoginButton from "./components/LoginButton";
import { useAuth } from "./auth/AuthContext";

// React Three Fiber imports
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';

import { BiColorFill } from "react-icons/bi";

import Model from './Model.js'

const api = "https://silver-space-pancake-97w4jq55q9v2xxxg-8000.app.github.dev";

// --- MemoryCard Component ---
// This component shows a preview of the memory text with a border.
// A small outlined tag displays the memory category.
// Clicking the card toggles expansion.
function MemoryCard({ memory, hue }) {
  const [expanded, setExpanded] = useState(false);
  const previewLimit = 100; // preview limit (characters)

  // If memory is a simple string, treat it as text with a default category.
  const text = typeof memory === 'string' ? memory : memory.text;
  const category = typeof memory === 'string'
    ? 'General'
    : memory.category.split("_").join(" ") || 'General';
  const previewText = text.length > previewLimit ? text.slice(0, previewLimit) + '...' : text;

  // Dynamic styles using the current hue value.
  // Here, the memory box background is made darker by lowering the lightness.
  const cardStyle = {
    border: `1px solid hsl(${hue}, 40%, 40%)`,
    borderRadius: '3px',
    backgroundColor: `hsl(${hue}, 40%, 20%)`, // Darker background
    fontSize: 14,
    padding: '8px',
    textAlign: 'left',
    color: `hsl(${hue}, 40%, 70%)`,
    cursor: 'pointer'
  };

  const tagStyle = {
    background: `hsl(${hue}, 40%, 25%)`,
    border: `1px solid hsl(${hue}, 40%, 40%)`,
    borderRadius: '3px',
    fontSize: '0.8rem',
    padding: '3px',
    marginTop: 3,
    color: `hsl(${hue}, 40%, 75%)`
  };

  return (
    <div style={{
      marginBottom: '1rem',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'flex-start'
    }}>
      <div style={cardStyle} onClick={() => setExpanded(!expanded)}>
        <div>
          {expanded ? text : previewText}
        </div>
      </div>
      <div style={tagStyle}>
        {category}
      </div>
    </div>
  );
}

// --- Background Scene Component ---
function BackgroundScene({ isPlaying }) {
  return (
    <Canvas
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background: 'transparent',
        zIndex: 0,
        pointerEvents: 'auto'
      }}
      camera={{ position: [0, -0.15, 1.25], fov: 60 }}
      gl={{ alpha: true }}
    >
      <ambientLight intensity={1.2} />
      <directionalLight position={[10, 10, 5]} intensity={2.0} />
      <pointLight position={[0, 5, 0]} intensity={2.0} />
      <hemisphereLight intensity={1.2} skyColor="#f5efe7" groundColor="#444444" position={[0, 10, 0]} />
      <Suspense fallback={null}>
        <Model isPlaying={isPlaying} />
        <OrbitControls enableZoom={false} enableRotate={false} enablePan={false} />
      </Suspense>
    </Canvas>
  );
}

function App() {
  const { token } = useAuth();
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [memories, setMemories] = useState([]);

  // New state for hue and showing the slider.
  const [hue, setHue] = useState(260);
  const [showHueSlider, setShowHueSlider] = useState(false);
  const sliderRef = useRef(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const mediaSourceRef = useRef(null);
  const readerRef = useRef(null);

  useEffect(() => {
    const createSession = async () => {
      try {
        const res = await fetch(api + "/new_session", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        setSessionId(data.session_id);
      } catch (error) {
        console.error("Error creating new session:", error);
      }
    };

    const fetchMemories = async () => {
      try {
        const res = await fetch(api + "/retrieve_memories", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        setMemories(data.memories);
      } catch (error) {
        console.error("Error fetching memories:", error);
      }
    };

    createSession();
    if (token) {
      fetchMemories();
    }
  }, [token]);

  // Close the hue slider when clicking outside.
  useEffect(() => {
    function handleClickOutside(event) {
      if (sliderRef.current && !sliderRef.current.contains(event.target)) {
        setShowHueSlider(false);
      }
    }
    if (showHueSlider) {
      document.addEventListener("mousedown", handleClickOutside);
    } else {
      document.removeEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showHueSlider]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }

        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/mp3" });
        audioChunksRef.current = [];
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.mp3");

        try {
          const response = await fetch(api + `/process_audio?tts=true&session_id=${sessionId}`, {
            method: "POST",
            body: formData,
            headers: { Authorization: `Bearer ${token}` },
          });
          if (!response.ok) {
            throw new Error("Server error: " + response.statusText);
          }

          mediaSourceRef.current = new MediaSource();
          const audioElement = new Audio();
          audioElement.src = URL.createObjectURL(mediaSourceRef.current);
          document.body.appendChild(audioElement);
          setIsPlaying(true);
          audioElement.play();

          mediaSourceRef.current.addEventListener("sourceopen", async () => {
            const sourceBuffer = mediaSourceRef.current.addSourceBuffer("audio/mpeg");
            const reader = response.body.getReader();
            readerRef.current = reader;

            async function pushData() {
              const { done, value } = await reader.read();
              if (done) {
                if (mediaSourceRef.current.readyState === "open") {
                  try {
                    mediaSourceRef.current.endOfStream();
                  } catch (error) {
                    console.log("Error ending stream", error);
                  }
                }
                return;
              }
              sourceBuffer.addEventListener("updateend", function onUpdateEnd() {
                sourceBuffer.removeEventListener("updateend", onUpdateEnd);
                pushData();
              });
              try {
                sourceBuffer.appendBuffer(value);
              } catch (err) {
                console.error("Error appending buffer", err);
              }
            }
            pushData();

            audioElement.onended = () => {
              setIsPlaying(false);
              setIsRecording(false);
            };
          });

          window.currentAudio = audioElement;
        } catch (error) {
          console.error("Error sending audio:", error);
          setIsRecording(false);
          setIsPlaying(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const stopPlaying = () => {
    if (window.currentAudio) {
      if (readerRef.current) {
        readerRef.current.cancel();
      }
      window.currentAudio.pause();
      window.currentAudio.currentTime = 0;
      window.currentAudio.src = "";
    }
    setIsPlaying(false);
    fetch(api + "/stop_playing", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const finalizeConversation = async () => {
    if (sessionId && token) {
      try {
        await fetch(api + `/finalize_conversation?session_id=${sessionId}`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        });
      } catch (error) {
        console.error("Error finalizing session:", error);
      }
    }
  };

  // Define dynamic style for buttons using the current hue.
  const dynamicButtonStyle = {
    background: `hsl(${hue}, 40%, 30%)`,
    border: `1px solid hsl(${hue}, 40%, 40%)`,
    color: 'white',
    appearance: 'none',
    borderRadius: '3px',
    padding: '5px'
  };

  return (
    <div className="App" style={{ position: "relative", background: "transparent", height: "100vh" }}>
      {/* 3D background scene */}
      <BackgroundScene isPlaying={isPlaying} />

      {/* Psychologist Name Box (Atlas) */}
      <div style={{
        position: 'absolute',
        top: '18%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        zIndex: 2,
        background: `hsl(${hue}, 40%, 30%)`,
        border: `1px solid hsl(${hue}, 40%, 40%)`,
        borderRadius: '4px',
        padding: '0.5rem 1rem',
        color: 'white',
        fontSize: '1.2rem'
      }}>
        Atlas
      </div>

      {/* Login Panel */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        zIndex: 2,
        backgroundColor: 'rgba(20, 20, 20, 0.85)',
        borderRadius: '4px',
        padding: '10px',
        color: 'white'
      }}>
        <LoginButton dynamicButtonStyle={dynamicButtonStyle} />
      </div>

      {/* Hue Button on top right */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        position: 'absolute',
        top: '20px',
        right: '20px',
        zIndex: 2,
        background: `hsl(${hue}, 40%, 30%)`,
        border: `1px solid hsl(${hue}, 40%, 40%)`,
        color: 'white',
        appearance: 'none',
        borderRadius: '3px',
        padding: '5px',
        cursor: 'pointer',
        padding: 12
      }}>
        <BiColorFill size={23} />
        <button style={{background: 'transparent', border: 'none', color: 'white', cursor: 'pointer', fontSize: 16}} onClick={() => setShowHueSlider(!showHueSlider)}>
          Pick Color
        </button>
      </div>

      {/* Hue Slider Panel */}
      {showHueSlider && (
        <div ref={sliderRef} style={{
          position: 'absolute',
          top: '75px',
          right: '20px',
          zIndex: 2,
          backgroundColor: 'rgba(20,20,20,0.85)',
          borderRadius: '4px',
          padding: '10px',
          color: 'white'
        }}>
          <div style={{ marginBottom: '5px' }}>Color</div>
          <input
            type="range"
            min="0"
            max="360"
            value={hue}
            onChange={(e) => setHue(e.target.value)}
          />
        </div>
      )}

      {/* Memories Panel */}
      <div style={{
        position: 'absolute',
        top: '120px',
        left: '20px',
        bottom: '20px',
        zIndex: 2,
        width: '300px',
        paddingBottom: 15
      }}>
        <div style={{
          backgroundColor: 'rgba(20, 20, 20, 0.85)',
          borderRadius: '4px',
          padding: '10px',
          width: "100%",
          maxHeight: "100%",
          overflowY: 'auto',
          color: 'white',
          display: 'flex',
          flexDirection: 'column',
          textAling: 'left',
          alignItems: 'flex-start'
        }}>
          {token && (
            <>
              <div style={{ marginTop: 0, fontSize: 21, marginBottom: 15 }}>Memories</div>
              {memories && memories.map((memory, i) => (
                <MemoryCard key={i} memory={memory} hue={hue} />
              ))}
            </>
          )}
        </div>
      </div>

      {/* Right overlay: Recording controls and End Conversation */}
      <div style={{
        position: 'absolute',
        top: '50%',
        right: '20px',
        transform: 'translateY(-50%)',
        zIndex: 2,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        borderRadius: '4px',
        color: 'white',
        textAlign: 'center',
        padding: '1rem',
        maxWidth: '250px'
      }}>
        <div style={{ marginBottom: "1rem" }}>
          <button 
            onClick={startRecording} 
            disabled={isRecording || isPlaying || !sessionId}
            style={{ marginBottom: "0.5rem", ...dynamicButtonStyle }}
          >
            {isRecording ? "Recording..." : "Start Recording"}
          </button>
          <br />
          <button style={dynamicButtonStyle} onClick={() => {
            if (isPlaying) {
              stopPlaying();
            } else {
              stopRecording();
            }
          }}>
            {isPlaying ? "Stop Playing" : "Stop Recording"}
          </button>
        </div>
        {isPlaying && <p style={{ margin: 0 }}>Playing response...</p>}
        {!isRecording && !isPlaying && <p style={{ margin: 0 }}>Ready to record</p>}
        {token && (
          <div style={{ marginTop: '1rem' }}>
            <button style={dynamicButtonStyle} onClick={finalizeConversation}>
              End Conversation
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
