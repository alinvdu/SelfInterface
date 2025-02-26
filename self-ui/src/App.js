import React, { useState, useEffect, useRef, Suspense } from "react";
import "./App.css";
import LoginButton from "./components/LoginButton";
import { useAuth } from "./auth/AuthContext";

// React Three Fiber imports
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import { BiColorFill } from "react-icons/bi";
import { LuPhoneCall } from "react-icons/lu";
import Model from "./Model.js";

const api = "https://selfinterface-simple-env.up.railway.app";

// --- MemoryCard Component ---
function MemoryCard({ memory, hue }) {
  const [expanded, setExpanded] = useState(false);
  const previewLimit = 100;
  const text = typeof memory === "string" ? memory : memory.text;
  const category =
    typeof memory === "string"
      ? "General"
      : memory.category.split("_").join(" ") || "General";
  const previewText =
    text.length > previewLimit ? text.slice(0, previewLimit) + "..." : text;

  const cardStyle = {
    border: `1px solid hsl(${hue}, 40%, 40%)`,
    borderRadius: "3px",
    backgroundColor: `hsl(${hue}, 40%, 20%)`,
    fontSize: 14,
    padding: "8px",
    textAlign: "left",
    color: `hsl(${hue}, 40%, 70%)`,
    cursor: "pointer",
  };

  const tagStyle = {
    background: `hsl(${hue}, 40%, 25%)`,
    border: `1px solid hsl(${hue}, 40%, 40%)`,
    borderRadius: "3px",
    fontSize: "0.8rem",
    padding: "3px",
    marginTop: 3,
    color: `hsl(${hue}, 40%, 75%)`,
  };

  return (
    <div
      style={{
        marginBottom: "1rem",
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
      }}
    >
      <div style={cardStyle} onClick={() => setExpanded(!expanded)}>
        <div>{expanded ? text : previewText}</div>
      </div>
      <div style={tagStyle}>{category}</div>
    </div>
  );
}

// --- Background Scene Component ---
function BackgroundScene({ isPlaying }) {
  return (
    <Canvas
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        background: "transparent",
        zIndex: 0,
        pointerEvents: "auto",
      }}
      camera={{ position: [0, -0.15, 1.25], fov: 60 }}
      gl={{ alpha: true }}
    >
      <ambientLight intensity={1.2} />
      <directionalLight position={[10, 10, 5]} intensity={2.0} />
      <pointLight position={[0, 5, 0]} intensity={2.0} />
      <Suspense fallback={null}>
        <Model isPlaying={isPlaying} />
        <OrbitControls
          enableZoom={false}
          enableRotate={false}
          enablePan={false}
        />
      </Suspense>
    </Canvas>
  );
}

function App() {
  // Audio context and oscillator refs
  const audioContextRef = useRef(null);

  // Function to initialize AudioContext and start continuous sound
  const initializeProactive = () => {
    if (!audioContextRef.current) {
      // Trigger proactive message after initialization
      if (token && sessionId) {
        setConversing(true);
        fetchAndPlayProactiveMessage(
          `${api}/proactive_message?session_id=${sessionId}`
        ).catch((error) =>
          console.error("Error triggering proactive message:", error)
        );
      }
    }
  };

  const peerConnectionRef = useRef(null);
  const wsRef = useRef(null); // WebSocket for signaling

  // Initialize WebRTC connection
  const initiateWebRTC = async () => {
    try {
      // 1) Get local microphone track 2.
      const localStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      peerConnectionRef.current = new RTCPeerConnection({
        iceServers: [
            {
              urls: "stun:stun.relay.metered.ca:80",
            },
            {
              urls: "turn:global.relay.metered.ca:80?transport=tcp",
              username: "6975b17010809692e9b965f6",
              credential: "P+JbvCClSCMe6XW1",
            },
            {
              urls: "turns:global.relay.metered.ca:443?transport=tcp",
              username: "6975b17010809692e9b965f6",
              credential: "P+JbvCClSCMe6XW1",
            },
        ]
      });

      // 2) Add the microphone track(s) to the connection
      // localStream.getTracks().forEach(track => {
      //   console.log("Adding local mic track:", track);
      //   peerConnectionRef.current.addTrack(track, localStream);
      // });

      // WebSocket for signaling
      const wsUrl = api.replace("https", "wss").replace("http", "ws") + "/ws";
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = async () => {
        console.log("WebSocket opened");
        setIsConnected(true);

        // Now that the WebSocket is open, create the offer
        const offer = await peerConnectionRef.current.createOffer({
          offerToReceiveAudio: true,
        });
        console.log("JS SDP Offer:", offer.sdp);
        await peerConnectionRef.current.setLocalDescription(offer);

        // 4) Send the offer to the server
        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({
              type: "offer",
              sdp: offer.sdp,
              sessionId,
              token,
            })
          );
        }
      };

      // 5) When we get the answer, set it as remote description
      wsRef.current.onmessage = async (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "answer") {
          console.log("Received answer from server");
          await peerConnectionRef.current.setRemoteDescription(
            new RTCSessionDescription({ type: "answer", sdp: message.sdp })
          );
          console.log(
            "Client SDP:",
            peerConnectionRef.current.remoteDescription.sdp
          );
        } else if (message.type === "ice-candidate") {
          console.log("ice candidate received");
          await peerConnectionRef.current.addIceCandidate(
            new RTCIceCandidate(message.candidate)
          );
        }
      };

      // 7) Handle sending our local ICE candidates up to the server
      peerConnectionRef.current.onicecandidate = (event) => {
        console.log("ice event", event);
        if (event.candidate) {
          wsRef.current.send(
            JSON.stringify({
              type: "ice-candidate",
              candidate: event.candidate,
              sessionId,
            })
          );
        }
      };

      // 8) This is where we receive the **remote** TTS track
      peerConnectionRef.current.ontrack = (event) => {
        console.log(
          "ontrack event with remote track(s):",
          event.streams[0].getTracks()
        );
        console.log(
          "tracks right now",
          peerConnectionRef.current.getReceivers()
        );
        // We create an audio element and attach the remote stream
        const audio = new Audio();
        audio.srcObject = event.streams[0];
        audio.muted = false;
        audio.volume = 1;
        audio.controls = true; // for debugging
        audio.autoplay = true;

        audio.onplaying = () => {
          console.log("Playing remote TTS track");
          setIsPlaying(true);
        };
        audio.onended = () => {
          console.log("Remote TTS track ended");
          setIsPlaying(false);
        };

        document.body.appendChild(audio);
        audio.play().catch((e) => console.error("Audio play failed:", e));
      };
    } catch (error) {
      console.error("Error initiating WebRTC:", error);
    }
  };

  const { token } = useAuth();
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false); // For regular process_audio playback
  const [isProactivePlaying, setIsProactivePlaying] = useState(false); // Proactive audio playback active
  const [isProactiveLoading, setIsProactiveLoading] = useState(false); // Proactive API call pending
  const [isProcessingAudio, setIsProcessingAudio] = useState(false); // process_audio API call pending
  const [sessionId, setSessionId] = useState(null);
  const [memories, setMemories] = useState([]);
  const [conversing, setConversing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  // Control button style: if disabled due to proactive loading, processing audio, etc.
  const getControlButtonStyle = (baseStyle, allowIsPlaying = false) => {
    if (
      isProactiveLoading ||
      isProcessingAudio ||
      (!allowIsPlaying && isPlaying) ||
      isProactivePlaying
    ) {
      return {
        ...baseStyle,
        background: "gray",
        color: "#ccc",
        opacity: 0.6,
        cursor: "not-allowed",
      };
    }
    return baseStyle;
  };

  const [hue, setHue] = useState(() => {
    const storedHue = localStorage.getItem("hue");
    return storedHue ? Number(storedHue) : 260;
  });

  // Base dynamicButtonStyle remains fixed.
  const dynamicButtonStyle = {
    background: `hsl(${hue}, 40%, 30%)`,
    border: `1px solid hsl(${hue}, 40%, 40%)`,
    color: "white",
    borderRadius: "3px",
    padding: "5px",
  };

  useEffect(() => {
    localStorage.setItem("hue", hue);
  }, [hue]);

  const [showHueSlider, setShowHueSlider] = useState(false);
  const sliderRef = useRef(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  // ----- Common Audio Playback Helpers -----
  async function playAudioFromResponse(response, onPlay, onEnded) {
    try {
      const reader = response.body.getReader();
      const chunks = [];
      let done = false;
      while (!done) {
        const { value, done: readingDone } = await reader.read();
        if (value) chunks.push(value);
        done = readingDone;
      }
      const blob = new Blob(chunks, { type: "audio/mpeg" });
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);
      window.currentAudio = audio;
      audio.onplaying = () => {
        if (onPlay) onPlay();
      };
      audio.onended = () => {
        if (onEnded) onEnded();
      };
      await audio.play();
    } catch (error) {
      console.error("Error in playAudioFromResponse:", error);
    }
  }

  async function fetchAndPlayAudio(endpoint, options = {}, onPlay, onEnded) {
    try {
      const response = await fetch(endpoint, {
        method: options.method || "GET",
        headers: { Authorization: `Bearer ${token}` },
        body: options.body,
      });
      if (!response.ok)
        throw new Error("Error fetching audio: " + response.statusText);
      await playAudioFromResponse(response, onPlay, onEnded);
    } catch (error) {
      console.error("Error in fetchAndPlayAudio:", error);
    }
  }

  async function fetchAndPlayProactiveMessage(endpoint) {
    setIsProactiveLoading(true);
    await fetchAndPlayAudio(
      endpoint,
      { method: "POST" },
      () => {
        // When audio starts playing:
        setIsProactiveLoading(false);
        setIsProactivePlaying(true);
      },
      () => {
        // When proactive audio ends:
        setIsProactivePlaying(false);
      }
    );
  }
  // ----- End of Audio Playback Helpers -----

  // Combined new_session and proactive message call.
  useEffect(() => {
    const createSession = async () => {
      if (token && !sessionId) {
        try {
          const res = await fetch(api + "/new_session", {
            headers: { Authorization: `Bearer ${token}` },
          });
          const data = await res.json();
          setSessionId(data.session_id);
        } catch (error) {
          console.error("Error creating session and proactive message:", error);
        }
      }
    };
    createSession();
  }, [token, sessionId]);

  // Fetch memories (this can remain separate)
  useEffect(() => {
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
    if (token) fetchMemories();
  }, [token]);

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
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showHueSlider]);

  const startRecording = async () => {
    if (isProactiveLoading || isProactivePlaying || isProcessingAudio) return;
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
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/mp3",
        });
        audioChunksRef.current = [];
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.mp3");
        try {
          setIsProcessingAudio(true);
          const response = await fetch(
            api + `/process_audio?tts=true&session_id=${sessionId}`,
            {
              method: "POST",
              body: formData,
              headers: { Authorization: `Bearer ${token}` },
            }
          );
          if (!response.ok)
            throw new Error("Server error: " + response.statusText);
          await playAudioFromResponse(
            response,
            () => {
              setIsProcessingAudio(false);
              setIsPlaying(true);
            },
            () => {
              setIsPlaying(false);
            }
          );
        } catch (error) {
          console.error("Error sending audio:", error);
          setIsRecording(false);
          setIsPlaying(false);
          setIsProcessingAudio(false);
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
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
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

  // Status text: show "Atlas is thinking..." if proactive loading or process_audio is pending,
  // "Atlas is speaking..." if audio is playing, otherwise "Ready to record".
  const statusText =
    isProactiveLoading || isProcessingAudio
      ? "Atlas is thinking..."
      : isProactivePlaying || isPlaying
      ? "Atlas is speaking..."
      : "Ready to record";

  // Fixed panel background.
  const panelBackground = "rgba(0, 0, 0, 0.6)";

  return (
    <div
      className="App"
      style={{
        position: "relative",
        background: "transparent",
        height: "100vh",
      }}
    >
      <BackgroundScene isPlaying={isPlaying || isProactivePlaying} />

      <div
        style={{
          position: "absolute",
          top: "18%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          zIndex: 2,
          background: `hsl(${hue}, 40%, 30%)`,
          border: `1px solid hsl(${hue}, 40%, 40%)`,
          borderRadius: "4px",
          padding: "0.5rem 1rem",
          color: "white",
          fontSize: "1.2rem",
        }}
      >
        Atlas
      </div>

      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          zIndex: 2,
          backgroundColor: "rgba(20, 20, 20, 0.85)",
          borderRadius: "4px",
          padding: "10px",
          color: "white",
        }}
      >
        <LoginButton dynamicButtonStyle={dynamicButtonStyle} />
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          position: "absolute",
          top: "20px",
          right: "20px",
          zIndex: 2,
          background: `hsl(${hue}, 40%, 30%)`,
          border: `1px solid hsl(${hue}, 40%, 40%)`,
          color: "white",
          borderRadius: "3px",
          padding: "12px",
          cursor: "pointer",
        }}
      >
        <BiColorFill size={23} />
        <button
          style={{
            background: "transparent",
            border: "none",
            color: "white",
            cursor: "pointer",
            fontSize: 16,
          }}
          onClick={() => setShowHueSlider(!showHueSlider)}
        >
          Pick Color
        </button>
      </div>

      {showHueSlider && (
        <div
          ref={sliderRef}
          style={{
            position: "absolute",
            top: "75px",
            right: "20px",
            zIndex: 2,
            backgroundColor: "rgba(20,20,20,0.85)",
            borderRadius: "4px",
            padding: "10px",
            color: "white",
          }}
        >
          <div style={{ marginBottom: "5px" }}>Color</div>
          <input
            type="range"
            min="0"
            max="360"
            value={hue}
            onChange={(e) => setHue(e.target.value)}
          />
        </div>
      )}

      {token && (
        <div
          style={{
            position: "absolute",
            top: "120px",
            left: "20px",
            bottom: "20px",
            zIndex: 2,
            width: "300px",
            paddingBottom: 15,
          }}
        >
          <div
            style={{
              backgroundColor: "rgba(20, 20, 20, 0.85)",
              borderRadius: "4px",
              padding: "10px",
              width: "100%",
              maxHeight: "100%",
              overflowY: "auto",
              color: "white",
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-start",
            }}
          >
            <>
              <div style={{ marginTop: 0, fontSize: 21, marginBottom: 15 }}>
                Memories
              </div>
              {memories &&
                memories.map((memory, i) => (
                  <MemoryCard key={i} memory={memory} hue={hue} />
                ))}
            </>
          </div>
        </div>
      )}
      <div
        style={{
          position: "absolute",
          top: "50%",
          right: "20px",
          transform: "translateY(-50%)",
          zIndex: 2,
          backgroundColor: panelBackground,
          borderRadius: "4px",
          color: "white",
          textAlign: "center",
          padding: "1rem",
          maxWidth: "250px",
        }}
      >
        {!conversing ? (
          <div>
            <h3 style={{ marginBottom: "1rem" }}>Start Atlas Session</h3>
            <button
              onClick={initiateWebRTC}
              style={{
                background: `hsl(${hue}, 40%, 30%)`,
                border: `1px solid hsl(${hue}, 40%, 40%)`,
                color: "white",
                borderRadius: "50%",
                padding: "10px",
                fontSize: "1.5rem",
                cursor: "pointer",
              }}
              aria-label="Start Atlas session"
            >
              <LuPhoneCall />
            </button>
          </div>
        ) : (
          <div>
            <div style={{ marginBottom: "1rem" }}>
              <button
                onClick={startRecording}
                disabled={
                  isRecording ||
                  isPlaying ||
                  isProactivePlaying ||
                  isProactiveLoading ||
                  isProcessingAudio ||
                  !sessionId
                }
                style={{
                  marginBottom: "0.5rem",
                  ...getControlButtonStyle(dynamicButtonStyle),
                }}
              >
                {isRecording ? "Recording..." : "Speak"}
              </button>
              <button
                style={{
                  ...getControlButtonStyle(dynamicButtonStyle, true),
                  marginLeft: 10,
                }}
                onClick={() => {
                  isPlaying ? stopPlaying() : stopRecording();
                }}
                disabled={
                  isProactivePlaying || isProactiveLoading || isProcessingAudio
                }
              >
                {isPlaying ? "Stop Atlas" : "Stop Recording"}
              </button>
            </div>

            <p style={{ margin: 0 }}>{statusText}</p>

            {token && (
              <div style={{ marginTop: "1rem" }}>
                <button
                  style={getControlButtonStyle(dynamicButtonStyle)}
                  onClick={finalizeConversation}
                  disabled={
                    isProactivePlaying ||
                    isProactiveLoading ||
                    isProcessingAudio
                  }
                >
                  End Conversation
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
