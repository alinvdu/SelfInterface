import React, { useState, useEffect, useRef, Suspense } from "react";
import "./App.css";
import LoginButton from "./components/LoginButton";
import { useAuth } from "./auth/AuthContext";

// React Three Fiber imports
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import { HiOutlinePhone } from "react-icons/hi2";
import { HiOutlinePhoneXMark } from "react-icons/hi2";
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
    border: `1px solid rgba(255, 255, 255, 0.4)`,
    borderRadius: "5px",
    background: 'rgba(255, 255, 255, 0.65)',
    fontSize: 14,
    padding: "8px",
    textAlign: "left",
    color: 'black',
    cursor: "pointer",
  };

  const tagStyle = {
    border: `1px solid rgba(255, 255, 255, 0.3)`,
    borderRadius: "5px",
    background: 'rgba(255, 255, 255, 0.55)',
    fontSize: "0.8rem",
    padding: "3px",
    marginTop: 3,
    color: `black`,
  };

  return (
    <div
      style={{
        marginBottom: "1.5rem",
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
function BackgroundScene({ isTalking }) {
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
        <Model isPlaying={isTalking} />
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

  const peerConnectionRef = useRef(null);
  const wsRef = useRef(null); // WebSocket for signaling
  const analyserRef = useRef(null); // For audio analysis

  // Initialize WebRTC connection
  const initiateWebRTC = async () => {
    try {
      // 1) Get local microphone track 2.
      const localStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      peerConnectionRef.current = new RTCPeerConnection({
        iceTransportPolicy: "relay",
        iceServers: [
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
      ],
      });

      // 2) Add the microphone track(s) to the connection
      localStream.getTracks().forEach(track => {
        peerConnectionRef.current.addTrack(track, localStream);
      });

      // WebSocket for signaling
      const wsUrl = api.replace("https", "wss").replace("http", "ws") + `/ws?token=${token}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = async () => {
        // Now that the WebSocket is open, create the offer
        peerConnectionRef.current.addTransceiver('audio', { direction: 'recvonly' });
        const offer = await peerConnectionRef.current.createOffer();
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
          await peerConnectionRef.current.setRemoteDescription(message)
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
              sessionId
            })
          );
        }
      };

      // 8) This is where we receive the **remote** TTS track
      peerConnectionRef.current.ontrack = (event) => {
        console.log('got new track')
        // We create an audio element and attach the remote stream
        // Keep your working playback logic
        setPhoneCalling(false);
        setConversing(true);
        const audio = new Audio();
        audio.srcObject = event.streams[0];
        audio.muted = false;
        audio.volume = 1;
        audio.autoplay = true;
        document.body.appendChild(audio);
        audio.play().catch((e) => console.error("Audio play failed:", e));

        // Add Web Audio API for analysis only (no playback through it)
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            // Resume AudioContext if suspended (due to autoplay policies)
            if (audioContextRef.current.state === 'suspended') {
                audioContextRef.current.resume().then(() => {
                    console.log("AudioContext resumed");
                });
            }
        }
        const audioContext = audioContextRef.current;

        // Create a MediaStreamSource from the same stream
        const source = audioContext.createMediaStreamSource(event.streams[0]);

        // Set up AnalyserNode for silence detection
        analyserRef.current = audioContext.createAnalyser();
        analyserRef.current.fftSize = 2048;
        source.connect(analyserRef.current);

        let lastUpdateTime = 0;
        const debounceTime = 150; // 150ms to smooth out word-by-word toggling
        let silenceTimeout = null; // For delayed silence detection

        // Check audio activity
        const checkAudioActivity = (timestamp) => {
            const bufferLength = analyserRef.current.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            analyserRef.current.getByteTimeDomainData(dataArray);

            // Calculate RMS
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                const value = (dataArray[i] / 128) - 1; // Normalize to -1 to 1
                sum += value * value;
            }
            const rms = Math.sqrt(sum / bufferLength);

            const silenceThreshold = 0.02; // Tune this
            const isActive = rms > silenceThreshold;
            
            if (timestamp - lastUpdateTime >= debounceTime) {
              if (isActive) {
                  // Start talking immediately
                  setIsTalking(isActive)
                  // Clear any pending silence timeout
                  if (silenceTimeout) {
                      clearTimeout(silenceTimeout);
                      silenceTimeout = null;
                  }
              } else {
                  // Delay stopping to bridge short gaps
                  if (!silenceTimeout) {
                      silenceTimeout = setTimeout(() => {
                          setIsTalking(isActive);
                          console.log("Silence detected - isTalking set to false");
                      }, 200);
                  }
              }
              lastUpdateTime = timestamp;
          }

            if (audioContext.state !== 'closed') {
                requestAnimationFrame(checkAudioActivity);
            }
        };

        // Start analysis
        requestAnimationFrame(checkAudioActivity);
      };
    } catch (error) {
      console.error("Error initiating WebRTC:", error);
    }
  };

  const { token, user } = useAuth();
  const [isTalking, setIsTalking] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [memories, setMemories] = useState([]);
  const [conversing, setConversing] = useState(false);
  const [calling, setPhoneCalling] = useState(false);

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

  // Fixed panel background.
  const panelBackground = "rgba(0, 0, 0, 0.45)";

  return (
    <div
      className="App"
      style={{
        position: "relative",
        background: "transparent",
        height: "100vh",
      }}
    >
      <BackgroundScene isTalking={isTalking} />

      <div
        style={{
          position: "absolute",
          top: "18%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          zIndex: 2,
          background: 'rgba(0, 0, 0, 0.25)',
          "backdrop-filter": "blur(8px)",
          "-webkit-backdrop-filter": "blur(8px)",
          border: "1px solid rgba(255, 255, 255, 0.35)",
          borderRadius: "26px",
          padding: "0.7rem 2rem",
          color: "white",
          fontSize: "1.2rem"
        }}
      >
        Atlas
      </div>

      <div
        style={{
          position: "absolute",
          top: "20px",
          right: user ? "20px" : "50%",
          transform: user ? "" : "translateX(50%)",
          zIndex: 2,
          background: 'rgba(0, 0, 0, 0.25)',
          "backdrop-filter": "blur(15px)",
          "-webkit-backdrop-filter": "blur(15px)",
          border: "1px solid rgba(255, 255, 255, 0.3)",
          borderRadius: "26px",
          padding: "15px",
          color: "white",
          fontSize: 17
        }}
      >
        <LoginButton />
      </div>
      {token && (
        <div
          style={{
            position: "absolute",
            top: "20px",
            left: "20px",
            bottom: "20px",
            zIndex: 2,
            width: "300px",
            paddingBottom: 15,
          }}
        >
          <div
            style={{
              "backdrop-filter": "blur(12px)",
              "-webkit-backdrop-filter": "blur(12px)",
              background: 'rgba(0, 0, 0, 0.25)',
              border: "1px solid rgba(255, 255, 255, 0.35)",
              borderRadius: "16px",
              padding: "21px",
              width: "100%",
              maxHeight: "95%",
              overflowY: "auto",
              color: "white",
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-start",
            }}
          >
            <>
              <div style={{ marginTop: 0, fontSize: 21, marginBottom: 15, borderRadius: 6 }}>
                Memories
              </div>
              {memories &&
                memories.map((memory, i) => (
                  <MemoryCard key={i} memory={memory} />
                ))}
            </>
          </div>
        </div>
      )}
      <div
        style={{
          position: "absolute",
          bottom: "50px",
          right: "50%",
          transform: "translateX(50%)",
          zIndex: 2,
          "backdrop-filter": "blur(8px)",
          "-webkit-backdrop-filter": "blur(8px)",
          background: 'rgba(0, 0, 0, 0.25)',
          border: "1px solid rgba(255, 255, 255, 0.4)",
          borderRadius: "46px",
          color: "white",
          textAlign: "center",
          padding: "0.8rem",
          maxWidth: "250px"
        }}
      >
        {!conversing ? (
          <div style={{
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer'
          }} onClick={() => {
            setPhoneCalling(true)
            initiateWebRTC()
          }}>
            <button
              style={{
                background: 'transparent',
                border: '1px solid rgba(255, 255, 255, 0.5)',
                color: "white",
                borderRadius: "50%",
                padding: "10px",
                fontSize: "1.6rem",
                cursor: "pointer",
                lineHeight: '17px',
                width: 46,
                height: 46
              }}
              aria-label="Start Atlas session"
            >
              <HiOutlinePhone />
            </button>
            <div style={{ marginLeft: "1rem", marginRight: "0.5rem", fontSize: "18px" }}>Let's Talk</div>
          </div>
        ) : (
          <div style={{
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer'
          }} onClick={() => {
            setPhoneCalling(true)
            initiateWebRTC()
          }}>
            <div style={{ marginRight: "1rem", marginLeft: "0.5rem", fontSize: "18px" }}>Talking...</div>
            <button
              style={{
                background: 'transparent',
                border: '1px solid rgba(255, 255, 255, 0.5)',
                color: "white",
                borderRadius: "50%",
                padding: "10px",
                fontSize: "1.6rem",
                cursor: "pointer",
                lineHeight: '17px',
                width: 46,
                height: 46
              }}
              aria-label="Start Atlas session"
            >
              <HiOutlinePhoneXMark />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
