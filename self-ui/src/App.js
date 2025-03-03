import React, { useState, useEffect, useRef, Suspense } from "react";
import "./App.css";
import LoginButton from "./components/LoginButton";
import { useAuth } from "./auth/AuthContext";

// React Three Fiber imports
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import { HiOutlinePhone, HiOutlinePhoneXMark } from "react-icons/hi2";
import { IoEarOutline } from "react-icons/io5";
import { BiUserVoice } from "react-icons/bi";
import { LuBrainCog } from "react-icons/lu";

import Model from "./Model.js";

import LoadingDiv from "./components/LoadingDiv";
import CollapsibleMemoriesPanel from "./components/CollapsiblePanel.js";
import Chat from "./components/Chat.js";

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
      camera={{ position: [0.25, -0.05, 0.6], fov: 60 }}
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
      const localStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      peerConnectionRef.current = new RTCPeerConnection({
        // iceTransportPolicy: "relay",
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

      localStream.getTracks().forEach(track => {
        peerConnectionRef.current.addTrack(track, localStream);
      });

      peerConnectionRef.current.addTransceiver('audio', { direction: 'recvonly' });
      const offer = await peerConnectionRef.current.createOffer();
      await peerConnectionRef.current.setLocalDescription(offer);

      wsRef.current.send(
        JSON.stringify({
          type: "offer",
          sdp: offer.sdp,
          sessionId,
          token,
        })
      );

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

      peerConnectionRef.current.ontrack = (event) => {
        console.log('got new track')
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
              animationFrameIdRef.current = requestAnimationFrame(checkAudioActivity);
            }
        };

        // Start analysis
        animationFrameIdRef.current = requestAnimationFrame(checkAudioActivity);
      };
    } catch (error) {
      console.error("Error initiating WebRTC:", error);
    }
  };

  const { token, user, loading } = useAuth();
  const [isTalking, setIsTalking] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [memories, setMemories] = useState([]);
  const [conversing, setConversing] = useState(false);
  const [calling, setPhoneCalling] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [chat, setChat] = useState([])
  const [loadingChat, setLoadingChat] = useState(false)
  const [disconnecting, setDisconnecting] = useState(false);
  const animationFrameIdRef = useRef(null);
  const [isWsOpen, toggleWsOpen] = useState(false);

  const createAndConnectWs = (currentSessionId, currentToken) => {
    const wsUrlPath = currentToken ? `/ws?token=${currentToken}&session_id=${currentSessionId}` : '/ws?session_id=' + currentSessionId
    const wsUrl = api.replace("https", "wss").replace("http", "ws") + wsUrlPath;
    wsRef.current = new WebSocket(wsUrl);

    // 5) When we get the answer, set it as remote description
    wsRef.current.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "answer") {
        await peerConnectionRef.current.setRemoteDescription(message)
        console.log(
          "Client SDP:",
          peerConnectionRef.current.remoteDescription.sdp
        );
      } else if (message.type === "ice-candidate") {
        await peerConnectionRef.current.addIceCandidate(
          new RTCIceCandidate(message.candidate)
        );
      } else if (message.type === "PROCESSING") {
        setProcessing(true);
      } else if (message.type === "FINISHED_PROCESSING") {
        setProcessing(false);
      } else if (message.type === "CHAT_MESSAGE") {
        setChat(chat => [...chat, {
          "role": "assistant",
          "content": message.message
        }])
        setLoadingChat(false)
      } else if (message.type === "rtc_disconnected") {
        setDisconnecting(false);
        setConversing(false);
        setPhoneCalling(false);
        setIsTalking(false);
        setProcessing(false);
      }
    };

    wsRef.current.onopen = () => {
      toggleWsOpen(true);
    }

    wsRef.current.onclose = () => {
      toggleWsOpen(false);
      setDisconnecting(false);
      setConversing(false);
      setPhoneCalling(false);
      setIsTalking(false);
      setProcessing(false);
      setTimeout(() => {
        createAndConnectWs(currentSessionId, currentToken)
      }, 500)
    }
  }

  // Combined new_session and proactive message call.
  useEffect(() => {
    if (!loading) {
      const createSession = async () => {
        if (!sessionId) {
          try {
            const res = await fetch(api + "/new_session", {
              headers: { Authorization: `Bearer ${token}` },
            });
            const data = await res.json();
            setSessionId(data.session_id);

            createAndConnectWs(data.session_id, token)
          } catch (error) {
            console.error("Error creating session and proactive message:", error);
          }
        }
      };
      createSession();
    }
  }, [loading]);

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

  const renderActivityIcon = () => {
    if (processing) {
      return <LuBrainCog style={{fontSize: 21}} />
    }

    return <IoEarOutline style={{fontSize: 21}} />
  }

  const handleDisconnect = async () => {
    if (!sessionId) return;
    
    setDisconnecting(true);
    
    try {
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
        animationFrameIdRef.current = null;
      }

      // Close local tracks
      if (peerConnectionRef.current) {
        const senders = peerConnectionRef.current.getSenders();
        senders.forEach(sender => {
          if (sender.track) {
            sender.track.stop();
          }
        });
        
        // Close the peer connection
        peerConnectionRef.current.close();
        peerConnectionRef.current = null;
      }
      
      // Reset audio context
      if (audioContextRef.current) {
        await audioContextRef.current.close();
        audioContextRef.current = null;
      }
      
      if (analyserRef.current) {
        analyserRef.current = null;
      }
      
      // Tell the server to clean up this session's connections - we are not really disconnecting everything
      // Tell the server to clean up this session's WebRTC connection
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: "rtc_disconnect",
          sessionId
        }));
      }
    } catch (error) {
      console.error("Error during disconnection:", error);
      setDisconnecting(false);
    }
  };

  const renderConversing = () => {
    if (!isWsOpen) {
      return null
    }

    if (!conversing) {
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          padding: 12
        }} onClick={() => {
          setPhoneCalling(true)
          initiateWebRTC()
        }}>
          <LoadingDiv
            isLoading={calling} 
            duration={0.75} 
            width={`${46}px`}
            height={`${46}px`}
            borderWidth={1}
            loadingColor="#FFFFFF"
            borderColor="rgba(255, 255, 255, 0.5)"
            borderRadius={`${46}px`}
            backgroundColor="transparent"
            loadingSegmentPercentage={25}
          >
            <HiOutlinePhone style={{ fontSize: 21 }} />
          </LoadingDiv>
          <div style={{ marginLeft: "1rem", marginRight: "0.5rem", fontSize: "18px" }}>{calling ? "Calling Atlas..." : "Let's Talk"}</div>
        </div>
      )
    }
  
    return (
        <div style={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          padding: 12
        }}>
          <div style={{ marginRight: "1rem", marginLeft: "0.5rem", fontSize: "18px" }}>{processing ? "Processing thoughts" : "I'm listening..."}</div>
          <LoadingDiv
            isLoading={processing} 
            duration={0.75} 
            width={`${46}px`}
            height={`${46}px`}
            borderWidth={1}
            loadingColor="#FFFFFF"
            borderColor="rgba(255, 255, 255, 0.5)"
            borderRadius={`${46}px`}
            backgroundColor="transparent"
            loadingSegmentPercentage={25}
            isGlowing
          >
            {renderActivityIcon()}
          </LoadingDiv>
        </div>
    )
  }

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
      <div style={{
          position: "absolute",
          top: "10px",
          left: "10px",
          zIndex: 2,
          background: 'rgba(0, 0, 0, 0.25)',
          "backdrop-filter": "blur(8px)",
          "-webkit-backdrop-filter": "blur(8px)",
          border: "1px solid rgba(255, 255, 255, 0.35)",
          borderRadius: "21px",
          padding: "0.7rem 2rem",
          color: "white",
          fontSize: "23px",
          width: 140,
          height: 40,
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-start"
        }}>
          <img style={{ width: 55 }} src="selfai-logo.png" />
          <span style={{marginLeft: 10}}>Self AI</span>
      </div>
      <div
        style={{
          position: "absolute",
          top: "100px",
          left: "53%",
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

      {!loading &&
      <div
        style={{
          position: "absolute",
          top: "10px",
          right: "20px",
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
      </div>}
      <div style={{
        position: "absolute",
        top: "100px",
        left: "10px",
        bottom: "20px",
      }}>
          <CollapsibleMemoriesPanel
            token={token}
            requiresAccount
            memories={memories}
            MemoryCard={MemoryCard}
            title="Memories"
          >
            {memories &&
                memories.map((memory, i) => (
                  <MemoryCard key={i} memory={memory} />
                ))}
          </CollapsibleMemoriesPanel>
          <CollapsibleMemoriesPanel
            memories={[]}
            MemoryCard={() => {}}
            title="Chat"
            openedByDefault
            canBeToggled={!conversing && !calling}
          >
            <div style={{
              position: "relative",
              width: "100%",
              height: "100%",
              minHeight: 20
            }}>
              <div style={{
                position: "absolute",
                width: "100%",
                height: 1,
                background: "rgba(255, 255, 255, 0.25)"
              }} />
              {!chat || !chat.length || !isWsOpen ?
              <div style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                paddingTop: 15
              }}>
                <LoadingDiv
                  isLoading 
                  duration={0.75} 
                  width={`${25}px`}
                  height={`${25}px`}
                  borderWidth={1}
                  loadingColor="#FFFFFF"
                  borderColor="rgba(255, 255, 255, 0.5)"
                  borderRadius={`${10}px`}
                  backgroundColor="transparent"
                  loadingSegmentPercentage={25}
                />
              </div>
              : <Chat chat={chat} onSendMessage={message => {
                if (wsRef.current) {
                  setChat([...chat, {
                    "role": "user",
                    "content": message
                  }])

                  wsRef.current.send(JSON.stringify({
                    "type": "CHAT_MESSAGE",
                    "message": message
                  }))

                  setLoadingChat(true);
                }
              }} isLoading={loadingChat} />}
            </div>
          </CollapsibleMemoriesPanel>
      </div>
      <div style={{
        position: "absolute",
        bottom: "50px",
        right: "50%",
        display: "flex",
        transform: "translateX(50%)",
      }}>
        <div
          style={{
            zIndex: 2,
            "backdrop-filter": "blur(8px)",
            "-webkit-backdrop-filter": "blur(8px)",
            background: 'rgba(0, 0, 0, 0.25)',
            border: "1px solid rgba(255, 255, 255, 0.4)",
            borderRadius: "46px",
            color: "white",
            textAlign: "center",
            minWidth: 140,
            minHeight: 60
          }}
        >
          {!isWsOpen && <div style={{
              width: '100%',
              height: '100%',
              alignItems: 'center',
              justifyItems: 'center',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <LoadingDiv
                isLoading 
                duration={0.75} 
                width={`${25}px`}
                height={`${25}px`}
                borderWidth={1}
                loadingColor="#FFFFFF"
                borderColor="rgba(255, 255, 255, 0.5)"
                borderRadius={`${10}px`}
                backgroundColor="transparent"
                loadingSegmentPercentage={25}
              />
            </div>}
            {renderConversing()}
          </div>
          {conversing && (
            <div style={{
                "backdrop-filter": "blur(8px)",
                "-webkit-backdrop-filter": "blur(8px)",
                background: 'rgba(0, 0, 0, 0.25)',
                border: "1px solid rgba(255, 255, 255, 0.4)",
                padding: 12,
                borderRadius: 46,
                marginLeft: 10,
                cursor: "pointer"
              }}
              onClick={handleDisconnect}
            >
              <LoadingDiv
                isLoading={disconnecting} 
                duration={0.75}
                width={`${46}px`}
                height={`${46}px`}
                borderWidth={1}
                loadingColor="#FFFFFF"
                borderColor="rgba(255, 255, 255, 0.5)"
                borderRadius={`${46}px`}
                backgroundColor="#ed7878"
                loadingSegmentPercentage={25}
              >
              <HiOutlinePhoneXMark style={{fontSize: 21}} />
            </LoadingDiv>
          </div>)}
        </div>
    </div>
  );
}

export default App;
