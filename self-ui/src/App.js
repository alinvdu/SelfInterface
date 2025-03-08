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
import { formatDateSeparator, formatDuration } from "./utils.js";
import MemoryCard from "./components/MemoryCard.js";
import Switch from "./components/Switch.js";

const api = "https://selfinterface-simple-env.up.railway.app";

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
  const convDetails = useRef(null);

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

  const handleClearMemories = () => {
    setMemories([]);
  };
  
  const handleClearChat = () => {
    setChat([]);
  };

  const handleDeleteMessage = async (message) => {
    // Update the chat state by removing the deleted message
    setChat(chat.filter(item => item.id !== message.id));
  };
  
  const handleDeleteMemory = async (memory) => {
    // Update the memories state by removing the deleted memory
    setMemories(memories.filter(m => 
      !(m.text === memory.text && m.category === memory.category)
    ));
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
  const [chatLoading, setChatLoading] = useState(true);

  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const isMobile = windowWidth < 786;
  const [isChatExpanded, toggleExpandChat] = useState(!isMobile)

  const [isMemoryEnabled, setIsMemoryEnabled] = useState(true);
  const [isChatEnabled, setIsChatEnabled] = useState(true);
  const [isTogglingMemory, setIsTogglingMemory] = useState(false);
  const [isTogglingChat, setIsTogglingChat] = useState(false);
  const [loadingPreferences, setLoadingPreferences] = useState(true);

  const processPhoneCallEvents = (messages) => {
    const processed = [];
    let i = 0;
  
    while (i < messages.length) {
      // Check if the current message is a "Phone call started" event
      if (messages[i].type === "CONVERSATION_EVENT" && messages[i].content === "Phone call started") {
        let j = i + 1;
  
        // Look for the next "Phone call ended" event
        while (j < messages.length && !(messages[j].type === "CONVERSATION_EVENT" && messages[j].content === "Phone call ended")) {
          j++;
        }
  
        if (j < messages.length) {
          // Found a matching "Phone call ended" event
          const startTime = messages[i].timestamp;
          const endTime = messages[j].timestamp;
          const duration = endTime - startTime;
  
          // Create a new event with start timestamp and duration
          processed.push({
            id: messages[i].id,
            type: "CONVERSATION_EVENT",
            content: `Phone call duration: ${duration} seconds`,
            timestamp: startTime,
            duration: formatDuration(duration)
          });
  
          // Skip past the "Phone call ended" event
          i = j + 1;
        } else {
          // No "Phone call ended" found, keep the "Phone call started" event as is
          processed.push(messages[i]);
          i++;
        }
      } else {
        // Not a "Phone call started" event, add the message as is
        processed.push(messages[i]);
        i++;
      }
    }
  
    return processed;
  };

  // Add this function to the App component
  const fetchConversationHistory = async () => {
    if (!token || !user) return;
  
    try {
      const res = await fetch(`${api}/user_conversations`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
  
      // Extract messages from the response
      const messages = data.messages || [];

      const processedMessages = processPhoneCallEvents(messages);
  
      // Process messages to add date separators
      const allMessages = [];
      let currentDate = null;
  
      processedMessages.forEach(message => {
        const messageDate = new Date(message.timestamp * 1000);
        const year = messageDate.getFullYear();
        const month = (messageDate.getMonth() + 1).toString().padStart(2, '0');
        const day = messageDate.getDate().toString().padStart(2, '0');
        const dateStr = `${year}-${month}-${day}`;
  
        // If the date has changed (or it's the first message), add a date separator
        if (dateStr !== currentDate) {
          currentDate = dateStr;
          allMessages.push({
            type: 'DATE_SEPARATOR',
            content: formatDateSeparator(messageDate), // e.g., "2025-02-01"
            timestamp: message.timestamp // Use the timestamp of the first message of this day
          });
        }
  
        // Add the original message
        allMessages.push(message);
      });
  

      setChatLoading(false);
      // Set the chat with the processed messages, even if empty
      setChat(allMessages);
    } catch (error) {
      console.error("Error fetching conversation history:", error);
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

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

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
        toggleExpandChat(true);
      } else if (message.type === "CONV_START") {
        convDetails.current = message.timestamp
      } else if (message.type === "CONV_END") {
        const duration = message.timestamp - convDetails.current
        setChat(chat => [...chat, {
          type: "CONVERSATION_EVENT",
          content: `Phone call duration: ${duration || 0} seconds`,
          timestamp: convDetails.current,
          duration: formatDuration(duration || 0)
        }])
        convDetails.current = null 
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
            // Start the new session API call
          const newSessionPromise = fetch(api + "/new_session", {
            headers: { Authorization: `Bearer ${token}` },
          }).then(res => res.json());

            // Start fetchConversationHistory (assumed to be async and handle its own state)
            const historyPromise = token ? fetchConversationHistory() : Promise.resolve();

            // Wait for both to complete
            const [newSessionData] = await Promise.all([newSessionPromise, historyPromise]);

            if (token) {
              fetchConversationHistory();
            } else {
              setChatLoading(false);
            }

            // Set sessionId and create WebSocket connection
            setSessionId(newSessionData.session_id);
            createAndConnectWs(newSessionData.session_id, token);
          } catch (error) {
            console.error("Error creating session and proactive message:", error);
          }
        }
      };
      createSession();
    }
  }, [loading, token]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        if (token) {
          fetchMemories();

          const prefsRes = await fetch(`${api}/user_preferences`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          const prefsData = await prefsRes.json();

          setIsMemoryEnabled(prefsData.memory_enabled !== false); // Default to true if not set
          setIsChatEnabled(prefsData.chat_enabled !== false); // Default to true if not set
          setLoadingPreferences(false);
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };
  
    if (token) {
      fetchData();
    }
  }, [token]);

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

  const handleMemoryToggle = async (value, ntoken) => {
    if (ntoken) {
      setIsTogglingMemory(true);
      try {
        const response = await fetch(`${api}/update_preferences`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${ntoken}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ 
            memory_enabled: value 
          })
        });
        
        if (response.ok) {
          setIsMemoryEnabled(value);
        }
      } catch (error) {
        console.error("Error updating memory preference:", error);
      } finally {
        setIsTogglingMemory(false);
      }
    }
  };
  
  const handleChatToggle = async (value, ntoken) => {
    if (ntoken) {
      setIsTogglingChat(true);
      try {
        const response = await fetch(`${api}/update_preferences`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${ntoken}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ 
            chat_enabled: value 
          })
        });
        
        if (response.ok) {
          setIsChatEnabled(value);
        }
      } catch (error) {
        console.error("Error updating chat preference:", error);
      } finally {
        setIsTogglingChat(false);
      }
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
          toggleExpandChat(false)
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
          height: 35,
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-start"
        }}>
          <img style={{ width: 50 }} src="selfai-logo.png" />
          <span style={{marginLeft: 10}}>Self AI</span>
      </div>
      <div
        style={{
          position: "absolute",
          top: "110px",
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
        <LoginButton isMobile={isMobile} />
      </div>}
      <div style={{
        position: "absolute",
        top: isMobile ? "150px" : "100px",
        left: "10px",
        bottom: isMobile ? "150px" : "20px",
      }}>
          <CollapsibleMemoriesPanel
            token={token}
            requiresAccount
            memories={memories}
            MemoryCard={MemoryCard}
            title="Memories"
            onClear={handleClearMemories}
            api={api}
            toggleComponent={token &&
              <Switch
                isChecked={isMemoryEnabled}
                onChange={(value) => handleMemoryToggle(value, token)}
                isDisabled={isTogglingMemory}
                isLoading={loadingPreferences}
              />
            }
          >
            {memories &&
                memories.map((memory, i) => (
                  <MemoryCard key={i} memory={memory} token={token} api={api} onDelete={handleDeleteMemory} />
                ))}
          </CollapsibleMemoriesPanel>
          <CollapsibleMemoriesPanel
            memories={[]}
            MemoryCard={() => {}}
            title="Chat"
            expanded={isChatExpanded}
            toggleExpanded={() => {
              if (!conversing && !calling) {
                toggleExpandChat(prev => !prev)
              }
            }}
            onClear={handleClearChat}
            api={api}
            token={token}
            toggleComponent={token &&
              <Switch 
                isChecked={isChatEnabled}
                onChange={value => handleChatToggle(value, token)}
                isDisabled={isTogglingChat}
                isLoading={loadingPreferences}
              />
            }
            toggleLabel="Save"
          >
            <div style={{
              position: "relative",
              width: "100%",
              height: "100%",
              minHeight: 20,
              display: "flex"
            }}>
              <div style={{
                position: "absolute",
                width: "100%",
                height: 1,
                background: "rgba(255, 255, 255, 0.25)",
                display: "flex"
              }} />
              {chatLoading || !isWsOpen ?
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
              }} isLoading={loadingChat} token={token} api={api} onDeleteMessage={handleDeleteMessage} />}
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
