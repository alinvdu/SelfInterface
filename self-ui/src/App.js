import React, { useState, useEffect, useRef, Suspense, memo } from "react";
import "./App.css";
import { useAuth } from "./auth/AuthContext";
import { GiBrain } from "react-icons/gi";
import { FiBox } from "react-icons/fi";
import { IoCloseCircleOutline } from "react-icons/io5";

import { motion, AnimatePresence } from "framer-motion"; // You'll need to install framer-motion
import { RxAvatar } from "react-icons/rx";

// React Three Fiber imports
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import { HiOutlinePhone, HiOutlinePhoneXMark } from "react-icons/hi2";
import { RiVoiceprintFill } from "react-icons/ri";
import { LuBrainCog } from "react-icons/lu";

import Model from "./Model.js";
// import Model from "./ModelAnimation.js";

import LoadingDiv from "./components/LoadingDiv";
import CollapsibleMemoriesPanel from "./components/CollapsiblePanel.js";
import Chat from "./components/Chat.js";
import { formatDateSeparator, formatDuration, needsTodaySeparator } from "./utils.js";
import MemoryCard from "./components/MemoryCard.js";
import Switch from "./components/Switch.js";

import { FiUser } from "react-icons/fi";
import LoadingDots from "./components/LoadingDots.js";
import { useProgress } from "@react-three/drei";
import Overlay from "./components/Overlay.js";
import { BsArrowRight } from "react-icons/bs";
import PrivacyPolicy from "./PrivacyPolicy.js";

const WS_RECONNECT_TIMEOUT = 1500

const api = "https://selfai.live";

const AVAILABLE_MODELS = [
  {
    id: "ft:gpt-4o-mini-2024-07-18:personal::BANPHZFe",
    name: "Atlas",
    description: "Calm, empathetic and attentive. Great for a psychologist."
  },
  {
    id: "ft:gpt-4o-mini-2024-07-18:personal::B3Ti7zzf",
    name: "Unfiltered Friend",
    description: "Empathetic, truthful and funny. Might say the wrong thing!"
  }
];

const DEFAULT_MODEL = AVAILABLE_MODELS[0];

function ModelLoader() {
  return (
    <div className="model-loader-container">
      <div className="model-loader-content">
      <div style={{
        position: "relative",
        display: "flex",
        flexDirection: "row",
        alignItems: "center"
      }}>
        <LoadingDiv
            isLoading 
            duration={0.75} 
            width={`${32}px`}
            height={`${32}px`}
            borderWidth={2}
            loadingColor="#FFFFFF"
            borderColor="rgba(255, 255, 255, 0.5)"
            borderRadius={`${10}px`}
            backgroundColor="transparent"
            loadingSegmentPercentage={25}
        />
        <span style={{
          fontSize: 17,
          marginLeft: 15
        }}>Loading Graphics...</span>
      </div>
      </div>
    </div>
  );
}

function BackgroundScene({ 
  isTalking, 
  assistantTalking, 
  visemeSequence, 
  currentEmote, 
  setCurrentEmote, 
  isIntroMode = false,
  onModelLoaded,
  isModelVisible=true,
  isSmallSize=false
}) {
  return (
    <Canvas
      style={{
        position: "fixed",
        top: 0,
        right: 0,
        width: isIntroMode ? isSmallSize ? "0%" : "45%" : "100%",
        height: "100%",
        background: "transparent",
        zIndex: 0,
        pointerEvents: "auto",
      }}
      camera={{ 
        position: [0.09, 0.012, 0], // Adjust camera position for intro mode
        fov: 60, 
        near: 0.001 
      }}
      gl={{ alpha: true }}
    >
      <ambientLight color="#ffffff" intensity={isIntroMode ? 0.62 : 0.72} />
      <directionalLight color="#ffffff" position={[10, 7, 2]} intensity={isIntroMode ? 2.95 : 3.2} />
      <Suspense fallback={null}>
        {isModelVisible && (
          <Model 
            isPlaying={isTalking} 
            assistantTalking={assistantTalking} 
            visemeSequence={visemeSequence} 
            currentEmote={currentEmote} 
            setCurrentEmote={setCurrentEmote} 
            introPosition={isIntroMode}
            onLoad={onModelLoaded}
          />
        )}
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
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  
  const handleModelLoaded = () => {
    setModelLoaded(true);
    setModelLoading(false);
  };

  // Audio context and oscillator refs
  const audioContextRef = useRef(null);

  const peerConnectionRef = useRef(null);
  const wsRef = useRef(null); // WebSocket for signaling
  const analyserRef = useRef(null); // For audio analysis
  const convDetails = useRef(null);

  const assistantTalkingRef = useRef(null);

  // Initialize WebRTC connection
  const initiateWebRTC = async (withCamera = false) => {
    try {
      const localStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: withCamera
      });

      if (withCamera) {
        const videoElement = document.getElementById('user-video');
        if (videoElement) {
          videoElement.srcObject = localStream;
        }
      }

      peerConnectionRef.current = new RTCPeerConnection({
        // iceTransportPolicy: "relay",
        iceServers: [
          {
            urls: "turn:standard.relay.metered.ca:80?transport=tcp",
            username: process.env.REACT_APP_TURN_SERVER_USERNAME,
            credential: process.env.REACT_APP_TURN_SERVER_CREDENTIAL,
          },
          {
            urls: "turns:standard.relay.metered.ca:443?transport=tcp",
            username: process.env.REACT_APP_TURN_SERVER_USERNAME,
            credential: process.env.REACT_APP_TURN_SERVER_CREDENTIAL,
          },
      ],
      });

      localStream.getTracks().forEach(track => {
        peerConnectionRef.current.addTrack(track, localStream);
      });

      function removeRtxCodecs(sdp) {
        const lines = sdp.split('\r\n');
        const rtxPayloadTypes = [];
    
        // First, identify all RTX payload types
        lines.forEach(line => {
            if (line.includes('a=rtpmap') && line.includes('rtx/')) {
                const match = line.match(/a=rtpmap:(\d+) rtx\/\d+/);
                if (match && match[1]) {
                    rtxPayloadTypes.push(match[1]);
                }
            }
        });
    
        const filteredLines = lines.filter(line => {
            // Remove RTX related lines
            for (let payloadType of rtxPayloadTypes) {
                if (line.startsWith(`a=rtpmap:${payloadType}`) ||
                    line.startsWith(`a=fmtp:${payloadType}`) ||
                    line.startsWith(`a=rtcp-fb:${payloadType}`)) {
                    return false;
                }
            }
            return true;
        });
    
        // Now, remove RTX payload types from the m=video line
        const finalLines = filteredLines.map(line => {
            if (line.startsWith('m=video')) {
                const parts = line.split(' ');
                const mLine = parts.slice(0, 3); // "m=video", port, protocol
                const payloads = parts.slice(3).filter(pt => !rtxPayloadTypes.includes(pt));
                return [...mLine, ...payloads].join(' ');
            }
            return line;
        });
    
        return finalLines.join('\r\n');
    }

      peerConnectionRef.current.addTransceiver('audio', { direction: 'recvonly' });
      const offer = await peerConnectionRef.current.createOffer();
      offer.sdp = removeRtxCodecs(offer.sdp);
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
        const debounceTime = 0; // 150ms to smooth out word-by-word toggling
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
                  if (!assistantTalkingRef.current) {
                    setAssistantTalking(true);
                  }
                  if (silenceTimeout) {
                      clearTimeout(silenceTimeout);
                      silenceTimeout = null;
                  }
              } else {
                  // Delay stopping to bridge short gaps
                  if (!silenceTimeout) {
                      silenceTimeout = setTimeout(() => {
                          setIsTalking(isActive);
                      }, 120);
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

  const { token, user, loading, signInWithGoogle, logout } = useAuth()
  // const { token, user, loading } = {
  //   token: null,
  //   user: null,
  //   loading: false
  // }
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
  const [currentEmote, setCurrentEmote] = useState(null);

  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const isMobile = windowWidth < 786;
  const isSmallSize = windowWidth < 1200;
  const smallerThan850 = windowWidth < 850;
  const [isChatExpanded, toggleExpandChat] = useState(!isMobile)
  const [isMemoryExpanded, toggleMemoryExpanded] = useState(false)

  const [isMemoryEnabled, setIsMemoryEnabled] = useState(true);
  const [isChatEnabled, setIsChatEnabled] = useState(true);
  const [isTogglingMemory, setIsTogglingMemory] = useState(false);
  const [isTogglingChat, setIsTogglingChat] = useState(false);
  const [loadingPreferences, setLoadingPreferences] = useState(true);
  const [userVoiceMessage, setUserVoiceMessage] = useState("");
  // const [assistantMessage, setAssistantVoiceMessage] = useState("I can see from your facial expressions and your voice that you are in a good mood! I think it would be perfect for us to explore ");
  const [assistantMessage, setAssistantVoiceMessage] = useState(null);
  const [latestMessageType, setLatestMessageType] = useState(null);

  const [userMessageKey, setUserMessageKey] = useState(0);
  const [assistantMessageKey, setAssistantMessageKey] = useState(0);
  const [isCallDropdownVisible, setIsCallDropdownVisible] = useState(false);
  const callDropdownRef = useRef(null);
  const [showVideo, setShowVideo] = useState(false);
  const [assistantTalking, setAssistantTalking] = useState(false);
  const [visemes, setVisemes] = useState();

  const modelDropdownRef = useRef(null);
  const [isModelVisible, setIsModelVisible] = useState(true);

  const [showIntroMode, setShowIntroMode] = useState(false);
  const [showLoginView, setShowLoginView] = useState(false);
  const [showCreateAccount, setShowCreateAccount] = useState(false);

  const [showPrivacyPolicyDialog, setShowPrivacyPolicyDialog] = useState(false);

  const handleStartApp = () => {
    setIsModelVisible(false)
    setTimeout(() => {
      setShowIntroMode(false);
    }, 50)
    setTimeout(() => {
      setIsModelVisible(true)
    }, 200);
  };

  useEffect(() => {
    assistantTalkingRef.current = assistantTalking
  }, [assistantTalking])

  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState(() => {
    const savedModel = localStorage.getItem("selectedModel");
    if (savedModel) {
      try {
        return JSON.parse(savedModel);
      } catch (e) {
        return DEFAULT_MODEL;
      }
    }
    return DEFAULT_MODEL;
  });

    // Add this with your other useEffect hooks
  useEffect(() => {
    // Function to handle clicks outside the dropdown
    function handleClickOutside(event) {
      if (modelDropdownRef.current && 
          !modelDropdownRef.current.contains(event.target) && 
          isModelDropdownOpen) {
        setIsModelDropdownOpen(false);
      }
    }
    
    // Add event listener when dropdown is open
    if (isModelDropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    
    // Clean up the event listener
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isModelDropdownOpen]); // Re-run effect when dropdown state changes

  useEffect(() => {
    // Function to handle clicks outside the dropdown
    function handleClickOutside(event) {
      if (modelDropdownRef.current && 
          !modelDropdownRef.current.contains(event.target) && 
          isModelDropdownOpen) {
        setIsModelDropdownOpen(false);
      }
    }
    
    // Add event listener when dropdown is open
    if (isModelDropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    
    // Clean up the event listener
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isModelDropdownOpen]); // Re-run effect when dropdown state changes

  useEffect(() => {
    const handleClickOutside = (event) => {
      // Only close if dropdown is open and click is outside dropdown and button
      if (isCallDropdownVisible && 
          callDropdownRef.current && 
          !callDropdownRef.current.contains(event.target) &&
          !event.target.closest('.lets-connect-button')) {
        setIsCallDropdownVisible(false);
      }
    };
    
    // Add event listener when dropdown is open
    if (isCallDropdownVisible) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    
    // Clean up
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isCallDropdownVisible]);
  
  const handleModelSelect = (model) => {
    setSelectedModel(model);
    localStorage.setItem("selectedModel", JSON.stringify(model));
    setIsModelDropdownOpen(false);

    window.location.reload();
  };

  const prevUserMessageRef = useRef(null);
  const prevAssistantMessageRef = useRef(null);
  const wsReconnectRef = useRef(null);
  const manualCloseRef = useRef(false);

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
      } else if (message.type === "ice-candidate") {
        await peerConnectionRef.current.addIceCandidate(
          new RTCIceCandidate(message.candidate)
        );
      } else if (message.type === "PROCESSING") {
        setProcessing(true);
      } else if (message.type === "FINISHED_PROCESSING") {
        setProcessing(false);
      } else if (message.type === "CHAT_MESSAGE") {
        setChat(chat => {
          // Check if we need to add a Today separator
          const updatedChat = [...chat];
          const currentTimestamp = Math.floor(Date.now() / 1000); // Current time in seconds
          
          // If we need a "Today" separator, add it
          if (needsTodaySeparator(updatedChat)) {
            updatedChat.push({
              type: 'DATE_SEPARATOR',
              content: 'Today',
              timestamp: currentTimestamp
            });
          }
          
          // Add the new message
          updatedChat.push({
            "role": "assistant",
            "content": message.message,
            "timestamp": currentTimestamp
          });
          
          return updatedChat;
        });
        setLoadingChat(false)
      } else if (message.type === "rtc_disconnected") {
        setDisconnecting(false);
        setConversing(false);
        setPhoneCalling(false);
        setIsTalking(false);
        setProcessing(false);
        toggleExpandChat(true);
        setAssistantVoiceMessage(null);
        setUserVoiceMessage(null);
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
      } else if (message.type === "user_voice_message") {
        // Check if the message is actually new by comparing with the ref
        if (message.text !== prevUserMessageRef.current) {
          setUserMessageKey(prev => prev + 1);
          prevUserMessageRef.current = message.text;
        }
        setUserVoiceMessage(message.text);
        setLatestMessageType('user');
        setTimeout(() => {
          setAssistantVoiceMessage("...")
          if ("..." !== prevAssistantMessageRef.current) {
            setAssistantMessageKey(prev => prev + 1);
            prevAssistantMessageRef.current = "...";
          }
          setLatestMessageType('assistant');
        }, 200)
      } else if (message.type === "assistant_voice_message") {
        // Check if the message is actually new by comparing with the ref
        if (message.text !== prevAssistantMessageRef.current) {
          setAssistantMessageKey(prev => prev + 1);
          setVisemes(message.visemes)
          console.log('visemes are:', message.visemes)
          prevAssistantMessageRef.current = message.text;
        }
        setAssistantVoiceMessage(message.text);
        setLatestMessageType('assistant');
      } else if (message.type === "voice_message_start") {
        // Check if "..." is actually new
        if ("..." !== prevUserMessageRef.current) {
          setUserMessageKey(prev => prev + 1);
          prevUserMessageRef.current = "...";
        }
        setUserVoiceMessage("...");
        setLatestMessageType('user');
      } else if (message.type === "SESSION_NOT_FOUND") {
        console.log("Session not found, creating a new one");
        clearTimeout(wsReconnectRef.current);
        manualCloseRef.current = true;
        
        // Create a new session
        const res = await fetch(api + "/new_session", {
          headers: { Authorization: `Bearer ${token}` },
        });
        
        const newSessionData = await res.json();
        setSessionId(newSessionData.session_id);
        
        // Close existing WebSocket
        if (wsRef.current) {
          wsRef.current.close();
        }

        // Create a new WebSocket connection with the new session ID
        createAndConnectWs(newSessionData.session_id, token);
      } else if (message.type === "START_TALK") {
        setAssistantTalking(true);
      } else if (message.type === "FINISHED_TALK") {
        setAssistantTalking(false);
        if (message.emote_type) {
          setCurrentEmote(message.emote_type);
        }
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
      if (!manualCloseRef.current) {
        wsReconnectRef.current = setTimeout(() => {  
          createAndConnectWs(currentSessionId, currentToken)
        }, WS_RECONNECT_TIMEOUT);
      } else {
        manualCloseRef.current = false; // Reset flag
      }
    }
  }

  // Combined new_session and proactive message call.
  useEffect(() => {
    if (!loading) {
      if (!token) {
        setShowIntroMode(true);
      }

      const createSession = async () => {
        if (!sessionId) {
          try {
            // Start the new session API call
          const newSessionPromise = fetch(api + `/new_session?model_version=${encodeURIComponent(selectedModel.id)}`, {
            headers: { Authorization: `Bearer ${token}` },
          }).then(res => res.json());

            // Start fetchConversationHistory (assumed to be async and handle its own state)
            const historyPromise = token ? fetchConversationHistory() : Promise.resolve();

            // Wait for both to complete
            const [newSessionData] = await Promise.all([newSessionPromise, historyPromise]);

            setChatLoading(false);

            // Set sessionId and create WebSocket connection
            setSessionId(newSessionData.session_id);

            await new Promise(resolve => setTimeout(resolve, 100));

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

    return <RiVoiceprintFill style={{fontSize: 21}} />
  }

  const handleDisconnect = async () => {
    if (!sessionId) return;

    setShowVideo(false);
    
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
        <div 
          className="lets-connect-button"
          style={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          padding: isMobile ? 6 : 12,
          position: 'relative' // Added for dropdown positioning
        }} onClick={(e) => {
          e.stopPropagation(); // Prevent body click from closing it immediately
          setIsCallDropdownVisible(true); // Show dropdown instead of dialog
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
          <div style={{ marginLeft: isMobile ? 7 : "1rem", marginRight: "0.5rem", fontSize: isMobile ? "15px" : "18px" }}>{calling ? "Calling Atlas..." : "Let's Connect"}</div>
          
          {/* Call Options Dropdown */}
          {isCallDropdownVisible && (
            <div 
              ref={callDropdownRef}
              className="callOptionsDropdown"
              onClick={(e) => e.stopPropagation()} // Prevent clicks from reaching document
              style={{
                position: 'absolute',
                bottom: '100%',
                marginBottom: '12px',
                width: '280px',
                background: isMobile ? 'rgba(83, 83, 83, 0.85)' : 'rgba(0, 0, 0, 0.25)',
                backdropFilter: 'blur(8px)',
                WebkitBackdropFilter: 'blur(8px)',
                border: '1px solid rgba(255, 255, 255, 0.4)',
                borderRadius: '16px',
                padding: '12px',
                zIndex: 10,
                animation: 'fadeInUp 0.3s ease-out forwards', // Animation
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)'
              }}
            >
              {/* Audio Call Option */}
              <div 
                style={{
                  padding: '12px 16px',
                  borderRadius: '12px',
                  marginBottom: '8px',
                  cursor: 'pointer',
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  flexDirection: 'column'
                }}
                onClick={(e) => {
                  e.stopPropagation(); // Prevent triggering parent onClick
                  setIsCallDropdownVisible(false);
                  toggleExpandChat(false);
                  setPhoneCalling(true);
                  initiateWebRTC(false); // Audio call
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.5)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                }}
              >
                <div style={{ fontWeight: '600', marginBottom: '4px', fontSize: '16px' }}>Audio Call</div>
                <div style={{ fontSize: '14px', opacity: 0.85 }}>Real time voice sentiment emotion recognition</div>
              </div>
              
              {/* Video Call Option */}
              <div 
                style={{
                  padding: '12px 16px',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  flexDirection: 'column'
                }}
                onClick={(e) => {
                  e.stopPropagation(); // Prevent triggering parent onClick
                  setIsCallDropdownVisible(false);
                  toggleExpandChat(false);
                  setPhoneCalling(true);
                  initiateWebRTC(true); // Video call
                  setShowVideo(true);
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.5)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                }}
              >
                <div style={{ fontWeight: '600', marginBottom: '4px', fontSize: '16px' }}>Video Call</div>
                <div style={{ fontSize: '14px', opacity: 0.85 }}>Real time video and voice sentiment emotion recognition</div>
              </div>
            </div>
          )}
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

  const CHAT_CHAR_DISPLAY = isMobile ? 125 : isSmallSize ? 300 : 700

  const renderUserMessage = (key) => (
    <div style={{
      display: "flex",
      maxWidth: "75%",
      alignSelf: "flex-start",
      marginTop: 10
    }}
    key={`user-${key}`}
    className="userSlideIn"
    >
      <div style={{
        display: 'flex',
        alignItems: "center",
        justifyContent: "center",
        width: 40,
        height: 40,
        borderRadius: "50%",
        border: "1px solid white",
        marginRight: 8,
        backgroundColor: 'rgba(0, 0, 0, 0.25)',
        color: 'white',
        border: '1px solid rgba(255, 255, 255, 0.4)',
        "backdrop-filter": "blur(8px)",
        "-webkit-backdrop-filter": "blur(8px)",
      }}>
        <FiUser fontSize={21} style={{marginTop: -1}} />
      </div>
      <div style={{
        backgroundColor: 'rgba(0, 0, 0, 0.25)',
        color: 'white',
        border: '1px solid rgba(255, 255, 255, 0.4)',
        "backdrop-filter": "blur(8px)",
        "-webkit-backdrop-filter": "blur(8px)",
        padding: "6px 8px",
        borderRadius: 8,
        maxWidth: "70%",
        fontSize: isMobile ? 14 : 15
      }}
      title={userVoiceMessage}
      >
        {userVoiceMessage === "..." ? <LoadingDots size={4} /> : userVoiceMessage.length > CHAT_CHAR_DISPLAY ? userVoiceMessage.substring(0, CHAT_CHAR_DISPLAY) + '...' : userVoiceMessage}
      </div>
    </div>
  );

  const getStylesForVideo = () => {
    const commonStyles = {
      position: "absolute",
      borderRadius: "8px",
      overflow: "hidden",
      border: "1px solid rgba(255, 255, 255, 0.6)",
      backdropFilter: "blur(4px)",
      WebkitBackdropFilter: "blur(4px)",
      background: 'rgba(0, 0, 0, 0.3)'
    }

    if (isMobile) {
      return {
        ...commonStyles,
        bottom: 200,
        left: 20,
        width: 120,
        height: 120
      }
    }

    if (windowWidth < 1200 && windowWidth > 900) {
      return {
        ...commonStyles,
        top: 100,
        right: 20,
        width: 280,
        height: 170
      }
    }

    if (windowWidth < 900) {
      return {
        ...commonStyles,
        top: 240,
        left: 20,
        width: 280,
        height: 170
      }
    }

    return {
      ...commonStyles,
      bottom: 280,
      right: 250,
      width: 280,
      height: 170
    }
  }
  // Function to render assistant message
  const renderAssistantMessage = (key) => (
    <div style={{
      display: "flex",
      maxWidth: "75%",
      alignSelf: "flex-end",
      marginTop: 10
    }}
    key={`assistant-${key}`}
    className="assistantSlideIn"
    >
      <div style={{
        backgroundColor: 'rgba(255, 255, 255, 0.35)',
        border: '1px solid rgba(255, 255, 255, 0.3)',
        "backdrop-filter": "blur(8px)",
        "-webkit-backdrop-filter": "blur(8px)",
        padding: "6px 8px",
        borderRadius: 8,
        flex: 1,
        color: "rgba(0, 0, 0, 0.65)",
        fontSize: isMobile ? 14 : 15
      }}
      title={assistantMessage}
      >
        {assistantMessage === "..." ? <LoadingDots size={4} /> : assistantMessage.length > CHAT_CHAR_DISPLAY ? assistantMessage.substring(0, CHAT_CHAR_DISPLAY) + '...' : assistantMessage}
      </div>
      <img style={{
        width: 40,
        height: 40,
        borderRadius: "50%",
        border: "1px solid white",
        marginLeft: 8
      }} src={`${process.env.PUBLIC_URL}/assets/atlas-avatar.png`} />
    </div>
  );

  return (
    <div
      className="App"
      style={{
        position: "relative",
        background: "transparent",
        height: "100vh",
      }}
    >
      {modelLoading && <ModelLoader />}
      {modelLoaded && (
        <AnimatePresence>
          {showIntroMode && <Overlay smallerThan850={smallerThan850} isSmallSize={isSmallSize} token={token} showCreateAccount={showCreateAccount} signInWithGoogle={signInWithGoogle} showLoginView={showLoginView} handleStartApp={handleStartApp} toggleLoginView={() => {
            setShowCreateAccount(false)
            setShowLoginView(true)
          }} toggleCreateAccountView={() => {
            setShowLoginView(false)
            setShowCreateAccount(true)
          }} navigateBack={() => {
            setShowCreateAccount(false);
            setShowLoginView(false);
          }}
          setShowPrivacyPolicyDialog={setShowPrivacyPolicyDialog}/>}
        </AnimatePresence>
      )}
      <BackgroundScene isSmallSize={isSmallSize} isTalking={isTalking} assistantTalking={assistantTalking} visemeSequence={visemes} currentEmote={currentEmote} setCurrentEmote={setCurrentEmote} isIntroMode={showIntroMode} onModelLoaded={handleModelLoaded} isModelVisible={isModelVisible} />
      {showIntroMode && !showLoginView && !showCreateAccount && !modelLoading ? !token ? <div style={{
        position: "absolute",
        top: smallerThan850 ? 28 : 35,
        right: smallerThan850 ? 15 : 50,
        fontSize: 19,
        color: 'rgba(255, 255, 255, 1)',
        opacity: 0.85,
        cursor: 'pointer'
      }}
      onMouseOver={e => {
        e.currentTarget.style.opacity = 1
        e.currentTarget.style.transform = "scale(1.05)";
      }}
      onMouseLeave={e => {
        e.currentTarget.style.opacity = 0.85
        e.currentTarget.style.transform = "scale(1)";
      }}
      onClick={() => {
        setShowLoginView(true)
      }}
      >
        Log In
      </div> : <div style={{
        position: "absolute",
        top: smallerThan850 ? 22 : 35,
        right: 50,
        fontSize: 19,
        color: 'rgba(255, 255, 255, 1)',
        opacity: 0.85,
        cursor: 'pointer',
        display: "flex"
      }}
      >
        {!smallerThan850 && 
        <><div style={{
          marginRight: 15
        }}>
          Hello {user.displayName || user.email}
        </div>
        <div style={{
          marginRight: 15
        }}>
          |
        </div></>}
        <div style={{
          
        }}
        onMouseOver={e => {
          e.currentTarget.style.opacity = 1
          e.currentTarget.style.transform = "scale(1.05)";
        }}
        onMouseLeave={e => {
          e.currentTarget.style.opacity = 0.85
          e.currentTarget.style.transform = "scale(1)";
        }}
        onClick={logout}
        >
        Log Out
        </div>
        <div style={{
          marginLeft: 25,
          display: "flex",
          alignItems: "center"
        }}
        onMouseOver={e => {
          e.currentTarget.style.opacity = 1
          e.currentTarget.style.transform = "scale(1.05)";
        }}
        onMouseLeave={e => {
          e.currentTarget.style.opacity = 0.85
          e.currentTarget.style.transform = "scale(1)";
        }}
        onClick={() => {
          setIsModelVisible(false)
          setTimeout(() => {
            setShowIntroMode(false);
          }, 50)
          setTimeout(() => {
            setIsModelVisible(true)
            }, 200);
        }}
        >
          <span>Back</span>
          <BsArrowRight style={{
            marginLeft: 5
          }} />
        </div>
      </div> : null}
      {/* <BackgroundScene isTalking={isTalking} assistantTalking={assistantTalking} /> */}
      {!showIntroMode && modelLoaded && (
      <>
        <div style={{
            position: "absolute",
            top: "16px",
            left: "16px",
            zIndex: 2,
            background: 'rgba(0, 0, 0, 0.25)',
            "backdrop-filter": "blur(8px)",
            "-webkit-backdrop-filter": "blur(8px)",
            border: "1px solid rgba(255, 255, 255, 0.35)",
            borderRadius: "21px",
            color: "white",
            fontSize: isMobile ? 18 : "23px",
            width: isMobile ? 120 : 155,
            height: isMobile ? 45 : 60,
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}>
            <GiBrain style={{
              fontSize: 38,
              color: "white",
              marginLeft: -5
            }} />
            <div style={{marginLeft: isMobile ? 5 : 10, display: "flex", flexDirection: "column", alignItems: 'flex-start'}}>
              <div>Self AI</div>
            </div>
        </div>
        <div
          ref={modelDropdownRef}
          style={{
            position: "absolute",
            top: isMobile ? "70px" : "85px",
            left: "50%",
            transform: !isMobile ? "translate(-50%, -35%)" : "translateX(-50%)",
            zIndex: 2,
            background: 'rgba(0, 0, 0, 0.25)',
            backdropFilter: "blur(8px)",
            WebkitBackdropFilter: "blur(8px)",
            border: "1px solid rgba(255, 255, 255, 0.35)",
            borderRadius: "26px",
            padding: isModelDropdownOpen ? "0.8rem" : isMobile ? "0px 12px" : "0.5rem 1.2rem",
            color: "white",
            fontSize: isMobile ? 15 : "1.2rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            transition: "all 0.3s ease-in-out",
            height: isModelDropdownOpen ? "auto" : "48px",
            flexDirection: isModelDropdownOpen ? "column" : "row",
            overflow: "hidden",
            maxHeight: isMobile && isModelDropdownOpen ? "calc(100vh - 200px)" : "none",
            width: isMobile && isModelDropdownOpen ? "80%" : "auto",
            zIndex: 99
          }}
          onClick={() => !isModelDropdownOpen && setIsModelDropdownOpen(true)}
        >
          {!isModelDropdownOpen ? (
            <>
              {selectedModel.name}
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
                style={{ 
                  marginLeft: "8px",
                  transition: "transform 0.2s ease",
                }}
              >
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </>
          ) : (
            <>
              <div style={{ 
                display: "flex", 
                justifyContent: "space-between", 
                width: "100%", 
                marginBottom: "10px",
                alignItems: "center"
              }}>
                <div style={{ fontSize: "1rem", fontWeight: "bold" }}>Model Selection</div>
                <div 
                  style={{ 
                    cursor: "pointer", 
                    fontSize: 18, 
                    width: 24, 
                    height: 24, 
                    display: "flex", 
                    alignItems: "center", 
                    justifyContent: "center",
                    borderRadius: "50%",
                    background: "rgba(255, 255, 255, 0.1)",
                    transition: "background-color 0.2s ease"
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsModelDropdownOpen(false);
                  }}
                  onMouseOver={(e) => e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.2)"}
                  onMouseOut={(e) => e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.1)"}
                >
                  ×
                </div>
              </div>
              <div style={{ maxHeight: "190px", overflowY: "auto", width: "100%" }}>
                {AVAILABLE_MODELS.map((model) => (
                  <div
                    key={model.id}
                    style={{
                      padding: "8px 10px",
                      margin: "3px 0",
                      borderRadius: "10px",
                      background: selectedModel.id === model.id ? "rgba(255, 255, 255, 0.95)" : "rgba(0, 0, 0, 0.2)",
                      border: selectedModel.id === model.id 
                        ? "1px solid rgba(255, 255, 255, 0.95)" 
                        : "1px solid rgba(255, 255, 255, 0.3)",
                      color: selectedModel.id === model.id ? "rgba(0, 0, 0, 0.8)" : "white",
                      cursor: "pointer",
                      transition: "all 0.2s ease",
                      display: "flex",
                      flexDirection: "column",
                      position: "relative",
                      alignItems: "flex-start",
                      boxShadow: selectedModel.id === model.id ? "0 2px 8px rgba(0, 0, 0, 0.1)" : "none",
                      marginBottom: 8
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleModelSelect(model);
                    }}
                    onMouseOver={(e) => {
                      if (selectedModel.id !== model.id) {
                        e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.5)";
                        e.currentTarget.style.background = "rgba(0, 0, 0, 0.3)";
                      }
                    }}
                    onMouseOut={(e) => {
                      if (selectedModel.id !== model.id) {
                        e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.3)";
                        e.currentTarget.style.background = "rgba(0, 0, 0, 0.2)";
                      }
                    }}
                  >
                    <div style={{ 
                      fontWeight: "500", 
                      display: "flex", 
                      alignItems: "center",
                      justifyContent: "space-between",
                      width: "100%",
                      fontSize: 16
                    }}>
                      {model.name}
                      {selectedModel.id === model.id && (
                        <svg 
                          xmlns="http://www.w3.org/2000/svg" 
                          width="16" 
                          height="16" 
                          viewBox="0 0 24 24" 
                          fill="none" 
                          stroke="currentColor" 
                          strokeWidth="2" 
                          strokeLinecap="round" 
                          strokeLinejoin="round"
                          style={{ color: "rgba(0, 0, 0, 0.7)" }}
                        >
                          <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                      )}
                    </div>
                    <div style={{ 
                      fontSize: "0.9rem", 
                      opacity: selectedModel.id === model.id ? 0.8 : 0.9,
                      maxWidth: 300,
                      paddingRight: 15,
                      textAlign: "left"
                    }}>
                      {model.description}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {!loading &&
        <div
          style={{
            position: "absolute",
            top: token ? "16px" : "22px",
            right: "16px",
            zIndex: 2,
            background: token ? 'rgba(0, 0, 0, 0.25)' : 'rgba(255, 255, 255, 0.85)',
            "backdrop-filter": "blur(15px)",
            "-webkit-backdrop-filter": "blur(15px)",
            border: "1px solid rgba(255, 255, 255, 0.3)",
            borderRadius: "26px",
            color: "white",
            fontSize: 17,
            height: token ? isMobile ? 47 : 62 : 45,
            boxSizing: "border-box",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            padding: "0 16px"
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "scale(1.05)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "scale(1)";
          }}
          onClick={() => {
            setIsModelVisible(false)
            setTimeout(() => {
              setShowIntroMode(true);
            }, 50)
            setTimeout(() => {
              setIsModelVisible(true)
            }, 200);

            if (!token) {
              setShowCreateAccount(true);
            }
          }}
        >
          {token && <RxAvatar style={{
            fontSize: 30,
            color: "white"
          }} />}
          {!token ? <div style={{ color: "black", fontSize: 15, fontWeight: "bold"}}>Create Account</div> : null}
        </div>}
        <div style={{
          position: "absolute",
          top: isMobile ? "140px" : "100px",
          left: "16px",
          bottom: isMobile ? "100px" : "20px",
          display: "flex",
          flexDirection: "column",
          minHeight: 0,
          marginTop: isMobile && !isChatExpanded && !isMemoryExpanded ? "50%" : 0
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
              expanded={isMemoryExpanded}
              toggleExpanded={() => {
                toggleMemoryExpanded(prev => !prev)
              }}
              extraStyles={{
                "maxHeight": isMobile && isChatExpanded ? "50%" : ""
              }}
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
              extraStyles={{
                "maxHeight": isMobile && isMemoryExpanded ? "50%" : ""
              }}
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
        {conversing &&
          <div style={{
            position: "absolute",
            bottom: isMobile ? 100 : "140px",
            right: "50%",
            display: "flex",
            transform: "translateX(50%)",
            display: "flex",
            flexDirection: "column",
            minWidth: isMobile ? 350 : 450,
            maxWidth: isMobile ? 350 : 700
          }}>
            {/* Render messages in the correct order based on which was received last */}
            {userVoiceMessage && assistantMessage ? (
              // Both messages exist
              latestMessageType === 'assistant' ? (
                // Assistant was the last to speak
                <>
                  {renderUserMessage(userMessageKey)}
                  {renderAssistantMessage(assistantMessageKey)}
                </>
              ) : (
                // User was the last to speak
                <>
                  {renderAssistantMessage(assistantMessageKey)}
                  {renderUserMessage(userMessageKey)}
                </>
              )
            ) : (
              // Only one message exists
              <>
                {assistantMessage && renderAssistantMessage(assistantMessageKey)}
                {userVoiceMessage && renderUserMessage(userMessageKey)}
              </>
            )}
          </div>}
        <div style={{
          position: "absolute",
          bottom: isMobile ? "20px" : "50px",
          right: "50%",
          display: "flex",
          transform: "translateX(50%)",
          zIndex: 3
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
          {showVideo && <div style={getStylesForVideo()}>
            <video
              id="user-video"
              autoPlay
              playsInline
              muted
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover"
              }}
            />
          </div>}
          </>)}
          {showIntroMode && <div style={{
            position: "absolute",
            bottom: 20,
            right: 20,
            background: "rgba(0, 0, 0, 0.5)",
            padding: "5px 10px",
            borderRadius: 8,
            fontSize: 14
          }}>
            © 2025 SelfAI.live • <span style={{
              cursor: "pointer"
            }} onClick={() => {
              setShowPrivacyPolicyDialog(true)
            }}>Privacy Policy</span>
          </div>}
          {showPrivacyPolicyDialog && <div style={{
            position: "absolute",
            left: 0,
            right: 0,
            bottom: 0,
            top: 0,
            zIndex: 99999,
            background: "rgba(0, 0, 0, 0.45",
            padding: 25,
            boxSizing: "border-box",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}>
            <div style={{
              background: "rgba(50, 50, 50, 1)",
              borderRadius: 16,
              maxWidth: 800,
              height: "100%",
              position: "relative",
              paddingTop: 40,
              boxSizing: "border-box",
              paddingRight: 40
            }}>
              <div style={{
                height: "100%",
                overflow: "auto"
              }}>
                <PrivacyPolicy />
              </div>
              <IoCloseCircleOutline onClick={() => {
                setShowPrivacyPolicyDialog(false);
              }} style={{
                fontSize: 32,
                position: "absolute",
                top: 10,
                right: 10,
                cursor: "pointer"
              }} />
            </div>
          </div>}
      </div>
  );
}

export default App;
