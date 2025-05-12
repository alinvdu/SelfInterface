import React, { useState, useEffect, useRef, Suspense, memo } from "react";
import "./App.css";
import { useAuth } from "./auth/AuthContext";
import { GiBrain } from "react-icons/gi";
import { FiBox } from "react-icons/fi";
import { IoCloseCircleOutline } from "react-icons/io5";

import { motion, AnimatePresence } from "framer-motion"; // You'll need to install framer-motion
import { RxAvatar } from "react-icons/rx";
import { FaCheck } from "react-icons/fa6";
import { PiShirtFoldedDuotone } from "react-icons/pi";

import env1Url from "./env1.jpg"

// React Three Fiber imports
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import { HiOutlinePhone, HiOutlinePhoneXMark } from "react-icons/hi2";
import { RiVoiceprintFill } from "react-icons/ri";
import { LuBrainCog } from "react-icons/lu";
import { IoIosInformationCircleOutline } from "react-icons/io";
import { LuMessagesSquare } from "react-icons/lu";

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
import Overlay from "./components/Overlay.js";
import { BsArrowRight } from "react-icons/bs";
import PrivacyPolicy from "./PrivacyPolicy.js";
import SaveDialog from "./components/SaveDialog.js";
import { IoIosLogOut } from "react-icons/io";
import { IoIosCloseCircleOutline } from "react-icons/io";
import MyPanelWithWaves from "./components/WavyPanel.js";
import EnvironmentModal from "./components/EnvironmentModal.js";

const WS_RECONNECT_TIMEOUT = 1500

const api = "http://localhost:8000";

function ModelLoader() {
  return (
    <div style={{
      width: "100%",
      height: "100%",
      position: "absolute",
      top: 0,
      left: 0
    }}>
      <div style={{
        position: "relative",
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        width: "100%",
        height: "100%",
        justifyContent: "center"
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
  currentMicroExpression,
  setCurrentMicroExpression,
  currentGesture,
  setCurrentGesture
}) {
  return (
    <Canvas
      style={{
        width: "100%",
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
            currentMicroExpression={currentMicroExpression}
            setCurrentMicroExpression={setCurrentMicroExpression}
            setCurrentGesture={setCurrentGesture}
            currentGesture={currentGesture}
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
  const apiSelectedModel = useRef(null);

  const peerConnectionRef = useRef(null);
  const wsRef = useRef(null);
  const analyserRef = useRef(null);
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
            urls: "stun:stun.relay.metered.ca:80",
          },
          {
            urls: "turn:standard.relay.metered.ca:80",
            username: process.env.REACT_APP_TURN_SERVER_USERNAME,
            credential: process.env.REACT_APP_TURN_SERVER_CREDENTIAL,
          },
          {
            urls: "turn:standard.relay.metered.ca:80?transport=tcp",
            username: process.env.REACT_APP_TURN_SERVER_USERNAME,
            credential: process.env.REACT_APP_TURN_SERVER_CREDENTIAL,
          },
          {
            urls: "turn:standard.relay.metered.ca:443",
            username: process.env.REACT_APP_TURN_SERVER_USERNAME,
            credential: process.env.REACT_APP_TURN_SERVER_CREDENTIAL,
          },
          {
            urls: "turns:standard.relay.metered.ca:443?transport=tcp",
            username: process.env.REACT_APP_TURN_SERVER_USERNAME,
            credential: process.env.REACT_APP_TURN_SERVER_CREDENTIAL,
          },
      ]
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
                  if (silenceTimeout) {
                      clearTimeout(silenceTimeout);
                      silenceTimeout = null;
                  }
              } else {
                  // Delay stopping to bridge short gaps
                  if (!silenceTimeout) {
                      silenceTimeout = setTimeout(() => {
                          setIsTalking(isActive);
                      }, 150);
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
  const [currentMicroExpression, setCurrentMicroExpression] = useState(null);
  const [currentGesture, setCurrentGesture] = useState(null);

  const [showModelSelectionScreen, setShowModelSelectionScreen] = useState(false);

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
  const [userVoiceMessage, setUserVoiceMessage] = useState(null);
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
  const [areEnvsLoaded, setAreEnvsLoaded] = useState(false);
  const [environments, setEnvironments] = useState([{
    name: "Cabinet"
  }]);

  const INTRO_MODE = "INTRO_MODE"
  const APP_MODE = "APP_MODE"
  const [mode, setMode] = useState(null);
  const showIntroMode = mode === INTRO_MODE
  const [showLoginView, setShowLoginView] = useState(false);
  const [showCreateAccount, setShowCreateAccount] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [showEnvironmentModal, setShowEnvironmentModal] = useState(false);

  const [showTipCard, setShowTipCard] = useState(false);
  const progressIntervalRef = useRef(null);
  const tipShownRef = useRef(false);

  const [isFirefox, setIsFirefox] = useState(false);
  const [showFirefoxTooltip, setShowFirefoxTooltip] = useState(false);

  const [isTranscriptionExpanded, setExpandTranscription] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [conversationToSave, setConversationToSave] = useState(null);

  const fullEmoteRef = useRef(null)
  const microEmoteRef = useRef(null)

  const [isUpdatingModel, setIsUpdatingModel] = useState(false);

  const updateModel = async (model) => {
    setIsUpdatingModel(true);
    try {
      const res = await fetch(`${api}/update_model_selection`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ modelSelection: model })
      });

      if (!res.ok) throw new Error("Failed to update model");
      
      setSelectedModel(model);
      setShowModelSelectionScreen(false);

      if (apiSelectedModel.current) {
        apiSelectedModel.current = model;
        createAndConnectWs(sessionId, token);
      } else {
        apiSelectedModel.current = model;
      }
    } catch (err) {
      console.error("Error updating model:", err);
    } finally {
      setIsUpdatingModel(false);
    }
  };

  useEffect(() => {
    const isFirefoxBrowser = navigator.userAgent.toLowerCase().indexOf('firefox') > -1;
    setIsFirefox(isFirefoxBrowser);
  }, []);;

  useEffect(() => {
    if (!showIntroMode && !token && !conversing && !tipShownRef.current && !loading) {
      setShowTipCard(true);
      tipShownRef.current = true;
    }
  }, [showIntroMode, token, conversing, loading]);

  useEffect(() => {
    if (showTipCard) {
      const startTime = Date.now();
      const duration = 20000;

      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }

      progressIntervalRef.current = setInterval(() => {
        const elapsedTime = Date.now() - startTime;
        const progressPercent = (elapsedTime / duration) * 100;

        const progressBar = document.getElementById('tip-progress-bar');
        if (progressBar) {
          progressBar.style.width = `${Math.min(progressPercent, 100)}%`;
        }

        if (elapsedTime >= duration) {
          clearInterval(progressIntervalRef.current);
          setShowTipCard(false);
        }
      }, 100);

      return () => {
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current);
        }
      };
    }
  }, [showTipCard]);

  const [showPrivacyPolicyDialog, setShowPrivacyPolicyDialog] = useState(false);

  const handleStartApp = () => {
    setMode(APP_MODE)
  };

  useEffect(() => {
    assistantTalkingRef.current = assistantTalking
  }, [assistantTalking])

  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedEnv, setSelectedEnv] = useState(null);

  useEffect(() => {
    if (selectedEnv) {
      if (selectedEnv.name === 'Cabinet') {
        document.body.style.backgroundImage = `url('${env1Url}')`;
      } else {
        document.body.style.backgroundImage = `url('${selectedEnv.url}')`
      }
    }
  }, [selectedEnv]);

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

  useEffect(() => {
    if (areEnvsLoaded) {
      const savedEnv = window.localStorage.getItem('selectedEnvironment');
      if (savedEnv && environments.find(env => env.name === savedEnv)) {
        const env = environments.find(env => env.name === savedEnv)
        setSelectedEnv({
          name: env.name,
          url: env.fullImage
        })
      } else {
        setSelectedEnv({
          name: "Cabinet"
        })
      }
    }
  }, [environments, areEnvsLoaded])

  useEffect(() => {
    if (selectedEnv) {
      window.localStorage.setItem('selectedEnvironment', selectedEnv.name);
    }
  }, [selectedEnv])

  const prevUserMessageRef = useRef(null);
  const prevAssistantMessageRef = useRef(null);
  const wsReconnectRef = useRef(null);
  const manualCloseRef = useRef(false);

  const handlePlayMessageEmote = (emoteType) => {
    if (emoteType) {
      setCurrentEmote(emoteType);
    }
  };

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
            "emote_type": message.emote_type,
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
      } else if (message.type === "FINISHED_TALK") {
        setAssistantTalking(false);
        if (fullEmoteRef.current) {
          // wait for the previous animations if any, to finish
          setTimeout(() => {
            setCurrentEmote(fullEmoteRef.current)
            fullEmoteRef.current = null;
          }, 250);
        }
      } else if (message.type === "STARTED_TALKING") {
        setAssistantTalking(true);
        if (microEmoteRef.current) {
          setTimeout(() => {
            setCurrentMicroExpression(microEmoteRef.current)
            microEmoteRef.current = null;
          }, 50)
        }
      } else if (message.type === "EMOTE_FOR_CONV") {
        if (message.emotional_response && message.emotional_response.response_type === "express_emote") {
          if (message.emotional_response.full_emote) {
            fullEmoteRef.current = message.emotional_response.full_emote
          }

          if (message.emotional_response.micro_emote) {
            if (assistantTalkingRef.current) {
              setCurrentMicroExpression(message.emotional_response.micro_emote)
            } else {
              // register it in the ref, will be played later.
              microEmoteRef.current = message.emotional_response.micro_emote
            }
          }

          if (message.emotional_response.gesture) {
            setCurrentGesture(message.emotional_response.gesture)
          }
        }
      }  else if (message.type === "CONVERSATION_HISTORY") {
        if (message.history && message.history.length > 0) {
          setConversationToSave(message.history);
          setShowSaveDialog(true);
        }
      } else if (message.type === "AUDIO_MESSAGE") {
        // Create and store a blob URL for the audio
        const audioBase64 = message.audio;
        const binaryAudio = atob(audioBase64);
        const audioArray = new Uint8Array(binaryAudio.length);
        for (let i = 0; i < binaryAudio.length; i++) {
          audioArray[i] = binaryAudio.charCodeAt(i);
        }
        const audioBlob = new Blob([audioArray], { type: 'audio/mp3' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Add to chat with necessary metadata
        const newMessage = {
          role: "assistant",
          type: "AUDIO_MESSAGE",
          content: message.text,
          audioUrl: audioUrl,
          timestamp: Math.floor(Date.now() / 1000)
        };
        
        // Add emote type if it exists
        if (message.emote_type) {
          newMessage.emote_type = message.emote_type;
        }
        
        // Update the chat state
        setChat(chat => {
          // Check if we need to add a Today separator
          const updatedChat = [...chat];
          const currentTimestamp = Math.floor(Date.now() / 1000);
          
          // If we need a "Today" separator, add it
          // have to revisit this
          if (needsTodaySeparator(updatedChat)) {
            updatedChat.push({
              type: 'DATE_SEPARATOR',
              content: 'Today',
              timestamp: currentTimestamp
            });
          }
          
          // Add the new message
          updatedChat.push(newMessage);
          return updatedChat;
        });
        
        setLoadingChat(false);
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
    if (!loading && token) {
      const createSession = async () => {
        if (!sessionId) {
          try {
            // Start the new session API call
          const newSessionPromise = fetch(api + `/new_session`, {
            headers: { Authorization: `Bearer ${token}` },
          }).then(res => res.json());

            // Start fetchConversationHistory (assumed to be async and handle its own state)
            const historyPromise = token ? fetchConversationHistory() : Promise.resolve();

            // Wait for both to complete
            const [newSessionData] = await Promise.all([newSessionPromise, historyPromise]);

            setChatLoading(false);

            // Set sessionId and create WebSocket connection
            setSessionId(newSessionData.session_id);
            if (!newSessionData.model_selection) {
              setShowModelSelectionScreen(true);
              return;
            } else {
              apiSelectedModel.current = newSessionData.model_selection
              setSelectedModel(newSessionData.model_selection)
            }

            if (newSessionData.environments && newSessionData.environments.length) {
              setEnvironments([...environments, ...newSessionData.environments])
              setAreEnvsLoaded(true)
            }

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
    if (!loading) {
      setMode(!token ? INTRO_MODE : APP_MODE)
    }
  }, [loading]);

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

    return <RiVoiceprintFill style={{fontSize: isMobile ? 18 : 21}} />
  }

  const handleSaveConversation = () => {
    if (!conversationToSave) return;

    const blob = new Blob([JSON.stringify(conversationToSave, null, 2)], { type: 'application/json' });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${sessionId}_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(a);
    a.click();

    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setShowSaveDialog(false);
      setConversationToSave(null);
    }, 100);
  };

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
  
  const shouldShowModelLoader = () => 
    modelLoading && isSmallSize 
      ? !showIntroMode 
      : modelLoading;

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
            position: 'relative',
            transition: "transform 0.3s ease"
          }}
          onClick={(e) => {
            if (!isFirefox) {
              e.stopPropagation();
              setIsCallDropdownVisible(false);
              toggleExpandChat(false);
              setPhoneCalling(true);
              initiateWebRTC(false);
              clearInterval(progressIntervalRef.current);
              setShowTipCard(false);
            }
          }}
          onMouseEnter={(e) => {
            if (isFirefox) {
              setShowFirefoxTooltip(true);
            }
          }}
          onMouseLeave={(e) => {
            if (isFirefox) {
              setShowFirefoxTooltip(false);
            }
          }}
        >
          <LoadingDiv
            isLoading={calling}
            duration={0.75} 
            width={`${isMobile ? 36 : 46}px`}
            height={`${isMobile ? 36 : 46}px`}
            borderWidth={1}
            loadingColor="#FFFFFF"
            borderColor="rgba(255, 255, 255, 0.5)"
            borderRadius={`${46}px`}
            backgroundColor="transparent"
            loadingSegmentPercentage={25}
          >
            <HiOutlinePhone style={{ fontSize: isMobile ? 18 : 21 }} />
          </LoadingDiv>
          <div style={{ marginLeft: isMobile ? 7 : "1rem", marginRight: "0.5rem", fontSize: isMobile ? "15px" : "18px" }}>
            {calling ? "Calling Atlas..." : "Let's Connect"}
          </div>
          
          {/* Firefox Warning Tooltip */}
          {isFirefox && showFirefoxTooltip && (
            <div
              style={{
                position: 'absolute',
                bottom: '100%',
                left: '50%',
                transform: 'translateX(-50%)',
                marginBottom: '10px',
                backgroundColor: 'rgba(0, 0, 0, 0.85)',
                color: 'white',
                padding: '8px 12px',
                borderRadius: '6px',
                fontSize: '14px',
                maxWidth: '250px',
                textAlign: 'center',
                zIndex: 1000,
                boxShadow: '0 2px 10px rgba(0, 0, 0, 0.2)',
                pointerEvents: 'none',
                width: 300
              }}
            >
              Unfortunately live WebRTC connection is not available for Firefox at this moment
              {/* Tooltip arrow */}
              <div
                style={{
                  position: 'absolute',
                  top: '100%',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: 0,
                  height: 0,
                  borderLeft: '8px solid transparent',
                  borderRight: '8px solid transparent',
                  borderTop: '8px solid rgba(0, 0, 0, 0.85)',
                }}
              />
            </div>
          )}

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
                  clearInterval(progressIntervalRef.current);
                  setShowTipCard(false);
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
                  clearInterval(progressIntervalRef.current);
                  setShowTipCard(false);
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
        <div style={{ marginRight: "1rem", marginLeft: "0.5rem", fontSize: isMobile ? 15 : "18px" }}>{processing ? "Processing thoughts" : "I'm listening..."}</div>
        <LoadingDiv
          isLoading={processing} 
          duration={0.75} 
          width={`${isMobile ? 36 : 46}px`}
          height={`${isMobile ? 36 : 46}px`}
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
      maxWidth: userVoiceMessage === "..." ? "none" : "calc(100% - 20px)",
      alignSelf: "flex-end", 
      marginTop: 10,
      padding: 8,
      boxSizing: "border-box",
      backgroundColor: 'rgba(100, 150, 255, 1)', // Blue background
      border: '1px solid rgba(255, 255, 255, 0.4)',
      borderRadius: 12,
      transition: "max-height 0.3s ease, max-width 0.3s ease"
    }}
    key={`user-${key}`}
    className="userSlideIn"
    >
      <div style={{
        color: 'white',
        boxSizing: "border-box",
        fontSize: 14,
        lineHeight: "18px",
        letterSpacing: "0.5px",
        overflow: isTranscriptionExpanded ? "auto" : "hidden",
        textWrap: isTranscriptionExpanded ? "wrap" : "nowrap",
        textOverflow: "ellipsis"
      }}
      >
        {userVoiceMessage}
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
      bottom: 350,
      right: 20,
      width: 280,
      height: 170
    }
  }

  // Function to render assistant message
  const renderAssistantMessage = (key) => (
    <div style={{
      display: "flex",
      maxWidth: assistantMessage === "..." ? "none" : "100%",
      alignSelf: "flex-end",
      marginTop: 10,
      backgroundColor: 'rgba(50, 50, 50, 0.55)',
      border: '1px solid rgba(255, 255, 255, 0.3)',
      backdropFilter: "blur(8px)",
      WebkitBackdropFilter: "blur(8px)",
      padding: 8,
      borderRadius: 12,
      transition: "max-height 0.3s ease, max-width 0.3s ease",
      boxSizing: "border-box"
    }}
    key={`assistant-${key}`}
    className="assistantSlideIn"
    >
      <div style={{
        flex: 1,
        color: "white",
        fontSize: 14,
        overflow: isTranscriptionExpanded ? "auto" : "hidden",
        textWrap: isTranscriptionExpanded ? "wrap" : "nowrap",
        lineHeight: "18px",
        letterSpacing: "0.5px",
        textOverflow: "ellipsis"
      }}
      >
        {assistantMessage}
      </div>
    </div>
  );

  if (loading) {
    return (
      <div
        className="App"
        style={{
          position: "relative",
          background: "transparent",
          height: "100%",
          width: "100%",
          background: "black"
        }}
      />
    )
  }

  const renderMenu = () => {
    return (
      <div style={{
        position: "absolute",
        top: 0,
        right: 0,
        left: 0,
        bottom: 0,
        background: 'rgba(20, 20, 20, 0.25)',
        backdropFilter: "blur(4px)",
        WebkitBackdropFilter: "blur(4px)",
        zIndex: 99999,
      }}>
      <div style={{
        position: "absolute",
        top: 15,
        width: 400,
        right: 25,
        bottom: 15,
        borderRadius: 15,
        background: 'rgba(0, 0, 0, 1)',
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        border: "1px solid rgba(255, 255, 255, 0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-start",
        boxSizing: "border-box",
        flexDirection: "column",
      }}>
        <div style={{
          width: "100%"
        }}>
        <div style={{
          position: "absolute",
          top: 1,
          right: 1,
          width: 200,
          height: 45,
          background: "#393939",
          fontSize: 17,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          borderBottom: "1px solid #1a1a1a",
          borderTopRightRadius: 13,
          clipPath: "polygon(20% 0, 100% 0, 100% 100%, 0 100%)",
          color: "white",
          fontWeight: "bold"
        }}>
          <div style={{position: "absolute", right: 150, top: 0, width: 70, height: 45, clipPath: "polygon(55% 0%, 100% 0%, 60% 100%, 0% 100%)", background: "#1a1a1a"}} />
          <span>Settings</span>
        </div>
        </div>
        <div className="convert-white-hover" style={{
          position: "absolute",
          top: 5,
          left: 5,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "6px 12px",
          borderRadius: 15,
          cursor: "pointer"
        }} onClick={() => {
          setShowMenu(false);
        }}>
          <span style={{
            fontSize: 16
          }}>Close</span>
          <IoIosCloseCircleOutline style={{
            fontSize: 25,
            marginLeft: 8
          }} />
        </div>
        <div style={{
          marginTop: 70,
          width: "100%"
        }}>
          <div style={{
            marginRight: 15,
            fontSize: 25
          }}>
            <span style={{
              fontWeight: "bold"
            }}>Hello</span> {user.displayName || user.email}
          </div>
          <div style={{
            width: "100%",
            flex: 1,
            padding: 25,
            paddingTop: 50,
            boxSizing: 'border-box'
          }}>
            <div style={{
              width: "100%",
              display: "flex",
              alignItems: "flex-start",
              flexDirection: "column",
              height: "100%",
              overflowY: "auto"
            }}>
              <div style={{
                fontSize: 18
              }}>Psychologist</div>
              <div style={{
                width: "100%",
                height: 1,
                marginTop: 8,
                borderBottom: "1px dotted rgba(255, 255, 255, 0.75)"
              }} />
              <div style={{
                width: "100%",
                display: "flex",
                flexDirection: "row",
                paddingTop: 8,
                marginBottom: 25,
                boxSizing: "border-box"
              }}>
                <img src={selectedModel === "Atlas" ? "philosophical-icon.png" : "fun-icon.png"} style={{
                  width: selectedModel === "Atlas" ? 55 : 40,
                  marginRight: selectedModel === "Atlas" ? 0 : 8,
                  marginTop: selectedModel === "Atlas" ? 0 : 8,
                  alignSelf: 'flex-start'
                }} />
                <div style={{
                  marginLeft: 8,
                  marginTop: 5,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-start",
                  justifyContent: "flex-start"
                }}>
                  <div style={{
                    fontWeight: "bold"
                  }}>{selectedModel}</div>
                  <div style={{
                    fontSize: 14,
                    textAlign: "left",
                    marginTop: 5
                  }}>
                  {selectedModel === "Atlas" ? "Fine-tuned for introspection, psychology and philosophy. Perfect for inner self exploration!" : "Fun and creative. The ideal partner to talk about your family, friends and life."}
                  </div>
                  <div style={{width: "100%", display: "flex"}}>
                    <div style={{
                      marginTop: 8,
                      width: 80,
                      height: 28  ,
                      background: 'white',
                      borderRadius: 5,
                      color: "black",
                      display: "flex",
                      fontSize: 14,
                      justifyContent: "center",
                      alignItems: "center",
                      cursor: "pointer",
                      transition: "transform 0.3s ease"
                    }} onMouseOver={(e) => {
                      e.currentTarget.style.transform = "scale(1.05)";
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.transform = "scale(1)";
                    }} onClick={() => {
                      setShowMenu(false);
                      setShowModelSelectionScreen(true);
                    }}>
                      <span>Change</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div style={{
              width: "100%",
              display: "flex",
              alignItems: "flex-start",
              flexDirection: "column"
            }}>
              <div style={{
                fontSize: 18
              }}>Environment</div>
              <div style={{
                width: "100%",
                height: 1,
                marginTop: 8,
                borderBottom: "1px dotted rgba(255, 255, 255, 0.75)"
              }} />
              <div style={{
                display: "flex",
                marginBottom: 15,
                paddingBottom: 10,
                width: "100%",
                paddingTop: 15,
                alignItems: "center",
                boxSizing: "border-box",
                justifyContent: "flex-start",
                width: "100%",
                overflowX: "auto"
              }}>
                {environments.map(env => {
                  const isSelected = env.name === selectedEnv.name
                  return (
                  <div style={{
                    display: "flex",
                    minHeight: 0,
                    flexDirection: "column",
                    alignItems: "flex-start",
                    justifyContent: "flex-start",
                    borderRadius: 8,
                    position: "relative",
                    cursor: "pointer",
                    border: `1px solid ${isSelected ? "#AD60FF" : 'rgba(255, 255, 255, 0.55)'}`,
                    padding: 5,
                    paddingBottom: 8,
                    boxSizing: "border-box",
                    marginRight: 10
                  }} onClick={() => {
                    setSelectedEnv({
                      name: env.name,
                      url: env.fullImage
                    })
                  }}>
                    <div style={{
                      width: 85,
                      height: 85,
                      borderRadius: 5,
                      backgroundImage: `url(${env.name === "Cabinet" ? "default-env-cabinet.jpg" : env.thumbnail})`,
                      backgroundSize: "cover",
                    }} />
                    <div style={{ width: "100%", height: 2, marginTop: 3, borderBottom: '1px dotted rgba(255, 255, 255, 0.5)'}} />
                    {isSelected ? <div style={{
                      position: "absolute",
                      bottom: -8,
                      right: -5,
                      borderRadius: 2,
                      width: 18,
                      height: 18,
                      background: "#AD60FF",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center"
                    }}><FaCheck style={{
                      fontSize: 13,
                    }} /></div> : null}
                    <div style={{
                      fontSize: 15,
                      marginTop: 4,
                      maxWidth: 85,
                      fontSize: 13,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap"
                    }} title={env.name}>
                      {env.name}
                    </div>
                  </div>)}
                )}
                <div style={{
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center"
                }}>
                <div onClick={() => {
                  setShowMenu(false);
                  setShowEnvironmentModal(true)
                }} onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(255, 255, 255, 1)"
                  e.currentTarget.style.color = "black"
                }} onMouseLeave={(e) => {
                  e.currentTarget.style.background = "#393939"
                  e.currentTarget.style.color = "rgba(255, 255, 255, 0.45)"
                }} style={{width: 42, cursor: "pointer", height: 42, background: "#393939", borderRadius: 8, marginLeft: 10, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 28, color: "rgba(255, 255, 255, 0.45)"}}>
                  <span style={{
                    marginTop: -3
                  }}>
                    +
                  </span>
                </div>
                </div>
              </div>
            </div>
            <div style={{
              width: "100%",
              display: "flex",
              alignItems: "flex-start",
              flexDirection: "column"
            }}>
              <div style={{
                fontSize: 18
              }}>Clothes</div>
              <div style={{
                width: "100%",
                height: 1,
                marginTop: 8,
                borderBottom: "1px dotted rgba(255, 255, 255, 0.75)"
              }} />
              <div style={{
                marginBottom: 30,
                width: "100%",
                paddingTop: 10,
                display: "flex"
              }}>
                <div style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  padding: 6,
                  boxSizing: "border-box",
                  border: '1px solid #AD60FF',
                  borderRadius: 5,
                  position: "relative"
                }}>
                  <img src="shirt.png" style={{
                    width: 28
                  }} />
                  <div style={{
                    marginTop: 5,
                    fontSize: 13
                  }}>
                    Formal
                  </div>
                  <div style={{
                      position: "absolute",
                      bottom: -8,
                      right: -5,
                      borderRadius: 2,
                      width: 15,
                      height: 15,
                      background: "#AD60FF",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center"
                    }}><FaCheck style={{
                      fontSize: 10,
                    }} /></div>
                </div>
                <div style={{
                  display: "flex",
                  flexDirection: "column",
                  flexGrow: 1,
                  alignItems: "flex-start",
                  justifyContent: "center",
                  marginLeft: 18,
                  fontSize: 14
                }}>
                  <div>
                    More To Come...
                  </div>
                  <div style={{
                    fontSize: 13,
                    opacity: 0.85
                  }}>
                    Stay tuned.
                  </div>
                </div>
              </div>
            </div>
            <div style={{
              width: "100%",
              display: "flex",
              alignItems: "flex-start",
              flexDirection: "column"
            }}>
              <div style={{
                fontSize: 18
              }}>Data Persistence</div>
              <div style={{
                width: "100%",
                height: 1,
                marginTop: 8,
                borderBottom: "1px dotted rgba(255, 255, 255, 0.75)"
              }} />
              <div style={{
                width: "100%",
                paddingTop: 15
              }}>
                <div style={{
                  display: "flex"
                }}>
                  <Switch 
                    isChecked={isChatEnabled}
                    onChange={value => handleChatToggle(value, token)}
                    isDisabled={isTogglingChat}
                    isLoading={loadingPreferences}
                  />
                  <div style={{
                    fontSize: 14,
                    marginLeft: 8
                  }}>Chat Persistence</div>
                </div>
                <div style={{
                  display: "flex",
                  marginTop: 15
                }}>
                  <Switch
                    isChecked={isMemoryEnabled}
                    onChange={(value) => handleMemoryToggle(value, token)}
                    isDisabled={isTogglingMemory}
                    isLoading={loadingPreferences}
                  />
                  <div style={{
                    fontSize: 14,
                    marginLeft: 8
                  }}>Memory Persistence</div>
                </div>
              </div>
            </div>
          </div>
          <div className="convert-white-hover" style={{
            position: "absolute",
            right: 12,
            bottom: 10,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "6px 12px",
            borderRadius: 15,
            cursor: "pointer"
          }}
          onClick={logout}
          >
          <IoIosLogOut style={{
            fontSize: 24,
            marginRight: 5
          }} />
          <span style={{
            fontSize: 16
          }}>Log Out</span>
          </div>
        </div>
      </div>
      </div>
    )
  }

  return (
    <div
      className="App"
      style={{
        position: "relative",
        background: "transparent",
        height: "100%",
      }}
    >
        {showIntroMode && <Overlay isMobile={isMobile} smallerThan850={smallerThan850} isSmallSize={isSmallSize} token={token} showCreateAccount={showCreateAccount} signInWithGoogle={signInWithGoogle} showLoginView={showLoginView} handleStartApp={handleStartApp} toggleLoginView={() => {
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
        {showMenu && renderMenu()}
      {!showIntroMode && <div style={{
        position: "fixed",
        top: 0,
        right: 0,
        width: "100%",
        height: "100%"
      }}>
          <div style={{
            position: "relative",
            width: "100%",
            height: "100%"
          }}>
          <BackgroundScene
            currentMicroExpression={currentMicroExpression}
            setCurrentMicroExpression={setCurrentMicroExpression}
            isSmallSize={isSmallSize} isTalking={isTalking}
            assistantTalking={assistantTalking}
            visemeSequence={visemes}
            currentEmote={currentEmote}
            setCurrentEmote={setCurrentEmote}
            isIntroMode={showIntroMode} onModelLoaded={handleModelLoaded} isModelVisible={isModelVisible}
            currentGesture={currentGesture}
            setCurrentGesture={setCurrentGesture}
            />
          {shouldShowModelLoader() && <ModelLoader />}
        </div>
      </div>}
      {showSaveDialog && (
        <SaveDialog
          onSave={handleSaveConversation} 
          onCancel={() => {
            setShowSaveDialog(false);
            setConversationToSave(null);
          }}
        />
      )}
      {!showIntroMode && (
      <>
        <div style={{
            position: "absolute",
            top: isMobile ? 10 : "16px",
            left: "16px",
            zIndex: 2,
            background: 'rgba(0, 0, 0, 0.45)',
            backdropFilter: "blur(8px)",
            WebkitBackdropFilter: "blur(8px)",
            border: "1px solid rgba(255, 255, 255, 0.45)",
            borderRadius: "21px",
            color: "white",
            fontSize: isMobile ? 19 : "23px",
            width: isMobile ? 120 : 155,
            height: isMobile ? 45 : 60,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer"
          }}
          onClick={() => {
            window.location.href = '/';
          }}
          >
            <GiBrain style={{
              fontSize: isMobile ? 28 : 38,
              color: "white",
              marginLeft: -5
            }} />
            <div style={{marginLeft: isMobile ? 8 : 10, display: "flex", flexDirection: "column", alignItems: 'flex-start'}}>
              <div>Self AI</div>
            </div>
        </div>
        {selectedModel && <div
          ref={modelDropdownRef}
          style={{
            position: "absolute",
            top: isMobile ? "58px" : "85px",
            left: "50%",
            transform: !isMobile ? "translate(-50%, -35%)" : "translateX(-50%)",
            zIndex: 2,
            background: 'rgba(0, 0, 0, 0.45)',
            backdropFilter: "blur(8px)",
            WebkitBackdropFilter: "blur(8px)",
            border: "1px solid rgba(255, 255, 255, 0.55)",
            borderRadius: "26px",
            padding: isModelDropdownOpen ? "0.8rem" : isMobile ? "3px 16px" : "0.5rem 1.2rem",
            color: "white",
            fontSize: isMobile ? 16 : "1.2rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            transition: "all 0.3s ease-in-out",
            height: isModelDropdownOpen ? "auto" : isMobile ? 40 : "48px",
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
              {selectedModel}
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
                <div style={{ fontSize: "1.2rem", flex: 1 }}>{selectedModel}</div>
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
              <div style={{ maxHeight: "190px", maxWidth: 280, overflowY: "auto", width: "100%", fontSize: 15 }}>
                {selectedModel === "Atlas" ? <div style={{
                  borderTop: '1px solid grey',
                  borderBottom: '1px solid grey',
                  paddingBottom: 8,
                  paddingTop: 8,
                  fontSize: 14
                }}>
                  Fine-tuned for introspection, psychology and philosophy. Perfect for inner self exploration!
                </div> : <div style={{
                  borderTop: '1px solid grey',
                  borderBottom: '1px solid grey',
                  paddingBottom: 8,
                  paddingTop: 8,
                  fontSize: 14
                }}>
                  Fun and creative. The ideal partner to talk about your family, friends and life.
                </div>}
                <div style={{width: "100%", display: "flex", alignItems: "center", justifyContent: "center"}}>
                <div style={{
                  width: 95,
                  height: 33,
                  borderRadius: 8,
                  color: "black",
                  fontSize: 14,
                  background: "rgba(255, 255, 255, 0.85)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginTop: 8
                }} onClick={() => {
                  setShowModelSelectionScreen(true);
                }}>
                  <span>Change</span>
                </div>
                </div>
              </div>
            </>
          )}
        </div>}

        {!loading &&
        <div
          style={{
            position: "absolute",
            top: isMobile ? 10 : token ? "16px" : "22px",
            right: "16px",
            zIndex: 2,
            background: token ? 'rgba(0, 0, 0, 0.25)' : 'rgba(255, 255, 255, 0.85)',
            backdropFilter: "blur(15px)",
            WebkitBackdropFilter: "blur(15px)",
            border: "1px solid rgba(255, 255, 255, 0.3)",
            borderRadius: "26px",
            color: "white",
            fontSize: 17,
            height: token ? isMobile ? 45 : 62 :  isMobile ? 45 : 45,
            boxSizing: "border-box",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            padding: isMobile ? "0 12px" : "0 16px"
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "scale(1.05)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "scale(1)";
          }}
          onClick={() => {
            setShowMenu(true);
          }}
        >
          {token && 
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}><RxAvatar style={{
            fontSize: isMobile ? 20 : 30,
            color: "white",
            marginRight: 8
          }} /><span>Settings</span></div>}
          {!token ? <div style={{ color: "black", fontSize: isMobile ? 14 : 15, fontWeight: "bold"}}>Create Account</div> : null}
        </div>}

        {/* Tip Card */}
        {showTipCard && !token && !showIntroMode && !isMobile && (
          <div
          className={showTipCard ? "tipCardEnter" : "tipCardExit"}
          style={{
            position: "absolute",
            top: "82px",
            right: "16px",
            width: isMobile ? "250px" : "320px",
            background: 'rgba(0, 0, 0, 0.35)',
            backdropFilter: "blur(8px)",
            WebkitBackdropFilter: "blur(8px)",
            border: "1px solid rgba(255, 255, 255, 0.4)",
            borderRadius: "12px",
            padding: "12px",
            color: "white",
            zIndex: 10,
            boxShadow: "0 4px 15px rgba(0, 0, 0, 0.2)",
            overflow: "hidden", // Important to contain the progress bar
            opacity: showTipCard ? 1 : 0,
            transform: showTipCard ? "translateY(0)" : "translateY(-20px)",
            transition: "opacity 0.5s ease, transform 0.5s ease"
          }}
        >
            {/* Header */}
            <div style={{ 
              display: "flex", 
              alignItems: "center", 
              marginBottom: "10px",
              position: "relative",
              width: "100%"
            }}>
              <IoCloseCircleOutline style={{
                fontSize: 24,
                position: "absolute",
                right: 0,
                top: 0,
                cursor: "pointer"
              }} onClick={(e) => {
                e.stopPropagation();
                clearInterval(progressIntervalRef.current);
                setShowTipCard(false);
              }} />
              <IoIosInformationCircleOutline style={{
                fontSize: 24,
                marginRight: 5
              }} />
              <div style={{ 
                fontWeight: "600", 
                fontSize: "16px"
              }}>
                Useful tip
              </div>
            </div>
            
            {/* Content */}
            <div style={{ 
              fontSize: "15px", 
              lineHeight: "1.5",
              marginBottom: "16px",
              textAlign: "left"
            }}>
              <span style={{fontWeight: "bold"}}>Voice connection</span> features facial expressions and voice cues recognition. This is the best way to connect with <span style={{fontWeight: "bold"}}>Atlas</span>.
            </div>
            <div style={{
              background: 'rgba(255, 255, 255, 0.85)',
              border: "1px solid rgba(255, 255, 255, 0.3)",
              borderRadius: "26px",
              color: "black",
              fontSize: 14,
              height: 32,
              width: 100,
              boxSizing: "border-box",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer"
            }}
            onClick={(e) => {
              e.stopPropagation(); // Prevent body click from closing it immediately
              setIsCallDropdownVisible(true); // Show dropdown instead of dialog
              clearInterval(progressIntervalRef.current);
              setShowTipCard(false);
            }}>
              Connect
            </div>
            
            {/* Progress Bar Container */}
            <div style={{
              position: "absolute",
              bottom: 0,
              left: 0,
              width: "100%",
              height: "3px",
              backgroundColor: "rgba(255, 255, 255, 0.1)"
            }}>
              {/* Actual Progress Bar */}
              <div
                id="tip-progress-bar"
                style={{
                  position: "absolute",
                  bottom: 0,
                  left: 0,
                  height: "100%",
                  width: "0%", // Starts at 0%
                  backgroundColor: "white",
                  boxShadow: "0 0 10px rgba(66, 135, 245, 0.7)",
                  transition: "none" // No transition, we update width directly
                }}
              />
            </div>
          </div>
        )}

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
              isMobile={isMobile}
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
              isMobile={isMobile}
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
                    // Check if the message is a text message (string) or an audio message (object)
                    if (typeof message === 'string') {
                      // It's a text message, handle it as before
                      setChat([...chat, {
                        "role": "user",
                        "content": message,
                        "timestamp": Math.floor(Date.now() / 1000)
                      }]);
              
                      wsRef.current.send(JSON.stringify({
                        "type": "CHAT_MESSAGE",
                        "message": message
                      }));
              
                      setLoadingChat(true);
                    } 
                    else if (message.type === 'AUDIO_MESSAGE') {
                      // It's an audio message
                      // Add to chat immediately for UI feedback
                      setChat([...chat, {
                        "role": "user",
                        "type": "AUDIO_MESSAGE",
                        "audioUrl": message.audioUrl,
                        "timestamp": Math.floor(Date.now() / 1000)
                      }]);
              
                      // If the audio message has base64 audio data, send it through WebSocket
                      if (message.audioData) {
                        wsRef.current.send(JSON.stringify({
                          "type": "VOICE_MESSAGE",
                          "audio": message.audioData,
                          "format": "webm",
                          "sessionId": sessionId
                        }));

                        setLoadingChat(true);
                      }
                    }
                  }
                }} isLoading={loadingChat} token={token} api={api} onDeleteMessage={handleDeleteMessage} isMobile={isMobile}
                onPlayEmote={handlePlayMessageEmote} />}
              </div>
            </CollapsibleMemoriesPanel>
        </div>
        {!isMobile && conversing &&
          <div 
            style={{
              position: "absolute",
              width: isTranscriptionExpanded ? "290px" : "220px",
              maxHeight: isTranscriptionExpanded ? "250px" : "none",
              bottom: "0px",
              right: "0px",
              display: "flex",
              flexDirection: "column",
              maxWidth: "400px",
              padding: "20px",
              zIndex: 3,
              alignItems: "flex-end",
              justifyContent: "flex-end",
              background: 'linear-gradient(to bottom, rgba(35, 35, 35, 0.75) 0%, rgba(50, 50, 50, 0) 70%, rgba(50, 50, 50, 0.15) 100%)',
              borderTopLeftRadius: 10,
              borderTop: "1px solid rgba(100, 100, 100, 1)",
              transition: 'all 0.2s ease',
              paddingTop: 10
            }}
            onMouseEnter={(e) => {
              setExpandTranscription(true);
            }}
            onMouseLeave={(e) => {
              setExpandTranscription(false);
            }}
          >
            <div style={{
              // background: "rgba(30, 30, 30, 0.55)",
              padding: "10px 14px",
              borderRadius: 6,
              // border: "1px solid rgba(80, 80, 80, 1)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "100%",
            }}>
              <LuMessagesSquare style={{
                fontSize: 21
              }} />
              <span style={{
                marginLeft: 8,
                fontSize: 16
              }}>Voice Transcriptions</span>
            </div>
            <div style={{
              width: "100%",
              height: 1,
              background: 'linear-gradient(to right, rgba(125, 125, 125, 0) 0%, rgba(180, 180, 180, 1) 50%, rgba(50, 50, 50, 0) 100%)',
            }} />
            <div style={{
              display: "flex",
              flexDirection: "column",
              width: "100%",
              height: "100%",
              overflowY: "auto",
              overflowX: "hidden",
              padding: 8,
              boxSizing: "border-box"
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
            </div>
          </div>
        }
        <div style={{
          position: "absolute",
          bottom: isMobile ? "20px" : "50px",
          right: "50%",
          display: "flex",
          transform: "translateX(50%)",
          zIndex: 3,
          flexDirection: "column"
        }}>
          {(
            <div style={{
              height: 25,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: 10
            }}>
              {conversing  && (userVoiceMessage === "..." || assistantMessage === "...") ? (
                <div style={{
                  background: 'rgba(50, 50, 50, 0.25)',
                  width: "60px",
                  height: "100%",
                  borderRadius: 5
                }}>
                  <LoadingDots size={5} />
                </div>
            ) : null}
            </div>
            )}
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}>
            <div
              style={{
                zIndex: 2,
                backdropFilter: "blur(8px)",
                WebkitBackdropFilter: "blur(8px)",
                background: 'rgba(0, 0, 0, 0.55)',
                border: "1px solid rgba(255, 255, 255, 0.65)",
                borderRadius: "46px",
                color: "white",
                textAlign: "center",
                minWidth: 140,
                minHeight: 60,
                display: "flex",
                alignItems: "center"
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
                    backdropFilter: "blur(8px)",
                    WebkitBackdropFilter: "blur(8px)",
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
                    width={`${isMobile ? 36 : 46}px`}
                    height={`${isMobile ? 36 : 46}px`}
                    borderWidth={1}
                    loadingColor="#FFFFFF"
                    borderColor="rgba(255, 255, 255, 0.5)"
                    borderRadius={`${46}px`}
                    backgroundColor="#ed7878"
                    loadingSegmentPercentage={25}
                  >
                  <HiOutlinePhoneXMark style={{fontSize: isMobile ? 17 : 21}} />
                </LoadingDiv>
              </div>)}
            </div>
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
          {showModelSelectionScreen && <MyPanelWithWaves
            apiSelectedModel={apiSelectedModel.current}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            updateModel={updateModel}
            isUpdatingModel={isUpdatingModel}
            setShowModelSelectionScreen={setShowModelSelectionScreen}
          />}
          {showEnvironmentModal &&
          <EnvironmentModal api={api} token={token} setShowEnvironmentModal={setShowEnvironmentModal} setSelectedEnv={setSelectedEnv} setEnvironments={setEnvironments} environments={environments} />}
      </div>
  );
}

export default App;
