import React, { useState, useRef, useEffect } from "react";
import { IoIosSend } from "react-icons/io";
import { FaMicrophone } from "react-icons/fa";
import { ImCross } from "react-icons/im";
import { MdMic } from "react-icons/md";
import LoadingDiv from "./LoadingDiv";
import { BsThreeDots } from "react-icons/bs";
import PlayEmoteButton from './PlayEmoteButton';
import { formatDuration } from "../utils";
import { RiVoiceprintLine } from "react-icons/ri";
import { IoPlayCircleOutline } from "react-icons/io5";
import { HiOutlinePauseCircle } from "react-icons/hi2";


// Audio Message component to display and play recorded audio
const AudioMessage = ({ audioUrl, index, role }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      const handleEnded = () => setIsPlaying(false);
      audio.addEventListener('ended', handleEnded);
      return () => {
        audio.removeEventListener('ended', handleEnded);
      };
    }
  }, []);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      width: '100%'
    }}>
      <button
        onClick={togglePlayPause}
        style={{
          border: 'none',
          background: "none",
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          color: 'white',
          marginRight: '8px',
          padding: 0
        }}
      >
        {isPlaying ? <HiOutlinePauseCircle style={{
          color: role === "user" ? "white" : "black"
        }} size={25} /> : <IoPlayCircleOutline size={25} style={{
          color: role === "user" ? "white" : "black"
        }} />}
      </button>
      <span><RiVoiceprintLine style={{fontSize: 17}} /><RiVoiceprintLine style={{fontSize: 17}} /><RiVoiceprintLine style={{fontSize: 17}} /><RiVoiceprintLine style={{fontSize: 17}} /></span>
      <div
        style={{
          flex: 1,
          height: '24px',
          background: 'rgba(0, 0, 0, 0.1)',
          borderRadius: '12px',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <audio ref={audioRef} src={audioUrl} />
      </div>
    </div>
  );
};

const ChatMessage = ({ message, token, api, onDeleteMessage, index, onPlayEmote }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const [showThreeDots, setShowThreeDots] = useState(false);
  const hasEmote = message.role === 'assistant' && message.emote_type;
  const isAudioMessage = message.type === 'AUDIO_MESSAGE';

  const handleDeleteMessage = async (messageItem) => {
    if (token && typeof onDeleteMessage === 'function') {
      try {
        console.log('message id', messageItem)
        const response = await fetch(`${api}/delete_message`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ 
            message_id: messageItem.id 
          })
        });
        
        if (response.ok) {
          onDeleteMessage(messageItem);
          setIsMenuOpen(false);
        }
      } catch (error) {
        console.error("Error deleting message:", error);
      }
    }
  };

  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setIsMenuOpen(false);
        setShowThreeDots(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  return (
    <div
      key={index}
      className={`message ${message.role}`}
      style={{
        alignSelf: message.role === "user" ? "flex-end" : "flex-start",
        marginLeft: message.role === "user" ? "20%" : "0",
        marginRight: message.role === "assistant" ? "20%" : "0",
        backgroundColor: message.role === "user" ? "rgba(100, 150, 255, 0.85)" : "rgba(255, 255, 255, 0.65)",
        color: message.role === "user" ? "white" : "black",
        padding: "12px 16px",
        borderRadius: "8px",
        maxWidth: "80%",
        wordWrap: "break-word",
        boxShadow: "0 1px 2px rgba(0, 0, 0, 0.1)",
        border: "1px solid rgba(255, 255, 255, 0.3)",
        fontSize: 15,
        textAlign: "left",
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: "center"
      }}
      onMouseEnter={() => {
        if (message.id) {
          setShowThreeDots(true);
        }
      }}
      onMouseLeave={() => {
        if (!isMenuOpen) {
          setShowThreeDots(false);
        }
      }}
    >
      {isAudioMessage ? (
        <AudioMessage audioUrl={message.audioUrl} index={index} role={message.role} />
      ) : (
        <span>{message.content}</span>
      )}
      
      {/* Add play button if the message has an emote_type */}
      {hasEmote && (
        <div style={{
          width: "100%",
          display: "flex",
          justifyContent: "flex-end"
        }}>
        <PlayEmoteButton 
          emoteType={message.emote_type} 
          onClick={() => onPlayEmote(message.emote_type)} 
        />
        </div>
      )}
      {showThreeDots &&
        <div 
            style={{ 
                position: "absolute", 
                top: "0px", 
                right: "5px", 
                cursor: "pointer",
                zIndex: 5,
                color: "black",
                fontSize: 17
            }}
            onClick={(e) => {
            e.stopPropagation();
            setIsMenuOpen(!isMenuOpen);
            }}
        >
            <BsThreeDots />
            
            {isMenuOpen && (
            <div 
                ref={menuRef}
                style={{
                    position: "absolute",
                    top: "15px",
                    right: "0",
                    background: "rgba(255, 255, 255, 0.95)",
                    border: "1px solid rgba(255, 255, 255, 1)",
                    borderRadius: "5px",
                    zIndex: 100,
                    minWidth: "120px",
                    color: "black",
                    fontSize: 14,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center"
                }}
            >
                <div 
                style={{
                    padding: "4px 6px",
                    cursor: "pointer",
                    hover: { background: "rgba(255, 255, 255, 0.1)" }
                }}
                onClick={() => handleDeleteMessage(message)}
                >
                Delete Message
                </div>
            </div>
            )}
        </div>}
    </div>
  )
}

// Tooltip component for hover effects
const Tooltip = ({ text, children, show }) => {
  if (!show) return children;
  
  return (
    <div style={{ position: 'relative' }}>
      {children}
      <div style={{
        position: 'absolute',
        bottom: '100%',
        right: 0,
        marginBottom: '5px', // Increased margin
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        color: '#000',
        padding: '4px 8px',
        borderRadius: '4px',
        fontSize: '12px',
        whiteSpace: 'nowrap',
        pointerEvents: 'none',
        zIndex: 1000, // Higher z-index
        boxShadow: '0 2px 4px rgba(0,0,0,0.2)' // Added shadow for visibility
      }}>
        {text}
      </div>
    </div>
  );
};

const Chat = ({ chat, onSendMessage, isLoading, token, api, onDeleteMessage, onPlayEmote }) => {
  const [newMessage, setNewMessage] = useState("");
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  
  // State for recording functionality
  const [isRecording, setIsRecording] = useState(false);
  const [isCountingDown, setIsCountingDown] = useState(false);
  const [countdownValue, setCountdownValue] = useState(3);
  const [recordingTime, setRecordingTime] = useState(0);
  const [showSendTooltip, setShowSendTooltip] = useState(false);
  
  // Refs for recording
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingTimerRef = useRef(null);
  const countdownTimerRef = useRef(null);
  const recordingStartTimeRef = useRef(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  // Focus input field when component mounts
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Cleanup timers when component unmounts
  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
      stopRecording();
    };
  }, []);

  const handleSendMessage = () => {
    if (newMessage.trim()) {
      onSendMessage(newMessage);
      setNewMessage("");
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Start countdown before recording
  const handleRecordClick = () => {
    setIsCountingDown(true);
    setCountdownValue(3);
    
    countdownTimerRef.current = setInterval(() => {
      setCountdownValue((prev) => {
        if (prev === 1) {
          clearInterval(countdownTimerRef.current);
          setIsCountingDown(false);
          startRecording();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  // Start the actual recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      
      // Start recording timer using timestamp-based approach
      setRecordingTime(0);
      recordingStartTimeRef.current = Date.now();
      recordingTimerRef.current = setInterval(() => {
        const elapsedSeconds = Math.floor((Date.now() - recordingStartTimeRef.current) / 1000);
        setRecordingTime(elapsedSeconds);
      }, 200); // Update more frequently for smoother display
    } catch (error) {
      console.error("Error starting recording:", error);
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      clearInterval(recordingTimerRef.current);
      
      // Stop all audio tracks
      if (mediaRecorderRef.current.stream) {
        mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      }
    }
  };

  // Cancel recording without sending
  const cancelRecording = () => {
    stopRecording();
    setIsRecording(false);
    audioChunksRef.current = [];
  };

  // Send the recording
  const sendRecording = () => {
    // Attach the onstop handler
    mediaRecorderRef.current.onstop = () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      const audioUrl = URL.createObjectURL(audioBlob);
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = function() {
        const base64Audio = reader.result.split(',')[1];
        const audioMessage = {
          role: "user",
          type: "AUDIO_MESSAGE",
          audioUrl: audioUrl,
          audioData: base64Audio,
          timestamp: Math.floor(Date.now() / 1000)
        };
        setIsRecording(false);
        audioChunksRef.current = [];
        if (typeof onSendMessage === 'function') {
          onSendMessage(audioMessage);
        }
      };
    };
  
    stopRecording(); // This will trigger the onstop event when done.
  };
  

  const renderMessage = (message, index) => {
    // Handle different message types
    if (message.type === "CONVERSATION_EVENT") {
      return (
        <div
          key={index}
          className={`message ${message.role}`}
          style={{
            alignSelf: "center",
            backgroundColor: "rgba(255, 255, 255, 0.45)",
            color: message.role === "user" ? "white" : "black",
            padding: "6px 4px",
            borderRadius: "8px",
            maxWidth: "80%",
            wordWrap: "break-word",
            boxShadow: "0 1px 2px rgba(0, 0, 0, 0.1)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            fontSize: 14,
            textAlign: "left"
          }}
        >
          {message.duration ? `Phone call conversation (${message.duration})` : "Phone call conversation"}
        </div>
      );
    } else if (message.type === "DATE_SEPARATOR") {
      return (
        <div key={index} style={{
          display: "flex",
          alignItems: "center",
          flexDirection: "column"
        }}>
          <div style={{
            color: "white",
            marginBottom: 8
          }}>
            {message.content}
          </div>
          <div style={{
            width: "100%",
            minHeight: 1,
            backgroundColor: 'rgba(255, 255, 255, 0.35)'
          }} />
        </div>
      );
    }

    return <ChatMessage 
      key={index}
      onDeleteMessage={onDeleteMessage} 
      api={api} 
      token={token} 
      message={message} 
      index={index} 
      onPlayEmote={onPlayEmote} 
    />;
  };

  return (
    <div className="chat-container" style={{ display: "flex", paddingTop: 15, flexDirection: "column", flex: 1 }}>
      <div 
        className="messages-container" 
        style={{ 
          flex: 1, 
          overflowY: "auto", 
          display: "flex", 
          flexDirection: "column",
          gap: "12px"
        }}
      >
        {chat.map((message, index) => renderMessage(message, index))}
        <div ref={messagesEndRef} />
      </div>
      
      {isLoading ? 
        <div style={{width: "100%", display: "flex", justifyContent: "center", marginBottom: 10}}>
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
      : null}
      
      <div 
        className="input-container" 
        style={{ 
          display: "flex",
          alignItems: "center"
        }}
      >
        {isRecording ? (
          <div style={{
            flex: 1,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: "68px",
            backgroundColor: "rgba(0, 0, 0, 0.2)",
            borderRadius: "8px",
            color: "white",
            padding: "0 12px"
          }}>
            <MdMic size={20} color="red" style={{ marginRight: "8px" }} />
            <span style={{ marginRight: "8px" }}>Recording</span>
            <span>{formatDuration(recordingTime)}</span>
          </div>
        ) : (
          <textarea
            ref={inputRef}
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Type your message..."
            className="text-areas"
            style={{
              flex: 1,
              resize: "none",
              border: "none",
              borderRadius: "8px",
              padding: "12px",
              background: "rgba(0, 0, 0, 0.2)",
              color: "white",
              minHeight: "44px",
              maxHeight: "120px",
              outline: "none",
              fontFamily: "inherit",
              fontSize: "15px"
            }}
          />
        )}
        <div style={{
          display: "flex",
          flexDirection: "column",
          marginLeft: 6
        }}>
        {/* Send button with tooltip during recording */}
        <Tooltip 
          text="Send Voice" 
          show={isRecording && showSendTooltip}
        >
          <button
            onClick={isRecording ? sendRecording : handleSendMessage}
            onMouseEnter={() => isRecording && setShowSendTooltip(true)}
            onMouseLeave={() => setShowSendTooltip(false)}
            style={{
              width: "32px",
              height: "32px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "8px",
              border: "none",
              background: "rgba(0, 0, 0, 0.35)",
              color: "white",
              cursor: "pointer",
              transition: "background-color 0.2s"
            }}
          >
            <IoIosSend size={20} />
          </button>
        </Tooltip>
        
        {/* Record/Cancel button */}
        {isRecording ? (
          <button
            onClick={cancelRecording}
            style={{
              marginTop: "4px",
              width: "32px",
              height: "32px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "8px",
              border: "none",
              background: "rgba(255, 0, 0, 0.5)",
              color: "white",
              cursor: "pointer",
              transition: "background-color 0.2s"
            }}
          >
            <ImCross size={12} />
          </button>
        ) : (
          <button
            onClick={handleRecordClick}
            style={{
              marginTop: "4px",
              width: "32px",
              height: "32px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "8px",
              border: "none",
              background: isCountingDown ? "rgba(255, 165, 0, 0.5)" : "rgba(0, 0, 0, 0.35)",
              color: "white",
              cursor: "pointer",
              transition: "background-color 0.2s",
              fontSize: isCountingDown ? "16px" : "inherit",
              fontWeight: isCountingDown ? "bold" : "normal"
            }}
          >
            {isCountingDown ? (
              countdownValue
            ) : (
              <FaMicrophone size={16} />
            )}
          </button>
        )}
        </div>
      </div>
    </div>
  );
};

export default Chat;
