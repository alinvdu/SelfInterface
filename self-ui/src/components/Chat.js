import React, { useState, useRef, useEffect } from "react";
import { GoArrowUp } from "react-icons/go";
import LoadingDiv from "./LoadingDiv";
import { BsThreeDots } from "react-icons/bs";

const ChatMessage = ({ message, token, api, onDeleteMessage, index }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const [showThreeDots, setShowThreeDots] = useState(false);

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
        position: 'relative'
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
      {message.content}
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
                    padding: "4px 0",
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

const Chat = ({ chat, onSendMessage, isLoading, token, api, onDeleteMessage }) => {
  const [newMessage, setNewMessage] = useState("");
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  // Focus input field when component mounts
  useEffect(() => {
    inputRef.current?.focus();
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
        <div style={{
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

    return <ChatMessage onDeleteMessage={onDeleteMessage} api={api} token={token} message={message} index={index} />
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
        {chat.map((message, index) => {
          return renderMessage(message, index)
        })}
        <div ref={messagesEndRef} />
      </div>
      {isLoading ? <div style={{width: "100%", display: "flex", justifyContent: "center", marginBottom: 10}}>
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
        </div> : null}
      <div 
        className="input-container" 
        style={{ 
          display: "flex", 
        }}
      >
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
        <button
          onClick={handleSendMessage}
          style={{
            marginLeft: "6px",
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
          <GoArrowUp size={22} />
        </button>
      </div>
    </div>
  );
};

export default Chat;
