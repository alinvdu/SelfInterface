import React, { useState, useRef, useEffect } from "react";
import { GoArrowUp } from "react-icons/go";
import LoadingDiv from "./LoadingDiv";

const Chat = ({ chat, onSendMessage, isLoading }) => {
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
          {`Phone call conversation (${message.duration})`}
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
            textAlign: "left"
          }}
        >
          {message.content}
        </div>
    );
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
