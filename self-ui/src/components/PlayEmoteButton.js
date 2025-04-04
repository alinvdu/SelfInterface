import React, { useState, useEffect, useRef } from 'react';
import { PiPlayCircleLight } from "react-icons/pi";

const animationDurations = {
  'happy': 6000,
  'sad': 6000,
  'anger': 7000,
  'disappointment': 7000
  // You can add more durations here
};

const PlayEmoteButton = ({ emoteType, onClick }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef(null);

  const getEmoteName = (type) => {
    const emoteNames = {
      'happy': 'Happy',
      'sad': 'Sad',
      'anger': 'Angry',
      'disappointment': 'Disappointed'
    };
    return emoteNames[type] || type;
  };

  const handleClick = () => {
    const duration = animationDurations[emoteType] || 3000;
    setIsPlaying(true);
    setProgress(0);
    onClick && onClick();

    const startTime = Date.now();
    intervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const newProgress = Math.min((elapsed / duration) * 100, 100);
      setProgress(newProgress);

      if (newProgress >= 100) {
        clearInterval(intervalRef.current);
        setIsPlaying(false);
        setProgress(0);
      }
    }, 100);
  };

  useEffect(() => {
    return () => clearInterval(intervalRef.current);
  }, []);

  return (
    <button
      className="play-emote-button"
      onClick={handleClick}
      title={`Play ${getEmoteName(emoteType)} animation`}
      style={{
        display: 'inline-flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0, 0, 0, 0.45)',
        backdropFilter: 'blur(4px)',
        borderRadius: '12px',
        padding: '4px 6px',
        border: "none",
        cursor: isPlaying ? 'not-allowed' : 'pointer',
        marginLeft: '8px',
        transition: 'all 0.2s ease',
        fontSize: '13px',
        marginTop: 10,
        opacity: isPlaying ? 0.6 : 1,
        position: 'relative',
        color: "white"
      }}
      disabled={isPlaying}
      onMouseOver={(e) => {
        if (!isPlaying) {
          e.currentTarget.style.background = 'rgba(0, 0, 0, 0.55)';
        }
      }}
      onMouseOut={(e) => {
        if (!isPlaying) {
          e.currentTarget.style.background = 'rgba(0, 0, 0, 0.45)';
        }
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <PiPlayCircleLight style={{ marginRight: '3px', fontSize: 19 }} />
        <span>Expression</span>
      </div>
      {isPlaying && (
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 6,
          height: '2px',
          background: 'white',
          width: `calc(${progress}% - 12px)`,
          borderBottomLeftRadius: '12px',
          borderBottomRightRadius: progress === 100 ? '12px' : 0,
          transition: 'width 0.1s linear'
        }} />
      )}
    </button>
  );
};

export default PlayEmoteButton;
