import React, { useState, useEffect } from 'react';

const RotatingText = () => {
  const texts = ['Micro Expressions', 'Emotion Detection', 'Gesture Animations'];
  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setFade(false);
      setTimeout(() => {
        setIndex((prev) => (prev + 1) % texts.length);
        setFade(true);
      }, 500);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      style={{
        fontSize: 18,
        color: 'white',
        opacity: fade ? 0.7 : 0,
        transition: 'opacity 0.5s ease-in-out',
        width: '180px', // Adjust to match longest text
        textAlign: 'center',
        whiteSpace: 'nowrap',
      }}
    >
      {texts[index]}
    </div>
  );
};

export default RotatingText;
