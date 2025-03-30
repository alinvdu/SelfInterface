import React, { useEffect, useState } from 'react';

const TypingText = ({ texts, colors, typingSpeed = 100, eraseSpeed = 60, pauseTime = 1500, fontSize = 35 }) => {
  const [currentTextIndex, setCurrentTextIndex] = useState(0);
  const [displayedText, setDisplayedText] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const handleTyping = () => {
      const fullText = texts[currentTextIndex];

      setDisplayedText(prev => {
        if (isDeleting) {
          return fullText.substring(0, prev.length - 1);
        }
        return fullText.substring(0, prev.length + 1);
      });

      let timeout = isDeleting ? eraseSpeed : typingSpeed;

      if (!isDeleting && displayedText === fullText) {
        timeout = pauseTime;
        setTimeout(() => setIsDeleting(true), timeout);
      } else if (isDeleting && displayedText === '') {
        setIsDeleting(false);
        setCurrentTextIndex((prev) => (prev + 1) % texts.length);
      }
    };

    const timer = setTimeout(handleTyping, isDeleting ? eraseSpeed : typingSpeed);

    return () => clearTimeout(timer);
  }, [displayedText, isDeleting, currentTextIndex, texts, typingSpeed, eraseSpeed, pauseTime]);

  const textColor = colors[currentTextIndex % colors.length];

  return (
    <div style={{
      fontSize: `${fontSize}px`,
      color: textColor,
      display: 'inline-block',
      fontWeight: "bold"
    }}>
      {displayedText}
      <span className="cursor">|</span>
    </div>
  );
};

export default TypingText;
