import React, { useEffect, useState } from 'react';

const LoadingDots = ({ color = 'white', size = 16 }) => {
  const [dotStates, setDotStates] = useState([0, 0, 0]);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setDotStates(prevState => {
        // Calculate which dot should be "up" in the animation sequence
        const activeDotIndex = (prevState.indexOf(1) + 1) % 3;
        return [0, 0, 0].map((_, index) => index === activeDotIndex ? 1 : 0);
      });
    }, 300); // Control animation speed
    
    return () => clearInterval(interval);
  }, []);
  
  // Container styles
  const containerStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    height: '100%',
    borderRadius: '8px',
  };
  
  // Dots container style
  const dotsContainerStyle = {
    display: 'flex',
    gap: '8px',
  };
  
  return (
    <div style={containerStyle}>
      <div style={dotsContainerStyle}>
        {dotStates.map((state, index) => {
          // Individual dot style
          const dotStyle = {
            width: `${size}px`,
            height: `${size}px`,
            backgroundColor: color,
            borderRadius: '50%',
            transition: 'transform 300ms ease-in-out',
            transform: state === 1 ? `translateY(-${size * 0.75}px)` : 'translateY(0)',
          };
          
          return <div key={index} style={dotStyle} />;
        })}
      </div>
    </div>
  );
};

export default LoadingDots;
