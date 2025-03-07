// In ./components/Switch.js
import React from 'react';

export function Switch({ isChecked, onChange, isDisabled, isLoading = false }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center' }}>
      <div
        style={{
          position: 'relative',
          width: '40px',
          height: '20px',
          backgroundColor: isLoading 
            ? 'rgba(150, 150, 150, 0.4)' 
            : isChecked 
              ? 'rgba(0, 122, 255, 0.8)' 
              : 'rgba(80, 80, 80, 0.32)',
          borderRadius: '10px',
          transition: 'background-color 0.2s',
          cursor: isLoading || isDisabled ? 'not-allowed' : 'pointer',
          opacity: isLoading ? 0.7 : isDisabled ? 0.6 : 1,
          overflow: 'hidden'
        }}
        onClick={() => {
          if (!isDisabled && !isLoading) {
            onChange(!isChecked);
          }
        }}
      >
        {/* Loading animation */}
        {isLoading && (
          <div 
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              height: '100%',
              width: '200%',
              background: 'linear-gradient(to right, transparent, rgba(255,255,255,0.3), transparent)',
              animation: 'shimmer 1.5s infinite',
            }}
          />
        )}
        
        <div
          style={{
            position: 'absolute',
            top: '2px',
            left: isLoading ? '11px' : isChecked ? '22px' : '2px',
            width: '16px',
            height: '16px',
            backgroundColor: 'white',
            borderRadius: '50%',
            transition: 'left 0.2s',
            opacity: isLoading ? 0.7 : 1
          }}
        />
      </div>
      
      {/* Add keyframe animation for the shimmer effect */}
      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
}

export default Switch;
