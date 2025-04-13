import React from 'react';

const SaveDialog = ({ onSave, onCancel }) => {
    return (
      <div style={{
        position: 'absolute',
        bottom: 150,
        left: "50%",
        transform: 'translateX(-50%)',
        width: "220px",
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        background: 'rgba(0, 0, 0, 0.35)',
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        border: "1px solid rgba(255, 255, 255, 0.4)",
        borderRadius: "12px",
        padding: "12px",
        color: "white",
        flexDirection: "column",
      }}>
        <h3 style={{ color: 'white', marginTop: 0 }}>Save Conversation</h3>
        <p style={{ color: 'white' }}>Would you like to save this conversation as a JSON file?</p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '20px' }}>
        <button 
            style={{
            background: 'rgba(100, 150, 255, 1)',
            border: 'none',
            borderRadius: '6px',
            padding: '8px 16px',
            color: 'white',
            cursor: 'pointer'
            }}
            onClick={onSave}
        >
            Yes
        </button>
        <button 
            style={{
            background: 'rgba(100, 100, 100, 0.5)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '6px',
            padding: '8px 16px',
            color: 'white',
            cursor: 'pointer'
            }}
            onClick={onCancel}
        >
            No
        </button>
        </div>
    </div>
    );
  };

  export default SaveDialog;
