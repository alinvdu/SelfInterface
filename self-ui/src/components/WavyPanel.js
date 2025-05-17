import React from 'react';
import { CiStar } from "react-icons/ci";
import { IoIosCheckmarkCircleOutline } from "react-icons/io";
import LoadingDots from './LoadingDots';


/**
 * WavePanel – balanced tone: darker base with subtle but noticeable color.
 */
const WavePanel = ({ selectedModel, setSelectedModel, updateModel, isUpdatingModel, apiSelectedModel, setShowModelSelectionScreen, windowWidth, isMobile }) => {
  const waveAnimation = `
    @keyframes wave {
      0%   { transform: translateX(0); }
      50%  { transform: translateX(-20px); }
      100% { transform: translateX(0); }
    }
  `;

  return (
    <div
      style={{
        position: 'absolute', inset: 0, zIndex: 999,
        background: 'rgba(3,3,8,0.65)',
        backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        style={{
          position: 'relative', width: windowWidth < 1140 ? 800 : 1100, height: isMobile ? "calc(100% - 20px)" : 780,
          margin: windowWidth < 820 ? 15 : 0,
          padding: windowWidth < 820 ? 10 : 0,
          boxSizing: 'border-box',
          maxWidth: "100%",
          borderRadius: 20,
          background: 'rgba(0,0,0,1)',
          backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
          border: '1px solid rgba(255,255,255,0.15)',
          overflow: 'hidden', boxShadow: '0 8px 40px rgba(0,0,0,0.5)'
        }}
      >
        <style>{waveAnimation}</style>

        {/* balanced gradients */}
        <svg style={{ height: 0 }}>
          <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="#5054A5" stopOpacity="0.55" />
              <stop offset="100%" stopColor="#6A6FB1" stopOpacity="0.5" />
            </linearGradient>
            <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="#49527D" stopOpacity="0.42" />
              <stop offset="100%" stopColor="#5E6296" stopOpacity="0.38" />
            </linearGradient>
            <linearGradient id="grad3" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="#64689C" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#4D507A" stopOpacity="0.26" />
            </linearGradient>
          </defs>
        </svg>

        {/* ---------- TOP WAVES (flipped) ---------- */}
        <svg
          style={{ position: 'absolute', top: -40, left: 0, width: '110%', height: 180, transform: 'scaleY(-1)', zIndex: 1 }}
          viewBox="0 0 1500 180" preserveAspectRatio="none"
        >
          {[
            { dY: 100, grad: 'grad1', dur: 4 },
            { dY: 110, grad: 'grad2', dur: 3.5 },
            { dY: 120, grad: 'grad3', dur: 3 }
          ].map(({ dY, grad, dur }, i) => (
            <path key={i}
              d={`M0,${dY} C 250,${dY-100} 500,${dY+100} 750,${dY} C 1000,${dY-100} 1250,${dY+100} 1500,${dY} L1500,180 L0,180 Z`}
              fill={`url(#${grad})`}
              style={{ animation: `wave ${dur}s infinite ease-in-out` }}
            />
          ))}
        </svg>

        {/* ---------- BOTTOM WAVES ---------- */}
        <svg
          style={{ position: 'absolute', bottom: -60, left: 0, width: '110%', height: 200, zIndex: 1 }}
          viewBox="0 0 1500 200" preserveAspectRatio="none"
        >
          {[
            { dY: 120, grad: 'grad1', dur: 8 },
            { dY: 130, grad: 'grad2', dur: 7 },
            { dY: 140, grad: 'grad3', dur: 6 }
          ].map(({ dY, grad, dur }, i) => (
            <path key={i}
              d={`M0,${dY} C 250,${dY-60} 500,${dY+60} 750,${dY} C 1000,${dY-60} 1250,${dY+60} 1500,${dY} L1500,200 L0,200 Z`}
              fill={`url(#${grad})`}
              style={{ animation: `wave ${dur}s infinite ease-in-out` }}
            />
          ))}
        </svg>
        <div style={{zIndex: 999, position: "relative", display: "flex", flexDirection: "column", height: isMobile ? "calc(100% - 40px)" : "auto"}}>
          <div style={{
            marginTop: 52,
            color: 'rgba(255, 255, 255, 0.88)',
            opacity: 0.88,
          }}>
            <div style={{
              fontSize: isMobile ? 20 : 27,
            }}><span style={{ fontWeight: 600 }}>Psychologist</span> Model Selection</div>
            <div style={{
              marginTop: 6,
              fontSize: isMobile ? 14 : 16,
            }}>Explore your true Self. Let go of what holds you back. Select what's best for you.</div>
          </div>
          <div style={{
            marginTop: 25,
            boxSizing: "border-box",
            display: "flex",
            flexDirection: isMobile ? "column" : "row",
            justifyContent: isMobile ? "flex-start" : "center",
            overflow: isMobile ? "auto" : "visible",
            alignItems: "center"
          }}>
            <div onMouseOver={(e) => {
              e.currentTarget.style.transform = "scale(1.05)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = "scale(1)";
            }} style={{
              width: isMobile ? "90%" : "50%",
              transition: "transform 0.3s ease",
              maxWidth: 350,
              display: "flex",
              justifyContent: "center",
              flexDirection: "column",
              alignItems: "center",
              border: selectedModel === "Atlas" ? "3px solid #AD60FF" : "3px solid rgb(106, 108, 134)",
              borderRadius: 15,
              padding: 25,
              boxSizing: "border-box",
              position: "relative",
              paddingBottom: 75
            }}>
              <div style={{
                position: "absolute",
                top: 5,
                right: 5,
                width: 140,
                height: 28,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                borderRadius: 8,
                background: "#4D507A",
                fontSize: 14
              }}>
                <CiStar style={{
                  fontSize: 17,
                  color: "white",
                  marginRight: 5,
                }} />
                <span>Recommended</span>
              </div>
              <div style={{height: 100}}>
                <img src="philosophical-icon.png" style={{
                  width: 80
                }} />
              </div>
              <div style={{
                fontSize: 22,
                opacity: 0.85
              }}>
                Atlas
              </div>
              <div style={{
                fontSize: 16,
                opacity: 0.7,
                marginTop: 25,
                paddingBottom: 20,
                borderBottom: '1px dotted rgba(255, 255, 255, 0.75)',
                width: "100%"
              }}>
                Psychology-oriented conversational base.
              </div>
              <div style={{
                fontSize: 16,
                opacity: 0.7,
                marginTop: 25,
                paddingBottom: 20,
                borderBottom: '1px dotted rgba(255, 255, 255, 0.75)',
                width: "100%"
              }}>
                Fine-tuned on a currated list of introspective and metaphysical texts.
              </div>
              <div style={{
                fontSize: 16,
                opacity: 0.7,
                marginTop: 25
              }}>
                Best for exploring your inner self, the connection with the universe and consciousness.
              </div>
              <div style={{
                position: "absolute",
                left: "50%",
                transform: "translateX(-50%)",
                bottom: 20,
                background: "#5054A5",
                border: "1px solid rgba(255, 255, 255, 0.25)",
                width: 100,
                height: 32,
                borderRadius: 5,
                fontSize: 12,
                letterSpacing: "0.5px",
                color: "white",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                background: selectedModel === "Atlas" ? "#A36DF7" : "#5054A5",
                border: "1px solid rgba(255, 255, 255, 0.25)",
                width: selectedModel === "Atlas" ? 120 : 100,
              }} onClick={() => {
                setSelectedModel("Atlas")
              }}>
                {selectedModel === "Atlas" ? <IoIosCheckmarkCircleOutline style={{
                  fontSize: 21,
                  color: "white",
                  marginRight: 5
                }} /> : null}
                <span>
                {selectedModel === "Atlas" ? "SELECTED" : "SELECT"}
                </span>
              </div>
            </div>
            <div onMouseOver={(e) => {
              e.currentTarget.style.transform = "scale(1.05)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = "scale(1)";
            }} style={{
              width: isMobile ? "90%" : "50%",
              display: "flex",
              maxWidth: 350,
              marginLeft: isMobile ? 0 : 55,
              justifyContent: "center",
              flexDirection: "column",
              marginTop: isMobile ? 25 : 0,
              alignItems: "center",
              border: selectedModel === "Echo" ? "3px solid #AD60FF" : "3px solid rgb(106, 108, 134)",
              transition: "transform 0.3s ease",
              borderRadius: 15,
              padding: 25,
              boxSizing: "border-box",
              position: "relative",
              paddingBottom: 75
            }}>
              <div style={{height: 100}}>
                <img src="fun-icon.png" style={{
                  marginTop: 13,
                  width: 60
                }} />
              </div>
              <div style={{
                fontSize: 22,
                opacity: 0.85
              }}>
                Echo
              </div>
              <div style={{
                fontSize: 16,
                opacity: 0.7,
                marginTop: 25,
                paddingBottom: 20,
                borderBottom: '1px dotted rgba(255, 255, 255, 0.75)',
                width: "100%"
              }}>
                Psychology-oriented conversational base.
              </div>
              <div style={{
                fontSize: 16,
                opacity: 0.7,
                marginTop: 25,
                paddingBottom: 20,
                borderBottom: '1px dotted rgba(255, 255, 255, 0.75)',
                width: "100%"
              }}>
                Fine-tuned on patient-therapist conversations
              </div>
              <div style={{
                fontSize: 16,
                opacity: 0.7,
                marginTop: 25
              }}>
                Best for relaxed, supportive chats focused on relationships and self-reflection.
              </div>
              <div style={{
                position: "absolute",
                left: "50%",
                transform: "translateX(-50%)",
                bottom: 20,
                background: selectedModel === "Echo" ? "#A36DF7" : "#5054A5",
                border: "1px solid rgba(255, 255, 255, 0.25)",
                width: selectedModel === "Echo" ? 120 : 100,
                height: 32,
                borderRadius: 5,
                fontSize: 12,
                letterSpacing: "0.5px",
                color: "white",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer"
              }} onClick={() => setSelectedModel("Echo")}>
                {selectedModel === "Echo" ? <IoIosCheckmarkCircleOutline style={{
                  fontSize: 21,
                  color: "white",
                  marginRight: 5
                }} /> : null}
                <span>
                {selectedModel === "Echo" ? "SELECTED" : "SELECT"}
                </span>
              </div>
            </div>
          </div>
          <div style={{
            marginTop: 25,
            color: 'rgba(255, 255, 255, 0.85)',
            opacity: 0.88,
          }}>
            <div style={{
              marginTop: 6,
              fontSize: isMobile ? 14 : 16,
            }}>Model selection can be changed later in Settings. For now, pick one.</div>
          </div>
          <div style={{
            marginTop: 12,
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}>
            <div style={{
              background: !selectedModel ? "rgba(255, 255, 255, 0.35)" : "rgba(255, 255, 255, 0.85)",
              border: !selectedModel ?  "1px solid rgba(255, 255, 255, 0.1)" : "1px solid rgba(255, 255, 255, 0.55)",
              width: 120,
              height: 38,
              borderRadius: 5,
              fontSize: 15,
              letterSpacing: "0.5px",
              color: !selectedModel ? "rgba(255, 255, 255, 0.25)" : "black",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: !selectedModel ? "not-allowed" : "pointer"
            }} onClick={() => {
              if (!selectedModel || isUpdatingModel) {
                return
              }

              if (apiSelectedModel && apiSelectedModel === selectedModel) {
                setShowModelSelectionScreen(false)
                return
              }

              updateModel(selectedModel)
            }}>
              {!isUpdatingModel ?
              <span>
              {!apiSelectedModel ? "Finish Setup" : "Save"}
              </span> : <LoadingDots size={7} color="rgba(20, 20, 20, 0.55)" />}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WavePanel;
