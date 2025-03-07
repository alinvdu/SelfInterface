import React, { useState, useEffect, useRef } from 'react';
import { MdOutlineChevronLeft } from "react-icons/md";
import { MdOutlineChevronRight } from "react-icons/md";
import LoginButton from './LoginButton';
import { HiDotsVertical } from "react-icons/hi";

const CollapsibleMemoriesPanel = ({ token, requiresAccount, openedByDefault, api, children, expanded=false,
    toggleExpanded, canBeToggled=true, title = "Memories", onClear=null, toggleComponent = null,
    toggleLabel = "" }) => {
  const [isExpandedInternal, setIsExpandedInternal] = useState(openedByDefault);
  const [isAnimating, setIsAnimating] = useState(false);

  const isExpanded = expanded !== undefined && toggleExpanded ? expanded : isExpandedInternal;

  const toggleExpand = (overrideValue) => {
    setIsAnimating(true);
    
    if (toggleExpanded && expanded !== undefined) {
      // Use external toggle function if provided
      toggleExpanded(overrideValue !== undefined ? overrideValue : !expanded);
    } else {
      // Otherwise use internal state
      if (overrideValue !== undefined) {
        setIsExpandedInternal(overrideValue);
      } else {
        setIsExpandedInternal(!isExpandedInternal);
      }
    }
    
    // Reset animation state after animation completes
    setTimeout(() => {
      setIsAnimating(false);
    }, 300); // Match this with the CSS transition duration
  };

  useEffect(() => {
    if (!canBeToggled) {
      setIsExpandedInternal(false)
    }
  }, [canBeToggled])

  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setIsMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleClear = async () => {
    if (token) {
      try {
        const endpoint = title === "Memories" ? "clear_memories" : "clear_chat";
        const response = await fetch(`${api}/${endpoint}`, {
          method: "POST",
          headers: { 
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json"
          }
        });
        
        if (response.ok) {
          setIsMenuOpen(false);
          if (onClear) onClear(); // Call the onClear callback
        }
      } catch (error) {
        console.error(`Error clearing ${title.toLowerCase()}:`, error);
      }
    }
  };

  return (
    <div
      style={{
        width: isExpanded ? "350px" : "200px",
        marginBottom: "10px",
        transition: "width 0.3s ease-in-out",
        zIndex: 2,
        cursor: !isExpanded ? "pointer" : "default",
        perspective: "500px",
        maxHeight: "50%",
        display: "flex"
      }}
      onClick={!isExpanded && canBeToggled ? () => toggleExpand() : null}
    >
        <div
        style={{
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          background: 'rgba(0, 0, 0, 0.25)',
          border: "1px solid rgba(255, 255, 255, 0.35)",
          borderRadius: "16px",
          padding: isExpanded ? "21px" : "12px 8px",
          width: "100%",
          height: isExpanded ? "auto" : "fit-content",
          maxHeight: isExpanded ? "400px" : "200px", // Limit height for multiple panels
          color: "white",
          display: "flex",
          flexDirection: "column",
          alignItems: isExpanded ? "flex-start" : "center",
          transition: "all 0.3s ease-in-out, transform 0.3s ease-in-out",
          transform: isExpanded ? "rotateY(0deg)" : "rotateY(20deg)",
          transformOrigin: "left center",
          transformStyle: "preserve-3d", // Important for nested 3D transforms
          boxShadow: isExpanded 
            ? "0 4px 30px rgba(0, 0, 0, 0.1)" 
            : "5px 0 15px rgba(0, 0, 0, 0.2)",
        }}
      >
        {isExpanded ? (
          <>
            <div style={{ 
              display: "flex", 
              width: "100%", 
              justifyContent: "space-between", 
              alignItems: "center",
              marginBottom: 15 
            }}>
              <div style={{ fontSize: 21, borderRadius: 6 }}>
                {title}
              </div>
              <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                {toggleComponent && (
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    marginRight: '15px',
                    cursor: 'default',
                    fontSize: '14px'
                  }}>
                    {toggleLabel && (
                      <span style={{ marginRight: '8px', color: 'rgba(255,255,255,0.9)' }}>
                        {toggleLabel}
                      </span>
                    )}
                    {toggleComponent}
                  </div>
                )}
                <button 
                  style={{
                    background: "none",
                    border: "none",
                    color: "white",
                    cursor: "pointer",
                    padding: "4px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity: 0.8,
                    transition: "opacity 0.2s",
                    borderRadius: 3
                  }}
                  onClick={() => toggleExpand()}
                  onMouseOver={(e) => {
                    e.currentTarget.style.opacity = 1
                    e.currentTarget.style.background = "rgba(0, 0, 0, 0.15)"
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.opacity = 0.8
                    e.currentTarget.style.background = "none"
                  }
                }
                >
                  <MdOutlineChevronLeft size={18} />
                </button>
                {/* Add the three-dot menu */}
                <div 
                  style={{ cursor: "pointer" }}
                  onClick={(e) => {
                    e.stopPropagation(); // Prevent panel from toggling
                    setIsMenuOpen(!isMenuOpen);
                  }}
                >
                  <div style={{
                    background: "none",
                    border: "none",
                    color: "white",
                    cursor: "pointer",
                    padding: "4px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity: 0.8,
                    transition: "opacity 0.2s",
                    borderRadius: 3,
                    marginTop: 0
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.opacity = 1
                    e.currentTarget.style.background = "rgba(0, 0, 0, 0.15)"
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.opacity = 0.8
                    e.currentTarget.style.background = "none"
                  }}
                  >
                    <HiDotsVertical />
                  </div>
                  
                  {/* Dropdown menu */}
                  {isMenuOpen && (
                    <div 
                      ref={menuRef}
                      style={{
                        position: "absolute",
                        top: "50px",
                        right: "-20px",
                        background: "rgba(255, 255, 255, 0.85)",
                        border: "1px solid rgba(255, 255, 255, 0.95)",
                        borderRadius: "5px",
                        padding: "5px 0",
                        zIndex: 100,
                        minWidth: "120px",
                        color: "black",
                        fontSize: 14
                      }}
                    >
                      <div 
                        style={{
                          padding: "4px 6px",
                          cursor: "pointer",
                          hover: { background: "rgba(255, 255, 255, 0.1)" }
                        }}
                        onClick={handleClear}
                      >
                        Clear {title}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            <div style={{ 
              width: "100%", 
              opacity: isAnimating ? 0 : 1,
              transition: "opacity 0.15s ease-in-out",
              overflow: "auto"
            }}>
              {requiresAccount && !token ? <LoginButton /> : children}
            </div>
          </>
        ) : (
          <button
            style={{
              background: "none",
              border: "none",
              color: "white",
              cursor: "pointer",
              padding: "4px 0px",
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              gap: "12px",
              opacity: 0.8,
              transition: "opacity 0.2s",
              height: "100%",
              justifyContent: "center",
            }}
            onMouseOver={(e) => e.currentTarget.style.opacity = 1}
            onMouseOut={(e) => e.currentTarget.style.opacity = 0.8}
          >
            <div style={{ 
              transform: "rotate(0deg)",
              letterSpacing: "1px",
              textTransform: "uppercase",
              fontSize: "12px",
              fontWeight: "500",
              display: "flex",
              flexDirection: "row",
            }}>
              <span style={{
                display: "inline-block",
                whiteSpace: "nowrap",
                transformOrigin: "center center",
                fontSize: 15
              }}>
                {title}
              </span>
            </div>
            <MdOutlineChevronRight size={20} />
          </button>
        )}
      </div>
    </div>
  );
};

export default CollapsibleMemoriesPanel;
// Also export the container component
