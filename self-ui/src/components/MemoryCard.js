import { useEffect, useRef, useState } from "react";
import { BsThreeDots } from "react-icons/bs";

function MemoryCard({ memory, hue, token, api, onDelete }) {
    const [expanded, setExpanded] = useState(false);
    const previewLimit = 100;
    const text = typeof memory === "string" ? memory : memory.text;
    const category =
      typeof memory === "string"
        ? "General"
        : memory.category.split("_").join(" ") || "General";
    const previewText =
      text.length > previewLimit ? text.slice(0, previewLimit) + "..." : text;
  
    const cardStyle = {
      border: `1px solid rgba(255, 255, 255, 0.4)`,
      borderRadius: "5px",
      background: 'rgba(255, 255, 255, 0.65)',
      fontSize: 14,
      padding: "12px 12px 8px 8px",
      textAlign: "left",
      color: 'black',
      cursor: "pointer",
    };
  
    const tagStyle = {
      border: `1px solid rgba(255, 255, 255, 0.3)`,
      borderRadius: "5px",
      background: 'rgba(255, 255, 255, 0.55)',
      fontSize: "0.8rem",
      padding: "3px",
      marginTop: 3,
      color: `black`,
    };

    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [showThree, showThreeDots] = useState(false);
    const menuRef = useRef(null);

    useEffect(() => {
        function handleClickOutside(event) {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setIsMenuOpen(false);
                showThreeDots(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    const handleDelete = async () => {
        if (token && typeof onDelete === 'function') {
            try {
            const response = await fetch(`${api}/delete_memory`, {
                method: "POST",
                headers: {
                Authorization: `Bearer ${token}`,
                "Content-Type": "application/json"
                },
                body: JSON.stringify({ 
                memory_id: memory.id || null,
                text: memory.text,
                category: memory.category
                })
            });
            
            if (response.ok) {
                onDelete(memory);
                setIsMenuOpen(false);
            }
            } catch (error) {
            console.error("Error deleting memory:", error);
            }
        }
    };
  
    return (
      <div
        style={{
          marginBottom: "1.5rem",
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          position: "relative"
        }}
      >
        <div style={cardStyle} onMouseOver={() => {
            showThreeDots(true)
        }} onMouseOut={() => {
            if (!isMenuOpen) {
                showThreeDots(false)
            }
        }} onClick={() => setExpanded(!expanded)}>
            {showThree &&
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
                    onClick={handleDelete}
                    >
                    Delete Memory
                    </div>
                </div>
                )}
            </div>}
          <div>{expanded ? text : previewText}</div>
        </div>
        <div style={tagStyle}>{category}</div>
      </div>
    );
  }

  export default MemoryCard;
