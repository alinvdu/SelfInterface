import { memo, useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { GiBrain } from 'react-icons/gi';
import { BsArrowRight, BsArrowLeft } from 'react-icons/bs';
import { BiConversation } from 'react-icons/bi';
import { RiEmotionHappyLine } from 'react-icons/ri';
import { MdOutlinePsychology } from 'react-icons/md';
import { IoIosArrowBack, IoIosArrowForward } from 'react-icons/io';
import TypingText from './TypingText';

const IntroductionPanel = memo(({ handleIntroScroll, introSectionsVisible, handleStartApp }) => {
  // Slider state
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const sliderInterval = useRef(null);
  
  // Slider content - extracted from the original divs
  const slides = [
    {
      icon: BiConversation,
      title: "Real-Time Psychological Conversations",
      content: (
        <div style={{
          marginLeft: 23,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column"
        }}>
        <div style={{ color: "rgba(255, 255, 255, 0.8)", fontSize: "1.1rem", maxWidth: 520, lineHeight: 1.6, textAlign: "left", marginTop: 8  }}>
            Fine-tuned on patient-therapist conversations and philosophy.
        </div>
          <video
            width={280}
            controls
            autoPlay={true}
            loop
            muted
            style={{
              borderRadius: 16,
              marginTop: 25,
              border: "1px solid rgba(255, 255, 255, 0.75)"
            }}
          >
            <source src="showcase.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      )
    },
    {
      icon: RiEmotionHappyLine,
      title: "Facial Expressions and Audio Cues Recognition",
      content: (
        <div style={{
          marginLeft: 23,
          display: "flex",
          marginTop: 22,
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column"
        }}>
            <div style={{
                display: "flex"
            }}>
                <img style={{
                    width: 150,
                    borderRadius: 16,
                    border: "1px solid rgba(255, 255, 255, 0.35)"
                }} src="face-recognition-graphic.png" alt="Face recognition" />
                <div style={{display: "flex", flexDirection: "column", marginLeft: 10}}>
                    <div style={{background: 'rgba(255, 255, 255, 0.85)', color: "black", display: "flex", alignItems: "center", justifyContent: "center", width: 130, height: 32, fontSize: 14, borderRadius: 5}}>
                        Happines (100%)
                    </div>
                    <div style={{background: 'rgba(255, 255, 255, 0.85)', marginTop: 5, color: "black", display: "flex", alignItems: "center", justifyContent: "center", width: 130, height: 32, fontSize: 14, borderRadius: 5}}>
                        Amusement (75%)
                    </div>
                    <div style={{background: 'rgba(255, 255, 255, 0.85)', marginTop: 5, color: "black", display: "flex", alignItems: "center", justifyContent: "center", width: 130, height: 32, fontSize: 14, borderRadius: 5}}>
                        Joy (85%)
                    </div>
                </div>
            </div>
          <div style={{display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", marginTop: 15, marginLeft: 15}}>
            <div style={{ color: "rgba(255, 255, 255, 0.8)", fontSize: "1.1rem", maxWidth: 520, lineHeight: 1.6, textAlign: "center", marginTop: 8  }}>
              Facial Expressions and Visual Cues recognition powered by Hume, integrated into real-time conversation.
            </div>
          </div>
        </div>
      )
    },
    {
      icon: MdOutlinePsychology,
      title: "Memory-Based Psychological Insights",
      content: (
        <div style={{
          marginLeft: 23,
          display: "flex",
          marginTop: 22,
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column"
        }}>
          <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center"
          }}>
            <div style={{
              backdropFilter: "blur(12px)",
              WebkitBackdropFilter: "blur(12px)",
              background: 'rgba(0, 0, 0, 0.25)',
              border: "1px solid rgba(255, 255, 255, 0.35)",
              borderRadius: "16px",
              padding: 16,
              maxWidth: 300,
              fontSize: 15,
              textAlign: "left",
              lineHeight: "18px",
              color: "white"
            }}>
              The user experiences frustration when confronting psychological limitations, reflecting an underlying tension between ambition and perceived constraints.
            </div>
            <div style={{
              display: "flex"
            }}>
              <div style={{background: 'rgba(255, 255, 255, 0.85)', marginTop: 8, color: "black", display: "flex", alignItems: "center", justifyContent: "center", width: 80, height: 32, fontSize: 14, borderRadius: 16}}>
                Insight
              </div>
              <div style={{background: 'rgba(255, 255, 255, 0.85)', marginTop: 8, marginLeft: 8, color: "black", display: "flex", alignItems: "center", justifyContent: "center", width: 80, height: 32, fontSize: 14, borderRadius: 16}}>
                Memory
              </div>
            </div>
          </div>
          <div style={{display: "flex", flexDirection: "column", marginLeft: 15}}>
            <div style={{ color: "rgba(255, 255, 255, 0.8)", fontSize: "1.1rem", maxWidth: 580, lineHeight: 1.6, textAlign: "center", marginTop: 15  }}>
              Conversation summarization and psychological insights are kept up to date to increase engagement and build a profile.
            </div>
          </div>
        </div>
      )
    }
  ];

  // Auto-rotate slides unless paused
  useEffect(() => {
    if (!isPaused) {
      sliderInterval.current = setInterval(() => {
        setCurrentSlide((prev) => (prev + 1) % slides.length);
      }, 5000); // Change slide every 5 seconds
    }
    
    return () => {
      if (sliderInterval.current) {
        clearInterval(sliderInterval.current);
      }
    };
  }, [isPaused, slides.length]);

  // Navigation handlers
  const goToPrevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  const goToNextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const goToSlide = (index) => {
    setCurrentSlide(index);
  };

  const CurrentIcon = slides[currentSlide].icon;

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="introduction-panel"
      transition={{ duration: 0.6 }}
      exit={{ opacity: 0, transition: { duration: 0 } }}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100vh",
        background: "radial-gradient(circle at center, rgba(15, 15, 15, 0.65) 10%, rgba(30, 30, 30, 1) 75%)",
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        padding: "25px 2rem",
        boxSizing: "border-box",
        overflowY: "auto",
        paddingRight: "45%"
      }}
      onScroll={handleIntroScroll}
    >
      <div style={{
        display: "flex",
        width: "100%",
        justifyContent: "space-between"
      }}>
        <div style={{ 
          display: "flex", 
          alignItems: "center", 
          marginBottom: "2rem" 
        }}>
          <GiBrain style={{
            fontSize: 38,
            color: "white",
            marginRight: 10
          }} />
          <div style={{ 
            color: "white", 
            fontSize: "23px",
            margin: 0
          }}>
            Self AI
          </div>
        </div>
      </div>
      <div style={{
        display: "flex",
        flexDirection: "column",
        position: "relative",
        justifyContent: "flex-end"
      }}>
        <div style={{right: 0, marginTop: 120,}}>
          <TypingText texts={[
            "Psychology, unlocked!",
            "Powerful and expressive!",
            "Real-time with deep conversations"
          ]} colors={[
            "rgb(151, 205, 223)",
            "rgb(151, 160, 223)",
            "rgb(223, 151, 222)"
          ]} />
        </div>
        <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
        }}>
            <div style={{ 
                display: "flex", 
                justifyContent: "center",
                marginTop: 25,
                marginRight: 15
            }}>
            <button 
                style={{
                backgroundColor: "rgba(255, 255, 255, 0.9)",
                color: "#000",
                border: "none",
                padding: "0.8rem 1.5rem",
                borderRadius: "2rem",
                fontSize: "16px",
                fontWeight: "600",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                transition: "all 0.2s ease",
                height: 50
                }}
                onMouseOver={(e) => {
                e.currentTarget.style.transform = "scale(1.05)";
                e.currentTarget.style.backgroundColor = "#fff";
                }}
                onMouseOut={(e) => {
                e.currentTarget.style.transform = "scale(1)";
                e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.9)";
                }}
                onClick={handleStartApp}
            >
                Create account <BsArrowRight style={{ marginLeft: "0.5rem", fontSize: "1.2rem" }} />
            </button>
            </div>
            <div style={{ 
                display: "flex", 
                justifyContent: "center",
                marginTop: 25
            }}>
            <button 
                style={{
                background: "transparent",
                border: "1px solid white",
                color: "white",
                padding: "0.8rem 1.5rem",
                borderRadius: "2rem",
                fontSize: "16px",
                fontWeight: "600",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                transition: "all 0.2s ease",
                height: 50
                }}
                onMouseOver={(e) => {
                    e.currentTarget.style.transform = "scale(1.05)";
                }}
                onMouseOut={(e) => {
                    e.currentTarget.style.transform = "scale(1)";
                }}
                onClick={handleStartApp}
            >
                Try without logging <BsArrowRight style={{ marginLeft: "0.5rem", fontSize: "1.2rem" }} />
            </button>
            </div>
        </div>
        
        {/* Slider Section */}
        <div 
          style={{
            marginTop: 50,
            padding: 25,
            boxSizing: "border-box",
            width: "100%",
            position: "relative",
            minHeight: 400 // Added minimum height to maintain consistency
          }}
          onMouseEnter={() => setIsPaused(true)}
          onMouseLeave={() => setIsPaused(false)}
        >
          {/* Slider Container with Animation */}
          <motion.div
            key={currentSlide}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            style={{
              justifyContent: "flex-end",
              display: "flex",
              boxSizing: "border-box",
              width: "100%",
              flexDirection: "column",
              minHeight: 300 // Ensure consistent height
            }}
          >
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center"
            }}>
              <div style={{ color: "white", fontSize: "1.8rem", textAlign: "center", marginLeft: 10, maxWidth: 550 }}>
                {slides[currentSlide].title}
              </div>
            </div>
            
            {slides[currentSlide].content}
          </motion.div>
          
          {/* Fixed Navigation Arrows */}
          <div style={{
            position: "absolute",
            width: "100%",
            left: 0,
            top: "50%",
            transform: "translateY(-50%)",
            display: "flex",
            justifyContent: "space-between",
            padding: "0 10px",
            boxSizing: "border-box",
            zIndex: 20,
            pointerEvents: "none"
          }}>
            <button
              onClick={goToPrevSlide}
              style={{
                background: "rgba(0, 0, 0, 0.5)",
                color: "white",
                border: "none",
                borderRadius: "50%",
                width: 40,
                height: 40,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                pointerEvents: "auto",
                position: "absolute",
                left: 20,
                transform: "translateY(-50%)"
              }}
            >
              <IoIosArrowBack size={24} />
            </button>
            <button
              onClick={goToNextSlide}
              style={{
                background: "rgba(0, 0, 0, 0.5)",
                color: "white",
                border: "none",
                borderRadius: "50%",
                width: 40,
                height: 40,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                pointerEvents: "auto",
                position: "absolute",
                right: 20,
                transform: "translateY(-50%)"
              }}
            >
              <IoIosArrowForward size={24} />
            </button>
          </div>
          
          {/* Bullet Navigation */}
          <div style={{
            display: "flex",
            justifyContent: "center",
            marginTop: 20,
            gap: 10
          }}>
            {slides.map((_, index) => (
              <button
                key={index}
                onClick={() => goToSlide(index)}
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  background: currentSlide === index ? "white" : "rgba(255, 255, 255, 0.4)",
                  border: "none",
                  cursor: "pointer",
                  transition: "all 0.3s ease",
                  padding: 0
                }}
                aria-label={`Go to slide ${index + 1}`}
              />
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
});

export default IntroductionPanel;
