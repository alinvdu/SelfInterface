// Overlay.jsx - Integration with Auth Components
import { memo, useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { GiBrain } from 'react-icons/gi';
import { BsArrowRight, BsArrowLeft } from 'react-icons/bs';
import { BiConversation } from 'react-icons/bi';
import { RiEmotionHappyLine } from 'react-icons/ri';
import { MdOutlinePsychology } from 'react-icons/md';
import { IoIosArrowBack, IoIosArrowForward } from 'react-icons/io';
import { FcGoogle } from 'react-icons/fc';
import TypingText from './TypingText';
import { useAuth } from '../auth/AuthContext';

// Import the components we created

const AssistantMessage = ({ text, extraStyles = {} }) => (
  <div style={{
      display: "flex",
      ...extraStyles
  }}>
    <div style={{
      backgroundColor: 'rgba(255, 255, 255, 0.75)',
      border: '1px solid rgba(255, 255, 255, 0.9)',
      backdropFilter: "blur(8px)",
      WebkitBackdropFilter: "blur(8px)",
      padding: "6px 8px",
      borderRadius: 8,
      flex: 1,
      color: "rgba(0, 0, 0, 0.65)",
      fontSize: 15
    }}>
      {text}
    </div>
    <img style={{
        width: 40,
        height: 40,
        borderRadius: "50%",
        border: "1px solid white",
        marginLeft: 8
      }} src={`${process.env.PUBLIC_URL}/assets/atlas-avatar.png`} />
  </div>
);

const Overlay = memo(({ showLoginView, smallerThan850, isSmallSize, navigateBack, showCreateAccount, handleStartApp, toggleLoginView, signInWithGoogle, token, toggleCreateAccountView }) => {
  // Slider state
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const sliderInterval = useRef(null);
  const { signInWithEmail, register, resetPassword } = useAuth();
  
  // Form states
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [formError, setFormError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [showResetForm, setShowResetForm] = useState(false);
  const [showVerificationMessage, setShowVerificationMessage] = useState(false);

  // Form handling functions
  const handleEmailLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setFormError('');
    
    try {
      const result = await signInWithEmail(email, password);
      
      if (result.success) {
        // Auth state will handle redirect/reload
      } else {
        // Handle verification needed
        if (result.error === "Email not verified") {
          setShowVerificationMessage(true);
        } else {
          setFormError(result.error || 'Login failed');
        }
      }
    } catch (error) {
      setFormError(error.message || 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegistration = async (e) => {
    e.preventDefault();
    
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email || !emailRegex.test(email)) {
      setFormError('Please enter a valid email address');
      return;
    }
    
    if (password.length < 6) {
      setFormError('Password must be at least 6 characters long');
      return;
    }
    
    if (password !== confirmPassword) {
      setFormError('Passwords do not match');
      return;
    }
    
    setIsLoading(true);
    setFormError('');
    
    try {
      const result = await register(email, password);
      
      if (result.success) {
        setSuccessMessage(result.message || 'Registration successful! Please check your email to verify your account.');
        // Clear form
        setEmail('');
        setPassword('');
        setConfirmPassword('');
      } else {
        setFormError(result.error || 'Registration failed');
      }
    } catch (error) {
      setFormError(error.message || 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePasswordReset = async (e) => {
    e.preventDefault();
    
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email || !emailRegex.test(email)) {
      setFormError('Please enter a valid email address');
      return;
    }
    
    setIsLoading(true);
    setFormError('');
    
    try {
      const result = await resetPassword(email);
      
      if (result.success) {
        setSuccessMessage('Password reset email sent! Please check your inbox.');
        setShowResetForm(false);
      } else {
        setFormError(result.error || 'Failed to send reset email');
      }
    } catch (error) {
      setFormError(error.message || 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  // Slider content - extracted from the original divs
  const slides = [
    {
      icon: BiConversation,
      title: "Interactive Real-Time Conversations",
      content: (
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column"
        }}>
        <div style={{ color: "rgba(255, 255, 255, 0.8)", fontSize: "1.1rem", maxWidth: 650, lineHeight: 1.6, textAlign: "center", marginTop: 8  }}>
            Real-time voice based interactive communication with interactive expressions, text based feedback and more.
        </div>
          <video
            width={smallerThan850 ? 390 : 580}
            controls
            autoPlay={true}
            loop
            muted
            style={{
              borderRadius: 16,
              marginTop: 25,
              border: "1px solid rgba(255, 255, 255, 0.75)"
            }}
            playsInline
          >
            <source src="Slide1.mp4" type="video/mp4" />
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
          display: "flex",
          marginTop: 0,
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          overflow: "auto"
        }}>
            <div style={{display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", marginTop: 0, marginLeft: 15}}>
              <div style={{ color: "rgba(255, 255, 255, 0.8)", fontSize: "1.1rem", maxWidth: 650, lineHeight: 1.6, textAlign: "center", marginTop: 8, fontSize: smallerThan850 ? 14 : 16  }}>
                Facial Expressions and Visual Cues recognition powered by Hume, integrated into real-time conversation so Atlas can understand more about you.
              </div>
            </div>
            <div style={{
                display: "flex",
                marginTop: 45,
                position: "relative",
                flexDirection: smallerThan850 ? "column" : "row"
            }}>
                <img style={{
                    width: 320,
                    borderRadius: 16,
                    border: "1px solid rgba(255, 255, 255, 0.35)"
                }} src="face-recognition-graphic.jpg" alt="Face recognition" />
                <div style={{display: "flex", flexDirection: smallerThan850 ? "row" : "column", marginLeft: smallerThan850 ? 0 : 10, marginTop: smallerThan850 ? 10 : 0}}>
                    <div style={{background: 'rgba(255, 255, 255, 0.85)', color: "black", display: "flex", alignItems: "center", justifyContent: "center", width: 120, height: 30, fontSize: 13, borderRadius: 5}}>
                        Happines (100%)
                    </div>
                    <div style={{background: 'rgba(255, 255, 255, 0.85)', marginLeft: smallerThan850 ? 5 : 0, marginTop: smallerThan850 ? 0 : 5, color: "black", display: "flex", alignItems: "center", justifyContent: "center", width: 80, height: 30, fontSize: 13, borderRadius: 5}}>
                        Joy (85%)
                    </div>
                </div>
                <AssistantMessage text="Great, I see that you are happy, yesterday you were feeling down so can I ask what happened in the meantime?" extraStyles={{
                  position: smallerThan850 ? "relative" : "absolute",
                  right: isSmallSize ? 0 : -150,
                  bottom: isSmallSize ? smallerThan850 ? 0 : -40 : 20,
                  marginTop: smallerThan850 ? 10 : 0,
                  width: 350
                }} />
            </div>
        </div>
      )
    },
    {
      icon: MdOutlinePsychology,
      title: "Memory-Based Psychological Insights",
      content: (
        <div style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "center",
          flexDirection: "column"
        }}>
          <div style={{display: "flex", flexDirection: "column"}}>
            <div style={{ color: "rgba(255, 255, 255, 0.8)", fontSize: "1.1rem", maxWidth: 580, lineHeight: 1.6, textAlign: "center", marginTop: 15  }}>
              Conversation summarization and psychological insights are kept up to date to increase engagement and build a profile.
            </div>
          </div>
          <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            position: "relative",
            marginTop: 35
          }}>
            <div style={{
              backdropFilter: "blur(12px)",
              WebkitBackdropFilter: "blur(12px)",
              background: 'rgba(50, 50, 50, 0.25)',
              border: "1px solid rgba(255, 255, 255, 0.35)",
              borderRadius: "8px",
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
              <div style={{background: 'rgba(255, 255, 255, 0.85)', marginTop: 8, color: "black", display: "flex", alignItems: "center", justifyContent: "center", width: 80, height: 32, fontSize: 14, borderRadius: 5}}>
                Insight
              </div>
              <div style={{background: 'rgba(255, 255, 255, 0.85)', marginTop: 8, marginLeft: 8, color: "black", display: "flex", alignItems: "center", justifyContent: "center", width: 80, height: 32, fontSize: 14, borderRadius: 5}}>
                Memory
              </div>
            </div>
            <AssistantMessage text="A sensitive topic I want to re-explore with you is the nature of limitation!" extraStyles={{
                position: smallerThan850 ? "relative" :"absolute",
                right: smallerThan850 ? 0 : -320,
                bottom: smallerThan850 ? 0 : 20,
                marginTop: smallerThan850 ? 10 : 0,
                width: 350
              }} />
          </div>
        </div>
      )
    }
  ];

  // Auto-rotate slides unless paused
  useEffect(() => {
    if (!showLoginView && !showCreateAccount) {
      sliderInterval.current = setInterval(() => {
        setCurrentSlide((prev) => (prev + 1) % slides.length);
      }, 30000); // Change slide every 30 seconds
    }
    
    return () => {
      if (sliderInterval.current) {
        clearInterval(sliderInterval.current);
      }
    };
  }, [slides.length, showLoginView, showCreateAccount]);

  // Navigation handlers
  const goToPrevSlide = () => {
    clearInterval(sliderInterval.current);
    sliderInterval.current = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slides.length);
    }, 30000);
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  const goToNextSlide = () => {
    clearInterval(sliderInterval.current);
    sliderInterval.current = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slides.length);
    }, 30000);
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const goToSlide = (index) => {
    setCurrentSlide(index);
  };

  // Reset form state when switching views
  useEffect(() => {
    setFormError('');
    setSuccessMessage('');
    setShowResetForm(false);
    setShowVerificationMessage(false);
    setEmail('');
    setPassword('');
    setConfirmPassword('');
  }, [showLoginView, showCreateAccount]);

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
        padding: smallerThan850 ? "15px 10px" : "25px 2rem",
        boxSizing: "border-box",
        overflowY: "auto",
        paddingRight: isSmallSize ? "0" : "45%",
        minHeight: "100%",
        display: "flex",
        flexDirection: "column"
      }}
    >
      <div style={{
        display: "flex",
        width: "100%",
        justifyContent: "space-between",
        alignItems: "center"
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
          
          {/* Back button for login and create account views */}
          {(showLoginView || showCreateAccount) && <button 
            style={{
              border: "none",
              background: "transparent",
              color: "white",
              padding: "0.5rem 1rem",
              borderRadius: "2rem",
              fontSize: "14px",
              fontWeight: "600",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              transition: "all 0.2s ease",
              height: 32,
              marginLeft: 20,
              marginTop: 2
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = "scale(1.05)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = "scale(1)";
            }}
            onClick={() => {
              navigateBack();
              setShowResetForm(false);
              setSuccessMessage('');
              setFormError('');
            }}
          >
            <BsArrowLeft style={{
              fontSize: 32,
              color: "white"
            }} />
            <div style={{
              fontSize: 18,
              marginLeft: 10
            }}>Back</div>
          </button>}
        </div>
      </div>
      
      {showCreateAccount ? (
        // Create Account View - Now using RegisterForm component
        <div style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          width: "100%",
          margin: "0 auto"
        }}>
          {/* Keep TypingText component */}
          <div style={{right: 0, marginTop: 60, marginBottom: 40}}>
            <TypingText fontSize={smallerThan850 ? 25 : 35} texts={[
              "Psychology, unlocked!",
              "Powerful and expressive!",
              "Real-time with deep conversations"
            ]} colors={[
              "rgb(151, 205, 223)",
              "rgb(151, 160, 223)",
              "rgb(223, 151, 222)"
            ]} />
          </div>
          
          <div style={{ width: "100%", maxWidth: 400 }}>
            {formError && (
              <div style={{
                backgroundColor: "rgba(255, 0, 0, 0.1)",
                border: "1px solid rgba(255, 0, 0, 0.3)",
                color: "white",
                padding: "10px",
                borderRadius: "5px",
                marginBottom: "15px",
                textAlign: "center"
              }}>
                {formError}
              </div>
            )}
            
            {successMessage ? (
              <div style={{
                backgroundColor: "rgba(0, 255, 0, 0.1)",
                border: "1px solid rgba(0, 255, 0, 0.3)",
                color: "white",
                padding: "15px",
                borderRadius: "5px",
                marginBottom: "15px",
                textAlign: "center"
              }}>
                <p>{successMessage}</p>
                <button 
                  style={{
                    backgroundColor: "rgba(255, 255, 255, 0.9)",
                    color: "#000",
                    border: "none",
                    padding: "0.5rem 1rem",
                    borderRadius: "2rem",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    marginTop: "10px"
                  }}
                  onClick={() => {
                    setSuccessMessage('');
                    toggleLoginView();
                  }}
                >
                  Go to Login
                </button>
              </div>
            ) : (
              <>
                {/* Login with Google button */}
                <button 
                  style={{
                    backgroundColor: "white",
                    color: "#333",
                    border: "none",
                    padding: "0.8rem 1.5rem",
                    borderRadius: "2rem",
                    fontSize: "16px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "all 0.2s ease",
                    width: "100%",
                    marginBottom: "1.5rem"
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.transform = "scale(1.02)";
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.transform = "scale(1)";
                  }}
                  onClick={signInWithGoogle}
                  disabled={isLoading}
                >
                  <FcGoogle style={{ fontSize: 20, marginRight: 10 }} />
                  Continue with Google
                </button>
                <div>OR</div>
                
                {/* Registration form */}
                <form onSubmit={handleRegistration} style={{ width: "100%" }}>
                  
                  <div style={{
                    width: "100%",
                    marginBottom: "1.5rem"
                  }}>
                    <div style={{
                      color: "white",
                      fontSize: "17px",
                      marginBottom: "0.5rem",
                      width: "100%",
                      textAlign: "left"
                    }}>
                      Email
                    </div>
                    <input 
                      type="email" 
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      disabled={isLoading}
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.3)",
                        background: "rgba(255, 255, 255, 0.1)",
                        color: "white",
                        fontSize: "16px",
                        boxSizing: "border-box",
                      }}
                    />
                  </div>
                  
                  <div style={{
                    width: "100%",
                    marginBottom: "1.5rem"
                  }}>
                    <div style={{
                      color: "white",
                      fontSize: "17px",
                      marginBottom: "0.5rem",
                      width: "100%",
                      textAlign: "left"
                    }}>
                      Password
                    </div>
                    <input 
                      type="password" 
                      placeholder="Create a password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={isLoading}
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.3)",
                        background: "rgba(255, 255, 255, 0.1)",
                        color: "white",
                        fontSize: "16px",
                        boxSizing: "border-box"
                      }}
                    />
                  </div>
                  
                  <div style={{
                    width: "100%",
                    marginBottom: "2rem"
                  }}>
                    <div style={{
                      color: "white",
                      fontSize: "17px",
                      marginBottom: "0.5rem",
                      width: "100%",
                      textAlign: "left"
                    }}>
                      Re-type Password
                    </div>
                    <input 
                      type="password" 
                      placeholder="Confirm your password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      disabled={isLoading}
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.3)",
                        background: "rgba(255, 255, 255, 0.1)",
                        color: "white",
                        fontSize: "16px",
                        boxSizing: "border-box"
                      }}
                    />
                  </div>
                  
                  {/* Register and Login buttons */}
                  <div style={{
                    display: "flex",
                    flexDirection: "column",
                    width: "100%",
                    marginTop: "1rem",
                    alignItems: "center"
                  }}>
                    {/* Register button */}
                    <button 
                      type="submit"
                      style={{
                        backgroundColor: "#4285F4", 
                        color: "white",
                        border: "none",
                        padding: "0.8rem 1.5rem",
                        borderRadius: "2rem",
                        fontSize: "16px",
                        fontWeight: "600",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        transition: "all 0.2s ease",
                        height: 50,
                        width: "100%",
                        marginBottom: "1.5rem"
                      }}
                      onMouseOver={(e) => {
                        if (!isLoading) e.currentTarget.style.transform = "scale(1.05)";
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.transform = "scale(1)";
                      }}
                      disabled={isLoading}
                    >
                      {isLoading ? 'Registering...' : 'Register'}
                    </button>
                    
                    {/* Already have an account text and button */}
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      marginTop: "0.5rem"
                    }}>
                      <div style={{ color: "white", marginRight: "0.5rem" }}>
                        Already have an account?
                      </div>
                      <button 
                        type="button"
                        style={{
                          background: "transparent",
                          border: "none",
                          color: "#4285F4",
                          fontWeight: "600",
                          cursor: "pointer",
                          padding: 0,
                          fontSize: "16px"
                        }}
                        onClick={() => {
                          toggleLoginView();
                          setFormError('');
                        }}
                        disabled={isLoading}
                      >
                        Log In
                      </button>
                    </div>
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      ) : showLoginView ? (
        // Login View - Using LoginForm or ResetPasswordForm components as needed
        <div style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          width: "100%",
          margin: "0 auto"
        }}>
          {/* Keep TypingText component */}
          <div style={{right: 0, marginTop: 60, marginBottom: 40}}>
            <TypingText fontSize={smallerThan850 ? 25 : 35} texts={[
              "Psychology, unlocked!",
              "Powerful and expressive!",
              "Real-time with deep conversations"
            ]} colors={[
              "rgb(151, 205, 223)",
              "rgb(151, 160, 223)",
              "rgb(223, 151, 222)"
            ]} />
          </div>
          
          <div style={{ width: "100%", maxWidth: 400 }}>
            {formError && (
              <div style={{
                backgroundColor: "rgba(255, 0, 0, 0.1)",
                border: "1px solid rgba(255, 0, 0, 0.3)",
                color: "white",
                padding: "10px",
                borderRadius: "5px",
                marginBottom: "15px",
                textAlign: "center"
              }}>
                {formError}
              </div>
            )}
            
            {successMessage && (
              <div style={{
                backgroundColor: "rgba(0, 255, 0, 0.1)",
                border: "1px solid rgba(0, 255, 0, 0.3)",
                color: "white",
                padding: "15px",
                borderRadius: "5px",
                marginBottom: "15px",
                textAlign: "center"
              }}>
                <p>{successMessage}</p>
              </div>
            )}
            
            {showVerificationMessage ? (
              <div style={{
                backgroundColor: "rgba(255, 165, 0, 0.1)",
                border: "1px solid rgba(255, 165, 0, 0.3)",
                color: "white",
                padding: "15px",
                borderRadius: "5px",
                marginBottom: "15px",
                textAlign: "center"
              }}>
                <p>Please verify your email before signing in.</p>
                <button 
                  style={{
                    backgroundColor: "rgba(255, 255, 255, 0.2)",
                    color: "white",
                    border: "1px solid rgba(255, 255, 255, 0.4)",
                    padding: "0.5rem 1rem",
                    borderRadius: "2rem",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    marginTop: "10px"
                  }}
                  onClick={() => setShowVerificationMessage(false)}
                >
                  Back to Login
                </button>
              </div>
            ) : showResetForm ? (
              // Password Reset Form
              <form onSubmit={handlePasswordReset} style={{ width: "100%" }}>
                <div style={{ fontSize: "20px", fontWeight: "bold", color: "white", marginBottom: "15px", textAlign: "center" }}>
                  Reset Your Password
                </div>
                
                <p style={{ color: "white", marginBottom: "20px", textAlign: "center" }}>
                  Enter your email address and we'll send you instructions to reset your password.
                </p>
                
                <div style={{
                  width: "100%",
                  marginBottom: "1.5rem"
                }}>
                  <div style={{
                    color: "white",
                    fontSize: "17px",
                    marginBottom: "0.5rem",
                    width: "100%",
                    textAlign: "left"
                  }}>
                    Email
                  </div>
                  <input 
                    type="email" 
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={isLoading}
                    style={{
                      width: "100%",
                      padding: "0.8rem",
                      borderRadius: "0.5rem",
                      border: "1px solid rgba(255, 255, 255, 0.3)",
                      background: "rgba(255, 255, 255, 0.1)",
                      color: "white",
                      fontSize: "16px",
                      boxSizing: "border-box",
                    }}
                  />
                </div>
                
                <div style={{
                  display: "flex",
                  justifyContent: "space-between",
                  width: "100%",
                  marginTop: "1.5rem"
                }}>
                  <button 
                    type="button"
                    style={{
                      backgroundColor: "rgba(255, 255, 255, 0.2)",
                      color: "white",
                      border: "1px solid rgba(255, 255, 255, 0.4)",
                      padding: "0.8rem 1.5rem",
                      borderRadius: "2rem",
                      fontSize: "16px",
                      fontWeight: "600",
                      cursor: "pointer",
                      transition: "all 0.2s ease",
                      height: 50
                    }}
                    onClick={() => setShowResetForm(false)}
                    disabled={isLoading}
                  >
                    Cancel
                  </button>
                  
                  <button 
                    type="submit"
                    style={{
                      backgroundColor: "#4285F4",
                      color: "white",
                      border: "none",
                      padding: "0.8rem 1.5rem",
                      borderRadius: "2rem",
                      fontSize: "16px",
                      fontWeight: "600",
                      cursor: "pointer",
                      transition: "all 0.2s ease",
                      height: 50
                    }}
                    disabled={isLoading}
                  >
                    {isLoading ? 'Sending...' : 'Send Reset Link'}
                  </button>
                </div>
              </form>
            ) : (
              // Regular Login Form
              <>
                {/* Login with Google button */}
                <button 
                  style={{
                    backgroundColor: "white",
                    color: "#333",
                    border: "none",
                    padding: "0.8rem 1.5rem",
                    borderRadius: "2rem",
                    fontSize: "16px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "all 0.2s ease",
                    width: "100%",
                    marginBottom: "1.5rem"
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.transform = "scale(1.02)";
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.transform = "scale(1)";
                  }}
                  onClick={signInWithGoogle}
                  disabled={isLoading}
                >
                  <FcGoogle style={{ fontSize: 20, marginRight: 10 }} />
                  Continue with Google
                </button>
                <div>OR</div>
                
                {/* Custom login */}
                <form onSubmit={handleEmailLogin} style={{ width: "100%" }}>
                  <div style={{
                    width: "100%",
                    marginBottom: "1.5rem",
                    marginTop: 10
                  }}>
                    <div style={{
                      color: "white",
                      fontSize: "17px",
                      marginBottom: "0.5rem",
                      width: "100%",
                      textAlign: "left"
                    }}>
                      Email
                    </div>
                    <input 
                      type="email" 
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      disabled={isLoading}
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.3)",
                        background: "rgba(255, 255, 255, 0.1)",
                        color: "white",
                        fontSize: "16px",
                        boxSizing: "border-box",
                      }}
                    />
                  </div>
                  
                  <div style={{
                    width: "100%",
                    marginBottom: "1rem"
                  }}>
                    <div style={{
                      color: "white",
                      fontSize: "17px",
                      marginBottom: "0.5rem",
                      width: "100%",
                      textAlign: "left"
                    }}>
                      Password
                    </div>
                    <input 
                      type="password" 
                      placeholder="Enter your password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={isLoading}
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.3)",
                        background: "rgba(255, 255, 255, 0.1)",
                        color: "white",
                        fontSize: "16px",
                        boxSizing: "border-box"
                      }}
                    />
                  </div>
                  
                  <div style={{
                    textAlign: "right",
                    marginBottom: "1.5rem"
                  }}>
                    <button
                      type="button"
                      style={{
                        background: "transparent",
                        border: "none",
                        color: "#4285F4",
                        cursor: "pointer",
                        fontSize: "14px"
                      }}
                      onClick={() => setShowResetForm(true)}
                      disabled={isLoading}
                    >
                      Forgot Password?
                    </button>
                  </div>
                  
                  {/* Log in and Create account buttons */}
                  <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    width: "100%",
                    marginTop: "1rem"
                  }}>
                    {/* Log in submit button */}
                    <button 
                      type="submit"
                      style={{
                        backgroundColor: "#4285F4",
                        color: "white",
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
                        if (!isLoading) e.currentTarget.style.transform = "scale(1.05)";
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.transform = "scale(1)";
                      }}
                      disabled={isLoading}
                    >
                      {isLoading ? 'Logging In...' : 'Log In'}
                    </button>
                    
                    {/* Create account button */}
                    <button 
                      type="button"
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
                        if (!isLoading) {
                          e.currentTarget.style.transform = "scale(1.05)";
                          e.currentTarget.style.backgroundColor = "#fff";
                        }
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.transform = "scale(1)";
                        e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.9)";
                      }}
                      onClick={toggleCreateAccountView}
                      disabled={isLoading}
                    >
                      Create account <BsArrowRight style={{ marginLeft: "0.5rem", fontSize: "1.2rem" }} />
                    </button>
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      ) : (
        // Original content with slides
        <div style={{
          display: "flex",
          flexDirection: "column",
          position: "relative",
          justifyContent: "flex-end",
          minHeight: 0,
          flex: 1
        }}>
          <div style={{display: "flex", height: token ? "15%" : "30%", flexDirection: "column", alignItems: "center", justifyContent: "center"}}>
            <div style={{right: 0, display: "flex", justifyContent: "center", alignItems: "center" }}>
              <TypingText fontSize={smallerThan850 ? 25 : 35} texts={[
                "Psychology, unlocked!",
                "Powerful and expressive!",
                "Real-time with deep conversations"
              ]} colors={[
                "rgb(151, 205, 223)",
                "rgb(151, 160, 223)",
                "rgb(223, 151, 222)"
              ]} />
            </div>
            {!token && <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexDirection: smallerThan850 ? "column" : "row",
            }}>
                <div style={{ 
                    display: "flex", 
                    justifyContent: "center",
                    marginTop: 25,
                    marginRight: smallerThan850 ? 0 : 15
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
                    onClick={toggleCreateAccountView}
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
                    Try without logging
                </button>
                </div>
            </div>}
          </div>
          
          {/* Slider Section */}
          <div 
            style={{
              padding: 25,
              boxSizing: "border-box",
              width: "100%",
              position: "relative",
              flex: 1
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
                justifyContent: "center",
                display: "flex",
                boxSizing: "border-box",
                width: "100%",
                flexDirection: "column",
                height: "100%",
                alignItems: "center",
              }}
            >
              <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }}>
                <div style={{ display: "flex", color: "white", fontSize: smallerThan850 ? "1.5rem" : "1.8rem", textAlign: "center", marginLeft: 10, maxWidth: 650 }}>
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
                  background: "rgba(255, 255, 255, 0.85)",
                  color: "black",
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
                  left: isSmallSize ? 0 : 20,
                  transform: "translateY(-50%)"
                }}
              >
                <IoIosArrowBack size={24} />
              </button>
              <button
                onClick={goToNextSlide}
                style={{
                  background: "rgba(255, 255, 255, 0.86)",
                  color: "black",
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
                  right: isSmallSize ? 0 : 20,
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
      )}
    </motion.div>
  );
});

export default Overlay;
