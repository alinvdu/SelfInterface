// Overlay.jsx - Integration with Auth Components
import { memo, useState, useEffect, useRef } from 'react';
import { GiBrain } from 'react-icons/gi';
import { BsArrowRight, BsArrowLeft } from 'react-icons/bs';
import { FcGoogle } from 'react-icons/fc';
import TypingText from './TypingText';
import { useAuth } from '../auth/AuthContext';
import RotatingText from './RotatingText';
import { TbPrompt } from "react-icons/tb";
import { IoIosShirt } from "react-icons/io";
import { FaTwitter } from "react-icons/fa";
import { FaDiscord } from "react-icons/fa";
import { FaYoutube } from "react-icons/fa";

const Overlay = memo(({ showLoginView, smallerThan850, isSmallSize, navigateBack, showCreateAccount, handleStartApp, toggleLoginView, signInWithGoogle, token, toggleCreateAccountView, setShowPrivacyPolicyDialog, isMobile }) => {
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
  const [menuUpScrolled, setMenuUpScrolled] = useState(false);
  const [showCreateAccountButton, setShowCreateAccountButton] = useState(false);

  const [scrollPosition, setScrollPosition] = useState(0);
  const scrollRef = useRef(null);
  const videoCallsOverlay = useRef(null);
  const textChatOverlay = useRef(null);
    
  useEffect(() => {
    const container = scrollRef.current
    const handleScroll = () => {
      setScrollPosition(container.scrollTop);
      if (container.scrollTop > 100) {
        setMenuUpScrolled(true);
      } else {
        setMenuUpScrolled(false);
      }

      if (container.scrollTop > 400) {
        setShowCreateAccountButton(true);
      } else {
        setShowCreateAccountButton(false);
      }
    };

    container.addEventListener('scroll', handleScroll);
    return () => {
      container.removeEventListener('scroll', handleScroll);
    };
  }, []);

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

  const firstGradientScrollPos = 80
  const fillPct = Math.min(scrollPosition / firstGradientScrollPos, 1) * 100;
  const fillPct2 = Math.min(Math.max(scrollPosition - 300, 0) / 80, 1) * 100;
  const fillPct3 = Math.min(Math.max(scrollPosition - 500, 0) / 200, 1) * 100;
  const fillPct4 = Math.min(Math.max(scrollPosition - 550, 0) / 80, 1) * 100;
  const fillPct5 = Math.min(Math.max(scrollPosition - 700, 0) / 550, 1) * 100;
  const fillPct6 = Math.min(Math.max(scrollPosition - 1020, 0) / 100, 1) * 100;
  const fillPct7 = Math.min(Math.max(scrollPosition - 980, 0) / 80, 1) * 100;
  const fillPct8 = Math.min(Math.max(scrollPosition - 1250, 0) / 80, 1) * 100;
  const fillPct9 = Math.min(Math.max(scrollPosition - 1920, 0) / 80, 1) * 100;
  const fillPct10 = Math.min(Math.max(scrollPosition - 2040, 0) / 80, 1) * 100;
  const fillPct11 = Math.min(Math.max(scrollPosition - 2300, 0) / 80, 1) * 100;
  const blue   = "#8DE1FF";
  const purple = "#AD60FF";
  const grey     = "#5B585B";
  const stop     = fillPct.toFixed(1) + "%";
  const stop2     = fillPct2.toFixed(1);
  const stop3     = fillPct3.toFixed(1);
  const stop4     = fillPct4.toFixed(1);
  const stop5     = fillPct5.toFixed(1);
  const stop6     = fillPct6.toFixed(1);
  const stop7     = fillPct7.toFixed(1);
  const stop8     = fillPct8.toFixed(1);
  const stop9     = fillPct9.toFixed(1);
  const stop10    = fillPct10.toFixed(1);
  const stop11    = fillPct11.toFixed(1);

  const SMOOTH_PCT = 30

  const gradient1Text =
    `linear-gradient(
      to right,
      ${blue} 0%,
      ${blue} ${stop},
      ${grey}   100%
    )`;

  const gradient2Text =
    `linear-gradient(
      to left,
      ${blue} 0%,
      ${blue} ${Math.max(stop2 == 100 ? stop2 : stop2 - 10, 0)}%,
      ${grey} ${Math.min(stop2 + 10, 100)}%,
      ${grey}   100%
    )`;

  const gradient3Down = 
  `linear-gradient(
      to bottom,
      ${purple} 0%,
      ${blue} ${Math.max(stop3 == 100 ? stop3 : stop3 - 10, 0)}%,
      ${grey} ${Math.min(stop3 + 10, 100)}%,
      ${grey}   100%
    )`;

  const gradient4Text =
    `linear-gradient(
      to bottom,
      ${blue} 0%,
      ${blue} ${Math.max(stop4 == 100 ? stop4 : stop4 - 30, 0)}%,
      ${grey} ${Math.min(stop4 + 30, 100)}%,
      ${grey}   100%
    )`;

  const gradient6Text = 
  `linear-gradient(
      to right,
      ${blue} 0%,
      ${blue} ${Math.max(stop6 == 100 ? stop6 : stop6 - 30, 0)}%,
      ${grey} ${Math.min(stop6 + 30, 100)}%,
      ${grey}   100%
    )`;

  const gradient7Text = 
    `linear-gradient(
        to right,
        ${purple} 0%,
        ${blue} ${Math.max(stop7 == 100 ? stop7 : stop7 - 30, 0)}%,
        ${grey} ${Math.min(stop7 + 30, 100)}%,
        ${grey}   100%
      )`;

  const gradient8Text =
  `linear-gradient(
    to bottom,
    ${blue} 0%,
    ${blue} ${Math.max(stop8 == 100 ? stop8 : stop8 - 30, 0)}%,
    ${grey} ${Math.min(stop8 + 30, 100)}%,
    ${grey}   100%
  )`;

  const gradient9Text =
  `linear-gradient(
    to bottom,
    ${blue} 0%,
    ${blue} ${Math.max(stop9 == 100 ? stop9 : stop9 - 30, 0)}%,
    ${grey} ${Math.min(stop9 + 30, 100)}%,
    ${grey}   100%
  )`;

  const gradient10Down = 
  `linear-gradient(
      to bottom,
      ${purple} 0%,
      ${purple} ${Math.max(stop10 == 100 ? stop10 : stop10 - 10, 0)}%,
      ${grey} ${Math.min(stop10 + 10, 100)}%,
      ${grey}   100%
    )`;

  const gradient11Text =
    `linear-gradient(
      to bottom,
      ${blue} 0%,
      ${blue} ${Math.max(stop11 == 100 ? stop11 : stop11 - 30, 0)}%,
      ${grey} ${Math.min(stop11 + 30, 100)}%,
      ${grey}   100%
    )`;

  const startFillAt   = firstGradientScrollPos;
  const endFillAt     = 200;
  const raw           = (scrollPosition - startFillAt) / (endFillAt - startFillAt);
  const fillFraction  = Math.min(Math.max(raw, 0), 1);
  const fillPctGradient2 = (fillFraction * 100).toFixed(1);

  const borderAnimStartScroll = 200;
  const borderAnimDuration = 200;

  const borderRawFill = (scrollPosition - borderAnimStartScroll) / borderAnimDuration;
  const borderFillFraction = Math.min(Math.max(borderRawFill, 0), 1);
  const borderFillPctStr = (borderFillFraction * 100).toFixed(1);

  const animatedBorderActiveColor = purple;
  const animatedBorderInactiveColor = grey;

  const fillPctBorder = parseFloat(borderFillPctStr); // Make sure this is a number
  const transitionRange = 20;
  const halfTransition = transitionRange / 2;

  const startTransition = fillPctBorder > 0 ? fillPctBorder === 100 ? fillPctBorder : Math.max(fillPctBorder - halfTransition, 0).toFixed(1) : 0;
  const endTransition = fillPctBorder > 0 ? Math.min(fillPctBorder + halfTransition, 100).toFixed(1) : 0;

  const animatedBorderGradient = `linear-gradient(
    to bottom,
    ${animatedBorderActiveColor} 0%,
    ${animatedBorderActiveColor} ${startTransition}%,
    ${animatedBorderInactiveColor} ${endTransition}%,
    ${animatedBorderInactiveColor} 100%
  )`;

  return (
    <div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="introduction-panel"
      transition={{ duration: 0.6 }}
      exit={{ opacity: 0, transition: { duration: 0 } }}
      style={{
        position: "relative",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        background: `linear-gradient(to right, rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 1) 50%, rgba(0, 0, 0, 1) 70%)`,
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        boxSizing: "border-box",
        minHeight: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        zIndex: 99,
        overflow: "hidden"
      }}
    >
      {!showLoginView && !showCreateAccount && <div style={{
        position: "absolute",
        top: 100,
        left: "50%",
        transform: `translateX(-50%) translateY(-${scrollPosition * 1.2}px)`,
        zIndex: 99999
      }}>
        <div style={{
          fontSize: 21
        }}>
          Scroll To Discover <b>Self AI</b>
        </div>
        <div>
          <img style={{
            marginTop: 15,
            width: 20
          }} className="bounce" src="arrow-down.png" />
        </div>
      </div>}
      <div style={{ 
          display: "flex", 
          justifyContent: "flex-end",
          alignItems: 'flex-end',
          flexDirection: "column",
          position: "absolute",
          left: 0,
          top: 0,
          width: '50%',
          transform: `translateY(-${scrollPosition * 1}px)`
      }}>
        <div style={{
          position: "relative",
          display: "flex", 
          justifyContent: "flex-end",
          alignItems: 'flex-end',
          flexDirection: "column"
        }}>
        <video
        style={{
          width: 720,
          zIndex: 99,
          opacity: showCreateAccount || showLoginView ? 0.7 : 1
        }}
          autoPlay={true}
          loop
          muted
        >
          <source src="floating-psychological-self-flipped.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div style={{
          width: 5,
          height: 245,
          position: "absolute",
          background: gradient3Down,
          top: 1093,
          left: "50%",
          zIndex: 999
        }} />
        <svg
          viewBox="0 0 380 710"
          style={{
            position: 'absolute',
            top: 1326,
            left: "calc(50%)",
            width: 376,
            height: 700,
            pointerEvents: 'none',
            zIndex: 999,
          }}
        >
          <defs>
            <linearGradient
              id="curveGrad3"
              gradientUnits="userSpaceOnUse"
              x1="0" y1="0"
              x2="380" y2="710"
            >
              <stop offset="0%" stopColor={blue} />
              <stop offset={`${stop5 > 0 ? stop5 == 100 ? stop5 : Math.max(stop5 - (SMOOTH_PCT-10)/2, 0) : stop5}%`} stopColor={purple} />
              <stop offset={`${stop5 > 0 ? Math.min(+stop5 + (SMOOTH_PCT - 10)/2, 100) : stop5}%`} stopColor={grey} />
            </linearGradient>
          </defs>

          <path
            d="M0 10 H280 A48 48 0 0 1 325 50 L325 710"
            stroke="url(#curveGrad3)"
            strokeWidth="5"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <div style={{
            position: "absolute",
            top: 1200,
            fontSize: 33,
            maxWidth: 340,
            textAlign: "left",
            zIndex: 9991,
            left: "50%",
            transform: "translateX(-50%)",
            background: "black"
        }}>
          <span
            style={{
              backgroundImage: gradient4Text,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}
          >
          Explore Your Inner Core
          </span>
        </div>
        <video
        style={{
          width: 650,
          marginTop: -30,
          zIndex: 98,
          marginRight: 30
        }}
          autoPlay={true}
          loop
          muted
        >
          <source src="sphere-layers.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        </div>
      </div>
      <img
        src="warp-effect.png"
        alt="Overlay"
        style={{
          position: "fixed",
          top: "0",
          left: "0",
          width: "100%",
          height: "100%",
          zIndex: 98
        }}
      />
      <div ref={scrollRef} style={{
        overflowY: 'auto',
        width: "100%",
        float: 'right',
        zIndex: 99,
        minHeight: "100%"
      }}>
      <div style={{
        display: "flex",
        width: "100%",
        justifyContent: "space-between",
        alignItems: "flex-start",
        flexDirection: "column",
        position: "fixed",
        paddingTop: 15,
        paddingLeft: 25,
        paddingRight: 25,
        boxSizing: "border-box",
        left: 0,
        zIndex: 99999,
        transition: 'background 0.3s ease, border-bottom 0.3s ease, backdrop-filter 0.3s ease',
        background: menuUpScrolled ? 'rgba(33, 26, 26, 0.35)' : 'transparent',
        borderBottom: menuUpScrolled ? '1px solid rgba(255, 255, 255, 0.2)' : 'transparent',
        backdropFilter: menuUpScrolled ? "blur(2px)" : "none",
        WebkitBackdropFilter: menuUpScrolled ? "blur(2px)" : "none",
      }}>
        <div style={{ 
          display: "flex", 
          alignItems: "center", 
          marginBottom: "10px" ,
          justifyContent: "space-between",
          width: "100%"
        }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            cursor: "pointer"
          }}
          >
            <div style={{
              display: "flex",
              alignItems: "center",
              cursor: "pointer"
            }}
            onClick={() => {
              window.location.href = '/';
            }}
            >
            <GiBrain style={{
              fontSize: isMobile ? 28 : 38,
              color: "white",
              marginRight: 10
            }} />
            <div style={{ 
              color: "white", 
              fontSize: isMobile ? 19 : "23px",
              margin: 0
            }}>
              Self AI
            </div>
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
                fontSize: isMobile ? 22 : 32,
                color: "white"
              }} />
              <div style={{
                fontSize: isMobile ? 17 : 18,
                marginLeft: 10
              }}>Back</div>
            </button>}
          </div>
          <div style={{
            display: "flex",
            alignItems: "center"
          }}>
            {
              <button 
                  className={`create-account-button ${showCreateAccountButton ? 'visible' : 'hidden'}`}
                  style={{
                  backgroundColor: "rgba(255, 255, 255, 0.9)",
                  color: "#000",
                  border: "none",
                  padding: "0.6rem 1rem",
                  borderRadius: "2rem",
                  fontSize: "16px",
                  fontWeight: "600",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  transition: "all 0.2s ease",
                  height: 38,
                  width: 180,
                  fontSize: 16,
                  marginRight: 15,
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
                  Create Account <BsArrowRight style={{ marginLeft: "0.5rem", fontSize: "1.2rem" }} />
              </button>
            }
            <div style={{ 
              color: "white", 
              fontSize: isMobile ? 19 : "19px",
              margin: 0,
              cursor: "pointer"
            }}
            onClick={() => {
              toggleLoginView(true)
            }}
            >
              Log In
            </div>
          </div>
        </div>
      </div>
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "row",
        width: "100%",
        minHeight: "100%"
      }}>
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
                      textAlign: "left",
                      textShadow: "2px 2px 4px rgba(0, 0, 0, 0.7)"
                    }}>
                      Email
                    </div>
                    <input 
                      type="email" 
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      disabled={isLoading}
                      className="placeholder-white"
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.3)",
                        background: "rgba(255, 255, 255, 0.1)",
                        backdropFilter: "blur(8px)",
                        WebkitBackdropFilter: "blur(8px)",
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
                      textAlign: "left",
                      textShadow: "2px 2px 4px rgba(0, 0, 0, 0.7)"
                    }}>
                      Password
                    </div>
                    <input 
                      type="password" 
                      placeholder="Create a password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={isLoading}
                      className='placeholder-white'
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.3)",
                        backdropFilter: "blur(8px)",
                        WebkitBackdropFilter: "blur(8px)",
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
                      textAlign: "left",
                      textShadow: "2px 2px 4px rgba(0, 0, 0, 0.7)"
                    }}>
                      Re-type Password
                    </div>
                    <input 
                      type="password" 
                      placeholder="Confirm your password"
                      value={confirmPassword}
                      className='placeholder-white'
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      disabled={isLoading}
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        backdropFilter: "blur(8px)",
                        WebkitBackdropFilter: "blur(8px)",
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

                    <div style={{
                      marginTop: "0.5rem"
                    }}>
                      By registering, you agree with our <span style={{
                        cursor: "pointer",
                        color: "#4285F4",
                        fontWeight: "bold"
                      }} onClick={() => {
                        setShowPrivacyPolicyDialog(true)
                      }}>Privacy Policy</span>.
                    </div>
                    
                    {/* Already have an account text and button */}
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      marginTop: "0.8rem"
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
                    textAlign: "left",
                    textShadow: "0px 3px rgba(0, 0, 0, 0.25)"
                  }}>
                    Email
                  </div>
                  <input 
                    type="email" 
                    placeholder="Enter your email"
                    value={email}
                    className="placeholder-white"
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={isLoading}
                    style={{
                      width: "100%",
                      padding: "0.8rem",
                      borderRadius: "0.5rem",
                      border: "1px solid rgba(255, 255, 255, 0.3)",
                      background: "rgba(255, 255, 255, 0.1)",
                      backdropFilter: "blur(8px)",
                      WebkitBackdropFilter: "blur(8px)",
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
                      className="placeholder-white"
                      onChange={(e) => setEmail(e.target.value)}
                      disabled={isLoading}
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.3)",
                        background: "rgba(255, 255, 255, 0.1)",
                        backdropFilter: "blur(8px)",
                        WebkitBackdropFilter: "blur(8px)",
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
                      className="placeholder-white"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={isLoading}
                      style={{
                        width: "100%",
                        padding: "0.8rem",
                        borderRadius: "0.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.3)",
                        background: "rgba(255, 255, 255, 0.1)",
                        backdropFilter: "blur(8px)",
                        WebkitBackdropFilter: "blur(8px)",
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
                  <div style={{
                    marginTop: "1rem",
                    width: "100%",
                    textAlign: "left"
                  }}>
                    By signing in, you agree with our <span style={{
                      cursor: "pointer",
                      color: "#4285F4",
                      fontWeight: "bold"
                    }} onClick={() => {
                      setShowPrivacyPolicyDialog(true)
                    }}>Privacy Policy</span>.
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      ) : (
        <div style={{
          display: "flex",
          flexDirection: "column",
          position: "relative",
          justifyContent: "flex-end",
          alignItems: "flex-end",
          minHeight: 0,
          width: '100%'
        }}>
            <div style={{display: "flex", minHeight: 0, width: "62%", flexDirection: "row", alignItems: "flex-start", justifyContent: "flex-start"}}>
            <div style={{
              zIndex: 100,
              position: "absolute",
              top: 330,
              width: "50%",
              right: 0,
              display: 'flex',
              flexDirection: "column",
              alignItems: "flex-start"
            }}>
              <div style={{
                position: 'absolute',
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                left: -66,
                top: 9,
                width: 38,
                height: 38,
                borderRadius: 38,
                border: "1px solid rgba(255, 255, 255, 0.5)",
                background: 'rgba(183, 183, 183, 0.32)',
                boxShadow: '0 0 10px 5px rgba(255, 255, 255, 0.3)'
              }}>
                <div style={{
                  width: 28,
                  height: 28,
                  borderRadius: 28,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "rgba(33, 33, 33, 0.5)",
                  border: "1px solid rgba(255, 255, 255, 0.3)"
                }}>
                  <div style={{
                  width: 15,
                  height: 15,
                  backgroundColor: "#AD60FF",
                  borderRadius: 15
                }}></div>
                </div>
              </div>
              <svg
                viewBox="0 0 66 78"
                style={{
                  position: 'absolute',
                  top: -70,
                  left: -116,
                  width: 76,
                  height: 80,
                  pointerEvents: 'none'
                }}
              >
                <defs>
                  <linearGradient
                    id="arcGradient"
                    gradientUnits="userSpaceOnUse"
                    x1="66" y1="78"
                    x2="0"  y2="0"
                  >
                    <stop offset="0%" stopColor="#AD60FF" />
                    <stop offset="100%" stopColor="#F9DFFD" />
                  </linearGradient>
                </defs>

                <path
                  d="M64 78 C66 20 20 0 0 4"
                  stroke="url(#arcGradient)"
                  strokeWidth="3"
                  fill="none"
                  strokeLinecap="round"
                />
              </svg>
              <svg
                viewBox="0 0 42 60"
                style={{
                  position: 'absolute',
                  top: 48,
                  left: -84,
                  width: 42,
                  height: 60,
                  pointerEvents: 'none'
                }}
              >
                <defs>
                  <linearGradient
                    id="curveGrad"
                    gradientUnits="userSpaceOnUse"
                    x1="42" y1="0"   /* gradient start at top-right */
                    x2="0"  y2="60"  /* gradient end at bottom-left */
                  >
                    <stop offset="0%"  stopColor="#AD60FF" />
                    <stop offset="100%" stopColor="#F9DFFD" />
                  </linearGradient>
                </defs>

                <path
                  d="M40 0 C42 40 22 60 0 58"
                  stroke="url(#curveGrad)"
                  strokeWidth="3"
                  fill="none"
                  strokeLinecap="round"
                />
              </svg>
              <svg
                viewBox="0 0 130 60"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: -26,
                  width: '130px',
                  height: '60px',
                  pointerEvents: 'none'
                }}
              >
                <defs>
                  <linearGradient
                    id="gradientStroke"
                    gradientUnits="userSpaceOnUse"
                    x1="0" y1="30"
                    x2="130" y2="30"
                  >
                    <stop offset="0%" stopColor="#AD60FF" />
                    <stop offset="100%" stopColor="#8DE1FF" />
                  </linearGradient>
                </defs>

                <path
                  d="M 0 30 L 130 30"
                  stroke="url(#gradientStroke)"
                  strokeWidth="4"
                  fill="none"
                />
              </svg>
              <div style={{
                marginLeft: 100
              }}>
              <div style={{right: 0, display: "flex", justifyContent: "center", alignItems: "flex-start", flexDirection: "column", maxWidth: 400, textAlign: "left"}}>
                  <div style={{
                    fontSize: 45,
                    textAlign: "left",
                    position: "relative"
                  }}>
                    <span style={{
                      backgroundImage: gradient1Text,
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent'
                    }}>Expand Your Self</span>
                    <svg
                      viewBox="0 0 120 250"
                      style={{
                        position: 'absolute',
                        top: 20,
                        right: -112,
                        width: 120,
                        height: 250,
                        pointerEvents: 'none'
                      }}
                    >
                      <defs>
                        <linearGradient
                          id="curveGrad2"
                          gradientUnits="userSpaceOnUse"
                          x1="0" y1="0"   /* gradient start at top-right */
                          x2="120"  y2="250"  /* gradient end at bottom-left */
                        >
                          <stop offset="0%" stopColor="#8DE1FF" />
                          <stop offset={`${fillPctGradient2 > 0 ? fillPctGradient2 == 100 ? fillPctGradient2 : Math.max(fillPctGradient2 - SMOOTH_PCT/2, 0) : fillPctGradient2}%`} stopColor={"#A36DF7"} />
                          <stop offset={`${fillPctGradient2 > 0 ? Math.min(+fillPctGradient2 + SMOOTH_PCT/2, 100) : fillPctGradient2}%`} stopColor="#5B585B" />
                        </linearGradient>
                      </defs>

                      <path
                        d="M0 10 H80 A48 48 0 0 1 115 50 L110 250"
                        stroke="url(#curveGrad2)"
                        strokeWidth="4"
                        fill="none"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </div>
                  <div style={{
                    fontSize: 18,
                    color: "white",
                    opacity: 0.7,
                    marginTop: 5
                  }}>
                    Interact with the most advanced AI platform build from the ground up for the soul.
                  </div>
                </div>
              <button 
                  style={{
                    marginTop: 25,
                  backgroundColor: "rgba(255, 255, 255, 0.9)",
                  color: "#000",
                  border: "none",
                  padding: "0.8rem 1rem 0.8rem 1.5rem",
                  borderRadius: "2rem",
                  fontSize: "16px",
                  fontWeight: "600",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  transition: "all 0.2s ease",
                  height: 50,
                  width: 190,
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
                  Create Account <BsArrowRight style={{ marginLeft: "0.5rem", fontSize: "1.2rem" }} />
              </button>
              </div>
            </div>
            <div style={{
                display: "flex", 
                justifyContent: "flex-start",
                alignItems: "center",
                flexDirection: "column",
                marginTop: 600,
                marginLeft: 0,
                position: "relative",
                width: "100%"
            }}>
              <div style={{
                marginTop: 600,
                fontSize: 30,
                marginLeft: 0,
                transform: `translateY(-${(scrollPosition > 500 ? 500 : scrollPosition) * 0.8}px)`,
                display: "flex",
                flexDirection: "column",
                width: "100%",
                alignItems: "flex-start",
                position: "relative"
              }}>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "flex-end"
                  }}
                >
                  <div style={{
                    fontSize: 33,
                    position: "relative"
                  }}>
                    <span style={{
                      backgroundImage: gradient2Text,
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent'
                    }}>Full Emotional Awareness</span>
                    <div style={{
                      position: 'absolute',
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      right: -50,
                      top: 0,
                      width: 38,
                      height: 38,
                      borderRadius: 38,
                      border: "1px solid rgba(255, 255, 255, 0.5)",
                      background: 'rgba(183, 183, 183, 0.32)',
                      boxShadow: '0 0 10px 5px rgba(255, 255, 255, 0.3)'
                    }}>
                      <div style={{
                        width: 28,
                        height: 28,
                        borderRadius: 28,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "rgba(33, 33, 33, 0.5)",
                        border: "1px solid rgba(255, 255, 255, 0.3)"
                      }}>
                        <div style={{
                        width: 15,
                        height: 15,
                        backgroundColor: scrollPosition > 305 ? purple : "#4A4747",
                        borderRadius: 15
                      }}></div>
                      </div>
                    </div>
                    <div style={{
                      width: 50,
                      height: 5,
                      backgroundColor: scrollPosition > 300 ? purple : '#4A4747',
                      position: "absolute",
                      right: -100,
                      top: 18
                    }}/>
                  </div>
                  <div style={{
                      fontSize: 18,
                      color: "white",
                      opacity: 0.7,
                      marginTop: 5,
                      maxWidth: 360,
                      textAlign: "right"
                  }}>
                    Atlas can pick-up emotions from your voice, use micro-expressions and gestures when suitable. So you feel like it’s real!
                  </div>
                  <div style={{
                    marginTop: 12,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "flex-end"
                  }}>
                    <img style={{
                      width: 50
                    }} src="micro-expressions-icon.png" />
                    <RotatingText />
                  </div>
                </div>
              </div>
              <div style={{
                backgroundColor: "black",
                borderRadius: 25,
                overflow: "hidden",
                position: "absolute",
                background: animatedBorderGradient,
                padding: 5,
                height: 600,
                backgroundClip: 'padding-box, border-box',
                top: 0,
                right: "10%",
                overflow: "hidden",
              }}>
                <div style={{
                  overflow: "hidden",
                  borderRadius: 25,
                  height: "100%",
                }}>
                  <img style={{
                    width: 500,
                    borderRadius: 25,
                    filter: "grayscale(100%)",
                    background: "black",
                    transition: "filter 0.5s ease-in-out"
                  }} onMouseEnter={e => {
                    e.target.style.filter = "grayscale(0%)"
                  }} onMouseLeave={e => {
                    e.target.style.filter = "grayscale(100%)"
                  }} src="atlas-smile-purple-filter.png" />
                </div>
              </div>
              <div
                style={{
                  marginTop: 1100,
                  alignSelf: "flex-start",
                  transform: `translateY(-${(scrollPosition > 1250 ? 1250 : scrollPosition) * 0.8}px)`,
                  marginLeft: "25%"
                }}
              >
              <div style={{
                position: "relative"
              }}>
              <div style={{
                  fontSize: 33,
                  maxWidth: 640,
                  textAlign: "left"
              }}>
                <span style={{
                    backgroundImage: gradient6Text,
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent'
                }}>
                  Pick Your Personality Type
                </span>
              </div>
              <div style={{
                  display: "flex",
                  marginTop: 5,
                  maxWidth: 550,
                  textAlign: "left",
                  alignItems: "flex-start",
                  justifyContent: "flex-start",
                  flexDirection: "column",
              }}>
                <div style={{
                  display: "flex",
                  flexDirection: "column",
                  marginTop: 25
                }}>
                  <div style={{
                    marginTop: 15,
                    opacity: 0.75,
                    fontSize: 17
                  }}>
                    Both start from a psychology-oriented conversational base.
                  </div>
                  <div style={{
                    fontSize: 23,
                    maxWidth: 100,
                    zIndex: 99,
                    marginTop: 35
                  }}>
                    Atlas
                  </div>
                  <img src="philosophical-icon.png" style={{
                    position: "absolute",
                    left: -85,
                    top: 140,
                    width: 75
                  }} />
                  <div style={{
                    marginTop: 15,
                    opacity: 0.75,
                    fontSize: 17
                  }}>
                    Fine-tuned on a currated list of introspective and metaphysical texts.
                  </div>
                  <div style={{
                    marginTop: 8,
                    opacity: 0.75,
                    fontSize: 17
                  }}>
                    Deep Introspection &bull; Psychology &bull; Philosophy &bull; Metaphysics. Explore your inner world.
                  </div>
                </div>
                <div style={{
                  display: "flex",
                  flexDirection: "column",
                  marginTop: 70,
                  position: "relative"
                }}>
                  <div style={{
                    fontSize: 22,
                    maxWidth: 80
                  }}>
                    Leif
                  </div>
                  <img src="fun-icon.png" style={{
                    position: "absolute",
                    left: -72,
                    top: 10,
                    width: 55
                  }} />
                  <div style={{
                    marginTop: 10,
                    opacity: 0.75
                  }}>
                    Fine-tuned on fun and creative datasets.
                  </div>
                  <div style={{
                    marginTop: 8,
                    opacity: 0.75
                  }}>
                    Fun &bull; Creative &bull; Relaxed &bull; Explores Family & Friends. Have casual, fun and interesting conversations.
                  </div>
                </div>
              </div>
              <div style={{
                position: 'absolute',
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                left: -120,
                top: 4,
                width: 38,
                height: 38,
                borderRadius: 38,
                border: "1px solid rgba(255, 255, 255, 0.5)",
                background: 'rgba(183, 183, 183, 0.32)',
                boxShadow: '0 0 10px 5px rgba(255, 255, 255, 0.3)'
              }}>
                <div style={{
                  width: 28,
                  height: 28,
                  borderRadius: 28,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "rgba(33, 33, 33, 0.5)",
                  border: "1px solid rgba(255, 255, 255, 0.3)"
                }}>
                  <div style={{
                  width: 15,
                  height: 15,
                  backgroundColor: scrollPosition > 980 ? "#AD60FF" : grey,
                  borderRadius: 15
                }}></div>
                </div>
              </div>
              <div style={{
                width: 84,
                height: 5,
                background: gradient7Text,
                position: "absolute",
                left: -80,
                top: 20
              }} />
              </div>
              </div>
          </div>
          </div>
          <div style={{ marginTop: -880, width: "100%", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginLeft: -79,
              marginTop: -18,
              width: 38,
              height: 38,
              borderRadius: 38,
              border: "1px solid rgba(255, 255, 255, 0.5)",
              background: 'rgba(183, 183, 183, 0.32)',
              boxShadow: '0 0 10px 5px rgba(255, 255, 255, 0.3)'
            }}>
              <div style={{
                width: 28,
                height: 28,
                borderRadius: 28,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "rgba(33, 33, 33, 0.5)",
                border: "1px solid rgba(255, 255, 255, 0.3)"
              }}>
                <div style={{
                width: 15,
                height: 15,
                backgroundColor: scrollPosition > 1250 ? "#AD60FF" : grey,
                borderRadius: 15
              }}></div>
              </div>
            </div>
            <div style={{
              display: "flex",
              alignItems: "flex-start",
              flexDirection: "column",
              marginTop: 15
            }}>
              <div style={{
                  fontSize: 33,
                  maxWidth: 240,
                  textAlign: "left",
              }}>
                <span
                style={{
                  backgroundImage: gradient8Text,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}
                >
                Everything You Need
                </span>
              </div>
              <div style={{
                  color: "white",
                  fontSize: 21,
                  maxWidth: 240,
                  textAlign: "left",
                  opacity: 0.7
              }}>
                In One Platform
              </div>
            </div>
          </div>
          <div style={{
            width: "100%",
            position: "relative",
            height: 1780,
            overflow: "hidden"
          }}>
          <div style={{width: "100%", display: "flex", position: "absolute", top: 470, alignItems: "center", justifyContent: "center", flexDirection: "column"}}>
            <div style={{
              marginTop: 45,
              display: "flex"
            }}>
              <div style={{
                width: 400,
                height: 550,
                borderRadius: 15,
                padding: 3,
                background: "linear-gradient(to bottom, rgba(255, 255, 255, 0.15) 0%, #6236BA 100%)",
                transform: `translateY(-${(scrollPosition >= 1600 ? 1600 : scrollPosition) * 0.3}px)`,
                overflow: 'hidden'
              }}>
                <div style={{
                  background: "black",
                  width: "100%",
                  height: "100%",
                  borderRadius: 15,
                  overflow: "hidden"
                }}
                onMouseEnter={() => {
                  videoCallsOverlay.current.style.opacity = 0
                }}
                onMouseLeave={() => {
                  videoCallsOverlay.current.style.opacity = 1
                }}
                >
                  <video
                    style={{
                      width: "100%"
                    }}
                    autoPlay
                    muted
                    loop
                  >
                    <source src="voice_talk_example.mp4" type="video/mp4" />
                  </video>
                  <div ref={videoCallsOverlay} style={{
                    position: 'absolute',
                    height: "100%",
                    width: "100%",
                    top: 0,
                    padding: 15,
                    boxSizing: 'border-box',
                    background: 'linear-gradient(to top left, #291D38 10%, rgba(0, 0, 0, 0) 100%)',
                    transition: 'opacity 0.5s ease-in-out',
                  }}>
                    <div
                      style={{
                        display: "flex",
                        width: "100%",
                        height: "100%",
                        justifyContent: "flex-end",
                        alignItems: "flex-end",
                        flexDirection: "column",
                        paddingRight: 10,
                        paddingBottom: 10,
                        boxSizing: "border-box"
                      }}
                    >
                      <div style={{
                        fontSize: 22,
                        fontWeight: 600,
                        textAlign: "right"
                      }}>
                        Video Calls
                      </div>
                      <div style={{
                        textAlign: "right",
                        marginTop: 8
                      }}>
                      Voice calls powered by empathic text to speech,
                      expressions, animations & more
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div style={{
                width: 350,
                height: 344,
                borderRadius: 15,
                padding: 3,
                marginLeft: 10,
                marginTop: 540,
                background: "linear-gradient(to bottom, rgba(255, 255, 255, 0.15) 0%, #6236BA 100%)",
                transform: `translateY(-${(scrollPosition >= 1700 ? 1700 : scrollPosition) * 0.6}px)`
              }}>
                <div style={{
                  background: "black",
                  width: "100%",
                  height: "100%",
                  borderRadius: 15,
                  overflow: "hidden",
                  position: 'relative'
                }}
                onMouseEnter={() => {
                  textChatOverlay.current.style.opacity = 0
                }}
                onMouseLeave={() => {
                  textChatOverlay.current.style.opacity = 1
                }}
                >
                  <video
                    style={{
                      width: "100%"
                    }}
                    autoPlay
                    muted
                    loop
                  >
                    <source src="chat_examplee.mp4" type="video/mp4" />
                  </video>
                  <div ref={textChatOverlay} style={{
                    position: 'absolute',
                    height: "100%",
                    width: "100%",
                    top: 0,
                    padding: 15,
                    boxSizing: 'border-box',
                    background: 'linear-gradient(to bottom, #1D1C26 0%, rgba(0, 0, 0, 0) 100%)',
                    transition: 'opacity 0.5s ease-in-out',
                  }}>
                    <div
                      style={{
                        fontSize: 22,
                        fontWeight: 600,
                        textAlign: "left"
                      }}
                    >
                      Text Chat & Voice <br /> Messages
                    </div>
                  </div>
                </div>
              </div>
              <div style={{
                display: "flex",
                flexDirection: "column",
                marginTop: 700,
                transform: `translateY(-${(scrollPosition >= 1810 ? 1810 : scrollPosition) * 0.65}px)`
              }}>
                <div style={{
                  width: 290,
                  height: 232,
                  borderRadius: 15,
                  padding: 3,
                  marginLeft: 10,
                  background: "linear-gradient(to bottom, rgba(255, 255, 255, 0.25) 0%, #6236BA 100%)",
                  position: "relative"
                }}>
                  <div style={{
                    background: "linear-gradient(to bottom right, #4A4646 0%, #172575 100%)",
                    width: "100%",
                    height: "100%",
                    borderRadius: 15,
                    display: "flex",
                    padding: 15,
                    boxSizing: "border-box",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    textAlign: "left"
                  }}>
                    <div style={{
                      fontSize: 22,
                      fontWeight: 600
                    }}>
                      Memory
                    </div>
                    <div style={{
                      marginTop: 10
                    }}>
                      Remembers important information.
                    </div>
                    <div style={{
                      marginTop: 8
                    }}>
                    Builds a timeline for what matters to you.
                    </div>
                  </div>
                  <img src="brain-graphic.png" style={{
                    position: "absolute",
                    bottom: -42,
                    right: -40,
                    width: 320
                  }} />
                </div>
                <div style={{
                  width: 290,
                  height: 90,
                  borderRadius: 15,
                  padding: 3,
                  marginTop: 10,
                  marginLeft: 10,
                  background: "linear-gradient(to bottom, rgba(255, 255, 255, 0.25) 0%, #6236BA 100%)",
                  position: "relative"
                }}>
                  <div style={{
                    background: "linear-gradient(to bottom right, #4A4646 0%, #172575 100%)",
                    width: "100%",
                    height: "100%",
                    borderRadius: 15,
                    display: "flex",
                    padding: 15,
                    boxSizing: "border-box",
                    flexDirection: "column",
                    alignItems: "flex-start"
                  }}>
                    <div style={{
                      fontSize: 22,
                      fontWeight: 600
                    }}>
                      Build-In Data Privacy
                    </div>
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      marginTop: 10
                    }}>
                      <div style={{
                      }}>
                        Control everything.
                      </div>
                      <img src="switch-graphic.png" style={{
                        width: 30,
                        marginLeft: 8
                      }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div style={{
            position: "absolute",
            top: 1170,
            display: "flex",
            justifyContent: "center",
            flexDirection: "column",
            width: "100%",
            transform: `translateY(-${(scrollPosition >= 2050 ? 2050 : scrollPosition) * 0.35}px)`
          }}>
            <div style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              marginLeft: 400 
            }}>
              <div style={{
                display: "flex",
                alignItems: "flex-start",
                flexDirection: "column",
                marginTop: 15
              }}>
                <div style={{
                    fontSize: 33,
                    maxWidth: 540,
                    textAlign: "left",
                }}>
                  <span
                  style={{
                    backgroundImage: gradient9Text,
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent'
                  }}
                  >
                  Customize Environment
                  </span>
                </div>
              </div>
              <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: 38,
                height: 38,
                marginTop: 15,
                borderRadius: 38,
                border: "1px solid rgba(255, 255, 255, 0.5)",
                background: 'rgba(183, 183, 183, 0.32)',
                boxShadow: '0 0 10px 5px rgba(255, 255, 255, 0.3)'
              }}>
                <div style={{
                  width: 28,
                  height: 28,
                  borderRadius: 28,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "rgba(33, 33, 33, 0.5)",
                  border: "1px solid rgba(255, 255, 255, 0.3)"
                }}>
                  <div style={{
                  width: 15,
                  height: 15,
                  backgroundColor: scrollPosition > 2000 ? "#AD60FF" : grey,
                  borderRadius: 15
                }}></div>
                </div>
              </div>
              <div style={{
                position: "absolute",
                height: 200,
                top: 95,
                width: 5,
                background: gradient10Down
              }} />
            </div>
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginTop: 300,
              transform: `translateY(-${(scrollPosition >= 2450 ? 2450 : scrollPosition) * 0.14}px)`
            }}>
              <div style={{
                display: "flex",
                marginLeft: 150,
                width: 300,
                position: "relative",
                height: 400,
                zIndex: 99
              }}>
                <div style={{
                    position: "absolute",
                    right: -120,
                    top: 70,
                    textAlign: "right", 
                    justifySelf: "flex-end"
                }}>
                  <div style={{
                    fontSize: 33,
                    maxWidth: 250,
                    justifySelf: "flex-end",
                    marginTop: 350,
                    transform: `translateY(-${(scrollPosition >= 2330 ? 2330 : scrollPosition) * 0.14}px)`
                  }}>
                    <span
                     style={{
                      backgroundImage: gradient11Text,
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent'
                    }}
                    >
                    Tune According To Your Mood
                    </span>
                  </div>
                  <div style={{
                    color: "white",
                    opacity: 0.7,
                    fontSize: 17,
                    marginTop: 8,
                    transform: `translateY(-${(scrollPosition >= 2520 ? 2520 : scrollPosition) * 0.13}px)`
                  }}>
                    Be what you want to be. <br />
                    Customise the background and the character.
                  </div>
                  <img style={{
                    marginTop: -15,
                    width: 70,
                    transform: `translateY(-${(scrollPosition >= 2540 ? 2540 : scrollPosition) * 0.12}px)`
                  }} src="customize-icon.png" />
                </div>
              </div>
              <div style={{
                position: "relative",
                marginTop: 80
              }}>
              <img src="env-view.png" style={{
                width: 1100,
                position: "relative"
              }} />
               <div style={{
                  position: "absolute",
                  bottom: -200,
                  left: 0,
                  zIndex: 9999
                }}>
                  <div style={{
                    width: 300,
                    height: 42,
                    background: 'rgba(25, 25, 25, 0.8)',
                    backdropFilter: "blur(8px)",
                    WebkitBackdropFilter: "blur(8px)",
                    zIndex: 999,
                    borderRadius: 8,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    border: "1px solid rgba(255, 255, 255, 0.3)",
                    transform: `translateY(-${(scrollPosition >= 2720 ? 2720 : scrollPosition) * 0.14}px)`
                  }}>
                    <TbPrompt style={{
                      marginRight: 5,
                      marginTop: 3,
                      fontSize: 19
                    }} />
                    <div style={{
                      fontSize: 15
                    }}>
                      A serene beach facing the calm sea.
                    </div>
                  </div>
                  <div style={{
                    width: 150,
                    height: 42,
                    background: 'rgba(25, 25, 25, 0.8)',
                    backdropFilter: "blur(8px)",
                    WebkitBackdropFilter: "blur(8px)",
                    zIndex: 999,
                    marginTop: 15,
                    borderRadius: 8,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    border: "1px solid rgba(255, 255, 255, 0.3)",
                    transform: `translateY(-${(scrollPosition >= 2720 ? 2720 : scrollPosition) * 0.14}px)`
                  }}>
                    <IoIosShirt style={{
                      marginRight: 5,
                      marginTop: 3,
                      fontSize: 19,
                      color: "#188EEE"
                    }} />
                    <div style={{
                      fontSize: 15
                    }}>
                      Relax Clothes
                    </div>
                  </div>
                </div>
                </div>
            </div>
            <video
            style={{
              position: "absolute",
              left: "50%",
              width: 850,
              opacity: 0.5,
              top: 780,
              transform: "translateX(-50%)"
            }}
              autoPlay={true}
              loop
              muted
            >
              <source src="orb-graphic.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
            <div style={{
              width: "100%",
              display: "flex",
              flexDirection: "column",
              marginTop: 100,
              justifyContent: "center",
              alignItems: "center",
              transform: `translateY(-${(scrollPosition >= 2850 ? 2850 : scrollPosition) * 0.13}px)`
            }}>
              <div style={{
                fontSize: 32,
                color: "#8DE1FF"
              }}>
                Still not sure?
              </div>
              <div style={{
                marginTop: 5,
                fontSize: 21,
                color: "white",
                opacity: 0.7
              }}>
                You're just one step away from your ideal virtual self!
              </div>
              <button 
                style={{
                  marginTop: 25,
                  backgroundColor: "rgba(255, 255, 255, 0.9)",
                  color: "#000",
                  border: "none",
                  padding: "0.8rem 1.3rem 0.8rem 1.5rem",
                  borderRadius: "2rem",
                  fontSize: "16px",
                  fontWeight: "600",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  transition: "all 0.2s ease",
                  height: 50,
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
                  Create Account For Free <BsArrowRight style={{ marginLeft: "0.5rem", fontSize: "1.2rem" }} />
              </button>
            </div>
            <div style={{
              marginTop: 50,
              backdropFilter: "blur(5px)",
              WebkitBackdropFilter: "blur(5px)",
              height: 93,
              width: "100%",
              borderTop: '1px solid rgba(30, 30, 30, 1)',
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              paddingLeft: 50,
              paddingRight: 50,
              boxSizing: "border-box",
              flexDirection: "row",
              transform: `translateY(-${(scrollPosition >= 2850 ? 2850 : scrollPosition) * 0.1}px)`
            }}>
              <div style={{
                fontSize: 17,
                opacity: 0.7
              }}>
                Copyright 2025 - <b>SelfAI.live</b>
              </div>
              <div style={{
                display: "flex",
                alignItems: "center",
                opacity: 0.8
              }}>
                <div style={{
                  display: "flex",
                  alignItems: "center"
                }}>
                  <FaTwitter style={{
                    color: "white",
                    fontSize: 19
                  }} />
                  <span style={{
                    marginLeft: 5
                  }}>Twitter</span>
                </div>
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  marginLeft: 15
                }}>
                  <FaDiscord style={{
                    color: "white",
                    fontSize: 19
                  }} />
                  <span style={{
                    marginLeft: 5
                  }}>Discord</span>
                </div>
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  marginLeft: 15
                }}>
                  <FaYoutube style={{
                    color: "white",
                    fontSize: 19
                  }} />
                  <span style={{
                    marginLeft: 5
                  }}>YouTube</span>
                </div>
              </div>
            </div>
          </div>
          </div>
        </div>
      )}
      </div>
      </div>
    </div>
  );
});

export default Overlay;
