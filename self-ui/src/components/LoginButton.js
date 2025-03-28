import React from "react";
import { useAuth } from "./../auth/AuthContext";

const LoginButton = ({ isMobile }) => {
  // const { user, signInWithGoogle, logout } = useAuth();
  const { user, signInWithGoogle, logout } = {
    user: undefined,
    signInWithGoogle: () => {},
    logout: () => {}
  }

  return (
    <div>
      {user ? (
        <div>
          {!isMobile && <span>Welcome, {user.displayName}!</span>}
          <button style={{
            marginLeft: isMobile ? 0 : 10,
            padding: "5px 15px",
            borderRadius: 26,
            background: 'rgba(255, 255, 255, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.6)',
            color: 'black',
            fontSize: '14px'
          }} onClick={logout}>Log Out</button>
        </div>
      ) : (
        <div>
          {!isMobile && <span>Sign in to access more features!</span>}
          <button style={{
            marginLeft: isMobile ? 0 : 10,
            padding: "5px 15px",
            borderRadius: 26,
            background: 'rgba(255, 255, 255, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.6)',
            color: 'black',
            fontSize: '14px'
          }} onClick={signInWithGoogle}>{isMobile ? "Log In" : "Log In with Google"}</button>
        </div>
      )}
    </div>
  );
};

export default LoginButton;
