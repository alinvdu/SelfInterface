import React from "react";
import { useAuth } from "./../auth/AuthContext";

const LoginButton = () => {
  const { user, signInWithGoogle, logout } = useAuth();

  return (
    <div>
      {user ? (
        <div>
          <span>Welcome, {user.displayName}!</span>
          <button style={{
            marginLeft: 10,
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
          <span>Sign in to access more features!</span>
          <button style={{
            marginLeft: 10,
            padding: "5px 15px",
            borderRadius: 26,
            background: 'rgba(255, 255, 255, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.6)',
            color: 'black',
            fontSize: '14px'
          }} onClick={signInWithGoogle}>Log In with Google</button>
        </div>
      )}
    </div>
  );
};

export default LoginButton;
