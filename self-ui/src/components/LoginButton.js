import React from "react";
import { useAuth } from "./../auth/AuthContext";

const LoginButton = ({ dynamicButtonStyle }) => {
  const { user, signInWithGoogle, logout } = useAuth();

  return (
    <div>
      {user ? (
        <div>
          <span>Welcome, {user.displayName}!</span>
          <button style={{
            marginLeft: 10,
            ...dynamicButtonStyle
          }} onClick={logout}>Log Out</button>
        </div>
      ) : (
        <div>
          <div>Power-up Atlas by Joining Now!</div>
          <button style={{...dynamicButtonStyle, marginTop: 10}} onClick={signInWithGoogle}>Log In with Google</button>
        </div>
      )}
    </div>
  );
};

export default LoginButton;
