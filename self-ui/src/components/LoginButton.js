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
        <button onClick={signInWithGoogle}>Log In with Google</button>
      )}
    </div>
  );
};

export default LoginButton;
