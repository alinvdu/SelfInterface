import { createContext, useState, useEffect, useContext } from "react";
import { auth, signInWithGoogle, logout } from "./firebase";
import { onAuthStateChanged } from "firebase/auth";

// Create context
const AuthContext = createContext();

// Modified sign-in and sign-out functions that reload the page
const handleSignInWithGoogle = async () => {
  try {
    await signInWithGoogle();
    window.location.reload(); // Reload page after sign-in
  } catch (error) {
    console.error("Sign in failed:", error);
  }
};

const handleLogout = async () => {
  try {
    await logout();
    window.location.reload(); // Reload page after sign-out
  } catch (error) {
    console.error("Logout failed:", error);
  }
};

// AuthProvider to wrap the app
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  // Listen for auth state changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        setUser(firebaseUser);
        const token = await firebaseUser.getIdToken();
        setToken(token);
      } else {
        setUser(null);
        setToken(null);
      }
      setLoading(false);
    });

    return () => unsubscribe(); // Cleanup subscription on unmount
  }, []);

  return (
    <AuthContext.Provider value={{ 
      user, 
      token, 
      loading, 
      signInWithGoogle: handleSignInWithGoogle, 
      logout: handleLogout 
    }}>
      {children}
    </AuthContext.Provider>
  );
};

// Hook for easy use in components
export const useAuth = () => useContext(AuthContext);
