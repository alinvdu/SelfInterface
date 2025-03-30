// AuthContext.js
import { createContext, useState, useEffect, useContext } from "react";
import { 
  auth, 
  signInWithGoogle, 
  signInWithEmail,
  registerWithEmailAndPassword,
  sendVerificationEmail,
  resetPassword,
  logout 
} from "./firebase";
import { onAuthStateChanged } from "firebase/auth";

// Create context
const AuthContext = createContext();

// AuthProvider to wrap the app
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

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

  // Handle sign in with Google
  const handleSignInWithGoogle = async () => {
    setAuthError(null);
    try {
      await signInWithGoogle();
      window.location.reload();
      // Page reload handled by auth state change
    } catch (error) {
      setAuthError(error.message);
      console.error("Sign in with Google failed:", error);
    }
  };

  // Handle registration with email/password
  const handleRegister = async (email, password) => {
    setAuthError(null);
    try {
      const user = await registerWithEmailAndPassword(email, password);
      return { success: true, user, message: "Registration successful! Please check your email to verify your account." };
    } catch (error) {
      setAuthError(error.message);
      console.error("Registration failed:", error);
      return { success: false, error: error.message };
    }
  };

  // Handle sign in with email/password
  const handleSignInWithEmail = async (email, password) => {
    setAuthError(null);
    try {
      const user = await signInWithEmail(email, password);
      
      // Check if email is verified
      if (!user.emailVerified) {
        // You might want to handle this differently
        // Option 1: Allow login but show a verification reminder
        // Option 2: Prevent login until verified (implemented below)
        setAuthError("Please verify your email before signing in.");
        await logout(); // Sign out the user
        return { success: false, error: "Email not verified" };
      }

      window.location.reload();
      
      return { success: true, user };
    } catch (error) {
      setAuthError(error.message);
      console.error("Sign in with email failed:", error);
      return { success: false, error: error.message };
    }
  };

  // Handle password reset
  const handlePasswordReset = async (email) => {
    setAuthError(null);
    try {
      await resetPassword(email);
      return { success: true, message: "Password reset email sent!" };
    } catch (error) {
      setAuthError(error.message);
      console.error("Password reset failed:", error);
      return { success: false, error: error.message };
    }
  };

  // Handle sending verification email
  const handleSendVerificationEmail = async () => {
    setAuthError(null);
    try {
      if (user && !user.emailVerified) {
        await sendVerificationEmail(user);
        return { success: true, message: "Verification email sent!" };
      } else {
        return { success: false, error: "No user to verify or already verified" };
      }
    } catch (error) {
      setAuthError(error.message);
      console.error("Sending verification email failed:", error);
      return { success: false, error: error.message };
    }
  };

  // Handle logout
  const handleLogout = async () => {
    setAuthError(null);
    try {
      await logout();
      window.location.reload()
      // Page reload handled by auth state change
    } catch (error) {
      setAuthError(error.message);
      console.error("Logout failed:", error);
    }
  };

  const value = {
    user,
    token,
    loading,
    error: authError,
    isEmailVerified: user?.emailVerified,
    signInWithGoogle: handleSignInWithGoogle,
    signInWithEmail: handleSignInWithEmail,
    register: handleRegister,
    sendVerificationEmail: handleSendVerificationEmail,
    resetPassword: handlePasswordReset,
    logout: handleLogout,
    clearError: () => setAuthError(null)
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Hook for easy use in components
export const useAuth = () => useContext(AuthContext);
