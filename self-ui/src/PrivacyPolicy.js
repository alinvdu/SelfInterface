import React from 'react';

const PrivacyPolicy = () => {
  const containerStyle = {
    maxWidth: '800px',
    margin: '0 auto',
    padding: '32px',
    paddingTop: 12,
    color: '#e0e0e0', // Light text for dark background
    backgroundColor: 'transparent', // Let it blend with your app’s dark theme
    textAlign: "left"
  };

  const headingStyle = {
    fontSize: '1.5rem',
    marginBottom: '12px',
    color: '#ffffff',
  };

  const subHeadingStyle = {
    fontSize: '1.2rem',
    marginTop: '16px',
    marginBottom: '8px',
    fontWeight: 'bold',
    color: '#f5f5f5',
  };

  const paragraphStyle = {
    fontSize: '1rem',
    lineHeight: '1.6',
    color: '#cccccc',
    marginBottom: '8px',
  };

  const listStyle = {
    marginBottom: '8px',
    paddingLeft: '20px',
  };

  const emailStyle = {
    color: '#4fa3ff',
    textDecoration: 'underline',
  };

  return (
    <div style={containerStyle}>
      <h1 style={headingStyle}>Privacy Policy</h1>
      <p style={paragraphStyle}><strong>Effective Date:</strong> April 4, 2025</p>
      <p style={paragraphStyle}><strong>Last Updated:</strong> April 4, 2025</p>

      <section>
        <h2 style={headingStyle}>1. Introduction</h2>
        <p style={paragraphStyle}>
          Welcome to <strong>SelfAI</strong>. Your privacy is important to us.
          This Privacy Policy explains what personal data we collect, how we use it, and the measures we take to protect it.
        </p>
        <p style={paragraphStyle}>
          By using our chatbot application and related services (the “Service”), you agree to the practices described in this policy.
        </p>
      </section>

      <section>
        <h2 style={headingStyle}>2. What Data We Collect</h2>

        <h3 style={subHeadingStyle}>a) Account Information</h3>
        <p style={paragraphStyle}>We use Firebase Authentication to allow users to sign in using:</p>
        <ul style={listStyle}>
          <li>Email and password</li>
          <li>Google accounts</li>
        </ul>
        <p style={paragraphStyle}>We collect basic account details including:</p>
        <ul style={listStyle}>
          <li>Email address (from Google)</li>
          <li>Display name (from Google)</li>
          <li>Unique user ID (UID) from Firebase</li>
        </ul>

        <h3 style={subHeadingStyle}>b) Chat Data</h3>
        <p style={paragraphStyle}>
          We store conversations between users and the chatbot to:
        </p>
        <ul style={listStyle}>
          <li>Provide contextual memory</li>
          <li>Improve responses</li>
          <li>Allow users to revisit previous chats</li>
        </ul>
        <p style={paragraphStyle}>
            All conversations are <b>encrypted</b>.
        </p>
        <p style={paragraphStyle}>
          Conversations may contain personal information if voluntarily submitted by the user.
          We do not recommend sharing sensitive or confidential information.
        </p>

        <h3 style={subHeadingStyle}>c) Vector Embeddings</h3>
        <p style={paragraphStyle}>
          To enable context-aware chat functionality, we generate and store semantic vector representations of chat content.
          These vectors are used only within our system and are not directly tied to user identities.
        </p>
        <p style={paragraphStyle}>
            Vector embeddings texts are not encrypted.
        </p>
      </section>

      <section>
        <h2 style={headingStyle}>3. Data Security</h2>
        <p style={paragraphStyle}>
          All user conversations are encrypted, with the exception of vector embeddings.
        </p>
        <p style={paragraphStyle}>
          We rely on Firebase and other GDPR-compliant infrastructure providers to store and manage data securely.
        </p>
        <p style={paragraphStyle}>
          Access to stored data is strictly limited to essential system functions.
        </p>
        <p style={paragraphStyle}>
            Data is shared with 3rd party Service providers, see 6 for a list.
        </p>
      </section>

      <section>
        <h2 style={headingStyle}>4. How We Use Your Data</h2>
        <ul style={listStyle}>
          <li>Authenticate and manage your account</li>
          <li>Deliver a personalized and contextual chatbot experience</li>
          <li>Maintain chat history across sessions</li>
        </ul>
        <p style={paragraphStyle}>
          We <strong>do not</strong> sell your data, serve ads, or use your information for marketing purposes.
        </p>
      </section>

      <section>
        <h2 style={headingStyle}>5. Data Retention</h2>
        <p style={paragraphStyle}>
          Account data is retained as long as your account is active.
        </p>
        <p style={paragraphStyle}>
          Chat data and vector embeddings are stored to support session continuity, unless you request deletion.
        </p>
        <p style={paragraphStyle}>
          You may request deletion of your account and all associated data at any time (see Section 7).
        </p>
      </section>

      <section>
        <h2 style={headingStyle}>6. Third party service providers</h2>
        <p style={paragraphStyle}>
          In order to run AI inference, emotion detection and speech to text transcription the following list of services are used:
        </p>
        <ul style={listStyle}>
          <li>OpenAI - data about conversation history, text based on vector embeddings are shared with OpenAI for inference. Voice messages are shared for transcription.</li>
          <li>DeepGram - voice data is shared for live conversations for transcription.</li>
          <li>HumeAI - face data & voice data shared for emotion analysis.</li>
        </ul>
      </section>

      <section>
        <h2 style={headingStyle}>7. Cookies and Tracking</h2>
        <p style={paragraphStyle}>
          We do <strong>not</strong> use analytics tools, tracking cookies, or third-party advertising services.
        </p>
        <p style={paragraphStyle}>
          Only essential cookies related to login sessions may be used to support authentication.
        </p>
      </section>

      <section>
        <h2 style={headingStyle}>8. Your Rights (GDPR and Other Regions)</h2>
        <p style={paragraphStyle}>
          If you are located in the EU, UK, or another region with applicable data protection laws, you have the right to:
        </p>
        <ul style={listStyle}>
          <li>Access your personal data</li>
          <li>Request correction or deletion</li>
          <li>Withdraw consent or object to processing</li>
          <li>Request data portability</li>
        </ul>
        <p style={paragraphStyle}>
          To exercise any of these rights, please contact us directly.
        </p>
      </section>

      <section>
        <h2 style={headingStyle}>9. Contact Us</h2>
        <p style={paragraphStyle}>
          For any questions about this policy or to request account/data deletion:
        </p>
        <p style={paragraphStyle}>
          📧 <a href="mailto:support@yourapp.com" style={emailStyle}>dumitru.alin25@gmail.com</a>
        </p>
      </section>

      <section>
        <h2 style={headingStyle}>9. Updates to This Policy</h2>
        <p style={paragraphStyle}>
          We may update this Privacy Policy as our features or data practices evolve.
          We will post any changes on this page with an updated "Last Updated" date.
        </p>
      </section>
    </div>
  );
};

export default PrivacyPolicy;
