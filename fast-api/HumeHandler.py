import asyncio
import base64
import cv2
import json
import logging
import numpy as np
import time
import websockets
import io

class HumeWebSocketHandler:
    def __init__(self, api_key, session_id):
        self.api_key = api_key
        self.session_id = session_id
        self.ws = None
        self.connected = False
        self.last_activity = 0
        self.keep_alive_task = None
        self.face_emotions = []
        self.voice_emotions = []
        self.logger = logging.getLogger(f"hume_{session_id}")
        self.is_capturing_emotion = False
        self.processing_timeout = 0.75
        
        # Define thresholds for emotions
        self.FACE_EMOTION_THRESHOLD = 0.42
        self.VOICE_EMOTION_THRESHOLD = 0.25
        
    def start_emotion_capture(self):
        """Mark the start of emotion capture"""
        self.face_emotions = []
        self.voice_emotions = []
    
    def extract_top_emotions(self, hume_response, mode="face", top_n=3):
        """
        Extract top N emotions from Hume API response that meet threshold requirements
        
        Args:
            hume_response: JSON response from Hume API
            mode: "face" or "prosody" to indicate which analysis to extract
            top_n: Number of top emotions to extract
            
        Returns:
            List of top emotions with their scores that meet threshold requirements
        """
        try:
            # Extract emotions based on mode
            if mode == "face" and "face" in hume_response:
                if "predictions" in hume_response["face"] and len(hume_response["face"]["predictions"]) > 0:
                    emotions = hume_response["face"]["predictions"][0].get("emotions", [])
                    threshold = self.FACE_EMOTION_THRESHOLD
                else:
                    return []
                    
            elif mode == "prosody" and "prosody" in hume_response:
                if "predictions" in hume_response["prosody"] and len(hume_response["prosody"]["predictions"]) > 0:
                    emotions = hume_response["prosody"]["predictions"][0].get("emotions", [])
                    threshold = self.VOICE_EMOTION_THRESHOLD
                else:
                    return []
            else:
                return []
            
            # Filter emotions by threshold
            filtered_emotions = [e for e in emotions if e.get("score", 0) >= threshold]
            
            # If none meet threshold, return empty list
            if not filtered_emotions:
                return []
                
            # Sort emotions by score (highest first)
            sorted_emotions = sorted(filtered_emotions, key=lambda x: x.get("score", 0), reverse=True)
            
            # Return top N emotions
            return [{"name": e.get("name"), "score": e.get("score")} for e in sorted_emotions[:top_n]]
            
        except Exception as e:
            logging.error(f"Error extracting emotions: {str(e)}")
            return []
    
    async def connect(self):
        """Connect to Hume API WebSocket"""
        try:
            self.ws = await websockets.connect(
                f'wss://api.hume.ai/v0/stream/models?apikey={self.api_key}'
            )
            self.connected = True
            self.last_activity = time.time()
            
            # Start keep-alive task
            self.keep_alive_task = asyncio.create_task(self._keep_alive())
            
            # Start listener task
            asyncio.create_task(self._listen())
            
            self.logger.info(f"Connected to Hume API for session {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Hume API: {str(e)}")
            return False
    
    async def _keep_alive(self):
        """Send keep-alive messages every 5 seconds"""
        while self.connected:
            try:
                current_time = time.time()
                if current_time - self.last_activity >= 5 and not self.is_capturing_emotion:
                    # Send a minimal model request as keep-alive
                    # Using a model we don't otherwise use with empty parameters
                    await self.ws.send(json.dumps({
                        "data": "",  # Empty data string
                        "models": {
                            "language": {}  # Using facemesh as keep-alive model
                        }
                    }))
                    self.last_activity = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Keep-alive error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _listen(self):
        """Listen for responses from Hume API"""
        while self.connected:
            try:
                response = await self.ws.recv()
                if not self.is_capturing_emotion:
                    continue
                data = json.loads(response)
                response_time = time.time()
                # Process face emotions
                if "face" in data:
                    emotions = self.extract_top_emotions(data, mode="face")
                    
                    if len(emotions):
                        self.face_emotions.append(emotions)
                
                # Process voice/prosody emotions
                if "prosody" in data:
                    emotions = self.extract_top_emotions(data, mode="prosody")
                
                    if len(emotions):
                        self.voice_emotions.append(emotions)
                
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("Hume WebSocket connection closed")
                self.connected = False
                break
            
            except Exception as e:
                self.logger.error(f"Error processing Hume response: {str(e)}")
    
    async def send_face_image(self, image_data):
        """Send a face image for emotion analysis"""
        if not self.connected:
            return False
        
        try:
            # Convert to base64
            if isinstance(image_data, np.ndarray):
                # Convert OpenCV image to JPEG bytes
                success, buffer = cv2.imencode('.jpg', image_data)
                if not success:
                    return False
                encoded_data = base64.b64encode(buffer).decode("utf-8")
            else:
                # Assume it's already bytes
                encoded_data = base64.b64encode(image_data).decode("utf-8")
            
            # Send to Hume API
            await self.ws.send(json.dumps({
                "data": encoded_data,
                "models": {
                    "face": {}
                }
            }))
            
            self.last_activity = time.time()
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending face image: {str(e)}")
            return False
        
    def pcm_to_wav(self, pcm_data, sample_rate=48000, channels=1):
        """Convert raw PCM data to WAV format"""
        import io
        import wave
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        
        buffer.seek(0)
        return buffer.read()
    
    async def send_audio(self, audio_data, duration_ms=3000):
        """Send audio data for emotion analysis"""
        if not self.connected:
            return False
        
        try:
            # Convert to base64
            # Convert PCM to WAV
            wav_data = self.pcm_to_wav(audio_data)
            
            # Encode to base64
            encoded_data = base64.b64encode(wav_data).decode("utf-8")

            await self.ws.send(json.dumps({
                "data": encoded_data,
                "models": {
                    "prosody": {}
                },
                "stream_window_ms": duration_ms
            }))
            
            self.last_activity = time.time()
            return True
            
        except Exception as e:
            
            self.logger.error(f"Error sending audio: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from Hume API"""
        try:
            if self.keep_alive_task:
                self.keep_alive_task.cancel()
            
            if self.ws:
                await self.ws.close()
            
            self.connected = False
            self.logger.info(f"Disconnected from Hume API for session {self.session_id}")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from Hume API: {str(e)}")
    
    def get_emotion_data(self):
        """Get current emotion data"""
        return {
            "facial": self.face_emotions,
            "vocal": self.voice_emotions,
            "timestamp": time.time()
        }
        
    async def wait_for_processing(self, timeout=None):
        """Wait for all pending requests to be processed"""
        if timeout is None:
            timeout = self.processing_timeout
        
        try:
            # Wait for requests of hume to finish
            return await asyncio.sleep(timeout)
        
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for emotion processing in session {self.session_id}")
            return False
