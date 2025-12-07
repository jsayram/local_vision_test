"""
LLM Client for Natural Language Conversation
Uses Llama 3.2 (1B or 3B) via Ollama for text-based conversation
"""
import requests
import json
from typing import Optional, List, Dict

class LLMClient:
    """Handles conversation generation using a text-only LLM"""
    
    def __init__(self, model_name: str = "llama3.2:3b", ollama_url: str = "http://localhost:11434"):
        """
        Initialize LLM client
        
        Args:
            model_name: Ollama model to use (llama3.2:1b for RPi, llama3.2:3b for desktop)
            ollama_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if self.model_name in model_names:
                    print(f"[LLM] ✓ {self.model_name} available")
                    return True
                else:
                    print(f"[LLM] ⚠️ Model {self.model_name} not found. Run: ollama pull {self.model_name}")
                    return False
            return False
        except Exception as e:
            print(f"[LLM] ⚠️ Ollama not running: {e}")
            print(f"[LLM] Start with: ollama serve")
            return False
    
    def generate_response(
        self,
        vision_description: str,
        user_message: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        person_name: Optional[str] = None,
        event_type: str = "CHAT_MESSAGE"
    ) -> str:
        """
        Generate a conversational response
        
        Args:
            vision_description: What Moondream sees in the image
            user_message: What the user said (if any)
            conversation_history: Recent chat messages
            person_name: User's name (if known)
            event_type: Type of event triggering response
            
        Returns:
            Natural language response
        """
        if not self.available:
            return self._fallback_response(user_message, event_type)
        
        # Build conversation prompt
        prompt = self._build_conversation_prompt(
            vision_description,
            user_message,
            conversation_history,
            person_name,
            event_type
        )
        
        try:
            # Call Ollama API
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 100,  # Limit response length
                        "top_p": 0.9
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '').strip()
                
                # Clean up response (remove any meta-commentary)
                text = self._clean_response(text)
                
                print(f"[LLM] Generated: {text[:100]}...")
                return text
            else:
                print(f"[LLM] Error: {response.status_code}")
                return self._fallback_response(user_message, event_type)
                
        except Exception as e:
            print(f"[LLM] Error generating response: {e}")
            return self._fallback_response(user_message, event_type)
    
    def _build_conversation_prompt(
        self,
        vision_description: str,
        user_message: Optional[str],
        conversation_history: Optional[List[Dict]],
        person_name: Optional[str],
        event_type: str
    ) -> str:
        """Build a detailed prompt for the LLM"""
        
        parts = []
        
        # System context
        parts.append("You are a magical living portrait hanging on a wall. You can see people through your frame and speak with them warmly and engagingly.")
        
        # What you can see
        parts.append(f"\nYou can see: {vision_description}")
        
        # Who you're talking to
        if person_name:
            parts.append(f"You're speaking with {person_name}.")
        else:
            parts.append("This is someone new - ask for their name.")
        
        # Recent conversation
        if conversation_history:
            parts.append("\nRecent conversation:")
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get('role', 'portrait')
                text = msg.get('text', '')
                if role == 'user':
                    parts.append(f"  Them: {text}")
                else:
                    parts.append(f"  You: {text}")
        
        # Current situation
        if event_type == "NEW_PERSON":
            parts.append("\nSituation: A new person just appeared. Greet them warmly and ask their name.")
        elif event_type == "VOICE_MESSAGE" and user_message:
            parts.append(f"\nSituation: They just said: \"{user_message}\"")
            parts.append("Respond naturally and directly to what they said.")
        elif event_type == "CHAT_MESSAGE" and user_message:
            parts.append(f"\nSituation: They typed: \"{user_message}\"")
            parts.append("Respond naturally and directly to their message.")
        elif event_type == "POSE_CHANGED":
            parts.append("\nSituation: They moved. Comment briefly or continue the conversation.")
        else:
            parts.append("\nSituation: Continue the conversation or ask an engaging question.")
        
        # Response instructions
        parts.append("\nRespond in 1-2 short sentences. Be warm, natural, and conversational. Don't repeat yourself.")
        
        return "\n".join(parts)
    
    def _clean_response(self, text: str) -> str:
        """Remove any unwanted patterns from LLM response"""
        # Remove common meta-commentary
        unwanted_phrases = [
            "Here's my response:",
            "I would say:",
            "My response:",
            "As a portrait,",
            "As an AI,",
        ]
        
        for phrase in unwanted_phrases:
            if text.startswith(phrase):
                text = text[len(phrase):].strip()
        
        # Ensure it ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _fallback_response(self, user_message: Optional[str], event_type: str) -> str:
        """Generate a simple fallback response when LLM is unavailable"""
        if event_type == "NEW_PERSON":
            return "Hello there! Welcome to my frame. What's your name?"
        elif user_message:
            # Try to respond based on keywords
            msg_lower = user_message.lower()
            if any(word in msg_lower for word in ['hello', 'hi', 'hey']):
                return "Greetings! How can I help you today?"
            elif any(word in msg_lower for word in ['how', 'what', 'why', 'when', 'where']):
                return "That's an interesting question. Tell me more about what you're curious about."
            else:
                return "I hear you. Please continue."
        else:
            return "How are you doing?"


def create_llm_client(model_name: str = "llama3.2:3b") -> LLMClient:
    """
    Factory function to create LLM client
    
    Args:
        model_name: Model to use
            - "llama3.2:1b" for Raspberry Pi (lighter, faster)
            - "llama3.2:3b" for desktop (better quality)
    """
    return LLMClient(model_name=model_name)
