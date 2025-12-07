"""
Chat Manager
Manages per-person conversation threads with memory and archiving
"""
import os
import json
from typing import Dict, Optional, List
from datetime import datetime

from models.models import (
    ChatMessage, 
    PersonConversation, 
    InteractionMode
)


class ChatManager:
    """Manages chat conversations for all people"""
    
    def __init__(self, 
                 conversations_file: str = "memory/conversations.json",
                 archive_dir: str = "memory/archives",
                 max_messages: int = 50):
        """
        Initialize chat manager
        
        Args:
            conversations_file: Path to active conversations storage
            archive_dir: Directory for archived conversations
            max_messages: Maximum messages per conversation before archiving
        """
        self.conversations_file = conversations_file
        self.archive_dir = archive_dir
        self.max_messages = max_messages
        
        # Active conversations: {person_id: PersonConversation}
        # None key = conversation with unknown person
        self.conversations: Dict[Optional[int], PersonConversation] = {}
        
        self._load_conversations()
    
    def get_conversation(self, person_id: Optional[int]) -> PersonConversation:
        """
        Get conversation for a person (creates if doesn't exist)
        
        Args:
            person_id: Person ID (None for unknown person)
            
        Returns:
            PersonConversation
        """
        if person_id not in self.conversations:
            self.conversations[person_id] = PersonConversation(
                person_id=person_id,
                max_messages=self.max_messages
            )
        
        return self.conversations[person_id]
    
    def add_user_message(self, person_id: Optional[int], text: str, 
                        is_voice: bool = False) -> ChatMessage:
        """
        Add a user message to conversation
        
        Args:
            person_id: Person ID (None if unknown)
            text: Message text
            is_voice: True if spoken, False if typed
            
        Returns:
            Created ChatMessage
        """
        conversation = self.get_conversation(person_id)
        message = ChatMessage.create_user_message(text, is_voice)
        conversation.add_message(message)
        
        self._save_conversations()
        
        print(f"[Chat] User message added (person_id={person_id}, voice={is_voice})")
        return message
    
    def add_portrait_message(self, person_id: Optional[int], text: str,
                            mood: str = "idle", is_voice: bool = False) -> ChatMessage:
        """
        Add a portrait response to conversation
        
        Args:
            person_id: Person ID (None if unknown)
            text: Response text
            mood: Portrait mood
            is_voice: True if spoken response
            
        Returns:
            Created ChatMessage
        """
        conversation = self.get_conversation(person_id)
        message = ChatMessage.create_portrait_message(text, mood, is_voice)
        conversation.add_message(message)
        
        self._save_conversations()
        
        print(f"[Chat] Portrait message added (person_id={person_id}, mood={mood})")
        return message
    
    def get_recent_messages_formatted(self, person_id: Optional[int], count: int = 10) -> List[Dict[str, str]]:
        """
        Get recent messages formatted for conversation context
        
        Args:
            person_id: Person ID (None if unknown)
            count: Number of recent messages to retrieve
            
        Returns:
            List of dicts with 'role' and 'text' keys
        """
        conversation = self.get_conversation(person_id)
        messages = conversation.get_recent_messages(count)
        
        formatted = []
        for msg in messages:
            formatted.append({
                'role': 'user' if msg.speaker == 'user' else 'portrait',
                'text': msg.text,
                'mood': getattr(msg, 'mood', 'neutral'),
                'is_voice': msg.is_voice,
                'timestamp': msg.timestamp
            })
        
        return formatted
    
    def get_recent_messages(self, person_id: Optional[int], 
                           count: int = 10) -> List[ChatMessage]:
        """
        Get recent messages for a person
        
        Args:
            person_id: Person ID
            count: Number of recent messages to get
            
        Returns:
            List of ChatMessage
        """
        conversation = self.get_conversation(person_id)
        return conversation.get_recent_messages(count)
    
    def delete_last_exchange(self, person_id: Optional[int]) -> int:
        """
        Delete last user message and portrait response (for "forget that")
        
        Args:
            person_id: Person ID
            
        Returns:
            Number of messages deleted
        """
        conversation = self.get_conversation(person_id)
        deleted = conversation.delete_last_exchange()
        
        if deleted > 0:
            self._save_conversations()
            print(f"[Chat] Deleted {deleted} messages (person_id={person_id})")
        
        return deleted
    
    def delete_message(self, person_id: Optional[int], message_id: str) -> bool:
        """
        Delete specific message
        
        Args:
            person_id: Person ID
            message_id: Message ID to delete
            
        Returns:
            True if deleted
        """
        conversation = self.get_conversation(person_id)
        deleted = conversation.delete_message(message_id)
        
        if deleted:
            self._save_conversations()
            print(f"[Chat] Deleted message {message_id}")
        
        return deleted
    
    def clear_conversation(self, person_id: Optional[int]):
        """Clear all messages for a person"""
        if person_id in self.conversations:
            # Archive before clearing
            self._archive_conversation(person_id)
            
            # Create fresh conversation
            self.conversations[person_id] = PersonConversation(
                person_id=person_id,
                max_messages=self.max_messages
            )
            
            self._save_conversations()
            print(f"[Chat] Cleared conversation (person_id={person_id})")
    
    def merge_conversations(self, from_person_id: Optional[int], 
                           to_person_id: int):
        """
        Merge unknown person's conversation into identified person
        
        Used when unknown person is identified via face recognition
        
        Args:
            from_person_id: Source person ID (typically None)
            to_person_id: Target person ID
        """
        if from_person_id not in self.conversations:
            return
        
        from_conv = self.conversations[from_person_id]
        to_conv = self.get_conversation(to_person_id)
        
        # Merge messages
        to_conv.messages.extend(from_conv.messages)
        to_conv.last_updated = datetime.now().isoformat()
        
        # Remove old conversation
        del self.conversations[from_person_id]
        
        self._save_conversations()
        print(f"[Chat] Merged conversation {from_person_id} -> {to_person_id}")
    
    def get_conversation_summary(self, person_id: Optional[int]) -> Dict:
        """Get summary info about a conversation"""
        conversation = self.get_conversation(person_id)
        
        user_count = sum(1 for msg in conversation.messages if msg.speaker == 'user')
        portrait_count = sum(1 for msg in conversation.messages if msg.speaker == 'portrait')
        voice_count = sum(1 for msg in conversation.messages if msg.is_voice)
        
        return {
            'person_id': person_id,
            'total_messages': len(conversation.messages),
            'user_messages': user_count,
            'portrait_messages': portrait_count,
            'voice_messages': voice_count,
            'created_at': conversation.created_at,
            'last_updated': conversation.last_updated
        }
    
    def _archive_conversation(self, person_id: Optional[int]):
        """Archive a conversation to disk"""
        if person_id not in self.conversations:
            return
        
        conversation = self.conversations[person_id]
        
        if not conversation.messages:
            return  # Nothing to archive
        
        try:
            # Create archive directory
            os.makedirs(self.archive_dir, exist_ok=True)
            
            # Generate archive filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            person_str = str(person_id) if person_id is not None else "unknown"
            archive_file = os.path.join(
                self.archive_dir, 
                f"conversation_{person_str}_{timestamp}.json"
            )
            
            # Save to archive
            with open(archive_file, 'w') as f:
                json.dump(conversation.to_dict(), f, indent=2)
            
            print(f"[Chat] Archived conversation to {archive_file}")
            
        except Exception as e:
            print(f"[Chat] Error archiving conversation: {e}")
    
    def _load_conversations(self):
        """Load conversations from disk"""
        if not os.path.exists(self.conversations_file):
            print("[Chat] No existing conversations file")
            return
        
        try:
            with open(self.conversations_file, 'r') as f:
                data = json.load(f)
            
            # Convert keys back to int/None
            for key_str, conv_data in data.items():
                person_id = None if key_str == "null" else int(key_str)
                self.conversations[person_id] = PersonConversation.from_dict(conv_data)
            
            print(f"[Chat] Loaded {len(self.conversations)} conversations")
            
        except Exception as e:
            print(f"[Chat] Error loading conversations: {e}")
            self.conversations = {}
    
    def _save_conversations(self):
        """Save conversations to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.conversations_file), exist_ok=True)
            
            # Convert to serializable format
            data = {}
            for person_id, conversation in self.conversations.items():
                key = "null" if person_id is None else str(person_id)
                data[key] = conversation.to_dict()
            
            with open(self.conversations_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            print(f"[Chat] Error saving conversations: {e}")
    
    def get_all_conversations(self) -> Dict[Optional[int], PersonConversation]:
        """Get all active conversations"""
        return self.conversations.copy()


# ============================================================================
# COMMAND PARSER
# ============================================================================

class CommandParser:
    """Parse special commands from user messages"""
    
    @staticmethod
    def parse_command(text: str) -> Optional[Dict]:
        """
        Parse text for special commands
        
        Returns:
            Command dict if found, None otherwise
            
        Supported commands:
            - "forget that" -> delete last exchange
            - "clear chat" -> clear conversation
            - "forget everything" -> clear all conversations
        """
        text_lower = text.lower().strip()
        
        # Forget that / forget it
        if any(phrase in text_lower for phrase in ['forget that', 'forget it', 'never mind']):
            return {'command': 'forget_last', 'original_text': text}
        
        # Clear chat
        if any(phrase in text_lower for phrase in ['clear chat', 'clear conversation', 'start over']):
            return {'command': 'clear_conversation', 'original_text': text}
        
        # Clear everything
        if 'forget everything' in text_lower:
            return {'command': 'forget_all', 'original_text': text}
        
        return None
    
    @staticmethod
    def is_command(text: str) -> bool:
        """Check if text contains a command"""
        return CommandParser.parse_command(text) is not None
