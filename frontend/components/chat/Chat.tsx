import React, { useState, useRef, useEffect } from 'react';
import { API_BASE } from '../../src/core/api';
import RichMarkdownRenderer from '../../src/shared/RichMarkdownRenderer';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

interface ChatProps {
  currentWorkspaceId: number | null;
}

const Chat: React.FC<ChatProps> = ({ currentWorkspaceId }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // TODO: Implement actual chat API call
      // For now, simulate a response
      setTimeout(() => {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: `I received your message: "${userMessage.content}". This is a placeholder response. Chat functionality will be implemented soon.`,
          role: 'assistant',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Failed to send message:', error);
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div id="chat-tab" className="tab-content active">
      <div className="chat-header">
        <h2>AI Chat Assistant</h2>
      </div>
      <div className="chat-container">
        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="chat-placeholder">
              <div className="placeholder-content">
                <h3>👋 Welcome to AI Chat!</h3>
                <p>Ask me anything about your code, get help with debugging, or discuss your projects.</p>
                <div className="placeholder-suggestions">
                  <div className="suggestion">💡 Try: "Explain this code snippet"</div>
                  <div className="suggestion">🔧 Try: "Help me debug this error"</div>
                  <div className="suggestion">📚 Try: "What are best practices for React?"</div>
                </div>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
              >
                <div className="message-content">
                  <RichMarkdownRenderer content={message.content} />
                </div>
                <div className="message-timestamp">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="message assistant-message loading">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-container">
          <div className="chat-input-wrapper">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              className="chat-input"
              rows={1}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="chat-send-btn"
            >
              {isLoading ? '⏳' : '📤'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
