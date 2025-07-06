import React, { createContext, useState, useContext, useEffect } from 'react';
import { sendChatMessageStreamed } from '../utils/apiClient';
import { useSettingsContext } from './SettingsContext';
import { APP_CONFIG } from '../config/api';

const ChatContext = createContext();

export function useChatContext() {
  return useContext(ChatContext);
}

export function ChatProvider({ children }) {
  // Get settings from context
  const { settings } = useSettingsContext();

  // Main state for all messages
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  // Set welcome message on component mount
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          id: 'welcome-' + Date.now(),
          sender: 'system',
          content: `Welcome to ${APP_CONFIG.APP_NAME}! I'm your ${APP_CONFIG.RESEARCH_AGENT_NAME}, ready to conduct comprehensive research using AI-powered analysis. What topic would you like me to investigate for you today?`,
          timestamp: new Date().toISOString(),
        }
      ]);
    }
  }, [messages.length]);

  // Debugging to track messages 
  useEffect(() => {
    console.log('Messages updated, count:', messages.length);
  }, [messages]);

  const sendMessage = async (content) => {
    if (!content.trim()) return;

    // Generate a unique conversation ID
    const conversationId = Date.now().toString();

    // Create the user message
    const userMessage = {
      id: `user-${conversationId}`,
      sender: 'user',
      content,
      timestamp: new Date().toISOString(),
    };

    // Add user message to the conversation
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);

    try {
      // Create a unique ID for the temporary message
      const tempId = `temp-${conversationId}`;
      const statusUpdates = [];
      let hasReceivedFinalResponse = false;

      // Start the streaming process with settings
      sendChatMessageStreamed(
        content,
        null, // No topic ID for now
        (chunkData, isJSON) => {
          // Handle both JSON status updates and plain text
          let tempMessage;

          console.log("Chunk received in ChatContext:", chunkData, isJSON);

          if (isJSON) {
            // JSON response - check if it's a status update or final
            if (chunkData.isStatus) {
              // It's a status update with structured data
              console.log("Status update in ChatContext:", chunkData);

              // Format the status update with agent name
              const statusUpdate = {
                agent: chunkData.agent,
                message: chunkData.message
              };

              statusUpdates.push(statusUpdate);
              console.log("Status updates so far:", statusUpdates);

              // Format status updates into a nicely formatted block
              const formattedStatusUpdates = statusUpdates.map(update => {
                if (update.agent) {
                  return `**${update.agent}**: ${update.message}`;
                }
                return update.message;
              });

              // Create/update the temporary message
              tempMessage = {
                id: tempId,
                sender: 'system',
                content: `${formattedStatusUpdates.join('\n\n')}\n\n*Processing...*`,
                timestamp: new Date().toISOString(),
                isTemp: true,
                statusUpdates: statusUpdates // Store original status updates for future formatting
              };

              console.log("Created temp message:", tempMessage);
              // Update the UI with the temporary message
              setMessages(prev => {
                const hasTempMessage = prev.some(m => m.id === tempId);

                if (hasTempMessage) {
                  // Replace the existing temp message
                  return prev.map(m => m.id === tempId ? tempMessage : m);
                } else {
                  // Add as a new message
                  return [...prev, tempMessage];
                }
              });
            } else if (chunkData.type === 'response' && chunkData.final_answer) {
              // Handle the structured final response
              console.log("Final response received:", chunkData);
              hasReceivedFinalResponse = true;

              // Extract debate details if included
              const debateDetails = chunkData.debate_details || [];

              // Create the final message that will replace the temporary one
              const finalMessage = {
                id: `system-${conversationId}`,
                sender: 'system',
                content: chunkData.final_answer.content || "I've processed your request.",
                timestamp: new Date().toISOString(),
                sources: chunkData.final_answer.sources || [],
                source_count: chunkData.final_answer.total_sources_found || chunkData.final_answer.source_count || 0,
                research_summary: chunkData.final_answer.research_summary || "",
                research_topic: chunkData.final_answer.research_topic || "",
                debate_transcript: chunkData.final_answer.debate_transcript || debateDetails || [],
              };

              // Update messages: keep status history and add final message
              setMessages(prev => {
                // Convert the temporary message to a permanent status history message
                const updatedMessages = prev.map(m => {
                  if (m.id === tempId) {
                    return {
                      ...m,
                      id: `status-${conversationId}`,
                      isTemp: false,
                      isStatusHistory: true,
                      content: m.content.replace('*Processing...*', ''), // Remove the processing indicator
                      statusUpdates: m.statusUpdates // Keep the structured status updates for rendering
                    };
                  }
                  return m;
                });

                // Add the final message
                return [...updatedMessages, finalMessage];
              });

              setIsTyping(false);
            } else {
              // It's a full response object that will be handled by onComplete
              return;
            }
          } else {
            // Plain text status update (fallback)
            statusUpdates.push({
              agent: null,
              message: chunkData
            });

            // Format status updates into a nicely formatted block
            const formattedStatusUpdates = statusUpdates.map(update => {
              if (update.agent) {
                return `**${update.agent}**: ${update.message}`;
              }
              return update.message;
            });

            // Create/update the temporary message
            tempMessage = {
              id: tempId,
              sender: 'system',
              content: `${formattedStatusUpdates.join('\n\n')}\n\n*Processing...*`,
              timestamp: new Date().toISOString(),
              isTemp: true,
              statusUpdates: statusUpdates
            };

            // Update the UI with the temporary message
            setMessages(prev => {
              const hasTempMessage = prev.some(m => m.id === tempId);

              if (hasTempMessage) {
                // Replace the existing temp message
                return prev.map(m => m.id === tempId ? tempMessage : m);
              } else {
                // Add as a new message
                return [...prev, tempMessage];
              }
            });
          }
        },
        (finalData) => {
          // Only process onComplete if we haven't already processed the final response
          if (!hasReceivedFinalResponse) {
            console.log("Processing final data in onComplete:", finalData);

            // Create the final message that will replace the temporary one
            const finalMessage = {
              id: `system-${conversationId}`,
              sender: 'system',
              content: finalData.content || "I've processed your request.",
              timestamp: new Date().toISOString(),
              sources: finalData.sources || [],
              source_count: finalData.total_sources_found || finalData.source_count || 0,
              research_summary: finalData.research_summary || "",
              research_topic: finalData.research_topic || "",
              debate_transcript: finalData.debate_transcript || [],
            };

            // Instead of replacing the temp message, we'll rename it and keep it
            // Then add a new final message
            setMessages(prev => {
              // First check if we already have a status history message for this conversation
              const hasStatusHistory = prev.some(m =>
                m.id === `status-${conversationId}` && m.isStatusHistory
              );

              if (hasStatusHistory) {
                // If we already have a status history, just remove the temp message
                // and add the final message
                const withoutTemp = prev.filter(m => m.id !== tempId);
                return [...withoutTemp, finalMessage];
              } else {
                // Convert the temporary message to a permanent status history message
                const updatedMessages = prev.map(m => {
                  if (m.id === tempId) {
                    return {
                      ...m,
                      id: `status-${conversationId}`,
                      isTemp: false,
                      isStatusHistory: true,
                      content: m.content.replace('*Processing...*', ''), // Remove the processing indicator
                      statusUpdates: m.statusUpdates // Keep the structured status updates for rendering
                    };
                  }
                  return m;
                });

                // Add the final message
                return [...updatedMessages, finalMessage];
              }
            });

            setIsTyping(false);
          }
        },
        (error) => {
          console.error('Error in chat stream:', error);

          // Create an error message to replace the temp message
          const errorMessage = {
            id: `system-${conversationId}`,
            sender: 'system',
            content: "I'm sorry, there was an error processing your request. Please try again.",
            timestamp: new Date().toISOString(),
            isError: true
          };

          // Update the state with the error message, removing the temp one
          setMessages(prev => {
            const withoutTemp = prev.filter(m => m.id !== tempId);
            return [...withoutTemp, errorMessage];
          });

          setIsTyping(false);
        },
        // Pass settings as options
        {
          includeDebateDetails: settings.includeDebateDetails,
          maxIterations: settings.maxIterations
        }
      );
    } catch (error) {
      console.error('Failed to set up streaming:', error);

      // Add an error message if we couldn't even start the stream
      const errorMessage = {
        id: `system-${conversationId}`,
        sender: 'system',
        content: "There was a problem connecting to the server. Please check your connection and try again.",
        timestamp: new Date().toISOString(),
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
      setIsTyping(false);
    }
  };

  const clearChat = () => {
    // Reset to welcome message only
    setMessages([
      {
        id: 'welcome-' + Date.now(),
        sender: 'system',
        content: `Welcome to ${APP_CONFIG.APP_NAME}! I'm your ${APP_CONFIG.RESEARCH_AGENT_NAME}, ready to conduct comprehensive research using AI-powered analysis. What topic would you like me to investigate for you today?`,
        timestamp: new Date().toISOString(),
      }
    ]);
  };

  return (
    <ChatContext.Provider value={{ messages, sendMessage, clearChat, isTyping }}>
      {children}
    </ChatContext.Provider>
  );
}