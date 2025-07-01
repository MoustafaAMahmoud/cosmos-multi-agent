import React, { useState } from 'react';
import { Box, Paper, Typography, useTheme, Fade, IconButton, Collapse } from '@mui/material';
import { useSettingsContext } from '../contexts/SettingsContext';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Avatar from '@mui/material/Avatar';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import ScienceIcon from '@mui/icons-material/Science';
import Sources from './References';
import { APP_CONFIG } from '../config/api';

// Custom component for rendering code blocks with syntax highlighting
const CodeBlock = ({ language, value, dark = true }) => {
  return (
    <Box sx={{ position: 'relative', my: 2 }}>
      <Box 
        sx={{ 
          position: 'absolute', 
          top: -6, 
          right: 16, 
          bgcolor: dark ? '#2d2d2d' : '#f5f5f5',
          color: dark ? '#aaa' : '#666',
          fontSize: '11px',
          borderRadius: '4px',
          px: 1,
          py: 0.25,
          zIndex: 1
        }}
      >
        {language || 'code'}
      </Box>
      <SyntaxHighlighter
        language={language || 'javascript'}
        style={dark ? atomDark : oneLight}
        customStyle={{
          borderRadius: '8px',
          margin: '0',
          marginTop: '4px',
          padding: '16px',
          fontSize: '13px',
          boxShadow: dark ? 'none' : '0 2px 6px rgba(0,0,0,0.1)'
        }}
      >
        {value}
      </SyntaxHighlighter>
    </Box>
  );
};



// StatusMessage component to handle the collapsible behavior
const StatusMessage = ({ message, darkMode, theme }) => {
  const [expanded, setExpanded] = useState(false);
  
  const toggleExpand = () => {
    setExpanded(!expanded);
  };
  
  // Function to render formatted status messages with colors
  const renderFormattedStatusMessages = () => {
    if (!message.statusUpdates) {
      // Fall back to regular markdown if no structured status updates
      return (
        <ReactMarkdown>
          {message.content}
        </ReactMarkdown>
      );
    }
    
    // Helper to get color for different agents
    const getAgentColor = (agent) => {
      if (!agent) return darkMode ? '#ffffff' : '#333333';
      
      const agentColors = {
        'CRITIC-TEAM': darkMode ? '#4dabf5' : '#1976d2',
        'CRITIC': darkMode ? '#ff9800' : '#ed6c02',
        'AGENT': darkMode ? '#66bb6a' : '#2e7d32',
        'APPROVED': darkMode ? '#9c27b0' : '#7b1fa2',
        'default': darkMode ? '#90caf9' : '#42a5f5'
      };
      
      return agentColors[agent] || agentColors.default;
    };
    
    return (
      <Box sx={{ mt: 1 }}>
        {message.statusUpdates.map((status, index) => (
          <Box 
            key={index} 
            sx={{ 
              mb: 2,
              display: 'flex',
              alignItems: 'flex-start'
            }}
          >
            {status.agent && (
              <Typography 
                variant="body2" 
                component="span"
                sx={{ 
                  mr: 1.5, 
                  color: getAgentColor(status.agent),
                  fontWeight: 600,
                  fontSize: '0.875rem',
                  whiteSpace: 'nowrap'
                }}
              >
                {status.agent}:
              </Typography>
            )}
            <Typography 
              variant="body2" 
              sx={{ 
                color: darkMode ? '#e0e0e0' : '#555',
                fontSize: '0.875rem'
              }}
            >
              {status.message}
            </Typography>
          </Box>
        ))}
      </Box>
    );
  };
  
  return (
    <Box sx={{ position: 'relative', width: '100%' }}>
      <Paper
        elevation={0}
        sx={{
          p: 2,
          width: '100%',
          borderRadius: '12px',
          borderLeft: `4px solid ${theme.palette.primary.main}`,
          bgcolor: darkMode ? '#232323' : '#f9f9f9',
          boxShadow: darkMode ? '0 2px 8px rgba(0, 0, 0, 0.25)' : '0 2px 8px rgba(0, 0, 0, 0.08)',
          color: darkMode ? 'white' : 'inherit',
        }}
      >
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            cursor: 'pointer',
            userSelect: 'none'
          }}
          onClick={toggleExpand}
        >
          <Typography 
            variant="subtitle2" 
            sx={{ 
              fontWeight: 500, 
              color: darkMode ? '#d0d0d0' : '#666',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <span style={{ marginRight: '8px', opacity: 0.7 }}>🧠</span>
            Thinking Process
            <span style={{ marginLeft: '8px', fontSize: '0.8em', color: darkMode ? '#aaa' : '#888' }}>
              (Click to {expanded ? 'hide' : 'show'} details)
            </span>
          </Typography>
          <IconButton 
            size="small" 
            onClick={toggleExpand}
            sx={{ color: darkMode ? '#aaa' : '#666' }}
          >
            {expanded ? <KeyboardArrowDownIcon /> : <KeyboardArrowRightIcon />}
          </IconButton>
        </Box>
        
        <Collapse in={expanded} timeout="auto" unmountOnExit>
          <Box sx={{ mt: 2, borderTop: `1px solid ${darkMode ? '#333' : '#eee'}`, pt: 2 }}>
            {renderFormattedStatusMessages()}
          </Box>
        </Collapse>
      </Paper>
    </Box>
  );
};

function MessageList({ messages, isTyping, darkMode }) {
  const theme = useTheme();
  const { settings } = useSettingsContext();
  
  // Filter messages based on includeDebateDetails setting
  const finalMessages = messages.filter(message => 
    // Remove status history messages completely when debate details are disabled
    !message.isStatusHistory || settings.includeDebateDetails
  );

  if (finalMessages.length === 0 && !isTyping) {
    return null;
  }

  return (
    <Box sx={{ width: '100%', maxWidth: '850px', mx: 'auto', px: 2, pb: 4 }}>
      {finalMessages.map((message, index) => (
        <Fade key={message.id} in={true} timeout={300} style={{ transitionDelay: `${index * 100}ms` }}>
          <Box
            sx={{
              display: 'flex',
              mb: 4,
              ...(message.sender === 'user' ? { justifyContent: 'flex-end' } : {}),
            }}
            className="message-bubble-container"
          >
            {message.sender === 'system' && (
              <Box 
                sx={{ 
                  mt: 0.5,
                  mr: 2, 
                  background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  color: 'white',
                  width: 36,
                  height: 36,
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  borderRadius: '8px',
                  flexShrink: 0,
                  boxShadow: '0 3px 10px rgba(46, 139, 87, 0.3)',
                }}
              >
                <ScienceIcon sx={{ fontSize: '1.3rem' }} />
              </Box>
            )}
            
            {message.isStatusHistory ? (
              <StatusMessage message={message} darkMode={darkMode} theme={theme} />
            ) : (
              <Paper
                elevation={0}
                className={message.sender === 'user' ? 'user-message' : 'system-message'}
                sx={{
                  p: 2,
                  maxWidth: message.sender === 'user' ? '75%' : '82%',
                  borderRadius: message.sender === 'user' ? '12px 12px 0 12px' : '12px 12px 12px 0',
                  ...(message.sender === 'user'
                    ? {
                        bgcolor: '#8F5EEC', // Distinct purple for user messages
                        color: 'white',
                      }
                    : {
                        bgcolor: darkMode ? '#1e1e1e' : 'white',
                        boxShadow: darkMode 
                          ? '0 2px 8px rgba(0, 0, 0, 0.25)' 
                          : '0 2px 8px rgba(0, 0, 0, 0.08)',
                        color: darkMode ? 'white' : 'inherit',
                        // Add subtle animation for status updates
                        ...(message.isTemp && {
                          borderLeft: `4px solid ${theme.palette.primary.main}`,
                          bgcolor: darkMode ? '#232323' : '#f9f9f9'
                        })
                      }),
                }}
              >
                {message.sender === 'user' ? (
                  <Typography variant="body1">{message.content}</Typography>
                ) : (
                  <ReactMarkdown
                    components={{
                      code: ({ node, inline, className, children, ...props }) => {
                        const match = /language-(\w+)/.exec(className || '');
                        return !inline && match ? (
                          <CodeBlock 
                            language={match[1]} 
                            value={String(children).replace(/\n$/, '')} 
                            dark={match[1] === 'json' ? false : true}
                          />
                        ) : (
                          <code
                            className={className}
                            style={{ 
                              padding: '2px 4px', 
                              borderRadius: '4px', 
                              backgroundColor: darkMode ? '#2d2d2d' : '#f0f0f0',
                              color: darkMode ? '#ff7b9e' : '#e01e5a',
                              fontFamily: "'Consolas', 'Monaco', monospace",
                              fontSize: '0.9em'
                            }}
                            {...props}
                          >
                            {children}
                          </code>
                        );
                      },
                      p: ({ children }) => {
                        // Process text for numbered citations [1], [2], etc.
                        const processedChildren = React.Children.map(children, (child) => {
                          if (typeof child === 'string') {
                            // Replace [number] with styled citation components
                            const parts = child.split(/(\[\d+\])/g);
                            return parts.map((part, index) => {
                              const citationMatch = part.match(/^\[(\d+)\]$/);
                              if (citationMatch) {
                                const citationNumber = citationMatch[1];
                                return (
                                  <Box
                                    key={index}
                                    component="span"
                                    sx={{
                                      display: 'inline-flex',
                                      alignItems: 'center',
                                      justifyContent: 'center',
                                      width: 16,
                                      height: 16,
                                      borderRadius: '50%',
                                      bgcolor: theme.palette.primary.main,
                                      color: 'white',
                                      fontSize: '0.7rem',
                                      fontWeight: 600,
                                      mx: 0.25,
                                      cursor: 'pointer',
                                      '&:hover': {
                                        bgcolor: theme.palette.primary.dark,
                                      }
                                    }}
                                    title={`Source ${citationNumber}`}
                                  >
                                    {citationNumber}
                                  </Box>
                                );
                              }
                              return part;
                            });
                          }
                          return child;
                        });
                        
                        return (
                          <Typography variant="body1" sx={{ my: 1.5 }}>
                            {processedChildren}
                          </Typography>
                        );
                      },
                      h1: ({ children }) => <Typography variant="h5" sx={{ mt: 3, mb: 1.5, fontWeight: 600 }}>{children}</Typography>,
                      h2: ({ children }) => <Typography variant="h6" sx={{ mt: 2.5, mb: 1.5, fontWeight: 600 }}>{children}</Typography>,
                      h3: ({ children }) => <Typography variant="subtitle1" sx={{ mt: 2, mb: 1, fontWeight: 600 }}>{children}</Typography>,
                      ul: ({ children }) => (
                        <Box component="ul" sx={{ pl: 2.5, my: 1 }}>
                          {children}
                        </Box>
                      ),
                      ol: ({ children }) => (
                        <Box component="ol" sx={{ pl: 2.5, my: 1 }}>
                          {children}
                        </Box>
                      ),
                      li: ({ children }) => (
                        <Box component="li" sx={{ mb: 0.75 }}>
                          <Typography variant="body1" component="span">{children}</Typography>
                        </Box>
                      ),
                      a: ({ children, href }) => (
                        <a 
                          href={href} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          style={{ 
                            color: darkMode ? '#5ea7fd' : theme.palette.primary.main, 
                            textDecoration: 'none',
                            fontWeight: 500,
                            borderBottom: `1px solid ${darkMode ? '#3a82da' : theme.palette.primary.light}`
                          }}
                        >
                          {children}
                        </a>
                      )
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                )}
                
                {/* Add Sources component for system messages with sources */}
                {message.sender === 'system' && message.sources && message.sources.length > 0 && (
                  <Sources 
                    sources={message.sources}
                    sourceCount={message.source_count}
                    researchSummary={message.research_summary}
                    researchTopic={message.research_topic}
                    darkMode={darkMode}
                  />
                )}
              </Paper>
            )}
            
            {message.sender === 'user' && (
              <Avatar 
                sx={{ 
                  ml: 2, 
                  mt: 0.5,
                  bgcolor: '#68768A', 
                  width: 36, 
                  height: 36,
                  flexShrink: 0
                }}
              >
                U
              </Avatar>
            )}
          </Box>
        </Fade>
      ))}

      {isTyping && (
        <Fade in={true} timeout={300}>
          <Box
            sx={{
              display: 'flex',
              mb: 3,
            }}
            className="typing-animation-container"
          >
            <Box 
              sx={{ 
                mt: 0.5,
                mr: 2, 
                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                color: 'white',
                width: 36,
                height: 36,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                borderRadius: '8px',
                flexShrink: 0,
                boxShadow: '0 3px 10px rgba(46, 139, 87, 0.3)'
              }}
            >
              <ScienceIcon sx={{ fontSize: '1.3rem' }} />
            </Box>
            
            <Paper
              elevation={0}
              sx={{
                py: 1.5,
                px: 2.5,
                minWidth: 100,
                borderRadius: '12px 12px 12px 0',
                bgcolor: darkMode ? '#1e1e1e' : 'white',
                boxShadow: darkMode ? '0 2px 8px rgba(0, 0, 0, 0.25)' : '0 2px 8px rgba(0, 0, 0, 0.08)',
                color: darkMode ? 'white' : 'inherit',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              <Box className="thinking-container">
                <div className="thinking-circle"></div>
                <div className="thinking-circle"></div>
                <div className="thinking-circle"></div>
              </Box>
            </Paper>
          </Box>
        </Fade>
      )}
    </Box>
  );
}

export default MessageList;