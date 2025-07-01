import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  IconButton, 
  TextField, 
  Paper, 
  Typography, 
  AppBar, 
  Toolbar,
  useTheme,
  useMediaQuery,
  Avatar,
  Tooltip,
  Button,
  Chip
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import SendIcon from '@mui/icons-material/Send';
import SettingsIcon from '@mui/icons-material/Settings';
import MicIcon from '@mui/icons-material/Mic';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import ScienceIcon from '@mui/icons-material/Science';
import SearchIcon from '@mui/icons-material/Search';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import CodeIcon from '@mui/icons-material/Code';
import { useChatContext } from '../contexts/ChatContext';
import { useTopicContext } from '../contexts/TopicContext';
import MessageList from './MessageList';
import CodeTestDisplay from './CodeTestDisplay';
import TopicSelector from './TopicSelector';
import { APP_CONFIG } from '../config/api';

function MainChat({ handleDrawerToggle, darkMode }) {
  const [message, setMessage] = useState('');
  const [showCodeTest, setShowCodeTest] = useState(false);
  const { messages, sendMessage, isTyping } = useChatContext();
  const { selectedTopic, selectTopic } = useTopicContext();
  const messagesEndRef = useRef(null);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleSend = (e) => {
    e.preventDefault();
    if (message.trim()) {
      // If no topic is selected, select a default one first
      if (selectedTopic === null) {
        // Select the "General Questions" topic by default
        selectTopic('general');
      }
      // Then send the message
      sendMessage(message);
      setMessage('');
    }
  };

  // Auto-scroll to bottom of message list
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <Box 
      component="main" 
      sx={{ 
        flexGrow: 1,
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
        bgcolor: darkMode ? '#121212' : '#f9f9f9',
        transition: 'all 0.3s ease'
      }}
    >
      <AppBar 
        position="static" 
        color="default" 
        elevation={0}
        sx={{ 
          borderBottom: '1px solid',
          borderColor: darkMode ? '#333' : '#e0e0e0',
          backgroundColor: darkMode ? '#1e1e1e' : 'white'
        }}
      >
        <Toolbar sx={{ minHeight: 64 }}>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 1.5 }}
              size="small"
            >
              <MenuIcon />
            </IconButton>
          )}
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <Box 
              sx={{ 
                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                color: 'white',
                width: 36,
                height: 36,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                borderRadius: '8px',
                mr: 1.5,
                boxShadow: '0 3px 10px rgba(46, 139, 87, 0.3)'
              }}
            >
              <ScienceIcon sx={{ fontSize: '1.4rem' }} />
            </Box>
            <Box>
              <Typography variant="subtitle1" component="div" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
                {APP_CONFIG.APP_NAME}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                <Chip 
                  icon={<AutoAwesomeIcon sx={{ fontSize: '0.8rem' }} />}
                  label="AI-Powered" 
                  size="small" 
                  sx={{ 
                    height: 18, 
                    fontSize: '0.65rem',
                    bgcolor: theme.palette.secondary.main,
                    color: 'white',
                    fontWeight: 600,
                    '& .MuiChip-icon': { color: 'white' }
                  }} 
                />
                <Chip 
                  icon={<SearchIcon sx={{ fontSize: '0.8rem' }} />}
                  label="Exhaustive Research" 
                  size="small" 
                  sx={{ 
                    height: 18, 
                    fontSize: '0.65rem',
                    bgcolor: theme.palette.primary.main,
                    color: 'white',
                    fontWeight: 600,
                    '& .MuiChip-icon': { color: 'white' }
                  }} 
                />
              </Box>
            </Box>
          </Box>
          
          <Tooltip title="Code Syntax Test">
            <Button
              startIcon={<CodeIcon />}
              variant={showCodeTest ? "contained" : "outlined"}
              size="small"
              onClick={() => setShowCodeTest(!showCodeTest)}
              sx={{ mr: 1 }}
            >
              Syntax Test
            </Button>
          </Tooltip>
          
          <Tooltip title="Settings">
            <IconButton color="inherit" size="small" sx={{ ml: 1 }}>
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          
          <Avatar 
            sx={{ 
              width: 32, 
              height: 32, 
              ml: 1.5, 
              bgcolor: '#68768A',
              fontSize: '0.875rem' 
            }}
          >
            DU
          </Avatar>
        </Toolbar>
      </AppBar>

      <Box 
        sx={{ 
          flexGrow: 1,
          overflowY: 'auto',
          backgroundColor: darkMode ? '#121212' : '#f9f9f9',
          display: 'flex',
          flexDirection: 'column',
          pt: 2,
          transition: 'all 0.3s ease'
        }}
      >
        {showCodeTest ? (
          <CodeTestDisplay darkMode={darkMode} />
        ) : selectedTopic === null ? (
          <TopicSelector darkMode={darkMode} />
        ) : (
          <MessageList messages={messages} isTyping={isTyping} darkMode={darkMode} />
        )}
        <div ref={messagesEndRef} />
      </Box>

      {!showCodeTest && (
        <Box sx={{ 
          px: 2, 
          pb: 2, 
          pt: 1, 
          bgcolor: darkMode ? '#121212' : '#f9f9f9',
          transition: 'all 0.3s ease'
        }}>
          <Paper 
            component="form" 
            onSubmit={handleSend}
            sx={{ 
              py: 1,
              px: 2,
              display: 'flex',
              alignItems: 'center',
              boxShadow: darkMode ? '0 2px 8px rgba(0, 0, 0, 0.2)' : '0 2px 8px rgba(0, 0, 0, 0.08)',
              borderRadius: 3,
              bgcolor: darkMode ? '#1e1e1e' : 'white',
              transition: 'all 0.3s ease'
            }}
            elevation={0}
          >
            <IconButton 
              size="small" 
              sx={{ 
                color: 'text.secondary',
                mr: 1
              }}
            >
              <AttachFileIcon fontSize="small" />
            </IconButton>
            
            <TextField
              fullWidth
              variant="standard"
              placeholder="Ask me anything - I'll research comprehensively using AI-powered analysis..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              // Input enabled by default
              disabled={false}
              multiline
              maxRows={4}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && message.trim()) {
                  e.preventDefault();
                  handleSend(e);
                }
              }}
              InputProps={{
                disableUnderline: true
              }}
              sx={{
                '& .MuiInputBase-root': {
                  py: 0.75,
                  fontSize: '0.95rem'
                }
              }}
            />
            
            <Tooltip title="Voice input">
              <IconButton 
                size="small"
                sx={{ 
                  mx: 1,
                  color: 'text.secondary',
                }}
              >
                <MicIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            
            <IconButton 
              color="primary" 
              type="submit" 
              disabled={!message.trim()} // Only disabled when message is empty
              sx={{ 
                bgcolor: message.trim() ? theme.palette.primary.main : 'transparent',
                color: message.trim() ? 'white' : 'text.disabled',
                '&:hover': {
                  bgcolor: message.trim() ? theme.palette.primary.dark : 'transparent',
                },
                width: 36,
                height: 36
              }}
            >
              <SendIcon fontSize="small" />
            </IconButton>
          </Paper>
        </Box>
      )}
    </Box>
  );
}

export default MainChat;