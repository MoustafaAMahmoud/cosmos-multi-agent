import React, { useState, useMemo, useEffect } from 'react';
import { Box, ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import Sidebar from './components/Sidebar';
import MainChat from './components/MainChat';
import ApiConfigManager from './components/ApiConfigManager';
import { TopicProvider } from './contexts/TopicContext';
import { ChatProvider } from './contexts/ChatContext';
import { SettingsProvider } from './contexts/SettingsContext';
import { APP_CONFIG, THEME_CONFIG, FEATURE_FLAGS } from './config/api';

function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    // Initialize dark mode from localStorage or default to false
    const saved = localStorage.getItem('research-agent-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  // Update document title and favicon based on configuration
  useEffect(() => {
    document.title = `${APP_CONFIG.APP_NAME} - ${APP_CONFIG.APP_SUBTITLE}`;
    
    // Update favicon if specified
    const favicon = document.querySelector('link[rel="icon"]');
    if (favicon && APP_CONFIG.FAVICON_PATH) {
      favicon.href = APP_CONFIG.FAVICON_PATH;
    }
  }, []);

  // Save dark mode preference to localStorage
  useEffect(() => {
    localStorage.setItem('research-agent-dark-mode', JSON.stringify(darkMode));
  }, [darkMode]);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };
  
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };
  
  // Create a research-focused theme using environment configuration
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: darkMode ? 'dark' : 'light',
          primary: {
            main: THEME_CONFIG.PRIMARY_COLOR, // Research green
            light: '#4CAF50',
            dark: '#1B5E20',
          },
          secondary: {
            main: THEME_CONFIG.SECONDARY_COLOR, // Research teal
            light: '#4DB6AC',
            dark: '#00695C',
          },
          accent: {
            main: THEME_CONFIG.ACCENT_COLOR, // Research accent
          },
          background: {
            default: darkMode ? THEME_CONFIG.BACKGROUND_DARK : THEME_CONFIG.BACKGROUND_LIGHT,
            paper: darkMode ? '#1e1e1e' : '#ffffff',
          },
          text: {
            primary: darkMode ? '#f5f5f5' : '#2c3e50',
            secondary: darkMode ? '#b0b0b0' : '#7f8c8d',
          },
        },
        typography: {
          fontFamily: '"Inter", "Segoe UI", "Roboto", "Helvetica", "Arial", sans-serif',
          h4: {
            fontWeight: 700,
            letterSpacing: '-0.02em',
          },
          h5: {
            fontWeight: 600,
            letterSpacing: '-0.01em',
          },
          h6: {
            fontWeight: 600,
            letterSpacing: '-0.01em',
          },
          body1: {
            lineHeight: 1.6,
          },
          body2: {
            lineHeight: 1.5,
          },
        },
        components: {
          MuiButton: {
            styleOverrides: {
              root: {
                borderRadius: 8,
                textTransform: 'none',
                fontWeight: 600,
                padding: '10px 20px',
                transition: 'all 0.2s ease-in-out',
              },
              contained: {
                boxShadow: 'none',
                '&:hover': {
                  boxShadow: '0 4px 12px rgba(46, 139, 87, 0.3)',
                  transform: 'translateY(-1px)',
                },
              },
            },
          },
          MuiCard: {
            styleOverrides: {
              root: {
                borderRadius: 12,
                boxShadow: darkMode 
                  ? '0px 4px 20px rgba(0, 0, 0, 0.3)' 
                  : '0px 4px 20px rgba(0, 0, 0, 0.1)',
                transition: 'all 0.3s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: darkMode 
                    ? '0px 8px 30px rgba(0, 0, 0, 0.4)' 
                    : '0px 8px 30px rgba(0, 0, 0, 0.15)',
                },
              },
            },
          },
          MuiPaper: {
            styleOverrides: {
              root: {
                borderRadius: 12,
                backgroundImage: 'none',
              },
            },
          },
          MuiChip: {
            styleOverrides: {
              root: {
                borderRadius: 6,
                fontWeight: 500,
              },
            },
          },
        },
      }),
    [darkMode],
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SettingsProvider>
        <TopicProvider>
          <ChatProvider>
          <Box sx={{ 
            display: 'flex', 
            height: '100vh',
            overflowX: 'hidden',
            backgroundColor: theme.palette.background.default,
            color: theme.palette.text.primary,
            transition: 'all 0.3s ease',
            // Add subtle research-themed background pattern
            backgroundImage: darkMode 
              ? 'radial-gradient(circle at 25% 25%, rgba(46, 139, 87, 0.1) 0%, transparent 50%)'
              : 'radial-gradient(circle at 75% 25%, rgba(46, 139, 87, 0.05) 0%, transparent 50%)',
          }}>
            <Sidebar 
              mobileOpen={mobileOpen} 
              handleDrawerToggle={handleDrawerToggle}
              darkMode={darkMode}
              toggleDarkMode={toggleDarkMode}
            />
            <MainChat 
              handleDrawerToggle={handleDrawerToggle}
              darkMode={darkMode}
            />
            {/* API Configuration Manager */}
            <ApiConfigManager />
          </Box>
        </ChatProvider>
      </TopicProvider>
      </SettingsProvider>
    </ThemeProvider>
  );
}

export default App;