import React, { useState } from 'react';
import { 
  Box, 
  Drawer, 
  Divider, 
  List, 
  ListItem, 
  ListItemButton, 
  ListItemIcon, 
  ListItemText, 
  IconButton,
  Typography, 
  useTheme,
  useMediaQuery,
  Avatar,
  Button,
  Tooltip,
  Switch,
  Chip
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import AddIcon from '@mui/icons-material/Add';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import SearchIcon from '@mui/icons-material/Search';
import HistoryIcon from '@mui/icons-material/History';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import ScienceIcon from '@mui/icons-material/Science';
import SourceIcon from '@mui/icons-material/Source';
import GitHubIcon from '@mui/icons-material/GitHub';
import EmailIcon from '@mui/icons-material/Email';
import { useTopicContext } from '../contexts/TopicContext';
import { useChatContext } from '../contexts/ChatContext';
import { useSettingsContext } from '../contexts/SettingsContext';
import { APP_CONFIG, FEATURE_FLAGS } from '../config/api';

const drawerWidth = 280;
const collapsedWidth = 60;

function Sidebar({ mobileOpen, handleDrawerToggle, darkMode, toggleDarkMode }) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { resetTopic } = useTopicContext();
  const { clearChat } = useChatContext();
  const { settings, toggleDebateDetails } = useSettingsContext();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleClearChat = () => {
    clearChat();
  };

  const handleGoToMainMenu = () => {
    clearChat();
    resetTopic();
  };

  const handleToggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  const drawer = (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%', 
      bgcolor: darkMode ? '#1a1a1a' : '#f8f9fa',
      borderRight: '1px solid',
      borderColor: darkMode ? '#333' : '#e0e0e0',
      transition: 'all 0.3s ease'
    }}>
      <Box sx={{ 
        p: isCollapsed ? 1 : 2, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between' 
      }}>
        {!isCollapsed && (
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <Box sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 36,
              height: 36,
              borderRadius: '8px',
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
              marginRight: '12px',
              boxShadow: '0 4px 12px rgba(46, 139, 87, 0.3)'
            }}>
              <ScienceIcon sx={{ color: 'white', fontSize: '1.5rem' }} />
            </Box>
            <Box>
              <Typography 
                variant="subtitle1" 
                component="div" 
                sx={{ 
                  fontWeight: 700, 
                  color: darkMode ? 'white' : 'inherit',
                  lineHeight: 1.2
                }}
              >
                {APP_CONFIG.APP_NAME}
              </Typography>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: darkMode ? 'rgba(255,255,255,0.7)' : 'text.secondary',
                  fontSize: '0.7rem',
                  fontWeight: 500
                }}
              >
                {APP_CONFIG.COMPANY_NAME}
              </Typography>
            </Box>
          </Box>
        )}
        
        {isCollapsed && (
          <Tooltip title={APP_CONFIG.APP_NAME} placement="right">
            <Box sx={{ 
              mx: 'auto', 
              my: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 40,
              height: 40,
              borderRadius: '10px',
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
              boxShadow: '0 4px 12px rgba(46, 139, 87, 0.3)'
            }}>
              <ScienceIcon sx={{ color: 'white', fontSize: '1.6rem' }} />
            </Box>
          </Tooltip>
        )}
        
        {(isMobile && !isCollapsed) && (
          <IconButton onClick={handleDrawerToggle} size="small" sx={{ color: darkMode ? 'white' : 'inherit' }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        )}
        
        {!isMobile && (
          <IconButton 
            onClick={handleToggleCollapse} 
            size="small" 
            sx={{ 
              color: darkMode ? 'white' : 'inherit',
              ...(isCollapsed && { mx: 'auto' })
            }}
          >
            {isCollapsed ? <ChevronRightIcon fontSize="small" /> : <ChevronLeftIcon fontSize="small" />}
          </IconButton>
        )}
      </Box>
      
      {!isCollapsed && <Divider sx={{ mb: 2, bgcolor: darkMode ? '#333' : undefined }} />}
      
      {!isCollapsed && (
        <Box sx={{ px: 2, mb: 2 }}>
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            mb: 1.5 
          }}>
            <Typography 
              variant="body2" 
              color={darkMode ? 'rgba(255,255,255,0.7)' : 'text.secondary'}
              sx={{ 
                letterSpacing: '0.5px',
                textTransform: 'uppercase',
                fontSize: '0.75rem',
                fontWeight: 600
              }}
            >
              Research Session
            </Typography>
          </Box>
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            mb: 2,
            borderRadius: 1,
            p: 1,
            '&:hover': {
              bgcolor: darkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)'
            }
          }}>
            <Avatar sx={{ 
              width: 32, 
              height: 32, 
              mr: 1.5, 
              bgcolor: theme.palette.primary.main,
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`
            }}>
              <ScienceIcon sx={{ fontSize: '1.1rem' }} />
            </Avatar>
            <Box>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontWeight: 500,
                  color: darkMode ? 'white' : 'inherit'
                }}
              >
                {APP_CONFIG.RESEARCH_AGENT_NAME}
              </Typography>
              <Chip 
                label="Active" 
                size="small" 
                sx={{ 
                  height: 16, 
                  fontSize: '0.65rem',
                  bgcolor: theme.palette.secondary.main,
                  color: 'white',
                  fontWeight: 600
                }} 
              />
            </Box>
          </Box>
        </Box>
      )}
      
      {isCollapsed ? (
        <Box sx={{ mt: 2 }}>
          <List disablePadding>
            <Tooltip title="New Research" placement="right">
              <ListItem disablePadding sx={{ mb: 1 }}>
                <ListItemButton 
                  onClick={handleGoToMainMenu}
                  sx={{ 
                    borderRadius: 2,
                    py: 1,
                    minHeight: 0,
                    mx: 'auto',
                    width: 40,
                    justifyContent: 'center',
                    bgcolor: theme.palette.primary.main,
                    color: 'white',
                    '&:hover': {
                      bgcolor: theme.palette.primary.dark,
                      transform: 'scale(1.05)'
                    }
                  }}
                >
                  <SearchIcon fontSize="small" />
                </ListItemButton>
              </ListItem>
            </Tooltip>
            
            <Tooltip title="Clear Session" placement="right">
              <ListItem disablePadding sx={{ mb: 1 }}>
                <ListItemButton 
                  onClick={handleClearChat}
                  sx={{ 
                    borderRadius: 2,
                    py: 1,
                    minHeight: 0,
                    mx: 'auto',
                    width: 40,
                    justifyContent: 'center'
                  }}
                >
                  <AddIcon fontSize="small" sx={{ color: darkMode ? 'white' : theme.palette.primary.main }} />
                </ListItemButton>
              </ListItem>
            </Tooltip>
            
            <Tooltip title="Research Documentation" placement="right">
              <ListItem disablePadding sx={{ mb: 1 }}>
                <ListItemButton
                  component="a"
                  href={APP_CONFIG.DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{ 
                    borderRadius: 2,
                    py: 1,
                    minHeight: 0,
                    mx: 'auto',
                    width: 40,
                    justifyContent: 'center'
                  }}
                >
                  <MenuBookIcon fontSize="small" sx={{ color: darkMode ? 'white' : theme.palette.primary.main }} />
                </ListItemButton>
              </ListItem>
            </Tooltip>
            
            <Tooltip title={darkMode ? "Light Mode" : "Dark Mode"} placement="right">
              <ListItem disablePadding sx={{ mb: 1 }}>
                <ListItemButton 
                  onClick={toggleDarkMode}
                  sx={{ 
                    borderRadius: 1,
                    py: 1,
                    minHeight: 0,
                    mx: 'auto',
                    width: 40,
                    justifyContent: 'center'
                  }}
                >
                  {darkMode ? 
                    <LightModeIcon fontSize="small" sx={{ color: 'white' }} /> :
                    <DarkModeIcon fontSize="small" sx={{ color: '#68768A' }} />
                  }
                </ListItemButton>
              </ListItem>
            </Tooltip>
          </List>
        </Box>
      ) : (
        <>
          <Box sx={{ px: 2, mb: 2 }}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              mb: 1 
            }}>
              <Typography 
                variant="body2" 
                color={darkMode ? 'rgba(255,255,255,0.7)' : 'text.secondary'}
                sx={{ 
                  letterSpacing: '0.5px',
                  textTransform: 'uppercase',
                  fontSize: '0.75rem',
                  fontWeight: 600
                }}
              >
                Research Tools
              </Typography>
            </Box>
            
            <Button
              variant="contained"
              fullWidth
              startIcon={<SearchIcon />}
              onClick={handleGoToMainMenu}
              sx={{ 
                mb: 2, 
                textTransform: 'none',
                borderRadius: '8px',
                py: 1,
                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                boxShadow: '0 4px 12px rgba(46, 139, 87, 0.3)',
                '&:hover': {
                  transform: 'translateY(-1px)',
                  boxShadow: '0 6px 16px rgba(46, 139, 87, 0.4)'
                }
              }}
            >
              New Research
            </Button>
            
            <Button
              variant="outlined"
              fullWidth
              startIcon={<DeleteSweepIcon />}
              onClick={handleClearChat}
              sx={{ 
                mb: 2, 
                textTransform: 'none',
                borderRadius: '8px',
                py: 1,
                borderColor: theme.palette.primary.main,
                color: theme.palette.primary.main,
                '&:hover': {
                  bgcolor: `${theme.palette.primary.main}15`,
                  borderColor: theme.palette.primary.dark
                }
              }}
            >
              Clear Session
            </Button>
          </Box>
          
          <Box sx={{ mt: 'auto' }}>
            <Divider sx={{ bgcolor: darkMode ? '#333' : undefined }} />
            <List disablePadding sx={{ p: 1 }}>
              <ListItem disablePadding>
                <ListItemButton 
                  component="a"
                  href={APP_CONFIG.DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{ 
                    borderRadius: 1,
                    color: darkMode ? 'white' : 'inherit'
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <MenuBookIcon fontSize="small" sx={{ color: darkMode ? 'white' : theme.palette.primary.main }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Research Docs" 
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItemButton>
              </ListItem>
              
              <ListItem disablePadding>
                <ListItemButton 
                  component="a"
                  href={APP_CONFIG.GITHUB_REPO}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{ 
                    borderRadius: 1,
                    color: darkMode ? 'white' : 'inherit'
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <GitHubIcon fontSize="small" sx={{ color: darkMode ? 'white' : theme.palette.primary.main }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Source Code" 
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItemButton>
              </ListItem>
              
              <ListItem disablePadding>
                <ListItemButton 
                  component="a"
                  href={`mailto:${APP_CONFIG.SUPPORT_EMAIL}`}
                  sx={{ 
                    borderRadius: 1,
                    color: darkMode ? 'white' : 'inherit'
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <EmailIcon fontSize="small" sx={{ color: darkMode ? 'white' : theme.palette.primary.main }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Support" 
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItemButton>
              </ListItem>
              <ListItem disablePadding>
                <ListItemButton 
                  onClick={toggleDarkMode}
                  sx={{ 
                    borderRadius: 1,
                    color: darkMode ? 'white' : 'inherit' 
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    {darkMode ? 
                      <LightModeIcon fontSize="small" sx={{ color: 'white' }} /> :
                      <DarkModeIcon fontSize="small" />
                    }
                  </ListItemIcon>
                  <ListItemText 
                    primary={darkMode ? "Light Mode" : "Dark Mode"} 
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItemButton>
              </ListItem>
              <ListItem disablePadding>
                <ListItemButton 
                  onClick={toggleDebateDetails}
                  sx={{ 
                    borderRadius: 1,
                    color: darkMode ? 'white' : 'inherit' 
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Switch
                      checked={settings.includeDebateDetails}
                      size="small"
                      sx={{ 
                        '& .MuiSwitch-thumb': {
                          backgroundColor: settings.includeDebateDetails ? theme.palette.primary.main : undefined
                        }
                      }}
                    />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Research Process Details" 
                    primaryTypographyProps={{ variant: 'body2' }}
                    secondary={settings.includeDebateDetails ? "Showing analysis steps" : "Hiding analysis steps"}
                    secondaryTypographyProps={{ 
                      variant: 'caption',
                      sx: { color: darkMode ? 'rgba(255,255,255,0.7)' : 'text.secondary' }
                    }}
                  />
                </ListItemButton>
              </ListItem>
            </List>
          </Box>
        </>
      )}
    </Box>
  );

  return (
    <>
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        ModalProps={{
          keepMounted: true, // Better mobile performance
        }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': { 
            boxSizing: 'border-box', 
            width: drawerWidth,
            boxShadow: 3
          },
        }}
      >
        {drawer}
      </Drawer>
      
      {/* Desktop drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': { 
            boxSizing: 'border-box', 
            width: isCollapsed ? collapsedWidth : drawerWidth,
            position: 'relative',
            height: '100%',
            border: 'none',
            transition: 'width 0.3s ease'
          },
        }}
        open
      >
        {drawer}
      </Drawer>
    </>
  );
}

export default Sidebar;