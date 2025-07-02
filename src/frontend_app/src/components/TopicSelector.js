import React from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  useTheme,
  Fade,
  Button,
  Zoom
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import TuneIcon from '@mui/icons-material/Tune';
import CodeIcon from '@mui/icons-material/Code';
import SpeedIcon from '@mui/icons-material/Speed';
import SecurityIcon from '@mui/icons-material/Security';
import BuildIcon from '@mui/icons-material/Build';
import ScienceIcon from '@mui/icons-material/Science';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import WhatshotIcon from '@mui/icons-material/Whatshot';
import BalanceIcon from '@mui/icons-material/Balance';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import BiotechIcon from '@mui/icons-material/Biotech';
import { useTopicContext } from '../contexts/TopicContext';
import { useChatContext } from '../contexts/ChatContext';
import { APP_CONFIG } from '../config/api';

// Map icon names to Material UI icon components for research topics
const iconMap = {
  'help_outline': <HealthAndSafetyIcon />,  // Vaping Health Effects
  'tune': <WhatshotIcon />,                 // Heating Technology  
  'code': <BalanceIcon />,                  // Regulatory Landscape
  'speed': <TrendingUpIcon />,              // Market Analysis
  'security': <VerifiedUserIcon />,         // Safety & Standards
  'build': <BiotechIcon />                  // Scientific Research
};

function TopicSelector({ darkMode }) {
  const { topics, selectTopic } = useTopicContext();
  const { sendMessage } = useChatContext();
  const theme = useTheme();
  
  const handleTopicClick = (topicId) => {
    const topic = selectTopic(topicId);
    if (topic && topic.sampleQuestion) {
      // Small delay to allow the welcome message to appear first
      setTimeout(() => {
        sendMessage(topic.sampleQuestion);
      }, 800);
    }
  };
  
  return (
    <Box sx={{ width: '100%', maxWidth: '900px', mx: 'auto', mt: 4, px: 2 }}>
      <Fade in={true} timeout={500}>
        <Box>
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <Zoom in={true} timeout={800}>
              <Box 
                sx={{ 
                  position: 'relative',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: 120,
                  height: 120,
                  mb: 2,
                  background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  borderRadius: '20px',
                  boxShadow: '0 8px 32px rgba(46, 139, 87, 0.3)',
                  '&:hover': {
                    transform: 'scale(1.05)',
                    boxShadow: '0 12px 40px rgba(46, 139, 87, 0.4)',
                  },
                  transition: 'all 0.3s ease'
                }}
              >
                <ScienceIcon sx={{ fontSize: '4rem', color: 'white' }} />
                <Box 
                  className="research-particle"
                  sx={{
                    position: 'absolute',
                    top: -10,
                    left: 10,
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    backgroundColor: theme.palette.secondary.main,
                    filter: 'blur(1px)',
                    animation: 'float 3s ease-in-out infinite'
                  }}
                />
                <Box 
                  className="research-particle"
                  sx={{
                    position: 'absolute',
                    bottom: 5,
                    right: -5,
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor: theme.palette.primary.light,
                    filter: 'blur(1px)',
                    animation: 'float 2s ease-in-out infinite reverse'
                  }}
                />
                <AutoAwesomeIcon 
                  sx={{
                    position: 'absolute',
                    top: 10,
                    right: 10,
                    fontSize: '1.2rem',
                    color: 'white',
                    opacity: 0.8,
                    animation: 'sparkle 2s ease-in-out infinite'
                  }}
                />
              </Box>
            </Zoom>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 700, mt: 1 }}>
              {APP_CONFIG.APP_NAME}
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto', lineHeight: 1.6 }}>
              {APP_CONFIG.APP_SUBTITLE} - Leverage AI-powered multi-agent collaboration to find comprehensive answers with exhaustive source analysis.
            </Typography>
          </Box>
          
          <Box sx={{ mb: 3, mt: 6 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Choose a research area to begin your comprehensive analysis:
            </Typography>
          </Box>
          
          <Box sx={{ 
            display: 'grid', 
            gridTemplateColumns: {
              xs: '1fr',
              sm: 'repeat(auto-fill, minmax(260px, 1fr))'
            },
            gap: 3
          }}>
            {topics.map((topic, index) => (
              <Zoom key={topic.id} in={true} style={{ transitionDelay: `${index * 100}ms` }}>
                <Card 
                  className="topic-card"
                  elevation={0}
                  sx={{ 
                    height: '100%',
                    cursor: 'pointer',
                    bgcolor: topic.id === 'performance' ? theme.palette.primary.main : 
                            topic.id === 'api' ? (darkMode ? '#1a334d' : '#E8F5FE') :
                            topic.id === 'security' ? (darkMode ? '#1a3047' : '#E6F3FF') : 
                            darkMode ? '#1e1e1e' : 'white',
                    color: topic.id === 'performance' ? 'white' : 
                            darkMode ? 'white' : 'inherit',
                    boxShadow: darkMode ? '0 4px 12px rgba(0, 0, 0, 0.3)' : 'none',
                    '& .topic-icon': {
                      bgcolor: topic.id === 'performance' ? 'rgba(255,255,255,0.2)' : 
                                topic.id === 'api' ? '#50B0F9' :
                                topic.id === 'security' ? '#0078D4' : theme.palette.primary.main,
                      color: topic.id === 'performance' ? 'white' : 
                              topic.id === 'api' ? 'white' :
                              topic.id === 'security' ? 'white' : 'white'
                    },
                    '& .topic-description': {
                      color: topic.id === 'performance' ? 'rgba(255,255,255,0.8)' : 
                             darkMode ? 'rgba(255,255,255,0.7)' : 'text.secondary'
                    }
                  }}
                  onClick={() => handleTopicClick(topic.id)}
                >
                  <CardContent sx={{ 
                    height: '100%', 
                    display: 'flex', 
                    flexDirection: 'column',
                    p: 3
                  }}>
                    <Box 
                      className="topic-icon"
                      sx={{ 
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: 40,
                        height: 40,
                        borderRadius: '8px',
                        mb: 2
                      }}
                    >
                      {iconMap[topic.icon]}
                    </Box>
                    <Typography variant="h6" component="div" sx={{ mb: 1, fontWeight: 600 }}>
                      {topic.title}
                    </Typography>
                    <Typography variant="body2" className="topic-description">
                      {topic.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Zoom>
            ))}
          </Box>
          
          <Box sx={{ textAlign: 'center', mt: 6 }}>
            <Button 
              variant="outlined" 
              color="primary"
              startIcon={<HelpOutlineIcon />}
              sx={{ 
                borderRadius: 20, 
                px: 3, 
                py: 1,
                boxShadow: darkMode 
                  ? '0 2px 8px rgba(0, 120, 212, 0.25)'
                  : '0 2px 8px rgba(0, 120, 212, 0.15)',
                color: darkMode ? '#fff' : undefined,
                borderColor: darkMode ? 'rgba(255, 255, 255, 0.3)' : undefined,
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: darkMode
                    ? '0 4px 12px rgba(0, 120, 212, 0.4)'
                    : '0 4px 12px rgba(0, 120, 212, 0.2)',
                  borderColor: darkMode ? 'rgba(255, 255, 255, 0.5)' : undefined,
                },
                transition: 'all 0.3s ease'
              }}
            >
              Ask a specific question to get started
            </Button>
          </Box>
        </Box>
      </Fade>
    </Box>
  );
}

export default TopicSelector;