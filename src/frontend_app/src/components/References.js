import React, { useState } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  IconButton, 
  Collapse, 
  Chip,
  useTheme,
  Divider
} from '@mui/material';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import ArticleIcon from '@mui/icons-material/Article';
import SourceIcon from '@mui/icons-material/Source';
import ImageIcon from '@mui/icons-material/Image';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';

function Sources({ sources = [], sourceCount = 0, darkMode = false, researchSummary = "", researchTopic = "" }) {
  const [expanded, setExpanded] = useState(false);
  const theme = useTheme();

  if (!sources || sources.length === 0) {
    return null;
  }

  const toggleExpand = () => {
    setExpanded(!expanded);
  };

  return (
    <Box sx={{ mt: 2, width: '100%' }}>
      {/* Sources Header */}
      <Box 
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          cursor: 'pointer',
          userSelect: 'none',
          mb: 1,
          py: 1
        }}
        onClick={toggleExpand}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SourceIcon 
            sx={{ 
              color: darkMode ? '#90caf9' : theme.palette.primary.main,
              fontSize: '1.1rem'
            }} 
          />
          <Typography 
            variant="subtitle2" 
            sx={{ 
              fontWeight: 600,
              color: darkMode ? '#e0e0e0' : '#333',
              fontSize: '0.875rem'
            }}
          >
            Sources
          </Typography>
        </Box>
        <IconButton 
          size="small" 
          onClick={toggleExpand}
          sx={{ color: darkMode ? '#aaa' : '#666', ml: 1 }}
        >
          {expanded ? <KeyboardArrowDownIcon fontSize="small" /> : <KeyboardArrowRightIcon fontSize="small" />}
        </IconButton>
      </Box>

      {/* Horizontal Sources Preview */}
      <Box sx={{ 
        display: 'flex', 
        gap: 1, 
        overflowX: 'auto',
        pb: 1,
        '&::-webkit-scrollbar': {
          height: 4,
        },
        '&::-webkit-scrollbar-track': {
          backgroundColor: darkMode ? '#333' : '#f1f1f1',
          borderRadius: 2,
        },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: darkMode ? '#666' : '#c1c1c1',
          borderRadius: 2,
        }
      }}>
        {sources.map((source, index) => (
          <Paper
            key={index}
            elevation={0}
            sx={{
              minWidth: 200,
              maxWidth: 250,
              p: 1.5,
              borderRadius: '8px',
              border: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
              bgcolor: darkMode ? '#1a1a1a' : '#fafafa',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              '&:hover': {
                bgcolor: darkMode ? '#2a2a2a' : '#f0f0f0',
                borderColor: theme.palette.primary.main,
              }
            }}
            onClick={(e) => {
              e.stopPropagation();
              if (source.document_url) {
                window.open(source.document_url, '_blank', 'noopener,noreferrer');
              } else {
                setExpanded(!expanded);
              }
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              {/* Citation Number */}
              <Box
                sx={{
                  width: 20,
                  height: 20,
                  borderRadius: '50%',
                  bgcolor: theme.palette.primary.main,
                  color: 'white',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  flexShrink: 0,
                  mt: 0.25
                }}
              >
                {source.citation_number}
              </Box>
              
              <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                <Typography 
                  variant="body2" 
                  sx={{ 
                    fontWeight: 500,
                    color: darkMode ? '#e0e0e0' : '#333',
                    fontSize: '0.8rem',
                    lineHeight: 1.2,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}
                >
                  {source.title}
                </Typography>
                
                {(source.topic_relation_summary || source.content_snippet) && (
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      color: darkMode ? '#aaa' : '#666',
                      fontSize: '0.7rem',
                      lineHeight: 1.2,
                      mt: 0.5,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                    }}
                  >
                    {source.topic_relation_summary || source.content_snippet}
                  </Typography>
                )}
              </Box>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                {source.content_type === 'image' && (
                  <ImageIcon 
                    sx={{ 
                      color: darkMode ? '#666' : '#999',
                      fontSize: '0.9rem',
                      flexShrink: 0
                    }} 
                  />
                )}
                {source.document_url && (
                  <OpenInNewIcon 
                    sx={{ 
                      color: theme.palette.primary.main,
                      fontSize: '0.8rem',
                      flexShrink: 0
                    }} 
                  />
                )}
              </Box>
            </Box>
          </Paper>
        ))}
      </Box>
      
      {/* Expanded Sources List */}
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <Paper
          elevation={0}
          sx={{
            mt: 2,
            borderRadius: '12px',
            border: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
            bgcolor: darkMode ? '#1a1a1a' : 'white',
            overflow: 'hidden'
          }}
        >
          <Box sx={{ p: 2 }}>
            <Typography 
              variant="subtitle2" 
              sx={{ 
                fontWeight: 600,
                color: darkMode ? '#e0e0e0' : '#333',
                mb: 1
              }}
            >
              Exhaustive Research: {researchTopic} ({sourceCount} sources)
            </Typography>
            
            {researchSummary && (
              <Typography 
                variant="body2" 
                sx={{ 
                  color: darkMode ? '#b0b0b0' : '#555',
                  fontSize: '0.85rem',
                  lineHeight: 1.4,
                  mb: 2,
                  p: 1.5,
                  bgcolor: darkMode ? '#2a2a2a' : '#f8f9fa',
                  borderRadius: 1,
                  borderLeft: `3px solid ${theme.palette.primary.main}`
                }}
              >
                {researchSummary}
              </Typography>
            )}
            
            <Typography 
              variant="body2" 
              sx={{ 
                fontWeight: 500,
                color: darkMode ? '#d0d0d0' : '#444',
                mb: 2
              }}
            >
              Source Details & Topic Relations
            </Typography>
            
            {sources.map((source, index) => (
              <Box key={index}>
                <Box 
                  sx={{ 
                    display: 'flex', 
                    alignItems: 'flex-start', 
                    gap: 2, 
                    py: 2,
                    cursor: source.document_url ? 'pointer' : 'default',
                    borderRadius: 1,
                    '&:hover': source.document_url ? {
                      bgcolor: darkMode ? '#2a2a2a' : '#f5f5f5'
                    } : {}
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (source.document_url) {
                      window.open(source.document_url, '_blank', 'noopener,noreferrer');
                    }
                  }}
                >
                  {/* Citation Number */}
                  <Box
                    sx={{
                      width: 24,
                      height: 24,
                      borderRadius: '50%',
                      bgcolor: theme.palette.primary.main,
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      flexShrink: 0
                    }}
                  >
                    {source.citation_number}
                  </Box>
                  
                  <Box sx={{ flexGrow: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      {source.content_type === 'image' ? (
                        <ImageIcon sx={{ color: darkMode ? '#90caf9' : theme.palette.primary.main, fontSize: '1rem' }} />
                      ) : (
                        <ArticleIcon sx={{ color: darkMode ? '#90caf9' : theme.palette.primary.main, fontSize: '1rem' }} />
                      )}
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          fontWeight: 500,
                          color: darkMode ? '#e0e0e0' : '#333',
                          flexGrow: 1
                        }}
                      >
                        {source.title}
                      </Typography>
                      {source.document_url && (
                        <OpenInNewIcon 
                          sx={{ 
                            color: theme.palette.primary.main,
                            fontSize: '0.9rem',
                            opacity: 0.7
                          }} 
                        />
                      )}
                    </Box>
                    
                    {source.topic_relation_summary && (
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          color: darkMode ? '#90caf9' : theme.palette.primary.main,
                          fontSize: '0.85rem',
                          lineHeight: 1.4,
                          fontWeight: 500,
                          mb: 1
                        }}
                      >
                        {source.topic_relation_summary}
                      </Typography>
                    )}
                    
                    {source.content_snippet && (
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          color: darkMode ? '#ccc' : '#555',
                          fontSize: '0.8rem',
                          lineHeight: 1.4,
                          fontStyle: 'italic'
                        }}
                      >
                        "{source.content_snippet}"
                      </Typography>
                    )}
                  </Box>
                </Box>
                
                {index < sources.length - 1 && (
                  <Divider sx={{ borderColor: darkMode ? '#333' : '#eee' }} />
                )}
              </Box>
            ))}
          </Box>
        </Paper>
      </Collapse>
    </Box>
  );
}

export default Sources;