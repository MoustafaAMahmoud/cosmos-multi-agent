import React, { createContext, useState, useContext } from 'react';

const TopicContext = createContext();

export function useTopicContext() {
  return useContext(TopicContext);
}

export function TopicProvider({ children }) {
  // Initialize with null to show the topic selector by default
  const [selectedTopic, setSelectedTopic] = useState(null);
  
  // List of research topics for comprehensive analysis
  const topics = [
    { 
      id: 'vaping-health', 
      title: 'Vaping Health Effects', 
      description: 'Research on health impacts of vaping', 
      icon: 'help_outline',
      sampleQuestion: 'What are the comprehensive health effects of vaping compared to traditional smoking?'
    },
    { 
      id: 'heating-technology', 
      title: 'Heating Technology', 
      description: 'Advanced heating systems and innovations', 
      icon: 'tune',
      sampleQuestion: 'What are the latest innovations in heat-not-burn tobacco heating technologies?'
    },
    { 
      id: 'regulatory', 
      title: 'Regulatory Landscape', 
      description: 'Legal and regulatory frameworks', 
      icon: 'code',
      sampleQuestion: 'What are the current regulatory requirements for vaping products across different countries?'
    },
    { 
      id: 'market-trends', 
      title: 'Market Analysis', 
      description: 'Industry trends and market dynamics', 
      icon: 'speed',
      sampleQuestion: 'What are the current market trends and growth projections for the vaping industry?'
    },
    { 
      id: 'safety-standards', 
      title: 'Safety & Standards', 
      description: 'Safety protocols and quality standards', 
      icon: 'security',
      sampleQuestion: 'What safety standards and quality control measures are applied in vaping device manufacturing?'
    },
    { 
      id: 'research-studies', 
      title: 'Scientific Research', 
      description: 'Latest research findings and studies', 
      icon: 'build',
      sampleQuestion: 'What do the latest clinical studies reveal about the long-term effects of heat-not-burn products?'
    }
  ];

  const selectTopic = (topicId) => {
    const topic = topics.find(t => t.id === topicId);
    setSelectedTopic(topic);
    return topic;
  };
  
  const resetTopic = () => {
    setSelectedTopic(null);
  };

  return (
    <TopicContext.Provider value={{ topics, selectedTopic, selectTopic, resetTopic }}>
      {children}
    </TopicContext.Provider>
  );
}