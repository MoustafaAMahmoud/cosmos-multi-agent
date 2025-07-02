/**
 * API Configuration File
 * 
 * This file contains the configuration for API endpoints.
 * All values are read from environment variables.
 */

// API Configuration from environment variables
const API_HOST = process.env.REACT_APP_API_HOST || 'localhost';
const API_PORT = process.env.REACT_APP_API_PORT || '8000';
const API_PROTOCOL = process.env.REACT_APP_API_PROTOCOL || 'http';
const API_BASE_PATH = process.env.REACT_APP_API_BASE_PATH || '/api/v1';

// Request configuration from environment
const REQUEST_TIMEOUT = parseInt(process.env.REACT_APP_REQUEST_TIMEOUT) || 30000;

// Construct the base URL
export const API_BASE_URL = `${API_PROTOCOL}://${API_HOST}:${API_PORT}${API_BASE_PATH}`;

// API endpoints for research-support
export const ENDPOINTS = {
  CHAT: `${API_BASE_URL}/research-support`,
};

// Export timeout configuration
export { REQUEST_TIMEOUT };

/**
 * Configure global fetch options
 * @returns {Object} Default fetch options
 */
export const getDefaultFetchOptions = () => ({
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  credentials: 'omit', // Don't send credentials for CORS
  mode: 'cors', // This ensures CORS is used properly
});

// Application configuration from environment
export const APP_CONFIG = {
  APP_NAME: process.env.REACT_APP_APP_NAME || 'Deep Research Agent',
  APP_SUBTITLE: process.env.REACT_APP_APP_SUBTITLE || 'AI-Powered Comprehensive Research Platform',
  COMPANY_NAME: process.env.REACT_APP_COMPANY_NAME || 'Microsoft Research',
  LOGO_PATH: process.env.REACT_APP_LOGO_PATH || '/research-agent-logo.png',
  FAVICON_PATH: process.env.REACT_APP_FAVICON_PATH || '/favicon.ico',
  RESEARCH_AGENT_NAME: process.env.REACT_APP_RESEARCH_AGENT_NAME || 'Research Assistant',
  MAX_SOURCES_DISPLAY: parseInt(process.env.REACT_APP_MAX_SOURCES_DISPLAY) || 50,
  DEFAULT_RESEARCH_MODE: process.env.REACT_APP_DEFAULT_RESEARCH_MODE || 'exhaustive',
  ENABLE_DEBATE_DETAILS: process.env.REACT_APP_ENABLE_DEBATE_DETAILS === 'true',
  DOCS_URL: process.env.REACT_APP_DOCS_URL || 'https://docs.microsoft.com/azure/ai/research',
  SUPPORT_EMAIL: process.env.REACT_APP_SUPPORT_EMAIL || 'research-support@microsoft.com',
  GITHUB_REPO: process.env.REACT_APP_GITHUB_REPO || 'https://github.com/microsoft/deep-research-agent'
};

// Theme configuration from environment
export const THEME_CONFIG = {
  PRIMARY_COLOR: process.env.REACT_APP_PRIMARY_COLOR || '#2E8B57',
  SECONDARY_COLOR: process.env.REACT_APP_SECONDARY_COLOR || '#20B2AA',
  ACCENT_COLOR: process.env.REACT_APP_ACCENT_COLOR || '#FF6B6B',
  BACKGROUND_LIGHT: process.env.REACT_APP_BACKGROUND_LIGHT || '#FAFAFA',
  BACKGROUND_DARK: process.env.REACT_APP_BACKGROUND_DARK || '#121212'
};

// Feature flags from environment
export const FEATURE_FLAGS = {
  ENABLE_DARK_MODE: process.env.REACT_APP_ENABLE_DARK_MODE !== 'false',
  ENABLE_SOURCE_PREVIEW: process.env.REACT_APP_ENABLE_SOURCE_PREVIEW !== 'false',
  ENABLE_CITATION_EXPORT: process.env.REACT_APP_ENABLE_CITATION_EXPORT !== 'false',
  ENABLE_RESEARCH_HISTORY: process.env.REACT_APP_ENABLE_RESEARCH_HISTORY !== 'false'
};

// Export as named configuration object
const apiConfig = {
  API_BASE_URL,
  ENDPOINTS,
  REQUEST_TIMEOUT,
  getDefaultFetchOptions,
  APP_CONFIG,
  THEME_CONFIG,
  FEATURE_FLAGS
};

export default apiConfig;