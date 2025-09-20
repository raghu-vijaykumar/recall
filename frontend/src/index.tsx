import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  console.log('Initializing Recall application...');

  const container = document.getElementById('app');
  if (container) {
    const root = createRoot(container);
    root.render(<App />);
    console.log('Recall application initialized successfully!');
  } else {
    console.error('App container not found');
  }
});
