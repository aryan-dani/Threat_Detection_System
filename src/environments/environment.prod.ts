export const environment = {
  production: true,
  apiUrl: '/api',
  websocketUrl: window.location.origin.replace('http', 'ws'),
  detectionRefreshRate: 500, // milliseconds
  confidenceThreshold: 0.6,
  enableDebugMode: false
};
