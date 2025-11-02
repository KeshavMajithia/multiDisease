// API Configuration
const isProduction = process.env.NODE_ENV === 'production';

// Use production URL if in production, otherwise use local development URL
export const API_URL = isProduction 
  ? 'https://multidisease-dxqd.onrender.com'  // Production backend URL
  : 'http://localhost:8000';

export default {
  API_URL,
};
