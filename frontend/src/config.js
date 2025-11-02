// API Configuration
const isProduction = process.env.NODE_ENV === 'production';

// Use production URL if in production, otherwise use local development URL
export const API_URL = isProduction 
  ? 'https://your-backend-api.herokuapp.com'  // Replace with your actual backend URL when deployed
  : 'http://localhost:8000';

export default {
  API_URL,
};
