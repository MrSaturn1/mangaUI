// frontend/config.ts

// For local development
const DEFAULT_API_URL = 'http://localhost:5000';

// Environment-aware configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_URL;
export const API_ENDPOINT = `${API_BASE_URL}/api`;

// Image path utilities
export function normalizeImagePath(path: string | undefined): string {
  if (!path) return '';
  
  // Already a data URL (base64)
  if (path.startsWith('data:')) return path;
  
  // Already an absolute URL
  if (path.startsWith('http')) return path;
  
  // Ensure path starts with /
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  
  // Combine with API base URL
  return `${API_BASE_URL}${normalizedPath}`;
}