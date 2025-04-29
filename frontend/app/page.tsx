// app/page.tsx
'use client';

import { useState, useEffect } from 'react';
import MangaEditor from '../components/MangaEditor';
import InitializeModels from '../components/InitializeModels';

export default function Home() {
  const [isInitialized, setIsInitialized] = useState<boolean>(false);
  const [characters, setCharacters] = useState([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  
  useEffect(() => {
    checkInitialization();
  }, []);
  
  const checkInitialization = async () => {
    try {
      setIsLoading(true);
      // Check if the backend is initialized
      const response = await fetch('http://localhost:5000/api/get_characters');
      const data = await response.json();
      
      if (data.status === 'success') {
        setIsInitialized(true);
        setCharacters(data.characters);
      }
    } catch (error) {
      console.error('Error checking initialization:', error);
      setIsInitialized(false);
    } finally {
      setIsLoading(false);
    }
  };
  
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8 flex justify-center items-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">Manga Creator Studio</h1>
      
      {isInitialized ? (
        <MangaEditor characters={characters} />
      ) : (
        <InitializeModels onInitialized={checkInitialization} />
      )}
    </div>
  );
}