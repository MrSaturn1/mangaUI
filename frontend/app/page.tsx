// app/page.tsx
"use client";
import { useState, useEffect } from 'react';
import MangaEditor from '../components/MangaEditor';
import InitializeModels from '../components/InitializeModels';
import { FloatingStatus } from '../components/FloatingStatus';
import ProjectManager from '../components/ProjectManager';

export default function Home() {
  // Model initialization states
  const [characters, setCharacters] = useState([]);
  const [loadingCharacters, setLoadingCharacters] = useState(false);
  
  // Status display states
  const [statusMinimized, setStatusMinimized] = useState(false);
  const [showModelStatus, setShowModelStatus] = useState(true);
  const [initStatus, setInitStatus] = useState({
    status: 'in_progress',
    message: 'Initializing models...',
    progress: 0
  });
  
  // Force "show initialize model" dialog
  const [showInitializeDialog, setShowInitializeDialog] = useState(false);

  // States for project management
  const [currentProject, setCurrentProject] = useState(null);
  const [showProjectManager, setShowProjectManager] = useState(false);
  const [pages, setPages] = useState([]);


  useEffect(() => {
    // Start initialization right away
    startInitialization();
    
    // Set up polling for initialization status
    const intervalId = setInterval(() => {
      checkInitializationStatus();
    }, 1000);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  // Function to check if models are already initialized
  const checkInitializationStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/status');
      const data = await response.json();
      
      // Update status for floating indicator
      setInitStatus({
        status: data.initialized ? 'success' : 
               data.initializing ? 'in_progress' : 'error',
        message: data.message || 'Checking model status...',
        progress: data.progress || 0
      });
      
      // If already initialized, fetch characters
      if (data.initialized && !loadingCharacters) {
        await fetchCharacters();
        
        // Hide the status indicator after 2 seconds when initialization completes
        if (initStatus.status !== 'success') {
          setTimeout(() => {
            setShowModelStatus(false);
          }, 2000);
        }
      }
    } catch (error) {
      console.error('Error checking initialization status:', error);
      setInitStatus({
        status: 'error',
        message: 'Could not connect to server',
        progress: 0
      });
    }
  };

  // Function to start model initialization
  const startInitialization = async () => {
    try {
      setInitStatus({
        status: 'in_progress',
        message: 'Starting model initialization...',
        progress: 5
      });
      
      const response = await fetch('http://localhost:5000/api/init', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model_path: './drawatoon-v1',
          character_data_path: './characters.json',
          character_embedding_path: './character_output/character_embeddings/character_embeddings_map.json',
          output_dir: './manga_output'
        })
      });
      
      const data = await response.json();
      
      // Update status for display
      setInitStatus({
        status: data.status === 'success' ? 'success' : 
               data.status === 'in_progress' ? 'in_progress' : 'error',
        message: data.message || 'Initializing models...',
        progress: data.progress || 0
      });
      
      // If already successful (unlikely for async initialization)
      if (data.status === 'success') {
        await fetchCharacters();
      }
    } catch (error) {
      console.error('Error starting initialization:', error);
      setInitStatus({
        status: 'error',
        message: 'Failed to start initialization. Server might be down.',
        progress: 0
      });
      setShowInitializeDialog(true);
    }
  };

  // Function to fetch character data
  const fetchCharacters = async () => {
    try {
      setLoadingCharacters(true);
      const response = await fetch('http://localhost:5000/api/get_characters');
      const data = await response.json();
      
      if (data.status === 'success') {
        setCharacters(data.characters);
      } else {
        console.error('Error fetching characters:', data.message);
      }
    } catch (error) {
      console.error('Error fetching characters:', error);
    } finally {
      setLoadingCharacters(false);
    }
  };

  // When initialization completes from the dialog
  const handleInitialized = async () => {
    setShowInitializeDialog(false);
    await fetchCharacters();
    setShowModelStatus(true); // Show status again briefly
    setInitStatus({
      status: 'success',
      message: 'Models initialized successfully',
      progress: 100
    });
    
    // Hide again after 2 seconds
    setTimeout(() => {
      setShowModelStatus(false);
    }, 2000);
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Always show the editor */}
      {showInitializeDialog ? (
        <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-6 my-10">
          <h2 className="text-2xl font-bold mb-4 text-center text-red-600">Model Configuration</h2>
          <p className="mb-4">Initialize models to start generating manga panels</p>
          <InitializeModels onInitialized={handleInitialized} />
        </div>
      ) : (
        <div className="flex-1 overflow-hidden h-full flex flex-col">
          <MangaEditor 
            characters={characters} 
            apiEndpoint="http://localhost:5000/api"
          />
        </div>
      )}

      {/* Show floating status indicator while initializing or on error */}
      {showModelStatus && (initStatus.status === 'in_progress' || initStatus.status === 'error') && (
        <FloatingStatus
          status={initStatus.status}
          message={initStatus.message}
          progress={initStatus.progress}
          isMinimized={statusMinimized}
          onToggle={() => setStatusMinimized(!statusMinimized)}
        />
      )}
    </div>
  );
}