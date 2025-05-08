// app/page.tsx
"use client";
import { useState, useEffect } from 'react';
import MangaEditor, { Page, Project, Character } from '../components/MangaEditor';
import InitializeModels from '../components/InitializeModels';
import { FloatingStatus } from '../components/FloatingStatus';
import ProjectManager from '../components/ProjectManager';
import { API_ENDPOINT, API_BASE_URL } from '../config';

export default function Home() {
  // Model initialization states
  const [characters, setCharacters] = useState<Character[]>([]);
  const [loadingCharacters, setLoadingCharacters] = useState(false);
  
  // Status display states
  const [statusMinimized, setStatusMinimized] = useState(false);
  const [showModelStatus, setShowModelStatus] = useState(true);
  const [initStatus, setInitStatus] = useState({
    status: 'in_progress',
    message: 'Initializing models...',
    progress: 0
  });
  
  const [showInitializeDialog, setShowInitializeDialog] = useState(false);

  // Project management states
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [showProjectManager, setShowProjectManager] = useState(false);
  const [pages, setPages] = useState<Page[]>([]);
  const [apiEndpoint] = useState(API_ENDPOINT);


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

  // Effect to check for projects after characters are loaded
  useEffect(() => {
    if (characters.length > 0) {
      checkForExistingProjects();
    }
  }, [characters]);

  // Function to check for existing projects
  const checkForExistingProjects = async () => {
    try {
      // Try to fetch projects from API
      const response = await fetch(`${apiEndpoint}/projects`);
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        if (data.projects && data.projects.length > 0) {
          // Load the first project
          setCurrentProject(data.projects[0]);
          const projectPages = await loadProjectPages(data.projects[0].id);
          setPages(projectPages);
        } else {
          // No projects exist, create a default one
          await createDefaultProject();
        }
      } else {
        // API error, try localStorage
        const localProjects = JSON.parse(localStorage.getItem('mangaProjects') || '[]');
        if (localProjects.length > 0) {
          setCurrentProject(localProjects[0]);
          const projectPages = await loadProjectPages(localProjects[0].id);
          setPages(projectPages);
        } else {
          await createDefaultProject();
        }
      }
    } catch (error) {
      console.error('Error checking projects:', error);
      // Try localStorage as fallback
      const localProjects = JSON.parse(localStorage.getItem('mangaProjects') || '[]');
      if (localProjects.length > 0) {
        setCurrentProject(localProjects[0]);
        const projectPages = await loadProjectPages(localProjects[0].id);
        setPages(projectPages);
      } else {
        await createDefaultProject();
      }
    }
  };
  
  // Function to create a default project
  const createDefaultProject = async () => {
    try {
      const defaultProject = {
        name: "Untitled Manga"
      };
      
      // Create via API
      const response = await fetch(`${apiEndpoint}/projects`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(defaultProject)
      });
      
      const data = await response.json();
      
      if (response.ok && data.status === 'success' && data.project) {
        setCurrentProject(data.project as Project);
        setPages([]);
      } else {
        throw new Error('API project creation failed');
      }
    } catch (error) {
      console.error('Error creating default project:', error);
      // Fallback to localStorage
      const newProject: Project = {
        id: `project-${Date.now()}`,
        name: "Untitled Manga",
        pages: 0,
        lastModified: new Date().toISOString()
      };
      
      const projects = [newProject];
      localStorage.setItem('mangaProjects', JSON.stringify(projects));
      setCurrentProject(newProject);
      setPages([]);
    }
  };
  
  // Function to load project pages
  const loadProjectPages = async (projectId: string): Promise<Page[]> => {
    try {
      // Try to load from API
      const response = await fetch(`${apiEndpoint}/projects/${projectId}`);
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.status === 'success' && data.pages) {
          return data.pages;
        }
      }
      
      // Fall back to localStorage
      const pagesJson = localStorage.getItem(`project_${projectId}`);
      if (pagesJson) {
        return JSON.parse(pagesJson);
      }
      
      return [];
    } catch (error) {
      console.error('Error loading project pages:', error);
      
      // Try localStorage as fallback
      const pagesJson = localStorage.getItem(`project_${projectId}`);
      if (pagesJson) {
        return JSON.parse(pagesJson);
      }
      
      return [];
    }
  };
  
  // Function to handle project selection
  const handleSelectProject = (project: Project, projectPages: Page[]) => {
    setCurrentProject(project);
    setPages(projectPages);
    setShowProjectManager(false);
  };
  
  // Function to handle project saving
  const handleSaveProject = (projectId: string, currentPages: Page[]) => {
    // This will be called from MangaEditor when pages change
    setPages(currentPages);
  };

  // Function to check if models are already initialized
  const checkInitializationStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/status`);
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
      {showInitializeDialog ? (
        <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-6 my-10">
          <h2 className="text-2xl font-bold mb-4 text-center text-red-600">Model Configuration</h2>
          <p className="mb-4">Initialize models to start generating manga panels</p>
          <InitializeModels onInitialized={handleInitialized} />
        </div>
      ) : (
        <div className="flex-1 overflow-hidden h-full flex flex-col">
          {/* Project manager overlay */}
          {showProjectManager && (
            <div className="absolute inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-4">
              <div className="w-full max-w-4xl">
                <ProjectManager
                  apiEndpoint={apiEndpoint}
                  onSelectProject={handleSelectProject}
                  onSaveProject={handleSaveProject}
                  currentPages={pages}
                />
              </div>
            </div>
          )}
          
          <MangaEditor 
            characters={characters} 
            apiEndpoint={apiEndpoint}
            currentProject={currentProject}
            setCurrentProject={setCurrentProject}
            pages={pages}
            setPages={setPages}
            onSaveProject={handleSaveProject}
            onShowProjectManager={() => setShowProjectManager(true)}
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