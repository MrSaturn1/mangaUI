// app/page.tsx
"use client";
import { useState, useEffect } from 'react';
import MangaEditor, { Page, Project, Character } from '../components/MangaEditor';
import InitializeModels from '../components/InitializeModels';
import { FloatingStatus } from '../components/FloatingStatus';
import ProjectManager from '../components/ProjectManager';
import ProjectExport from '../components/ProjectExport';
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
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [apiEndpoint] = useState(API_ENDPOINT);

  // Export states
  const [showExportDialog, setShowExportDialog] = useState(false);

  // Add state for tracking loading status
  const [isLoadingProject, setIsLoadingProject] = useState(true);
  const [isLoadingCharacters, setIsLoadingCharacters] = useState(false);

  // Page size configuration (should match your MangaEditor)
  const pageSize = { width: 1500, height: 2250 }; // Manga proportions

  // Separate initialization checks
  useEffect(() => {
    // Start all processes in parallel
    startInitialization();
    loadProjectData();
    loadCharactersImmediately(); // Load characters immediately, don't wait for models
    
    // Set up polling for initialization status
    const intervalId = setInterval(() => {
      checkInitializationStatus();
    }, 1000);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  // Load project data immediately, don't wait for models
  const loadProjectData = async () => {
    setIsLoadingProject(true);
    
    try {
      // Try to fetch projects from API
      const response = await fetch(`${apiEndpoint}/projects`);
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        if (data.projects && data.projects.length > 0) {
          // Try to load the last used project from localStorage
          const lastProjectId = localStorage.getItem('lastUsedProjectId');
          let projectToLoad = data.projects[0]; // Default to first project
          
          if (lastProjectId) {
            const lastProject = data.projects.find(p => p.id === lastProjectId);
            if (lastProject) {
              projectToLoad = lastProject;
            }
          }
          
          setCurrentProject(projectToLoad);
          
          // Load project pages
          const projectResponse = await fetch(`${apiEndpoint}/projects/${projectToLoad.id}`);
          const projectData = await projectResponse.json();
          
          if (projectResponse.ok && projectData.status === 'success') {
            setPages(projectData.pages || []);
          }
        } else {
          // No projects exist, create a default one
          await createDefaultProject();
        }
      } else {
        // Fallback to localStorage
        loadProjectsFromLocalStorage();
      }
    } catch (error) {
      console.error('Error loading projects:', error);
      loadProjectsFromLocalStorage();
    } finally {
      setIsLoadingProject(false);
    }
  };

  // Helper function to load from localStorage
  const loadProjectsFromLocalStorage = () => {
    const localProjects = JSON.parse(localStorage.getItem('mangaProjects') || '[]');
    if (localProjects.length > 0) {
      // Try to load the last used project from localStorage
      const lastProjectId = localStorage.getItem('lastUsedProjectId');
      let projectToLoad = localProjects[0]; // Default to first project
      
      if (lastProjectId) {
        const lastProject = localProjects.find(p => p.id === lastProjectId);
        if (lastProject) {
          projectToLoad = lastProject;
        }
      }
      
      setCurrentProject(projectToLoad);
      const projectPages = JSON.parse(
        localStorage.getItem(`project_${projectToLoad.id}`) || '[]'
      );
      setPages(projectPages);
    } else {
      createDefaultProject();
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
        const newProject = data.project as Project;
        setCurrentProject(newProject);
        // Remember this as the last used project
        localStorage.setItem('lastUsedProjectId', newProject.id);
        // New projects now start with 1 page from the backend
        setPages([{
          id: 'page-1',
          panels: []
        }]);
      } else {
        throw new Error('API project creation failed');
      }
    } catch (error) {
      console.error('Error creating default project:', error);
      // Fallback to localStorage
      const newProject: Project = {
        id: `project-${Date.now()}`,
        name: "Untitled Manga",
        pages: 1, // Changed from 0 to 1
        lastModified: new Date().toISOString()
      };
      
      const projects = [newProject];
      localStorage.setItem('mangaProjects', JSON.stringify(projects));
      setCurrentProject(newProject);
      // Remember this as the last used project
      localStorage.setItem('lastUsedProjectId', newProject.id);
      setPages([{
        id: 'page-1',
        panels: []
      }]);
    }
  };
  
  // Function to load project pages (not needed with new approach but keeping for compatibility)
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
  const handleSelectProject = async (project: Project, projectPages: Page[]) => {
    setCurrentProject(project);
    
    // Remember this as the last used project
    localStorage.setItem('lastUsedProjectId', project.id);
    
    // If pages weren't provided, fetch them
    if (!projectPages || projectPages.length === 0) {
      const pages = await loadProjectPages(project.id);
      setPages(pages);
    } else {
      setPages(projectPages);
    }
    
    setShowProjectManager(false);
  };
  
  // Function to handle project saving
  const handleSaveProject = (projectId: string, currentPages: Page[]) => {
    // This will be called from MangaEditor when pages change
    setPages(currentPages);
  };

  // Load characters immediately on page load, don't wait for models
  const loadCharactersImmediately = async () => {
    setIsLoadingCharacters(true);
    try {
      await fetchCharacters();
    } catch (error) {
      console.error('Error loading characters on startup:', error);
      // Don't show error for characters, just log it
    } finally {
      setIsLoadingCharacters(false);
    }
  };

  // Update checkInitializationStatus - simplified since characters load separately
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
      
      // Hide the status indicator after 2 seconds when initialization completes
      if (data.initialized && initStatus.status !== 'success') {
        setTimeout(() => {
          setShowModelStatus(false);
        }, 2000);
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
      
      const response = await fetch(`${apiEndpoint}/init`, {
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
      const response = await fetch(`${apiEndpoint}/get_characters`);
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

  // Handler for showing export dialog
  const handleShowExport = () => {
    setShowExportDialog(true);
  };

  // Handler for updating current page index (passed to MangaEditor)
  const handlePageIndexChange = (newIndex: number) => {
    setCurrentPageIndex(newIndex);
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
          {/* Show loading indicator while project is loading */}
          {isLoadingProject ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500 mx-auto mb-4"></div>
                <p className="text-gray-600">Loading project...</p>
              </div>
            </div>
          ) : (
            <>
              {/* Project manager overlay */}
              {showProjectManager && (
                <div 
                  className="absolute inset-0 z-50 bg-white bg-opacity-75 flex items-center justify-center p-4"
                  onClick={(e) => {
                    if (e.target === e.currentTarget) {
                      setShowProjectManager(false);
                    }
                  }}
                >
                  <div className="w-full max-w-4xl">
                    <ProjectManager
                      apiEndpoint={apiEndpoint}
                      onSelectProject={handleSelectProject}
                      onSaveProject={handleSaveProject}
                      onClose={() => setShowProjectManager(false)}
                      currentPages={pages}
                      activeProject={currentProject}
                    />
                  </div>
                </div>
              )}
              
              {/* Export dialog overlay */}
              <ProjectExport
                isOpen={showExportDialog}
                onClose={() => setShowExportDialog(false)}
                currentProject={currentProject}
                pages={pages}
                currentPageIndex={currentPageIndex}
                pageSize={pageSize}
                apiEndpoint={apiEndpoint}
              />
              
              <MangaEditor 
                characters={characters} 
                apiEndpoint={apiEndpoint}
                currentProject={currentProject}
                setCurrentProject={setCurrentProject}
                pages={pages}
                setPages={setPages}
                onSaveProject={handleSaveProject}
                onShowProjectManager={() => setShowProjectManager(true)}
                onShowExport={handleShowExport}
                onPageIndexChange={handlePageIndexChange}
              />
            </>
          )}
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
      
      {/* Show character loading indicator in corner if needed */}
      {isLoadingCharacters && !isLoadingProject && (
        <div className="fixed bottom-4 left-4 bg-white rounded-lg shadow-lg p-3">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-indigo-500 mr-3"></div>
            <span className="text-sm text-gray-600">Loading characters...</span>
          </div>
        </div>
      )}
    </div>
  );
}