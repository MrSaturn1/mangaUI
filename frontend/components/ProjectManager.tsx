import React, { useState, useEffect } from 'react';
import { Page, Project } from './MangaEditor';

interface ApiResponse {
  status: string;
  message?: string;
  projects?: Project[];
  project?: Project;
  pages?: Page[];
}

interface ProjectManagerProps {
  apiEndpoint: string;
  onSelectProject?: (project: Project, pages: Page[]) => void;
  onSaveProject?: (projectId: string, pages: Page[]) => void;
  onClose?: () => void;
  currentPages?: Page[];
  activeProject?: Project | null;
}

const ProjectManager: React.FC<ProjectManagerProps> = ({ 
  apiEndpoint,
  onSelectProject,
  onSaveProject,
  onClose,
  currentPages = [],
  activeProject
}) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [showNewProjectModal, setShowNewProjectModal] = useState<boolean>(false);
  const [newProjectName, setNewProjectName] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [useLocalStorage, setUseLocalStorage] = useState<boolean>(true);

  // Load projects on component mount
  useEffect(() => {
    fetchProjects();
  }, []);

  // Sync currentProject with activeProject from parent
  useEffect(() => {
    if (activeProject) {
      setCurrentProject(activeProject);
    }
  }, [activeProject]);

  // Fetch all projects from the API
  const fetchProjects = async (): Promise<void> => {
    setIsLoading(true);
    setError(null);
    
    // Try to fetch from API first
    try {
      const response = await fetch(`${apiEndpoint}/projects`);
      
      if (response.ok) {
        const data: ApiResponse = await response.json();
        
        if (data.status === 'success' && data.projects) {
          setProjects(data.projects);
          
          // Set the first project as current if there's no current project
          if (data.projects.length > 0 && !currentProject) {
            setCurrentProject(data.projects[0]);
          }
          
          // Also store in localStorage as backup
          localStorage.setItem('mangaProjects', JSON.stringify(data.projects));
          
          setIsLoading(false);
          return;
        }
      }
    } catch (err) {
      console.warn('Error fetching from API, falling back to localStorage:', err);
    }
    
    // If API fails, try localStorage
    try {
      const projectsJson = localStorage.getItem('mangaProjects') || '[]';
      const loadedProjects: Project[] = JSON.parse(projectsJson);
      setProjects(loadedProjects);
      
      if (loadedProjects.length > 0 && !currentProject) {
        setCurrentProject(loadedProjects[0]);
      }
    } catch (err) {
      console.error('Error loading projects from localStorage:', err);
      setError(err instanceof Error ? err.message : String(err));
      
      // Last fallback - dummy data
      const dummyProjects: Project[] = [
        { id: 'project-1', name: 'My First Manga', pages: 5, lastModified: new Date().toISOString() },
        { id: 'project-2', name: 'Sci-Fi Adventure', pages: 12, lastModified: new Date().toISOString() }
      ];
      setProjects(dummyProjects);
    } finally {
      setIsLoading(false);
    }
  };

  // Create a new project
  const handleCreateProject = async (): Promise<void> => {
    if (!newProjectName.trim()) {
      setError('Project name cannot be empty');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    // Create project in backend
    try {
      const response = await fetch(`${apiEndpoint}/projects`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: newProjectName
        })
      });
      
      if (response.ok) {
        const data: ApiResponse = await response.json();
        
        if (data.status === 'success' && data.project) {
          handleProjectCreated(data.project);
          return;
        }
      }
      
      // If API fails, create in localStorage
      throw new Error('API request failed');
    } catch (err) {
      console.warn('Error creating project in API, using localStorage instead:', err);
      
      // Create project in localStorage
      const newProject: Project = { 
        id: `project-${Date.now()}`, 
        name: newProjectName, 
        pages: 0, 
        lastModified: new Date().toISOString() 
      };
      
      handleProjectCreated(newProject);
    }
  };

  // Common handling after creating a project
  const handleProjectCreated = (newProject: Project): void => {
    // Update projects list
    const updatedProjects = [...projects, newProject];
    setProjects(updatedProjects);
    setCurrentProject(newProject);
    
    // Save to localStorage
    localStorage.setItem('mangaProjects', JSON.stringify(updatedProjects));
    
    // Clear form
    setNewProjectName('');
    setShowNewProjectModal(false);
    setIsLoading(false);
    
    // Trigger callback
    if (onSelectProject) {
      onSelectProject(newProject, []);
    }
  };

  // Delete a project
  const handleDeleteProject = async (projectId: string): Promise<void> => {
    // TypeScript-safe confirm dialog
    if (!window.confirm('Are you sure you want to delete this project? This action cannot be undone.')) {
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    // Try to delete from API
    try {
      const response = await fetch(`${apiEndpoint}/projects/${projectId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        throw new Error(`Failed to delete project: ${response.statusText}`);
      }
    } catch (err) {
      console.warn('Error deleting project from API:', err);
      // Continue with local deletion anyway
    }
    
    // Update local state
    const updatedProjects = projects.filter(p => p.id !== projectId);
    setProjects(updatedProjects);
    
    // Update localStorage
    localStorage.setItem('mangaProjects', JSON.stringify(updatedProjects));
    
    // Delete project data from localStorage
    localStorage.removeItem(`project_${projectId}`);
    
    // Update current project if needed
    if (currentProject && currentProject.id === projectId) {
      const newCurrentProject = updatedProjects.length > 0 ? updatedProjects[0] : null;
      setCurrentProject(newCurrentProject);
      if (onSelectProject && newCurrentProject) {
        const pages = loadProjectPages(newCurrentProject.id);
        onSelectProject(newCurrentProject, pages);
      }
    }
    
    setIsLoading(false);
  };

  // Format date for display
  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  // Load project pages
  const loadProjectPages = (projectId: string): Page[] => {
    try {
      // Check localStorage first
      const pagesJson = localStorage.getItem(`project_${projectId}`);
      if (pagesJson) {
        return JSON.parse(pagesJson);
      }
      
      // Return empty array if not found
      return [];
    } catch (err) {
      console.error('Error loading project pages:', err);
      return [];
    }
  };
  
  // Handle opening a project
  const handleOpenProject = async (project: Project): Promise<void> => {
    setIsLoading(true);
    setCurrentProject(project);
    
    // Try to load from API first
    try {
      const response = await fetch(`${apiEndpoint}/projects/${project.id}`);
      
      if (response.ok) {
        const data: ApiResponse = await response.json();
        
        if (data.status === 'success' && data.pages) {
          // Also save to localStorage as backup
          localStorage.setItem(`project_${project.id}`, JSON.stringify(data.pages));
          
          if (onSelectProject) {
            onSelectProject(project, data.pages);
          }
          
          setIsLoading(false);
          return;
        }
      }
    } catch (err) {
      console.warn('Error loading project from API, falling back to localStorage:', err);
    }
    
    // Fall back to localStorage
    const pages = loadProjectPages(project.id);
    
    if (onSelectProject) {
      onSelectProject(project, pages);
    }
    
    setIsLoading(false);
  };
  
  // Save project with all pages
  const handleSaveProject = async (): Promise<void> => {
    if (!currentProject || !currentPages || currentPages.length === 0) {
      return;
    }
    
    setIsSaving(true);
    
    // Save to localStorage first (immediate)
    localStorage.setItem(`project_${currentProject.id}`, JSON.stringify(currentPages));
    
    // Update project metadata
    const updatedProject = {
      ...currentProject,
      pages: currentPages.length,
      lastModified: new Date().toISOString()
    };
    
    // Update projects list
    const updatedProjects = projects.map(p => 
      p.id === currentProject.id ? updatedProject : p
    );
    
    setProjects(updatedProjects);
    setCurrentProject(updatedProject);
    
    // Save projects metadata to localStorage
    localStorage.setItem('mangaProjects', JSON.stringify(updatedProjects));
    
    // Try to save to backend
    try {
      const response = await fetch(`${apiEndpoint}/projects/${currentProject.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          pages: currentPages
        })
      });
      
      if (!response.ok) {
        console.warn('Project saved to localStorage but not to backend.');
      }
    } catch (err) {
      console.warn('Error saving project to backend:', err);
      // Already saved to localStorage, so no further action needed
    } finally {
      setIsSaving(false);
    }
  };

  // UI remains the same but with proper typing
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Manga Projects</h2>
        <div className="flex items-center gap-3">
          <button
            className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:ring-2 focus:ring-indigo-500"
            onClick={() => setShowNewProjectModal(true)}
          >
            Create New Project
          </button>
          {onClose && (
            <button
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md"
              onClick={onClose}
              title="Close Project Manager"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}
      
      {isLoading ? (
        <div className="flex justify-center items-center h-40">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
        </div>
      ) : (
        <>
          {projects.length === 0 ? (
            <div className="text-center py-8">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              <p className="text-gray-600 mb-4">No projects found. Create your first manga project!</p>
              <button
                className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                onClick={() => setShowNewProjectModal(true)}
              >
                Create New Project
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Project Name
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Pages
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Last Modified
                    </th>
                    <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {projects.map((project) => (
                    <tr 
                      key={project.id} 
                      className={`hover:bg-gray-50 ${currentProject && currentProject.id === project.id ? 'bg-blue-50' : ''}`}
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="flex-shrink-0 h-10 w-10 bg-gray-200 rounded-md flex items-center justify-center">
                            <span className="text-gray-500 font-medium">{project.name.charAt(0)}</span>
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900">
                              {project.name}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{project.pages}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-500">{formatDate(project.lastModified)}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button
                          onClick={() => handleOpenProject(project)}
                          className="text-indigo-600 hover:text-indigo-900 mr-4"
                        >
                          Open
                        </button>
                        <button
                          onClick={() => handleDeleteProject(project.id)}
                          className="text-red-600 hover:text-red-900"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          
          {/* Current Project Summary */}
          {currentProject && (
            <div className="mt-8 p-4 border rounded-lg bg-gray-50">
              <h3 className="text-lg font-medium text-gray-800 mb-2">
                Current Project: {currentProject.name}
              </h3>
              <div className="flex space-x-6">
                <div>
                  <span className="text-sm text-gray-500 block">Pages</span>
                  <span className="text-2xl font-bold text-gray-800">{currentProject.pages}</span>
                </div>
                <div>
                  <span className="text-sm text-gray-500 block">Last Modified</span>
                  <span className="text-md text-gray-800">{formatDate(currentProject.lastModified)}</span>
                </div>
              </div>
              <div className="mt-4 flex space-x-2">
                <button
                  className="px-3 py-1 bg-indigo-600 text-white text-sm rounded hover:bg-indigo-700"
                  onClick={handleSaveProject}
                  disabled={isSaving}
                >
                  {isSaving ? 'Saving...' : 'Save Project'}
                </button>
                <button
                  className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700"
                >
                  Export Project
                </button>
                <button
                  className="px-3 py-1 bg-purple-600 text-white text-sm rounded hover:bg-purple-700"
                >
                  Preview Manga
                </button>
              </div>
            </div>
          )}
        </>
      )}
      
      {/* New Project Modal */}
      {showNewProjectModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
            <div className="p-4 border-b flex justify-between items-center">
              <h3 className="text-lg font-medium text-gray-800">Create New Project</h3>
              <button
                onClick={() => setShowNewProjectModal(false)}
                className="p-1 rounded-full hover:bg-gray-100"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4">
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Project Name</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="My Awesome Manga"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                />
              </div>
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setShowNewProjectModal(false)}
                  className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateProject}
                  className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  disabled={isLoading || !newProjectName.trim()}
                >
                  {isLoading ? 'Creating...' : 'Create Project'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProjectManager;