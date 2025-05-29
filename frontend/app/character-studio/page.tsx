// app/character-studio/page.tsx
"use client";
import React, { useState, useEffect } from 'react';
import CharacterManager from '@/components/CharacterManager';
import Link from 'next/link';
import { API_ENDPOINT, API_BASE_URL } from '../../config';

interface Project {
  id: string;
  name: string;
}

const CharacterStudioPage = () => {
  // State management
  const [filterMode, setFilterMode] = useState('all'); // 'all', 'project', 'favorites', 'with-embedding', 'without-embedding'
  const [currentProject, setCurrentProject] = useState<string>('');
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [sortOrder, setSortOrder] = useState<string>('name-asc'); // 'name-asc', 'name-desc', 'date-asc', 'date-desc'

  // Fetch projects on mount
  useEffect(() => {
    fetchProjects();
  }, []);

  // Function to fetch projects from API
  const fetchProjects = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Attempt to fetch from API
      const response = await fetch(`${API_ENDPOINT}/projects`);
      
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success' && data.projects) {
          setProjects(data.projects);
        }
      } else {
        // Fallback to localStorage if API fails
        const projectsJson = localStorage.getItem('mangaProjects');
        if (projectsJson) {
          setProjects(JSON.parse(projectsJson));
        } else {
          // Sample projects if nothing is available
          setProjects([
            { id: 'project-1', name: 'My First Manga' },
            { id: 'project-2', name: 'Sci-Fi Adventure' }
          ]);
        }
      }
    } catch (err) {
      console.error('Error fetching projects:', err);
      setError('Failed to load projects. Please try again.');
      
      // Sample projects as fallback
      setProjects([
        { id: 'project-1', name: 'My First Manga' },
        { id: 'project-2', name: 'Sci-Fi Adventure' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle project selection
  const handleProjectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setCurrentProject(e.target.value);
  };

  // Handle search input
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };

  // Handle sort order change
  const handleSortChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSortOrder(e.target.value);
  };

  // Handle character batch actions
  const handleBatchAction = (action: string) => {
    switch (action) {
      case 'generate-all':
        alert('Generating all selected characters...');
        break;
      case 'delete-selected':
        alert('Deleting selected characters...');
        break;
      case 'export-selected':
        alert('Exporting selected characters...');
        break;
      default:
        break;
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Character Studio</h1>
          <p className="text-gray-600 mt-1">Create, manage, and organize your manga characters</p>
        </div>
        
        <div className="flex gap-2">
          <Link 
            href="/"
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition"
          >
            Back to Editor
          </Link>
          <button 
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
            onClick={() => document.getElementById('character-import-file')?.click()}
          >
            Import Characters
          </button>
          <input 
            type="file" 
            id="character-import-file" 
            className="hidden" 
            accept=".json"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                alert(`Importing characters from: ${e.target.files[0].name}`);
              }
            }}
          />
        </div>
      </div>
      
      {error && (
        <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
          {error}
        </div>
      )}
      
      <div className="mb-6 bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Character Filters & Tools</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Filter Mode
            </label>
            <select 
              className="mt-1 block w-full px-3 py-2 border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              value={filterMode}
              onChange={(e) => setFilterMode(e.target.value)}
            >
              <option value="all">All Characters</option>
              <option value="project">Project Characters</option>
              <option value="favorites">Favorites</option>
              <option value="with-embedding">With Embeddings</option>
              <option value="without-embedding">Without Embeddings</option>
            </select>
          </div>
          
          {filterMode === 'project' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Select Project
              </label>
              <select 
                className="mt-1 block w-full px-3 py-2 border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                value={currentProject}
                onChange={handleProjectChange}
                disabled={isLoading}
              >
                <option value="">All Projects</option>
                {projects.map(project => (
                  <option key={project.id} value={project.id}>
                    {project.name}
                  </option>
                ))}
              </select>
            </div>
          )}
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Search
            </label>
            <input
              type="text"
              placeholder="Search characters..."
              className="mt-1 block w-full px-3 py-2 border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              value={searchTerm}
              onChange={handleSearchChange}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Sort By
            </label>
            <select 
              className="mt-1 block w-full px-3 py-2 border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              value={sortOrder}
              onChange={handleSortChange}
            >
              <option value="name-asc">Name (A-Z)</option>
              <option value="name-desc">Name (Z-A)</option>
              <option value="date-desc">Newest First</option>
              <option value="date-asc">Oldest First</option>
            </select>
          </div>
        </div>
        
        <div className="mt-4 pt-4 border-t border-gray-200 flex flex-wrap gap-2">
          <button
            className="px-3 py-1 bg-indigo-600 text-white text-sm rounded hover:bg-indigo-700"
            onClick={() => handleBatchAction('generate-all')}
          >
            Generate Selected
          </button>
          <button
            className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700"
            onClick={() => handleBatchAction('export-selected')}
          >
            Export Selected
          </button>
          <button
            className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
            onClick={() => handleBatchAction('delete-selected')}
          >
            Delete Selected
          </button>
        </div>
      </div>
      
      {/* Character Statistics Summary */}
      <div className="mb-6 grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-4 flex items-center">
          <div className="rounded-full bg-indigo-100 p-3 mr-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
          <div>
            <div className="text-sm text-gray-500">Total Characters</div>
            <div className="text-xl font-bold">64</div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4 flex items-center">
          <div className="rounded-full bg-green-100 p-3 mr-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <div>
            <div className="text-sm text-gray-500">With Embeddings</div>
            <div className="text-xl font-bold">48</div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4 flex items-center">
          <div className="rounded-full bg-yellow-100 p-3 mr-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div>
            <div className="text-sm text-gray-500">Need Generation</div>
            <div className="text-xl font-bold">16</div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4 flex items-center">
          <div className="rounded-full bg-blue-100 p-3 mr-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
            </svg>
          </div>
          <div>
            <div className="text-sm text-gray-500">Favorites</div>
            <div className="text-xl font-bold">12</div>
          </div>
        </div>
      </div>
      
      {/* Character Manager with full-width display */}
      <CharacterManager 
        apiBaseUrl="http://localhost:8000"
        showUseInPanelButton={false} // Hide the "Use in Selected Panel" button
      />
    </div>
  );
};

export default CharacterStudioPage;