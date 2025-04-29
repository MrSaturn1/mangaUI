// mangaui/frontend/src/components/SceneBrowser.tsx
import React, { useState } from 'react';
import { useRouter } from 'next/navigation';

interface SceneElement {
  type: 'action' | 'dialogue';
  text?: string;
  character?: string;
  dialogue?: string[];
  dialogue_text?: string;
}

interface Scene {
  location: string;
  time: string;
  interior_exterior: string;
  estimated_panels: number;
  elements: SceneElement[];
  index: number;
}

const SceneBrowser: React.FC = () => {
  const router = useRouter();
  const [screenplay, setScreenplay] = useState<{path: string; title: string} | null>(null);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [selectedScene, setSelectedScene] = useState<Scene | null>(null);
  const [loadingScenes, setLoadingScenes] = useState<boolean>(false);
  
  // Function to load the screenplay
  const handleLoadScreenplay = async () => {
    try {
      setLoadingScenes(true);
      
      const response = await fetch('http://localhost:5000/api/parse_screenplay', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          screenplay_path: './the-rat.txt',
          character_data_path: './characters.json'
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setScreenplay({
          path: './the-rat.txt',
          title: 'The Rat'
        });
        setScenes(data.scenes);
        setSelectedScene(null);
      } else {
        console.error('Error parsing screenplay:', data.message);
        alert(`Error parsing screenplay: ${data.message}`);
      }
    } catch (error) {
      console.error('Error loading screenplay:', error);
      alert(`Error loading screenplay: ${error}`);
    } finally {
      setLoadingScenes(false);
    }
  };
  
  // Handler for selecting a scene
  const handleSelectScene = (scene: Scene) => {
    setSelectedScene(scene);
  };
  
  // Function to generate panels for a scene
  const handleGeneratePanels = (sceneIndex: number) => {
    // In a real implementation, this would redirect to the panel editor
    // with the scene data pre-loaded
    console.log(`Generating panels for scene ${sceneIndex}`);
    alert(`Generating panels for scene ${sceneIndex}`);
  };
  
  return (
    <div className="p-4">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-4">Scene Browser</h2>
        
        <div className="flex space-x-4 items-center">
          <button
            className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
            onClick={handleLoadScreenplay}
            disabled={loadingScenes}
          >
            {loadingScenes ? 'Loading...' : 'Load Screenplay'}
          </button>
          
          {screenplay && (
            <div className="text-gray-700">
              <span className="font-medium">{screenplay.title}</span>
              <span className="text-sm text-gray-500 ml-2">{screenplay.path}</span>
            </div>
          )}
        </div>
      </div>
      
      {scenes.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 bg-white rounded-lg shadow-lg p-4 h-[70vh] overflow-hidden flex flex-col">
            <h3 className="text-lg font-semibold mb-4">Scenes</h3>
            
            <div className="overflow-y-auto flex-grow">
              {scenes.map((scene, index) => (
                <div 
                  key={`scene-${index}`}
                  className={`p-3 mb-2 rounded-md cursor-pointer ${
                    selectedScene?.index === index
                      ? 'bg-indigo-100 border-l-4 border-indigo-500'
                      : 'bg-gray-50 hover:bg-gray-100'
                  }`}
                  onClick={() => handleSelectScene(scene)}
                >
                  <h4 className="font-medium text-gray-900">Scene {index + 1}</h4>
                  <p className="text-sm text-gray-600">
                    {scene.interior_exterior} {scene.location} - {scene.time}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Est. Panels: {scene.estimated_panels || 'N/A'}
                  </p>
                </div>
              ))}
            </div>
          </div>
          
          <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-4 h-[70vh] overflow-hidden flex flex-col">
            {selectedScene ? (
              <>
                <h3 className="text-lg font-semibold mb-2">Scene {selectedScene.index + 1} Details</h3>
                
                <div className="mb-4 bg-gray-50 p-3 rounded-md">
                  <h4 className="font-medium text-lg text-gray-900">
                    {selectedScene.interior_exterior} {selectedScene.location} - {selectedScene.time}
                  </h4>
                </div>
                
                <div className="overflow-y-auto flex-grow mb-4">
                  <h4 className="font-medium mb-2">Elements</h4>
                  
                  <div className="space-y-3">
                    {selectedScene.elements.map((element, index) => (
                      <div 
                        key={`element-${index}`} 
                        className={`p-3 rounded-md ${
                          element.type === 'action' 
                            ? 'bg-blue-50' 
                            : 'bg-green-50'
                        }`}
                      >
                        {element.type === 'action' ? (
                          <p className="text-sm">{element.text}</p>
                        ) : (
                          <div>
                            <h5 className="font-medium text-sm">{element.character}</h5>
                            <p className="text-sm italic">"{element.dialogue_text || element.dialogue?.join(' ')}"</p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="flex space-x-4">
                  <button 
                    className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    onClick={() => handleGeneratePanels(selectedScene.index)}
                  >
                    Generate Panels for this Scene
                  </button>
                  
                  <button
                    className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    onClick={() => router.push('/panel-editor')}
                  >
                    Edit Panels Manually
                  </button>
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                <p>Select a scene to view details</p>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto mb-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Screenplay Loaded</h3>
          <p className="text-gray-500 mb-4">Load a screenplay to view scenes and generate panels</p>
          <button
            className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            onClick={handleLoadScreenplay}
            disabled={loadingScenes}
          >
            {loadingScenes ? 'Loading...' : 'Load Screenplay'}
          </button>
        </div>
      )}
    </div>
  );
};

export default SceneBrowser;