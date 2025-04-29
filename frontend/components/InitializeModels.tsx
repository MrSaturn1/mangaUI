// mangaui/frontend/src/components/InitializeModels.tsx
import React, { useState } from 'react';

interface InitializeModelsProps {
  onInitialized: () => void;
}

const InitializeModels: React.FC<InitializeModelsProps> = ({ onInitialized }) => {
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [modelPath, setModelPath] = useState<string>('./drawatoon-v1');
  const [characterDataPath, setCharacterDataPath] = useState<string>('./characters.json');
  const [characterEmbeddingPath, setCharacterEmbeddingPath] = useState<string>('./character_output/character_embeddings/character_embeddings_map.json');
  const [outputDir, setOutputDir] = useState<string>('./manga_output');
  
  const handleInitialize = async () => {
    try {
      setIsInitializing(true);
      
      const response = await fetch('http://localhost:5000/api/init', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model_path: modelPath,
          character_data_path: characterDataPath,
          character_embedding_path: characterEmbeddingPath,
          output_dir: outputDir
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        onInitialized();
      } else {
        console.error('Error initializing models:', data.message);
        alert(`Error initializing models: ${data.message}`);
      }
    } catch (error) {
      console.error('Error initializing models:', error);
      alert(`Error initializing models: ${error}`);
    } finally {
      setIsInitializing(false);
    }
  };
  
  return (
    <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-6 text-center">Initialize Manga Creator</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Model Path</label>
          <input
            type="text"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Character Data Path</label>
          <input
            type="text"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            value={characterDataPath}
            onChange={(e) => setCharacterDataPath(e.target.value)}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Character Embedding Path</label>
          <input
            type="text"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            value={characterEmbeddingPath}
            onChange={(e) => setCharacterEmbeddingPath(e.target.value)}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Output Directory</label>
          <input
            type="text"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
          />
        </div>
        
        <button
          className="w-full px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
          onClick={handleInitialize}
          disabled={isInitializing}
        >
          {isInitializing ? 'Initializing...' : 'Initialize Models'}
        </button>
      </div>
      
      <div className="mt-6 text-sm text-gray-500">
        <p>This will initialize the Drawatoon model and load character data. It may take a few moments.</p>
      </div>
    </div>
  );
};

export default InitializeModels;