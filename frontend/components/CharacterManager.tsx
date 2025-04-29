// mangaui/frontend/src/components/CharacterManager.tsx
import React, { useState, useEffect } from 'react';

interface Character {
  name: string;
  descriptions: string[];
  hasEmbedding: boolean;
  imageData?: string;
}

interface CharacterManagerProps {
  // Add any props here if needed
}

const CharacterManager: React.FC<CharacterManagerProps> = () => {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(null);
  const [seed, setSeed] = useState<number>(Math.floor(Math.random() * 1000000));
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  
  // Effect to load characters on mount
  useEffect(() => {
    fetchCharacters();
  }, []);
  
  // Function to fetch characters from the server
  const fetchCharacters = async () => {
    try {
      setIsLoading(true);
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
      setIsLoading(false);
    }
  };
  
  // Handler for selecting a character
  const handleSelectCharacter = (character: Character) => {
    setSelectedCharacter(character);
  };
  
  // Handler for generating a character
  const handleGenerateCharacter = async () => {
    if (!selectedCharacter) {
      alert('Please select a character');
      return;
    }
    
    try {
      setIsGenerating(true);
      // In a real implementation, this would call the API to generate the character
      alert(`Generating character: ${selectedCharacter.name} with seed: ${seed}`);
      
      // After generation, update the characters list
      fetchCharacters();
    } catch (error) {
      console.error('Error generating character:', error);
      alert(`Error generating character: ${error}`);
    } finally {
      setIsGenerating(false);
    }
  };
  
  // Handler for saving a character to keepers
  const handleSaveToKeepers = async () => {
    if (!selectedCharacter || !selectedCharacter.hasEmbedding) {
      alert('Please select a generated character');
      return;
    }
    
    try {
      setIsLoading(true);
      // In a real implementation, this would call the API to save the character to keepers
      alert(`Saving character to keepers: ${selectedCharacter.name}`);
      
      // After saving, update the characters list
      fetchCharacters();
    } catch (error) {
      console.error('Error saving character:', error);
      alert(`Error saving character: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-4">
      <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-4">
        <h2 className="text-2xl font-bold mb-4">Character Gallery</h2>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="text-gray-500">Loading characters...</div>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {characters.map((character) => (
              <div
                key={character.name}
                className={`cursor-pointer rounded-lg overflow-hidden border-2 transition-colors ${
                  selectedCharacter?.name === character.name
                    ? 'border-indigo-500'
                    : 'border-gray-200 hover:border-gray-400'
                }`}
                onClick={() => handleSelectCharacter(character)}
              >
                {character.imageData ? (
                  <div className="aspect-square">
                    <img
                      src={character.imageData}
                      alt={character.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                ) : (
                  <div className="aspect-square bg-gray-200 flex items-center justify-center">
                    <span className="text-4xl text-gray-500">{character.name[0]}</span>
                  </div>
                )}
                <div className="p-2 bg-white">
                  <div className="font-medium truncate">{character.name}</div>
                  <div className="text-xs text-gray-500 flex items-center">
                    {character.hasEmbedding && (
                      <span className="text-green-500 mr-1">✓</span>
                    )}
                    {character.descriptions.length} descriptions
                  </div>
                </div>
              </div>
            ))}
            
            {characters.length === 0 && (
              <div className="col-span-full text-gray-500 italic text-center">
                No characters available. Check your character data.
              </div>
            )}
          </div>
        )}
      </div>
      
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h2 className="text-2xl font-bold mb-4">Character Details</h2>
        
        {selectedCharacter ? (
          <div>
            <div className="mb-4">
              <h3 className="text-xl font-semibold mb-2">{selectedCharacter.name}</h3>
              
              <div className="mb-4">
                {selectedCharacter.imageData ? (
                  <img
                    src={selectedCharacter.imageData}
                    alt={selectedCharacter.name}
                    className="w-full max-h-64 object-contain mb-2"
                  />
                ) : (
                  <div className="w-full h-48 bg-gray-200 flex items-center justify-center mb-2">
                    <span className="text-6xl text-gray-500">{selectedCharacter.name[0]}</span>
                  </div>
                )}
                
                <div className="text-sm text-gray-500">
                  {selectedCharacter.hasEmbedding ? (
                    <span className="text-green-500">Character has embedding ✓</span>
                  ) : (
                    <span className="text-yellow-500">Character not generated yet</span>
                  )}
                </div>
              </div>
              
              <div className="mb-4">
                <h4 className="font-medium mb-1">Descriptions</h4>
                {selectedCharacter.descriptions.length > 0 ? (
                  <ul className="list-disc pl-5 text-sm">
                    {selectedCharacter.descriptions.map((desc, index) => (
                      <li key={index}>{desc}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-500 italic">No descriptions available</p>
                )}
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Seed</label>
                <div className="flex space-x-2">
                  <input
                    type="number"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                  />
                  <button
                    className="px-3 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    onClick={() => setSeed(Math.floor(Math.random() * 1000000))}
                  >
                    Random
                  </button>
                </div>
              </div>
              
              <div className="flex space-x-4">
                <button
                  className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
                  onClick={handleGenerateCharacter}
                  disabled={isGenerating}
                >
                  {isGenerating
                    ? 'Generating...'
                    : selectedCharacter.hasEmbedding
                      ? 'Regenerate'
                      : 'Generate'
                  }
                </button>
                
                <button
                  className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400"
                  onClick={handleSaveToKeepers}
                  disabled={!selectedCharacter.hasEmbedding || isLoading}
                >
                  Save to Keepers
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            <p>Select a character to view details</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default CharacterManager;