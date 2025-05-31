// mangaui/frontend/src/components/CharacterManager.tsx - COMPLETE VERSION
import React, { useState, useEffect } from 'react';
import GenerationHistoryModal from './GenerationHistoryModal';

interface Character {
  name: string;
  descriptions: string[];
  hasEmbedding: boolean;
  hasHistory?: boolean;  // NEW
  imageData?: string;
}

interface CharacterManagerProps {
  apiBaseUrl: string;
  onSelectCharacter?: (character: Character) => void;
  onCharacterUpdated?: (characters: Character[]) => void;
  initialSelectedCharacter?: Character | null;
  showUseInPanelButton?: boolean;
}

const CharacterManager: React.FC<CharacterManagerProps> = ({ 
  apiBaseUrl, 
  onSelectCharacter,
  onCharacterUpdated,
  initialSelectedCharacter = null,
  showUseInPanelButton = true,
}) => {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(initialSelectedCharacter);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [seed, setSeed] = useState<number>(Math.floor(Math.random() * 1000000));
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [showAddCharacterModal, setShowAddCharacterModal] = useState<boolean>(false);
  const [newCharacterName, setNewCharacterName] = useState<string>('');
  const [newCharacterDescription, setNewCharacterDescription] = useState<string>('');
  
  // NEW: Generation history state
  const [showGenerationHistory, setShowGenerationHistory] = useState<boolean>(false);
  
  // Fetch characters on component mount
  useEffect(() => {
    fetchCharacters();
  }, []);
  
  // Function to fetch characters using the new endpoint with history
  const fetchCharacters = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Try the new endpoint first
      let response = await fetch(`${apiBaseUrl}/api/get_characters_with_history`);
      
      if (!response.ok) {
        // Fallback to original endpoint
        response = await fetch(`${apiBaseUrl}/api/get_characters`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setCharacters(data.characters || []);
      } else {
        throw new Error(data.message || 'Failed to fetch characters');
      }
    } catch (err) {
      console.error('Error fetching characters:', err);
      setError('Failed to load characters. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Filter characters based on search term
  const filteredCharacters = characters.filter(character => 
    character.name.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  // Handle character selection
  const handleSelectCharacter = (character: Character) => {
    setSelectedCharacter(character);
    if (onSelectCharacter) {
      onSelectCharacter(character);
    }
  };
  
  // Generate character with the new history-aware endpoint
  const handleGenerateCharacter = async () => {
    if (!selectedCharacter) return;
    
    setIsGenerating(true);
    setError(null);
    
    try {
      // Use the new generation endpoint that supports history
      const response = await fetch(`${apiBaseUrl}/api/generate_character_with_history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: selectedCharacter.name,
          seed: seed,
          regenerate: selectedCharacter.hasEmbedding || selectedCharacter.hasHistory
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        // Update the character in the list with new image data
        const updatedCharacters = characters.map(char => 
          char.name === selectedCharacter.name 
            ? { 
                ...char, 
                imageData: data.imageData, 
                hasEmbedding: true,
                hasHistory: true  // Now has generation history
              } 
            : char
        );
        
        setCharacters(updatedCharacters);
        setSelectedCharacter({ 
          ...selectedCharacter, 
          imageData: data.imageData, 
          hasEmbedding: true,
          hasHistory: true
        });
        
        // Notify parent component if needed
        if (onCharacterUpdated) {
          onCharacterUpdated(updatedCharacters);
        }
      } else {
        throw new Error(data.message || 'Failed to generate character');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      console.error('Error generating character:', err);
      setError(`Failed to generate character: ${errorMessage}`);
    } finally {
      setIsGenerating(false);
    }
  };
  
  // Save character to keepers
  const handleSaveToKeepers = async () => {
    if (!selectedCharacter || !selectedCharacter.hasEmbedding) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${apiBaseUrl}/api/save_to_keepers`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: selectedCharacter.name
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        alert(`Character ${selectedCharacter.name} saved to keepers!`);
      } else {
        throw new Error(data.message || 'Failed to save character to keepers');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      console.error('Error saving character to keepers:', err);
      setError(`Failed to save character: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Add a new character (keeping existing implementation)
  const handleAddCharacter = async () => {
    if (!newCharacterName.trim()) {
      setError('Character name cannot be empty');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${apiBaseUrl}/api/add_character`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: newCharacterName,
          descriptions: newCharacterDescription.split('\n').filter(d => d.trim())
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        const newCharacter = {
          name: newCharacterName,
          descriptions: newCharacterDescription.split('\n').filter(d => d.trim()),
          hasEmbedding: false,
          hasHistory: false
        };
        
        setCharacters([...characters, newCharacter]);
        setSelectedCharacter(newCharacter);
        setNewCharacterName('');
        setNewCharacterDescription('');
        setShowAddCharacterModal(false);
        
        if (onCharacterUpdated) {
          onCharacterUpdated([...characters, newCharacter]);
        }
      } else {
        throw new Error(data.message || 'Failed to add character');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      console.error('Error adding character:', err);
      setError(`Failed to add character: ${errorMessage}`);
      
      // Fallback implementation
      const newCharacter = {
        name: newCharacterName,
        descriptions: newCharacterDescription.split('\n').filter(d => d.trim()),
        hasEmbedding: false,
        hasHistory: false
      };
      
      setCharacters([...characters, newCharacter]);
      setSelectedCharacter(newCharacter);
      setNewCharacterName('');
      setNewCharacterDescription('');
      setShowAddCharacterModal(false);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Render history button only if character has history
  const renderHistoryButton = () => {
    if (!selectedCharacter || !selectedCharacter.hasHistory) {
      return null;
    }
    
    return (
      <button
        className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500"
        onClick={() => setShowGenerationHistory(true)}
      >
        View History
      </button>
    );
  };
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-4">
      {/* Character Gallery - Takes up 2/3 of the space */}
      <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-800">Character Gallery</h2>
          <button
            className="px-3 py-1 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 text-sm"
            onClick={() => setShowAddCharacterModal(true)}
          >
            Add New Character
          </button>
        </div>
        
        <div className="mb-4">
          <input
            type="text"
            placeholder="Search characters..."
            className="w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        {error && (
          <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}
        
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4" style={{ maxHeight: 'calc(100vh - 300px)', overflowY: 'auto' }}>
            {filteredCharacters.map((character) => (
              <div
                key={character.name}
                className={`cursor-pointer rounded-lg overflow-hidden border-2 transition-colors ${
                  selectedCharacter?.name === character.name
                    ? 'border-indigo-500'
                    : 'border-gray-200 hover:border-gray-400'
                }`}
                onClick={() => handleSelectCharacter(character)}
              >
                <div className="aspect-square relative">
                  {character.imageData ? (
                    <img
                      src={character.imageData}
                      alt={character.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="bg-gray-200 w-full h-full flex items-center justify-center">
                      <span className="text-4xl text-gray-500">{character.name[0]}</span>
                    </div>
                  )}
                  
                  {/* NEW: History indicator */}
                  {character.hasHistory && (
                    <div className="absolute top-2 right-2">
                      <div className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                    </div>
                  )}
                </div>
                <div className="p-2 bg-white">
                  <div className="font-medium truncate text-gray-800">{character.name}</div>
                  <div className="text-xs text-gray-500 flex items-center">
                    {character.hasEmbedding && (
                      <span className="text-green-500 mr-1">✓</span>
                    )}
                    {character.descriptions?.length || 0} descriptions
                    {character.hasHistory && (
                      <span className="ml-1 text-purple-500" title="Has generation history">
                        📚
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {filteredCharacters.length === 0 && (
              <div className="col-span-full text-gray-500 italic text-center py-12">
                {searchTerm 
                  ? `No characters matching "${searchTerm}"` 
                  : "No characters available. Add your first character!"}
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Character Details - Takes up 1/3 of the space */}
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Character Details</h2>
        
        {selectedCharacter ? (
          <div>
            <div className="mb-4">
              <h3 className="text-xl font-semibold mb-2 text-gray-800">{selectedCharacter.name}</h3>
              
              <div className="mb-4">
                {selectedCharacter.imageData ? (
                  <img
                    src={selectedCharacter.imageData}
                    alt={selectedCharacter.name}
                    className="w-full max-h-64 object-contain mb-2 rounded"
                  />
                ) : (
                  <div className="w-full h-48 bg-gray-200 flex items-center justify-center mb-2 rounded">
                    <span className="text-6xl text-gray-500">{selectedCharacter.name[0]}</span>
                  </div>
                )}
                
                <div className="text-sm text-gray-600 flex items-center justify-between">
                  <div>
                    {selectedCharacter.hasEmbedding ? (
                      <span className="text-green-500 font-medium">Character has embedding ✓</span>
                    ) : (
                      <span className="text-yellow-500 font-medium">Character not generated yet</span>
                    )}
                  </div>
                  {selectedCharacter.hasHistory && (
                    <span className="text-purple-500 text-xs bg-purple-100 px-2 py-1 rounded">
                      Has History
                    </span>
                  )}
                </div>
              </div>
              
              <div className="mb-4">
                <h4 className="font-medium mb-1 text-gray-800">Descriptions</h4>
                {selectedCharacter.descriptions && selectedCharacter.descriptions.length > 0 ? (
                  <ul className="list-disc pl-5 text-sm text-gray-600">
                    {selectedCharacter.descriptions.map((desc, index) => (
                      <li key={index} className="mb-1">{desc}</li>
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
              
              {/* Updated button layout with history button */}
              <div className="space-y-2">
                <div className="flex space-x-2">
                  <button
                    className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
                    onClick={handleGenerateCharacter}
                    disabled={isGenerating}
                  >
                    {isGenerating
                      ? 'Generating...'
                      : selectedCharacter.hasEmbedding || selectedCharacter.hasHistory
                        ? 'Generate New Version'
                        : 'Generate'}
                  </button>
                  
                  <button
                    className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400"
                    onClick={handleSaveToKeepers}
                    disabled={!selectedCharacter.hasEmbedding || isLoading}
                  >
                    Save to Keepers
                  </button>
                </div>
                
                {/* History button row */}
                <div className="flex space-x-2">
                  {renderHistoryButton()}
                  
                  {/* Use in Panel button */}
                  {selectedCharacter && showUseInPanelButton && (
                    <button
                      className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500"
                      onClick={() => {
                        if (onSelectCharacter) {
                          onSelectCharacter(selectedCharacter);
                        }
                      }}
                    >
                      Use in Selected Panel
                    </button>
                  )}
                </div>
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
      
      {/* Generation History Modal */}
      {showGenerationHistory && selectedCharacter && (
        <GenerationHistoryModal
          isOpen={showGenerationHistory}
          onClose={() => setShowGenerationHistory(false)}
          type="character"
          characterName={selectedCharacter.name}
          apiEndpoint={apiBaseUrl}
          onSelectionChanged={async (timestamp) => {
            // Close the modal
            setShowGenerationHistory(false);
          }}
        />
      )}
      
      {/* Add Character Modal - unchanged */}
      {showAddCharacterModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-lg">
            <div className="p-4 border-b flex justify-between items-center">
              <h3 className="text-lg font-medium text-gray-800">Add New Character</h3>
              <button
                onClick={() => setShowAddCharacterModal(false)}
                className="p-1 rounded-full hover:bg-gray-100"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4">
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Character Name</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="e.g., Samurai Jack"
                  value={newCharacterName}
                  onChange={(e) => setNewCharacterName(e.target.value)}
                />
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Descriptions (one per line)
                </label>
                <textarea
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="e.g., tall with broad shoulders&#10;wearing samurai armor&#10;has long black hair tied in a bun"
                  rows={6}
                  value={newCharacterDescription}
                  onChange={(e) => setNewCharacterDescription(e.target.value)}
                />
                <p className="mt-1 text-sm text-gray-500">
                  Add descriptive traits that define the character's appearance. Each line will be treated as a separate descriptor.
                </p>
              </div>
              
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setShowAddCharacterModal(false)}
                  className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  Cancel
                </button>
                <button
                  onClick={handleAddCharacter}
                  className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  disabled={isLoading || !newCharacterName.trim()}
                >
                  {isLoading ? 'Adding...' : 'Add Character'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CharacterManager;