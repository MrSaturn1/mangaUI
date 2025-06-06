// components/GenerationHistoryModal.tsx
import React, { useState, useEffect } from 'react';
import { X, Check, Download } from 'lucide-react';

interface Generation {
  timestamp: string;
  datetime: string;
  imageData: string;
  seed: number;
  prompt?: string;
  isActive: boolean;
  hasEmbedding?: boolean;
}

interface GenerationHistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  type: 'panel' | 'character';
  projectId?: string;
  panelIndex?: number;
  characterName?: string;
  apiEndpoint: string;
  onSelectionChanged?: (timestamp: string) => void;
}

const GenerationHistoryModal: React.FC<GenerationHistoryModalProps> = ({
  isOpen,
  onClose,
  type,
  projectId,
  panelIndex,
  characterName,
  apiEndpoint,
  onSelectionChanged
}) => {
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [currentGeneration, setCurrentGeneration] = useState<Generation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedGeneration, setSelectedGeneration] = useState<string | null>(null);

  // Load generation history when modal opens
  useEffect(() => {
    if (isOpen) {
      loadGenerationHistory();
    }
  }, [isOpen, type, projectId, panelIndex, characterName]);

  const loadGenerationHistory = async () => {
    setIsLoading(true);
    setError(null);

    try {
      let url = '';
      if (type === 'panel' && projectId && panelIndex !== undefined) {
        url = `${apiEndpoint}/panel_history/${projectId}/${panelIndex}`;
      } else if (type === 'character' && characterName) {
        url = `${apiEndpoint}/character_history/${encodeURIComponent(characterName)}`;
      } else {
        throw new Error('Invalid parameters for loading history');
      }

      const response = await fetch(url);
      const data = await response.json();

      if (data.status === 'success') {
        console.log('Loaded generation history:', data.history?.length, 'items');
        console.log('First item imageData preview:', data.history?.[0]?.imageData?.substring(0, 100));
        setGenerations(data.history || []);
        setCurrentGeneration(data.currentGeneration);
        
        // Set initial selection to current generation
        if (data.currentGeneration) {
          setSelectedGeneration(data.currentGeneration.timestamp);
        }
      } else {
        throw new Error(data.message || 'Failed to load generation history');
      }
    } catch (err) {
      console.error('Error loading generation history:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSetActive = async (timestamp: string, createEmbedding: boolean = false) => {
    setIsLoading(true);
    setError(null);

    try {
      let url = '';
      let requestBody: any = {};

      if (type === 'panel' && projectId && panelIndex !== undefined) {
        url = `${apiEndpoint}/set_active_panel_generation`;
        requestBody = {
          projectId,
          panelIndex,
          timestamp
        };
      } else if (type === 'character' && characterName) {
        url = `${apiEndpoint}/set_active_character_generation`;
        requestBody = {
          characterName,
          timestamp,
          createEmbedding
        };
      } else {
        throw new Error('Invalid parameters for setting active generation');
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });

      const data = await response.json();

      if (data.status === 'success') {
        // Update local state
        const updatedGenerations = generations.map(gen => ({
          ...gen,
          isActive: gen.timestamp === timestamp,
          hasEmbedding: type === 'character' && gen.timestamp === timestamp ? 
            (createEmbedding || gen.hasEmbedding) : 
            (type === 'character' ? false : gen.hasEmbedding)
        }));
        
        setGenerations(updatedGenerations);
        setCurrentGeneration(updatedGenerations.find(g => g.timestamp === timestamp) || null);
        setSelectedGeneration(timestamp);

        // Notify parent component
        if (onSelectionChanged) {
          onSelectionChanged(timestamp);
        }

        // Close modal after successful update
        setTimeout(() => {
          onClose();
        }, 1000);
      } else {
        throw new Error(data.message || 'Failed to set active generation');
      }
    } catch (err) {
      console.error('Error setting active generation:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (datetime: string) => {
    const date = new Date(datetime);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  const downloadGeneration = (generation: Generation) => {
    const link = document.createElement('a');
    link.href = generation.imageData;
    link.download = `${type}_${generation.timestamp}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        <div className="p-4 border-b flex justify-between items-center flex-shrink-0">
          <h3 className="text-xl font-bold">
            {type === 'panel' ? `Panel ${panelIndex} Generation History` : `${characterName} Generation History`}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X size={24} />
          </button>
        </div>

        {error && (
          <div className="p-4 bg-red-100 border-b border-red-200 text-red-700 flex-shrink-0">
            {error}
          </div>
        )}

        <div className="p-4 flex-1 overflow-y-auto min-h-0">
          {isLoading ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
            </div>
          ) : generations.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-gray-500">No generation history found.</p>
            </div>
          ) : (
            <>
              {/* Summary */}
              <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-gray-800">{generations.length}</div>
                    <div className="text-sm text-gray-500">Total Generations</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-green-600">
                      {generations.filter(g => g.isActive).length}
                    </div>
                    <div className="text-sm text-gray-500">Active</div>
                  </div>
                  {type === 'character' && (
                    <div>
                      <div className="text-2xl font-bold text-blue-600">
                        {generations.filter(g => g.hasEmbedding).length}
                      </div>
                      <div className="text-sm text-gray-500">With Embedding</div>
                    </div>
                  )}
                  <div>
                    <div className="text-2xl font-bold text-purple-600">
                      {new Set(generations.map(g => g.seed)).size}
                    </div>
                    <div className="text-sm text-gray-500">Unique Seeds</div>
                  </div>
                </div>
              </div>

              {/* Generation Grid */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {generations.map((generation) => (
                  <div
                    key={generation.timestamp}
                    className={`relative border rounded-lg overflow-hidden cursor-pointer transition-all hover:shadow-lg ${
                      selectedGeneration === generation.timestamp
                        ? 'border-indigo-500 ring-2 ring-indigo-200 shadow-lg'
                        : generation.isActive
                          ? 'border-green-500 shadow-md'
                          : 'border-gray-200 hover:border-gray-400'
                    }`}
                    onClick={() => setSelectedGeneration(generation.timestamp)}
                  >
                    {/* Image */}
                    <div className="w-full h-48 bg-gray-100 flex items-center justify-center overflow-hidden relative">
                      <img
                        src={generation.imageData}
                        alt={`Generation ${generation.timestamp}`}
                        className="max-w-full max-h-full object-contain relative z-10"
                        onError={(e) => {
                          console.error('Image failed to load:', generation.timestamp);
                          console.error('Image src length:', generation.imageData?.length);
                          console.error('Image src preview:', generation.imageData?.substring(0, 100));
                          e.currentTarget.style.display = 'none';
                        }}
                        onLoad={(e) => {
                          console.log('Image loaded successfully:', generation.timestamp);
                          console.log('Natural dimensions:', e.currentTarget.naturalWidth, 'x', e.currentTarget.naturalHeight);
                          console.log('Image src preview:', generation.imageData?.substring(0, 100));
                        }}
                      />
                    </div>

                    {/* Status badges - positioned over image */}
                    <div className="absolute top-2 right-2 flex gap-1">
                      {generation.isActive && (
                        <span className="bg-green-500 text-white text-xs px-2 py-1 rounded-full">
                          Active
                        </span>
                      )}
                      {type === 'character' && generation.hasEmbedding && (
                        <span className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
                          Embedded
                        </span>
                      )}
                    </div>

                    {/* Info Panel */}
                    <div className="p-3 bg-white">
                      <div className="text-xs text-gray-500 mb-1">
                        {formatDate(generation.datetime)}
                      </div>
                      <div className="text-sm font-medium text-gray-800 mb-1">
                        Seed: {generation.seed}
                      </div>
                      {generation.prompt && (
                        <div className="text-xs text-gray-600 truncate" title={generation.prompt}>
                          {generation.prompt}
                        </div>
                      )}
                      
                      {/* Action buttons */}
                      <div className="mt-2 flex gap-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            downloadGeneration(generation);
                          }}
                          className="flex-1 px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded flex items-center justify-center"
                          title="Download"
                        >
                          <Download size={12} />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Action Panel */}
              {selectedGeneration && (
                <div className="mt-6 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                  <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                    <div>
                      <h4 className="font-medium text-gray-800 mb-1">
                        Selected Generation: {formatDate(generations.find(g => g.timestamp === selectedGeneration)?.datetime || '')}
                      </h4>
                      <p className="text-sm text-gray-600">
                        {type === 'panel' 
                          ? 'Set this generation as the active panel image' 
                          : 'Set this generation as the active character image'
                        }
                      </p>
                    </div>
                    
                    <div className="flex gap-2">
                      {type === 'character' && (
                        <button
                          onClick={() => handleSetActive(selectedGeneration, true)}
                          disabled={isLoading}
                          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 flex items-center"
                        >
                          <Check size={16} className="mr-1" />
                          Set Active + Create Embedding
                        </button>
                      )}
                      
                      <button
                        onClick={() => handleSetActive(selectedGeneration, false)}
                        disabled={isLoading}
                        className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400 flex items-center"
                      >
                        <Check size={16} className="mr-1" />
                        {isLoading ? 'Setting...' : 'Set Active'}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default GenerationHistoryModal;