// mangaui/frontend/src/components/PanelEditor.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Stage, Layer, Rect, Text, Image as KonvaImage } from 'react-konva';
import { KonvaEventObject } from 'konva/lib/Node';

interface Character {
  name: string;
  descriptions: string[];
  hasEmbedding: boolean;
  imageData?: string;
}

interface DialogueItem {
  character: string;
  text: string;
}

interface ActionItem {
  text: string;
}

interface Position {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface PanelData {
  setting: string;
  prompt: string;
  panelIndex: number;
  seed: number;
  dialogues: DialogueItem[];
  actions: ActionItem[];
  characterNames: string[];
  characterPositions: Position[];
  dialoguePositions: Position[];
}

interface PanelEditorProps {
  characters: Character[];
}

const PanelEditor: React.FC<PanelEditorProps> = ({ characters }) => {
  const [panelData, setPanelData] = useState<PanelData>({
    setting: '',
    prompt: '',
    panelIndex: 0,
    seed: Math.floor(Math.random() * 1000000),
    dialogues: [],
    actions: [],
    characterNames: [],
    characterPositions: [],
    dialoguePositions: []
  });
  
  const [canvasSize, setCanvasSize] = useState({ width: 512, height: 512 });
  const [selectedCharacters, setSelectedCharacters] = useState<Character[]>([]);
  const [selectedItem, setSelectedItem] = useState<{ type: string; index: number } | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  
  const stageRef = useRef<any>(null);
  
  // Effect to update dialogue positions when dialogues change
  useEffect(() => {
    // Create default positions for dialogues
    const newPositions: Position[] = panelData.dialogues.map((_, index) => {
      if (index < panelData.dialoguePositions.length) {
        return panelData.dialoguePositions[index];
      }
      
      // Default positions based on index
      if (index === 0) {
        return { x: 0.6, y: 0.1, width: 0.3, height: 0.2 }; // Top right
      } else if (index === 1) {
        return { x: 0.1, y: 0.4, width: 0.3, height: 0.2 }; // Middle left
      } else {
        return { x: 0.6, y: 0.7, width: 0.3, height: 0.2 }; // Bottom right
      }
    });
    
    setPanelData(prev => ({
      ...prev,
      dialoguePositions: newPositions
    }));
  }, [panelData.dialogues.length]);
  
  // Effect to update character positions when selected characters change
  useEffect(() => {
    // Create default positions for characters
    const newPositions: Position[] = selectedCharacters.map((_, index) => {
      const totalChars = selectedCharacters.length;
      
      if (totalChars === 1) {
        // Single character in center
        return { x: 0.2, y: 0.2, width: 0.6, height: 0.6 };
      } else if (totalChars === 2) {
        // Two characters side by side
        return index === 0
          ? { x: 0.1, y: 0.2, width: 0.35, height: 0.6 }  // Left
          : { x: 0.55, y: 0.2, width: 0.35, height: 0.6 }; // Right
      } else {
        // Multiple characters - spread evenly
        const section = 1.0 / totalChars;
        return { 
          x: section * index, 
          y: 0.2, 
          width: section, 
          height: 0.6 
        };
      }
    });
    
    setPanelData(prev => ({
      ...prev,
      characterNames: selectedCharacters.map(c => c.name),
      characterPositions: newPositions
    }));
  }, [selectedCharacters]);
  
  // Handler for adding a dialogue
  const handleAddDialogue = () => {
    setPanelData(prev => ({
      ...prev,
      dialogues: [...prev.dialogues, { character: '', text: '' }]
    }));
  };
  
  // Handler for updating a dialogue
  const handleUpdateDialogue = (index: number, field: keyof DialogueItem, value: string) => {
    const newDialogues = [...panelData.dialogues];
    newDialogues[index] = { ...newDialogues[index], [field]: value };
    
    setPanelData(prev => ({
      ...prev,
      dialogues: newDialogues
    }));
  };
  
  // Handler for removing a dialogue
  const handleRemoveDialogue = (index: number) => {
    const newDialogues = [...panelData.dialogues];
    newDialogues.splice(index, 1);
    
    const newPositions = [...panelData.dialoguePositions];
    newPositions.splice(index, 1);
    
    setPanelData(prev => ({
      ...prev,
      dialogues: newDialogues,
      dialoguePositions: newPositions
    }));
  };
  
  // Handler for adding an action
  const handleAddAction = () => {
    setPanelData(prev => ({
      ...prev,
      actions: [...prev.actions, { text: '' }]
    }));
  };
  
  // Handler for updating an action
  const handleUpdateAction = (index: number, value: string) => {
    const newActions = [...panelData.actions];
    newActions[index] = { text: value };
    
    setPanelData(prev => ({
      ...prev,
      actions: newActions
    }));
  };
  
  // Handler for removing an action
  const handleRemoveAction = (index: number) => {
    const newActions = [...panelData.actions];
    newActions.splice(index, 1);
    
    setPanelData(prev => ({
      ...prev,
      actions: newActions
    }));
  };
  
  // Handler for selecting a character
  const handleSelectCharacter = (character: Character) => {
    if (!selectedCharacters.some(c => c.name === character.name)) {
      setSelectedCharacters([...selectedCharacters, character]);
    }
  };
  
  // Handler for removing a character
  const handleRemoveCharacter = (index: number) => {
    const newChars = [...selectedCharacters];
    newChars.splice(index, 1);
    
    const newPositions = [...panelData.characterPositions];
    newPositions.splice(index, 1);
    
    setSelectedCharacters(newChars);
    
    setPanelData(prev => ({
      ...prev,
      characterNames: newChars.map(c => c.name),
      characterPositions: newPositions
    }));
  };
  
  // Handler for dragging dialogue or character boxes
  const handleDragEnd = (index: number, e: KonvaEventObject<DragEvent>, type: 'dialogue' | 'character') => {
    if (!stageRef.current) return;
    
    const stage = stageRef.current;
    const { x, y } = e.target.position();
    
    // Convert from pixels to normalized coordinates
    const normalizedX = x / stage.width();
    const normalizedY = y / stage.height();
    
    if (type === 'dialogue') {
      const newPositions = [...panelData.dialoguePositions];
      newPositions[index] = {
        ...newPositions[index],
        x: normalizedX,
        y: normalizedY
      };
      setPanelData(prev => ({
        ...prev,
        dialoguePositions: newPositions
      }));
    } else if (type === 'character') {
      const newPositions = [...panelData.characterPositions];
      newPositions[index] = {
        ...newPositions[index],
        x: normalizedX,
        y: normalizedY
      };
      setPanelData(prev => ({
        ...prev,
        characterPositions: newPositions
      }));
    }
  };
  
  // Handler for generating the panel
  const handleGeneratePanel = async () => {
    try {
      setIsGenerating(true);
      
      // Prepare the data for the API
      const apiData = {
        prompt: panelData.prompt,
        setting: panelData.setting,
        dialoguePositions: panelData.dialoguePositions,
        characterPositions: panelData.characterPositions,
        characterNames: panelData.characterNames,
        dialogues: panelData.dialogues,
        actions: panelData.actions,
        panelIndex: panelData.panelIndex,
        seed: panelData.seed
      };
      
      // Call the API
      const response = await fetch('http://localhost:5000/api/generate_panel', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(apiData)
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setGeneratedImage(data.imageData);
        if (data.prompt) {
          setPanelData(prev => ({
            ...prev,
            prompt: data.prompt
          }));
        }
      } else {
        console.error('Error generating panel:', data.message);
        alert(`Error generating panel: ${data.message}`);
      }
    } catch (error) {
      console.error('Error calling API:', error);
      alert(`Error calling API: ${error}`);
    } finally {
      setIsGenerating(false);
    }
  };
  
  // Handler for saving the panel
  const handleSavePanel = () => {
    if (!generatedImage) {
      alert('Please generate a panel first');
      return;
    }
    
    // Create a download link
    const link = document.createElement('a');
    link.href = generatedImage;
    link.download = `panel_${panelData.panelIndex.toString().padStart(4, '0')}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-4">
      <div className="bg-white rounded-lg shadow-lg p-4 order-2 lg:order-1">
        <h2 className="text-2xl font-bold mb-4">Panel Settings</h2>
        
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Panel Index</label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                value={panelData.panelIndex}
                onChange={(e) => setPanelData(prev => ({ ...prev, panelIndex: parseInt(e.target.value) || 0 }))}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Seed</label>
              <div className="flex space-x-2">
                <input
                  type="number"
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  value={panelData.seed}
                  onChange={(e) => setPanelData(prev => ({ ...prev, seed: parseInt(e.target.value) || 0 }))}
                />
                <button
                  className="px-3 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  onClick={() => setPanelData(prev => ({ ...prev, seed: Math.floor(Math.random() * 1000000) }))}
                >
                  Random
                </button>
              </div>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Setting</label>
            <input
              type="text"
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              value={panelData.setting}
              onChange={(e) => setPanelData(prev => ({ ...prev, setting: e.target.value }))}
              placeholder="e.g., INT. HOTEL LOBBY - DAY, manga panel"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Custom Prompt (optional)</label>
            <textarea
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              value={panelData.prompt}
              onChange={(e) => setPanelData(prev => ({ ...prev, prompt: e.target.value }))}
              placeholder="Leave blank to auto-generate from panel elements"
              rows={4}
            />
          </div>
          
          <div className="flex space-x-4">
            <button
              className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
              onClick={handleGeneratePanel}
              disabled={isGenerating}
            >
              {isGenerating ? 'Generating...' : 'Generate Panel'}
            </button>
            
            <button
              className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400"
              onClick={handleSavePanel}
              disabled={!generatedImage}
            >
              Save Panel
            </button>
          </div>
        </div>
        
        <div className="mt-8">
            <h3 className="text-lg font-semibold mb-2">Characters</h3>
            <div className="flex flex-wrap gap-2 mb-4">
              {selectedCharacters.map((char, index) => (
                <div 
                  key={`selected-${char.name}`} 
                  className="flex items-center bg-gray-100 rounded-full px-3 py-1"
                >
                  <span className="mr-2">{char.name}</span>
                  <button 
                    onClick={() => handleRemoveCharacter(index)}
                    className="text-red-500 hover:text-red-700"
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
            
            <div className="mb-4">
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                value=""
                onChange={(e) => {
                  const selectedChar = characters.find(c => c.name === e.target.value);
                  if (selectedChar) {
                    handleSelectCharacter(selectedChar);
                  }
                }}
              >
                <option value="">Add Character...</option>
                {characters.map(char => (
                  <option key={char.name} value={char.name}>{char.name}</option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="mt-8">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-lg font-semibold">Dialogues</h3>
              <button
                onClick={handleAddDialogue}
                className="px-3 py-1 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
              >
                Add Dialogue
              </button>
            </div>
            
            <div className="space-y-4">
              {panelData.dialogues.map((dialogue, index) => (
                <div key={`dialogue-${index}`} className="p-4 bg-gray-50 rounded-md border border-gray-200">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-2">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Character</label>
                      <input
                        type="text"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        value={dialogue.character}
                        onChange={(e) => handleUpdateDialogue(index, 'character', e.target.value)}
                        placeholder="Character name"
                      />
                    </div>
                    <div className="flex items-end">
                      <select
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        value={dialogue.character}
                        onChange={(e) => handleUpdateDialogue(index, 'character', e.target.value)}
                      >
                        <option value="">Select Character...</option>
                        {characters.map(char => (
                          <option key={char.name} value={char.name}>{char.name}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                  
                  <div className="mb-2">
                    <label className="block text-sm font-medium text-gray-700 mb-1">Text</label>
                    <textarea
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      value={dialogue.text}
                      onChange={(e) => handleUpdateDialogue(index, 'text', e.target.value)}
                      placeholder="What they say..."
                      rows={2}
                    />
                  </div>
                  
                  <button
                    onClick={() => handleRemoveDialogue(index)}
                    className="text-red-500 hover:text-red-700"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
          
          <div className="mt-8">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-lg font-semibold">Actions</h3>
              <button
                onClick={handleAddAction}
                className="px-3 py-1 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
              >
                Add Action
              </button>
            </div>
            
            <div className="space-y-4">
              {panelData.actions.map((action, index) => (
                <div key={`action-${index}`} className="p-4 bg-gray-50 rounded-md border border-gray-200">
                  <div className="mb-2">
                    <label className="block text-sm font-medium text-gray-700 mb-1">Text</label>
                    <textarea
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      value={action.text}
                      onChange={(e) => handleUpdateAction(index, e.target.value)}
                      placeholder="Describe the action..."
                      rows={2}
                    />
                  </div>
                  
                  <button
                    onClick={() => handleRemoveAction(index)}
                    className="text-red-500 hover:text-red-700"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      
      <div className="bg-white rounded-lg shadow-lg p-4 order-1 lg:order-2">
        <h2 className="text-2xl font-bold mb-4">Panel Preview</h2>
        
        <div className="relative w-full" style={{ aspectRatio: '1/1' }}>
          <Stage 
            width={canvasSize.width} 
            height={canvasSize.height}
            ref={stageRef}
            className="bg-gray-100 shadow-inner"
          >
            <Layer>
              {/* Panel frame */}
              <Rect
                width={canvasSize.width}
                height={canvasSize.height}
                stroke="#000"
                fill="#fff"
              />
              
              {/* Generated image if available */}
              {generatedImage && (
                <KonvaImage
                  image={(() => {
                    const img = new window.Image();
                    img.src = generatedImage;
                    return img;
                  })()}
                  width={canvasSize.width}
                  height={canvasSize.height}
                />
              )}
              
              {/* Only show boxes if no generated image yet */}
              {!generatedImage && (
                <>
                  {/* Character boxes */}
                  {panelData.characterPositions.map((pos, index) => (
                    <Rect
                      key={`char-${index}`}
                      x={pos.x * canvasSize.width}
                      y={pos.y * canvasSize.height}
                      width={pos.width * canvasSize.width}
                      height={pos.height * canvasSize.height}
                      stroke="#4299e1"
                      strokeWidth={2}
                      dash={[5, 5]}
                      draggable
                      onDragEnd={(e) => handleDragEnd(index, e, 'character')}
                      onClick={() => setSelectedItem({ type: 'character', index })}
                    />
                  ))}
                  
                  {/* Dialogue boxes */}
                  {panelData.dialoguePositions.map((pos, index) => (
                    <Rect
                      key={`dialogue-${index}`}
                      x={pos.x * canvasSize.width}
                      y={pos.y * canvasSize.height}
                      width={pos.width * canvasSize.width}
                      height={pos.height * canvasSize.height}
                      stroke="#f56565"
                      strokeWidth={2}
                      dash={[5, 5]}
                      draggable
                      onDragEnd={(e) => handleDragEnd(index, e, 'dialogue')}
                      onClick={() => setSelectedItem({ type: 'dialogue', index })}
                    />
                  ))}
                </>
              )}
            </Layer>
          </Stage>
        </div>
        
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">Box Controls</h3>
          <p className="text-sm text-gray-600 mb-2">
            Drag boxes to position characters and dialogue bubbles
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 bg-blue-500 mr-2"></div>
                <span className="text-sm">Character Boxes</span>
              </div>
            </div>
            <div>
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 bg-red-500 mr-2"></div>
                <span className="text-sm">Dialogue Boxes</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PanelEditor;