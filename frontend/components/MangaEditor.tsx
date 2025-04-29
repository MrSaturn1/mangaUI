// components/MangaEditor.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Stage, Layer, Rect, Image as KonvaImage, Transformer } from 'react-konva';
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

interface Panel {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  imageData?: string;
  prompt?: string;
  setting?: string;
  seed?: number;
  panelIndex?: number;
  characterNames: string[];
  characterPositions: Position[];
  dialogues: DialogueItem[];
  dialoguePositions: Position[];
  actions: ActionItem[];
  isGenerating?: boolean;
}

interface MangaEditorProps {
  characters: Character[];
}

const MangaEditor: React.FC<MangaEditorProps> = ({ characters }) => {
  // Page state
  const [pageSize, setPageSize] = useState({ width: 1654, height: 2339 }); // A4 proportions
  const [pageIndex, setPageIndex] = useState<number>(0);
  const [scale, setScale] = useState<number>(0.3); // Scale for the canvas
  
  // Panels state
  const [panels, setPanels] = useState<Panel[]>([]);
  const [selectedPanelId, setSelectedPanelId] = useState<string | null>(null);
  
  // Refs
  const stageRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  
  // Get the selected panel
  const selectedPanel = panels.find(p => p.id === selectedPanelId);
  
  // Effect to add some default panels on first load
  useEffect(() => {
    if (panels.length === 0) {
      // Add some default panels in a 2x2 grid
      const panelWidth = pageSize.width / 2;
      const panelHeight = pageSize.height / 2;
      
      const defaultPanels: Panel[] = [
        {
          id: 'panel-1',
          x: 0,
          y: 0,
          width: panelWidth,
          height: panelHeight,
          characterNames: [],
          characterPositions: [],
          dialogues: [],
          dialoguePositions: [],
          actions: [],
          panelIndex: 0
        },
        {
          id: 'panel-2',
          x: panelWidth,
          y: 0,
          width: panelWidth,
          height: panelHeight,
          characterNames: [],
          characterPositions: [],
          dialogues: [],
          dialoguePositions: [],
          actions: [],
          panelIndex: 1
        },
        {
          id: 'panel-3',
          x: 0,
          y: panelHeight,
          width: panelWidth,
          height: panelHeight,
          characterNames: [],
          characterPositions: [],
          dialogues: [],
          dialoguePositions: [],
          actions: [],
          panelIndex: 2
        },
        {
          id: 'panel-4',
          x: panelWidth,
          y: panelHeight,
          width: panelWidth,
          height: panelHeight,
          characterNames: [],
          characterPositions: [],
          dialogues: [],
          dialoguePositions: [],
          actions: [],
          panelIndex: 3
        }
      ];
      
      setPanels(defaultPanels);
    }
  }, []);
  
  // Effect to update the transformer when a panel is selected
  useEffect(() => {
    if (selectedPanelId && transformerRef.current && stageRef.current) {
      // Find the Konva node for the selected panel
      const node = stageRef.current.findOne(`#${selectedPanelId}`);
      if (node) {
        transformerRef.current.nodes([node]);
        transformerRef.current.getLayer().batchDraw();
      }
    } else if (transformerRef.current) {
      transformerRef.current.nodes([]);
      transformerRef.current.getLayer().batchDraw();
    }
  }, [selectedPanelId]);
  
  // Handler for selecting a panel
  const handlePanelSelect = (panelId: string) => {
    setSelectedPanelId(panelId);
  };
  
  // Handler for adding a new panel
  const handleAddPanel = () => {
    // Find a good location for the new panel
    const newPanel: Panel = {
      id: `panel-${Date.now()}`,
      x: 50,
      y: 50,
      width: 400,
      height: 400,
      characterNames: [],
      characterPositions: [],
      dialogues: [],
      dialoguePositions: [],
      actions: [],
      panelIndex: panels.length
    };
    
    setPanels([...panels, newPanel]);
    setSelectedPanelId(newPanel.id);
  };
  
  // Handler for deleting a panel
  const handleDeletePanel = () => {
    if (!selectedPanelId) return;
    
    setPanels(panels.filter(p => p.id !== selectedPanelId));
    setSelectedPanelId(null);
  };
  
  // Handler for panel transform
  const handleTransformEnd = (e: KonvaEventObject<Event>) => {
    if (!selectedPanelId) return;
    
    // Get the transformer node
    const node = e.target;
    
    // Find the panel in our state
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the panel with new dimensions
    const updatedPanels = [...panels];
    updatedPanels[panelIndex] = {
      ...updatedPanels[panelIndex],
      x: node.x(),
      y: node.y(),
      width: node.width() * node.scaleX(),
      height: node.height() * node.scaleY()
    };
    
    // Reset scale after updating dimensions
    node.scaleX(1);
    node.scaleY(1);
    
    setPanels(updatedPanels);
  };
  
  // Handler for panel drag
  const handleDragEnd = (e: KonvaEventObject<DragEvent>) => {
    if (!selectedPanelId) return;
    
    // Get the node
    const node = e.target;
    
    // Find the panel in our state
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the panel position
    const updatedPanels = [...panels];
    updatedPanels[panelIndex] = {
      ...updatedPanels[panelIndex],
      x: node.x(),
      y: node.y()
    };
    
    setPanels(updatedPanels);
  };
  
  // Handler for adding a character to the selected panel
  const handleAddCharacter = (character: Character) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Check if character is already in the panel
    if (panels[panelIndex].characterNames.includes(character.name)) return;
    
    // Add the character to the panel
    const updatedPanels = [...panels];
    const totalChars = updatedPanels[panelIndex].characterNames.length;
    
    // Create a default position for the character
    let defaultPosition: Position;
    
    if (totalChars === 0) {
      // Single character in center
      defaultPosition = { x: 0.2, y: 0.2, width: 0.6, height: 0.6 };
    } else if (totalChars === 1) {
      // Two characters side by side
      defaultPosition = { x: 0.55, y: 0.2, width: 0.35, height: 0.6 };
      // Also update the first character's position
      updatedPanels[panelIndex].characterPositions[0] = { x: 0.1, y: 0.2, width: 0.35, height: 0.6 };
    } else {
      // Multiple characters - spread evenly
      const section = 1.0 / (totalChars + 1);
      defaultPosition = { x: section * totalChars, y: 0.2, width: section, height: 0.6 };
    }
    
    updatedPanels[panelIndex].characterNames.push(character.name);
    updatedPanels[panelIndex].characterPositions.push(defaultPosition);
    
    setPanels(updatedPanels);
  };
  
  // Handler for removing a character from the selected panel
  const handleRemoveCharacter = (index: number) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Remove the character
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].characterNames.splice(index, 1);
    updatedPanels[panelIndex].characterPositions.splice(index, 1);
    
    setPanels(updatedPanels);
  };
  
  // Handler for adding dialogue to the selected panel
  const handleAddDialogue = () => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Add a new dialogue
    const updatedPanels = [...panels];
    const dialogues = updatedPanels[panelIndex].dialogues;
    const dialoguePositions = updatedPanels[panelIndex].dialoguePositions;
    
    // Create a default position for the dialogue
    let defaultPosition: Position;
    
    if (dialogues.length === 0) {
      // Top right
      defaultPosition = { x: 0.6, y: 0.1, width: 0.3, height: 0.2 };
    } else if (dialogues.length === 1) {
      // Middle left
      defaultPosition = { x: 0.1, y: 0.4, width: 0.3, height: 0.2 };
    } else {
      // Bottom right
      defaultPosition = { x: 0.6, y: 0.7, width: 0.3, height: 0.2 };
    }
    
    updatedPanels[panelIndex].dialogues.push({ character: '', text: '' });
    updatedPanels[panelIndex].dialoguePositions.push(defaultPosition);
    
    setPanels(updatedPanels);
  };
  
  // Handler for updating dialogue in the selected panel
  const handleUpdateDialogue = (index: number, field: keyof DialogueItem, value: string) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the dialogue
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].dialogues[index][field] = value;
    
    setPanels(updatedPanels);
  };
  
  // Handler for removing dialogue from the selected panel
  const handleRemoveDialogue = (index: number) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Remove the dialogue
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].dialogues.splice(index, 1);
    updatedPanels[panelIndex].dialoguePositions.splice(index, 1);
    
    setPanels(updatedPanels);
  };
  
  // Handler for adding an action to the selected panel
  const handleAddAction = () => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Add a new action
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].actions.push({ text: '' });
    
    setPanels(updatedPanels);
  };
  
  // Handler for updating an action in the selected panel
  const handleUpdateAction = (index: number, value: string) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the action
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].actions[index].text = value;
    
    setPanels(updatedPanels);
  };
  
  // Handler for removing an action from the selected panel
  const handleRemoveAction = (index: number) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Remove the action
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].actions.splice(index, 1);
    
    setPanels(updatedPanels);
  };
  
  // Handler for updating the setting of the selected panel
  const handleUpdateSetting = (value: string) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the setting
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].setting = value;
    
    setPanels(updatedPanels);
  };
  
  // Handler for updating the prompt of the selected panel
  const handleUpdatePrompt = (value: string) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the prompt
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].prompt = value;
    
    setPanels(updatedPanels);
  };
  
  // Handler for generating a panel image
  const handleGeneratePanel = async () => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Mark the panel as generating
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].isGenerating = true;
    setPanels(updatedPanels);
    
    try {
      // Prepare the data for the API
      const panel = panels[panelIndex];
      const apiData = {
        prompt: panel.prompt || '',
        setting: panel.setting || '',
        dialoguePositions: panel.dialoguePositions,
        characterPositions: panel.characterPositions,
        characterNames: panel.characterNames,
        dialogues: panel.dialogues,
        actions: panel.actions,
        panelIndex: panel.panelIndex || 0,
        seed: panel.seed || Math.floor(Math.random() * 1000000)
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
        // Update the panel with the generated image
        const newPanels = [...panels];
        newPanels[panelIndex] = {
          ...newPanels[panelIndex],
          imageData: data.imageData,
          prompt: data.prompt || newPanels[panelIndex].prompt,
          isGenerating: false,
          seed: apiData.seed // Store the used seed
        };
        
        setPanels(newPanels);
      } else {
        console.error('Error generating panel:', data.message);
        alert(`Error generating panel: ${data.message}`);
        
        // Mark the panel as not generating
        const newPanels = [...panels];
        newPanels[panelIndex].isGenerating = false;
        setPanels(newPanels);
      }
    } catch (error) {
      console.error('Error calling API:', error);
      alert(`Error calling API: ${error}`);
      
      // Mark the panel as not generating
      const newPanels = [...panels];
      newPanels[panelIndex].isGenerating = false;
      setPanels(newPanels);
    }
  };
  
  // Handler for saving the page
  const handleSavePage = async () => {
    // TODO: Implement page saving
    alert('Page saving not implemented yet');
  };
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 p-4">
      {/* Canvas Area - Takes up 3/5 of the screen on large displays */}
      <div className="lg:col-span-3 bg-white rounded-lg shadow-lg p-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold">Page Editor</h2>
          
          <div className="flex space-x-2">
            <button
              className="px-3 py-1 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
              onClick={handleAddPanel}
            >
              Add Panel
            </button>
            <button
              className="px-3 py-1 bg-red-600 text-white rounded-md hover:bg-red-700"
              onClick={handleDeletePanel}
              disabled={!selectedPanelId}
            >
              Delete Panel
            </button>
            <button
              className="px-3 py-1 bg-green-600 text-white rounded-md hover:bg-green-700"
              onClick={handleSavePage}
            >
              Save Page
            </button>
          </div>
        </div>
        
        <div className="relative mx-auto" style={{ width: `${pageSize.width * scale}px`, height: `${pageSize.height * scale}px` }}>
          <Stage 
            width={pageSize.width * scale} 
            height={pageSize.height * scale}
            ref={stageRef}
            className="bg-gray-100 shadow-inner border border-gray-300"
          >
            <Layer>
              {/* Page background */}
              <Rect
                width={pageSize.width * scale}
                height={pageSize.height * scale}
                fill="white"
              />
              
              {/* Panels */}
              {panels.map((panel) => (
                <Rect
                  key={panel.id}
                  id={panel.id}
                  x={panel.x * scale}
                  y={panel.y * scale}
                  width={panel.width * scale}
                  height={panel.height * scale}
                  stroke={selectedPanelId === panel.id ? '#4299e1' : '#000'}
                  strokeWidth={selectedPanelId === panel.id ? 2 : 1}
                  fill={panel.imageData ? 'transparent' : '#f7fafc'}
                  onClick={() => handlePanelSelect(panel.id)}
                  onTap={() => handlePanelSelect(panel.id)}
                  draggable
                  onDragEnd={handleDragEnd}
                />
              ))}
              
              {/* Panel images */}
              {panels.map((panel) => (
                panel.imageData && (
                  <KonvaImage
                    key={`img-${panel.id}`}
                    x={panel.x * scale}
                    y={panel.y * scale}
                    width={panel.width * scale}
                    height={panel.height * scale}
                    image={(() => {
                      const img = new window.Image();
                      img.src = panel.imageData;
                      return img;
                    })()}
                    onClick={() => handlePanelSelect(panel.id)}
                    onTap={() => handlePanelSelect(panel.id)}
                  />
                )
              ))}
              
              {/* Transformer for resizing panels */}
              <Transformer
                ref={transformerRef}
                boundBoxFunc={(oldBox, newBox) => {
                  // Limit size to prevent tiny panels
                  if (newBox.width < 20 || newBox.height < 20) {
                    return oldBox;
                  }
                  return newBox;
                }}
                onTransformEnd={handleTransformEnd}
                padding={5}
                enabledAnchors={['top-left', 'top-right', 'bottom-left', 'bottom-right']}
                rotateEnabled={false}
              />
            </Layer>
          </Stage>
          
          {/* Loading overlay for panel generation */}
          {selectedPanel?.isGenerating && (
            <div className="absolute inset-0 bg-black bg-opacity-40 flex justify-center items-center">
              <div className="text-white text-lg">Generating panel...</div>
            </div>
          )}
        </div>
      </div>
      
      {/* Panel Editor Controls - Takes up 2/5 of the screen on large displays */}
      <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-4 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 2rem)' }}>
        {selectedPanel ? (
          <div>
            <h2 className="text-2xl font-bold mb-4">Panel Editor</h2>
            
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">Panel Settings</h3>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Panel Index</label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      value={selectedPanel.panelIndex || 0}
                      onChange={(e) => {
                        const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
                        if (panelIndex === -1) return;
                        
                        const updatedPanels = [...panels];
                        updatedPanels[panelIndex].panelIndex = parseInt(e.target.value) || 0;
                        
                        setPanels(updatedPanels);
                      }}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Seed</label>
                    <div className="flex space-x-2">
                      <input
                        type="number"
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        value={selectedPanel.seed || 0}
                        onChange={(e) => {
                          const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
                          if (panelIndex === -1) return;
                          
                          const updatedPanels = [...panels];
                          updatedPanels[panelIndex].seed = parseInt(e.target.value) || 0;
                          
                          setPanels(updatedPanels);
                        }}
                      />
                      <button
                        className="px-3 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                        onClick={() => {
                          const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
                          if (panelIndex === -1) return;
                          
                          const updatedPanels = [...panels];
                          updatedPanels[panelIndex].seed = Math.floor(Math.random() * 1000000);
                          
                          setPanels(updatedPanels);
                        }}
                      >
                        Random
                      </button>
                    </div>
                  </div>
                </div>
                
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Setting</label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    value={selectedPanel.setting || ''}
                    onChange={(e) => handleUpdateSetting(e.target.value)}
                    placeholder="e.g., INT. HOTEL LOBBY - DAY, manga panel"
                  />
                </div>
                
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Custom Prompt (optional)</label>
                  <textarea
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    value={selectedPanel.prompt || ''}
                    onChange={(e) => handleUpdatePrompt(e.target.value)}
                    placeholder="Leave blank to auto-generate from panel elements"
                    rows={3}
                  />
                </div>
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-lg font-semibold">Characters</h3>
                  <select
                    className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    value=""
                    onChange={(e) => {
                      const selectedChar = characters.find(c => c.name === e.target.value);
                      if (selectedChar) {
                        handleAddCharacter(selectedChar);
                      }
                    }}
                  >
                    <option value="">Add Character...</option>
                    {characters.map(char => (
                      <option key={char.name} value={char.name}>{char.name}</option>
                    ))}
                  </select>
                </div>
                
                <div className="flex flex-wrap gap-2 mb-4">
                  {selectedPanel.characterNames.map((name, index) => (
                    <div 
                      key={`char-${index}`} 
                      className="flex items-center bg-indigo-100 rounded-full px-3 py-1"
                    >
                      <span className="mr-2">{name}</span>
                      <button 
                        onClick={() => handleRemoveCharacter(index)}
                        className="text-red-500 hover:text-red-700"
                      >
                        &times;
                      </button>
                    </div>
                  ))}
                  
                  {selectedPanel.characterNames.length === 0 && (
                    <div className="text-gray-500 italic">No characters added</div>
                  )}
                </div>
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-lg font-semibold">Dialogues</h3>
                  <button
                    className="px-3 py-1 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                    onClick={handleAddDialogue}
                  >
                    Add Dialogue
                  </button>
                </div>
                
                <div className="space-y-4">
                  {selectedPanel.dialogues.map((dialogue, index) => (
                    <div key={`dialogue-${index}`} className="p-3 bg-gray-50 rounded-md border border-gray-200">
                      <div className="mb-2">
                        <label className="block text-sm font-medium text-gray-700 mb-1">Character</label>
                        <select
                          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                          value={dialogue.character}
                          onChange={(e) => handleUpdateDialogue(index, 'character', e.target.value)}
                        >
                          <option value="">Select Character...</option>
                          {selectedPanel.characterNames.map(name => (
                            <option key={name} value={name}>{name}</option>
                          ))}
                        </select>
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
                  
                  {selectedPanel.dialogues.length === 0 && (
                    <div className="text-gray-500 italic">No dialogues added</div>
                  )}
                </div>
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-lg font-semibold">Actions</h3>
                  <button
                    className="px-3 py-1 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                    onClick={handleAddAction}
                  >
                    Add Action
                  </button>
                </div>
                
                <div className="space-y-4">
                  {selectedPanel.actions.map((action, index) => (
                    <div key={`action-${index}`} className="p-3 bg-gray-50 rounded-md border border-gray-200">
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
                  
                  {selectedPanel.actions.length === 0 && (
                    <div className="text-gray-500 italic">No actions added</div>
                  )}
                </div>
              </div>
              
              <button
                className="w-full px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
                onClick={handleGeneratePanel}
                disabled={selectedPanel.isGenerating}
              >
                {selectedPanel.isGenerating 
                  ? 'Generating...' 
                  : selectedPanel.imageData 
                    ? 'Regenerate Panel' 
                    : 'Generate Panel'}
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p>Select a panel to edit</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MangaEditor;