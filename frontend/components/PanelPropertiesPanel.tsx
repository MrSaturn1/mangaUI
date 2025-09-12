import React from 'react';
import { History } from 'lucide-react';
import { Character, Panel } from './MangaEditor';

interface DialogueItem {
  character: string;
  text: string;
}

interface PanelPropertiesPanelProps {
  selectedPanel: Panel | undefined;
  characters: Character[];
  panels: Panel[];
  selectedPanelId: string | null;
  selectedBoxType: 'character' | 'text' | null;
  selectedBoxIndex: number | null;
  panelMode: 'adjust' | 'character-box' | 'text-box';
  activeCharacter: string | null;
  showBoxes: boolean;
  showNegativePrompt: boolean;
  panelsWithHistory: Set<number>;
  DEFAULT_NEGATIVE_PROMPT: string;
  
  // Event handlers
  onAddCharacter: (character: Character) => void;
  onBoxSelect: (type: 'character' | 'text', index: number) => void;
  onDeleteBox: (type: 'character' | 'text', index: number) => void;
  onSetPanelMode: (mode: 'adjust' | 'character-box' | 'text-box') => void;
  onAddDialogue: () => void;
  onUpdateDialogue: (index: number, field: keyof DialogueItem, value: string) => void;
  onUpdateSeed: (seed: number) => void;
  onRandomizeSeed: () => void;
  onUpdatePrompt: (prompt: string) => void;
  onUpdateNegativePrompt: (prompt: string) => void;
  onToggleNegativePrompt: () => void;
  onGeneratePanel: () => void;
  onShowPanelHistory: (panelId: string) => void;
}

const PanelPropertiesPanel: React.FC<PanelPropertiesPanelProps> = ({
  selectedPanel,
  characters,
  panels,
  selectedPanelId,
  selectedBoxType,
  selectedBoxIndex,
  panelMode,
  activeCharacter,
  showBoxes,
  showNegativePrompt,
  panelsWithHistory,
  DEFAULT_NEGATIVE_PROMPT,
  onAddCharacter,
  onBoxSelect,
  onDeleteBox,
  onSetPanelMode,
  onAddDialogue,
  onUpdateDialogue,
  onUpdateSeed,
  onRandomizeSeed,
  onUpdatePrompt,
  onUpdateNegativePrompt,
  onToggleNegativePrompt,
  onGeneratePanel,
  onShowPanelHistory,
}) => {
  if (!selectedPanel) {
    return (
      <div className="w-[32rem] bg-white border-l border-gray-200 overflow-y-auto overflow-x-hidden flex-shrink-0" style={{ maxHeight: '100vh - 128px' }}>
        <div className="flex flex-col items-center justify-center h-64 text-black p-4">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 002 2z" />
          </svg>
          <p>Select a panel to edit</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-[32rem] bg-white border-l border-gray-200 overflow-y-auto overflow-x-hidden flex-shrink-0" style={{ maxHeight: '100vh - 128px' }}>
      <div className="p-4 max-w-full">
        <h2 className="text-2xl font-bold mb-4">Panel Editor</h2>
        {!showBoxes && (
          <div className="px-3 py-1 mb-4 bg-indigo-100 text-yellow-800 text-sm rounded-md">
            Character and Text Boxes Hidden - Toggle "Show Boxes" to edit positioning
          </div>
        )}
        
        <div className="space-y-6 max-w-full">
          {/* Character Section */}
          <div className="pb-4 border-b border-gray-200">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-lg font-semibold">Characters</h3>
              <select
                className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                value=""
                onChange={(e) => {
                  const selectedChar = characters.find(c => c.name === e.target.value);
                  if (selectedChar) {
                    onAddCharacter(selectedChar);
                  }
                  // Reset the select
                  e.target.value = "";
                }}
              >
                <option value="">Add Character...</option>
                {characters.map(char => (
                  <option key={char.name} value={char.name}>{char.name}</option>
                ))}
              </select>
            </div>
            
            {/* Character Boxes - now the single source of truth */}
            {selectedPanel.characterBoxes && selectedPanel.characterBoxes.length > 0 ? (
              <div className="space-y-2">
                <h4 className="font-medium text-sm mb-2">Character Placement:</h4>
                <div className="max-h-60 overflow-y-auto space-y-2">
                  {selectedPanel.characterBoxes.map((box, idx) => {
                    const isSelected = selectedBoxType === 'character' && selectedBoxIndex === idx;
                    
                    return (
                      <div 
                        key={`char-box-${idx}`}
                        className={`flex items-center text-sm p-3 rounded-md border cursor-pointer transition-colors ${
                          isSelected
                            ? 'border-indigo-500 bg-indigo-50'
                            : 'border-gray-200 hover:bg-gray-50'
                        }`}
                        onClick={() => {
                          onBoxSelect('character', idx);
                          onSetPanelMode('adjust');
                        }}
                      >
                        <div 
                          className="w-4 h-4 mr-3 rounded" 
                          style={{backgroundColor: box.color}}
                        ></div>
                        <div className="flex-1">
                          <div className="font-medium">{box.character}</div>
                          <div className="text-xs text-gray-500">
                            Position: ({Math.round(box.x * 100)}%, {Math.round(box.y * 100)}%) • 
                            Size: {Math.round(box.width * 100)}% × {Math.round(box.height * 100)}%
                          </div>
                        </div>
                        {isSelected && (
                          <span className="ml-2 px-2 py-1 text-xs bg-indigo-200 text-indigo-800 rounded">
                            Selected
                          </span>
                        )}
                        <button
                          className="ml-2 px-2 py-1 text-xs rounded bg-red-100 text-red-700 hover:bg-red-200"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteBox('character', idx);
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="text-gray-500 italic text-center py-4">
                No characters added. Select a character from the dropdown to add them to this panel.
              </div>
            )}
            
            {panelMode === 'character-box' && (
              <div className="mt-3 p-3 bg-indigo-50 rounded-md border border-indigo-200">
                <p className="text-sm font-medium mb-2">Drawing Character Box for: <span className="font-bold">{activeCharacter}</span></p>
                <p className="text-xs text-gray-600">Click and drag on the panel to draw a placement box for this character.</p>
              </div>
            )}
          </div>

          {/* Dialogues Section with Text Box Drawing capability */}
          <div className="pb-4 border-b border-gray-200">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-lg font-semibold">Dialogue</h3>
              <div className="flex space-x-2">
                <button
                  className="px-3 py-1 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                  onClick={onAddDialogue}
                >
                  Add Dialogue
                </button>
              </div>
            </div>
            
            {panelMode === 'text-box' && (
              <div className="mt-3 p-3 bg-indigo-50 rounded-md border border-indigo-200">
                <p className="text-sm font-medium mb-2">Drawing Text Box</p>
                <p className="text-xs text-gray-600">Click and drag on the panel to draw a placement box for dialogue text.</p>
              </div>
            )}
            
            {/* Combined Text Boxes and Dialogues */}
            <div className="space-y-4 mt-4">
              {selectedPanel.dialogues.map((dialogue, index) => {
                const textBox = selectedPanel.textBoxes?.[index];
                const isSelected = selectedBoxType === 'text' && selectedBoxIndex === index;
                
                return (
                  <div 
                    key={`dialogue-${index}`} 
                    className={`p-3 rounded-md border transition-colors cursor-pointer ${
                      isSelected 
                        ? 'border-indigo-500 bg-indigo-50' 
                        : 'border-gray-200 bg-gray-50 hover:bg-gray-100'
                    }`}
                    onClick={() => {
                      onBoxSelect('text', index);
                      onSetPanelMode('adjust');
                    }}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center">
                        <span className="text-sm font-medium text-gray-700 mr-2">
                          Text Box {index + 1}
                        </span>
                        {textBox && (
                          <span className="text-xs text-gray-500">
                            ({Math.round(textBox.x * 100)}%, {Math.round(textBox.y * 100)}%)
                          </span>
                        )}
                        {isSelected && (
                          <span className="ml-2 px-2 py-1 text-xs bg-indigo-200 text-indigo-800 rounded">
                            Selected
                          </span>
                        )}
                      </div>
                      <button
                        className="px-2 py-1 text-xs rounded bg-red-100 text-red-700 hover:bg-red-200"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteBox('text', index);
                        }}
                      >
                        Delete
                      </button>
                    </div>
                    
                    <div className="mb-2">
                      <label className="block text-sm font-medium text-black mb-1">Text</label>
                      <textarea
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        value={dialogue.text}
                        onChange={(e) => {
                          e.stopPropagation();
                          onUpdateDialogue(index, 'text', e.target.value);
                        }}
                        onClick={(e) => e.stopPropagation()}
                        placeholder="What they say..."
                        rows={2}
                      />
                    </div>
                    
                    {!textBox && (
                      <div className="text-xs text-red-600 mt-1">
                        ⚠️ Missing text box - click "Draw Text Box" to add positioning
                      </div>
                    )}
                  </div>
                );
              })}
              
              {selectedPanel.dialogues.length === 0 && (
                <div className="text-gray-500 italic text-center py-4">
                  No dialogue added. Click "Add Dialogue" to get started.
                </div>
              )}
            </div>
          </div>
          
          {/* Panel Settings */}
          <div className="pb-2 border-b border-gray-200">
            <div className="flex flex-wrap mb-4 max-w-xl">
              <div className="pr-4">
                <label className="block text-sm font-medium text-black mb-1">Panel Index</label>
                <div className="w-20 px-3 py-2 border border-gray-300 rounded-md bg-gray-50 text-gray-700">
                  {selectedPanel.panelIndex || 0}
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-black mb-1">Seed</label>
                <div className="flex space-x-1">
                  <input
                    type="number"
                    className="w-40 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    value={selectedPanel.seed || 0}
                    onChange={(e) => {
                      onUpdateSeed(parseInt(e.target.value) || 0);
                    }}
                  />
                  <button
                    className="px-3 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    onClick={onRandomizeSeed}
                  >
                    Random
                  </button>
                </div>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-black mb-1">Prompt</label>
              <textarea
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                value={selectedPanel.prompt || ''}
                onChange={(e) => onUpdatePrompt(e.target.value)}
                placeholder="Deep in the undergrowth, ferns shake and a RAT emerges..."
                rows={3}
              />
              <div className="mb-4">
                <button
                  className="w-full flex items-center justify-between p-2 bg-gray-50 hover:bg-gray-100 border border-gray-300 rounded-md transition-colors"
                  onClick={onToggleNegativePrompt}
                >
                  <span className="text-sm font-medium text-black">Negative Prompt</span>
                  <div className="flex items-center">
                    {!showNegativePrompt && (
                      <span className="text-xs text-gray-500 mr-2">
                        Specify what you want to avoid in the generated image
                      </span>
                    )}
                    <svg 
                      className={`w-4 h-4 transition-transform ${showNegativePrompt ? 'rotate-180' : ''}`}
                      fill="none" 
                      viewBox="0 0 24 24" 
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </button>
                
                {showNegativePrompt && (
                  <div className="mt-2 space-y-2">
                    <textarea
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 text-sm"
                      value={selectedPanel.negativePrompt || DEFAULT_NEGATIVE_PROMPT}
                      onChange={(e) => onUpdateNegativePrompt(e.target.value)}
                      placeholder="Things you don't want in the image..."
                      rows={4}
                    />
                    <button
                      className="px-3 py-2 bg-indigo-600 text-white text-xs rounded-md hover:bg-indigo-700 transition-colors"
                      onClick={() => onUpdateNegativePrompt(DEFAULT_NEGATIVE_PROMPT)}
                    >
                      Reset to Default
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Generate Panel Button */}
          <button
            className="w-full px-4 py-3 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
            onClick={onGeneratePanel}
            disabled={selectedPanel.isGenerating}
          >
            {selectedPanel.isGenerating 
              ? 'Generating...' 
              : selectedPanel.imageData 
                ? 'Regenerate Panel' 
                : 'Generate Panel'}
          </button>
          
          {/* Panel History Button */}
          {selectedPanelId && panelsWithHistory.has(panels.findIndex(p => p.id === selectedPanelId)) && (
            <button
              className="w-full mt-3 px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 flex items-center justify-center"
              onClick={() => selectedPanelId && onShowPanelHistory(selectedPanelId)}
            >
              <History size={16} className="mr-2" />
              View Panel History
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default PanelPropertiesPanel;