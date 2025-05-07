import React from 'react';
import { Square, MessageSquare } from 'lucide-react';
import { Panel } from './MangaEditor'; // Assuming you have a types file

interface ModeStatusBarProps {
  /**
   * The current interaction mode
   */
  mode: 'adjust' | 'character-box' | 'text-box';
  
  /**
   * The currently selected panel, if any
   */
  selectedPanel: Panel | null | undefined;
  
  /**
   * The currently active character for character box drawing
   */
  activeCharacter: string | null;
  
  /**
   * Type of the box currently selected for editing
   */
  selectedBoxType: 'character' | 'text' | null;
  
  /**
   * Index of the box currently selected for editing
   */
  selectedBoxIndex: number | null;
}

/**
 * A status bar component that displays the current mode and context
 * with helpful hints based on the current interaction state
 */
const ModeStatusBar: React.FC<ModeStatusBarProps> = ({ 
  mode, 
  selectedPanel, 
  activeCharacter, 
  selectedBoxType, 
  selectedBoxIndex 
}) => {
  // Define colors and messages based on mode
  let bgColor = 'bg-gray-100';
  let textColor = 'text-gray-700';
  let icon = null;
  let message = 'Ready';
  
  if (mode === 'adjust') {
    if (selectedBoxType === 'character' && selectedBoxIndex !== null) {
      bgColor = 'bg-indigo-100';
      textColor = 'text-indigo-700';
      icon = <Square size={16} className="mr-2" />;
      const box = selectedPanel?.characterBoxes?.[selectedBoxIndex];
      message = `Editing character box: ${box?.character || 'Unknown'}`;
    } else if (selectedBoxType === 'text' && selectedBoxIndex !== null) {
      bgColor = 'bg-indigo-100';
      textColor = 'text-indigo-700';
      icon = <MessageSquare size={16} className="mr-2" />;
      message = `Editing text box #${selectedBoxIndex + 1}`;
    } else if (selectedPanel) {
      bgColor = 'bg-green-100';
      textColor = 'text-green-700';
      message = 'Panel selected - Adjust mode';
    }
  } else if (mode === 'character-box') {
    bgColor = 'bg-purple-100';
    textColor = 'text-purple-700';
    icon = <Square size={16} className="mr-2" />;
    message = `Drawing character box for: ${activeCharacter || 'Select a character'}`;
  } else if (mode === 'text-box') {
    bgColor = 'bg-blue-100';
    textColor = 'text-blue-700';
    icon = <MessageSquare size={16} className="mr-2" />;
    message = 'Drawing text box';
  }
  
  return (
    <div className={`flex items-center px-3 py-1.5 ${bgColor} ${textColor} rounded-md`}>
      {icon}
      <span className="text-sm font-medium">{message}</span>
      
      {/* Keys help */}
      {selectedBoxType && selectedBoxIndex !== null && (
        <span className="ml-4 text-xs bg-white px-2 py-0.5 rounded">
          Delete key: Remove box
        </span>
      )}
      
      {/* Mouse gestures */}
      {(mode === 'character-box' || mode === 'text-box') && (
        <span className="ml-4 text-xs bg-white px-2 py-0.5 rounded">
          Click and drag to draw
        </span>
      )}
      
      {/* Additional help for adjust mode */}
      {mode === 'adjust' && selectedPanel && !selectedBoxType && (
        <span className="ml-4 text-xs bg-white px-2 py-0.5 rounded">
          Drag to move • Handles to resize
        </span>
      )}
      
      {/* Escape key hint when in drawing mode */}
      {mode !== 'adjust' && (
        <span className="ml-auto text-xs bg-white px-2 py-0.5 rounded">
          ESC: Cancel
        </span>
      )}
    </div>
  );
};

export default ModeStatusBar;