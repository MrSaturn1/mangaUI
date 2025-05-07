import React from 'react';
import { Circle } from 'react-konva';
import { KonvaEventObject } from 'konva/lib/Node';

interface PanelAdjustmentHandlesProps {
  /**
   * The panel to render adjustment handles for
   */
  panel: {
    id: string;
    x: number;
    y: number;
    width: number;
    height: number;
  };
  
  /**
   * Whether the panel is currently selected
   */
  isSelected: boolean;
  
  /**
   * The current interaction mode
   */
  mode: 'adjust' | 'character-box' | 'text-box';
  
  /**
   * The current scale of the canvas
   */
  scale: number;
  
  /**
   * Optional callback for when a handle is dragged
   */
  onHandleDrag?: (position: { x: number; y: number }, handleType: HandleType) => void;
  
  /**
   * Optional callback for when a handle drag ends
   */
  onHandleDragEnd?: (e: KonvaEventObject<DragEvent>, handleType: HandleType) => void;
}

/**
 * Types of handles that can be dragged
 */
export type HandleType = 
  | 'top-left' 
  | 'top-center' 
  | 'top-right'
  | 'middle-left' 
  | 'middle-right'
  | 'bottom-left' 
  | 'bottom-center' 
  | 'bottom-right';

/**
 * A component that renders adjustment handles for a panel
 * to allow direct manipulation resizing
 */
const PanelAdjustmentHandles: React.FC<PanelAdjustmentHandlesProps> = ({ 
  panel, 
  isSelected, 
  mode, 
  scale,
  onHandleDrag,
  onHandleDragEnd
}) => {
  // Only show handles when panel is selected and in adjust mode
  if (!isSelected || mode !== 'adjust') return null;
  
  // Handle size in canvas coordinates (gets scaled properly)
  const handleSize = 8 / scale;
  const handleColor = '#4299e1';
  
  // Calculate positions for all 8 handles
  const handles: Array<{ x: number; y: number; type: HandleType }> = [
    // Corner handles
    { x: panel.x, y: panel.y, type: 'top-left' },
    { x: panel.x + panel.width, y: panel.y, type: 'top-right' },
    { x: panel.x, y: panel.y + panel.height, type: 'bottom-left' },
    { x: panel.x + panel.width, y: panel.y + panel.height, type: 'bottom-right' },
    
    // Edge handles
    { x: panel.x + panel.width/2, y: panel.y, type: 'top-center' },
    { x: panel.x + panel.width, y: panel.y + panel.height/2, type: 'middle-right' },
    { x: panel.x + panel.width/2, y: panel.y + panel.height, type: 'bottom-center' },
    { x: panel.x, y: panel.y + panel.height/2, type: 'middle-left' },
  ];
  
  const handleDrag = (e: KonvaEventObject<DragEvent>, type: HandleType) => {
    if (onHandleDrag) {
      const pos = e.target.position();
      onHandleDrag(pos, type);
    }
  };
  
  const handleDragEnd = (e: KonvaEventObject<DragEvent>, type: HandleType) => {
    if (onHandleDragEnd) {
      onHandleDragEnd(e, type);
    }
  };
  
  return (
    <>
      {handles.map((handle, index) => (
        <Circle
          key={`${panel.id}-handle-${index}`}
          x={handle.x}
          y={handle.y}
          radius={handleSize}
          fill={handleColor}
          stroke="#ffffff"
          strokeWidth={1 / scale}
          draggable={true}
          onDragMove={(e) => handleDrag(e, handle.type)}
          onDragEnd={(e) => handleDragEnd(e, handle.type)}
          // Handle dragging constraints
          dragBoundFunc={(pos) => {
            const bounds = { x: pos.x, y: pos.y };
            
            // Constrain based on handle type
            switch (handle.type) {
              case 'top-left':
                // No constraints, can move freely
                break;
              case 'top-center':
                // Can only move vertically
                bounds.x = panel.x + panel.width/2;
                break;
              case 'top-right':
                // Right edge can't go left of left edge
                bounds.x = Math.max(bounds.x, panel.x + handleSize);
                break;
              case 'middle-left':
                // Can only move horizontally
                bounds.y = panel.y + panel.height/2;
                break;
              case 'middle-right':
                // Can only move horizontally
                bounds.y = panel.y + panel.height/2;
                // Right edge can't go left of left edge
                bounds.x = Math.max(bounds.x, panel.x + handleSize);
                break;
              case 'bottom-left':
                // Bottom edge can't go above top edge
                bounds.y = Math.max(bounds.y, panel.y + handleSize);
                break;
              case 'bottom-center':
                // Can only move vertically
                bounds.x = panel.x + panel.width/2;
                // Bottom edge can't go above top edge
                bounds.y = Math.max(bounds.y, panel.y + handleSize);
                break;
              case 'bottom-right':
                // Right edge can't go left of left edge
                bounds.x = Math.max(bounds.x, panel.x + handleSize);
                // Bottom edge can't go above top edge
                bounds.y = Math.max(bounds.y, panel.y + handleSize);
                break;
            }
            
            return bounds;
          }}
        />
      ))}
    </>
  );
};

export default PanelAdjustmentHandles;