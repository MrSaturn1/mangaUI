// components/MangaEditor.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Stage, Layer, Rect, Image as KonvaImage, Transformer, Line } from 'react-konva';
import { KonvaEventObject } from 'konva/lib/Node';
import { saveAs } from 'file-saver';
import { PageTemplate, pageTemplates } from '../utils/pageTemplates';
import { 
  ChevronLeft, 
  ChevronRight, 
  Plus, 
  Save, 
  FileDown, 
  Grid, 
  Folder, 
  ZoomIn, 
  ZoomOut, 
  Maximize, 
  Trash2,
  SquarePen,
  Download,
  Layout,
  Square,
  MessageSquare
} from 'lucide-react';


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

export interface Panel {
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
  generationQueued?: boolean;
  queueMessage?: string;
  characterBoxes?: {
    character: string;
    x: number;
    y: number;
    width: number;
    height: number;
    color: string; // For different colors per character
  }[];
  textBoxes?: {
    text: string;
    x: number;
    y: number;
    width: number;
    height: number;
  }[];
}

interface Page {
  id: string;
  panels: Panel[]; // This explicitly tells TypeScript that panels is an array of Panel
}

interface MangaEditorProps {
  characters: Character[];
  apiEndpoint?: string; // Base API endpoint URL
  currentProject?: any;
  setCurrentProject?: (project: any) => void;
  onProjectSave?: (projectId: string, pages: Page[]) => void;
}

const MangaEditor: React.FC<MangaEditorProps> = ({ characters, apiEndpoint = 'http://localhost:5000/api', currentProject, setCurrentProject, onProjectSave }) => {
  // Page state
  const [pageSize, setPageSize] = useState({ width: 1500, height: 2250 }); // A4 proportions
  const [pageIndex, setPageIndex] = useState<number>(0);
  const [scale, setScale] = useState<number>(0.3); // Scale for the canvas
  const [pages, setPages] = useState([{ id: 'page-1', panels: [] as Panel[] }]);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [showTemplateDialog, setShowTemplateDialog] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  
  // Panels state
  const panels: Panel[] = pages[currentPageIndex]?.panels || [];
  const [selectedPanelId, setSelectedPanelId] = useState<string | null>(null);
  // State for tracking queued panel requests
  const [queuedPanelRequests, setQueuedPanelRequests] = useState<{[key: string]: {
    panelId: string;
    requestId: string;
    checkInterval: NodeJS.Timeout | null;
  }}>({});

  // Character and Text Boxes
  const [isDrawingCharacterBox, setIsDrawingCharacterBox] = useState<boolean>(false);
  const [isDrawingTextBox, setIsDrawingTextBox] = useState<boolean>(false);
  const [activeCharacter, setActiveCharacter] = useState<string | null>(null);
  const [drawingStartPos, setDrawingStartPos] = useState<{x: number, y: number} | null>(null);

  // Refs
  const stageRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  
  // Get the selected panel
  const selectedPanel = panels.find(p => p.id === selectedPanelId);

  // Project Management
  const [showProjectManager, setShowProjectManager] = useState<boolean>(false);

  // Helper lines and snapping
  const [guides, setGuides] = useState<{x: number[], y: number[]}>({x: [], y: []});
  const [showGuides, setShowGuides] = useState(false);
  const [snapThreshold, setSnapThreshold] = useState<number>(10); // In pixels, adjust as needed
  const [isSnappingEnabled, setIsSnappingEnabled] = useState<boolean>(true);
  
  // Effect to add some default panels on first load
  useEffect(() => {
    // Check if the current page has no panels
    if (pages[currentPageIndex]?.panels.length === 0) {
      const defaultPanels = createDefaultPanelsForPage();
      const updatedPages = [...pages];
      updatedPages[currentPageIndex].panels = defaultPanels;
      setPages(updatedPages);
    }
  }, [currentPageIndex, pages]);
  
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

  {/* useEffect(() => {
    if (currentProject && currentProject.characters) {
      // This assumes your character list is passed in as props
      const projectCharacters = characters.filter(
        character => currentProject.characters.includes(character.name)
      );
      
    }
  }, [currentProject]);

  const handleProjectSelect = (project) => {
    setCurrentProject(project);
    
    // If the project has pages, load them
    if (project.pages && project.pages.length > 0) {
      setPages(project.pages);
    } else {
      // Otherwise create a default page
      setPages([
        {
          id: `page-${uuidv4()}`,
          panels: createDefaultPanelsForPage()
        }
      ]);
    }
    
    setShowProjectManager(false);
  }; */}

  // Clean up intervals on unmount
  useEffect(() => {
    return () => {
      // Clean up any pending check intervals
      Object.values(queuedPanelRequests).forEach(request => {
        if (request.checkInterval) {
          clearInterval(request.checkInterval);
        }
      });
    };
  }, [queuedPanelRequests]);
  
  // Handler for selecting a panel
  const handlePanelSelect = (panelId: string) => {
    setSelectedPanelId(panelId);
  };

  const updatePanelsForCurrentPage = (newPanels: Panel[]) => {
    const updatedPages = [...pages];
    updatedPages[currentPageIndex] = {
      ...updatedPages[currentPageIndex],
      panels: newPanels
    };
    setPages(updatedPages);
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
    
    updatePanelsForCurrentPage([...panels, newPanel]);
    setSelectedPanelId(newPanel.id);
  };
  
  // Handler for deleting a panel
  const handleDeletePanel = () => {
    if (!selectedPanelId) return;
    
    updatePanelsForCurrentPage(panels.filter(p => p.id !== selectedPanelId));
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
    
    // Calculate actual dimensions (accounting for scale)
    let actualX = node.x() / scale;
    let actualY = node.y() / scale;
    let actualWidth = (node.width() * node.scaleX()) / scale;
    let actualHeight = (node.height() * node.scaleY()) / scale;
    
    // Apply snapping if enabled
    if (isSnappingEnabled) {
      // Generate guides for panels and page edges (similar to handleDragMove)
      const xGuides: number[] = [];
      const yGuides: number[] = [];
      
      // Add guides for each panel edge and centers
      panels.forEach(panel => {
        if (panel.id !== selectedPanelId) {
          // Panel edges
          const panelLeft = panel.x;
          const panelRight = panel.x + panel.width;
          const panelTop = panel.y;
          const panelBottom = panel.y + panel.height;
          const panelCenterX = panel.x + panel.width / 2;
          const panelCenterY = panel.y + panel.height / 2;
          
          // Edge guides
          xGuides.push(panelLeft);           // Left edge
          xGuides.push(panelRight);          // Right edge
          xGuides.push(panelCenterX);        // Center X
          
          yGuides.push(panelTop);            // Top edge
          yGuides.push(panelBottom);         // Bottom edge
          yGuides.push(panelCenterY);        // Center Y
        }
      });
      
      // Add page boundary guides
      xGuides.push(0);                       // Left page edge
      xGuides.push(pageSize.width);          // Right page edge
      xGuides.push(pageSize.width / 2);      // Page center X
      
      yGuides.push(0);                       // Top page edge
      yGuides.push(pageSize.height);         // Bottom page edge
      yGuides.push(pageSize.height / 2);     // Page center Y
      
      // Find snap positions
      const snapX = getSnapPosition(actualX, xGuides);
      const snapRight = getSnapPosition(actualX + actualWidth, xGuides);
      const snapY = getSnapPosition(actualY, yGuides);
      const snapBottom = getSnapPosition(actualY + actualHeight, yGuides);
      
      // Apply snapping
      if (snapX !== null) {
        actualX = snapX;
      }
      
      if (snapRight !== null) {
        actualWidth = snapRight - actualX;
      }
      
      if (snapY !== null) {
        actualY = snapY;
      }
      
      if (snapBottom !== null) {
        actualHeight = snapBottom - actualY;
      }
    }
    
    // Update the panel with new dimensions
    const updatedPanels = [...panels];
    updatedPanels[panelIndex] = {
      ...updatedPanels[panelIndex],
      x: actualX,
      y: actualY,
      width: actualWidth,
      height: actualHeight
    };
    
    // Reset scale after updating dimensions
    node.scaleX(1);
    node.scaleY(1);
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Helper function to find the closest guide
  const getSnapPosition = (position: number, guides: number[]): number | null => {
    // Find closest guide within threshold
    let closest = null;
    let minDistance = snapThreshold;
    
    guides.forEach(guide => {
      const distance = Math.abs(position - guide);
      if (distance < minDistance) {
        minDistance = distance;
        closest = guide;
      }
    });
    
    return closest;
  };

  // Modified drag move handler
  const handleDragMove = (e: KonvaEventObject<DragEvent>) => {
    if (!selectedPanelId) return;
    
    const node = e.target;
    const nodeWidth = node.width() * node.scaleX();
    const nodeHeight = node.height() * node.scaleY();

    // If snapping is disabled, just clear guides and return
    if (!isSnappingEnabled) {
      setGuides({ x: [], y: [] });
      setShowGuides(false);
      return;
    }
    
    // Initial position
    let x = node.x();
    let y = node.y();
    
    // Generate guides for other panel edges and center lines
    const xGuides: number[] = [];
    const yGuides: number[] = [];
    
    // Add guides for each panel edge and centers
    panels.forEach(panel => {
      if (panel.id !== selectedPanelId) {
        // Panel edges (scaled for canvas)
        const panelLeft = panel.x * scale;
        const panelRight = (panel.x + panel.width) * scale;
        const panelTop = panel.y * scale;
        const panelBottom = (panel.y + panel.height) * scale;
        const panelCenterX = (panel.x + panel.width / 2) * scale;
        const panelCenterY = (panel.y + panel.height / 2) * scale;
        
        // Edge guides
        xGuides.push(panelLeft);           // Left edge
        xGuides.push(panelRight);          // Right edge
        xGuides.push(panelCenterX);        // Center X
        
        // Also add guides for aligned right/left edges
        xGuides.push(panelLeft - nodeWidth);  // My right to their left
        xGuides.push(panelRight - nodeWidth); // My right to their right
        
        yGuides.push(panelTop);            // Top edge
        yGuides.push(panelBottom);         // Bottom edge
        yGuides.push(panelCenterY);        // Center Y
        
        // Also add guides for aligned top/bottom edges
        yGuides.push(panelTop - nodeHeight);   // My bottom to their top
        yGuides.push(panelBottom - nodeHeight); // My bottom to their bottom
      }
    });
    
    // Add page boundary guides
    xGuides.push(0);                       // Left page edge
    xGuides.push(pageSize.width * scale);  // Right page edge
    xGuides.push((pageSize.width * scale) / 2); // Page center X
    
    yGuides.push(0);                       // Top page edge
    yGuides.push(pageSize.height * scale); // Bottom page edge
    yGuides.push((pageSize.height * scale) / 2); // Page center Y
    
    // Add guides for equal spacing
    // (more complex - implement if needed)
    
    // Find snap positions
    const snapX = getSnapPosition(x, xGuides);
    const snapY = getSnapPosition(y, yGuides);
    
    // Apply snapping if within threshold
    if (snapX !== null) {
      node.x(snapX);
    }
    
    if (snapY !== null) {
      node.y(snapY);
    }
    
    // Show guidelines
    setGuides({ 
      x: snapX !== null ? [snapX] : [], 
      y: snapY !== null ? [snapY] : [] 
    });
    setShowGuides(true);
  };

  // Update the drag end handler
  const handleDragEnd = (e: KonvaEventObject<DragEvent>) => {
    if (!selectedPanelId) return;
    
    // Hide guides
    setShowGuides(false);
    
    // Get the node
    const node = e.target;
    
    // Find the panel in our state
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the panel position
    const updatedPanels = [...panels];
    updatedPanels[panelIndex] = {
      ...updatedPanels[panelIndex],
      x: node.x() / scale, // Convert back from canvas units to model units
      y: node.y() / scale
    };
    
    updatePanelsForCurrentPage(updatedPanels);
  };

  // Zoom control functions
  const handleZoomIn = () => {
    setScale(prevScale => Math.min(prevScale + 0.1, 1.0));
  };

  const handleZoomOut = () => {
    setScale(prevScale => Math.max(prevScale - 0.1, 0.1));
  };

  const handleZoomReset = () => {
    setScale(0.3); // Reset to default
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
    
    updatePanelsForCurrentPage(updatedPanels);
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
    
    updatePanelsForCurrentPage(updatedPanels);
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
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Handler for updating dialogue in the selected panel
  const handleUpdateDialogue = (index: number, field: keyof DialogueItem, value: string) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the dialogue
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].dialogues[index][field] = value;
    
    updatePanelsForCurrentPage(updatedPanels);
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
    
    updatePanelsForCurrentPage(updatedPanels);
  };

  // Add mouse handlers for drawing
  const handleStageMouseDown = (e: KonvaEventObject<MouseEvent>) => {
    // If clicking directly on the stage (background), deselect current panel
    if (e.target === e.currentTarget) {
      setSelectedPanelId(null);
      return;
    }
    
    if (!selectedPanelId || (!isDrawingCharacterBox && !isDrawingTextBox)) return;
    
    // Get pointer position relative to the stage
    const pos = e.target.getStage()?.getPointerPosition();
    if (!pos) return;
    
    // Find the selected panel
    const panel = panels.find(p => p.id === selectedPanelId);
    if (!panel) return;
    
    // Check if click is inside the selected panel
    const panelX = panel.x * scale;
    const panelY = panel.y * scale;
    const panelWidth = panel.width * scale;
    const panelHeight = panel.height * scale;
    
    if (
      pos.x >= panelX && 
      pos.x <= panelX + panelWidth && 
      pos.y >= panelY && 
      pos.y <= panelY + panelHeight
    ) {
      // Convert from canvas coordinates to panel-relative coordinates (0-1)
      const relativeX = (pos.x - panelX) / panelWidth;
      const relativeY = (pos.y - panelY) / panelHeight;
      
      // Store starting position
      setDrawingStartPos({ x: relativeX, y: relativeY });
    }
  };

  const handleStageMouseMove = (e: KonvaEventObject<MouseEvent>) => {
    // Preview logic can be added here
  };

  const handleStageMouseUp = (e: KonvaEventObject<MouseEvent>) => {
    if (!selectedPanelId || !drawingStartPos || (!isDrawingCharacterBox && !isDrawingTextBox)) return;
    
    const pos = e.target.getStage()?.getPointerPosition();
    if (!pos) return;
    
    // Find the selected panel
    const panel = panels.find(p => p.id === selectedPanelId);
    if (!panel) return;
    
    // Check if mouse up is inside the panel
    const panelX = panel.x * scale;
    const panelY = panel.y * scale;
    const panelWidth = panel.width * scale;
    const panelHeight = panel.height * scale;
    
    if (
      pos.x >= panelX && 
      pos.x <= panelX + panelWidth && 
      pos.y >= panelY && 
      pos.y <= panelY + panelHeight
    ) {
      // Convert to panel-relative coordinates (0-1)
      const relativeX = (pos.x - panelX) / panelWidth;
      const relativeY = (pos.y - panelY) / panelHeight;
      
      // Calculate width and height
      const width = Math.abs(relativeX - drawingStartPos.x);
      const height = Math.abs(relativeY - drawingStartPos.y);
      
      // Calculate top-left corner
      const x = Math.min(drawingStartPos.x, relativeX);
      const y = Math.min(drawingStartPos.y, relativeY);
      
      // Minimum size check
      if (width < 0.05 || height < 0.05) {
        setDrawingStartPos(null);
        return;
      }
      
      // Find the panel index
      const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
      if (panelIndex === -1) return;
      
      const updatedPanels = [...panels];
      
      if (isDrawingCharacterBox && activeCharacter) {
        // Get a color based on the character name (for consistency)
        const characterColors = [
          '#FF5733', '#33FF57', '#3357FF', '#FF33F5', '#F5FF33', 
          '#33FFF5', '#F533FF', '#FF3333', '#33FF33', '#3333FF'
        ];
        const colorIndex = activeCharacter.charCodeAt(0) % characterColors.length;
        
        // Add the character box
        const characterBoxes = [...(updatedPanels[panelIndex].characterBoxes || [])];
        characterBoxes.push({
          character: activeCharacter,
          x, y, width, height,
          color: characterColors[colorIndex]
        });
        updatedPanels[panelIndex].characterBoxes = characterBoxes;
      } else if (isDrawingTextBox) {
        // Add the text box
        const textBoxes = [...(updatedPanels[panelIndex].textBoxes || [])];
        textBoxes.push({
          text: '',
          x, y, width, height
        });
        updatedPanels[panelIndex].textBoxes = textBoxes;
      }
      
      updatePanelsForCurrentPage(updatedPanels);
    }
    
    setDrawingStartPos(null);
  };
  
  // Handler for adding an action to the selected panel
  const handleAddAction = () => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Add a new action
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].actions.push({ text: '' });
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Handler for updating an action in the selected panel
  const handleUpdateAction = (index: number, value: string) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the action
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].actions[index].text = value;
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Handler for removing an action from the selected panel
  const handleRemoveAction = (index: number) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Remove the action
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].actions.splice(index, 1);
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Handler for updating the setting of the selected panel
  const handleUpdateSetting = (value: string) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the setting
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].setting = value;
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Handler for updating the prompt of the selected panel
  const handleUpdatePrompt = (value: string) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Update the prompt
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].prompt = value;
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Handler for generating a panel image
  const handleGeneratePanel = async () => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Mark the panel as generating
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].isGenerating = true;
    updatePanelsForCurrentPage(updatedPanels);
    
    try {
      // Prepare the data for the API
      const panel = panels[panelIndex];
      
      // Convert panel dimensions from model coordinates to absolute pixel values
      const pixelWidth = Math.round(panel.width);
      const pixelHeight = Math.round(panel.height);
      
      const apiData = {
        prompt: panel.prompt || '',
        setting: panel.setting || '',
        dialoguePositions: panel.dialoguePositions,
        characterPositions: panel.characterPositions,
        characterNames: panel.characterNames,
        dialogues: panel.dialogues,
        actions: panel.actions,
        characterBoxes: panel.characterBoxes || [], // Include character boxes
        textBoxes: panel.textBoxes || [],  
        panelIndex: panel.panelIndex || 0,
        seed: panel.seed || Math.floor(Math.random() * 1000000),
        width: pixelWidth,
        height: pixelHeight
      };
      
      // Call the API
      const response = await fetch(`${apiEndpoint}/generate_panel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(apiData)
      });
      
      const data = await response.json();
      
      // Check if the request was queued (models still initializing)
      if (data.status === 'queued') {
        // Show a message that the request is queued
        console.log(`Panel generation queued. Request ID: ${data.request_id}`);
        
        // Setup polling to check status
        const requestId = data.request_id;
        
        // Create an interval to check the status
        const checkInterval = setInterval(() => {
          checkPanelStatus(requestId, selectedPanelId);
        }, 1000); // Check every second
        
        // Save the request info
        setQueuedPanelRequests(prev => ({
          ...prev,
          [selectedPanelId]: {
            panelId: selectedPanelId,
            requestId: requestId,
            checkInterval
          }
        }));
        
        // Update the panel with "queued" status
        const newPanels = [...panels];
        newPanels[panelIndex] = {
          ...newPanels[panelIndex],
          isGenerating: true,
          generationQueued: true,
          queueMessage: data.message
        };
        
        updatePanelsForCurrentPage(newPanels);
        
      } else if (data.status === 'success') {
        // Handle immediate success (models were already initialized)
        handlePanelGenerationComplete(data, selectedPanelId);
      } else {
        // Handle error
        console.error('Error generating panel:', data.message);
        alert(`Error generating panel: ${data.message}`);
        
        // Mark the panel as not generating
        const newPanels = [...panels];
        newPanels[panelIndex].isGenerating = false;
        updatePanelsForCurrentPage(newPanels);
      }
    } catch (error) {
      console.error('Error calling API:', error);
      alert(`Error calling API: ${error}`);
      
      // Mark the panel as not generating
      const newPanels = [...panels];
      newPanels[panelIndex].isGenerating = false;
      updatePanelsForCurrentPage(newPanels);
    }
  };

  // Function to check status of queued panel request
  const checkPanelStatus = async (requestId: string, panelId: string) => {
    try {
      const response = await fetch(`${apiEndpoint}/check_panel_status?request_id=${requestId}`);
      const data = await response.json();
      
      if (data.status === 'success') {
        // Panel generation completed successfully
        handlePanelGenerationComplete(data, panelId);
        
        // Clean up the interval
        if (queuedPanelRequests[panelId]?.checkInterval) {
          clearInterval(queuedPanelRequests[panelId].checkInterval);
        }
        
        // Remove from queued requests
        setQueuedPanelRequests(prev => {
          const newRequests = {...prev};
          delete newRequests[panelId];
          return newRequests;
        });
        
      } else if (data.status === 'error') {
        // Handle error
        console.error('Error in queued panel generation:', data.message);
        alert(`Error generating panel: ${data.message}`);
        
        // Mark the panel as not generating
        const panelIndex = panels.findIndex(p => p.id === panelId);
        if (panelIndex !== -1) {
          const newPanels = [...panels];
          newPanels[panelIndex].isGenerating = false;
          newPanels[panelIndex].generationQueued = false;
          updatePanelsForCurrentPage(newPanels);
        }
        
        // Clean up the interval
        if (queuedPanelRequests[panelId]?.checkInterval) {
          clearInterval(queuedPanelRequests[panelId].checkInterval);
        }
        
        // Remove from queued requests
        setQueuedPanelRequests(prev => {
          const newRequests = {...prev};
          delete newRequests[panelId];
          return newRequests;
        });
      }
      // If status is 'processing', continue polling
    } catch (error) {
      console.error('Error checking panel status:', error);
    }
  };
  
  // Handle successful panel generation completion
  const handlePanelGenerationComplete = (data: any, panelId: string) => {
    // Find the panel
    const panelIndex = panels.findIndex(p => p.id === panelId);
    if (panelIndex === -1) return;
    
    // Update the panel with the generated image
    const newPanels = [...panels];
    newPanels[panelIndex] = {
      ...newPanels[panelIndex],
      imageData: data.imageData,
      prompt: data.prompt || newPanels[panelIndex].prompt,
      isGenerating: false,
      generationQueued: false,
      seed: data.seed // Store the used seed
    };
    
    updatePanelsForCurrentPage(newPanels);
  };
  
  // Handler for saving the page
  const handleSavePage = async () => {
    // TODO: Implement page saving
    alert('Page saving not implemented yet');
  };

  // Page navigation functions
  const handlePrevPage = () => {
    if (currentPageIndex > 0) {
      setSelectedPanelId(null); // Clear selection when switching pages
      setCurrentPageIndex(currentPageIndex - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPageIndex < pages.length - 1) {
      setSelectedPanelId(null); // Clear selection when switching pages
      setCurrentPageIndex(currentPageIndex + 1);
    }
  };

  // Add this function to create default panels for a new page
  const createDefaultPanelsForPage = () => {
    const gap = 20;
    const cols = 2;
    const rows = 3;
    
    const panelWidth = (pageSize.width - (gap * (cols + 1))) / cols;
    const panelHeight = (pageSize.height - (gap * (rows + 1))) / rows;
    
    const defaultPanels = [];
    
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        defaultPanels.push({
          id: `panel-page-${pages.length}-${row * cols + col}`,
          x: gap + col * (panelWidth + gap),
          y: gap + row * (panelHeight + gap),
          width: panelWidth,
          height: panelHeight,
          characterNames: [],
          characterPositions: [],
          dialogues: [],
          dialoguePositions: [],
          actions: [],
          panelIndex: row * cols + col,
          characterBoxes: [], // Add this for character boxes
          textBoxes: []  
        });
      }
    }
    
    return defaultPanels;
  };


  const handleAddPage = () => {
    const newPageIndex = pages.length;
    const newPage = {
      id: `page-${newPageIndex + 1}`,
      panels: createDefaultPanelsForPage()
    };
    
    setPages([...pages, newPage]);
    setCurrentPageIndex(newPageIndex);
    setSelectedPanelId(null); // Clear selection when switching pages
  };


  const handleSaveSinglePage = async () => {
    if (!stageRef.current) return;
    
    const dataURL = stageRef.current.toDataURL();
    saveAs(dataURL, `manga-page-${currentPageIndex + 1}.png`);
  };

  // Add function to save all pages
  const handleSaveAllPages = async () => {
    // Show loading indicator
    setIsSaving(true);
    
    try {
      // Create a zip file with all pages
      const JSZip = (await import('jszip')).default;
      const zip = new JSZip();
      
      // Process each page
      for (let i = 0; i < pages.length; i++) {
        // Save current page index
        const currentPage = currentPageIndex;
        
        // Temporarily switch to the page we want to save
        setCurrentPageIndex(i);
        
        // Need to wait for the stage to update
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Get the data URL and add to zip
        const dataURL = stageRef.current.toDataURL();
        const base64Data = dataURL.split(',')[1];
        zip.file(`page-${i + 1}.png`, base64Data, {base64: true});
        
        // Switch back to original page
        setCurrentPageIndex(currentPage);
      }
      
      // Generate and save the zip file
      const content = await zip.generateAsync({type: 'blob'});
      saveAs(content, 'manga-pages.zip');
    } catch (error) {
      console.error('Error saving pages:', error);
      alert('Error saving pages: ' + error);
    } finally {
      setIsSaving(false);
    }
  };

  const applyPageTemplate = (templateId: string) => {
    const template = pageTemplates.find(t => t.id === templateId);
    if (!template) return;
    
    const gap = 20;
    const panelTemplates = template.layoutFunction(pageSize.width, pageSize.height, gap);
    
    // Create panels from the template
    const newPanels = panelTemplates.map((template, index) => ({
      id: `panel-page-${currentPageIndex}-${index}`,
      x: template.x,
      y: template.y,
      width: template.width,
      height: template.height,
      characterNames: [],
      characterPositions: [],
      dialogues: [],
      dialoguePositions: [],
      actions: [],
      panelIndex: index
    }));
    
    // Update the current page
    const updatedPages = [...pages];
    updatedPages[currentPageIndex].panels = newPanels;
    setPages(updatedPages);
    setSelectedPanelId(null);
    setShowTemplateDialog(false);
  };

  {/* const exportManga = async () => {
    if (!currentProject || !pages.length) {
      alert('No project or pages to export');
      return;
    }
    
    setIsExporting(true);
    
    try {
      // If using local browser-only export
      if (!apiEndpoint) {
        // Create a zip file using JSZip
        const JSZip = (await import('jszip')).default;
        const zip = new JSZip();
        
        // Add metadata
        zip.file('metadata.json', JSON.stringify({
          name: currentProject.name,
          created: currentProject.created,
          modified: new Date().toISOString(),
          pageCount: pages.length,
          characters: currentProject.characters || []
        }));
        
        // Add pages
        const pagesFolder = zip.folder('pages');
        
        // For each page, take a screenshot and add to zip
        for (let i = 0; i < pages.length; i++) {
          setCurrentPageIndex(i);
          
          // Wait for the page to render
          await new Promise(resolve => setTimeout(resolve, 100));
          
          // Take a screenshot using the stage ref
          if (stageRef.current) {
            const dataURL = stageRef.current.toDataURL({
              pixelRatio: 2 // Higher quality
            });
            
            // Convert data URL to blob
            const imageData = dataURL.split(',')[1];
            const binaryData = atob(imageData);
            const array = new Uint8Array(binaryData.length);
            
            for (let j = 0; j < binaryData.length; j++) {
              array[j] = binaryData.charCodeAt(j);
            }
            
            // Add to zip
            pagesFolder.file(`page-${i+1}.png`, array);
          }
        }
        
        // Generate the zip file
        const content = await zip.generateAsync({type: 'blob'});
        
        // Trigger download
        saveAs(content, `${currentProject.name.replace(/\s+/g, '-')}.zip`);
      } else {
        // If using backend export API
        const response = await fetch(`${apiEndpoint}/export/${currentProject.id}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            pages,
            characters: currentProject.characters || []
          })
        });
        
        if (!response.ok) {
          throw new Error(`Export failed: ${response.status}`);
        }
        
        // Assuming backend returns a URL to download the zip
        const { downloadUrl } = await response.json();
        
        // Trigger download
        window.location.href = downloadUrl;
      }
      
      // Show success message
      setStatusMessage('Export completed successfully');
      setTimeout(() => setStatusMessage(''), 3000);
    } catch (error) {
      console.error('Export error:', error);
      setStatusMessage('Export failed: ' + error.message);
      setTimeout(() => setStatusMessage(''), 3000);
    } finally {
      setIsExporting(false);
    }
  }; */}

  const handleBackgroundClick = (e: KonvaEventObject<MouseEvent>) => {
    // If clicking directly on the stage (background), deselect current panel
    // Check that the target is the stage itself
    if (e.target === e.currentTarget) {
      setSelectedPanelId(null);
    }
  };
  
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Left Sidebar - Vertical Toolbar */}
      <div className="w-14 bg-gray-100 shadow-md flex flex-col items-center py-4 space-y-4">
        <button
          className="p-2 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200"
          onClick={handleAddPanel}
          title="Add Panel"
        >
          <Plus size={20} />
        </button>
        
        <button
          className="p-2 bg-indigo-100 text-red-600 rounded-md hover:bg-indigo-200"
          onClick={handleDeletePanel}
          disabled={!selectedPanelId}
          title="Delete Panel"
        >
          <Trash2 size={20} />
        </button>
        
        <button
          className="p-2 bg-indigo-100 text-green-600 rounded-md hover:bg-indigo-200"
          onClick={handleSavePage}
          title="Save Page"
        >
          <Save size={20} />
        </button>
        
        <button
          className="p-2 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200"
          onClick={() => setShowTemplateDialog(true)}
          title="Page Templates"
        >
          <Layout size={20} />
        </button>
        
        <button
          className="p-2 bg-indigo-100 text-green-600 rounded-md hover:bg-indigo-200"
          onClick={handleSaveSinglePage}
          title="Export Current Page"
        >
          <Download size={20} />
        </button>
        
        <button
          className="p-2 bg-indigo-100 text-green-600 rounded-md hover:bg-indigo-200"
          onClick={handleSaveAllPages}
          disabled={isSaving}
          title="Export All Pages"
        >
          <FileDown size={20} />
        </button>
        
        <button
          onClick={() => setShowProjectManager(true)}
          className="p-2 bg-indigo-100 text-blue-600 rounded-md hover:bg-indigo-200"
          title="Project Manager"
        >
          <Folder size={20} />
        </button>
      </div>
  
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Controls */}
        <div className="bg-white shadow-sm p-2 flex items-center space-x-4">
          {/* Page Navigation Controls */}
          <div className="flex items-center space-x-1 border rounded-md px-1">
            <button
              onClick={handlePrevPage}
              disabled={currentPageIndex === 0}
              className="p-1 rounded hover:bg-indigo-100 text-gray-800 disabled:text-gray-400"
              title="Previous Page"
            >
              <ChevronLeft size={18} />
            </button>
            
            <span className="text-sm font-medium text-gray-800 px-2">
              {currentPageIndex + 1}/{pages.length}
            </span>
            
            <button
              onClick={handleNextPage}
              disabled={currentPageIndex >= pages.length - 1}
              className="p-1 rounded hover:bg-indigo-100 text-gray-800 disabled:text-gray-400"
              title="Next Page"
            >
              <ChevronRight size={18} />
            </button>
          </div>
          
          <button
            onClick={handleAddPage}
            className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200"
            title="Add Page"
          >
            <div className="flex items-center">
              <Plus size={16} className="mr-1" />
              <span>Page</span>
            </div>
          </button>
          
          {/* Snapping Toggle */}
          <div className="flex items-center">
            <label htmlFor="snapping-toggle" className="inline-flex items-center cursor-pointer">
              <span className="mr-3 text-sm font-medium text-gray-900">Snapping</span>
              <div className="relative">
                <input 
                  id="snapping-toggle" 
                  type="checkbox" 
                  checked={isSnappingEnabled}
                  onChange={(e) => setIsSnappingEnabled(e.target.checked)}
                  className="sr-only peer" 
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
              </div>
            </label>
          </div>
          
          {/* Zoom Controls */}
          <div className="flex items-center space-x-1 bg-gray-100 rounded-lg p-1 ml-auto">
            <button
              onClick={handleZoomOut}
              className="p-1 rounded hover:bg-indigo-100 text-gray-800"
              title="Zoom out"
            >
              <ZoomOut size={16} />
            </button>
            
            <button
              onClick={handleZoomReset}
              className="px-2 py-1 rounded hover:bg-indigo-100 text-gray-800 text-sm font-medium"
              title="Reset zoom"
            >
              {Math.round(scale * 100)}%
            </button>
            
            <button
              onClick={handleZoomIn}
              className="p-1 rounded hover:bg-indigo-100 text-gray-800"
              title="Zoom in"
            >
              <ZoomIn size={16} />
            </button>
          </div>
        </div>
  
        {/* Canvas and Panel Editor */}
        <div className="flex-1 flex overflow-hidden">
          {/* Canvas Area */}
          <div className="flex-1 bg-white overflow-auto p-4 flex items-center justify-center">
            <div className="relative" style={{ width: `${pageSize.width * scale}px`, height: `${pageSize.height * scale}px` }}>
              <Stage 
                width={pageSize.width * scale} 
                height={pageSize.height * scale}
                ref={stageRef}
                className="bg-gray-100 shadow-inner border border-gray-300"
                onClick={handleBackgroundClick}
                onMouseDown={handleStageMouseDown}
                onMouseMove={handleStageMouseMove}
                onMouseUp={handleStageMouseUp}
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
                      onDragMove={handleDragMove}
                      onDragEnd={handleDragEnd}
                    />
                  ))}
  
                  {panels.map(panel => (
                    <React.Fragment key={`boxes-${panel.id}`}>
                      {/* Character boxes */}
                      {panel.characterBoxes?.map((box, index) => (
                        <Rect
                          key={`char-box-${panel.id}-${index}`}
                          x={(panel.x + box.x * panel.width) * scale}
                          y={(panel.y + box.y * panel.height) * scale}
                          width={box.width * panel.width * scale}
                          height={box.height * panel.height * scale}
                          stroke={box.color}
                          strokeWidth={2}
                          dash={[5, 5]}
                          fill={box.color + '33'} // Add transparency
                        />
                      ))}
                      
                      {/* Text boxes */}
                      {panel.textBoxes?.map((box, index) => (
                        <Rect
                          key={`text-box-${panel.id}-${index}`}
                          x={(panel.x + box.x * panel.width) * scale}
                          y={(panel.y + box.y * panel.height) * scale}
                          width={box.width * panel.width * scale}
                          height={box.height * panel.height * scale}
                          stroke="#000000"
                          strokeWidth={2}
                          dash={[5, 5]}
                          fill="#FFFFFF88" // Semi-transparent white
                        />
                      ))}
                    </React.Fragment>
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
                    // Enable all anchors for full resizing control
                    enabledAnchors={[
                      'top-left', 'top-center', 'top-right',
                      'middle-left', 'middle-right',
                      'bottom-left', 'bottom-center', 'bottom-right'
                    ]}
                    rotateEnabled={false}
                  />
                  {/* Guide lines */}
                  {showGuides && guides.x.map((x, i) => (
                    <Line 
                      key={`x-${i}`}
                      points={[x, 0, x, pageSize.height * scale]}
                      stroke="#0066FF"
                      strokeWidth={1}
                      dash={[5, 5]}
                    />
                  ))}
                  {showGuides && guides.y.map((y, i) => (
                    <Line 
                      key={`y-${i}`}
                      points={[0, y, pageSize.width * scale, y]}
                      stroke="#0066FF"
                      strokeWidth={1}
                      dash={[5, 5]}
                    />
                  ))}
                </Layer>
              </Stage>
  
              {/* Loading overlay for panel generation */}
              {panels.map(panel => (
                panel.isGenerating && (
                  <div 
                    key={`overlay-${panel.id}`}
                    className="absolute bg-black bg-opacity-50 flex justify-center items-center"
                    style={{
                      left: panel.x * scale,
                      top: panel.y * scale,
                      width: panel.width * scale,
                      height: panel.height * scale
                    }}
                  >
                    <div className="text-white text-lg">
                      {panel.generationQueued 
                        ? `Queued: ${panel.queueMessage || 'Waiting...'}`
                        : 'Generating...'}
                    </div>
                  </div>
                )
              ))}
            </div>
          </div>
  
          {/* Panel Editor - Increased width */}
          <div className="w-96 bg-white border-l border-gray-200 overflow-y-auto" style={{ maxHeight: '100vh' }}>
            {selectedPanel ? (
              <div className="p-4">
                <h2 className="text-2xl font-bold mb-4">Panel Editor</h2>
                
                <div className="space-y-6">
                  {/* Box Drawing Tools - Moved from below canvas to Panel Editor */}
                  <div className="pb-4 border-b border-gray-200">
                    <h3 className="text-lg font-semibold mb-2">Box Drawing Tools</h3>
                    <div className="flex flex-wrap gap-2 mb-2">
                      <div className="flex flex-col">
                        <button
                          className={`px-3 py-2 rounded-md flex items-center ${isDrawingCharacterBox ? 'bg-indigo-600 text-white' : 'bg-indigo-100 text-indigo-700'}`}
                          onClick={() => {
                            setIsDrawingCharacterBox(!isDrawingCharacterBox);
                            setIsDrawingTextBox(false);
                          }}
                        >
                          <Square size={16} className="mr-2" />
                          Character Box
                        </button>
                        {isDrawingCharacterBox && (
                          <select
                            className="mt-2 px-2 py-1 border rounded"
                            value={activeCharacter || ''}
                            onChange={e => setActiveCharacter(e.target.value)}
                          >
                            <option value="">Select Character</option>
                            {selectedPanel.characterNames.map(name => (
                              <option key={name} value={name}>{name}</option>
                            ))}
                          </select>
                        )}
                      </div>
                      
                      <button
                        className={`px-3 py-2 rounded-md flex items-center ${isDrawingTextBox ? 'bg-indigo-600 text-white' : 'bg-indigo-100 text-indigo-700'}`}
                        onClick={() => {
                          setIsDrawingTextBox(!isDrawingTextBox);
                          setIsDrawingCharacterBox(false);
                        }}
                      >
                        <MessageSquare size={16} className="mr-2" />
                        Text Box
                      </button>
                    </div>
                    
                    {/* Display boxes for this panel */}
                    {selectedPanel && selectedPanel.characterBoxes && selectedPanel.characterBoxes.length > 0 && (
                      <div className="mt-3">
                        <h4 className="font-medium text-sm mb-1">Character Boxes:</h4>
                        <div className="max-h-32 overflow-y-auto">
                          {selectedPanel.characterBoxes.map((box, idx) => (
                            <div key={idx} className="flex items-center text-sm mb-1">
                              <div className="w-3 h-3 mr-2" style={{backgroundColor: box.color}}></div>
                              <span>{box.character}</span>
                              <button
                                className="ml-auto text-red-500 hover:text-red-700"
                                onClick={() => {
                                  const updatedPanels = [...panels];
                                  const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
                                  if (panelIndex !== -1 && updatedPanels[panelIndex].characterBoxes) {
                                    updatedPanels[panelIndex].characterBoxes = updatedPanels[panelIndex].characterBoxes.filter((_, i) => i !== idx);
                                    updatePanelsForCurrentPage(updatedPanels);
                                  }
                                }}
                              >
                                Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
  
                    {selectedPanel && selectedPanel.textBoxes && selectedPanel.textBoxes.length > 0 && (
                      <div className="mt-3">
                        <h4 className="font-medium text-sm mb-1">Text Boxes:</h4>
                        <div className="max-h-32 overflow-y-auto">
                          {selectedPanel.textBoxes.map((box, idx) => (
                            <div key={idx} className="flex items-center text-sm mb-1">
                              <span>Text Box {idx + 1}</span>
                              <button
                                className="ml-auto text-red-500 hover:text-red-700"
                                onClick={() => {
                                  const updatedPanels = [...panels];
                                  const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
                                  if (panelIndex !== -1 && updatedPanels[panelIndex].textBoxes) {
                                    updatedPanels[panelIndex].textBoxes = updatedPanels[panelIndex].textBoxes.filter((_, i) => i !== idx);
                                    updatePanelsForCurrentPage(updatedPanels);
                                  }
                                }}
                              >
                                Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
  
                  <div className="pb-4 border-b border-gray-200">
                    <h3 className="text-lg font-semibold mb-2">Panel Settings</h3>
                    
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <label className="block text-sm font-medium text-black mb-1">Panel Index</label>
                        <input
                          type="number"
                          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                          value={selectedPanel.panelIndex || 0}
                          onChange={(e) => {
                            const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
                            if (panelIndex === -1) return;
                            
                            const updatedPanels = [...panels];
                            updatedPanels[panelIndex].panelIndex = parseInt(e.target.value) || 0;
                            
                            updatePanelsForCurrentPage(updatedPanels);
                          }}
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-black mb-1">Seed</label>
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
                              
                              updatePanelsForCurrentPage(updatedPanels);
                            }}
                          />
                          <button
                            className="px-3 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                            onClick={() => {
                              const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
                              if (panelIndex === -1) return;
                              
                              const updatedPanels = [...panels];
                              updatedPanels[panelIndex].seed = Math.floor(Math.random() * 1000000);
                              
                              updatePanelsForCurrentPage(updatedPanels);
                            }}
                          >
                            Random
                          </button>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-black mb-1">Setting</label>
                      <input
                        type="text"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        value={selectedPanel.setting || ''}
                        onChange={(e) => handleUpdateSetting(e.target.value)}
                        placeholder="e.g., INT. HOTEL LOBBY - DAY, manga panel"
                      />
                    </div>
                    
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-black mb-1">Custom Prompt (optional)</label>
                      <textarea
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        value={selectedPanel.prompt || ''}
                        onChange={(e) => handleUpdatePrompt(e.target.value)}
                        placeholder="Leave blank to auto-generate from panel elements"
                        rows={3}
                      />
                    </div>
                  </div>
                  
                  <div className="pb-4 border-b border-gray-200">
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
                        <div className="text-black italic">No characters added</div>
                      )}
                    </div>
                  </div>
                  
                  <div className="pb-4 border-b border-gray-200">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="text-lg font-semibold">Dialogues</h3>
                      <button
                        className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200"
                        onClick={handleAddDialogue}
                      >
                        Add Dialogue
                      </button>
                    </div>
                    
                    <div className="space-y-4">
                      {selectedPanel.dialogues.map((dialogue, index) => (
                        <div key={`dialogue-${index}`} className="p-3 bg-gray-50 rounded-md border border-gray-200">
                          <div className="mb-2">
                            <label className="block text-sm font-medium text-black mb-1">Character</label>
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
                            <label className="block text-sm font-medium text-black mb-1">Text</label>
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
                        <div className="text-black italic">No dialogues added</div>
                      )}
                    </div>
                  </div>
                  
                  <div className="pb-4 border-b border-gray-200">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="text-lg font-semibold">Actions</h3>
                      <button
                        className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200"
                        onClick={handleAddAction}
                      >
                        Add Action
                      </button>
                    </div>
                    
                    <div className="space-y-4">
                      {selectedPanel.actions.map((action, index) => (
                        <div key={`action-${index}`} className="p-3 bg-gray-50 rounded-md border border-gray-200">
                          <div className="mb-2">
                            <label className="block text-sm font-medium text-black mb-1">Text</label>
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
                        <div className="text-black italic">No actions added</div>
                      )}
                    </div>
                  </div>
                  
                  <button
                    className="w-full px-4 py-3 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
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
              <div className="flex flex-col items-center justify-center h-64 text-black p-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p>Select a panel to edit</p>
              </div>
            )}
          </div>
        </div>
      </div>
  
      {/* Template Dialog Modal */}
      {showTemplateDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl p-6 max-h-[90vh] overflow-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-bold">Choose a Page Template</h3>
              <button
                onClick={() => setShowTemplateDialog(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {pageTemplates.map(template => (
                <div 
                  key={template.id}
                  className="border rounded-lg p-4 cursor-pointer hover:bg-indigo-50"
                  onClick={() => applyPageTemplate(template.id)}
                >
                  <h4 className="font-bold mb-1">{template.name}</h4>
                  <p className="text-sm text-gray-600 mb-2">{template.description}</p>
                  <div className="bg-gray-100 border aspect-[2/3] p-1">
                    <svg viewBox={`0 0 ${pageSize.width} ${pageSize.height}`} className="w-full h-full">
                      {template.layoutFunction(pageSize.width, pageSize.height, 20).map((panel, i) => (
                        <rect
                          key={i}
                          x={panel.x}
                          y={panel.y}
                          width={panel.width}
                          height={panel.height}
                          fill="white"
                          stroke="black"
                          strokeWidth="2"
                        />
                      ))}
                    </svg>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MangaEditor;