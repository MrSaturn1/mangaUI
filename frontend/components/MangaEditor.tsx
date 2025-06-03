// components/MangaEditor.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Stage, Layer, Rect, Image as KonvaImage, Transformer, Line, Group } from 'react-konva';
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
  MessageSquare,
  X
} from 'lucide-react';
import ModeStatusBar from './ModeStatusBar';
import PanelAdjustmentHandles from './PanelAdjustmentHandles';
import { API_ENDPOINT, normalizeImagePath } from '../config';


export interface Character {
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
  imagePath?: string;
  imageData?: string;
  prompt?: string;
  setting?: string;
  seed?: number;
  panelIndex?: number;
  isGenerating?: boolean;
  generationQueued?: boolean;
  queueMessage?: string;
  
  // Characters should remain as separate arrays as they might be 
  // needed independently by the character management system
  characterNames: string[];  // Keep for character identification
  
  // Character boxes for positioning with drawing capability
  characterBoxes?: {
    character: string;    // Must match a name in characterNames
    x: number;           // Relative coordinates (0-1)
    y: number;
    width: number;
    height: number;
    color: string;       // For UI display
  }[];
  
  // Text boxes for dialogue
  textBoxes?: {
    text: string;        // The content of the text/dialogue
    x: number;           // Relative coordinates (0-1)
    y: number;
    width: number;
    height: number;
  }[];
  
  // Keep dialogues as they provide structure for character attribution
  dialogues: DialogueItem[];  // {character: string, text: string}
  
  // Actions for scene descriptions
  actions: ActionItem[];      // {text: string}
}

export interface Page {
  id: string;
  panels: Panel[]; // This explicitly tells TypeScript that panels is an array of Panel
}

export interface Project {
  id: string;
  name: string;
  pages: number;
  lastModified: string;
}

interface MangaEditorProps {
  characters: Character[];
  apiEndpoint?: string;
  currentProject?: Project | null; 
  setCurrentProject?: (project: any) => void; // Add this
  pages?: Page[]; // Add this
  setPages?: (pages: Page[]) => void; // Add this
  onSaveProject?: (projectId: string, pages: Page[]) => void; // Add this
  onShowProjectManager?: () => void; // Add this
}

// Then update the function declaration to use these props
const MangaEditor: React.FC<MangaEditorProps> = ({ 
  characters, 
  apiEndpoint = API_ENDPOINT, 
  currentProject,
  setCurrentProject,
  pages = [],
  setPages,
  onSaveProject,
  onShowProjectManager
}) => {
  // Page state
  const [pageSize, setPageSize] = useState({ width: 1500, height: 2250 }); // A4 proportions
  const [pageIndex, setPageIndex] = useState<number>(0);
  const [scale, setScale] = useState<number>(0.3); // Scale for the canvas
  // Use a local copy of pages to avoid TypeScript errors with setPages
  const [localPages, setLocalPages] = useState<Page[]>([]);
  // const [pages, setPages] = useState([{ id: 'page-1', panels: [] as Panel[] }]);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [showTemplateDialog, setShowTemplateDialog] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  
  // Panels state
  const panels: Panel[] = localPages[currentPageIndex]?.panels || [];
  const [selectedPanelId, setSelectedPanelId] = useState<string | null>(null);
  // State for tracking queued panel requests
  const [queuedPanelRequests, setQueuedPanelRequests] = useState<{[key: string]: {
    panelId: string;
    requestId: string;
    checkInterval: NodeJS.Timeout | null;
  }}>({});
  // Panel, character, and text box modes and previews
  const [panelMode, setPanelMode] = useState<'adjust' | 'character-box' | 'text-box'>('adjust');
  const [previewBox, setPreviewBox] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
    character?: string;
    color?: string;
  } | null>(null);
  // Panel Focus Mode
  const [isPanelFocusMode, setIsPanelFocusMode] = useState<boolean>(false);
  const [focusedPanelId, setFocusedPanelId] = useState<string | null>(null);
  const [focusScale, setFocusScale] = useState<number>(1.0); // Scale for focused panel
  const focusedPanel = isPanelFocusMode && focusedPanelId ? 
    panels.find(p => p.id === focusedPanelId) : null;
  const currentScale = isPanelFocusMode ? focusScale : scale;

  // Character and Text Boxes
  const [isDrawingCharacterBox, setIsDrawingCharacterBox] = useState<boolean>(false);
  const [isDrawingTextBox, setIsDrawingTextBox] = useState<boolean>(false);
  const [activeCharacter, setActiveCharacter] = useState<string | null>(null);
  const [drawingStartPos, setDrawingStartPos] = useState<{x: number, y: number} | null>(null);
  const [selectedBoxType, setSelectedBoxType] = useState<'character' | 'text' | null>(null);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);
  const transformerBoxRef = useRef<any>(null);

  // Refs
  const stageRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  
  // Get the selected panel
  const selectedPanel = panels.find(p => p.id === selectedPanelId);

  // Project Management
  const [showProjectManager, setShowProjectManager] = useState<boolean>(false);
  // Auto-save functionality
  // Add state for tracking changes
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState<boolean>(false);
  const [lastSaveTime, setLastSaveTime] = useState<Date | null>(null);

  // Helper lines and snapping
  const [guides, setGuides] = useState<{x: number[], y: number[]}>({x: [], y: []});
  const [showGuides, setShowGuides] = useState(false);
  const [snapThreshold, setSnapThreshold] = useState<number>(10); // In pixels, adjust as needed
  const [isSnappingEnabled, setIsSnappingEnabled] = useState<boolean>(true);

  // Project export
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [exportProgress, setExportProgress] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [showExportDialog, setShowExportDialog] = useState<boolean>(false);
  const [exportConfig, setExportConfig] = useState<{
    type: 'png' | 'pdf';
    pageRange: 'all' | 'current' | 'custom';
    customRange: string;
    quality: 'normal' | 'high';
  }>({
    type: 'png',
    pageRange: 'all',
    customRange: '',
    quality: 'normal'
  });

  // Status Indicators
  const [statusType, setStatusType] = useState<'success' | 'error' | 'info' | 'loading'>('info');
  const [showStatus, setShowStatus] = useState<boolean>(false);
  const [statusTimeout, setStatusTimeout] = useState<NodeJS.Timeout | null>(null);

  

  // Initialize localPages from the props pages and sync them
  useEffect(() => {
    // Initialize localPages with pages from props if they exist
    if (pages.length > 0) {
      setLocalPages(pages);
    } else if (localPages.length === 0) {
      // Initialize with a default page if no pages exist yet
      setLocalPages([{ id: 'page-1', panels: [] }]);
    }
  }, []);
  
  // Sync props pages to localPages when they change externally
  useEffect(() => {
    // Only update if the pages prop has changed and is different from localPages
    if (pages.length > 0 && JSON.stringify(pages) !== JSON.stringify(localPages)) {
      setLocalPages(pages);
    }
  }, [pages]);
  
  // Sync localPages back to parent component
  useEffect(() => {
    // Only update if setPages exists and localPages has been initialized
    if (setPages && localPages.length > 0) {
      setPages(localPages);
    }
  }, [localPages, setPages]);
  
  // Effect to add some default panels on first load
  useEffect(() => {
    // Check if the current page has no panels
    if (localPages[currentPageIndex]?.panels?.length === 0) {
      const defaultPanels = createDefaultPanelsForPage();
      const updatedPages = [...localPages];
      
      // Ensure the page exists
      if (!updatedPages[currentPageIndex]) {
        updatedPages[currentPageIndex] = { id: `page-${currentPageIndex + 1}`, panels: [] };
      }
      
      updatedPages[currentPageIndex].panels = defaultPanels;
      setLocalPages(updatedPages);
    }
  }, [currentPageIndex, localPages]);
  
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

  // Add this effect to handle transformer for selected boxes
  useEffect(() => {
    if (selectedBoxType && selectedBoxIndex !== null && transformerBoxRef.current && stageRef.current) {
      // Always show transformer when a box is selected, regardless of mode
      const panelBoxes = stageRef.current.findOne(`#panel-boxes-${selectedPanelId}`);
      if (panelBoxes) {
        const node = panelBoxes.findOne(`.${selectedBoxType}-box-${selectedBoxIndex}`);
        if (node) {
          transformerBoxRef.current.nodes([node]);
          transformerBoxRef.current.getLayer().batchDraw();
        }
      }
    } else if (transformerBoxRef.current) {
      transformerBoxRef.current.nodes([]);
      transformerBoxRef.current.getLayer().batchDraw();
    }
  }, [selectedBoxType, selectedBoxIndex, selectedPanelId]);

  // Add keyboard handling for delete key when a box is selected
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && 
          selectedBoxType && 
          selectedBoxIndex !== null) {
        handleDeleteBox(selectedBoxType, selectedBoxIndex);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedBoxType, selectedBoxIndex]);

  // Add these helper functions to handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Handle Escape key to cancel current drawing mode
      if (e.key === 'Escape') {
        if (panelMode !== 'adjust') {
          setPanelMode('adjust');
          setPreviewBox(null);
          setDrawingStartPos(null);
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [panelMode]);

  // Add useEffect to load images when the current page changes
  useEffect(() => {
    // Load images for current page
    loadPageImages(currentPageIndex);
    
    // Optional: Preload images for adjacent pages for smoother navigation
    if (currentPageIndex > 0) {
      setTimeout(() => loadPageImages(currentPageIndex - 1), 500);
    }
    if (currentPageIndex < localPages.length - 1) {
      setTimeout(() => loadPageImages(currentPageIndex + 1), 500);
    }
  }, [currentPageIndex]);

  // Update useEffect for localPages to track changes
  useEffect(() => {
    // Only track changes once pages are loaded
    if (localPages.length > 0) {
      setHasUnsavedChanges(true);
    }
  }, [localPages]);

  // Add auto-save effect
  useEffect(() => {
    // Auto-save every 1 minute if there are unsaved changes
    const autoSaveInterval = setInterval(() => {
      if (hasUnsavedChanges && currentProject) {
        console.log('Auto-saving project...');
        saveToProject();
        setHasUnsavedChanges(false);
        setLastSaveTime(new Date());
      }
    }, 60000); // 1 minute
    
    // Clean up
    return () => clearInterval(autoSaveInterval);
  }, [hasUnsavedChanges, currentProject, localPages]);

  // Add keyboard shortcut to exit focus mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape to exit focus mode
      if (e.key === 'Escape' && isPanelFocusMode) {
        exitPanelFocusMode();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isPanelFocusMode]);

  // Add beforeunload handler to warn about unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (hasUnsavedChanges) {
        // Standard way to show a confirmation dialog when closing/reloading
        e.preventDefault();
        e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
        return e.returnValue;
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [hasUnsavedChanges]);

  // Add useEffect to handle memory cleanup when component unmounts
  useEffect(() => {
    return () => {
      // Clear all image data from memory
      setLocalPages(prevPages => 
        prevPages.map(page => ({
          ...page,
          panels: page.panels.map(panel => ({
            ...panel,
            imageData: undefined
          }))
        }))
      );
    };
  }, []);

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
    // Clear box selection when switching panels
    setSelectedBoxType(null);
    setSelectedBoxIndex(null);
  };

  const updatePanelsForCurrentPage = (newPanels: Panel[]) => {
    const updatedPages = [...localPages];
    
    // Ensure the page exists
    if (!updatedPages[currentPageIndex]) {
      updatedPages[currentPageIndex] = { id: `page-${currentPageIndex + 1}`, panels: [] };
    }
    
    updatedPages[currentPageIndex] = {
      ...updatedPages[currentPageIndex],
      panels: newPanels
    };
    
    setLocalPages(updatedPages);
  };

  const loadPageImages = async (pageIndex: number) => {
    if (pageIndex < 0 || pageIndex >= localPages.length) return;
    
    const page = localPages[pageIndex];
    if (!page || !page.panels || page.panels.length === 0) return;
    
    const panelsNeedingImages = page.panels.filter(panel => !panel.imageData && panel.imagePath);
    if (panelsNeedingImages.length === 0) return;
    
    const updatedPanels = [...page.panels];
    const loadingPromises = [];
    
    // For each panel with imagePath but no imageData, load the image
    for (let i = 0; i < updatedPanels.length; i++) {
      const panel = updatedPanels[i];
      
      // Skip if panel already has imageData or doesn't have an imagePath
      if (panel.imageData || !panel.imagePath) continue;
      
      // Create a promise for loading this panel's image
      const loadPromise = (async () => {
        try {
          const normalizedPath = normalizeImagePath(panel.imagePath);
          // Fetch the image
          const response = await fetch(normalizedPath!);
          
          if (!response.ok) {
            console.error(`Failed to load image for panel ${panel.id}: ${response.statusText}`);
            return false;
          }
          
          const blob = await response.blob();
          const base64Data = await blobToBase64(blob);
          
          // Update panel with image data
          updatedPanels[i] = {
            ...updatedPanels[i],
            imageData: base64Data
          };
          
          return true;
        } catch (error) {
          console.error(`Error loading image for panel ${panel.id}:`, error);
          return false;
        }
      })();
      
      loadingPromises.push(loadPromise);
    }
    
    // Wait for all images to load
    await Promise.all(loadingPromises);
    
    // Update state with all loaded images
    const updatedPages = [...localPages];
    updatedPages[pageIndex] = {
      ...updatedPages[pageIndex],
      panels: updatedPanels
    };
    
    setLocalPages(updatedPages);
  };
  
  // Helper function to convert blob to base64
  const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };
  
  // Function to clear image data for a specific page
  const clearPageImages = (pageIndex: number) => {
    if (pageIndex < 0 || pageIndex >= localPages.length) return;
    
    const page = localPages[pageIndex];
    if (!page || !page.panels || page.panels.length === 0) return;
    
    const updatedPanels = page.panels.map(panel => ({
      ...panel,
      imageData: undefined // Clear the image data but keep the path
    }));
    
    const updatedPages = [...localPages];
    updatedPages[pageIndex] = {
      ...updatedPages[pageIndex],
      panels: updatedPanels
    };
    
    setLocalPages(updatedPages);
  };

  const handlePageChange = (newPageIndex: number) => {
    // Clear image data from the current page to free memory
    const updatedPages = [...localPages];
    
    if (currentPageIndex >= 0 && currentPageIndex < updatedPages.length) {
      updatedPages[currentPageIndex].panels = updatedPages[currentPageIndex].panels.map(panel => ({
        ...panel,
        imageData: undefined // Clear the image data
      }));
    }
    
    setLocalPages(updatedPages);
    setCurrentPageIndex(newPageIndex);
    
    // Load images for the new page
    loadPageImages(newPageIndex);
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
      dialogues: [],
      actions: [],
      panelIndex: panels.length,
      // Initialize empty arrays for our box fields
      characterBoxes: [],
      textBoxes: []
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

  // Handler for selecting a box
  const handleBoxSelect = (type: 'character' | 'text', index: number) => {
    setSelectedBoxType(type);
    setSelectedBoxIndex(index);
    // Ensure we're in adjust mode
    setPanelMode('adjust');
  };

  // Handler for character box transform end
  const handleBoxTransformEnd = (e: KonvaEventObject<Event>, type: 'character' | 'text', index: number) => {
    if (!selectedPanelId) return;
    
    // Get the transformer node
    const node = e.target;
    
    // Find the panel in our state
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    const panel = panels[panelIndex];
    
    // Calculate actual dimensions (accounting for scale)
    const relativeX = node.x() / (panel.width * scale);
    const relativeY = node.y() / (panel.height * scale);
    const relativeWidth = (node.width() * node.scaleX()) / (panel.width * scale);
    const relativeHeight = (node.height() * node.scaleY()) / (panel.height * scale);
    
    // Ensure values stay within 0-1 range
    const x = Math.max(0, Math.min(1, relativeX));
    const y = Math.max(0, Math.min(1, relativeY));
    const width = Math.max(0.05, Math.min(1 - x, relativeWidth));
    const height = Math.max(0.05, Math.min(1 - y, relativeHeight));
    
    // Update the box
    const updatedPanels = [...panels];
    
    if (type === 'character') {
      if (!updatedPanels[panelIndex].characterBoxes) return;
      updatedPanels[panelIndex].characterBoxes[index] = {
        ...updatedPanels[panelIndex].characterBoxes[index],
        x, y, width, height
      };
    } else if (type === 'text') {
      if (!updatedPanels[panelIndex].textBoxes) return;
      updatedPanels[panelIndex].textBoxes[index] = {
        ...updatedPanels[panelIndex].textBoxes[index],
        x, y, width, height
      };
    }
    
    // Reset scale after updating dimensions
    node.scaleX(1);
    node.scaleY(1);
    
    updatePanelsForCurrentPage(updatedPanels);
  };

  // Handler for box drag end
  const handleBoxDragEnd = (e: KonvaEventObject<DragEvent>, type: 'character' | 'text', index: number) => {
    if (!selectedPanelId) return;
    
    // Get the node
    const node = e.target;
    
    // Find the panel in our state
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    const panel = panels[panelIndex];
    
    // Calculate actual dimensions (accounting for scale)
    const relativeX = node.x() / (panel.width * scale);
    const relativeY = node.y() / (panel.height * scale);
    
    // Ensure values stay within 0-1 range
    const x = Math.max(0, Math.min(1 - node.width() / (panel.width * scale), relativeX));
    const y = Math.max(0, Math.min(1 - node.height() / (panel.height * scale), relativeY));
    
    // Update the box
    const updatedPanels = [...panels];
    
    if (type === 'character') {
      if (!updatedPanels[panelIndex].characterBoxes) return;
      updatedPanels[panelIndex].characterBoxes[index] = {
        ...updatedPanels[panelIndex].characterBoxes[index],
        x, y
      };
    } else if (type === 'text') {
      if (!updatedPanels[panelIndex].textBoxes) return;
      updatedPanels[panelIndex].textBoxes[index] = {
        ...updatedPanels[panelIndex].textBoxes[index],
        x, y
      };
    }
    
    updatePanelsForCurrentPage(updatedPanels);
  };

  // Handler for deleting a box
  const handleDeleteBox = (type: 'character' | 'text', index: number) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    const updatedPanels = [...panels];
    
    if (type === 'character') {
      if (!updatedPanels[panelIndex].characterBoxes) return;
      updatedPanels[panelIndex].characterBoxes = updatedPanels[panelIndex].characterBoxes.filter((_, i) => i !== index);
    } else if (type === 'text') {
      if (!updatedPanels[panelIndex].textBoxes) return;
      // Remove the text box
      updatedPanels[panelIndex].textBoxes = updatedPanels[panelIndex].textBoxes.filter((_, i) => i !== index);
      
      // ALSO remove the corresponding dialogue
      if (updatedPanels[panelIndex].dialogues && updatedPanels[panelIndex].dialogues[index]) {
        updatedPanels[panelIndex].dialogues.splice(index, 1);
      }
    }
    
    updatePanelsForCurrentPage(updatedPanels);
    
    // Clear selection
    setSelectedBoxType(null);
    setSelectedBoxIndex(null);
  };

  // Add double-click handler for panels
  const handlePanelDoubleClick = (panelId: string) => {
    if (isPanelFocusMode && focusedPanelId === panelId) {
      // Exit focus mode if double-clicking the focused panel
      exitPanelFocusMode();
    } else {
      // Enter focus mode for this panel - ALWAYS update both states
      setSelectedPanelId(panelId); // Set as selected first
      enterPanelFocusMode(panelId);
    }
  };

  // Function to enter panel focus mode
  const enterPanelFocusMode = (panelId: string) => {
    const panel = panels.find(p => p.id === panelId);
    if (!panel) return;
    
    // Update states in correct order
    setFocusedPanelId(panelId);
    setSelectedPanelId(panelId);
    setIsPanelFocusMode(true);
    
    // Calculate optimal scale to fit panel in available viewport
    // Account for: header (64px), padding (64px), exit button space (64px)
    const availableWidth = window.innerWidth * 0.6 - 64; // Account for sidebar and padding
    const availableHeight = window.innerHeight - 192; // Account for header, padding, and button space
    
    const scaleX = availableWidth / panel.width;
    const scaleY = availableHeight / panel.height;
    const optimalScale = Math.min(scaleX, scaleY, 1.5); // Cap at 1.5x zoom
    
    setFocusScale(Math.max(0.3, optimalScale)); // At least 0.3x zoom
  };

  // Function to exit panel focus mode
  const exitPanelFocusMode = () => {
    setIsPanelFocusMode(false);
    setFocusedPanelId(null);
    setFocusScale(1.0);
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
    
    const updatedPanels = [...panels];
    const panel = updatedPanels[panelIndex];
    
    // Check if character is already in the panel
    if (panel.characterNames.includes(character.name)) return;
    
    // Add to characterNames array
    panel.characterNames.push(character.name);
    
    // Check if character already has a box
    const existingBox = panel.characterBoxes?.find(box => box.character === character.name);
    
    if (!existingBox) {
      // Initialize characterBoxes array if it doesn't exist
      if (!panel.characterBoxes) {
        panel.characterBoxes = [];
      }
      
      // Get the color for visualization
      const characterColor = getCharacterColor(character.name);
      
      // Determine position based on existing characters
      const totalChars = panel.characterBoxes.length;
      
      let defaultX, defaultY, defaultWidth, defaultHeight;
      
      if (totalChars === 0) {
        // Single character in center
        defaultX = 0.2;
        defaultY = 0.2;
        defaultWidth = 0.6;
        defaultHeight = 0.6;
      } else if (totalChars === 1) {
        // Two characters side by side
        defaultX = 0.55;
        defaultY = 0.2;
        defaultWidth = 0.35;
        defaultHeight = 0.6;
        
        // Also update the first character's position
        panel.characterBoxes[0] = {
          ...panel.characterBoxes[0],
          x: 0.1,
          y: 0.2,
          width: 0.35,
          height: 0.6
        };
      } else {
        // Multiple characters - spread evenly
        const section = 1.0 / (totalChars + 1);
        defaultX = section * totalChars;
        defaultY = 0.2;
        defaultWidth = Math.min(section, 0.3);
        defaultHeight = 0.6;
      }
      
      // Add the character box with a stable ID
      const newCharacterBox = {
        character: character.name,
        x: defaultX,
        y: defaultY,
        width: defaultWidth,
        height: defaultHeight,
        color: characterColor
      };
      
      panel.characterBoxes.push(newCharacterBox);
      
      // AUTO-SELECT THE NEW CHARACTER BOX - use the new index after push
      const newBoxIndex = panel.characterBoxes.length - 1;
      setSelectedBoxType('character');
      setSelectedBoxIndex(newBoxIndex);
      setPanelMode('adjust'); // Ensure we're in adjust mode
    }
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Handler for removing a character from the selected panel
  const handleRemoveCharacter = (index: number) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Remove the character
    const updatedPanels = [...panels];
    const charName = updatedPanels[panelIndex].characterNames[index];
    
    // Remove from characterNames array
    updatedPanels[panelIndex].characterNames.splice(index, 1);
    
    // Also remove from characterBoxes array
    if (updatedPanels[panelIndex].characterBoxes) {
      updatedPanels[panelIndex].characterBoxes = updatedPanels[panelIndex].characterBoxes.filter(
        box => box.character !== charName
      );
    }
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Handler for adding dialogue to the selected panel
  const handleAddDialogue = () => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    const updatedPanels = [...panels];
    const currentDialogues = updatedPanels[panelIndex].dialogues || [];
    const currentTextBoxes = updatedPanels[panelIndex].textBoxes || [];
    
    // Create a default position for the dialogue
    let defaultX, defaultY, defaultWidth, defaultHeight;
    
    if (currentDialogues.length === 0) {
      // Top right
      defaultX = 0.6;
      defaultY = 0.1;
      defaultWidth = 0.3;
      defaultHeight = 0.2;
    } else if (currentDialogues.length === 1) {
      // Middle left
      defaultX = 0.1;
      defaultY = 0.4;
      defaultWidth = 0.3;
      defaultHeight = 0.2;
    } else {
      // Bottom right, or spread them out
      defaultX = 0.6;
      defaultY = 0.7;
      defaultWidth = 0.3;
      defaultHeight = 0.2;
    }
    
    // Add to dialogues array
    updatedPanels[panelIndex].dialogues = [
      ...currentDialogues,
      { character: '', text: '' }
    ];
    
    // Add corresponding text box
    updatedPanels[panelIndex].textBoxes = [
      ...currentTextBoxes,
      {
        text: '',
        x: defaultX,
        y: defaultY,
        width: defaultWidth,
        height: defaultHeight
      }
    ];
    
    updatePanelsForCurrentPage(updatedPanels);
    
    // Auto-select the new text box
    const newBoxIndex = updatedPanels[panelIndex].textBoxes.length - 1;
    setSelectedBoxType('text');
    setSelectedBoxIndex(newBoxIndex);
    setPanelMode('adjust');
  };

  const handleRemoveDialogue = (index: number) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Remove the dialogue
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].dialogues.splice(index, 1);
    
    // Also remove from textBoxes array
    if (updatedPanels[panelIndex].textBoxes) {
      updatedPanels[panelIndex].textBoxes.splice(index, 1);
    }
    
    updatePanelsForCurrentPage(updatedPanels);
  };
  
  // Handler for updating dialogue in the selected panel
  const handleUpdateDialogue = (index: number, field: keyof DialogueItem, value: string) => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    const updatedPanels = [...panels];
    
    // Ensure dialogues array exists and has the right length
    if (!updatedPanels[panelIndex].dialogues) {
      updatedPanels[panelIndex].dialogues = [];
    }
    
    // Ensure the dialogue exists at this index
    if (!updatedPanels[panelIndex].dialogues[index]) {
      updatedPanels[panelIndex].dialogues[index] = { character: '', text: '' };
    }
    
    // Update the dialogue
    updatedPanels[panelIndex].dialogues[index][field] = value;
    
    // If updating text, also update the corresponding textBox
    if (field === 'text' && updatedPanels[panelIndex].textBoxes && updatedPanels[panelIndex].textBoxes[index]) {
      updatedPanels[panelIndex].textBoxes[index].text = value;
    }
    
    updatePanelsForCurrentPage(updatedPanels);
  };

  // Mouse handlers for drawing
  const handleStageMouseDown = (e: KonvaEventObject<MouseEvent>) => {
    // If clicking directly on the stage (background), deselect current panel
    if (e.target === e.currentTarget) {
      setSelectedPanelId(null);
      return;
    }
    
    if (!selectedPanelId) return;
    
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
      
      if (panelMode === 'adjust') {
        // In adjust mode, do nothing and let Konva handle dragging
        return;
      } else if (panelMode === 'character-box' || panelMode === 'text-box') {
        // Store starting position for drawing
        setDrawingStartPos({ x: relativeX, y: relativeY });
        
        // Initialize preview box
        setPreviewBox({
          x: relativeX,
          y: relativeY,
          width: 0,
          height: 0,
          character: activeCharacter || undefined,
          color: panelMode === 'character-box' && activeCharacter ? 
                 getCharacterColor(activeCharacter) : undefined
        });
        
        // Prevent event propagation to avoid panel dragging
        e.cancelBubble = true;
      }
    }
  };
  
  // Modified handleStageMouseMove function
  const handleStageMouseMove = (e: KonvaEventObject<MouseEvent>) => {
    if (!selectedPanelId || !drawingStartPos || !previewBox) return;
    
    const panel = panels.find(p => p.id === selectedPanelId);
    if (!panel) return;
    
    const pos = e.target.getStage()?.getPointerPosition();
    if (!pos) return;
    
    // Panel coordinates
    const panelX = panel.x * scale;
    const panelY = panel.y * scale;
    const panelWidth = panel.width * scale;
    const panelHeight = panel.height * scale;
    
    // Calculate current position relative to panel
    const relativeX = Math.max(0, Math.min(1, (pos.x - panelX) / panelWidth));
    const relativeY = Math.max(0, Math.min(1, (pos.y - panelY) / panelHeight));
    
    // Update preview box
    const x = Math.min(drawingStartPos.x, relativeX);
    const y = Math.min(drawingStartPos.y, relativeY);
    const width = Math.abs(relativeX - drawingStartPos.x);
    const height = Math.abs(relativeY - drawingStartPos.y);
    
    setPreviewBox({
      ...previewBox,
      x, y, width, height
    });
    
    // Prevent panel dragging during box drawing
    e.cancelBubble = true;
  };
  
  // Modified handleStageMouseUp function
  const handleStageMouseUp = (e: KonvaEventObject<MouseEvent>) => {
    if (!selectedPanelId || !drawingStartPos || !previewBox) {
      setPreviewBox(null);
      setDrawingStartPos(null);
      return;
    }
    
    const pos = e.target.getStage()?.getPointerPosition();
    if (!pos) {
      setPreviewBox(null);
      setDrawingStartPos(null);
      return;
    }
    
    // Find the selected panel
    const panel = panels.find(p => p.id === selectedPanelId);
    if (!panel) {
      setPreviewBox(null);
      setDrawingStartPos(null);
      return;
    }
    
    // Minimum size check
    if (previewBox.width < 0.05 || previewBox.height < 0.05) {
      setPreviewBox(null);
      setDrawingStartPos(null);
      return;
    }
    
    // Find the panel index
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) {
      setPreviewBox(null);
      setDrawingStartPos(null);
      return;
    }
    
    const updatedPanels = [...panels];
    
    if (panelMode === 'character-box' && activeCharacter) {
      // Add the character box
      const characterBoxes = [...(updatedPanels[panelIndex].characterBoxes || [])];
      characterBoxes.push({
        character: activeCharacter,
        x: previewBox.x,
        y: previewBox.y,
        width: previewBox.width,
        height: previewBox.height,
        color: getCharacterColor(activeCharacter)
      });
      updatedPanels[panelIndex].characterBoxes = characterBoxes;
    } else if (panelMode === 'text-box') {
      // Add the text box
      const textBoxes = [...(updatedPanels[panelIndex].textBoxes || [])];
      textBoxes.push({
        text: '',
        x: previewBox.x,
        y: previewBox.y,
        width: previewBox.width,
        height: previewBox.height
      });
      updatedPanels[panelIndex].textBoxes = textBoxes;
    }
    
    updatePanelsForCurrentPage(updatedPanels);
    
    // Reset drawing state
    setPreviewBox(null);
    setDrawingStartPos(null);
    
    // After drawing a box, return to adjust mode
    setPanelMode('adjust');
  };
  
  // Helper function to get a consistent color for a character
  const getCharacterColor = (characterName: string) => {
    const characterColors = [
      '#FF5733', '#33FF57', '#3357FF', '#FF33F5', '#F5FF33', 
      '#33FFF5', '#F533FF', '#FF3333', '#33FF33', '#3333FF'
    ];
    const colorIndex = characterName.charCodeAt(0) % characterColors.length;
    return characterColors[colorIndex];
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

  // Status notification function
  const showStatusMessage = (message: string, type: 'success' | 'error' | 'info' | 'loading' = 'info', duration = 3000) => {
    // Clear any existing timeout
    if (statusTimeout) {
      clearTimeout(statusTimeout);
    }
    
    // Update status
    setStatusMessage(message);
    setStatusType(type);
    setShowStatus(true);
    
    // Hide after duration (unless it's a loading message)
    if (type !== 'loading' && duration > 0) {
      const timeout = setTimeout(() => {
        setShowStatus(false);
      }, duration);
      
      setStatusTimeout(timeout);
    }
  };

  // Clear status notification
  const clearStatusMessage = () => {
    setShowStatus(false);
    if (statusTimeout) {
      clearTimeout(statusTimeout);
      setStatusTimeout(null);
    }
  };


  // Update saveToProject function to properly save to backend
  const saveToProject = async () => {
    if (!currentProject) return;
    
    showStatusMessage('Saving project...', 'loading');
    setIsSaving(true);
    
    try {
      // Update project timestamp
      const updatedProject = {
        ...currentProject,
        pages: localPages.length,
        lastModified: new Date().toISOString()
      };
      
      if (setCurrentProject) {
        setCurrentProject(updatedProject);
      }
      
      // Save to backend with full panel data
      const response = await fetch(`${apiEndpoint}/projects/${currentProject.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          pages: localPages // Send full pages with panel data
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        showStatusMessage(data.message || 'Project saved successfully', 'success');
        setHasUnsavedChanges(false);
        setLastSaveTime(new Date());
      } else {
        throw new Error('Failed to save project');
      }
      
      // Also save to localStorage as backup
      localStorage.setItem(`project_${currentProject.id}`, JSON.stringify(localPages));
      
      // Call onSaveProject callback
      if (onSaveProject) {
        onSaveProject(currentProject.id, localPages);
      }
      
    } catch (error) {
      console.error('Error saving project:', error);
      showStatusMessage('Error saving project', 'error');
    } finally {
      setIsSaving(false);
    }
  };

  const handleManualSave = () => {
    if (currentProject) {
      saveToProject();
    }
  };

  const handleGeneratePanel = async () => {
    if (!selectedPanelId) return;
    
    const panelIndex = panels.findIndex(p => p.id === selectedPanelId);
    if (panelIndex === -1) return;
    
    // Mark the panel as generating
    const updatedPanels = [...panels];
    updatedPanels[panelIndex].isGenerating = true;
    updatePanelsForCurrentPage(updatedPanels);
    
    showStatusMessage('Generating panel...', 'loading');
    
    try {
      const panel = panels[panelIndex];
      
      // Convert panel dimensions from model coordinates to absolute pixel values
      const pixelWidth = Math.round(panel.width);
      const pixelHeight = Math.round(panel.height);
      
      // Format character boxes in the way the API expects
      // The API expects arrays of [x, y, x+width, y+height] for bounding boxes
      const formattedCharacterBoxes = panel.characterBoxes?.map(box => ({
        character: box.character,
        x: box.x,
        y: box.y,
        width: box.width, 
        height: box.height,
        color: box.color
      })) || [];
      
      // Format text boxes in the way the API expects
      const formattedTextBoxes = panel.textBoxes?.map(box => ({
        text: box.text,
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height
      })) || [];
      
      // Prepare the API data using snake_case for backend and camelCase for frontend
      const apiData = {
        projectId: currentProject?.id || 'default',
        prompt: panel.prompt || '',
        setting: panel.setting || '',
        characterNames: panel.characterNames,
        dialogues: panel.dialogues,
        actions: panel.actions,
        characterBoxes: formattedCharacterBoxes,
        textBoxes: formattedTextBoxes,
        panelIndex: panel.panelIndex || 0,
        seed: panel.seed || Math.floor(Math.random() * 1000000),
        width: pixelWidth,
        height: pixelHeight
      };
      
      console.log("Sending panel data to API:", apiData);
      
      // Call the API
      const response = await fetch(`${apiEndpoint}/generate_panel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(apiData)
      });
      
      if (!response.ok) {
        throw new Error(`API returned status ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Check if the request was queued (models still initializing)
      if (data.status === 'queued') {
        // Show a message that the request is queued
        console.log(`Panel generation queued. Request ID: ${data.request_id}`);
        showStatusMessage('Panel generation queued. Please wait...', 'info');
        
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
        showStatusMessage('Panel generated successfully', 'success');
      } else {
        // Handle error
        console.error('Error generating panel:', data.message);
        showStatusMessage(`Error: ${data.message}`, 'error');
        
        // Mark the panel as not generating
        const newPanels = [...panels];
        newPanels[panelIndex].isGenerating = false;
        updatePanelsForCurrentPage(newPanels);
      }
    } catch (error) {
      console.error('Error calling API:', error);
      showStatusMessage(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
      
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
      imageData: data.imageData,          // Keep immediate display data
      imagePath: data.imagePath ? normalizeImagePath(data.imagePath) : data.imageData,
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

  // Modified page navigation function with image loading/unloading
  // Enhanced page loading with status indicators
  const handlePageNavigation = (newPageIndex: number) => {
    if (newPageIndex < 0 || newPageIndex >= localPages.length) return;
    
    // Check if a panel is generating, and if so, warn the user
    const isGeneratingOnCurrentPage = panels.some(panel => panel.isGenerating);
    if (isGeneratingOnCurrentPage) {
      if (!window.confirm("A panel is currently generating. Navigating away may interrupt this process. Continue anyway?")) {
        return;
      }
    }
    
    // Show loading status if the page has images to load
    const targetPage = localPages[newPageIndex];
    const hasImagesToLoad = targetPage?.panels.some(panel => panel.imagePath && !panel.imageData);
    
    if (hasImagesToLoad) {
      showStatusMessage('Loading page...', 'loading');
    }
    
    // Clear images from current page to free memory
    clearPageImages(currentPageIndex);
    
    // Update page index
    setCurrentPageIndex(newPageIndex);
    setSelectedPanelId(null); // Clear selection
    
    // Load images for the new page
    loadPageImages(newPageIndex)
      .then(() => {
        if (hasImagesToLoad) {
          showStatusMessage('Page loaded', 'success');
        }
      })
      .catch(error => {
        console.error('Error loading page images:', error);
        showStatusMessage('Error loading some images', 'error');
      });
  };

  // Update the existing navigation handlers to use the new function
  const handlePrevPage = () => {
    if (currentPageIndex > 0) {
      handlePageNavigation(currentPageIndex - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPageIndex < localPages.length - 1) {
      handlePageNavigation(currentPageIndex + 1);
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
    const newPageIndex = localPages.length;
    const newPage = {
      id: `page-${newPageIndex + 1}`,
      panels: createDefaultPanelsForPage()
    };
    
    // Update local state first
    const updatedPages = [...localPages, newPage];
    setLocalPages(updatedPages);
    
    // Update parent state if needed
    if (setPages) {
      setPages(updatedPages);
    }
    
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
    
    // Update the current page using localPages
    const updatedPages = [...localPages];
    
    // Ensure the page exists
    if (!updatedPages[currentPageIndex]) {
      updatedPages[currentPageIndex] = { id: `page-${currentPageIndex + 1}`, panels: [] };
    }
    
    updatedPages[currentPageIndex].panels = newPanels;
    
    // Update local state
    setLocalPages(updatedPages);
    
    // Update parent state if needed
    if (setPages) {
      setPages(updatedPages);
    }
    
    setSelectedPanelId(null);
    setShowTemplateDialog(false);
  };

  // Function to parse page range input
  const parsePageRange = (rangeInput: string, totalPages: number): number[] => {
    const pages: number[] = [];
    
    if (!rangeInput.trim()) {
      return pages;
    }
    
    const parts = rangeInput.split(',');
    
    for (const part of parts) {
      if (part.includes('-')) {
        // Range like "1-5"
        const [start, end] = part.split('-').map(num => parseInt(num.trim()));
        
        if (!isNaN(start) && !isNaN(end)) {
          const validStart = Math.max(1, Math.min(start, totalPages));
          const validEnd = Math.max(validStart, Math.min(end, totalPages));
          
          for (let i = validStart; i <= validEnd; i++) {
            pages.push(i - 1); // Convert to 0-based index
          }
        }
      } else {
        // Single page like "3"
        const pageNum = parseInt(part.trim());
        
        if (!isNaN(pageNum) && pageNum > 0 && pageNum <= totalPages) {
          pages.push(pageNum - 1); // Convert to 0-based index
        }
      }
    }
    
    // Return unique pages in order
    return [...new Set(pages)].sort((a, b) => a - b);
  };

  // Export manga function
  const exportManga = async () => {
    if (!currentProject || !localPages.length) {
      setStatusMessage('No project or pages to export');
      setTimeout(() => setStatusMessage(''), 3000);
      return;
    }
    
    setIsExporting(true);
    setExportProgress(0);
    
    try {
      // Parse page range
      let pagesToExport: number[];
      
      if (exportConfig.pageRange === 'current') {
        pagesToExport = [currentPageIndex];
      } else if (exportConfig.pageRange === 'custom') {
        pagesToExport = parsePageRange(exportConfig.customRange, localPages.length);
        
        if (pagesToExport.length === 0) {
          throw new Error('Invalid page range. Please use format like "1-3, 5, 7-9"');
        }
      } else {
        // 'all' is the default
        pagesToExport = Array.from({ length: localPages.length }, (_, i) => i);
      }
      
      // Different export methods based on type
      if (exportConfig.type === 'png') {
        await exportAsPNG(pagesToExport);
      } else {
        await exportAsPDF(pagesToExport);
      }
      
      // Show success message
      setStatusMessage('Export completed successfully');
      setTimeout(() => setStatusMessage(''), 3000);
    } catch (error) {
      console.error('Export error:', error);
      setStatusMessage('Export failed: ' + (error instanceof Error ? error.message : String(error)));
      setTimeout(() => setStatusMessage(''), 3000);
    } finally {
      setIsExporting(false);
      setExportProgress(100);
      // Reset to 0 after a delay
      setTimeout(() => setExportProgress(0), 1000);
    }
  };

  // PNG Export (as ZIP file)
  const exportAsPNG = async (pageIndices: number[]) => {
    // Create a zip file using JSZip
    const JSZip = (await import('jszip')).default;
    const zip = new JSZip();
    
    // Add metadata
    zip.file('metadata.json', JSON.stringify({
      name: currentProject?.name,
      created: new Date().toISOString(),
      modified: new Date().toISOString(),
      pageCount: pageIndices.length,
      exportDate: new Date().toISOString()
    }));
    
    // Add pages
    const pagesFolder = zip.folder('pages');
    
    // Original page index to restore later
    const originalPageIndex = currentPageIndex;
    
    // Pixel ratio for export quality
    const pixelRatio = exportConfig.quality === 'high' ? 3 : 2;
    
    // For each page, take a screenshot and add to zip
    for (let i = 0; i < pageIndices.length; i++) {
      const pageIndex = pageIndices[i];
      
      // Update progress
      setExportProgress(Math.round((i / pageIndices.length) * 100));
      
      // Switch to page
      setCurrentPageIndex(pageIndex);
      
      // Wait for the page to render
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Take a screenshot using the stage ref
      if (stageRef.current) {
        const dataURL = stageRef.current.toDataURL({
          pixelRatio,
          mimeType: 'image/png'
        });
        
        // Convert data URL to blob
        const imageData = dataURL.split(',')[1];
        const binaryData = atob(imageData);
        const array = new Uint8Array(binaryData.length);
        
        for (let j = 0; j < binaryData.length; j++) {
          array[j] = binaryData.charCodeAt(j);
        }
        
        // Add to zip with original page number for clarity
        pagesFolder?.file(`page-${pageIndex + 1}.png`, array);
      }
    }
    
    // Restore original page index
    setCurrentPageIndex(originalPageIndex);
    
    // Generate the zip file
    const content = await zip.generateAsync({
      type: 'blob',
      compression: 'DEFLATE',
      compressionOptions: {
        level: 9 // Maximum compression
      }
    });
    
    // Create name based on project and export type
    const fileName = `${currentProject?.name.replace(/\s+/g, '-')}-pages.zip`;
    
    // Trigger download
    saveAs(content, fileName);
  };

  // PDF Export
  const exportAsPDF = async (pageIndices: number[]) => {
    // Check if apiEndpoint is available
    if (apiEndpoint) {
      // Server-side PDF generation
      await serverPDFExport(pageIndices);
    } else {
      // Client-side PDF generation
      await clientPDFExport(pageIndices);
    }
  };

  // Client-side PDF generation
  const clientPDFExport = async (pageIndices: number[]) => {
    // Import jsPDF and dependencies dynamically
    const { default: jsPDF } = await import('jspdf');
    
    // Original page index to restore later
    const originalPageIndex = currentPageIndex;
    
    // Create a new PDF document
    // A4 size: 210 × 297 mm (8.27 × 11.69 inches)
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    });
    
    // PDF dimensions
    const pdfWidth = 210;  // mm
    const pdfHeight = 297; // mm
    
    // Manga dimensions with margin
    const margin = 10; // mm
    const mangaWidth = pdfWidth - (margin * 2);
    const mangaHeight = pdfHeight - (margin * 2);
    
    // Pixel ratio for quality
    const pixelRatio = exportConfig.quality === 'high' ? 3 : 2;
    
    // For each page
    for (let i = 0; i < pageIndices.length; i++) {
      const pageIndex = pageIndices[i];
      
      // Update progress
      setExportProgress(Math.round((i / pageIndices.length) * 100));
      
      // Add new page to PDF if not the first page
      if (i > 0) {
        pdf.addPage();
      }
      
      // Switch to page
      setCurrentPageIndex(pageIndex);
      
      // Wait for the page to render
      await new Promise(resolve => setTimeout(resolve, 150));
      
      // Get screenshot
      if (stageRef.current) {
        const dataURL = stageRef.current.toDataURL({
          pixelRatio,
          mimeType: 'image/jpeg',
          quality: 0.9
        });
        
        // Add image to PDF
        pdf.addImage(
          dataURL,
          'JPEG',
          margin,
          margin,
          mangaWidth,
          mangaHeight,
          `page-${pageIndex + 1}`,
          'MEDIUM'
        );
      }
    }
    
    // Restore original page index
    setCurrentPageIndex(originalPageIndex);
    
    // Create file name
    const fileName = `${currentProject?.name.replace(/\s+/g, '-')}.pdf`;
    
    // Save PDF
    pdf.save(fileName);
  };

  // Server-side PDF generation
  const serverPDFExport = async (pageIndices: number[]) => {
    // Store current page index to restore later
    const originalPageIndex = currentPageIndex;
    
    // We'll send image data for each page to the server
    const pageImages = [];
    
    // Pixel ratio for quality
    const pixelRatio = exportConfig.quality === 'high' ? 3 : 2;
    
    // For each page in the range
    for (let i = 0; i < pageIndices.length; i++) {
      const pageIndex = pageIndices[i];
      
      // Update progress
      setExportProgress(Math.round((i / pageIndices.length) * 50)); // First half of progress
      
      // Switch to page
      setCurrentPageIndex(pageIndex);
      
      // Wait for the page to render
      await new Promise(resolve => setTimeout(resolve, 150));
      
      // Get screenshot
      if (stageRef.current) {
        const dataURL = stageRef.current.toDataURL({
          pixelRatio,
          mimeType: 'image/jpeg',
          quality: 0.9
        });
        
        pageImages.push({
          pageIndex,
          dataURL
        });
      }
    }
    
    // Restore original page index
    setCurrentPageIndex(originalPageIndex);
    
    // Send to server
    const response = await fetch(`${normalizeImagePath('/api/export/pdf')}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        projectId: currentProject?.id,
        projectName: currentProject?.name,
        pages: pageImages,
        quality: exportConfig.quality
      })
    });
    
    if (!response.ok) {
      throw new Error(`PDF generation failed: ${response.status}`);
    }
    
    // Get the download URL
    const { downloadUrl } = await response.json();
    
    // Update progress for second half
    setExportProgress(75);
    
    // Download the PDF
    const pdfResponse = await fetch(downloadUrl);
    const pdfBlob = await pdfResponse.blob();
    
    // Update progress
    setExportProgress(95);
    
    // Create a file name
    const fileName = `${currentProject?.name.replace(/\s+/g, '-')}.pdf`;
    
    // Save the file
    saveAs(pdfBlob, fileName);
  };

  // Add this to render the export dialog modal
  const renderExportDialog = () => {
    if (!showExportDialog) return null;
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold">Export Manga</h3>
            <button
              onClick={() => setShowExportDialog(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div className="space-y-4">
            {/* Export Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Export Format</label>
              <div className="flex space-x-4">
                <label className="inline-flex items-center">
                  <input
                    type="radio"
                    className="form-radio h-4 w-4 text-indigo-600"
                    value="png"
                    checked={exportConfig.type === 'png'}
                    onChange={() => setExportConfig(prev => ({ ...prev, type: 'png' }))}
                  />
                  <span className="ml-2">PNG Images (ZIP)</span>
                </label>
                <label className="inline-flex items-center">
                  <input
                    type="radio"
                    className="form-radio h-4 w-4 text-indigo-600"
                    value="pdf"
                    checked={exportConfig.type === 'pdf'}
                    onChange={() => setExportConfig(prev => ({ ...prev, type: 'pdf' }))}
                  />
                  <span className="ml-2">PDF Document</span>
                </label>
              </div>
            </div>
            
            {/* Page Range */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Pages to Export</label>
              <div className="space-y-2">
                <label className="inline-flex items-center">
                  <input
                    type="radio"
                    className="form-radio h-4 w-4 text-indigo-600"
                    value="all"
                    checked={exportConfig.pageRange === 'all'}
                    onChange={() => setExportConfig(prev => ({ ...prev, pageRange: 'all' }))}
                  />
                  <span className="ml-2">All Pages ({localPages.length})</span>
                </label>
                
                <label className="inline-flex items-center">
                  <input
                    type="radio"
                    className="form-radio h-4 w-4 text-indigo-600"
                    value="current"
                    checked={exportConfig.pageRange === 'current'}
                    onChange={() => setExportConfig(prev => ({ ...prev, pageRange: 'current' }))}
                  />
                  <span className="ml-2">Current Page Only ({currentPageIndex + 1})</span>
                </label>
                
                <label className="inline-flex items-center">
                  <input
                    type="radio"
                    className="form-radio h-4 w-4 text-indigo-600"
                    value="custom"
                    checked={exportConfig.pageRange === 'custom'}
                    onChange={() => setExportConfig(prev => ({ ...prev, pageRange: 'custom' }))}
                  />
                  <span className="ml-2">Custom Range</span>
                </label>
                
                {exportConfig.pageRange === 'custom' && (
                  <div className="ml-6">
                    <input
                      type="text"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      placeholder="e.g. 1-3, 5, 7-9"
                      value={exportConfig.customRange}
                      onChange={(e) => setExportConfig(prev => ({ ...prev, customRange: e.target.value }))}
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Enter page numbers or ranges (e.g. "1-3, 5, 7-9")
                    </p>
                  </div>
                )}
              </div>
            </div>
            
            {/* Quality */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Export Quality</label>
              <div className="flex space-x-4">
                <label className="inline-flex items-center">
                  <input
                    type="radio"
                    className="form-radio h-4 w-4 text-indigo-600"
                    value="normal"
                    checked={exportConfig.quality === 'normal'}
                    onChange={() => setExportConfig(prev => ({ ...prev, quality: 'normal' }))}
                  />
                  <span className="ml-2">Normal</span>
                </label>
                <label className="inline-flex items-center">
                  <input
                    type="radio"
                    className="form-radio h-4 w-4 text-indigo-600"
                    value="high"
                    checked={exportConfig.quality === 'high'}
                    onChange={() => setExportConfig(prev => ({ ...prev, quality: 'high' }))}
                  />
                  <span className="ml-2">High Resolution</span>
                </label>
              </div>
            </div>
            
            {/* Export Button */}
            <div className="pt-4">
              <button
                onClick={() => {
                  setShowExportDialog(false);
                  exportManga();
                }}
                disabled={isExporting}
                className="w-full px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
              >
                {isExporting ? 'Exporting...' : 'Export'}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Add the status message component to render
  const renderStatusMessage = () => {
    if (!showStatus) return null;
    
    const bgColors = {
      success: 'bg-green-100 border-green-500',
      error: 'bg-red-100 border-red-500',
      info: 'bg-blue-100 border-blue-500',
      loading: 'bg-indigo-100 border-indigo-500'
    };
    
    const textColors = {
      success: 'text-green-700',
      error: 'text-red-700',
      info: 'text-blue-700',
      loading: 'text-indigo-700'
    };
    
    const icons = {
      success: (
        <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
        </svg>
      ),
      error: (
        <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
      ),
      info: (
        <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zm-1 7a1 1 0 002 0v-3a1 1 0 00-2 0v3z" clipRule="evenodd" />
        </svg>
      ),
      loading: (
        <div className="animate-spin w-5 h-5 mr-2 border-2 border-dashed rounded-full border-current"></div>
      )
    };
    
    return (
      <div className="fixed bottom-4 right-4 max-w-sm z-50">
        <div className={`p-3 rounded-lg shadow-md border-l-4 ${bgColors[statusType]}`}>
          <div className="flex items-center">
            <div className={textColors[statusType]}>
              {icons[statusType]}
            </div>
            <div className={`ml-3 ${textColors[statusType]}`}>
              <p className="text-sm font-medium">{statusMessage}</p>
            </div>
            {statusType !== 'loading' && (
              <button 
                onClick={clearStatusMessage}
                className="ml-auto text-gray-400 hover:text-gray-500"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    );
  };


  const handleBackgroundClick = (e: KonvaEventObject<MouseEvent>) => {
    // If clicking directly on the stage (background), deselect current panel
    // Check that the target is the stage itself
    if (e.target === e.currentTarget) {
      setSelectedPanelId(null);
    }
  };

  const renderSaveStatus = () => {
    return (
      <div className="text-xs text-gray-500 flex items-center">
        {hasUnsavedChanges ? (
          <>
            <span className="w-2 h-2 bg-yellow-500 rounded-full mr-1"></span>
            Unsaved changes
          </>
        ) : (
          <>
            <span className="w-2 h-2 bg-green-500 rounded-full mr-1"></span>
            Saved {lastSaveTime ? `at ${lastSaveTime.toLocaleTimeString()}` : ''}
          </>
        )}
      </div>
    );
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
          onClick={() => setShowExportDialog(true)}
          title="Export Manga"
        >
          <FileDown size={20} />
        </button>

        {/* Save Project */}
        <button
          className="p-2 bg-indigo-100 text-green-600 rounded-md hover:bg-indigo-200"
          onClick={handleManualSave}
          title="Save Project"
          disabled={isSaving || !hasUnsavedChanges}
        >
          <Save size={20} />
        </button>
        
        {/* Project Manager button */}
        <button
          onClick={onShowProjectManager}
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
            {currentProject && (
              <>
                <h2 className="text-sm font-medium">{currentProject.name}</h2>
                {renderSaveStatus()}
              </>
            )}
          </div>
        </div>
  
        {/* Canvas and Panel Editor */}
        <div className="flex-1 flex overflow-hidden">
          {/* Canvas Area */}
          <div className="flex-1 bg-white overflow-auto">
            
            <div 
              className="min-h-full min-w-full flex items-center justify-center"
              style={{
                // Dynamic sizing based on focus mode
                width: isPanelFocusMode && focusedPanel 
                  ? Math.max(window.innerWidth * 0.6, focusedPanel.width * focusScale + 128)
                  : pageSize.width * scale + 100,
                height: isPanelFocusMode && focusedPanel
                  ? Math.max(window.innerHeight - 128, focusedPanel.height * focusScale + 128) 
                  : pageSize.height * scale + 150,
                padding: isPanelFocusMode ? '32px' : '32px 32px 96px 32px' // Less bottom padding in focus mode
              }}
            >
              <div className="relative">
                {/* Exit button - positioned better for focus mode */}
                {isPanelFocusMode && focusedPanel && (
                  <button
                    onClick={exitPanelFocusMode}
                    className="absolute bg-none hover:bg-indigo-100 text-white rounded-full p-3 shadow-lg transition-colors duration-200 z-50 group"
                    style={{
                      // Position in top-left of container, not relative to panel
                      left: -48,
                      top: -32,
                    }}
                    title="Exit Focus Mode (ESC)"
                  >
                    <X size={16} />
                    
                    {/* Tooltip - shows on hover */}
                    <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-indigo-100 text-white text-sm px-3 py-2 rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
                      Exit Focus Mode (ESC)
                    </div>
                  </button>
                )}

                <Stage 
                  width={isPanelFocusMode && focusedPanel ? focusedPanel.width * focusScale : pageSize.width * scale} 
                  height={isPanelFocusMode && focusedPanel ? focusedPanel.height * focusScale : pageSize.height * scale}
                  ref={stageRef}
                  className={`bg-gray-100 shadow-inner border border-gray-300 ${
                    !selectedPanel ? 'cursor-default' :
                    panelMode === 'adjust' ? 'cursor-move' :
                    panelMode === 'character-box' || panelMode === 'text-box' ? 'cursor-crosshair' :
                    'cursor-default'
                  }`}
                  onClick={handleBackgroundClick}
                  onMouseDown={handleStageMouseDown}
                  onMouseMove={handleStageMouseMove}
                  onMouseUp={handleStageMouseUp}
                >
                  <Layer>
                    {/* Page background - only show in normal mode */}
                    {!isPanelFocusMode && (
                      <Rect
                        width={pageSize.width * scale}
                        height={pageSize.height * scale}
                        fill="white"
                      />
                    )}
                    
                    {/* In focus mode, show white background for the focused panel */}
                    {isPanelFocusMode && focusedPanel && (
                      <Rect
                        width={focusedPanel.width * focusScale}
                        height={focusedPanel.height * focusScale}
                        fill="white"
                      />
                    )}
                    
                    {/* Panels */}
                    {panels.map((panel) => {
                      // In focus mode, only show the focused panel
                      if (isPanelFocusMode && panel.id !== focusedPanelId) {
                        return null;
                      }
                      
                      // Calculate position - in focus mode, center the panel at origin
                      const panelX = isPanelFocusMode && panel.id === focusedPanelId ? 0 : panel.x;
                      const panelY = isPanelFocusMode && panel.id === focusedPanelId ? 0 : panel.y;
                      
                      return (
                        <Rect
                          key={panel.id}
                          id={panel.id}
                          x={panelX * currentScale}
                          y={panelY * currentScale}
                          width={panel.width * currentScale}
                          height={panel.height * currentScale}
                          stroke={selectedPanelId === panel.id ? '#4299e1' : '#000'}
                          strokeWidth={selectedPanelId === panel.id ? 3 : 2}
                          fill={panel.imageData ? 'transparent' : '#f7fafc'}
                          onClick={() => handlePanelSelect(panel.id)}
                          onDblClick={() => handlePanelDoubleClick(panel.id)}
                          onTap={() => handlePanelSelect(panel.id)}
                          onDblTap={() => handlePanelDoubleClick(panel.id)}
                          draggable={selectedPanelId === panel.id && panelMode === 'adjust' && !isPanelFocusMode}
                          onDragMove={handleDragMove}
                          onDragEnd={handleDragEnd}
                        />
                      );
                    })}
                    
                    {/* Panel images */}
                    {panels.map((panel) => {
                      // In focus mode, only show the focused panel's image
                      if (isPanelFocusMode && panel.id !== focusedPanelId) {
                        return null;
                      }
                      
                      if (!panel.imageData) return null;
                      
                      const panelX = isPanelFocusMode && panel.id === focusedPanelId ? 0 : panel.x;
                      const panelY = isPanelFocusMode && panel.id === focusedPanelId ? 0 : panel.y;
                      
                      return (
                        <KonvaImage
                          key={`img-${panel.id}`}
                          x={panelX * currentScale}
                          y={panelY * currentScale}
                          width={panel.width * currentScale}
                          height={panel.height * currentScale}
                          image={(() => {
                            const img = new window.Image();
                            img.src = panel.imageData;
                            return img;
                          })()}
                          onClick={() => handlePanelSelect(panel.id)}
                          onTap={() => handlePanelSelect(panel.id)}
                        />
                      );
                    })}

                    {/* Character and Text boxes */}
                    {panels.map(panel => {
                      // In focus mode, only show boxes for the focused panel
                      if (isPanelFocusMode && panel.id !== focusedPanelId) {
                        return null;
                      }
                      
                      const panelX = isPanelFocusMode && panel.id === focusedPanelId ? 0 : panel.x;
                      const panelY = isPanelFocusMode && panel.id === focusedPanelId ? 0 : panel.y;
                      
                      return (
                        <Group key={`panel-boxes-${panel.id}`} id={`panel-boxes-${panel.id}`}>
                          {/* Character boxes */}
                          {panel.characterBoxes?.map((box, index) => (
                            <Rect
                              key={`char-box-${panel.id}-${index}`}
                              name={`character-box-${index}`}
                              x={(panelX + box.x * panel.width) * currentScale}
                              y={(panelY + box.y * panel.height) * currentScale}
                              width={box.width * panel.width * currentScale}
                              height={box.height * panel.height * currentScale}
                              stroke={box.color}
                              strokeWidth={selectedBoxType === 'character' && selectedBoxIndex === index && panel.id === selectedPanelId ? 3 : 2}
                              dash={[5, 5]}
                              fill={box.color + '33'}
                              draggable={panel.id === selectedPanelId}
                              onClick={() => {
                                if (panel.id === selectedPanelId) {
                                  handleBoxSelect('character', index);
                                }
                              }}
                              onTap={() => {
                                if (panel.id === selectedPanelId) {
                                  handleBoxSelect('character', index);
                                }
                              }}
                              onDragEnd={(e) => handleBoxDragEnd(e, 'character', index)}
                              onTransformEnd={(e) => handleBoxTransformEnd(e, 'character', index)}
                            />
                          ))}
                        
                          {/* Text boxes */}
                          {panel.textBoxes?.map((box, index) => (
                            <Rect
                              key={`text-box-${panel.id}-${index}`}
                              name={`text-box-${index}`}
                              x={(panelX + box.x * panel.width) * currentScale}
                              y={(panelY + box.y * panel.height) * currentScale}
                              width={box.width * panel.width * currentScale}
                              height={box.height * panel.height * currentScale}
                              stroke="#000000"
                              strokeWidth={selectedBoxType === 'text' && selectedBoxIndex === index && panel.id === selectedPanelId ? 3 : 2}
                              dash={[5, 5]}
                              fill="#FFFFFF88"
                              draggable={panel.id === selectedPanelId}
                              onClick={() => {
                                if (panel.id === selectedPanelId) {
                                  handleBoxSelect('text', index);
                                }
                              }}
                              onTap={() => {
                                if (panel.id === selectedPanelId) {
                                  handleBoxSelect('text', index);
                                }
                              }}
                              onDragEnd={(e) => handleBoxDragEnd(e, 'text', index)}
                              onTransformEnd={(e) => handleBoxTransformEnd(e, 'text', index)}
                            />
                          ))}
                        </Group>
                      );
                    })}
                    
                    {/* Preview box while drawing - only show for selected panel */}
                    {previewBox && selectedPanel && (!isPanelFocusMode || selectedPanel.id === focusedPanelId) && (
                      <Rect
                        x={(selectedPanel.x + previewBox.x * selectedPanel.width) * currentScale}
                        y={(selectedPanel.y + previewBox.y * selectedPanel.height) * currentScale}
                        width={previewBox.width * selectedPanel.width * currentScale}
                        height={previewBox.height * selectedPanel.height * currentScale}
                        stroke={panelMode === 'character-box' ? (previewBox.color || '#FF5733') : '#000000'}
                        strokeWidth={2}
                        dash={[5, 5]}
                        fill={panelMode === 'character-box' ? (previewBox.color + '33' || '#FF573333') : '#FFFFFF44'}
                      />
                    )}
                    
                    {/* Transformer for resizing panels */}
                    <Transformer
                      ref={transformerRef}
                      boundBoxFunc={(oldBox, newBox) => {
                        if (newBox.width < 20 || newBox.height < 20) {
                          return oldBox;
                        }
                        return newBox;
                      }}
                      onTransformEnd={handleTransformEnd}
                      padding={5}
                      enabledAnchors={[
                        'top-left', 'top-center', 'top-right',
                        'middle-left', 'middle-right',
                        'bottom-left', 'bottom-center', 'bottom-right'
                      ]}
                      rotateEnabled={false}
                    />
                    
                    {/* Box Transformer */}
                    <Transformer
                      ref={transformerBoxRef}
                      boundBoxFunc={(oldBox, newBox) => {
                        if (newBox.width < 10 || newBox.height < 10) {
                          return oldBox;
                        }
                        return newBox;
                      }}
                      padding={5}
                      enabledAnchors={[
                        'top-left', 'top-center', 'top-right',
                        'middle-left', 'middle-right',
                        'bottom-left', 'bottom-center', 'bottom-right'
                      ]}
                      rotateEnabled={false}
                    />
                    
                    {/* Guide lines - only in normal mode */}
                    {!isPanelFocusMode && showGuides && guides.x.map((x, i) => (
                      <Line 
                        key={`x-${i}`}
                        points={[x, 0, x, pageSize.height * scale]}
                        stroke="#0066FF"
                        strokeWidth={1}
                        dash={[5, 5]}
                      />
                    ))}
                    {!isPanelFocusMode && showGuides && guides.y.map((y, i) => (
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
                {panels.map(panel => {
                  // Only show overlay for generating panels that are visible
                  if (!panel.isGenerating) return null;
                  if (isPanelFocusMode && panel.id !== focusedPanelId) return null;
                  
                  const panelX = isPanelFocusMode && panel.id === focusedPanelId ? 0 : panel.x;
                  const panelY = isPanelFocusMode && panel.id === focusedPanelId ? 0 : panel.y;
                  
                  return (
                    <div 
                      key={`overlay-${panel.id}`}
                      className="absolute bg-black bg-opacity-50 flex justify-center items-center"
                      style={{
                        left: panelX * currentScale,
                        top: panelY * currentScale,
                        width: panel.width * currentScale,
                        height: panel.height * currentScale
                      }}
                    >
                      <div className="text-white text-lg">
                        {panel.generationQueued 
                          ? `Queued: ${panel.queueMessage || 'Waiting...'}`
                          : 'Generating...'}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
  
          {/* Panel Editor - Right Sidebar */}
          <div className="w-[32rem] bg-white border-l border-gray-200 overflow-y-auto overflow-x-hidden flex-shrink-0" style={{ maxHeight: '100vh - 128px' }}>
            {selectedPanel ? (
              <div className="p-4 max-w-full">
                <h2 className="text-2xl font-bold mb-4">Panel Editor</h2>
                
                <div className="space-y-6 max-w-full">
                  {/* Character Section with Box Drawing capability */}
                  <div className="pb-4 border-b border-gray-200">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="text-lg font-semibold">Characters</h3>
                      <div className="flex space-x-2">
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
                        
                        {selectedPanel.characterNames.length > 0 && (
                          <button
                            className={`px-3 py-2 rounded-md flex items-center ${panelMode === 'character-box' ? 'bg-indigo-600 text-white' : 'bg-indigo-100 text-indigo-700'}`}
                            onClick={() => {
                              if (panelMode === 'character-box') {
                                setPanelMode('adjust');
                              } else {
                                setPanelMode('character-box');
                                setActiveCharacter(selectedPanel.characterNames[0]); // Default to first character
                                setSelectedBoxType(null);
                                setSelectedBoxIndex(null);
                              }
                            }}
                            disabled={selectedPanel.characterNames.length === 0}
                          >
                            <Square size={16} className="mr-2" />
                            {panelMode === 'character-box' ? 'Cancel' : 'Draw Box'}
                          </button>
                        )}
                      </div>
                    </div>
                    
                    {/* Character list with chips */}
                    <div className="flex flex-wrap gap-2 mb-4">
                      {selectedPanel.characterNames.map((name, index) => (
                        <div 
                          key={`char-${index}`} 
                          className={`flex items-center rounded-full px-3 py-1 ${
                            activeCharacter === name && panelMode === 'character-box' 
                              ? 'bg-indigo-600 text-white' 
                              : 'bg-indigo-100 text-indigo-700'
                          }`}
                          onClick={() => {
                            if (panelMode === 'character-box') {
                              setActiveCharacter(name);
                            }
                          }}
                        >
                          <span className="mr-2">{name}</span>
                          <button 
                            onClick={(e) => {
                              e.stopPropagation(); // Prevent character selection when removing
                              handleRemoveCharacter(index);
                            }}
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
                    
                    {/* Character Boxes section */}
                    {selectedPanel.characterBoxes && selectedPanel.characterBoxes.length > 0 && (
                      <div className="mt-4">
                        <h4 className="font-medium text-sm mb-2">Character Placement Boxes:</h4>
                        <div className="max-h-40 overflow-y-auto space-y-2">
                          {selectedPanel.characterBoxes.map((box, idx) => (
                            <div 
                              key={idx} 
                              className={`flex items-center text-sm p-2 rounded-md border ${
                                selectedBoxType === 'character' && selectedBoxIndex === idx
                                  ? 'border-indigo-500 bg-indigo-50'
                                  : 'border-gray-200 hover:bg-gray-50'
                              }`}
                              onClick={() => {
                                handleBoxSelect('character', idx);
                                setPanelMode('adjust');
                              }}
                            >
                              <div className="w-4 h-4 mr-2" style={{backgroundColor: box.color}}></div>
                              <span className="flex-1">{box.character}</span>
                              <button
                                className="px-2 py-1 text-xs rounded bg-red-100 text-red-700 hover:bg-red-200"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteBox('character', idx);
                                }}
                              >
                                Delete
                              </button>
                            </div>
                          ))}
                        </div>
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
                      <h3 className="text-lg font-semibold">Dialogues</h3>
                      <div className="flex space-x-2">
                        <button
                          className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200"
                          onClick={handleAddDialogue}
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
                              handleBoxSelect('text', index);
                              setPanelMode('adjust');
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
                                  handleDeleteBox('text', index);
                                }}
                              >
                                Delete
                              </button>
                            </div>
                            
                            <div className="mb-2">
                              <label className="block text-sm font-medium text-black mb-1">Character</label>
                              <select
                                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                                value={dialogue.character}
                                onChange={(e) => {
                                  e.stopPropagation();
                                  handleUpdateDialogue(index, 'character', e.target.value);
                                }}
                                onClick={(e) => e.stopPropagation()}
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
                                onChange={(e) => {
                                  e.stopPropagation();
                                  handleUpdateDialogue(index, 'text', e.target.value);
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
                          No dialogues added. Click "Add Dialogue" to get started.
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
                      <label className="block text-sm font-medium text-black mb-1">Prompt</label>
                      <textarea
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                        value={selectedPanel.prompt || ''}
                        onChange={(e) => handleUpdatePrompt(e.target.value)}
                        placeholder="Deep in the undergrowth, ferns shake and a RAT emerges..."
                        rows={3}
                      />
                    </div>
                  </div>
                  
                  {/* Generate Panel Button */}
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

      {renderExportDialog()}
      {renderStatusMessage()}
  
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