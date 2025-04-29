// mangaui/frontend/src/components/PageComposer.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Stage, Layer, Rect, Image as KonvaImage } from 'react-konva';

interface PanelData {
  index: number;
  imageData: string;
  data: any;
}

interface PageComposerProps {
  // Add any props here if needed
}

const PageComposer: React.FC<PageComposerProps> = () => {
  const [availablePanels, setAvailablePanels] = useState<PanelData[]>([]);
  const [selectedPanels, setSelectedPanels] = useState<PanelData[]>([]);
  const [pageLayout, setPageLayout] = useState<'grid' | 'vertical' | 'custom'>('grid');
  const [pageIndex, setPageIndex] = useState<number>(0);
  const [generatedPage, setGeneratedPage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  const pageRef = useRef<any>(null);
  const pageSize = { width: 1654, height: 2339 }; // A4 proportions
  const previewScale = 0.3; // Scale the preview to fit the screen
  
  // Effect to load available panels on mount
  useEffect(() => {
    fetchAvailablePanels();
  }, []);
  
  // Function to fetch available panels from the server
  const fetchAvailablePanels = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:5000/api/get_generated_panels');
      const data = await response.json();
      
      if (data.status === 'success') {
        setAvailablePanels(data.panels);
      } else {
        console.error('Error fetching panels:', data.message);
      }
    } catch (error) {
      console.error('Error fetching panels:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handler for selecting a panel
  const handleSelectPanel = (panel: PanelData) => {
    if (!selectedPanels.some(p => p.index === panel.index)) {
      setSelectedPanels([...selectedPanels, panel]);
    }
  };
  
  // Handler for removing a panel from selection
  const handleRemovePanel = (index: number) => {
    setSelectedPanels(selectedPanels.filter((_, i) => i !== index));
  };
  
  // Handler for changing the page layout
  const handleChangeLayout = (layout: 'grid' | 'vertical' | 'custom') => {
    setPageLayout(layout);
  };
  
  // Handler for creating the page
  const handleCreatePage = async () => {
    if (selectedPanels.length === 0) {
      alert('Please select at least one panel');
      return;
    }
    
    try {
      setIsLoading(true);
      // Call the API to create the page
      const response = await fetch('http://localhost:5000/api/create_page', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          panelIndices: selectedPanels.map(p => p.index),
          layout: pageLayout,
          pageIndex
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setGeneratedPage(data.imageData);
      } else {
        console.error('Error creating page:', data.message);
        alert(`Error creating page: ${data.message}`);
      }
    } catch (error) {
      console.error('Error calling API:', error);
      alert(`Error calling API: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handler for saving the page
  const handleSavePage = () => {
    if (!generatedPage) {
      alert('Please generate a page first');
      return;
    }
    
    // Create a download link
    const link = document.createElement('a');
    link.href = generatedPage;
    link.download = `page_${pageIndex.toString().padStart(3, '0')}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  // Generate the panel preview layout based on the selected layout
  const getPanelLayout = () => {
    const panels = selectedPanels.map(p => p.index);
    const layout: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      panelId: number;
    }> = [];
    
    if (pageLayout === 'grid') {
      // Grid layout (2x2 or 3x2 depending on panel count)
      const numPanels = panels.length;
      
      if (numPanels <= 2) {
        // Simple vertical stack
        const panelHeight = pageSize.height / numPanels;
        
        for (let i = 0; i < numPanels; i++) {
          layout.push({
            x: 0,
            y: i * panelHeight,
            width: pageSize.width,
            height: panelHeight,
            panelId: panels[i]
          });
        }
      } else if (numPanels <= 4) {
        // 2x2 grid
        const panelWidth = pageSize.width / 2;
        const panelHeight = pageSize.height / 2;
        
        for (let i = 0; i < Math.min(numPanels, 4); i++) {
          const row = Math.floor(i / 2);
          const col = i % 2;
          
          layout.push({
            x: col * panelWidth,
            y: row * panelHeight,
            width: panelWidth,
            height: panelHeight,
            panelId: panels[i]
          });
        }
      } else {
        // 3x2 grid (up to 6 panels)
        const panelWidth = pageSize.width / 2;
        const panelHeight = pageSize.height / 3;
        
        for (let i = 0; i < Math.min(numPanels, 6); i++) {
          const row = Math.floor(i / 2);
          const col = i % 2;
          
          layout.push({
            x: col * panelWidth,
            y: row * panelHeight,
            width: panelWidth,
            height: panelHeight,
            panelId: panels[i]
          });
        }
      }
    } else if (pageLayout === 'vertical') {
      // Simple vertical stack
      const numPanels = panels.length;
      const panelHeight = pageSize.height / numPanels;
      
      for (let i = 0; i < numPanels; i++) {
        layout.push({
          x: 0,
          y: i * panelHeight,
          width: pageSize.width,
          height: panelHeight,
          panelId: panels[i]
        });
      }
    }
    
    return layout;
  };
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-4">
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h2 className="text-2xl font-bold mb-4">Page Preview</h2>
        
        <div className="relative mx-auto" style={{ width: `${pageSize.width * previewScale}px`, height: `${pageSize.height * previewScale}px` }}>
          <Stage 
            width={pageSize.width * previewScale} 
            height={pageSize.height * previewScale}
            ref={pageRef}
            className="bg-gray-100 shadow-inner"
          >
            <Layer>
              <Rect
                width={pageSize.width * previewScale}
                height={pageSize.height * previewScale}
                fill="white"
                stroke="black"
              />
              
              {generatedPage ? (
                <KonvaImage
                  image={(() => {
                    const img = new window.Image();
                    img.src = generatedPage;
                    return img;
                  })()}
                  width={pageSize.width * previewScale}
                  height={pageSize.height * previewScale}
                />
              ) : (
                getPanelLayout().map((item, index) => (
                  <Rect
                    key={`layout-${index}`}
                    x={item.x * previewScale}
                    y={item.y * previewScale}
                    width={item.width * previewScale}
                    height={item.height * previewScale}
                    stroke="black"
                    fill="#f0f0f0"
                  />
                ))
              )}
            </Layer>
          </Stage>
        </div>
        
        <div className="mt-4">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Page Index</label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                value={pageIndex}
                onChange={(e) => setPageIndex(parseInt(e.target.value) || 0)}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Layout</label>
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                value={pageLayout}
                onChange={(e) => handleChangeLayout(e.target.value as 'grid' | 'vertical' | 'custom')}
              >
                <option value="grid">Grid</option>
                <option value="vertical">Vertical Stack</option>
                <option value="custom">Custom</option>
              </select>
            </div>
          </div>
          
          <div className="flex space-x-4">
            <button
              className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
              onClick={handleCreatePage}
              disabled={isLoading || selectedPanels.length === 0}
            >
              {isLoading ? 'Creating...' : 'Create Page'}
            </button>
            
            <button
              className="flex-1 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400"
              onClick={handleSavePage}
              disabled={!generatedPage}
            >
              Save Page
            </button>
          </div>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow-lg p-4">
        <div>
          <h2 className="text-2xl font-bold mb-4">Selected Panels</h2>
          <div className="flex flex-wrap gap-2 mb-6">
            {selectedPanels.map((panel, index) => (
              <div 
                key={`selected-${panel.index}`} 
                className="relative group"
              >
                <img 
                  src={panel.imageData} 
                  alt={`Panel ${panel.index}`}
                  className="w-24 h-24 object-cover border-2 border-indigo-500 rounded"
                />
                <button
                  onClick={() => handleRemovePanel(index)}
                  className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  &times;
                </button>
                <div className="text-center text-xs mt-1">#{panel.index}</div>
              </div>
            ))}
            
            {selectedPanels.length === 0 && (
              <div className="text-gray-500 italic">No panels selected</div>
            )}
          </div>
        </div>
        
        <div>
          <h2 className="text-2xl font-bold mb-4">Available Panels</h2>
          {isLoading ? (
            <div className="flex justify-center items-center h-64">
              <div className="text-gray-500">Loading panels...</div>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 max-h-96 overflow-y-auto p-2">
              {availablePanels.map((panel) => (
                <div
                  key={`available-${panel.index}`}
                  className="cursor-pointer group"
                  onClick={() => handleSelectPanel(panel)}
                >
                  <img
                    src={panel.imageData}
                    alt={`Panel ${panel.index}`}
                    className="w-full aspect-square object-cover border border-gray-300 rounded group-hover:border-indigo-500 transition-colors"
                  />
                  <div className="text-center text-xs mt-1">Panel #{panel.index}</div>
                </div>
              ))}
              
              {availablePanels.length === 0 && (
                <div className="col-span-full text-gray-500 italic text-center">
                  No panels available. Generate some panels first.
                </div>
              )}
            </div>
          )}
          
          <button
            className="mt-4 w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
            onClick={fetchAvailablePanels}
          >
            Refresh Panels
          </button>
        </div>
      </div>
    </div>
  );
};

export default PageComposer;