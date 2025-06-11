// components/ProjectExport.tsx
import React, { useState, useRef } from 'react';
import { saveAs } from 'file-saver';
import { X } from 'lucide-react';
import { Stage, Layer, Rect, Image as KonvaImage } from 'react-konva';
import Konva from 'konva';
import { API_ENDPOINT, normalizeImagePath } from '../config';
import { Panel, Page, Project } from './MangaEditor';

export interface ExportConfig {
  type: 'png' | 'pdf';
  pageRange: 'all' | 'current' | 'custom';
  customRange: string;
  quality: 'normal' | 'high';
  includePanelOutlines: boolean;
}

interface ProjectExportProps {
  isOpen: boolean;
  onClose: () => void;
  currentProject?: Project | null;
  pages: Page[];
  currentPageIndex: number;
  pageSize: { width: number; height: number };
  apiEndpoint?: string;
}

const ProjectExport: React.FC<ProjectExportProps> = ({
  isOpen,
  onClose,
  currentProject,
  pages,
  currentPageIndex,
  pageSize,
  apiEndpoint = API_ENDPOINT
}) => {
  const [exportConfig, setExportConfig] = useState<ExportConfig>({
    type: 'png',
    pageRange: 'all',
    customRange: '',
    quality: 'normal',
    includePanelOutlines: false // Default to false as requested
  });

  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  // Ref for the hidden stage used for rendering
  const hiddenStageRef = useRef<any>(null);

  // Function to parse page range input
  const parsePageRange = (rangeInput: string, totalPages: number): number[] => {
    const pageIndices: number[] = [];
    
    if (!rangeInput.trim()) {
      return pageIndices;
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
            pageIndices.push(i - 1); // Convert to 0-based index
          }
        }
      } else {
        // Single page like "3"
        const pageNum = parseInt(part.trim());
        
        if (!isNaN(pageNum) && pageNum > 0 && pageNum <= totalPages) {
          pageIndices.push(pageNum - 1); // Convert to 0-based index
        }
      }
    }
    
    // Return unique pages in order
    return [...new Set(pageIndices)].sort((a, b) => a - b);
  };

  // Function to load image data for a panel
  const loadPanelImage = async (panel: Panel): Promise<string | null> => {
    // If we already have imageData, use it
    if (panel.imageData) {
      return panel.imageData;
    }
    
    // If we have an imagePath, fetch it
    if (panel.imagePath) {
      try {
        const normalizedPath = normalizeImagePath(panel.imagePath);
        const response = await fetch(normalizedPath);
        
        if (!response.ok) {
          console.error(`Failed to load image from path ${panel.imagePath}: ${response.statusText}`);
          return null;
        }
        
        const blob = await response.blob();
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      } catch (error) {
        console.error(`Error loading image for panel ${panel.id}:`, error);
        return null;
      }
    }
    
    return null;
  };

  // Function to render a page to canvas
  const renderPageToCanvas = async (pageIndex: number): Promise<string> => {
    const page = pages[pageIndex];
    if (!page) throw new Error(`Page ${pageIndex + 1} not found`);

    // First, load all images for this page
    const panelsWithImages = await Promise.all(
      page.panels.map(async (panel) => {
        const imageData = await loadPanelImage(panel);
        return {
          ...panel,
          loadedImageData: imageData
        };
      })
    );

    return new Promise((resolve, reject) => {
      // Create a temporary div to hold our stage
      const container = document.createElement('div');
      container.style.position = 'absolute';
      container.style.top = '-9999px';
      container.style.left = '-9999px';
      container.style.width = `${pageSize.width}px`;
      container.style.height = `${pageSize.height}px`;
      document.body.appendChild(container);

      try {
        // Create stage using imported Konva
        const stage = new Konva.Stage({
          container: container,
          width: pageSize.width,
          height: pageSize.height
        });

        const layer = new Konva.Layer();
        stage.add(layer);

        // Add page background
        const background = new Konva.Rect({
          width: pageSize.width,
          height: pageSize.height,
          fill: 'white'
        });
        layer.add(background);

        let imagesLoaded = 0;
        const totalImages = panelsWithImages.filter(panel => panel.loadedImageData).length;

        const checkComplete = () => {
          if (imagesLoaded === totalImages) {
            // Add panel outlines if requested
            if (exportConfig.includePanelOutlines) {
              panelsWithImages.forEach(panel => {
                const outline = new Konva.Rect({
                  x: panel.x,
                  y: panel.y,
                  width: panel.width,
                  height: panel.height,
                  stroke: '#000000',
                  strokeWidth: 2,
                  fill: 'transparent'
                });
                layer.add(outline);
              });
            }

            // Ensure layer is drawn before converting
            layer.draw();
            
            // Small delay to ensure rendering is complete
            setTimeout(() => {
              try {
                // Convert to data URL with specific settings for better compatibility
                const pixelRatio = exportConfig.quality === 'high' ? 2 : 1; // Reduced from 3/2 to 2/1
                const dataURL = stage.toDataURL({
                  pixelRatio,
                  mimeType: 'image/png',
                  quality: 1.0
                });

                // Cleanup
                stage.destroy();
                document.body.removeChild(container);

                resolve(dataURL);
              } catch (renderError) {
                console.error('Error converting stage to dataURL:', renderError);
                stage.destroy();
                document.body.removeChild(container);
                reject(new Error('Failed to convert stage to image'));
              }
            }, 100);
          }
        };

        // Add panel images
        panelsWithImages.forEach((panel, panelIndex) => {
          if (panel.loadedImageData) {
            const imageObj = new Image();
            imageObj.crossOrigin = 'anonymous'; // Add this for better compatibility
            
            imageObj.onload = () => {
              try {
                const konvaImage = new Konva.Image({
                  x: panel.x,
                  y: panel.y,
                  width: panel.width,
                  height: panel.height,
                  image: imageObj
                });
                layer.add(konvaImage);
                
                imagesLoaded++;
                checkComplete();
              } catch (imageError) {
                console.error(`Error adding image to layer for panel ${panel.id}:`, imageError);
                imagesLoaded++;
                checkComplete();
              }
            };
            
            imageObj.onerror = () => {
              console.error(`Failed to load image for panel ${panel.id}`);
              imagesLoaded++;
              checkComplete();
            };
            
            imageObj.src = panel.loadedImageData;
          }
        });

        // If no images, complete immediately
        if (totalImages === 0) {
          checkComplete();
        }

        // Timeout fallback
        setTimeout(() => {
          if (document.body.contains(container)) {
            console.error('Rendering timeout for page', pageIndex + 1);
            stage.destroy();
            document.body.removeChild(container);
            reject(new Error(`Rendering timeout for page ${pageIndex + 1}`));
          }
        }, 30000); // 30 second timeout

      } catch (error) {
        console.error('Error in renderPageToCanvas:', error);
        if (document.body.contains(container)) {
          document.body.removeChild(container);
        }
        reject(error);
      }
    });
  };

  // PNG Export (as ZIP file)
  const exportAsPNG = async (pageIndices: number[]) => {
    const JSZip = (await import('jszip')).default;
    const zip = new JSZip();
    
    // Add metadata
    zip.file('metadata.json', JSON.stringify({
      name: currentProject?.name,
      created: new Date().toISOString(),
      pageCount: pageIndices.length,
      exportDate: new Date().toISOString(),
      panelOutlinesIncluded: exportConfig.includePanelOutlines
    }));
    
    const pagesFolder = zip.folder('pages');
    
    // Export each page
    for (let i = 0; i < pageIndices.length; i++) {
      const pageIndex = pageIndices[i];
      
      setExportProgress(Math.round((i / pageIndices.length) * 100));
      setStatusMessage(`Loading and rendering page ${pageIndex + 1}...`);
      
      try {
        const dataURL = await renderPageToCanvas(pageIndex);
        
        // Convert data URL to blob
        const imageData = dataURL.split(',')[1];
        const binaryData = atob(imageData);
        const array = new Uint8Array(binaryData.length);
        
        for (let j = 0; j < binaryData.length; j++) {
          array[j] = binaryData.charCodeAt(j);
        }
        
        pagesFolder?.file(`page-${pageIndex + 1}.png`, array);
      } catch (error) {
        console.error(`Error rendering page ${pageIndex + 1}:`, error);
        throw new Error(`Failed to render page ${pageIndex + 1}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
    
    setStatusMessage('Creating ZIP file...');
    
    // Generate the zip file
    const content = await zip.generateAsync({
      type: 'blob',
      compression: 'DEFLATE',
      compressionOptions: {
        level: 9
      }
    });
    
    // Create filename
    const fileName = `${currentProject?.name.replace(/\s+/g, '-') || 'manga'}-pages.zip`;
    
    // Trigger download
    saveAs(content, fileName);
  };

  // Client-side PDF generation
  const clientPDFExport = async (pageIndices: number[]) => {
    const { default: jsPDF } = await import('jspdf');
    
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    });
    
    const pdfWidth = 210;  // mm
    const pdfHeight = 297; // mm
    const margin = 10; // mm
    const mangaWidth = pdfWidth - (margin * 2);
    const mangaHeight = pdfHeight - (margin * 2);
    
    for (let i = 0; i < pageIndices.length; i++) {
      const pageIndex = pageIndices[i];
      
      setExportProgress(Math.round((i / pageIndices.length) * 100));
      setStatusMessage(`Loading and rendering page ${pageIndex + 1} for PDF...`);
      
      if (i > 0) {
        pdf.addPage();
      }
      
      try {
        const dataURL = await renderPageToCanvas(pageIndex);
        
        // Convert to JPEG for better PDF compatibility and smaller file size
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        await new Promise((resolve, reject) => {
          img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            
            // Fill with white background first
            ctx!.fillStyle = 'white';
            ctx!.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw the image
            ctx!.drawImage(img, 0, 0);
            
            // Convert to JPEG with higher quality
            const jpegDataURL = canvas.toDataURL('image/jpeg', 0.95);
            
            try {
              pdf.addImage(
                jpegDataURL,
                'JPEG',
                margin,
                margin,
                mangaWidth,
                mangaHeight,
                `page-${pageIndex + 1}`,
                'FAST' // Use FAST compression instead of MEDIUM
              );
              resolve(undefined);
            } catch (pdfError) {
              console.error('Error adding image to PDF:', pdfError);
              reject(pdfError);
            }
          };
          
          img.onerror = () => {
            reject(new Error('Failed to load rendered image'));
          };
          
          img.src = dataURL;
        });
        
      } catch (error) {
        console.error(`Error rendering page ${pageIndex + 1} for PDF:`, error);
        throw new Error(`Failed to render page ${pageIndex + 1} for PDF: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
    
    const fileName = `${currentProject?.name.replace(/\s+/g, '-') || 'manga'}.pdf`;
    
    try {
      pdf.save(fileName);
    } catch (saveError) {
      console.error('Error saving PDF:', saveError);
      throw new Error('Failed to save PDF file');
    }
  };

  // Server-side PDF generation
  const serverPDFExport = async (pageIndices: number[]) => {
    const pageImages = [];
    
    for (let i = 0; i < pageIndices.length; i++) {
      const pageIndex = pageIndices[i];
      
      setExportProgress(Math.round((i / pageIndices.length) * 50));
      setStatusMessage(`Loading and rendering page ${pageIndex + 1}...`);
      
      try {
        const dataURL = await renderPageToCanvas(pageIndex);
        pageImages.push({
          pageIndex,
          dataURL
        });
      } catch (error) {
        console.error(`Error rendering page ${pageIndex + 1}:`, error);
        throw new Error(`Failed to render page ${pageIndex + 1}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
    
    setStatusMessage('Sending to server for PDF generation...');
    setExportProgress(60);
    
    const response = await fetch(`${normalizeImagePath('/api/export/pdf')}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        projectId: currentProject?.id,
        projectName: currentProject?.name,
        pages: pageImages,
        quality: exportConfig.quality,
        includePanelOutlines: exportConfig.includePanelOutlines
      })
    });
    
    if (!response.ok) {
      throw new Error(`PDF generation failed: ${response.status}`);
    }
    
    const { downloadUrl } = await response.json();
    
    setExportProgress(80);
    setStatusMessage('Downloading PDF...');
    
    const pdfResponse = await fetch(downloadUrl);
    const pdfBlob = await pdfResponse.blob();
    
    const fileName = `${currentProject?.name.replace(/\s+/g, '-') || 'manga'}.pdf`;
    saveAs(pdfBlob, fileName);
  };

  // PDF Export
  const exportAsPDF = async (pageIndices: number[]) => {
    // Try server-side PDF generation first
    if (apiEndpoint) {
      try {
        await serverPDFExport(pageIndices);
        return;
      } catch (error) {
        console.warn('Server-side PDF export failed, falling back to client-side:', error);
      }
    }
    
    // Fallback to client-side PDF generation
    await clientPDFExport(pageIndices);
  };
  const handleExport = async () => {
    if (!currentProject || !pages.length) {
      setStatusMessage('No project or pages to export');
      return;
    }
    
    setIsExporting(true);
    setExportProgress(0);
    setStatusMessage('Preparing export...');
    
    try {
      // Parse page range
      let pagesToExport: number[];
      
      if (exportConfig.pageRange === 'current') {
        pagesToExport = [currentPageIndex];
      } else if (exportConfig.pageRange === 'custom') {
        pagesToExport = parsePageRange(exportConfig.customRange, pages.length);
        
        if (pagesToExport.length === 0) {
          throw new Error('Invalid page range. Please use format like "1-3, 5, 7-9"');
        }
      } else {
        // 'all' is the default
        pagesToExport = Array.from({ length: pages.length }, (_, i) => i);
      }
      
      // Export based on type
      if (exportConfig.type === 'png') {
        await exportAsPNG(pagesToExport);
      } else {
        await exportAsPDF(pagesToExport);
      }
      
      setStatusMessage('Export completed successfully!');
      setExportProgress(100);
      
      // Close dialog after a brief delay
      setTimeout(() => {
        onClose();
      }, 1500);
      
    } catch (error) {
      console.error('Export error:', error);
      setStatusMessage(`Export failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setTimeout(() => {
        setIsExporting(false);
        setExportProgress(0);
        setStatusMessage('');
      }, 2000);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-bold">Export Manga</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
            disabled={isExporting}
          >
            <X className="h-6 w-6" />
          </button>
        </div>
        
        {isExporting ? (
          <div className="space-y-4">
            <div className="text-center">
              <div className="text-lg font-medium mb-2">Exporting...</div>
              <div className="text-sm text-gray-600 mb-4">{statusMessage}</div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${exportProgress}%` }}
                ></div>
              </div>
              <div className="text-sm text-gray-500 mt-2">{exportProgress}%</div>
            </div>
          </div>
        ) : (
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
                  <span className="ml-2">All Pages ({pages.length})</span>
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

            {/* Panel Outlines Toggle */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Panel Appearance</label>
              <label className="inline-flex items-center">
                <input
                  type="checkbox"
                  className="form-checkbox h-4 w-4 text-indigo-600"
                  checked={exportConfig.includePanelOutlines}
                  onChange={(e) => setExportConfig(prev => ({ ...prev, includePanelOutlines: e.target.checked }))}
                />
                <span className="ml-2">Include panel outlines</span>
              </label>
              <p className="text-xs text-gray-500 mt-1">
                Uncheck to export without editor panel borders (recommended when AI generates its own outlines)
              </p>
            </div>
            
            {/* Export Button */}
            <div className="pt-4">
              <button
                onClick={handleExport}
                className="w-full px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Export
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProjectExport;