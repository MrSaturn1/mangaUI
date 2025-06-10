// components/ProjectExport.tsx
import React, { useState } from 'react';
import { saveAs } from 'file-saver';
import { X } from 'lucide-react';
import { API_ENDPOINT, normalizeImagePath } from '../config';

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
  onExport: (config: ExportConfig) => Promise<void>;
  currentProject?: { id: string; name: string } | null;
  totalPages: number;
  currentPageIndex: number;
  isExporting: boolean;
  exportProgress: number;
}

const ProjectExport: React.FC<ProjectExportProps> = ({
  isOpen,
  onClose,
  onExport,
  currentProject,
  totalPages,
  currentPageIndex,
  isExporting,
  exportProgress
}) => {
  const [exportConfig, setExportConfig] = useState<ExportConfig>({
    type: 'png',
    pageRange: 'all',
    customRange: '',
    quality: 'normal',
    includePanelOutlines: true
  });

  const handleExport = async () => {
    onClose();
    await onExport(exportConfig);
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
          >
            <X className="h-6 w-6" />
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
                <span className="ml-2">All Pages ({totalPages})</span>
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
              Uncheck to export without editor panel borders (useful when Drawatoon generates its own outlines)
            </p>
          </div>
          
          {/* Export Button */}
          <div className="pt-4">
            <button
              onClick={handleExport}
              disabled={isExporting}
              className="w-full px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
            >
              {isExporting ? `Exporting... ${exportProgress}%` : 'Export'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectExport;