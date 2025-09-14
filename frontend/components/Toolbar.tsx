import React from 'react';
import { 
  Plus, 
  Save, 
  FileDown, 
  Folder, 
  Layout,
  Trash2,
  Undo,
  Redo,
} from 'lucide-react';

interface ToolbarProps {
  selectedPanelId: string | null;
  isSaving: boolean;
  hasUnsavedChanges: boolean;
  canUndo?: boolean;
  canRedo?: boolean;
  onAddPanel: () => void;
  onDeletePanel: () => void;
  onShowTemplateDialog: () => void;
  onSaveProject: () => void;
  onShowProjectManager?: () => void;
  onShowExport?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({
  selectedPanelId,
  isSaving,
  hasUnsavedChanges,
  canUndo = false,
  canRedo = false,
  onAddPanel,
  onDeletePanel,
  onShowTemplateDialog,
  onSaveProject,
  onShowProjectManager,
  onShowExport,
  onUndo,
  onRedo,
}) => {
  return (
    <div className="w-14 bg-gray-100 shadow-md flex flex-col items-center py-4 space-y-4">
      {/* Add Panel */}
      <div className="relative group">
        <button
          className="p-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
          onClick={onAddPanel}
        >
          <Plus size={20} />
        </button>
        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-indigo-600 text-white text-sm px-3 py-2 rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
          Add Panel
          <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-indigo-600"></div>
        </div>
      </div>

      {/* Delete Panel */}
      <div className="relative group">
        <button
          className="p-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400 disabled:hover:bg-gray-400"
          onClick={onDeletePanel}
          disabled={!selectedPanelId}
        >
          <Trash2 size={20} />
        </button>
        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-indigo-600 text-white text-sm px-3 py-2 rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
          Delete Panel
          <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-indigo-600"></div>
        </div>
      </div>

      {/* Divider */}
      <div className="w-8 h-px bg-gray-300"></div>

      {/* Undo */}
      <div className="relative group">
        <button
          className="p-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:bg-gray-300 disabled:hover:bg-gray-300"
          onClick={onUndo}
          disabled={!canUndo}
          title="Undo (Ctrl/Cmd + Z)"
        >
          <Undo size={20} />
        </button>
        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-gray-600 text-white text-sm px-3 py-2 rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
          Undo (Ctrl/Cmd + Z)
          <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-600"></div>
        </div>
      </div>

      {/* Redo */}
      <div className="relative group">
        <button
          className="p-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:bg-gray-300 disabled:hover:bg-gray-300"
          onClick={onRedo}
          disabled={!canRedo}
          title="Redo (Ctrl/Cmd + Shift + Z)"
        >
          <Redo size={20} />
        </button>
        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-gray-600 text-white text-sm px-3 py-2 rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
          Redo (Ctrl/Cmd + Shift + Z)
          <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-600"></div>
        </div>
      </div>

      {/* Page Templates */}
      <div className="relative group">
        <button
          className="p-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
          onClick={onShowTemplateDialog}
        >
          <Layout size={20} />
        </button>
        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-indigo-600 text-white text-sm px-3 py-2 rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
          Page Templates
          <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-indigo-600"></div>
        </div>
      </div>

      {/* Export */}
      <div className="relative group">
        <button
          className="p-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
          onClick={() => onShowExport?.()}
        >
          <FileDown size={20} />
        </button>
        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-indigo-600 text-white text-sm px-3 py-2 rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
          Export Manga
          <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-indigo-600"></div>
        </div>
      </div>

      {/* Save Project */}
      <div className="relative group">
        <button
          className="p-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400 disabled:hover:bg-gray-400"
          onClick={onSaveProject}
          disabled={isSaving || !hasUnsavedChanges}
        >
          <Save size={20} />
        </button>
        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-indigo-600 text-white text-sm px-3 py-2 rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
          Save Project
          <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-indigo-600"></div>
        </div>
      </div>

      {/* Project Manager */}
      <div className="relative group">
        <button
          onClick={() => onShowProjectManager?.()}
          className="p-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
        >
          <Folder size={20} />
        </button>
        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-indigo-600 text-white text-sm px-3 py-2 rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
          Project Manager
          <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-indigo-600"></div>
        </div>
      </div>
    </div>
  );
};

export default Toolbar;