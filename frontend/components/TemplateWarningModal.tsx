import React from 'react';

interface TemplateWarningModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  templateName: string;
}

const TemplateWarningModal: React.FC<TemplateWarningModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  templateName,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
        <div className="flex items-center mb-4">
          <div className="flex-shrink-0">
            <svg 
              className="h-8 w-8 text-orange-500" 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.882 16.5c-.77.833.192 2.5 1.732 2.5z" 
              />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-lg font-medium text-gray-900">
              Apply Template
            </h3>
          </div>
        </div>
        
        <div className="mb-6">
          <p className="text-gray-700">
            Applying the <strong>"{templateName}"</strong> template will <strong>replace all existing panels</strong> on the current page. 
            This action cannot be undone.
          </p>
          <p className="text-gray-600 mt-2 text-sm">
            Are you sure you want to continue?
          </p>
        </div>
        
        <div className="flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 focus:outline-none focus:ring-2 focus:ring-orange-500"
          >
            Apply Template
          </button>
        </div>
      </div>
    </div>
  );
};

export default TemplateWarningModal;