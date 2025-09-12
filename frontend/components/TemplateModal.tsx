import React from 'react';
import { PageTemplate, pageTemplates } from '../utils/pageTemplates';

interface TemplateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onApplyTemplate: (templateId: string) => void;
  pageSize: { width: number; height: number };
}

const TemplateModal: React.FC<TemplateModalProps> = ({
  isOpen,
  onClose,
  onApplyTemplate,
  pageSize,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl p-6 max-h-[90vh] overflow-auto">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-bold">Choose a Page Template</h3>
          <button
            onClick={onClose}
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
              onClick={() => onApplyTemplate(template.id)}
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
  );
};

export default TemplateModal;