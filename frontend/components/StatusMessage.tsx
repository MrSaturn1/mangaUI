import React from 'react';

interface StatusMessageProps {
  show: boolean;
  message: string;
  type: 'success' | 'error' | 'info' | 'loading';
  onClose: () => void;
}

const StatusMessage: React.FC<StatusMessageProps> = ({ 
  show, 
  message, 
  type, 
  onClose 
}) => {
  if (!show) return null;

  const bgColors = {
    success: 'bg-green-50 border-green-500',
    error: 'bg-red-50 border-red-500', 
    info: 'bg-blue-50 border-blue-500',
    loading: 'bg-yellow-50 border-yellow-500'
  };

  const textColors = {
    success: 'text-green-700',
    error: 'text-red-700',
    info: 'text-blue-700', 
    loading: 'text-yellow-700'
  };

  const LoadingIcon = () => (
    <div className="animate-spin w-5 h-5 mr-2 border-2 border-dashed rounded-full border-current"></div>
  );

  return (
    <div className="fixed bottom-4 right-4 max-w-sm z-50">
      <div className={`p-3 rounded-lg shadow-md border-l-4 ${bgColors[type]}`}>
        <div className="flex items-center">
          <div className={textColors[type]}>
            {type === 'loading' && <LoadingIcon />}
          </div>
          <div className={`ml-3 ${textColors[type]}`}>
            <p className="text-sm font-medium">{message}</p>
          </div>
          <button 
            onClick={onClose}
            className={`ml-auto pl-3 ${textColors[type]} hover:opacity-70`}
          >
            <span className="sr-only">Close</span>
            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default StatusMessage;