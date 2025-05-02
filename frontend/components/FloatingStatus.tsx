interface FloatingStatusProps {
  status: string;
  message: string;
  progress: number;
  isMinimized: boolean;
  onToggle: () => void;
}

// Simple floating status component
export const FloatingStatus: React.FC<FloatingStatusProps> = ({ 
  status, 
  message, 
  progress, 
  isMinimized, 
  onToggle 
}) => {
  if (!status) return null;
  
  const getStatusColor = () => {
    switch (status) {
      case 'success': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-indigo-600';
    }
  };
  
  if (isMinimized) {
    return (
      <div 
        className={`fixed bottom-4 right-4 ${getStatusColor()} text-white rounded-full p-2 shadow-lg flex items-center cursor-pointer z-50`}
        onClick={onToggle}
      >
        {status === 'in_progress' && (
          <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
        )}
        {status === 'success' && (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        )}
        {status === 'error' && (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        )}
      </div>
    );
  }
  
  return (
    <div className={`fixed bottom-4 right-4 ${getStatusColor()} text-white rounded-lg shadow-lg p-4 max-w-sm z-50`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-bold">Model Initialization</h3>
        <div className="flex space-x-2">
          <button 
            className="text-white hover:text-gray-200"
            onClick={onToggle}
            title="Minimize"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>
      
      <div className="flex items-center mb-2">
        {status === 'in_progress' && (
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
        )}
        <span>{message}</span>
      </div>
      
      {status === 'in_progress' && progress > 0 && (
        <div className="w-full bg-white bg-opacity-30 rounded-full h-2">
          <div 
            className="bg-white h-2 rounded-full" 
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      )}
    </div>
  );
};