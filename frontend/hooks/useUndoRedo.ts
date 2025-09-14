import { useState, useCallback, useRef } from 'react';

export interface UndoRedoAction {
  type: string;
  description: string;
  timestamp: number;
  undo: () => void;
  redo: () => void;
}

export interface UndoRedoState<T> {
  present: T;
  past: UndoRedoAction[];
  future: UndoRedoAction[];
  maxHistorySize: number;
}

export interface UseUndoRedoReturn<T> {
  state: T;
  canUndo: boolean;
  canRedo: boolean;
  undo: () => void;
  redo: () => void;
  pushAction: (action: Omit<UndoRedoAction, 'timestamp'>) => void;
  clearHistory: () => void;
  getHistorySize: () => number;
}

export function useUndoRedo<T>(
  initialState: T,
  maxHistorySize: number = 50
): UseUndoRedoReturn<T> {
  const [undoRedoState, setUndoRedoState] = useState<UndoRedoState<T>>({
    present: initialState,
    past: [],
    future: [],
    maxHistorySize
  });

  // Track if we're currently performing an undo/redo to prevent recursion
  const isPerformingAction = useRef(false);

  const pushAction = useCallback((actionData: Omit<UndoRedoAction, 'timestamp'>) => {
    if (isPerformingAction.current) return;

    setUndoRedoState(prev => {
      const action: UndoRedoAction = {
        ...actionData,
        timestamp: Date.now()
      };

      let newPast = [...prev.past, action];
      
      // Limit history size
      if (newPast.length > maxHistorySize) {
        newPast = newPast.slice(-maxHistorySize);
      }

      return {
        ...prev,
        past: newPast,
        future: [] // Clear future when new action is performed
      };
    });
  }, [maxHistorySize]);

  const undo = useCallback(() => {
    if (undoRedoState.past.length === 0) return;

    isPerformingAction.current = true;
    
    try {
      const actionToUndo = undoRedoState.past[undoRedoState.past.length - 1];
      
      // Perform the undo
      actionToUndo.undo();
      
      setUndoRedoState(prev => ({
        ...prev,
        past: prev.past.slice(0, -1),
        future: [actionToUndo, ...prev.future]
      }));
    } finally {
      isPerformingAction.current = false;
    }
  }, [undoRedoState.past]);

  const redo = useCallback(() => {
    if (undoRedoState.future.length === 0) return;

    isPerformingAction.current = true;
    
    try {
      const actionToRedo = undoRedoState.future[0];
      
      // Perform the redo
      actionToRedo.redo();
      
      setUndoRedoState(prev => ({
        ...prev,
        past: [...prev.past, actionToRedo],
        future: prev.future.slice(1)
      }));
    } finally {
      isPerformingAction.current = false;
    }
  }, [undoRedoState.future]);

  const clearHistory = useCallback(() => {
    setUndoRedoState(prev => ({
      ...prev,
      past: [],
      future: []
    }));
  }, []);

  const getHistorySize = useCallback(() => {
    return undoRedoState.past.length + undoRedoState.future.length;
  }, [undoRedoState.past.length, undoRedoState.future.length]);

  return {
    state: undoRedoState.present,
    canUndo: undoRedoState.past.length > 0,
    canRedo: undoRedoState.future.length > 0,
    undo,
    redo,
    pushAction,
    clearHistory,
    getHistorySize
  };
}