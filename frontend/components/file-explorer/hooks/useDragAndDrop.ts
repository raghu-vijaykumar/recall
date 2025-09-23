import { FolderTreeNode, Workspace } from '../../core/types.js';

export const useDragAndDrop = (
  draggedItem: FolderTreeNode | null,
  workspace: Workspace | null,
  setDraggedItem: (item: FolderTreeNode | null) => void,
  loadFolderTree: () => void
) => {
  const handleDragStart = (e: React.DragEvent, item: FolderTreeNode) => {
    setDraggedItem(item);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = async (e: React.DragEvent, targetItem: FolderTreeNode) => {
    e.preventDefault();
    if (!draggedItem || !workspace?.folder_path || draggedItem.path === targetItem.path) return;

    // Only allow dropping on directories
    if (targetItem.type !== 'directory') return;

    const sourcePath = `${workspace.folder_path}/${draggedItem.path}`;
    const destPath = `${workspace.folder_path}/${targetItem.path}`;

    try {
      await (window as any).electronAPI.moveFile(sourcePath, destPath);
      setDraggedItem(null);
      loadFolderTree();
    } catch (error) {
      console.error('Failed to move item:', error);
      alert('Failed to move item');
      setDraggedItem(null);
    }
  };

  return {
    handleDragStart,
    handleDragOver,
    handleDrop,
  };
};
