import { FolderTreeNode, Workspace } from '../../core/types.js';

export const useFileOperations = (
  workspace: Workspace | null,
  selectedItem: FolderTreeNode | null,
  newName: string,
  fileName: string,
  createBasePath: string,
  setShowCreateModal: (show: boolean) => void,
  setShowRenameModal: (show: boolean) => void,
  setShowMoveModal: (show: boolean) => void,
  setShowCopyModal: (show: boolean) => void,
  setShowDeleteConfirm: (show: boolean) => void,
  setFileName: (name: string) => void,
  setNewName: (name: string) => void,
  setSelectedItem: (item: FolderTreeNode | null) => void,
  setCreateBasePath: (path: string) => void,
  loadFolderTree: () => void
) => {
  const handleCreateFile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!fileName.trim() || !workspace?.folder_path) return;

    const basePath = createBasePath || workspace.folder_path;

    try {
      await (window as any).electronAPI.createFile(basePath, fileName);
      setShowCreateModal(false);
      setFileName('');
      setCreateBasePath('');
      loadFolderTree();
    } catch (error) {
      console.error('Failed to create file:', error);
      alert('Failed to create file');
    }
  };

  const handleCreateDirectory = async (basePath: string, name: string) => {
    if (!name.trim()) return;

    try {
      await (window as any).electronAPI.createDirectory(basePath, name);
      loadFolderTree();
    } catch (error) {
      console.error('Failed to create directory:', error);
      alert('Failed to create directory');
    }
  };

  const handleRename = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim() || !selectedItem || !workspace?.folder_path) return;

    const fullPath = `${workspace.folder_path}/${selectedItem.path}`;

    try {
      await (window as any).electronAPI.renameFile(fullPath, newName);
      setShowRenameModal(false);
      setNewName('');
      setSelectedItem(null);
      loadFolderTree();
    } catch (error) {
      console.error('Failed to rename:', error);
      alert('Failed to rename');
    }
  };

  const handleMove = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim() || !selectedItem || !workspace?.folder_path) return;

    const sourcePath = `${workspace.folder_path}/${selectedItem.path}`;
    const destPath = `${workspace.folder_path}/${newName}`;

    try {
      await (window as any).electronAPI.moveFile(sourcePath, destPath);
      setShowMoveModal(false);
      setNewName('');
      setSelectedItem(null);
      loadFolderTree();
    } catch (error) {
      console.error('Failed to move:', error);
      alert('Failed to move');
    }
  };

  const handleCopy = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim() || !selectedItem || !workspace?.folder_path) return;

    const sourcePath = `${workspace.folder_path}/${selectedItem.path}`;
    const destPath = `${workspace.folder_path}/${newName}`;

    try {
      await (window as any).electronAPI.copyFile(sourcePath, destPath);
      setShowCopyModal(false);
      setNewName('');
      setSelectedItem(null);
      loadFolderTree();
    } catch (error) {
      console.error('Failed to copy:', error);
      alert('Failed to copy');
    }
  };

  const handleDelete = async () => {
    if (!selectedItem || !workspace?.folder_path) return;

    const fullPath = `${workspace.folder_path}/${selectedItem.path}`;

    try {
      await (window as any).electronAPI.deleteFile(fullPath);
      setShowDeleteConfirm(false);
      setSelectedItem(null);
      loadFolderTree();
    } catch (error) {
      console.error('Failed to delete:', error);
      alert('Failed to delete');
    }
  };

  return {
    handleCreateFile,
    handleCreateDirectory,
    handleRename,
    handleMove,
    handleCopy,
    handleDelete,
  };
};
