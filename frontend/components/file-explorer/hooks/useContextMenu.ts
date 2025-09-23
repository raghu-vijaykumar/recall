import { FolderTreeNode, Workspace } from '../../core/types.js';

export const useContextMenu = (
  contextMenu: { x: number; y: number; item: FolderTreeNode | null } | null,
  workspace: Workspace | null,
  copiedItemPath: string | null,
  setContextMenu: (menu: { x: number; y: number; item: FolderTreeNode | null } | null) => void,
  setSelectedItem: (item: FolderTreeNode | null) => void,
  setCreateBasePath: (path: string) => void,
  setShowCreateModal: (show: boolean) => void,
  setNewName: (name: string) => void,
  setShowRenameModal: (show: boolean) => void,
  setShowMoveModal: (show: boolean) => void,
  setCopiedItemPath: (path: string | null) => void,
  setShowDeleteConfirm: (show: boolean) => void,
  handleCreateDirectory: (basePath: string, name: string) => void,
  loadFolderTree: () => void
) => {
  const handleContextMenu = (e: React.MouseEvent, item: FolderTreeNode | null) => {
    e.preventDefault();

    // Calculate position to keep menu within viewport
    const menuWidth = 180;
    const menuHeight = 200; // Approximate height
    let x = e.clientX;
    let y = e.clientY;

    // Adjust horizontal position
    if (x + menuWidth > window.innerWidth) {
      x = window.innerWidth - menuWidth - 10;
    }

    // Adjust vertical position
    if (y + menuHeight > window.innerHeight) {
      y = window.innerHeight - menuHeight - 10;
    }

    setContextMenu({ x, y, item });
  };

  const handleContextMenuAction = (action: string) => {
    if (!contextMenu?.item) return;

    setSelectedItem(contextMenu.item);
    setContextMenu(null);

    const basePath = workspace?.folder_path ? `${workspace.folder_path}/${contextMenu.item.path}` : '';

    switch (action) {
      case 'new-file':
        setCreateBasePath(basePath);
        setShowCreateModal(true);
        break;
      case 'new-folder':
        const folderName = prompt('Enter folder name:');
        if (folderName && basePath) {
          handleCreateDirectory(basePath, folderName);
        }
        break;
      case 'rename':
        setNewName(contextMenu.item.name);
        setShowRenameModal(true);
        break;
      case 'move':
        setNewName(contextMenu.item.path);
        setShowMoveModal(true);
        break;
      case 'copy':
        setCopiedItemPath(`${workspace?.folder_path}/${contextMenu.item.path}`);
        break;
      case 'paste':
        if (copiedItemPath && basePath) {
          (window as any).electronAPI.copyFile(copiedItemPath, basePath).then(() => {
            loadFolderTree();
          }).catch((error: any) => {
            console.error('Failed to paste:', error);
            alert('Failed to paste');
          });
        }
        break;
      case 'delete':
        setShowDeleteConfirm(true);
        break;
    }
  };

  return {
    handleContextMenu,
    handleContextMenuAction,
  };
};
