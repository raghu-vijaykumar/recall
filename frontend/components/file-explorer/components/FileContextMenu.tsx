import React from 'react';
import { FolderTreeNode } from '../../core/types.js';

interface FileContextMenuProps {
  contextMenu: { x: number; y: number; item: FolderTreeNode | null } | null;
  copiedItemPath: string | null;
  handleContextMenuAction: (action: string) => void;
  setContextMenu: (menu: { x: number; y: number; item: FolderTreeNode | null } | null) => void;
}

const FileContextMenu: React.FC<FileContextMenuProps> = ({
  contextMenu,
  copiedItemPath,
  handleContextMenuAction,
  setContextMenu,
}) => {
  if (!contextMenu) return null;

  return (
    <div
      className="context-menu"
      style={{ left: contextMenu.x, top: contextMenu.y }}
      onClick={() => setContextMenu(null)}
    >
      <div className="context-menu-content" onClick={(e) => e.stopPropagation()}>
        <button onClick={() => handleContextMenuAction('new-file')}>New File</button>
        <button onClick={() => handleContextMenuAction('new-folder')}>New Folder</button>
        {contextMenu.item && (
          <>
            <hr />
            <button onClick={() => handleContextMenuAction('rename')}>Rename</button>
            <button onClick={() => handleContextMenuAction('move')}>Move</button>
            <button onClick={() => handleContextMenuAction('copy')}>Copy</button>
            {copiedItemPath && (
              <button onClick={() => handleContextMenuAction('paste')}>Paste</button>
            )}
            <hr />
            <button
              className="danger"
              onClick={() => handleContextMenuAction('delete')}
            >
              Delete
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default FileContextMenu;
