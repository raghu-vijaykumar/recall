import React from 'react';
import { FolderTreeNode, Tab } from '../../../src/core/types';
import { getFileIcon } from '../../../src/shared/utils';

interface FileTreeProps {
  folderTree: FolderTreeNode[];
  expandedDirs: Set<string>;
  openFiles: Tab[];
  draggedItem: FolderTreeNode | null;
  searchQuery: string;
  toggleDirectory: (path: string) => void;
  openFile: (node: FolderTreeNode) => void;
  handleContextMenu: (e: React.MouseEvent, item: FolderTreeNode | null) => void;
  handleDragStart: (e: React.DragEvent, item: FolderTreeNode) => void;
  handleDragOver: (e: React.DragEvent) => void;
  handleDrop: (e: React.DragEvent, targetItem: FolderTreeNode) => void;
  filterTreeBySearch: (nodes: FolderTreeNode[], query: string) => FolderTreeNode[];
}

const FileTree: React.FC<FileTreeProps> = ({
  folderTree,
  expandedDirs,
  openFiles,
  draggedItem,
  searchQuery,
  toggleDirectory,
  openFile,
  handleContextMenu,
  handleDragStart,
  handleDragOver,
  handleDrop,
  filterTreeBySearch,
}) => {
  const renderTreeNode = (node: FolderTreeNode, level: number = 0): React.ReactNode => {
    const isExpanded = expandedDirs.has(node.path);
    const isActive = openFiles.some(tab => tab.file.path === node.path);

    return (
      <div key={node.path}>
        <div
          className={`tree-item ${isActive ? 'active' : ''} ${draggedItem?.path === node.path ? 'dragging' : ''}`}
          style={{ paddingLeft: `${level * 16 + 8}px` }}
          onClick={() => node.type === 'directory' ? toggleDirectory(node.path) : openFile(node)}
          onContextMenu={(e) => handleContextMenu(e, node)}
          draggable={true}
          onDragStart={(e) => handleDragStart(e, node)}
          onDragOver={node.type === 'directory' ? handleDragOver : undefined}
          onDrop={node.type === 'directory' ? (e) => handleDrop(e, node) : undefined}
        >
          {node.type === 'directory' ? (
            <span className="tree-icon">
              {isExpanded ? '📂' : '📁'}
            </span>
          ) : (
            <span className="tree-icon">{getFileIcon(node.name)}</span>
          )}
          <span className="tree-name">{node.name}</span>
        </div>
        {node.type === 'directory' && isExpanded && node.children && (
          <div>
            {node.children.map(child => renderTreeNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  const filteredTree = searchQuery ? filterTreeBySearch(folderTree, searchQuery) : folderTree;

  return (
    <div className="file-tree">
      {filteredTree.length > 0 ? (
        filteredTree.map(node => renderTreeNode(node))
      ) : searchQuery ? (
        <div className="empty-tree">
          <p>No files match your search</p>
        </div>
      ) : (
        <div className="empty-tree">
          <p>No files found in workspace folder</p>
        </div>
      )}
    </div>
  );
};

export default FileTree;
