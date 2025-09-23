import React from 'react';
import { FolderTreeNode } from '../../core/types.js';

interface FileOperationModalsProps {
  showCreateModal: boolean;
  showRenameModal: boolean;
  showMoveModal: boolean;
  showCopyModal: boolean;
  showDeleteConfirm: boolean;
  fileName: string;
  newName: string;
  selectedItem: FolderTreeNode | null;
  createBasePath: string;
  setFileName: (name: string) => void;
  setNewName: (name: string) => void;
  setSelectedItem: (item: FolderTreeNode | null) => void;
  setCreateBasePath: (path: string) => void;
  setShowCreateModal: (show: boolean) => void;
  setShowRenameModal: (show: boolean) => void;
  setShowMoveModal: (show: boolean) => void;
  setShowCopyModal: (show: boolean) => void;
  setShowDeleteConfirm: (show: boolean) => void;
  handleCreateFile: (e: React.FormEvent) => void;
  handleRename: (e: React.FormEvent) => void;
  handleMove: (e: React.FormEvent) => void;
  handleCopy: (e: React.FormEvent) => void;
  handleDelete: () => void;
}

const FileOperationModals: React.FC<FileOperationModalsProps> = ({
  showCreateModal,
  showRenameModal,
  showMoveModal,
  showCopyModal,
  showDeleteConfirm,
  fileName,
  newName,
  selectedItem,
  createBasePath,
  setFileName,
  setNewName,
  setSelectedItem,
  setCreateBasePath,
  setShowCreateModal,
  setShowRenameModal,
  setShowMoveModal,
  setShowCopyModal,
  setShowDeleteConfirm,
  handleCreateFile,
  handleRename,
  handleMove,
  handleCopy,
  handleDelete,
}) => {
  return (
    <>
      {showCreateModal && (
        <div className="modal">
          <div className="modal-content">
            <h3>Create New File</h3>
            <form onSubmit={handleCreateFile}>
              <input
                type="text"
                placeholder="filename.ext (e.g., script.py, notes.md)"
                value={fileName}
                onChange={(e) => setFileName(e.target.value)}
                required
              />
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => {
                    setShowCreateModal(false);
                    setFileName('');
                  }}
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary">Create</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showRenameModal && selectedItem && (
        <div className="modal">
          <div className="modal-content">
            <h3>Rename {selectedItem.type === 'directory' ? 'Folder' : 'File'}</h3>
            <form onSubmit={handleRename}>
              <input
                type="text"
                placeholder="New name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                required
              />
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => {
                    setShowRenameModal(false);
                    setNewName('');
                    setSelectedItem(null);
                  }}
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary">Rename</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showMoveModal && selectedItem && (
        <div className="modal">
          <div className="modal-content">
            <h3>Move {selectedItem.type === 'directory' ? 'Folder' : 'File'}</h3>
            <form onSubmit={handleMove}>
              <input
                type="text"
                placeholder="New path (relative to workspace root)"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                required
              />
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => {
                    setShowMoveModal(false);
                    setNewName('');
                    setSelectedItem(null);
                  }}
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary">Move</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showCopyModal && selectedItem && (
        <div className="modal">
          <div className="modal-content">
            <h3>Copy {selectedItem.type === 'directory' ? 'Folder' : 'File'}</h3>
            <form onSubmit={handleCopy}>
              <input
                type="text"
                placeholder="Destination path (relative to workspace root)"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                required
              />
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => {
                    setShowCopyModal(false);
                    setNewName('');
                    setSelectedItem(null);
                  }}
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary">Copy</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showDeleteConfirm && selectedItem && (
        <div className="modal">
          <div className="modal-content">
            <h3>Delete {selectedItem.type === 'directory' ? 'Folder' : 'File'}</h3>
            <p>Are you sure you want to delete "{selectedItem.name}"? This action cannot be undone.</p>
            <div className="modal-actions">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  setShowDeleteConfirm(false);
                  setSelectedItem(null);
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-danger"
                onClick={handleDelete}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default FileOperationModals;
