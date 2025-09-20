import React, { useState, useEffect } from 'react';
import { API_BASE } from '../../core/api.js';

interface Workspace {
  id: number;
  name: string;
  description: string;
  folder_path?: string;
  file_count?: number;
  progress_percentage?: number;
}

interface WorkspacesProps {
  showCreateModal: boolean;
  setShowCreateModal: React.Dispatch<React.SetStateAction<boolean>>;
}

interface WorkspacesProps {
  showCreateModal: boolean;
  setShowCreateModal: React.Dispatch<React.SetStateAction<boolean>>;
  folderToCreate: { path: string; name: string } | null;
  setFolderToCreate: React.Dispatch<React.SetStateAction<{ path: string; name: string } | null>>;
}

const Workspaces: React.FC<WorkspacesProps> = ({ showCreateModal, setShowCreateModal, folderToCreate, setFolderToCreate }) => {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [formData, setFormData] = useState({ name: '', description: '' });
  const [currentWorkspaceId, setCurrentWorkspaceId] = useState<number | null>(null);

  useEffect(() => {
    loadWorkspaces();
  }, []);

  useEffect(() => {
    if (folderToCreate) {
      // Automatically create workspace if folderToCreate is set
      const createAutoWorkspace = async () => {
        try {
          const response = await fetch(`${API_BASE}/workspaces/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: folderToCreate.name, description: `Workspace for folder: ${folderToCreate.path}`, folder_path: folderToCreate.path })
          });

          if (response.ok) {
            console.log('Workspace created automatically for folder:', folderToCreate.name);
            await loadWorkspaces();
            setFolderToCreate(null); // Clear the folderToCreate after creation
          } else {
            console.error('Failed to create automatic workspace, status:', response.status);
          }
        } catch (error) {
          console.error('Failed to create automatic workspace - network error:', error);
        }
      };
      createAutoWorkspace();
    }
  }, [folderToCreate, setFolderToCreate]);

  const loadWorkspaces = async () => {
    try {
      const response = await fetch(`${API_BASE}/workspaces/`);
      const data = await response.json();
      setWorkspaces(data);
    } catch (error) {
      console.error('Failed to load workspaces:', error);
    }
  };

  const handleCreateWorkspace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name.trim()) return;

    // If a folder was selected, include its path
    const workspaceData = folderToCreate
      ? { ...formData, folder_path: folderToCreate.path }
      : formData;

    try {
      const response = await fetch(`${API_BASE}/workspaces/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(workspaceData)
      });

      if (response.ok) {
        setShowCreateModal(false);
        setFormData({ name: '', description: '' });
        await loadWorkspaces();
      } else {
        console.error('Failed to create workspace, status:', response.status);
      }
    } catch (error) {
      console.error('Failed to create workspace - network error:', error);
    }
  };

  const handleSelectWorkspace = (id: number) => {
    setCurrentWorkspaceId(id);
    // Emit event to switch to files tab
    window.dispatchEvent(new CustomEvent('workspace-selected', { detail: { workspaceId: id } }));
  };

  const handleDeleteWorkspace = async (id: number) => {
    if (!confirm('Are you sure you want to delete this workspace?')) return;

    try {
      const response = await fetch(`${API_BASE}/workspaces/${id}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        await loadWorkspaces();
      }
    } catch (error) {
      console.error('Failed to delete workspace:', error);
    }
  };

  return (
    <div id="workspaces-tab" className="tab-content active">
      <div className="workspace-header">
        <h2>Your Workspaces</h2>
        {/* The "Create Workspace" button is now driven by the Electron menu */}
      </div>
      <div id="workspaces-grid" className="workspace-grid">
        {workspaces.map(workspace => (
          <div key={workspace.id} className="workspace-card">
            <div className="workspace-header">
              <h3>
                {workspace.name}
                {workspace.folder_path && (
                  <span className="folder-indicator" title="Linked to folder">📁</span>
                )}
              </h3>
            </div>
            <p>{workspace.description || 'No description'}</p>
            <div className="workspace-stats">
              <span>Files: {workspace.file_count || 0}</span>
              <span>Progress: {workspace.progress_percentage || 0}%</span>
            </div>
            <div className="workspace-actions">
              <button className="btn-secondary" onClick={() => handleSelectWorkspace(workspace.id)}>
                Open
              </button>
              <button className="btn-danger" onClick={() => handleDeleteWorkspace(workspace.id)}>
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      {showCreateModal && (
        <div id="create-workspace-modal" className="modal">
          <div className="modal-content">
            <h3>Create New Workspace</h3>
            <form id="create-workspace-form" onSubmit={handleCreateWorkspace}>
              <input
                type="text"
                id="workspace-name"
                placeholder="Workspace Name"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
              <textarea
                id="workspace-description"
                placeholder="Description (optional)"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              />
              <div className="modal-actions">
                <button type="button" id="cancel-create" className="btn-secondary" onClick={() => {
                  setShowCreateModal(false);
                  setFormData({ name: '', description: '' });
                }}>
                  Cancel
                </button>
                <button type="submit" className="btn-primary">Create</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Workspaces;
