import { File, Tab, FolderTreeNode, MonacoEditor, Workspace } from '../../../src/core/types';
import { getFileTypeFromName } from '../../../src/shared/utils';

export const useFileManagement = (
  workspace: Workspace | null,
  currentWorkspaceId: number | null,
  openFiles: Tab[],
  activeFileId: number | null,
  monacoEditorRef: React.MutableRefObject<MonacoEditor | null>,
  setOpenFiles: (tabs: Tab[] | ((prev: Tab[]) => Tab[])) => void,
  setActiveFileId: (id: number | null) => void
) => {
  const openFile = async (node: FolderTreeNode) => {
    if (!workspace?.folder_path) return;

    const fullPath = `${workspace.folder_path}/${node.path}`;

    try {
      const content = await (window as any).electronAPI.readFileContent(fullPath);

      // Create a temporary file object for the tab
      const tempFile: File = {
        id: Date.now(), // Temporary ID
        name: node.name,
        path: node.path,
        file_type: getFileTypeFromName(node.name),
        size: content.length,
        workspace_id: currentWorkspaceId!,
        content: content
      };

      const existingTab = openFiles.find(f => f.file.path === node.path);
      if (existingTab) {
        setActiveFileId(existingTab.id);
        if (monacoEditorRef.current) {
          monacoEditorRef.current.setValue(content);
        }
        return;
      }

      const tab: Tab = {
        id: tempFile.id,
        name: tempFile.name,
        file: tempFile,
        isActive: false
      };

      setOpenFiles(prev => [...prev, tab]);
      setActiveFileId(tempFile.id);

      if (monacoEditorRef.current) {
        monacoEditorRef.current.setValue(content);
      }
    } catch (error) {
      console.error('Failed to open file:', error);
      alert('Failed to open file');
    }
  };

  const closeTab = (fileId: number) => {
    const remainingTabs = openFiles.filter(f => f.id !== fileId);
    setOpenFiles(remainingTabs);
    if (activeFileId === fileId) {
      if (remainingTabs.length > 0) {
        const newActiveId = remainingTabs[remainingTabs.length - 1].id;
        setActiveFileId(newActiveId);
        if (monacoEditorRef.current) {
          const newActiveTab = remainingTabs.find(t => t.id === newActiveId);
          if (newActiveTab) {
            monacoEditorRef.current.setValue(newActiveTab.file.content || '');
          }
        }
      } else {
        setActiveFileId(null);
      }
    }
  };

  const openSearchResult = async (result: any) => {
    if (!workspace?.folder_path) return;

    const fullPath = `${workspace.folder_path}/${result.path}`;

    try {
      const content = await (window as any).electronAPI.readFileContent(fullPath);

      const tempFile: File = {
        id: Date.now(),
        name: result.name,
        path: result.path,
        file_type: getFileTypeFromName(result.name),
        size: content.length,
        workspace_id: currentWorkspaceId!,
        content: content
      };

      const existingTab = openFiles.find(f => f.file.path === result.path);
      if (existingTab) {
        setActiveFileId(existingTab.id);
        if (monacoEditorRef.current) {
          monacoEditorRef.current.setValue(content);
        }
        return;
      }

      const tab: Tab = {
        id: tempFile.id,
        name: tempFile.name,
        file: tempFile,
        isActive: false
      };

      setOpenFiles(prev => [...prev, tab]);
      setActiveFileId(tempFile.id);

      if (monacoEditorRef.current) {
        monacoEditorRef.current.setValue(content);
      }
    } catch (error) {
      console.error('Failed to open search result:', error);
      alert('Failed to open file');
    }
  };

  return {
    openFile,
    closeTab,
    openSearchResult,
  };
};
