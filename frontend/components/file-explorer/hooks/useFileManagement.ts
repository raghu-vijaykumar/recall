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
  const isBinaryFile = (content: string): boolean => {
    // Check for null bytes or other binary indicators
    return content.includes('\0') || /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/.test(content);
  };

  const openFile = async (node: FolderTreeNode) => {
    if (!workspace?.folder_path || !currentWorkspaceId) return;

    const fullPath = `${workspace.folder_path}/${node.path}`;
    const isNonTextFile = /\.(jpg|jpeg|png|gif|bmp|webp|svg|pdf|mp4|avi|mov|mkv|webm|flv|wmv|mpg|mpeg|3gp|m4v|mp3|wav|ogg|flac|aac|m4a|wma|aiff|au)$/i.test(node.name);

    try {
      // For non-text files (images, PDFs, videos, audio), we don't need to read content as text
      if (isNonTextFile) {
        // Get file stats to get size
        const stats = await (window as any).electronAPI.getFileStats(fullPath);

        const tempFile: File = {
          id: Date.now(), // Temporary ID
          name: node.name,
          path: node.path, // Use relative path for consistency
          file_type: getFileTypeFromName(node.name),
          size: stats.size,
          workspace_id: currentWorkspaceId,
          content: '' // Not needed for viewer
        };

        const existingTab = openFiles.find(f => f.file.path === node.path);
        if (existingTab) {
          setActiveFileId(existingTab.id);
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
        return;
      }

      const content = await (window as any).electronAPI.readFileContent(fullPath);

      // Check if file is binary
      if (isBinaryFile(content)) {
        // Create a special file object for binary files
        const tempFile: File = {
          id: Date.now(), // Temporary ID
          name: node.name,
          path: node.path,
          file_type: 'binary',
          size: content.length,
          workspace_id: currentWorkspaceId,
          content: `This is a binary file and cannot be opened in the editor.\n\nFile: ${node.name}\nPath: ${node.path}\nSize: ${content.length} bytes`
        };

        const existingTab = openFiles.find(f => f.file.path === node.path);
        if (existingTab) {
          setActiveFileId(existingTab.id);
          if (monacoEditorRef.current) {
            monacoEditorRef.current.setValue(tempFile.content);
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
          monacoEditorRef.current.setValue(tempFile.content);
        }
        return;
      }

      // Create a temporary file object for the tab
      const tempFile: File = {
        id: Date.now(), // Temporary ID
        name: node.name,
        path: node.path,
        file_type: getFileTypeFromName(node.name),
        size: content.length,
        workspace_id: currentWorkspaceId,
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
    if (!workspace?.folder_path || !currentWorkspaceId || !result.path) return;

    const fullPath = `${workspace.folder_path}/${result.path}`;
    const isNonTextFile = /\.(jpg|jpeg|png|gif|bmp|webp|svg|pdf|mp4|avi|mov|mkv|webm|flv|wmv|mpg|mpeg|3gp|m4v|mp3|wav|ogg|flac|aac|m4a|wma|aiff|au)$/i.test(result.name);

    try {
      // For non-text files (images, PDFs, videos, audio), we don't need to read content as text
      if (isNonTextFile) {
        // Get file stats to get size
        const stats = await (window as any).electronAPI.getFileStats(fullPath);

        const tempFile: File = {
          id: Date.now(),
          name: result.name,
          path: result.path, // Use relative path for consistency
          file_type: getFileTypeFromName(result.name),
          size: stats.size,
          workspace_id: currentWorkspaceId,
          content: '' // Not needed for viewer
        };

        const existingTab = openFiles.find(f => f.file.path === result.path);
        if (existingTab) {
          setActiveFileId(existingTab.id);
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
        return;
      }

      const content = await (window as any).electronAPI.readFileContent(fullPath);

      // Check if file is binary
      if (isBinaryFile(content)) {
        // Create a special file object for binary files
        const tempFile: File = {
          id: Date.now(),
          name: result.name,
          path: result.path,
          file_type: 'binary',
          size: content.length,
          workspace_id: currentWorkspaceId,
          content: `This is a binary file and cannot be opened in the editor.\n\nFile: ${result.name}\nPath: ${result.path}\nSize: ${content.length} bytes`
        };

        const existingTab = openFiles.find(f => f.file.path === result.path);
        if (existingTab) {
          setActiveFileId(existingTab.id);
          if (monacoEditorRef.current) {
            monacoEditorRef.current.setValue(tempFile.content);
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
          monacoEditorRef.current.setValue(tempFile.content);
        }
        return;
      }

      const tempFile: File = {
        id: Date.now(),
        name: result.name,
        path: result.path,
        file_type: getFileTypeFromName(result.name),
        size: content.length,
        workspace_id: currentWorkspaceId,
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
