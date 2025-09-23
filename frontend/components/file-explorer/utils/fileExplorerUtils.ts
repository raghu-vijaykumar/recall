import { FolderTreeNode } from '../../core/types.js';

export const filterTreeBySearch = (nodes: FolderTreeNode[], query: string): FolderTreeNode[] => {
  if (!query.trim()) return nodes;

  const lowerQuery = query.toLowerCase();

  return nodes.reduce((filtered: FolderTreeNode[], node) => {
    const matchesName = node.name.toLowerCase().includes(lowerQuery);

    if (node.type === 'directory' && node.children) {
      const filteredChildren = filterTreeBySearch(node.children, query);
      if (matchesName || filteredChildren.length > 0) {
        filtered.push({
          ...node,
          children: filteredChildren
        });
      }
    } else if (matchesName) {
      filtered.push(node);
    }

    return filtered;
  }, []);
};
