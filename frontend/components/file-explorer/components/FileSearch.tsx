import React from 'react';
import { API_BASE } from '../../../src/core/api';
import { FolderTreeNode, Workspace } from '../../../src/core/types';
import { getFileIcon } from '../../../src/shared/utils';

interface FileSearchProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  searchResults: any[];
  setSearchResults: (results: any[]) => void;
  isSearching: boolean;
  setIsSearching: (searching: boolean) => void;
  currentWorkspaceId: number | null;
  workspace: Workspace | null;
  openSearchResult: (result: any) => void;
}

const FileSearch: React.FC<FileSearchProps> = ({
  searchQuery,
  setSearchQuery,
  searchResults,
  setSearchResults,
  isSearching,
  setIsSearching,
  currentWorkspaceId,
  workspace,
  openSearchResult,
}) => {
  const performContentSearch = async () => {
    if (!searchQuery.trim() || !workspace?.folder_path) return;

    setIsSearching(true);
    try {
      const response = await fetch(`${API_BASE}/search/content`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workspace_id: currentWorkspaceId,
          query: searchQuery,
          folder_path: workspace.folder_path
        })
      });

      if (response.ok) {
        const results = await response.json();
        setSearchResults(results);
      } else {
        console.error('Search failed:', response.statusText);
        setSearchResults([]);
      }
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const renderSearchResults = () => {
    // Group results by directory path
    const groupedResults = searchResults.reduce((groups: any, result: any) => {
      const pathParts = result.path.split('/');
      const dirPath = pathParts.slice(0, -1).join('/') || '/';

      if (!groups[dirPath]) {
        groups[dirPath] = [];
      }
      groups[dirPath].push(result);
      return groups;
    }, {});

    return Object.entries(groupedResults).map(([dirPath, files]: [string, any]) => (
      <div key={dirPath} className="search-group">
        <div className="search-group-header">
          <span className="search-group-path">{dirPath === '/' ? 'Root' : dirPath}</span>
        </div>
        <div className="search-group-files">
          {files.map((result: any) => (
            <div
              key={result.path}
              className="search-result-item"
              onClick={() => openSearchResult(result)}
            >
              <div className="search-result-header">
                <span className="search-result-icon">{getFileIcon(result.name)}</span>
                <span className="search-result-name">{result.name}</span>
              </div>
              {result.matches && result.matches.length > 0 && (
                <div className="search-result-matches">
                  {result.matches.slice(0, 3).map((match: any, index: number) => (
                    <div key={index} className="search-match">
                      <span className="search-match-line">Line {match.line}:</span>
                      <span className="search-match-text">{match.text}</span>
                    </div>
                  ))}
                  {result.matches.length > 3 && (
                    <div className="search-match-more">
                      ... and {result.matches.length - 3} more matches
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    ));
  };

  return (
    <div className="search-results">
      <div className="search-input-container">
        <input
          type="text"
          placeholder="Search file contents..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="search-input"
          onKeyPress={(e) => e.key === 'Enter' && performContentSearch()}
        />
        <button
          className="search-btn"
          onClick={performContentSearch}
          disabled={isSearching || !searchQuery.trim()}
        >
          {isSearching ? '🔄' : '🔍'}
        </button>
        {searchQuery && (
          <button
            className="clear-search-btn"
            onClick={() => {
              setSearchQuery('');
              setSearchResults([]);
            }}
            title="Clear search"
          >
            ×
          </button>
        )}
      </div>
      <div className="search-results-list">
        {searchResults.length > 0 ? (
          renderSearchResults()
        ) : searchQuery && !isSearching ? (
          <div className="empty-search">
            <p>No files found containing "{searchQuery}"</p>
          </div>
        ) : isSearching ? (
          <div className="searching">
            <p>Searching...</p>
          </div>
        ) : (
          <div className="empty-search">
            <p>Enter a search term to find files by content</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileSearch;
