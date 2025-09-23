// Shared utility functions

export function getFileIcon(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase();

  const iconMap: { [key: string]: string } = {
    'py': '🐍',
    'js': '🟨',
    'ts': '🔷',
    'html': '🌐',
    'css': '🎨',
    'json': '📋',
    'md': '📝',
    'markdown': '📝',
    'txt': '📄',
    'pdf': '📕',
    'jpg': '🖼️',
    'jpeg': '🖼️',
    'png': '🖼️',
    'gif': '🖼️',
    'xml': '📄',
  };

  return iconMap[ext || ''] || '📄';
}

export function getFileTypeFromName(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase();
  const typeMap: { [key: string]: string } = {
    'py': 'code',
    'js': 'code',
    'ts': 'code',
    'html': 'code',
    'css': 'code',
    'json': 'code',
    'xml': 'code',
    'md': 'markdown',
    'markdown': 'markdown',
    'txt': 'text',
    'pdf': 'pdf',
    'jpg': 'image',
    'jpeg': 'image',
    'png': 'image',
    'gif': 'image'
  };
  return typeMap[ext || ''] || 'text';
}

export function getDefaultContentForFile(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase();

  const templates: { [key: string]: string } = {
    'py': '# Python file\n\ndef main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()\n',
    'js': '// JavaScript file\n\nconsole.log("Hello, World!");\n',
    'ts': '// TypeScript file\n\nconsole.log("Hello, World!");\n',
    'html': '<!DOCTYPE html>\n<html>\n<head>\n    <title>New Page</title>\n</head>\n<body>\n    <h1>Hello, World!</h1>\n</body>\n</html>\n',
    'css': '/* CSS file */\n\nbody {\n    font-family: Arial, sans-serif;\n    margin: 0;\n    padding: 20px;\n}\n',
    'json': '{\n    "name": "example",\n    "version": "1.0.0"\n}\n',
    'md': '# Markdown File\n\nThis is a new markdown file.\n\n## Features\n\n- Easy to write\n- Easy to read\n- Easy to edit\n',
    'txt': 'This is a text file.\n\nYou can write anything here.\n'
  };

  return templates[ext || ''] || '// New file\n\nStart writing your content here...\n';
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

export function showModal(modalId: string): void {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.add('show');
  }
}

export function hideModal(modalId: string): void {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.remove('show');
  }
}

export function showTab(tabName: string): void {
  // Remove active class from all tab contents
  const tabContents = document.querySelectorAll('.tab-content');
  tabContents.forEach(content => content.classList.remove('active'));

  // Add active class to the target tab content
  const targetContent = document.getElementById(`${tabName}-tab`);
  if (targetContent) {
    targetContent.classList.add('active');
  }
}

export function createElementFromHTML(htmlString: string): HTMLElement {
  const div = document.createElement('div');
  div.innerHTML = htmlString.trim();
  return div.firstChild as HTMLElement;
}

export function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}
