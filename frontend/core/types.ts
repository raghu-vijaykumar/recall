// Shared TypeScript interfaces and types

export interface Workspace {
  id: number;
  name: string;
  description?: string;
  folder_path?: string;
  file_count?: number;
  progress_percentage?: number;
  created_at?: string;
  updated_at?: string;
}

export interface File {
  id: number;
  name: string;
  path: string;
  file_type: string;
  size: number;
  workspace_id: number;
  content?: string;
  created_at?: string;
  updated_at?: string;
}

export interface Quiz {
  id: number;
  question: string;
  options: string[];
  correct_answer: number;
  explanation?: string;
  workspace_id: number;
  created_at?: string;
}

export interface Progress {
  id: number;
  workspace_id: number;
  quiz_id?: number;
  score?: number;
  completed_at?: string;
}

export interface QuizResult {
  quiz_id: number;
  score: number;
  total_questions: number;
  correct_answers: number;
}

export interface WorkspaceStats {
  total_files: number;
  quizzes_taken: number;
  average_score: number;
  total_quizzes: number;
}

// Monaco Editor types
export interface MonacoEditor {
  setValue(value: string): void;
  getValue(): string;
  updateOptions(options: any): void;
  layout(): void;
}

// Tab management types
export interface Tab {
  id: number;
  name: string;
  file: File;
  isActive: boolean;
}

// Menu event types
export interface MenuEvent {
  type: string;
  data?: any;
}

// Component interfaces
export interface Component {
  initialize(): void;
  destroy(): void;
}

export interface TabComponent extends Component {
  getActiveTabId(): number | null;
  openFile(file: File): void;
  closeTab(fileId: number): void;
  switchToTab(fileId: number): void;
}

export interface EditorComponent extends Component {
  getValue(): string;
  setValue(value: string): void;
  focus(): void;
  resize(): void;
}
