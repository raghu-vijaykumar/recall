// HTML template loader for components
export class HtmlLoader {
  private static componentCache: Map<string, string> = new Map();

  static async loadComponent(componentName: string): Promise<string> {
    // Check cache first
    if (this.componentCache.has(componentName)) {
      return this.componentCache.get(componentName)!;
    }

    try {
      const response = await fetch(`./components/${componentName}/${componentName}.html`);
      if (!response.ok) {
        throw new Error(`Failed to load component ${componentName}: ${response.status}`);
      }
      const html = await response.text();
      this.componentCache.set(componentName, html);
      return html;
    } catch (error) {
      console.error(`Error loading component ${componentName}:`, error);
      return `<div class="error">Failed to load ${componentName} component</div>`;
    }
  }

  static async loadAllComponents(): Promise<void> {
    const components = ['workspaces', 'file-explorer', 'quiz', 'progress'];
    const loadPromises = components.map(component => this.loadComponent(component));

    try {
      const results = await Promise.all(loadPromises);

      // Insert components into DOM
      components.forEach((component, index) => {
        const container = document.getElementById(`${component}-container`);
        if (container) {
          container.innerHTML = results[index];
        }
      });

      console.log('All components loaded successfully');
    } catch (error) {
      console.error('Error loading components:', error);
    }
  }
}
