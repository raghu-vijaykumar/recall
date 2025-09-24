import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { ApiService } from '../../src/core/api';

interface Node {
  id: string;
  name: string;
  description?: string;
  x?: number;
  y?: number;
}

interface Edge {
  id: string;
  source: string;
  target: string;
  type: string;
  strength?: number;
}

interface KnowledgeGraphData {
  nodes: Node[];
  edges: Edge[];
}

interface KnowledgeGraphProps {
  workspaceId: number;
  onNodeClick?: (node: Node) => void;
  onEdgeClick?: (edge: Edge) => void;
  width?: number;
  height?: number;
}

export const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({
  workspaceId,
  onNodeClick,
  onEdgeClick,
  width = 800,
  height = 600,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<KnowledgeGraphData>({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  useEffect(() => {
    loadGraphData();
  }, [workspaceId]);

  useEffect(() => {
    if (data.nodes.length > 0 && svgRef.current) {
      renderGraph();
    }
  }, [data, selectedNode]);

  const loadGraphData = async () => {
    try {
      setLoading(true);
      const response = await ApiService.get(`/knowledge-graph/workspaces/${workspaceId}/graph`);
      setData(response);
      setError(null);
    } catch (err) {
      setError('Failed to load knowledge graph');
      console.error('Error loading graph:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderGraph = () => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous render

    const simulation = d3.forceSimulation(data.nodes as d3.SimulationNodeDatum[])
      .force('link', d3.forceLink(data.edges)
        .id((d: any) => d.id)
        .distance((d: any) => 100 - (d.strength || 0.5) * 50)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Create arrow markers for directed edges
    const defs = svg.append('defs');

    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('xoverflow', 'visible')
      .append('svg:path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#999');

    // Create links
    const link = svg.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(data.edges)
      .enter().append('line')
      .attr('stroke', (d) => getEdgeColor(d.type))
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d) => Math.sqrt(d.strength || 1) * 2)
      .attr('marker-end', 'url(#arrowhead)')
      .on('click', (event, d) => {
        event.stopPropagation();
        onEdgeClick?.(d);
      });

    // Create link labels
    const linkLabels = svg.append('g')
      .attr('class', 'link-labels')
      .selectAll('text')
      .data(data.edges)
      .enter().append('text')
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#666')
      .text((d) => d.type.replace('_', ' '));

    // Create nodes
    const node = svg.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(data.nodes)
      .enter().append('circle')
      .attr('r', (d) => getNodeRadius(d))
      .attr('fill', (d) => d.id === selectedNode?.id ? '#ff6b6b' : getNodeColor(d))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .call(d3.drag<SVGCircleElement, Node>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          (d as any).fx = (d as any).x;
          (d as any).fy = (d as any).y;
        })
        .on('drag', (event, d) => {
          (d as any).fx = event.x;
          (d as any).fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          (d as any).fx = null;
          (d as any).fy = null;
        })
      )
      .on('click', (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
        onNodeClick?.(d);
      })
      .on('dblclick', (event, d) => {
        // Expand node connections
        expandNode(d);
      });

    // Create node labels
    const nodeLabels = svg.append('g')
      .attr('class', 'node-labels')
      .selectAll('text')
      .data(data.nodes)
      .enter().append('text')
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .attr('dy', -25)
      .text((d) => truncateText(d.name, 15));

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      linkLabels
        .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
        .attr('y', (d: any) => (d.source.y + d.target.y) / 2);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      nodeLabels
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        svg.select('g').attr('transform', event.transform);
      });

    svg.call(zoom);
  };

  const expandNode = async (node: Node) => {
    // Load subgraph for the selected node
    try {
      const response = await ApiService.get(`/knowledge-graph/workspaces/${workspaceId}/graph?concept_id=${node.id}&depth=3`);
      setData(response);
    } catch (err) {
      console.error('Error expanding node:', err);
    }
  };

  const getNodeColor = (node: Node): string => {
    // Color nodes based on some property (could be enhanced with more logic)
    const colors = ['#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'];
    const hash = node.name.split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0);
      return a & a;
    }, 0);
    return colors[Math.abs(hash) % colors.length];
  };

  const getNodeRadius = (node: Node): number => {
    // Size nodes based on connectivity
    const connections = data.edges.filter(e => e.source === node.id || e.target === node.id).length;
    return Math.max(8, Math.min(20, 8 + connections * 2));
  };

  const getEdgeColor = (type: string): string => {
    const colorMap: { [key: string]: string } = {
      'relates_to': '#666',
      'dives_deep_to': '#e74c3c',
      'has_type': '#27ae60',
      'uses': '#f39c12',
    };
    return colorMap[type] || '#999';
  };

  const truncateText = (text: string, maxLength: number): string => {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        <span className="ml-2">Loading knowledge graph...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 text-red-500">
        <span>{error}</span>
        <button
          onClick={loadGraphData}
          className="ml-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="knowledge-graph-container">
      <div className="mb-4 flex justify-between items-center">
        <h3 className="text-lg font-semibold">Knowledge Graph</h3>
        <div className="flex gap-2">
          <button
            onClick={loadGraphData}
            className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600"
          >
            Refresh
          </button>
          <button
            onClick={() => setData({ nodes: [], edges: [] })}
            className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="border rounded-lg overflow-hidden">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="bg-gray-50"
        />
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p><strong>Instructions:</strong></p>
        <ul className="list-disc list-inside mt-1">
          <li>Click nodes to select and view details</li>
          <li>Double-click nodes to expand their connections</li>
          <li>Drag nodes to reposition them</li>
          <li>Scroll to zoom in/out</li>
        </ul>
      </div>
    </div>
  );
};
