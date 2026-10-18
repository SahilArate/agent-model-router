import {
  Background,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import { usd } from "../format";
import type { CallTrace } from "../types";

interface AgentNodeData extends Record<string, unknown> {
  trace: CallTrace;
  selected: boolean;
}

type AgentNode = Node<AgentNodeData, "agent">;

function finalModel(trace: CallTrace): string {
  return trace.attempts.at(-1)?.model_id ?? "none";
}

function AgentCard({ data }: NodeProps<AgentNode>) {
  const { trace, selected } = data;
  const tone = trace.final_action === "accept" ? "ok" : "bad";
  return (
    <div className={`agent-node ${tone} ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <div className="agent-name">{trace.agent}</div>
      <div className="agent-model">{finalModel(trace)}</div>
      <div className="agent-meta">
        {usd(trace.cost_usd)}
        {trace.escalations > 0 && <span className="badge warn">↑ {trace.escalations}</span>}
        {trace.final_action !== "accept" && <span className="badge bad">abstained</span>}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { agent: AgentCard };

const COLUMN_GAP = 250;
const ROW_GAP = 120;

/** Planner on the left, workers stacked in the middle, synthesizer on the right. */
function layout(traces: CallTrace[], selectedIndex: number) {
  const planner = traces.findIndex((t) => t.agent === "planner");
  const synthesizer = traces.findIndex((t) => t.agent === "synthesizer");
  const workers = traces.flatMap((t, i) => (t.agent === "worker" ? [i] : []));

  const nodes: AgentNode[] = traces.map((trace, index) => {
    const workerRow = workers.indexOf(index);
    const column = trace.agent === "planner" ? 0 : trace.agent === "synthesizer" ? 2 : 1;
    const centre = ((Math.max(workers.length, 1) - 1) * ROW_GAP) / 2;
    const y = trace.agent === "worker" ? workerRow * ROW_GAP : centre;
    return {
      id: String(index),
      type: "agent",
      position: { x: column * COLUMN_GAP, y },
      data: { trace, selected: index === selectedIndex },
    };
  });

  const edges: Edge[] = [];
  const link = (from: number, to: number) =>
    edges.push({ id: `${from}-${to}`, source: String(from), target: String(to), animated: true });
  workers.forEach((w) => {
    if (planner >= 0) link(planner, w);
    if (synthesizer >= 0) link(w, synthesizer);
  });
  if (workers.length === 0 && planner >= 0 && synthesizer >= 0) link(planner, synthesizer);

  return { nodes, edges };
}

interface Props {
  traces: CallTrace[];
  selected: number;
  onSelect: (index: number) => void;
}

export default function AgentGraph({ traces, selected, onSelect }: Props) {
  const { nodes, edges } = useMemo(() => layout(traces, selected), [traces, selected]);
  return (
    <div className="graph" role="img" aria-label="Agent workflow graph">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.25 }}
        nodesDraggable={false}
        nodesConnectable={false}
        zoomOnScroll={false}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, node) => onSelect(Number(node.id))}
      >
        <Background gap={20} />
      </ReactFlow>
    </div>
  );
}
