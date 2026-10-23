from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TrajectoryStep(BaseModel):
    step_number: int
    thought: str
    action_tool: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None
    observation: Optional[Any] = None

class AgentScratchpad(BaseModel):
    user_query: str
    plan_subgoals: List[str] = Field(default_factory=list)
    steps: List[TrajectoryStep] = Field(default_factory=list)
    gathered_passages: List[Dict[str, Any]] = Field(default_factory=list)
    visited_entities: List[str] = Field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 5

    def add_step(self, thought: str, tool_name: Optional[str] = None, tool_args: Optional[Dict[str, Any]] = None, observation: Optional[Any] = None):
        self.current_iteration += 1
        step = TrajectoryStep(
            step_number=self.current_iteration,
            thought=thought,
            action_tool=tool_name,
            action_args=tool_args,
            observation=observation
        )
        self.steps.append(step)

    def format_history_for_prompt(self) -> str:
        lines = []
        for s in self.steps:
            lines.append(f"Step {s.step_number}:")
            lines.append(f"Thought: {s.thought}")
            if s.action_tool:
                lines.append(f"Action: {s.action_tool}({s.action_args})")
            if s.observation is not None:
                lines.append(f"Observation: {s.observation}")
            lines.append("")
        return "\n".join(lines)
