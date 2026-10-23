import asyncio
import inspect
from typing import Callable, Dict, Any, List, Optional
from pydantic import BaseModel, Field

class AgentTool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Optional[Any] = Field(default=None, exclude=True)

    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Converts to OpenAI/Groq function calling tool specification.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    async def execute(self, **kwargs) -> Any:
        if not self.func:
            raise ValueError(f"Tool '{self.name}' has no executable function assigned.")
        
        # Filter kwargs to only those accepted by self.func, unless self.func accepts **kwargs
        try:
            sig = inspect.signature(self.func)
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            filtered_kwargs = kwargs if has_var_keyword else {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception:
            filtered_kwargs = kwargs

        if inspect.iscoroutinefunction(self.func):
            return await self.func(**filtered_kwargs)
        else:
            return await asyncio.to_thread(self.func, **filtered_kwargs)

class ToolRegistry:
    """
    Central registry managing available tools for the Agentic RAG core.
    Provides schema export for LLM function calling and safe asynchronous execution.
    """

    def __init__(self):
        self._tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[AgentTool]:
        return list(self._tools.values())

    def get_openai_tool_specs(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

    async def execute_tool(self, name: str, arguments: Dict[str, Any], timeout_sec: float = 3.0) -> Dict[str, Any]:
        tool = self.get_tool(name)
        if not tool:
            return {
                "status": "error",
                "tool": name,
                "error": f"Tool '{name}' is not registered in the agent registry."
            }

        try:
            result = await asyncio.wait_for(tool.execute(**arguments), timeout=timeout_sec)
            return {
                "status": "success",
                "tool": name,
                "output": result
            }
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "tool": name,
                "error": f"Tool '{name}' timed out after {timeout_sec}s."
            }
        except Exception as e:
            return {
                "status": "error",
                "tool": name,
                "error": str(e)
            }
