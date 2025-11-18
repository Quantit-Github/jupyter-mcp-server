# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""Read cell tool implementation."""

from typing import Any, Optional
import json
from jupyter_server_client import JupyterServerClient
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.models import Notebook
from jupyter_mcp_server.config import get_config
from mcp.types import ImageContent


class ReadCellTool(BaseTool):
    """Tool to read a specific cell from a notebook."""
    
    async def execute(
        self,
        mode: ServerMode,
        server_client: Optional[JupyterServerClient] = None,
        kernel_client: Optional[Any] = None,
        contents_manager: Optional[Any] = None,
        kernel_manager: Optional[Any] = None,
        kernel_spec_manager: Optional[Any] = None,
        notebook_manager: Optional[NotebookManager] = None,
        # Tool-specific parameters
        cell_index: int = None,
        include_outputs: bool = True,
        session_id: Optional[str] = None,  # ARK-165: Multi-client session support
        **kwargs
    ) -> dict:
        """Read cell content with session-aware context (ARK-165).

        This tool reads a cell from the notebook associated with the given session_id.
        It automatically retrieves the correct notebook from the SessionStore and supports
        reading both source code and execution outputs.

        Operation Modes:
            1. JUPYTER_SERVER (file-based):
               - Reads notebook directly from file system
               - Uses contents_manager for efficient access
               - No need for YDoc (read-only operation)

            2. MCP_SERVER (WebSocket):
               - Uses WebSocket connection to remote server
               - Accesses YDoc through NbModelClient
               - Real-time access to collaborative documents

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            contents_manager: Direct API access for JUPYTER_SERVER mode
            notebook_manager: Notebook manager instance
            cell_index: Index of the cell to read (0-based)
                - Must be within notebook range
                - Negative indices not supported
            include_outputs: Include execution outputs in response (default: True)
                - Only applies to code cells
                - Includes text, images, and error outputs
                - Markdown cells have no outputs
            session_id: Client session ID for multi-client support (ARK-165)
                - Required for multi-client environments
                - Used to lookup notebook context from SessionStore
                - If omitted, falls back to config default
            **kwargs: Additional parameters

        Returns:
            Structured dict with cell information (if session_id provided):
            ```python
            {
                "cell_info": List[str | ImageContent],  // Cell metadata, source, outputs
                "session_id": str | None,  // Client session ID
                "notebook_path": str,      // Path to notebook
                "metadata": {
                    "cell_index": int,        // Index of read cell
                    "cell_type": str,         // "code" or "markdown"
                    "include_outputs": bool,  // Whether outputs included
                    "timestamp": str          // ISO 8601 format
                }
            }
            ```
            Or simple list (backward compatible, if no session_id)

        Raises:
            ValueError: If cell_index out of range or invalid mode
            FileNotFoundError: If notebook file doesn't exist

        Example:
            Single client (backward compatible):
            ```python
            cell_info = await read_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=0,
                include_outputs=True,
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )
            # Returns list: ["=====Cell 0...", "source code", "outputs..."]
            ```

            Multi-client (ARK-165):
            ```python
            # Client A reads cell from their notebook
            result_a = await read_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=0,
                session_id="session-A-uuid",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )

            # Client B reads cell from their notebook (independent)
            result_b = await read_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=0,
                session_id="session-B-uuid",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )

            # Verify isolation
            assert result_a["session_id"] == "session-A-uuid"
            assert result_b["session_id"] == "session-B-uuid"
            assert result_a["notebook_path"] != result_b["notebook_path"]
            assert result_a["metadata"]["cell_index"] == 0
            ```

        Notes:
            - ARK-165: Session-based context lookup from SessionStore
            - Read-only operation (no need for YDoc in JUPYTER_SERVER mode)
            - Returns formatted cell information with metadata
            - Supports both text and image outputs
        """
        if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
            # Local mode: read notebook directly from file system
            # ARK-165: Session-based context lookup with auto-healing (Task 4)
            from jupyter_mcp_server.utils import get_notebook_context_from_session_async
            notebook_path, _ = await get_notebook_context_from_session_async(
                session_id=session_id,
                auto_heal=True,
                kernel_manager=kernel_manager,
                mode=mode  # Use mode parameter (ServerMode already imported at top)
            )
            
            model = await contents_manager.get(notebook_path, content=True, type='notebook')
            if 'content' not in model:
                raise ValueError(f"Could not read notebook content from {notebook_path}")
            notebook = Notebook(**model['content'])
        elif mode == ServerMode.MCP_SERVER and notebook_manager is not None:
            # Remote mode: use WebSocket connection to Y.js document
            async with notebook_manager.get_current_connection() as notebook_content:
                notebook = Notebook(**notebook_content.as_dict())
        else:
            raise ValueError(f"Invalid mode or missing required clients: mode={mode}")
        
        if cell_index >= len(notebook):
            error_msg = f"Cell index {cell_index} is out of range. Notebook has {len(notebook)} cells."
            # ARK-165: Error case with structured output (backward compatible)
            if session_id is not None:
                from datetime import datetime
                error_with_session = json.dumps({
                    "error": error_msg,
                    "session_id": session_id,
                    "notebook_path": notebook_path,
                    "metadata": {
                        "cell_index": cell_index,
                        "timestamp": datetime.now().isoformat()
                    }
                }, indent=2)
                return [f"❌ Error:\n{error_with_session}"]
            else:
                # Backward compatibility: return list with error message
                return [error_msg]

        cell = notebook[cell_index]
        info_list = []
        # add cell metadata
        info_list.append(f"=====Cell {cell_index} | type: {cell.cell_type} | execution count: {cell.execution_count if cell.execution_count else 'N/A'}=====")
        # add cell source
        info_list.append(cell.get_source('readable'))
        # add cell outputs for code cells
        if cell.cell_type == "code" and include_outputs:
            info_list.extend(cell.get_outputs('readable'))

        # ARK-165: Structured output with session_id (backward compatible)
        if session_id is not None:
            from datetime import datetime
            # Prepend session metadata as JSON string
            session_metadata = json.dumps({
                "session_id": session_id,
                "notebook_path": notebook_path,
                "metadata": {
                    "cell_index": cell_index,
                    "cell_type": cell.cell_type,
                    "include_outputs": include_outputs,
                    "timestamp": datetime.now().isoformat()
                }
            }, indent=2)
            return [f"📊 Session Info:\n{session_metadata}"] + info_list
        else:
            # Backward compatibility: return list only
            return info_list
