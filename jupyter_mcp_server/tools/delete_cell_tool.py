# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""Delete cell tool implementation."""

from typing import Any, Optional
from pathlib import Path
import json
import nbformat
from jupyter_server_client import JupyterServerClient
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.utils import get_notebook_model, clean_notebook_outputs


class DeleteCellTool(BaseTool):
    """Tool to delete specific cells from a notebook."""

    def _get_cell_source(self, cell: Any) -> str:
        """Get the cell source from the cell"""
        cell_source = cell.get("source", "")
        if isinstance(cell_source, list):
            return "".join(cell_source)
        else:
            return str(cell_source)

    async def _delete_cell_ydoc(
        self,
        serverapp: Any,
        notebook_path: str,
        cell_indices: list[int]
    ) -> list:
        """Delete cell using YDoc (collaborative editing mode).
        
        Args:
            serverapp: Jupyter ServerApp instance
            notebook_path: Path to the notebook
            cell_indices: List of indices of cells to delete
            
        Returns:
            NotebookNode
        """
        nb = await get_notebook_model(serverapp, notebook_path)
        if nb:
            if max(cell_indices) >= len(nb):
                raise ValueError(
                    f"Cell index {max(cell_indices)} is out of range. Notebook has {len(nb)} cells."
                )
            
            cells = nb.delete_many_cells(cell_indices)
            return cells
        else:
            # YDoc not available, use file operations
            return await self._delete_cell_file(notebook_path, cell_indices)
    
    async def _delete_cell_file(
        self,
        notebook_path: str,
        cell_indices: list[int]
    ) -> list:
        """Delete cell using file operations (non-collaborative mode).
        
        Args:
            notebook_path: Absolute path to the notebook
            cell_indices: List of indices of cells to delete
            
        Returns:
            List of deleted cells
        """
        # Read notebook file as version 4 for consistency
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        
        clean_notebook_outputs(notebook)
        
        if max(cell_indices) >= len(notebook.cells):
            raise ValueError(
                f"Cell index {max(cell_indices)} is out of range. Notebook has {len(notebook.cells)} cells."
            )
        
        deleted_cells = []
        for cell_index in cell_indices:
            cell = notebook.cells[cell_index]
            result = {
                "index": cell_index,
                "cell_type": cell.cell_type,
                "source": self._get_cell_source(cell),
            }
            deleted_cells.append(result)
        
        # Delete the cell
        for cell_index in sorted(cell_indices, reverse=True):
            notebook.cells.pop(cell_index)
        
        # Write back to file
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        
        return deleted_cells
    
    async def _delete_cell_websocket(
        self,
        notebook_manager: NotebookManager,
        cell_indices: list[int]
    ) -> list:
        """Delete cell using WebSocket connection (MCP_SERVER mode).
        
        Args:
            notebook_manager: Notebook manager instance
            cell_indices: List of indices of cells to delete
            
        Returns:
            List of deleted cell information
        """
        async with notebook_manager.get_current_connection() as notebook:
            if max(cell_indices) >= len(notebook):
                raise ValueError(
                    f"Cell index {max(cell_indices)} is out of range. Notebook has {len(notebook)} cells."
                )

            cells = notebook.delete_many_cells(cell_indices)
            return cells
    
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
        cell_indices: list[int] = None,
        include_source: bool = True,
        session_id: Optional[str] = None,  # ARK-165: Multi-client session support
        **kwargs
    ) -> dict:
        """Delete cells from notebook with session-aware context (ARK-165).

        This tool deletes cells from the notebook associated with the given session_id.
        It automatically retrieves the correct notebook from the SessionStore and supports
        multiple execution modes for different collaboration scenarios.

        Operation Modes:
            1. JUPYTER_SERVER with YDoc (collaborative):
               - Uses YDoc for real-time collaborative editing
               - Changes visible immediately to all connected users
               - Preferred when notebook is open in JupyterLab

            2. JUPYTER_SERVER without YDoc (file-based):
               - Falls back to nbformat file operations
               - Suitable when notebook is not actively edited

            3. MCP_SERVER (WebSocket):
               - Uses WebSocket connection to remote server
               - Accesses YDoc through NbModelClient

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            server_client: HTTP client for MCP_SERVER mode
            contents_manager: Direct API access for JUPYTER_SERVER mode
            notebook_manager: Notebook manager instance
            cell_indices: List of cell indices to delete (0-based)
                - Can delete multiple cells at once
                - Indices are sorted in reverse order for safe deletion
            include_source: Include deleted cell source in response (default: True)
            session_id: Client session ID for multi-client support (ARK-165)
                - Required for multi-client environments
                - Used to lookup notebook context from SessionStore
                - If omitted, falls back to config default
            **kwargs: Additional parameters

        Returns:
            Structured dict with deletion results (if session_id provided):
            ```python
            {
                "result": str,  // Success message with deleted cell info
                "session_id": str | None,  // Client session ID
                "notebook_path": str,  // Path to modified notebook
                "metadata": {
                    "cell_indices": List[int],  // Deleted cell indices
                    "include_source": bool,     // Whether source was included
                    "timestamp": str  // ISO 8601 format
                }
            }
            ```
            Or simple string (backward compatible, if no session_id)

        Raises:
            ValueError: If cell index out of range or invalid mode
            FileNotFoundError: If notebook file doesn't exist
            RuntimeError: If deletion operation fails

        Example:
            Single client (backward compatible):
            ```python
            result = await delete_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_indices=[0, 2],
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )
            ```

            Multi-client (ARK-165):
            ```python
            # Client A deletes cells in their notebook
            result_a = await delete_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_indices=[0],
                session_id="session-A-uuid",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )

            # Client B deletes cells in their notebook (independent)
            result_b = await delete_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_indices=[1, 2],
                session_id="session-B-uuid",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )
            ```

        Notes:
            - ARK-165: Session-based context lookup from SessionStore
            - Supports batch deletion of multiple cells
            - Automatic fallback to file mode if YDoc unavailable
        """
        if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
            # JUPYTER_SERVER mode: Try YDoc first, fall back to file operations
            from jupyter_mcp_server.jupyter_extension.context import get_server_context
            from jupyter_mcp_server.utils import get_notebook_context_from_session_async

            context = get_server_context()
            serverapp = context.serverapp
            # ARK-165: Session-based context lookup with auto-healing (Task 4)
            notebook_path, _ = await get_notebook_context_from_session_async(
                session_id=session_id,
                auto_heal=True,
                kernel_manager=kernel_manager,
                mode=mode  # Use mode parameter (ServerMode already imported at top)
            )

            # Resolve to absolute path
            if serverapp and not Path(notebook_path).is_absolute():
                root_dir = serverapp.root_dir
                notebook_path = str(Path(root_dir) / notebook_path)
            
            if serverapp:
                # Try YDoc approach first
                cells = await self._delete_cell_ydoc(serverapp, notebook_path, cell_indices)
            else:
                # Fall back to file operations
                cells = await self._delete_cell_file(notebook_path, cell_indices)
                
        elif mode == ServerMode.MCP_SERVER and notebook_manager is not None:
            # MCP_SERVER mode: Use WebSocket connection
            cells = await self._delete_cell_websocket(notebook_manager, cell_indices)
        else:
            raise ValueError(f"Invalid mode or missing required clients: mode={mode}")
        
        info_list = []
        for cell_index, cell_info in zip(cell_indices, cells):
            info_list.append(f"Cell {cell_index} ({cell_info['cell_type']}) deleted successfully.")
            if include_source:
                info_list.append(f"deleted cell source:\n{cell_info['source']}")
                info_list.append("\n---\n")

        result_str = "\n".join(info_list)

        # ARK-165: Structured output with session_id (backward compatible)
        if session_id is not None:
            from datetime import datetime
            return json.dumps({
                "result": result_str,
                "session_id": session_id,
                "notebook_path": notebook_path,
                "metadata": {
                    "cell_indices": cell_indices,
                    "include_source": include_source,
                    "timestamp": datetime.now().isoformat()
                }
            }, indent=2)
        else:
            # Backward compatibility: return string only
            return result_str
