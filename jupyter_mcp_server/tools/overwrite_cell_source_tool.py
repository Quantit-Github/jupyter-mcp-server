# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""Overwrite cell source tool implementation."""

import difflib
import json
import nbformat
from pathlib import Path
from typing import Any, Optional
from jupyter_server_client import JupyterServerClient
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.utils import get_notebook_model, clean_notebook_outputs


class OverwriteCellSourceTool(BaseTool):
    """Tool to overwrite the source of an existing cell."""
    
    def _generate_diff(self, old_source: str, new_source: str) -> str:
        """Generate unified diff between old and new source."""
        old_lines = old_source.splitlines(keepends=False)
        new_lines = new_source.splitlines(keepends=False)
        
        diff_lines = list[str](difflib.unified_diff(
            old_lines, 
            new_lines, 
            lineterm='',
            n=3  # Number of context lines
        ))
        
        if len(diff_lines) > 3:
            return '\n'.join(diff_lines)
        return "no changes detected"
    
    async def _overwrite_cell_ydoc(
        self,
        serverapp: Any,
        notebook_path: str,
        cell_index: int,
        cell_source: str
    ) -> str:
        """Overwrite cell source using YDoc (collaborative editing mode).
        
        Args:
            serverapp: Jupyter ServerApp instance
            notebook_path: Path to the notebook
            cell_index: Index of the cell to overwrite
            cell_source: New cell source content
            
        Returns:
            Diff showing changes made
            
        Raises:
            RuntimeError: When file_id_manager is not available
            ValueError: When cell_index is out of range
        """
        # Get notebook model
        nb = await get_notebook_model(serverapp, notebook_path)

        if nb:
            # Notebook is open in collaborative mode, use YDoc
            if cell_index >= len(nb):
                raise ValueError(
                    f"Cell index {cell_index} is out of range. Notebook has {len(nb)} cells."
                )
            
            old_source = nb.get_cell_source(cell_index)
            if isinstance(old_source, list):
                old_source = "".join(old_source)
            else:
                old_source = str(old_source)
            nb.set_cell_source(cell_index, cell_source)
            
            return self._generate_diff(old_source, cell_source)
        else:
            # YDoc not available, use file operations
            return await self._overwrite_cell_file(notebook_path, cell_index, cell_source)
    
    async def _overwrite_cell_file(
        self,
        notebook_path: str,
        cell_index: int,
        cell_source: str
    ) -> str:
        """Overwrite cell using file operations (non-collaborative mode).
        
        Args:
            notebook_path: Path to the notebook file
            cell_index: Index of the cell to overwrite
            cell_source: New cell source content
            
        Returns:
            Diff showing changes made
            
        Raises:
            ValueError: When cell_index is out of range
        """
        # Read notebook file as version 4 for consistency
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        clean_notebook_outputs(notebook)
        
        if cell_index >= len(notebook.cells):
            raise ValueError(
                f"Cell index {cell_index} is out of range. Notebook has {len(notebook.cells)} cells."
            )
        
        # Get original cell content
        old_source = notebook.cells[cell_index].source
        
        # Set new cell source
        notebook.cells[cell_index].source = cell_source
        
        # Write back to file
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        
        return self._generate_diff(old_source, cell_source)
    
    async def _overwrite_cell_websocket(
        self,
        notebook_manager: NotebookManager,
        cell_index: int,
        cell_source: str
    ) -> str:
        """Overwrite cell using WebSocket connection (MCP_SERVER mode).
        
        Args:
            notebook_manager: Notebook manager instance
            cell_index: Index of the cell to overwrite
            cell_source: New cell source content
            
        Returns:
            Diff showing changes made
            
        Raises:
            ValueError: When cell_index is out of range
        """
        async with notebook_manager.get_current_connection() as notebook:
            if cell_index >= len(notebook):
                raise ValueError(f"Cell index {cell_index} out of range")
            
            # Get original cell content
            old_source = notebook.get_cell_source(cell_index)
            if isinstance(old_source, list):
                old_source = "".join(old_source)
            else:
                old_source = str(old_source)
            notebook.set_cell_source(cell_index, cell_source)
            return self._generate_diff(old_source, cell_source)
    
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
        cell_source: str = None,
        session_id: Optional[str] = None,  # ARK-165: Multi-client session support
        **kwargs
    ) -> dict:
        """Overwrite cell source with session-aware context (ARK-165).

        This tool overwrites the source code of a cell in the notebook associated with
        the given session_id. It automatically retrieves the correct notebook from the
        SessionStore and supports multiple execution modes with diff generation.

        Operation Modes:
            1. JUPYTER_SERVER with YDoc (collaborative):
               - Uses YDoc for real-time collaborative editing
               - Changes visible immediately to all connected users
               - Protected by thread locks and YDoc transactions (atomic)

            2. JUPYTER_SERVER without YDoc (file-based):
               - Falls back to nbformat file operations
               - Suitable when notebook is not actively edited

            3. MCP_SERVER (WebSocket):
               - Uses WebSocket connection to remote server
               - Remote server handles synchronization

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            server_client: HTTP client for MCP_SERVER mode
            contents_manager: Direct API access for JUPYTER_SERVER mode
            notebook_manager: Notebook manager instance
            cell_index: Index of the cell to overwrite (0-based)
                - Must be within notebook range
                - Both code and markdown cells supported
            cell_source: New cell source content
                - Replaces entire cell source
                - Preserves cell type and metadata
            session_id: Client session ID for multi-client support (ARK-165)
                - Required for multi-client environments
                - Used to lookup notebook context from SessionStore
                - If omitted, falls back to config default
            **kwargs: Additional parameters

        Returns:
            Structured dict with overwrite results (if session_id provided):
            ```python
            {
                "result": str,  // Success message with unified diff
                "session_id": str | None,  // Client session ID
                "notebook_path": str,  // Path to modified notebook
                "metadata": {
                    "cell_index": int,     // Index of overwritten cell
                    "diff": str,           // Unified diff output
                    "timestamp": str       // ISO 8601 format
                }
            }
            ```
            Or simple string (backward compatible, if no session_id)

        Raises:
            ValueError: If mode invalid or cell_index out of range
            FileNotFoundError: If notebook file doesn't exist
            RuntimeError: If overwrite operation fails

        Example:
            Single client (backward compatible):
            ```python
            result = await overwrite_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=0,
                cell_source="print('Updated code')",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )
            ```

            Multi-client (ARK-165):
            ```python
            # Client A overwrites cell in their notebook
            result_a = await overwrite_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=0,
                cell_source="# Updated markdown",
                session_id="session-A-uuid",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )

            # Client B overwrites cell in their notebook (independent)
            result_b = await overwrite_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=0,
                cell_source="result = 42",
                session_id="session-B-uuid",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )

            # Verify changes are isolated
            assert result_a["session_id"] == "session-A-uuid"
            assert result_b["session_id"] == "session-B-uuid"
            assert "Updated markdown" in result_a["metadata"]["diff"]
            assert "result = 42" in result_b["metadata"]["diff"]
            ```

        Notes:
            - ARK-165: Session-based context lookup from SessionStore
            - Generates unified diff showing exact changes made
            - Thread-safe operations with YDoc transactions
            - Automatic fallback to file mode if YDoc unavailable
            - Preserves cell metadata and outputs
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
                # Try YDoc approach first (with thread safety and transactions)
                diff = await self._overwrite_cell_ydoc(serverapp, notebook_path, cell_index, cell_source)
            else:
                # Fall back to file operations
                diff = await self._overwrite_cell_file(notebook_path, cell_index, cell_source)
                
        elif mode == ServerMode.MCP_SERVER and notebook_manager is not None:
            # MCP_SERVER mode: Use WebSocket connection with remote transaction management
            diff = await self._overwrite_cell_websocket(notebook_manager, cell_index, cell_source)
        else:
            raise ValueError(f"Invalid mode or missing required clients: mode={mode}")
        
        if not diff.strip() or diff == "no changes detected":
            result_str = f"Cell {cell_index} overwritten successfully - no changes detected"
        else:
            result_str = f"Cell {cell_index} overwritten successfully!\n\n```diff\n{diff}\n```"

        # ARK-165: Structured output with session_id (backward compatible)
        if session_id is not None:
            from datetime import datetime
            return json.dumps({
                "result": result_str,
                "session_id": session_id,
                "notebook_path": notebook_path,
                "metadata": {
                    "cell_index": cell_index,
                    "diff": diff,
                    "timestamp": datetime.now().isoformat()
                }
            }, indent=2)
        else:
            # Backward compatibility: return string only
            return result_str
