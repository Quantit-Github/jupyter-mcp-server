# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""Insert cell tool implementation."""

from typing import Any, Optional, Literal
from pathlib import Path
import json
import nbformat
from jupyter_server_client import JupyterServerClient
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.utils import get_notebook_model, clean_notebook_outputs
from jupyter_mcp_server.models import Notebook



class InsertCellTool(BaseTool):
    """Tool to insert a cell at a specified position."""
    
    def _validate_cell_insertion_params(
        self,
        cell_index: int,
        total_cells: int,
        cell_type: str
    ) -> int:
        """Validate and normalize cell insertion parameters.
        
        Args:
            cell_index: Target index for insertion (-1 for append)
            total_cells: Total number of cells in the notebook
            cell_type: Type of cell to insert
            
        Returns:
            Normalized actual_index for insertion
            
        Raises:
            IndexError: When cell_index is out of valid range
            ValueError: When cell_type is invalid
        """
        if cell_index < -1 or cell_index > total_cells:
            raise IndexError(
                f"Index {cell_index} is outside valid range [-1, {total_cells}]. "
                f"Use -1 to append at end."
            )
        
        # Normalize -1 to append position
        actual_index = cell_index if cell_index != -1 else total_cells
        return actual_index
    
    async def _insert_cell_ydoc(
        self,
        serverapp: Any,
        notebook_path: str,
        cell_index: int,
        cell_type: Literal["code", "markdown"],
        cell_source: str
    ) -> tuple[Notebook, int, int]:
        """Insert cell using YDoc (collaborative editing mode).
        
        Args:
            serverapp: Jupyter ServerApp instance
            notebook_path: Path to the notebook
            cell_index: Index to insert at (-1 for append)
            cell_type: Type of cell to insert ("code", "markdown")
            cell_source: Source content for the cell
            
        Returns:
            Tuple of (notebook, actual_index, total_cells_after_insertion)
            
        Raises:
            IndexError: When cell_index is out of range
        """
        nb = await get_notebook_model(serverapp, notebook_path)
        
        if nb:
            # Notebook is open in collaborative mode, use YDoc
            total_cells = len(nb)
            
            # Validate insertion parameters
            actual_index = self._validate_cell_insertion_params(
                cell_index, total_cells, cell_type
            )
            
            nb.insert_cell(actual_index, cell_source, cell_type)
            
            return Notebook(**nb.as_dict()), actual_index, len(nb)
        else:
            # YDoc not available, use file operations
            return await self._insert_cell_file(notebook_path, cell_index, cell_type, cell_source)
    
    async def _insert_cell_file(
        self,
        notebook_path: str,
        cell_index: int,
        cell_type: Literal["code", "markdown"],
        cell_source: str
    ) -> tuple[Notebook, int, int]:
        """Insert cell using file operations (non-collaborative mode).
                
        Args:
            notebook_path: Absolute path to the notebook
            cell_index: Index to insert at (-1 for append)
            cell_type: Type of cell to insert ("code", "markdown")
            cell_source: Source content for the cell
            
        Returns:
            Tuple of (notebook, actual_index, total_cells_after_insertion)
            
        Raises:
            IndexError: When cell_index is out of range
            ValueError: When cell_type is invalid
        """
        # Read notebook file
        with open(notebook_path, "r", encoding="utf-8") as f:
            # Read as version 4 (latest) to ensure consistency and support for cell IDs
            notebook = nbformat.read(f, as_version=4)
        
        # Clean any transient fields from existing outputs (kernel protocol field not in nbformat schema)
        clean_notebook_outputs(notebook)
        
        total_cells = len(notebook.cells)
        
        # Validate insertion parameters
        actual_index = self._validate_cell_insertion_params(
            cell_index, total_cells, cell_type
        )
        
        # Create and insert the cell using unified method
         # Create and insert the cell
        if cell_type == "code":
            new_cell = nbformat.v4.new_code_cell(source=cell_source or "")
        elif cell_type == "markdown":
            new_cell = nbformat.v4.new_markdown_cell(source=cell_source or "")
        notebook.cells.insert(actual_index, new_cell)
        
        # Write back to file
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        
        notebook = Notebook(**notebook)
        
        return notebook, actual_index, len(notebook.cells)
    
    async def _insert_cell_websocket(
        self,
        notebook_manager: NotebookManager,
        cell_index: int,
        cell_type: Literal["code", "markdown"],
        cell_source: str
    ) -> tuple[Notebook, int, int]:
        """Insert cell using WebSocket connection (MCP_SERVER mode).
        
        Args:
            notebook_manager: Notebook manager instance
            cell_index: Index to insert at (-1 for append)
            cell_type: Type of cell to insert ("code", "markdown")
            cell_source: Source content for the cell
            
        Returns:
            Tuple of (notebook, actual_index, total_cells_after_insertion)
            
        Raises:
            IndexError: When cell_index is out of range
            ValueError: When cell_type is invalid
        """
        async with notebook_manager.get_current_connection() as notebook:
            total_cells = len(notebook)
            
            # Validate insertion parameters
            actual_index = self._validate_cell_insertion_params(
                cell_index, total_cells, cell_type
            )
            
            # Use the unified insert_cell method pattern
            # The remote notebook should have: insert_cell(index, source, cell_type)
            notebook.insert_cell(actual_index, cell_source, cell_type)
            
            return Notebook(**notebook.as_dict()), actual_index, len(notebook)

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
        cell_type: Literal["code", "markdown"] = None,
        cell_source: str = None,
        session_id: Optional[str] = None,  # ARK-165: Multi-client session support
        **kwargs
    ) -> dict:
        """Insert a cell into notebook with session-aware context (ARK-165).

        This tool inserts a cell into the notebook associated with the given session_id.
        It automatically retrieves the correct notebook from the SessionStore and supports
        multiple execution modes with thread-safe operations.

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
            cell_index: Target index for insertion (0-based)
                - Use -1 to append at end
                - Must be <= notebook length
            cell_type: Type of cell to insert
                - "code": Executable code cell
                - "markdown": Markdown documentation cell
            cell_source: Source content for the new cell
                - Code or markdown text
            session_id: Client session ID for multi-client support (ARK-165)
                - Required for multi-client environments
                - Used to lookup notebook context from SessionStore
                - If omitted, falls back to config default
            **kwargs: Additional parameters

        Returns:
            Structured dict with insertion results (if session_id provided):
            ```python
            {
                "result": str,  // Success message with context
                "session_id": str | None,  // Client session ID
                "notebook_path": str,  // Path to modified notebook
                "metadata": {
                    "cell_index": int,     // Actual insertion index
                    "cell_type": str,      // Type of inserted cell
                    "total_cells": int,    // New total cell count
                    "timestamp": str       // ISO 8601 format
                }
            }
            ```
            Or simple string (backward compatible, if no session_id)

        Raises:
            ValueError: If mode invalid, cell_type invalid, or clients missing
            IndexError: If cell_index out of range
            FileNotFoundError: If notebook file doesn't exist

        Example:
            Single client (backward compatible):
            ```python
            result = await insert_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=0,
                cell_type="code",
                cell_source="print('Hello')",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )
            ```

            Multi-client (ARK-165):
            ```python
            # Client A inserts cell in their notebook
            result_a = await insert_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=0,
                cell_type="markdown",
                cell_source="# My Notes",
                session_id="session-A-uuid",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )

            # Client B inserts cell in their notebook (independent)
            result_b = await insert_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=-1,  # append
                cell_type="code",
                cell_source="result = 42",
                session_id="session-B-uuid",
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )
            ```

        Notes:
            - ARK-165: Session-based context lookup from SessionStore
            - Thread-safe operations with YDoc transactions
            - Automatic fallback to file mode if YDoc unavailable
            - Returns surrounding cells for context
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
                notebook, actual_index, new_total_cells = await self._insert_cell_ydoc(
                    serverapp, notebook_path, cell_index, cell_type, cell_source
                )
            else:
                # Fall back to file operations
                notebook, actual_index, new_total_cells = await self._insert_cell_file(
                    notebook_path, cell_index, cell_type, cell_source
                )
                
        elif mode == ServerMode.MCP_SERVER and notebook_manager is not None:
            # MCP_SERVER mode: Use WebSocket connection with unified insert_cell pattern
            notebook, actual_index, new_total_cells = await self._insert_cell_websocket(
                notebook_manager, cell_index, cell_type, cell_source
            )
        else:
            raise ValueError(f"Invalid mode or missing required clients: mode={mode}")
        
        info_list = [f"Cell inserted successfully at index {actual_index} ({cell_type})!"]
        info_list.append(f"Notebook now has {new_total_cells} cells, showing surrounding cells:")
        # Show context near the insertion
        if new_total_cells - actual_index < 5:
            start_index = max(0, new_total_cells - 10)
        else:
            start_index = max(0, actual_index - 5)
        info_list.append(notebook.format_output(response_format="brief", start_index=start_index, limit=10))

        result_str = "\n".join(info_list)

        # ARK-165: Structured output with session_id (backward compatible)
        if session_id is not None:
            from datetime import datetime
            return json.dumps({
                "result": result_str,
                "session_id": session_id,
                "notebook_path": notebook_path,
                "metadata": {
                    "cell_index": actual_index,
                    "cell_type": cell_type,
                    "new_total_cells": new_total_cells,
                    "timestamp": datetime.now().isoformat()
                }
            }, indent=2)
        else:
            # Backward compatibility: return string only
            return result_str
        

