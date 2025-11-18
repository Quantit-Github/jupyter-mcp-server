# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""Restart notebook tool implementation."""

import logging
from typing import Any, Optional
from jupyter_server_client import JupyterServerClient
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.session_store import SessionStore

logger = logging.getLogger(__name__)


class RestartNotebookTool(BaseTool):
    """Tool to restart the kernel for a specific session."""

    async def execute(
        self,
        mode: ServerMode,
        server_client: Optional[JupyterServerClient] = None,
        kernel_client: Optional[Any] = None,
        contents_manager: Optional[Any] = None,
        kernel_manager: Optional[Any] = None,
        kernel_spec_manager: Optional[Any] = None,
        notebook_manager: Optional[NotebookManager] = None,
        session_store: Optional[SessionStore] = None,
        # Tool-specific parameters
        session_id: Optional[str] = None,
        notebook_name: Optional[str] = None,  # Backward compatibility
        **kwargs
    ) -> str:
        """Execute the restart_notebook tool.

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            kernel_manager: Kernel manager for JUPYTER_SERVER mode
            notebook_manager: Notebook manager instance (for backward compatibility)
            session_store: SessionStore instance (for session-based operation)
            session_id: Client session ID for multi-client support
            notebook_name: Notebook identifier (backward compatibility, deprecated)
            **kwargs: Additional parameters

        Returns:
            Success message
        """
        # ARK-165: Session-based operation (preferred)
        if session_id and session_store:
            ctx = session_store.get(session_id)
            if not ctx:
                return f"Session '{session_id[:8]}...' not found. Use list_sessions to see active sessions."

            if not ctx.kernel_id:
                return f"Session '{session_id[:8]}...' has no active kernel."

            kernel_id = ctx.kernel_id
            notebook_name = ctx.current_notebook or "Unknown"
        # Backward compatibility: notebook_name based operation
        elif notebook_name and notebook_manager:
            if notebook_name not in notebook_manager:
                return f"Notebook '{notebook_name}' is not connected. All currently connected notebooks: {list(notebook_manager.list_all_notebooks().keys())}"

            kernel_id = notebook_manager.get_kernel_id(notebook_name)
            if not kernel_id:
                return f"Failed to restart notebook '{notebook_name}': kernel ID not found."
        else:
            return "Either session_id or notebook_name must be provided."

        if mode == ServerMode.JUPYTER_SERVER:
            # JUPYTER_SERVER mode: Use kernel_manager to restart the kernel
            if kernel_manager is None:
                return f"Failed to restart: kernel_manager is required in JUPYTER_SERVER mode."

            try:
                logger.info(f"Restarting kernel {kernel_id} for notebook '{notebook_name}' in JUPYTER_SERVER mode")
                await kernel_manager.restart_kernel(kernel_id)
                return f"Kernel for notebook '{notebook_name}' restarted successfully. Memory state and imported packages have been cleared."
            except Exception as e:
                logger.error(f"Failed to restart kernel {kernel_id}: {e}")
                return f"Failed to restart: {e}"

        elif mode == ServerMode.MCP_SERVER:
            # MCP_SERVER mode: Use notebook_manager's restart_notebook method (backward compatibility)
            if not notebook_manager or not notebook_name:
                return "MCP_SERVER mode requires notebook_manager and notebook_name"

            success = notebook_manager.restart_notebook(notebook_name)

            if success:
                return f"Notebook '{notebook_name}' kernel restarted successfully. Memory state and imported packages have been cleared."
            else:
                return f"Failed to restart notebook '{notebook_name}'. The kernel may not support restart operation."
        else:
            return f"Invalid mode: {mode}"
