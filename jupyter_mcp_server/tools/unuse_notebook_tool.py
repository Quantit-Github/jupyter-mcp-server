# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""Unuse notebook tool implementation."""

import logging
from typing import Any, Optional
from jupyter_server_client import JupyterServerClient
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.session_store import SessionStore

logger = logging.getLogger(__name__)


class UnuseNotebookTool(BaseTool):
    """Tool to unuse from a session and release its resources"""

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
        """Execute the unuse_notebook tool.

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            kernel_manager: Kernel manager for JUPYTER_SERVER mode (optional kernel shutdown)
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

            notebook_name = ctx.current_notebook or "Unknown"
            kernel_id = ctx.kernel_id

            # Shutdown kernel if in JUPYTER_SERVER mode and kernel exists
            kernel_shutdown_msg = ""
            if mode == ServerMode.JUPYTER_SERVER and kernel_id and kernel_manager:
                try:
                    logger.info(f"Shutting down kernel {kernel_id} for session '{session_id[:8]}...' in JUPYTER_SERVER mode")
                    await kernel_manager.shutdown_kernel(kernel_id)
                    logger.info(f"Kernel {kernel_id} shutdown successfully")
                    kernel_shutdown_msg = " Kernel shutdown completed."
                except Exception as e:
                    logger.warning(f"Could not shutdown kernel {kernel_id}: {e}")
                    kernel_shutdown_msg = f" Kernel shutdown failed: {e}"

            # ARK-165: Remove session from SessionStore immediately
            removed = session_store.remove(session_id)
            if removed:
                logger.info(f"✓ [ARK-165] Session '{session_id[:8]}...' removed from SessionStore")
                return f"Session '{session_id[:8]}...' (notebook: '{notebook_name}') unused successfully.{kernel_shutdown_msg} Session removed from store."
            else:
                logger.warning(f"✗ [ARK-165] Session '{session_id[:8]}...' not found in SessionStore (may have already expired)")
                return f"Session '{session_id[:8]}...' (notebook: '{notebook_name}') unused.{kernel_shutdown_msg} Session was not found in store."

        # Backward compatibility: notebook_name based operation
        elif notebook_name and notebook_manager:
            if notebook_name not in notebook_manager:
                return f"Notebook '{notebook_name}' is not connected. All currently connected notebooks: {list(notebook_manager.list_all_notebooks().keys())}"

            # Get info about which notebook was current
            current_notebook = notebook_manager.get_current_notebook()
            was_current = current_notebook == notebook_name

            if mode == ServerMode.JUPYTER_SERVER:
                # JUPYTER_SERVER mode: Optionally shutdown kernel before removing
                kernel_id = notebook_manager.get_kernel_id(notebook_name)
                if kernel_id and kernel_manager:
                    try:
                        logger.info(f"Notebook '{notebook_name}' is being unused in JUPYTER_SERVER mode. Kernel {kernel_id} remains running.")
                        # Optional: Uncomment to shutdown kernel when unused
                        # await kernel_manager.shutdown_kernel(kernel_id)
                        # logger.info(f"Kernel {kernel_id} shutdown successfully")
                    except Exception as e:
                        logger.warning(f"Note: Could not access kernel {kernel_id}: {e}")

                success = notebook_manager.remove_notebook(notebook_name)

            elif mode == ServerMode.MCP_SERVER:
                # MCP_SERVER mode: Use notebook_manager's remove_notebook method
                success = notebook_manager.remove_notebook(notebook_name)
            else:
                return f"Invalid mode: {mode}"

            if success:
                message = f"Notebook '{notebook_name}' unused successfully."

                if was_current:
                    new_current = notebook_manager.get_current_notebook()
                    if new_current:
                        message += f" Current notebook switched to '{new_current}'."
                    else:
                        message += " No notebooks remaining."

                return message
            else:
                return f"Notebook '{notebook_name}' was not found."
        else:
            return "Either session_id or notebook_name must be provided."
