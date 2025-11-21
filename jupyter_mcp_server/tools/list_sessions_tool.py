# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""List sessions tool implementation."""

from typing import Any, Optional
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.session_store import SessionStore
from jupyter_mcp_server.utils import format_TSV


class ListSessionsTool(BaseTool):
    """Tool to list all active sessions with their notebook and kernel information."""

    async def execute(
        self,
        mode: ServerMode,
        session_store: SessionStore,
        server_client: Optional[Any] = None,
        contents_manager: Optional[Any] = None,
        kernel_manager: Optional[Any] = None,
        kernel_spec_manager: Optional[Any] = None,
        **kwargs
    ) -> str:
        """Execute the list_sessions tool.

        This tool lists all active sessions with their associated notebook and kernel information.
        Each session represents a client's isolated notebook context (session : notebook : kernel = 1:1:1).

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            session_store: SessionStore instance containing all active sessions
            **kwargs: Additional parameters (unused)

        Returns:
            TSV formatted table with session information
        """
        if session_store is None:
            return "No session store available."

        # Get all active sessions
        all_sessions = session_store.list_all()

        if not all_sessions:
            return "No active sessions. Use the use_notebook tool with a session_id to create a session."

        # Create TSV formatted output
        headers = ["Session_ID", "Notebook_Name", "Notebook_Path", "Kernel_ID", "Last_Accessed"]
        rows = []

        # Sort by last_accessed (most recent first)
        sorted_sessions = sorted(
            all_sessions.items(),
            key=lambda x: x[1].last_accessed,
            reverse=True
        )

        for session_id, ctx in sorted_sessions:
            # Truncate session_id for readability (show first 8 chars)
            short_session_id = session_id[:8] + "..." if len(session_id) > 8 else session_id

            rows.append([
                short_session_id,
                ctx.current_notebook or "-",
                ctx.notebook_path or "-",
                ctx.kernel_id or "-",
                ctx.last_accessed.strftime("%Y-%m-%d %H:%M:%S") if ctx.last_accessed else "-"
            ])

        return format_TSV(headers, rows)
