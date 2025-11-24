# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""List sessions tool implementation."""

import json
from typing import Any, Optional
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.session_store import SessionStore


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
            JSON formatted list with session information
        """
        if session_store is None:
            return json.dumps({
                "sessions": [],
                "message": "No session store available."
            }, ensure_ascii=False)

        # Get all active sessions
        all_sessions = session_store.list_all()

        if not all_sessions:
            return json.dumps({
                "sessions": [],
                "message": "No active sessions. Use the use_notebook tool with a session_id to create a session."
            }, ensure_ascii=False)

        # Sort by last_accessed (most recent first)
        sorted_sessions = sorted(
            all_sessions.items(),
            key=lambda x: x[1].last_accessed,
            reverse=True
        )

        # Build JSON response
        sessions = []
        for session_id, ctx in sorted_sessions:
            sessions.append({
                "session_id": session_id,
                "notebook_name": ctx.current_notebook,
                "notebook_path": ctx.notebook_path,
                "kernel_id": ctx.kernel_id,
                "last_accessed": ctx.last_accessed.isoformat() if ctx.last_accessed else None
            })

        return json.dumps({
            "sessions": sessions
        }, ensure_ascii=False)
