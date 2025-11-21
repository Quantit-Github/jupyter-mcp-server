# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""Use notebook tool implementation."""

import logging
import json
import os
from datetime import datetime
from typing import Any, Optional, Literal
from pathlib import Path
from jupyter_server_client import JupyterServerClient, NotFoundError
from jupyter_kernel_client import KernelClient
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.models import Notebook
from jupyter_mcp_server.session_manager import SessionManager

logger = logging.getLogger(__name__)


class UseNotebookTool(BaseTool):
    """Tool to use (connect to or create) a notebook file."""

    def _determine_action(self, message: str, use_mode: Optional[str]) -> str:
        """
        Determine action type from result message (ARK-165).

        Args:
            message: Result message from operation
            use_mode: Optional use_mode parameter

        Returns:
            Action type: "created" | "connected" | "switched" | "unknown"
        """
        message_lower = message.lower()
        if "successfully activate" in message_lower:
            return "created"
        elif "already created" in message_lower or "already activated" in message_lower:
            return "switched"
        elif "reactivating" in message_lower:
            return "switched"
        elif "connected to kernel" in message_lower:
            return "connected"
        return use_mode if use_mode else "unknown"

    def _detect_error(self, result: str) -> bool:
        """
        Detect if result indicates an error (ARK-165).

        Args:
            result: Result message to check

        Returns:
            True if error detected, False otherwise
        """
        # First check for success indicators
        is_success = (
            "Successfully" in result or
            "[INFO]" in result or
            "Connected to kernel" in result or
            "already created" in result or
            "already activated" in result
        )
        if is_success:
            return False

        # Check for error keywords
        error_keywords = [
            "not found", "failed", "error", "invalid",
            "not the correct", "no session_manager", "no valid"
        ]
        return any(keyword in result.lower() for keyword in error_keywords)

    async def _start_kernel_local(self, kernel_manager: Any):
        # Start a new kernel using local API
        kernel_id = await kernel_manager.start_kernel()
        logger.info(f"Started kernel '{kernel_id}', waiting for it to be ready...")
        
        # CRITICAL: Wait for the kernel to actually start and be ready
        # The start_kernel() call returns immediately, but kernel takes time to start
        import asyncio
        max_wait_time = 30  # seconds
        wait_interval = 0.5  # seconds
        elapsed = 0
        kernel_ready = False
        
        while elapsed < max_wait_time:
            try:
                # Get kernel model to check its state
                kernel_model = kernel_manager.get_kernel(kernel_id)
                if kernel_model is not None:
                    # Kernel exists, check if it's ready
                    # In Jupyter, we can try to get connection info which indicates readiness
                    try:
                        kernel_manager.get_connection_info(kernel_id)
                        kernel_ready = True
                        logger.info(f"Kernel '{kernel_id}' is ready (took {elapsed:.1f}s)")
                        break
                    except:
                        # Connection info not available yet, kernel still starting
                        pass
            except Exception as e:
                logger.debug(f"Waiting for kernel to start: {e}")
            
            await asyncio.sleep(wait_interval)
            elapsed += wait_interval
        
        if not kernel_ready:
            logger.warning(f"Kernel '{kernel_id}' may not be fully ready after {max_wait_time}s wait")
        
        return {"id": kernel_id}

    async def _check_path_http(
        self, 
        server_client: JupyterServerClient, 
        notebook_path: str, 
        mode: str
    ) -> tuple[bool, Optional[str]]:
        """Check if path exists using HTTP API."""
        path = Path(notebook_path)
        try:
            parent_path = path.parent.as_posix() if path.parent.as_posix() != "." else ""
            
            if parent_path:
                dir_contents = server_client.contents.list_directory(parent_path)
            else:
                dir_contents = server_client.contents.list_directory("")
                
            if mode == "connect":
                file_exists = any(file.name == path.name for file in dir_contents)
                if not file_exists:
                    return False, f"'{notebook_path}' not found in jupyter server, please check the notebook already exists."
            
            return True, None
        except NotFoundError:
            parent_dir = path.parent.as_posix() if path.parent.as_posix() != "." else "root directory"
            return False, f"'{parent_dir}' not found in jupyter server, please check the directory path already exists."
        except Exception as e:
            return False, f"Failed to check the path '{notebook_path}': {e}"
    
    async def _check_path_local(
        self,
        contents_manager: Any,
        notebook_path: str,
        mode: str
    ) -> tuple[bool, Optional[str]]:
        """Check if path exists using local contents_manager API."""
        path = Path(notebook_path)
        try:
            parent_path = str(path.parent) if str(path.parent) != "." else ""
            
            # Get directory contents using local API
            model = await contents_manager.get(parent_path, content=True, type='directory')
            
            if mode == "connect":
                file_exists = any(item['name'] == path.name for item in model.get('content', []))
                if not file_exists:
                    return False, f"'{notebook_path}' not found in jupyter server, please check the notebook already exists."
            
            return True, None
        except Exception as e:
            parent_dir = str(path.parent) if str(path.parent) != "." else "root directory"
            return False, f"'{parent_dir}' not found in jupyter server: {e}"
    
    async def execute(
        self,
        mode: ServerMode,
        server_client: Optional[JupyterServerClient] = None,
        kernel_client: Optional[Any] = None,
        contents_manager: Optional[Any] = None,
        kernel_manager: Optional[Any] = None,
        kernel_spec_manager: Optional[Any] = None,
        session_manager: Optional[Any] = None,
        notebook_manager: Optional[NotebookManager] = None,
        session_store: Optional[Any] = None,  # SessionStore instance for testing
        # Tool-specific parameters
        notebook_name: str = None,
        notebook_path: str = None,
        use_mode: Literal["connect", "create"] = "connect",
        runtime_url: Optional[str] = None,
        runtime_token: Optional[str] = None,
        session_id: Optional[str] = None,  # ARK-165: Multi-client session support
        kernel_id: Optional[str] = None,  # ARK-165: Kernel ID for reuse
        **kwargs
    ) -> str:
        """Connect to or create a notebook and register it in the session (ARK-165).

        This tool establishes a connection to a Jupyter notebook and registers it in the
        SessionStore for multi-client support. Each client can maintain independent notebook
        contexts using unique session_ids.

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            server_client: HTTP client for MCP_SERVER mode
            contents_manager: Direct API access for JUPYTER_SERVER mode
            kernel_manager: Direct kernel manager for JUPYTER_SERVER mode
            session_manager: Session manager for creating kernel-notebook associations
            notebook_manager: Notebook manager instance
            session_store: SessionStore instance for testing (optional, defaults to global)
            notebook_name: Unique identifier for the notebook
            notebook_path: Path to the notebook file (.ipynb)
                - Relative path from Jupyter server root
                - For create mode, file will be created if it doesn't exist
                - For connect mode, file must already exist
            use_mode: Operation mode
                - "connect": Connect to existing notebook (default)
                - "create": Create new notebook file
            runtime_url: Runtime URL for HTTP mode (MCP_SERVER only)
            runtime_token: Runtime token for HTTP mode (MCP_SERVER only)
            session_id: Client session ID for multi-client support (ARK-165)
                - Required for multi-client environments
                - UUID format recommended (e.g., str(uuid.uuid4()))
                - If omitted, falls back to config default (single-client mode)
                - Same session_id should be used for all operations by the same client
            **kwargs: Additional parameters

        Note:
            Kernel ID is automatically managed via SessionStore:
            - First call with session_id: Creates new kernel, stores in SessionStore
            - Subsequent calls: Reuses kernel from SessionStore automatically
            - No need to manually specify kernel_id

        Returns:
            Structured JSON string with session info and notebook details:
            ```json
            {
                "result": {
                    "status": "success" | "error",
                    "message": str,  // Human-readable result message
                    "action": "created" | "connected" | "switched" | "unknown",
                    "notebook_url": str | None,  // JupyterHub URL (if available)
                    "jupyter_token": str | None  // JupyterHub token (if available)
                },
                "session_id": str | None,  // Client session ID
                "notebook_path": str,      // Path to notebook
                "metadata": {
                    "notebook_name": str,  // Notebook identifier
                    "kernel_id": str,      // Kernel ID used
                    "mode": "JUPYTER_SERVER" | "MCP_SERVER",
                    "timestamp": str       // ISO 8601 format
                }
            }
            ```

        Raises:
            ValueError: If notebook_path or notebook_name is missing
            FileNotFoundError: If notebook doesn't exist in connect mode
            RuntimeError: If kernel connection fails

        Example:
            Single client (backward compatible):
            ```python
            result = await use_notebook.execute(
                mode=ServerMode.JUPYTER_SERVER,
                notebook_path="analysis.ipynb",
                notebook_name="analysis",
                kernel_manager=kernel_manager,
                contents_manager=contents_manager,
                notebook_manager=notebook_manager,
                use_mode="connect"
            )
            ```

            Multi-client (ARK-165):
            ```python
            import uuid

            # Client A
            session_a = str(uuid.uuid4())
            result_a = await use_notebook.execute(
                mode=ServerMode.JUPYTER_SERVER,
                notebook_path="notebook_a.ipynb",
                notebook_name="a",
                session_id=session_a,
                kernel_manager=kernel_manager,
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )

            # Client B (works independently)
            session_b = str(uuid.uuid4())
            result_b = await use_notebook.execute(
                mode=ServerMode.JUPYTER_SERVER,
                notebook_path="notebook_b.ipynb",
                notebook_name="b",
                session_id=session_b,
                kernel_manager=kernel_manager,
                contents_manager=contents_manager,
                notebook_manager=notebook_manager
            )
            ```

        Notes:
            - ARK-165: This tool now supports multi-client sessions
            - Duplicate kernel bug fix: Kernel ID injection prevents creating multiple kernels
            - JupyterHub integration: Automatically generates notebook_url in JupyterHub env
            - Structured output: All responses follow consistent JSON format
            - Kernel Auto-Recovery: If a kernel is not found (e.g., manually terminated),
              a new kernel is automatically created with a warning log. The SessionStore
              is updated with the new kernel_id automatically.
        """
        # ARK-165: session_id is required for multi-client support
        if not session_id:
            structured_output = {
                "result": {
                    "status": "error",
                    "message": "session_id is required for multi-client support (ARK-165). Please provide a valid session_id.",
                    "action": "rejected"
                },
                "metadata": {
                    "mode": mode.value if mode else None,
                    "timestamp": datetime.now().isoformat()
                }
            }
            return json.dumps(structured_output, ensure_ascii=False)

        # ARK-165: Check SessionStore first for existing session context
        # This enables early detection of same-session reuse and kernel_id lookup
        # Use provided session_store parameter (for testing), or fall back to global
        if session_store is None:
            from jupyter_mcp_server.server import session_store

        # ARK-165: SessionManager를 사용한 커널 헬스 체크 (리팩토링)
        # SessionManager가 커널 존재 여부를 확인하여 early exit 판단
        # Note: Use different variable name to avoid shadowing Jupyter's session_manager parameter
        mcp_session_manager = SessionManager(session_store)
        ctx, kernel_healthy = await mcp_session_manager.get_session_with_kernel_check(
            session_id=session_id,
            kernel_manager=kernel_manager,
            server_client=server_client,
            mode=mode
        )

        # ARK-165: Early exit if same session is reusing same notebook AND kernel is still valid
        # Scenario: A session already using a.ipynb → A session calls use_notebook(a.ipynb) again
        # Result: No need to recreate/reconnect, just return success
        if ctx and ctx.notebook_path == notebook_path and kernel_healthy:
            # Kernel is valid, safe to early return
            logger.info(f"✓ [ARK-165] Session '{session_id[:8]}...' already using '{notebook_path}' with valid kernel")
            structured_output = {
                "result": {
                    "status": "success",
                    "message": f"Already using notebook '{notebook_name}' with kernel '{ctx.kernel_id}'",
                    "action": "reused",
                    "session_id": session_id,
                    "notebook_name": notebook_name,
                    "notebook_path": notebook_path,
                    "kernel_id": ctx.kernel_id,
                },
                "metadata": {
                    "notebook_name": notebook_name,
                    "kernel_id": ctx.kernel_id,
                    "mode": mode.value if mode else None,
                    "timestamp": datetime.now().isoformat()
                }
            }
            return json.dumps(structured_output, ensure_ascii=False)
        elif ctx and ctx.notebook_path == notebook_path and not kernel_healthy:
            # Kernel is invalid, continue to recovery logic
            logger.warning(f"⚠ [ARK-165] Session '{session_id[:8]}...' has invalid kernel '{ctx.kernel_id}', will recover")

        # ARK-165: Prevent multiple sessions from using the same notebook file
        # Scenario: A session using a.ipynb → B session tries to use a.ipynb
        # Result: Reject with error (prevent file-level conflicts)
        # Note: session_store is already available from parameter or global import above
        for other_session_id, ctx in session_store.list_all().items():
            if other_session_id != session_id and ctx.notebook_path == notebook_path:
                error_msg = (
                    f"Notebook '{notebook_path}' is already in use by another session "
                    f"(session: {other_session_id[:8]}...). "
                    f"Multiple sessions cannot use the same notebook file simultaneously to prevent conflicts."
                )
                logger.warning(f"✗ [ARK-165] {error_msg}")
                structured_output = {
                    "result": {
                        "status": "error",
                        "message": error_msg,
                        "action": "rejected",
                        "session_id": session_id,
                        "notebook_path": notebook_path,
                        "conflicting_session": other_session_id[:8] + "...",
                    },
                    "metadata": {
                        "mode": mode.value if mode else None,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                return json.dumps(structured_output, ensure_ascii=False)

        # ARK-165: Extract kernel_id from existing session if available
        # Scenario: A session using a.ipynb → A session calls use_notebook(b.ipynb)
        # Result: Reuse same kernel, just switch notebook
        # Only override kernel_id if not explicitly provided
        if kernel_id is None and ctx:
            kernel_id = ctx.kernel_id
            if kernel_id:
                logger.info(f"✓ [ARK-165] Reusing kernel '{kernel_id}' from session '{session_id[:8]}...'")

        # Check the path exists
        if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
            path_ok, error_msg = await self._check_path_local(contents_manager, notebook_path, use_mode)
        elif mode == ServerMode.MCP_SERVER and server_client is not None:
            path_ok, error_msg = await self._check_path_http(server_client, notebook_path, use_mode)
        else:
            return f"Invalid mode or missing required clients: mode={mode}"
        
        if not path_ok:
            return error_msg
        
        info_list = []

        # ARK-165: SessionManager를 사용한 커널 힐링 (리팩토링)
        # kernel_id가 있지만 커널이 unhealthy이고 session이 존재하면 SessionManager로 자동 복구
        if kernel_id and not kernel_healthy and ctx:
            logger.warning(
                f"Kernel '{kernel_id}' not found in session '{session_id[:8]}...'. "
                f"Attempting auto-recovery with SessionManager."
            )
            new_kernel_id = await mcp_session_manager.heal_kernel(
                session_id=session_id,
                kernel_manager=kernel_manager,
                server_client=server_client,
                mode=mode,
                notebook_path=notebook_path
            )
            if new_kernel_id:
                kernel_id = new_kernel_id
                kernel_healthy = True  # 힐링 성공 - 커널이 이제 healthy함
                info_list.append(
                    f"[INFO] Previous kernel was not found (may have been terminated). "
                    f"Created new kernel '{kernel_id}'."
                )
            else:
                # 힐링 실패 - 새 커널 생성으로 fallback
                logger.warning(f"Kernel healing failed for session '{session_id[:8]}...', will create new kernel")
                kernel_id = None  # 새 커널 생성하도록 None으로 설정

        # Create/connect to kernel based on mode
        if mode == ServerMode.MCP_SERVER and server_client is not None:
            try:
                kernel = KernelClient(
                    server_url=runtime_url,
                    token=runtime_token,
                    kernel_id=kernel_id
                )
                # FIXED: Ensure kernel is started with the same path as the notebook
                kernel.start(path=notebook_path)
                info_list.append(f"[INFO] Connected to kernel '{kernel.id}'.")
            except Exception as e:
                # Kernel creation/connection failed - return clear error message
                logger.error(f"Failed to create/connect kernel for session '{session_id[:8]}...': {e}")
                return f"Failed to connect to kernel: {e}"
        elif mode == ServerMode.JUPYTER_SERVER and kernel_manager is not None:
            # JUPYTER_SERVER mode: Use local kernel manager API directly
            if kernel_id:
                # kernel_id가 있으면 재사용 시도
                if kernel_id in kernel_manager:
                    # 커널이 실제로 존재하면 재사용
                    kernel = {"id": kernel_id}
                else:
                    # 커널이 없으면 새로 생성 (이미 heal을 시도했거나 ctx가 없어서 heal을 못함)
                    kernel = await self._start_kernel_local(kernel_manager)
                    kernel_id = kernel['id']
                    info_list.append(
                        f"[INFO] Previous kernel was not found (may have been terminated). "
                        f"Created new kernel '{kernel_id}'."
                    )
            else:
                # kernel_id가 없으면 새 커널 생성 (first connection 또는 healing 실패 후)
                kernel = await self._start_kernel_local(kernel_manager)
                kernel_id = kernel['id']
                # ctx가 있으면 healing 실패 후 새 커널 생성
                if ctx:
                    info_list.append(
                        f"[INFO] Previous kernel was not found (may have been terminated). "
                        f"Created new kernel '{kernel_id}'."
                    )

            if not any("[INFO] Previous kernel" in msg for msg in info_list):
                info_list.append(f"[INFO] Connected to kernel '{kernel_id}'.")
            # Create a Jupyter session to associate the kernel with the notebook
            # This is CRITICAL for JupyterLab to recognize the kernel-notebook connection
            if session_manager is not None:
                try:
                    # create_session is an async method, so we await it directly
                    session_dict = await session_manager.create_session(
                        path=notebook_path,
                        kernel_id=kernel_id,
                        type="notebook",
                        name=notebook_path
                    )
                    logger.info(f"Created Jupyter session '{session_dict.get('id')}' for notebook '{notebook_path}' with kernel '{kernel_id}'")
                except Exception as e:
                    logger.warning(f"Failed to create Jupyter session: {e}. Notebook may not be properly connected in JupyterLab UI.")
            else:
                logger.warning("No session_manager available. Notebook may not be properly connected in JupyterLab UI.")

        # Create notebook if needed
        if use_mode == "create":
            content = {
                "cells": [{
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "New Notebook Created by Jupyter MCP Server",
                    ]
                }],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 4
            }
            if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
                # Use local API to create notebook
                await contents_manager.new(model={'type': 'notebook'}, path=notebook_path)
            elif mode == ServerMode.MCP_SERVER and server_client is not None:
                server_client.contents.create_notebook(notebook_path, content=content)

        # Add notebook to notebook_manager
        if mode == ServerMode.MCP_SERVER and runtime_url:
            notebook_manager.add_notebook(
                notebook_name,
                kernel,
                server_url=runtime_url,
                token=runtime_token,
                path=notebook_path
            )
        elif mode == ServerMode.JUPYTER_SERVER and kernel_manager is not None:
            notebook_manager.add_notebook(
                notebook_name,
                kernel,
                server_url="local",
                token=None,
                path=notebook_path
            )
        else:
            return f"Invalid configuration: mode={mode}, runtime_url={runtime_url}, kernel_manager={kernel_manager is not None}"

        notebook_manager.set_current_notebook(notebook_name)
        info_list.append(f"[INFO] Successfully activate notebook '{notebook_name}'.")
        
        # Return the quick overview of currently activated notebook
        try:
            if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
                # Read notebook to get cell count and first 20 cells
                model = await contents_manager.get(notebook_path, content=True, type='notebook')
                if 'content' in model:
                    notebook = Notebook(**model['content'])
                else:
                    notebook = Notebook()
            
            elif mode == ServerMode.MCP_SERVER and notebook_manager is not None:
                # Use notebook manager to get cell info
                async with notebook_manager.get_current_connection() as notebook_content:
                    notebook = Notebook(**notebook_content.as_dict())

            info_list.append(f"\nNotebook has {len(notebook)} cells.")
            info_list.append(f"Showing first {min(20, len(notebook))} cells:\n")
            info_list.append(notebook.format_output(response_format="brief", start_index=0, limit=20))
        except Exception as e:
            logger.debug(f"Failed to get notebook summary: {e}")
        
        # Check if we should open in JupyterLab UI (when JupyterLab mode is enabled)
        try:
            from jupyter_mcp_server.jupyter_extension.context import get_server_context
            context = get_server_context()
            
            if context.is_jupyterlab_mode():
                logger.info(f"JupyterLab mode enabled, attempting to open notebook '{notebook_path}' in JupyterLab UI")
                
                # Determine base_url and token based on mode
                base_url = None
                token = None
                
                if mode == ServerMode.JUPYTER_SERVER and context.serverapp is not None:
                    # JUPYTER_SERVER mode: Use ServerApp connection details
                    base_url = context.serverapp.connection_url
                    token = context.serverapp.token
                elif mode == ServerMode.MCP_SERVER and runtime_url:
                    # MCP_SERVER mode: Use runtime_url and runtime_token
                    base_url = runtime_url
                    token = runtime_token
                
                if base_url and token:
                    try:
                        from jupyter_mcp_tools.client import MCPToolsClient
                        
                        async with MCPToolsClient(base_url=base_url, token=token) as client:
                            execution_result = await client.execute_tool(
                                tool_id="docmanager_open",  # docmanager:open converted to underscore format
                                parameters={"path": notebook_path}
                            )
                            
                            if execution_result.get('success'):
                                logger.info(f"Successfully opened notebook '{notebook_path}' in JupyterLab UI")
                            else:
                                logger.warning(f"Failed to open notebook in JupyterLab UI: {execution_result}")
                                
                    except ImportError:
                        logger.warning("jupyter_mcp_tools not available, skipping JupyterLab UI opening")
                    except Exception as e:
                        logger.warning(f"Failed to open notebook in JupyterLab UI: {e}")
                else:
                    logger.warning("No valid base_url or token available for opening notebook in JupyterLab UI")
        except Exception as e:
            logger.debug(f"Could not check JupyterLab mode: {e}")

        # ARK-165: Extract actual kernel_id from kernel object (if kernel was created/connected)
        actual_kernel_id = None
        if 'kernel' in locals():
            if isinstance(kernel, dict):
                actual_kernel_id = kernel.get('id')
            elif hasattr(kernel, 'id'):
                actual_kernel_id = kernel.id

        # ARK-165: Build original result message
        original_result = "\n".join(info_list)

        # ARK-165 FIX 2: Register in SessionStore (Multi-Client Session Persistence)
        # Purpose: Store notebook context for this session_id
        # Benefits:
        #   - Other tools can lookup notebook_path/kernel_id using session_id
        #   - Enables tools to work without explicitly passing notebook context
        #   - Creates isolation between different client sessions
        # Example:
        #   Client A: session_id="A" → notebook="a.ipynb", kernel="kernel-A"
        #   Client B: session_id="B" → notebook="b.ipynb", kernel="kernel-B"
        #   execute_cell(session_id="A") → automatically executes in a.ipynb with kernel-A
        if session_id and actual_kernel_id:
            # Note: session_store is already available from parameter or global import above
            session_store.update_notebook(
                session_id=session_id,
                notebook_name=notebook_name,
                notebook_path=notebook_path,
                kernel_id=actual_kernel_id
            )
            logger.info(f"✓ [ARK-165] Notebook '{notebook_name}' registered to session '{session_id}'")

        # ARK-165 FIX 3: Determine action type (user-facing messaging)
        # Classifies what operation was performed for structured output
        # Actions: "created" | "connected" | "switched" | "unknown"
        action = self._determine_action(original_result, use_mode)

        # ARK-165 FIX 4: Generate JupyterHub URL (multi-tenant environment support)
        # Purpose: Provide direct link to notebook in JupyterHub UI
        # Environment Variables Required:
        #   - JUPYTERHUB_API_TOKEN: Auth token (presence indicates JupyterHub)
        #   - JUPYTERHUB_BASE_URL: Base URL (e.g., "/hub/")
        #   - JUPYTERHUB_USER: Current user name
        # Generated URL Format:
        #   {JUPYTERHUB_BASE_URL}user/{JUPYTERHUB_USER}/notebooks/{notebook_path}
        # Example:
        #   "/hub/user/john/notebooks/work/analysis.ipynb"
        jupyter_token = os.environ.get('JUPYTERHUB_API_TOKEN')
        notebook_url = None
        if notebook_path and jupyter_token:
            jupyterhub_base_url = os.environ.get('JUPYTERHUB_BASE_URL', '')
            jupyterhub_user = os.environ.get('JUPYTERHUB_USER', '')
            if jupyterhub_user:
                notebook_url = f"{jupyterhub_base_url}user/{jupyterhub_user}/notebooks/{notebook_path}"
                logger.info(f"✓ [ARK-165] Generated JupyterHub URL: {notebook_url}")

        # ARK-165 FIX 5: Return structured JSON output (API consistency)
        # Purpose: Standardize output format across all tools
        # Structure:
        #   result: {status, message, action, notebook_url, jupyter_token}
        #   session_id: Client session ID (or None)
        #   notebook_path: Path to notebook
        #   metadata: {notebook_name, kernel_id, mode, timestamp}
        # Benefits:
        #   - Clients can programmatically parse responses
        #   - session_id enables context tracking
        #   - Timestamp for debugging/auditing
        #   - notebook_url for direct browser navigation
        is_error = self._detect_error(original_result)
        structured_output = {
            "result": {
                "status": "error" if is_error else "success",
                "message": original_result,
                "action": action,
                "notebook_url": notebook_url,
                "jupyter_token": jupyter_token
            },
            "session_id": session_id,
            "notebook_path": notebook_path,
            "metadata": {
                "notebook_name": notebook_name,
                "kernel_id": actual_kernel_id,
                "mode": mode.value if mode else None,
                "timestamp": datetime.now().isoformat()
            }
        }

        return json.dumps(structured_output, ensure_ascii=False)
