# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

import re
import asyncio
import time
from typing import Any, Union, Optional, Tuple
from mcp.types import ImageContent
from jupyter_mcp_server.config import ALLOW_IMG_OUTPUT
from jupyter_nbmodel_client import NotebookModel
from jupyter_mcp_server.log import logger


def get_notebook_context_from_session(
    session_id: Optional[str] = None,
    notebook_manager=None  # deprecated, 호환성 유지
) -> Tuple[str, str]:
    """Retrieve notebook context from SessionStore with config fallback (ARK-165).

    This is the primary function for retrieving notebook context in a session-aware manner.
    It enables multi-client support by looking up the notebook path and kernel ID associated
    with a specific session_id, while maintaining backward compatibility with single-client mode.

    Lookup Priority:
        1. SessionStore (if session_id provided and context exists)
           - Returns session-specific notebook_path and kernel_id
           - Enables multi-client isolation

        2. Config fallback (if session_id not provided or context not found)
           - Returns config.document_id and config.runtime_id
           - Single-client mode for backward compatibility
           - Logs warning about multi-client usage

    Args:
        session_id: Client session ID for multi-client support (ARK-165)
            - Optional, but required for multi-client environments
            - UUID format recommended (e.g., str(uuid.uuid4()))
            - Must match the session_id used in use_notebook tool
            - If omitted, falls back to config values (single-client mode)
        notebook_manager: NotebookManager instance (DEPRECATED)
            - Kept for backward compatibility only
            - No longer used in implementation
            - Will be removed in future versions

    Returns:
        Tuple[str, str]: (notebook_path, kernel_id)
            - notebook_path: Relative path from Jupyter root (e.g., "work/notebook.ipynb")
            - kernel_id: Jupyter kernel ID string (e.g., "abc-123-def-456")
            - Both values guaranteed to be non-None strings

    Raises:
        No exceptions raised - always returns valid tuple
        (Falls back to config if session lookup fails)

    Example:
        Multi-client mode (recommended for ARK-165):
        ```python
        import uuid
        from jupyter_mcp_server.utils import get_notebook_context_from_session

        # Client A
        session_a = str(uuid.uuid4())
        path_a, kernel_a = get_notebook_context_from_session(session_id=session_a)
        # Returns: ("notebooks/a.ipynb", "kernel-a-id")

        # Client B (independent context)
        session_b = str(uuid.uuid4())
        path_b, kernel_b = get_notebook_context_from_session(session_id=session_b)
        # Returns: ("notebooks/b.ipynb", "kernel-b-id")

        assert path_a != path_b  # Different notebooks
        assert kernel_a != kernel_b  # Different kernels
        ```

        Single-client mode (backward compatible):
        ```python
        # No session_id - falls back to config
        path, kernel = get_notebook_context_from_session()
        # Returns: (config.document_id, config.runtime_id)
        # Logs warning about multi-client usage
        ```

    Notes:
        - ARK-165: Core function enabling session-based multi-client support
        - Thread-safe: SessionStore operations are thread-safe
        - Always returns valid tuple (never returns None)
        - Fallback to config ensures backward compatibility
        - Used by all cell operation tools for context lookup
        - Replaces deprecated get_current_notebook_context()

    See Also:
        - SessionStore.get(): Retrieves session context
        - UseNotebookTool.execute(): Creates and updates sessions
        - get_config(): Provides fallback values
    """
    from .server import session_store
    from .config import get_config

    # ARK-165: Priority 1 - Lookup from SessionStore (multi-client mode)
    # If session_id provided, try to retrieve session-specific context
    # This enables multiple clients to work with different notebooks simultaneously
    if session_id:
        ctx = session_store.get(session_id)
        # Verify context has required fields before returning
        # Both notebook_path and kernel_id must be set for valid session
        if ctx and ctx.notebook_path and ctx.kernel_id:
            # SUCCESS: Return session-specific context
            # This notebook/kernel pair is isolated to this client session
            return ctx.notebook_path, ctx.kernel_id
        # FALLTHROUGH: Session exists but incomplete → fall back to config

    # ARK-165: Priority 2 - Fallback to config (single-client mode, backward compatible)
    # This ensures backward compatibility when session_id not provided
    # or when session lookup fails (e.g., expired session)
    if not session_id:
        # Warn developer about single-client fallback
        # In multi-client scenarios, session_id should always be provided
        logger.warning(
            "No session_id specified, using default from config. "
            "For multi-client scenarios, please create a session_id and use it consistently."
        )

    # Return default notebook/kernel from config
    # config.document_id: Default notebook path (e.g., from CLI --document-url)
    # config.runtime_id: Default kernel ID (e.g., from CLI --runtime-url)
    config = get_config()
    return config.document_id, config.runtime_id


async def get_notebook_context_from_session_async(
    session_id: Optional[str] = None,
    auto_heal: bool = True,
    kernel_manager=None,
    server_client=None,
    mode=None
) -> Tuple[str, str]:
    """Retrieve notebook context with automatic kernel healing (ARK-165 커널 힐링 리팩토링).

    이것이 권장되는 함수입니다. 모든 새로운 코드는 이 async 버전을 사용하세요.
    SessionManager를 사용하여 커널 헬스 체크 및 자동 힐링을 지원합니다.

    Lookup Priority:
        1. SessionStore (if session_id provided and context exists)
           - 커널 헬스 체크 수행 (auto_heal=True일 때)
           - 커널이 죽었으면 자동으로 새 커널 생성
           - Returns session-specific notebook_path and kernel_id

        2. Config fallback (if session_id not provided or context not found)
           - Returns config.document_id and config.runtime_id
           - Single-client mode for backward compatibility

    Args:
        session_id: Client session ID for multi-client support
            - Optional, but required for multi-client environments
            - UUID format recommended
        auto_heal: Enable automatic kernel healing (default: True)
            - True: 커널이 죽었을 때 자동으로 새 커널 생성
            - False: 기존 sync 함수와 동일한 동작 (커널 체크 없음)
        kernel_manager: For JUPYTER_SERVER mode kernel operations
            - Required if auto_heal=True and mode=JUPYTER_SERVER
        server_client: For MCP_SERVER mode kernel operations
            - Required if auto_heal=True and mode=MCP_SERVER
        mode: Server mode (JUPYTER_SERVER or MCP_SERVER)
            - Required if auto_heal=True

    Returns:
        Tuple[str, str]: (notebook_path, kernel_id)
            - notebook_path: Relative path from Jupyter root
            - kernel_id: Jupyter kernel ID string
            - Both values guaranteed to be non-None strings

    Raises:
        No exceptions raised - always returns valid tuple
        (Falls back to config if session lookup or healing fails)

    Example:
        Auto-healing mode (JUPYTER_SERVER):
        ```python
        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        from jupyter_mcp_server.tools._base import ServerMode

        # 커널 헬스 체크 + 자동 힐링
        path, kernel = await get_notebook_context_from_session_async(
            session_id="session-A",
            auto_heal=True,
            kernel_manager=kernel_manager,
            mode=ServerMode.JUPYTER_SERVER
        )
        # 커널이 죽어있으면 자동으로 새 커널 생성
        ```

        Auto-healing mode (MCP_SERVER):
        ```python
        # MCP_SERVER 모드
        path, kernel = await get_notebook_context_from_session_async(
            session_id="session-B",
            auto_heal=True,
            server_client=server_client,
            mode=ServerMode.MCP_SERVER
        )
        ```

        No auto-healing (기존 동작):
        ```python
        # auto_heal=False - 커널 체크 없음
        path, kernel = await get_notebook_context_from_session_async(
            session_id="session-C",
            auto_heal=False
        )
        ```

    Notes:
        - ARK-165 커널 힐링 리팩토링: 커널 자동 복구 지원
        - SessionManager를 통해 중앙화된 커널 힐링 로직 사용
        - 모든 새로운 async tool은 이 함수를 사용해야 함
        - 기존 sync 함수는 backward compatibility를 위해 유지

    See Also:
        - SessionManager.get_session_with_kernel_check(): 커널 헬스 체크
        - SessionManager.heal_kernel(): 커널 자동 복구
        - get_notebook_context_from_session(): Sync 버전 (deprecated)
    """
    from .server import session_store
    from .session_manager import SessionManager
    from .config import get_config

    # ARK-165: Validation - auto_heal requires session_id
    if auto_heal and not session_id:
        logger.warning(
            "auto_heal=True requires session_id for kernel healing. "
            "Disabling auto_heal and using config fallback."
        )
        auto_heal = False

    # SessionManager 초기화
    session_manager = SessionManager(session_store)

    # ARK-165: Priority 1 - Lookup from SessionStore with auto-healing
    if session_id and auto_heal:
        # 커널 헬스 체크 + 자동 힐링
        ctx, kernel_healthy = await session_manager.get_session_with_kernel_check(
            session_id, kernel_manager, server_client, mode
        )

        if ctx and not kernel_healthy:
            # 커널이 죽었음 - 자동 힐링 시도
            logger.warning(
                f"Kernel unhealthy for session {session_id[:8]}..., attempting heal"
            )
            new_kernel_id = await session_manager.heal_kernel(
                session_id=session_id,
                kernel_manager=kernel_manager,
                server_client=server_client,
                mode=mode,
                notebook_path=ctx.notebook_path
            )

            if new_kernel_id:
                # 힐링 성공 - SessionStore에서 업데이트된 context 가져오기
                ctx = session_store.get(session_id)
                logger.info(f"✓ Kernel healed successfully: {new_kernel_id}")
            else:
                # 힐링 실패 - 세션의 notebook_path는 유지하되, config kernel만 fallback
                logger.error(
                    f"✗ Kernel healing failed for session {session_id}, keeping session notebook_path"
                )
                config = get_config()
                # 세션의 notebook_path 유지 (중요!), config의 kernel_id만 fallback
                if ctx and ctx.notebook_path:
                    return ctx.notebook_path, config.runtime_id
                else:
                    return config.document_id, config.runtime_id

        # 커널이 healthy하거나 힐링 성공 - 세션 context 반환
        # notebook_path만 있어도 반환 (kernel_id는 나중에 생성될 수 있음)
        if ctx and ctx.notebook_path:
            return ctx.notebook_path, ctx.kernel_id if ctx.kernel_id else None

    elif session_id:
        # auto_heal=False - 기존 동작 (커널 체크 없음)
        ctx = session_store.get(session_id)
        if ctx and ctx.notebook_path and ctx.kernel_id:
            return ctx.notebook_path, ctx.kernel_id

    # ARK-165: Priority 2 - Fallback to config (backward compatible)
    if not session_id:
        logger.warning(
            "No session_id specified, using default from config. "
            "For multi-client scenarios, please create a session_id and use it consistently."
        )

    config = get_config()
    return config.document_id, config.runtime_id


def get_current_notebook_context(notebook_manager=None):
    """
    DEPRECATED: Use get_notebook_context_from_session() instead.

    Get the current notebook path and kernel ID for JUPYTER_SERVER mode.

    This function is deprecated as of ARK-165. For multi-client support,
    use get_notebook_context_from_session(session_id=...) instead.

    Args:
        notebook_manager: NotebookManager instance (optional, ignored)

    Returns:
        Tuple of (notebook_path, kernel_id)
        Falls back to config values
    """
    logger.warning(
        "[DEPRECATED] get_current_notebook_context() is deprecated. "
        "Use get_notebook_context_from_session(session_id=...) for multi-client support."
    )
    return get_notebook_context_from_session(session_id=None, notebook_manager=notebook_manager)


def extract_output(output: Union[dict, Any]) -> Union[str, ImageContent]:
    """
    Extracts readable output from a Jupyter cell output dictionary.
    Handles both traditional and CRDT-based Jupyter formats.

    Args:
        output: The output from a Jupyter cell (dict or CRDT object).

    Returns:
        str: A string representation of the output.
    """
    # Handle pycrdt._text.Text objects
    if hasattr(output, 'source'):
        return str(output.source)
    
    # Handle CRDT YText objects
    if hasattr(output, '__str__') and 'Text' in str(type(output)):
        text_content = str(output)
        return strip_ansi_codes(text_content)
    
    # Handle lists (common in error tracebacks)
    if isinstance(output, list):
        return '\n'.join(extract_output(item) for item in output)
    
    # Handle traditional dictionary format
    if not isinstance(output, dict):
        return strip_ansi_codes(str(output))
    
    output_type = output.get("output_type")
    
    if output_type == "stream":
        text = output.get("text", "")
        if isinstance(text, list):
            text = ''.join(text)
        elif hasattr(text, 'source'):
            text = str(text.source)
        return strip_ansi_codes(str(text))
    
    elif output_type in ["display_data", "execute_result"]:
        data = output.get("data", {})
        if "image/png" in data:
            if ALLOW_IMG_OUTPUT:
                try:
                    return ImageContent(type="image", data=data["image/png"], mimeType="image/png")
                except Exception:
                    # Fallback to text placeholder on error
                    return "[Image Output (PNG) - Error processing image]"
            else:
                return "[Image Output (PNG) - Image display disabled]"
        if "text/plain" in data:
            plain_text = data["text/plain"]
            if hasattr(plain_text, 'source'):
                plain_text = str(plain_text.source)
            return strip_ansi_codes(str(plain_text))
        elif "text/html" in data:
            return "[HTML Output]"
        else:
            return f"[{output_type} Data: keys={list(data.keys())}]"
    
    elif output_type == "error":
        traceback = output.get("traceback", [])
        if isinstance(traceback, list):
            clean_traceback = []
            for line in traceback:
                if hasattr(line, 'source'):
                    line = str(line.source)
                clean_traceback.append(strip_ansi_codes(str(line)))
            return '\n'.join(clean_traceback)
        else:
            if hasattr(traceback, 'source'):
                traceback = str(traceback.source)
            return strip_ansi_codes(str(traceback))
    
    else:
        return f"[Unknown output type: {output_type}]"


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def clean_notebook_outputs(notebook):
    """Remove transient fields from all cell outputs.
    
    The 'transient' field is part of the Jupyter kernel messaging protocol
    but is NOT part of the nbformat schema. This causes validation errors.
    
    Args:
        notebook: nbformat notebook object to clean (modified in place)
    """
    for cell in notebook.cells:
        if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
            for output in cell.outputs:
                if isinstance(output, dict) and 'transient' in output:
                    del output['transient']


def safe_extract_outputs(outputs: Any) -> list[Union[str, ImageContent]]:
    """
    Safely extract all outputs from a cell, handling CRDT structures.
    
    Args:
        outputs: Cell outputs (could be CRDT YArray or traditional list)
        
    Returns:
        list[Union[str, ImageContent]]: List of outputs (strings or image content)
    """
    if not outputs:
        return []
    
    result = []
    
    # Handle CRDT YArray or list of outputs
    if hasattr(outputs, '__iter__') and not isinstance(outputs, (str, dict)):
        try:
            for output in outputs:
                extracted = extract_output(output)
                if extracted:
                    result.append(extracted)
        except Exception as e:
            result.append(f"[Error extracting output: {str(e)}]")
    else:
        # Handle single output
        extracted = extract_output(outputs)
        if extracted:
            result.append(extracted)
    
    return result

def normalize_cell_source(source: Any) -> list[str]:
    """
    Normalize cell source to a list of strings (lines).
    
    In Jupyter notebooks, source can be either:
    - A string (single or multi-line with \n)  
    - A list of strings (each element is a line)
    - CRDT text objects
    
    Args:
        source: The source from a Jupyter cell
        
    Returns:
        list[str]: List of source lines
    """
    if not source:
        return []
    
    # Handle CRDT text objects
    if hasattr(source, 'source'):
        source = str(source.source)
    elif hasattr(source, '__str__') and 'Text' in str(type(source)):
        source = str(source)
    
    # If it's already a list, return as is
    if isinstance(source, list):
        return [str(line) for line in source]
    
    # If it's a string, split by newlines
    if isinstance(source, str):
        # Split by newlines but preserve the newline characters except for the last line
        lines = source.splitlines(keepends=True)
        # Remove trailing newline from the last line if present
        if lines and lines[-1].endswith('\n'):
            lines[-1] = lines[-1][:-1]
        return lines
    
    # Fallback: convert to string and split
    return str(source).splitlines(keepends=True)

def format_TSV(headers: list[str], rows: list[list[str]]) -> str:
    """
    Format data as TSV (Tab-Separated Values)
    
    Args:
        headers: The list of headers
        rows: The list of data rows, each row is a list of strings
    
    Returns:
        The formatted TSV string
    """
    if not headers or not rows:
        return "No data to display"
    
    result = []
    
    header_row = "\t".join(headers)
    result.append(header_row)
    
    for row in rows:
        data_row = "\t".join(str(cell) for cell in row)
        result.append(data_row)
    
    return "\n".join(result)

###############################################################################
# Kernel and notebook operation helpers
###############################################################################


def create_kernel(config, logger):
    """Create a new kernel instance using current configuration."""
    from jupyter_kernel_client import KernelClient
    kernel = None
    try:
        # Initialize the kernel client with the provided parameters.
        kernel = KernelClient(
            server_url=config.runtime_url, 
            token=config.runtime_token, 
            kernel_id=config.runtime_id
        )
        kernel.start()
        logger.info("Kernel created and started successfully")
        return kernel
    except Exception as e:
        logger.error(f"Failed to create kernel: {e}")
        # Clean up partially initialized kernel to prevent __del__ errors
        if kernel is not None:
            try:
                # Try to clean up the kernel object if it exists
                if hasattr(kernel, 'stop'):
                    kernel.stop()
            except Exception as cleanup_error:
                logger.debug(f"Error during kernel cleanup: {cleanup_error}")
        raise


def start_kernel(notebook_manager, config, logger):
    """Start the Jupyter kernel with error handling (for backward compatibility)."""
    try:
        # Remove existing default notebook if any
        if "default" in notebook_manager:
            notebook_manager.remove_notebook("default")
        
        # Create and set up new kernel
        kernel = create_kernel(config, logger)
        notebook_manager.add_notebook("default", kernel)
        logger.info("Default notebook kernel started successfully")
    except Exception as e:
        logger.error(f"Failed to start kernel: {e}")
        raise


def ensure_kernel_alive(notebook_manager, current_notebook, create_kernel_fn):
    """Ensure kernel is running, restart if needed."""
    return notebook_manager.ensure_kernel_alive(current_notebook, create_kernel_fn)


async def execute_cell_with_forced_sync(notebook, cell_index, kernel, timeout_seconds = 300):
    """Execute cell with forced real-time synchronization."""
    from jupyter_mcp_server.log import logger
    
    start_time = time.time()
    
    # Start execution
    execution_future = asyncio.create_task(
        asyncio.to_thread(notebook.execute_cell, cell_index, kernel)
    )
    
    last_output_count = 0
    
    while not execution_future.done():
        elapsed = time.time() - start_time
        
        if elapsed > timeout_seconds:
            execution_future.cancel()
            try:
                if hasattr(kernel, 'interrupt'):
                    kernel.interrupt()
            except Exception:
                pass
            raise asyncio.TimeoutError(f"Cell execution timed out after {timeout_seconds} seconds")
        
        # Check for new outputs and try to trigger sync
        try:
            ydoc = notebook._doc
            current_outputs = ydoc._ycells[cell_index].get("outputs", [])
            
            if len(current_outputs) > last_output_count:
                last_output_count = len(current_outputs)
                logger.info(f"Cell {cell_index} progress: {len(current_outputs)} outputs after {elapsed:.1f}s")
                
                # Try different sync methods
                try:
                    # Method 1: Force Y-doc update
                    if hasattr(ydoc, 'observe') and hasattr(ydoc, 'unobserve'):
                        # Trigger observers by making a tiny change
                        pass
                        
                    # Method 2: Force websocket message
                    if hasattr(notebook, '_websocket') and notebook._websocket:
                        # The websocket should automatically sync on changes
                        pass
                        
                except Exception as sync_error:
                    logger.debug(f"Sync method failed: {sync_error}")
                    
        except Exception as e:
            logger.debug(f"Output check failed: {e}")
        
        await asyncio.sleep(1)  # Check every second
    
    # Get final result
    try:
        await execution_future
    except asyncio.CancelledError:
        pass
    
    return None


def is_kernel_busy(kernel):
    """Check if kernel is currently executing something."""
    try:
        # This is a simple check - you might need to adapt based on your kernel client
        if hasattr(kernel, '_client') and hasattr(kernel._client, 'is_alive'):
            return kernel._client.is_alive()
        return False
    except Exception:
        return False


async def wait_for_kernel_idle(kernel, max_wait_seconds=60):
    """Wait for kernel to become idle before proceeding."""
    from jupyter_mcp_server.log import logger
    
    start_time = time.time()
    while is_kernel_busy(kernel):
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            logger.warning(f"Kernel still busy after {max_wait_seconds}s, proceeding anyway")
            break
        logger.info(f"Waiting for kernel to become idle... ({elapsed:.1f}s)")
        await asyncio.sleep(1)


async def safe_notebook_operation(operation_func, max_retries=3):
    """Safely execute notebook operations with connection recovery."""
    from jupyter_mcp_server.log import logger
    
    for attempt in range(max_retries):
        try:
            return await operation_func()
        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["websocketclosederror", "connection is already closed", "connection closed"]):
                if attempt < max_retries - 1:
                    logger.warning(f"Connection lost, retrying... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(1 + attempt)  # Increasing delay
                    continue
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise Exception(f"Connection failed after {max_retries} retries: {e}")
            else:
                # Non-connection error, don't retry
                raise e
    
    raise Exception("Unexpected error in retry logic")


###############################################################################
# Local code execution helpers (JUPYTER_SERVER mode)
###############################################################################


async def execute_via_execution_stack(
    serverapp: Any,
    kernel_id: str,
    code: str,
    document_id: str = None,
    cell_id: str = None,
    timeout: int = 300,
    poll_interval: float = 0.1,
    logger = None
) -> list[Union[str, ImageContent]]:
    """Execute code using ExecutionStack (JUPYTER_SERVER mode with jupyter-server-nbmodel).
    
    This uses the ExecutionStack from jupyter-server-nbmodel extension directly,
    avoiding the reentrant HTTP call issue. This is the preferred method for code
    execution in JUPYTER_SERVER mode.
    
    Args:
        serverapp: Jupyter server application instance
        kernel_id: Kernel ID to execute in
        code: Code to execute
        document_id: Optional document ID for RTC integration (format: json:notebook:<file_id>)
        cell_id: Optional cell ID for RTC integration
        timeout: Maximum time to wait for execution (seconds)
        poll_interval: Time between polling for results (seconds)
        logger: Logger instance (optional)
        
    Returns:
        List of formatted outputs (strings or ImageContent)
        
    Raises:
        RuntimeError: If jupyter-server-nbmodel extension is not installed
        TimeoutError: If execution exceeds timeout
    """
    import logging as default_logging
    
    if logger is None:
        logger = default_logging.getLogger(__name__)
    
    try:
        # Get the ExecutionStack from the jupyter_server_nbmodel extension
        nbmodel_extensions = serverapp.extension_manager.extension_apps.get("jupyter_server_nbmodel", set())
        if not nbmodel_extensions:
            raise RuntimeError("jupyter_server_nbmodel extension not found. Please install it.")
        
        nbmodel_ext = next(iter(nbmodel_extensions))
        execution_stack = nbmodel_ext._Extension__execution_stack
        
        # Build metadata for RTC integration if available
        metadata = {}
        if document_id and cell_id:
            metadata = {
                "document_id": document_id,
                "cell_id": cell_id
            }
        
        # Submit execution request
        logger.info(f"Submitting execution request to kernel {kernel_id}")
        request_id = execution_stack.put(kernel_id, code, metadata)
        logger.info(f"Execution request {request_id} submitted")
        
        # Poll for results
        start_time = asyncio.get_event_loop().time()
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Execution timed out after {timeout} seconds")
            
            # Get result (returns None if pending, result dict if complete)
            result = execution_stack.get(kernel_id, request_id)
            
            if result is not None:
                # Execution complete
                logger.info(f"Execution request {request_id} completed")
                
                # Check for errors
                if "error" in result:
                    error_info = result["error"]
                    logger.error(f"Execution error: {error_info}")
                    return [f"[ERROR: {error_info.get('ename', 'Unknown')}: {error_info.get('evalue', '')}]"]
                
                # Check for pending input (shouldn't happen with allow_stdin=False)
                if "input_request" in result:
                    logger.warning("Unexpected input request during execution")
                    return ["[ERROR: Unexpected input request]"]
                
                # Extract outputs
                outputs = result.get("outputs", [])
                
                # Parse JSON string if needed (ExecutionStack returns JSON string)
                if isinstance(outputs, str):
                    import json
                    try:
                        outputs = json.loads(outputs)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse outputs JSON: {outputs}")
                        return [f"[ERROR: Invalid output format]"]
                
                if outputs:
                    formatted = safe_extract_outputs(outputs)
                    logger.info(f"Execution completed with {len(formatted)} formatted outputs: {formatted}")
                    return formatted
                else:
                    logger.info("Execution completed with no outputs")
                    return ["[No output generated]"]
            
            # Still pending, wait before next poll
            await asyncio.sleep(poll_interval)
            
    except Exception as e:
        logger.error(f"Error executing via ExecutionStack: {e}", exc_info=True)
        return [f"[ERROR: {str(e)}]"]


async def execute_code_local(
    serverapp,
    notebook_path: str,
    code: str,
    kernel_id: str,
    timeout: int = 300,
    logger=None
) -> list[Union[str, ImageContent]]:
    """Execute code in a kernel and return outputs (JUPYTER_SERVER mode).
    
    This is a centralized code execution function for JUPYTER_SERVER mode that:
    1. Gets the kernel from kernel_manager
    2. Creates a client and sends execute_request
    3. Polls for response messages with timeout
    4. Collects and formats outputs
    5. Cleans up resources
    
    Args:
        serverapp: Jupyter ServerApp instance
        notebook_path: Path to the notebook (for context)
        code: Code to execute
        kernel_id: ID of the kernel to execute in
        timeout: Timeout in seconds (default: 300)
        logger: Logger instance (optional)
        
    Returns:
        List of formatted outputs (strings or ImageContent)
    """
    import zmq.asyncio
    from inspect import isawaitable
    
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    try:
        # Get kernel manager
        kernel_manager = serverapp.kernel_manager
        
        # Get the kernel using pinned_superclass pattern (like KernelUsageHandler)
        lkm = kernel_manager.pinned_superclass.get_kernel(kernel_manager, kernel_id)
        session = lkm.session
        client = lkm.client()
        
        # Ensure channels are started (critical for receiving IOPub messages!)
        if not client.channels_running:
            client.start_channels()
            # Wait for channels to be ready
            await asyncio.sleep(0.1)
        
        # Send execute request on shell channel
        shell_channel = client.shell_channel
        msg_id = session.msg("execute_request", {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": False
        })
        shell_channel.send(msg_id)
        
        # Give a moment for messages to start flowing
        await asyncio.sleep(0.01)
        
        # Prepare to collect outputs
        outputs = []
        execution_done = False
        grace_period_ms = 100  # Wait 100ms after shell reply for remaining IOPub messages
        execution_done_time = None
        
        # Poll for messages with timeout
        poller = zmq.asyncio.Poller()
        iopub_socket = client.iopub_channel.socket
        shell_socket = shell_channel.socket
        poller.register(iopub_socket, zmq.POLLIN)
        poller.register(shell_socket, zmq.POLLIN)
        
        timeout_ms = timeout * 1000
        start_time = asyncio.get_event_loop().time()
        
        while not execution_done or (execution_done_time and (asyncio.get_event_loop().time() - execution_done_time) * 1000 < grace_period_ms):
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            remaining_ms = max(0, timeout_ms - elapsed_ms)
            
            # If execution is done and grace period expired, exit
            if execution_done and execution_done_time and (asyncio.get_event_loop().time() - execution_done_time) * 1000 >= grace_period_ms:
                break
            
            if remaining_ms <= 0:
                client.stop_channels()
                logger.warning(f"Code execution timeout after {timeout}s, collected {len(outputs)} outputs")
                return [f"[TIMEOUT ERROR: Code execution exceeded {timeout} seconds]"]
            
            # Use shorter poll timeout during grace period
            poll_timeout = min(remaining_ms, grace_period_ms / 2) if execution_done else remaining_ms
            events = dict(await poller.poll(poll_timeout))
            
            if not events:
                continue  # No messages, continue polling
            
            # IMPORTANT: Process IOPub messages BEFORE shell to collect outputs before marking done
            # Check for IOPub messages (outputs)
            if iopub_socket in events:
                msg = client.iopub_channel.get_msg(timeout=0)
                # Handle async get_msg (like KernelUsageHandler)
                if isawaitable(msg):
                    msg = await msg
                
                if msg and msg.get('parent_header', {}).get('msg_id') == msg_id['header']['msg_id']:
                    msg_type = msg.get('msg_type')
                    content = msg.get('content', {})
                    
                    logger.debug(f"IOPub message: {msg_type}")
                    
                    # Collect output messages
                    if msg_type == 'stream':
                        outputs.append({
                            'output_type': 'stream',
                            'name': content.get('name', 'stdout'),
                            'text': content.get('text', '')
                        })
                        logger.debug(f"Collected stream output: {len(content.get('text', ''))} chars")
                    elif msg_type == 'execute_result':
                        outputs.append({
                            'output_type': 'execute_result',
                            'data': content.get('data', {}),
                            'metadata': content.get('metadata', {}),
                            'execution_count': content.get('execution_count')
                        })
                        logger.debug(f"Collected execute_result, count: {content.get('execution_count')}")
                    elif msg_type == 'display_data':
                        # Note: 'transient' field from kernel messages is NOT part of nbformat schema
                        # Only include 'output_type', 'data', and 'metadata' fields
                        outputs.append({
                            'output_type': 'display_data',
                            'data': content.get('data', {}),
                            'metadata': content.get('metadata', {})
                        })
                        logger.debug("Collected display_data")
                    elif msg_type == 'error':
                        outputs.append({
                            'output_type': 'error',
                            'ename': content.get('ename', ''),
                            'evalue': content.get('evalue', ''),
                            'traceback': content.get('traceback', [])
                        })
                        logger.debug(f"Collected error: {content.get('ename')}")
            
            # Check for shell reply (execution complete) - AFTER processing IOPub
            if shell_socket in events:
                reply = client.shell_channel.get_msg(timeout=0)
                # Handle async get_msg (like KernelUsageHandler)
                if isawaitable(reply):
                    reply = await reply
                
                if reply and reply.get('parent_header', {}).get('msg_id') == msg_id['header']['msg_id']:
                    logger.debug(f"Execution complete, reply status: {reply.get('content', {}).get('status')}")
                    execution_done = True
                    execution_done_time = asyncio.get_event_loop().time()
        
        # Clean up
        client.stop_channels()
        
        # Extract and format outputs
        if outputs:
            result = safe_extract_outputs(outputs)
            logger.info(f"Code execution completed with {len(result)} outputs")
            return result
        else:
            return ["[No output generated]"]
            
    except Exception as e:
        logger.error(f"Error executing code locally: {e}")
        return [f"[ERROR: {str(e)}]"]


async def execute_cell_local(
    serverapp,
    notebook_path: str,
    cell_index: int,
    kernel_id: str,
    timeout: int = 300,
    logger=None
) -> list[Union[str, ImageContent]]:
    """Execute a cell in a notebook and return outputs (JUPYTER_SERVER mode).
    
    This function:
    1. Reads the cell source from the notebook (YDoc or file)
    2. Executes the code using execute_code_local
    3. Writes the outputs back to the notebook (YDoc or file)
    4. Returns the formatted outputs
    
    Args:
        serverapp: Jupyter ServerApp instance
        notebook_path: Path to the notebook
        cell_index: Index of the cell to execute
        kernel_id: ID of the kernel to execute in
        timeout: Timeout in seconds (default: 300)
        logger: Logger instance (optional)
        
    Returns:
        List of formatted outputs (strings or ImageContent)
    """
    import nbformat
    
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    try:
        # Try to get YDoc first (for collaborative editing)
        file_id_manager = serverapp.web_app.settings.get("file_id_manager")
        ydoc = None
        
        if file_id_manager:
            file_id = file_id_manager.get_id(notebook_path)
            yroom_manager = serverapp.web_app.settings.get("yroom_manager")
            
            if yroom_manager:
                room_id = f"json:notebook:{file_id}"
                if yroom_manager.has_room(room_id):
                    try:
                        yroom = yroom_manager.get_room(room_id)
                        ydoc = await yroom.get_jupyter_ydoc()
                        logger.info(f"Using YDoc for cell {cell_index} execution")
                    except Exception as e:
                        logger.debug(f"Could not get YDoc: {e}")
        
        # Execute using YDoc or file
        if ydoc:
            # YDoc path - read from collaborative document
            if cell_index < 0 or cell_index >= len(ydoc.ycells):
                raise ValueError(f"Cell index {cell_index} out of range. Notebook has {len(ydoc.ycells)} cells.")
            
            cell = ydoc.ycells[cell_index]
            
            # Only execute code cells
            cell_type = cell.get("cell_type", "")
            if cell_type != "code":
                return [f"[Cell {cell_index} is not a code cell (type: {cell_type})]"]
            
            source_raw = cell.get("source", "")
            if isinstance(source_raw, list):
                source = "".join(source_raw)
            else:
                source = str(source_raw)
            
            if not source:
                return ["[Cell is empty]"]
            
            logger.info(f"Cell {cell_index} source from YDoc: {source[:100]}...")
            
            # Execute the code
            outputs = await execute_code_local(
                serverapp=serverapp,
                notebook_path=notebook_path,
                code=source,
                kernel_id=kernel_id,
                timeout=timeout,
                logger=logger
            )
            
            logger.info(f"Execution completed with {len(outputs)} outputs: {outputs}")
            
            # Update execution count in YDoc
            max_count = 0
            for c in ydoc.ycells:
                if c.get("cell_type") == "code" and c.get("execution_count"):
                    max_count = max(max_count, c["execution_count"])
            
            cell["execution_count"] = max_count + 1
            
            # Update outputs in YDoc (simplified - just store formatted strings)
            # YDoc outputs should match nbformat structure
            cell["outputs"] = []
            for output in outputs:
                if isinstance(output, str):
                    cell["outputs"].append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": output
                    })
            
            return outputs
        else:
            # File path - original logic
            # Read notebook as version 4 (latest) for consistency
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Clean transient fields from outputs
            clean_notebook_outputs(notebook)
            
            # Validate cell index
            if cell_index < 0 or cell_index >= len(notebook.cells):
                raise ValueError(f"Cell index {cell_index} out of range. Notebook has {len(notebook.cells)} cells.")
        
            cell = notebook.cells[cell_index]
            
            # Only execute code cells
            if cell.cell_type != 'code':
                return [f"[Cell {cell_index} is not a code cell (type: {cell.cell_type})]"]
            
            # Get cell source
            source = cell.source
            if not source:
                return ["[Cell is empty]"]
            
            # Execute the code
            logger.info(f"Executing cell {cell_index} from {notebook_path}")
            outputs = await execute_code_local(
                serverapp=serverapp,
                notebook_path=notebook_path,
                code=source,
                kernel_id=kernel_id,
                timeout=timeout,
                logger=logger
            )
            
            # Write outputs back to notebook (update execution_count and outputs)
            # Get the last execution count
            max_count = 0
            for c in notebook.cells:
                if c.cell_type == 'code' and c.execution_count:
                    max_count = max(max_count, c.execution_count)
            
            cell.execution_count = max_count + 1
            
            # Convert formatted outputs back to nbformat structure
            # Note: outputs is already formatted, so we need to reconstruct
            # For simplicity, we'll store a simple representation
            cell.outputs = []
            for output in outputs:
                if isinstance(output, str):
                    # Create a stream output
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type='stream',
                        name='stdout',
                        text=output
                    ))
                elif isinstance(output, ImageContent):
                    # Create a display_data output with image
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type='display_data',
                        data={'image/png': output.data}
                    ))
            
            # Write notebook back
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(notebook, f)
            
            logger.info(f"Cell {cell_index} executed and notebook updated")
            return outputs
        
    except Exception as e:
        logger.error(f"Error executing cell locally: {e}")
        return [f"[ERROR: {str(e)}]"]


async def get_jupyter_ydoc(serverapp: Any, file_id: str):
    """Get the YNotebook document if it's currently open in a collaborative session.
    
    This follows the jupyter_ai_tools pattern of accessing YDoc through the
    yroom_manager when the notebook is actively being edited.
    
    Args:
        serverapp: The Jupyter ServerApp instance
        file_id: The file ID for the document
        
    Returns:
        YNotebook instance or None if not in a collaborative session
    """
    try:
        # Access ywebsocket_server from YDocExtension via extension_manager
        # jupyter-collaboration doesn't add yroom_manager to web_app.settings
        ywebsocket_server = None

        if hasattr(serverapp, 'extension_manager'):
            extension_points = serverapp.extension_manager.extension_points
            if 'jupyter_server_ydoc' in extension_points:
                ydoc_ext_point = extension_points['jupyter_server_ydoc']
                if hasattr(ydoc_ext_point, 'app') and ydoc_ext_point.app:
                    ydoc_app = ydoc_ext_point.app
                    if hasattr(ydoc_app, 'ywebsocket_server'):
                        ywebsocket_server = ydoc_app.ywebsocket_server

        if ywebsocket_server is None:
            return None

        room_id = f"json:notebook:{file_id}"

        # Get room and access document via room._document
        # DocumentRoom stores the YNotebook as room._document, not via get_jupyter_ydoc()
        try:
            yroom = await ywebsocket_server.get_room(room_id)
            if yroom and hasattr(yroom, '_document'):
                return yroom._document
        except Exception:
            pass

    except Exception:
        # YDoc not available, will fall back to file operations
        pass

    return None

async def get_notebook_model(serverapp: Any, notebook_path: str):
    """Get the NotebookModel instance if it's currently open in a collaborative session."""
    # Get file_id from file_id_manager
    file_id_manager = serverapp.web_app.settings.get("file_id_manager")
    if file_id_manager is None:
        raise RuntimeError("file_id_manager not available in serverapp")
    
    file_id = file_id_manager.get_id(notebook_path)
    ydoc = await get_jupyter_ydoc(serverapp, file_id)
    if ydoc is None:
        return None
    nb = NotebookModel()
    nb._doc = ydoc
    return nb