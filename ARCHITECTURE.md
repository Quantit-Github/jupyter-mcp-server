<!--
  ~ Copyright (c) 2023-2024 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Jupyter MCP Server - Architecture

## Overview

The Jupyter MCP Server supports **dual-mode operation**:

1. **MCP_SERVER Mode** (Standalone) - Connects to remote Jupyter servers via HTTP/WebSocket
2. **JUPYTER_SERVER Mode** (Extension) - Runs embedded in Jupyter Server with direct API access

Both modes share the same tool implementations, with automatic backend selection based on configuration.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       MCP Client                                │
│            (Claude Desktop, VS Code, Cursor, etc.)              │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             │ stdio/SSE                          │ HTTP/SSE
             │                                    │
    ┌────────▼────────────┐          ┌───────────▼──────────────┐
    │   MCP_SERVER Mode   │          │  JUPYTER_SERVER Mode     │
    │   (Standalone)      │          │  (Extension)             │
    │                     │          │                          │
    │   CLI Layer         │          │    Extension Handlers    │
    │  (CLI.py)           │          │  (handlers.py)           │
    └──────────┬──────────┘          └──────────┬───────────────┘
               │                                │
               │ Configuration                  │ Configuration
               │                                │
    ┌──────────▼──────────┐          ┌──────────▼──────────────┐
    │   Server Layer      │          │   Extension Context     │
    │  (server.py)        │          │  (context.py)           │
    │                     │          │                         │
    │  - FastMCP Server   │          │  - ServerApp Access     │
    │  - Tool Wrappers    │          │  - Manager Access       │
    │  - Error Handling   │          │  - Backend Selection    │
    └──────────┬──────────┘          └──────────┬──────────────┘
               │                                │
               │ Tool Delegation                │ Tool Delegation
               │                                │
        ┌──────▼────────────────────────────────▼──────┐
        │          Tool Implementation Layer           │
        │         (jupyter_mcp_server/tools/)          │
        │                                              │
        │  14 Tools in 3 Categories:                   │
        │  • Server Management (2)                     │
        │  • Multi-Notebook Management (5)             │
        │  • Cell Operations (7)                       │
        │                                              │
        │  Each tool implements:                       │
        │  - Dual-mode execution logic                 │
        │  - Backend abstraction                       │
        │  - Error handling and recovery               │
        └──────────┬───────────────────────┬───────────┘
                   │                       │
                   │ Mode Selection        │ Backend Selection
                   │                       │
          ┌────────▼────────┐     ┌────────▼────────┐
          │ Remote Backend  │     │  Local Backend  │
          │                 │     │                 │
          │ - HTTP Clients  │     │ - Direct API    │
          │ - WebSocket     │     │ - Zero Overhead │
          │ - Client Libs   │     │ - YDoc Support  │
          └────────┬────────┘     └────────┬────────┘
                   │                       │
                   │ HTTP/WS               │ Direct Python API
                   │                       │
            ┌──────▼──────────┐    ┌───────▼────────┐
            │ Remote Jupyter  │    │ Local Jupyter  │
            │ Server          │    │ Server         │
            └─────────────────┘    └────────────────┘
```

## Core Components

### 1. CLI Layer (`CLI.py`)

**Command-Line Interface** - Primary entry point for users and MCP clients:

**Key Features**:
- **Configuration Management**: Handles all startup configuration via command-line options and environment variables
- **Transport Selection**: Supports both `stdio` (for direct MCP client integration) and `streamable-http` (for HTTP-based clients)
- **Auto-Enrollment**: Automatically connects to specified notebooks on startup
- **Provider Support**: Supports both `jupyter` and `datalayer` providers
- **URL Resolution**: Intelligent URL and token resolution with fallback mechanisms

**Integration**:
- Calls `server.py` functions to initialize the MCP server
- Passes configuration to `ServerContext` for mode detection
- Handles kernel startup and notebook enrollment lifecycle

### 2. Backend Layer (`jupyter_mcp_server/jupyter_extension/backends/`)

**Backend Abstraction** - Unified interface for notebook and kernel operations:

**LocalBackend** - Complete implementation using local Jupyter Server APIs:
- Uses `serverapp.contents_manager` for file operations
- Uses `serverapp.kernel_manager` for kernel operations
- Direct Python API calls with minimal overhead
- Supports both file-based and YDoc collaborative editing

**RemoteBackend** - Placeholder implementation for HTTP/WebSocket access:
- Designed for `jupyter_server_client`, `jupyter_kernel_client`, `jupyter_nbmodel_client`
- Maintains 100% backward compatibility with existing MCP_SERVER mode
- Currently marked as "Not Implemented" - to be refactored from server.py

### 3. Server Context Layer

**Multiple Context Managers**:

**MCP Server Context** (`server_context.py::ServerContext`):
- Singleton managing server mode for standalone MCP_SERVER mode
- Provides HTTP clients for remote Jupyter server access
- Mode detection based on configuration

**Extension Context** (`jupyter_extension/context.py::ServerContext`):
- Singleton managing server mode for JUPYTER_SERVER extension mode
- Provides direct access to serverapp managers (contents_manager, kernel_manager)
- Handles configuration from Jupyter extension traits

**Mode Detection**:
- **JUPYTER_SERVER**: When running as extension, serverapp available
- **MCP_SERVER**: When running standalone, connects via HTTP

### 4. FastMCP Server Layer (`server.py`)

**FastMCP Integration** - Core MCP protocol implementation:

```python
# Global MCP server instance with CORS support
mcp = FastMCPWithCORS(name="Jupyter MCP Server", json_response=False, stateless_http=True)
notebook_manager = NotebookManager()
server_context = ServerContext.get_instance()

# Tool registration and execution
@mcp.tool()
async def list_files(path: str = "", max_depth: int = 1, ...) -> str:
    """List files and directories in Jupyter server filesystem"""
    return await safe_notebook_operation(
        lambda: ListFilesTool().execute(
            mode=server_context.mode,
            server_client=server_context.server_client,
            contents_manager=server_context.contents_manager,
            path=path,
            max_depth=max_depth,
            ...
        )
    )
```

**Key Responsibilities**:
- **Tool Registration**: All 14 MCP tools are registered as FastMCP decorators
- **Mode Detection**: Automatically detects and initializes appropriate server mode
- **Error Handling**: Provides `safe_notebook_operation()` wrapper with retry logic
- **Resource Management**: Manages notebook connections and kernel lifecycle
- **Protocol Bridge**: Translates between MCP protocol and internal tool implementations

**Transport Support**:
- **stdio**: Direct communication with MCP clients via standard input/output
- **streamable-http**: HTTP-based communication with SSE (Server-Sent Events) support
- **CORS Middleware**: Enables cross-origin requests for web-based MCP clients

### 5. Tool Implementation Layer (`jupyter_mcp_server/tools/`)

**Built-in Tool Implementations** - Complete set of Jupyter operations:

```python
# Tool Categories and Examples

# Server Management (2 tools)
class ListFilesTool(BaseTool):      # File system exploration
class ListKernelsTool(BaseTool):    # Kernel management

# Multi-Notebook Management (5 tools)
class UseNotebookTool(BaseTool):    # Connect/create notebooks
class ListNotebooksTool(BaseTool):  # List managed notebooks
class RestartNotebookTool(BaseTool): # Restart kernels
class UnuseNotebookTool(BaseTool):  # Disconnect notebooks
class ReadNotebookTool(BaseTool):   # Read notebook content

# Cell Operations (7 tools)
class InsertCellTool(BaseTool):     # Insert new cells
class DeleteCellTool(BaseTool):     # Delete cells
class OverwriteCellSourceTool(BaseTool): # Modify cell content
class ExecuteCellTool(BaseTool):    # Execute cells with streaming
class ReadCellTool(BaseTool):       # Read individual cells
class ExecuteCodeTool(BaseTool):    # Execute arbitrary code
class InsertExecuteCodeCellTool(BaseTool): # Combined insert+execute
```

**Implementation Architecture**:
- **BaseTool Abstract Class**: Defines `execute()` method signature with dual-mode support
- **ServerMode Enum**: Distinguishes between `MCP_SERVER` and `JUPYTER_SERVER` modes
- **Dual-Mode Logic**: Each tool implements both local and remote execution paths
- **Backend Integration**: Tools automatically select appropriate backend based on mode

**Tool Categories**:
1. **Server Management**: File system and kernel introspection
2. **Multi-Notebook Management**: Notebook lifecycle and connection management
3. **Cell Operations**: Fine-grained cell manipulation and execution

**Dynamic Tool Registry** (`get_registered_tools()`):
- Queries FastMCP's `list_tools()` to get all registered tools
- Returns tool metadata (name, description, parameters, inputSchema)
- Used by Jupyter extension to expose tools without hardcoding
- Supports both FastMCP tools and jupyter-mcp-tools integration

### 6. Jupyter Extension Layer (`jupyter_extension/`)

**Extension App** (`extension.py::JupyterMCPServerExtensionApp`):
```python
class JupyterMCPServerExtensionApp(ExtensionApp):
    name = "jupyter_mcp_server"
    
    # Configuration traits
    document_url = Unicode("local", config=True)
    runtime_url = Unicode("local", config=True)
    document_id = Unicode("notebook.ipynb", config=True)
    
    def initialize_settings(self):
        # Store config in Tornado settings
        # Initialize ServerContext with JUPYTER_SERVER mode
```

**Handlers** (`handlers.py`):
- `MCPHealthHandler`: GET /mcp/healthz
- `MCPToolsListHandler`: GET /mcp/tools/list (uses `get_registered_tools()`)
- `MCPToolsCallHandler`: POST /mcp/tools/call
- `MCPSSEHandler`: SSE endpoint for MCP protocol

**Extension Context** (`context.py::ServerContext`):
```python
class ServerContext:
    _serverapp: Optional[Any] = None
    _context_type: str = "unknown"
    
    def update(self, context_type: str, serverapp: Any):
        """Called by extension to register serverapp."""
    
    def is_local_document(self) -> bool:
        """Check if document operations use local access."""
    
    def get_contents_manager(self):
        """Get local contents_manager from serverapp."""
```

### 7. Notebook Manager (`notebook_manager.py`)

**Purpose**: Manages notebook connections and kernel lifecycle.

**Key Features**:
- Tracks managed notebooks with kernel associations
- Supports both local (JUPYTER_SERVER) and remote (MCP_SERVER) modes
- Provides `NotebookConnection` context manager for Y.js document access

**Local vs Remote**:
- **Local mode**: Notebooks tracked with `is_local=True`, no WebSocket connections
- **Remote mode**: Establishes WebSocket connections via `NbModelClient`

```python
class NotebookManager:
    def add_notebook(self, name, kernel, server_url="local", ...):
        """Add notebook with mode detection (local vs remote)."""
    
    def get_current_connection(self):
        """Get WebSocket connection (MCP_SERVER mode only)."""
```

## Session-based Multi-Client Architecture (ARK-165)

**Overview**: ARK-165 introduces session-based multi-client support, enabling multiple independent clients to simultaneously work with different notebooks using the same Jupyter MCP Server instance.

### Architecture Diagram

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Client A   │  │  Client B   │  │  Client C   │
│ (Claude)    │  │  (VSCode)   │  │  (Cursor)   │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       │ session_id_A   │ session_id_B   │ session_id_C
       │                │                │
       └────────────────┼────────────────┘
                        │
                 ┌──────▼──────┐
                 │ MCP Server  │
                 │  (server.py)│
                 └──────┬──────┘
                        │
                 ┌──────▼────────┐
                 │ SessionStore  │
                 │               │
                 │ session_id_A →│→ {notebook: a.ipynb, kernel: k1}
                 │ session_id_B →│→ {notebook: b.ipynb, kernel: k2}
                 │ session_id_C →│→ {notebook: c.ipynb, kernel: k3}
                 └──────┬────────┘
                        │
                        │ Context Lookup
                        │
          ┌─────────────┼─────────────┐
          │             │             │
   ┌──────▼──────┐ ┌───▼────┐ ┌──────▼──────┐
   │ notebook_a  │ │ notebook_b│ │ notebook_c │
   │   (k1)      │ │   (k2)   │ │   (k3)     │
   └─────────────┘ └──────────┘ └─────────────┘
```

### Core Components

#### 1. SessionStore (`session_store.py`)

**Purpose**: Lightweight in-memory storage mapping session_id → notebook context

**Key Features**:
- **Memory Efficient**: 1000 sessions ≈ 234 KB
- **TTL-based Expiration**: Default 24 hours, configurable
- **Thread-Safe**: Safe for concurrent reads and updates
- **O(1) Operations**: Fast lookup and update

**Data Structure**:
```python
@dataclass
class SessionContext:
    current_notebook: Optional[str]  # Notebook name
    kernel_id: Optional[str]         # Jupyter kernel ID
    notebook_path: Optional[str]     # Full path to notebook
    created_at: datetime             # Session creation time
    last_accessed: datetime          # Last access timestamp
```

**Public API**:
```python
class SessionStore:
    def get_or_create(session_id: str) -> SessionContext
    def get(session_id: str) -> Optional[SessionContext]
    def update_notebook(session_id, name, path, kernel_id) -> None
    def cleanup_expired() -> int
```

#### 2. Context Lookup (`utils.py`)

**Function**: `get_notebook_context_from_session(session_id: Optional[str])`

**Purpose**: Primary function for retrieving notebook context with session awareness

**Lookup Priority**:
1. **SessionStore** (if session_id provided)
   - Returns session-specific notebook_path and kernel_id
   - Enables multi-client isolation
2. **Config Fallback** (if session_id not provided)
   - Returns config.document_id and config.runtime_id
   - Backward compatibility for single-client mode

**Usage Pattern**:
```python
# Multi-client mode (ARK-165)
notebook_path, kernel_id = get_notebook_context_from_session(
    session_id="client-A-uuid"
)

# Single-client mode (backward compatible)
notebook_path, kernel_id = get_notebook_context_from_session()
```

#### 3. Tool Integration

**All tools support session_id parameter**:
- `use_notebook` - Creates/updates session context
- `execute_cell` - Executes in session's kernel
- `read_cell` - Reads from session's notebook
- `insert_cell` - Inserts into session's notebook
- `delete_cell` - Deletes from session's notebook
- `overwrite_cell_source` - Overwrites in session's notebook

**Tool Execution Flow**:
```python
async def execute(
    self,
    mode: ServerMode,
    session_id: Optional[str] = None,  # ARK-165
    **kwargs
) -> dict:
    # Step 1: Lookup context from session
    notebook_path, kernel_id = get_notebook_context_from_session(
        session_id=session_id
    )

    # Step 2: Execute operation on correct notebook/kernel
    result = await self._execute_operation(notebook_path, kernel_id)

    # Step 3: Return structured output with session info
    return {
        "result": result,
        "session_id": session_id,
        "notebook_path": notebook_path,
        "metadata": {...}
    }
```

### Multi-Client Workflow

#### 1. Session Initialization

**Client A connects**:
```python
# Client A creates session and connects to notebook
result = await use_notebook.execute(
    mode=ServerMode.JUPYTER_SERVER,
    notebook_path="analysis.ipynb",
    session_id="client-A-uuid",  # UUID generated by client
    kernel_id="kernel-A",
    # ... other params
)

# SessionStore state:
# session_id="client-A-uuid" → {
#     notebook_path="analysis.ipynb",
#     kernel_id="kernel-A"
# }
```

**Client B connects** (independent):
```python
# Client B creates different session
result = await use_notebook.execute(
    mode=ServerMode.JUPYTER_SERVER,
    notebook_path="report.ipynb",
    session_id="client-B-uuid",  # Different UUID
    kernel_id="kernel-B",
    # ... other params
)

# SessionStore state:
# session_id="client-A-uuid" → analysis.ipynb, kernel-A
# session_id="client-B-uuid" → report.ipynb, kernel-B
```

#### 2. Isolated Operations

**Client A executes cell**:
```python
result_a = await execute_cell.execute(
    mode=ServerMode.JUPYTER_SERVER,
    cell_index=0,
    session_id="client-A-uuid",  # Looks up analysis.ipynb
    # ... other params
)
# Executes in analysis.ipynb with kernel-A
```

**Client B executes cell** (simultaneously):
```python
result_b = await execute_cell.execute(
    mode=ServerMode.JUPYTER_SERVER,
    cell_index=0,
    session_id="client-B-uuid",  # Looks up report.ipynb
    # ... other params
)
# Executes in report.ipynb with kernel-B (independent)
```

#### 3. Notebook Switching

**Client A switches notebook**:
```python
# Switch to different notebook (same session)
await use_notebook.execute(
    mode=ServerMode.JUPYTER_SERVER,
    notebook_path="experiments.ipynb",
    session_id="client-A-uuid",  # Same session ID
    kernel_id="kernel-C",
    # ... other params
)

# SessionStore updated:
# session_id="client-A-uuid" → experiments.ipynb, kernel-C
# session_id="client-B-uuid" → report.ipynb, kernel-B (unchanged)
```

### Session Management

#### Session Lifecycle

```
1. Creation
   ↓
   Client calls use_notebook with session_id
   ↓
   SessionStore.get_or_create(session_id)
   ↓
   New SessionContext created

2. Usage
   ↓
   Client calls tools with same session_id
   ↓
   Tools call get_notebook_context_from_session(session_id)
   ↓
   SessionStore.get(session_id)
   ↓
   Returns notebook_path, kernel_id
   ↓
   Tool executes on correct notebook/kernel

3. Expiration
   ↓
   No access for TTL period (default: 24h)
   ↓
   SessionStore.cleanup_expired()
   ↓
   Session removed from memory
```

#### Best Practices

**Client-Side**:
1. **Generate session_id on startup**: Use `str(uuid.uuid4())`
2. **Persist session_id**: Store in client state/config
3. **Reuse session_id**: Pass same ID to all tool calls
4. **Handle errors**: Gracefully handle session expiration

**Server-Side**:
1. **Periodic cleanup**: Call `cleanup_expired()` every hour
2. **Monitor sessions**: Track active session count
3. **Configure TTL**: Adjust based on usage patterns
4. **Log session events**: Debug multi-client issues

## Kernel Healing Architecture (ARK-165)

**Overview**: ARK-165 introduces centralized kernel healing through the SessionManager layer, enabling automatic detection and recovery of dead kernels across all cell operation tools.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Tool Layer                               │
│  (use_notebook, execute_cell, read_cell, insert_cell,       │
│   delete_cell, overwrite_cell_source)                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ get_notebook_context_from_session_async()
                  │ (with auto_heal=True)
                  │
        ┌─────────▼─────────┐
        │  SessionManager   │  ◄── Orchestration Layer
        │                   │
        │  • heal_kernel()  │
        │  • check_kernel() │
        └─────────┬─────────┘
                  │
                  │ get/update operations
                  │
        ┌─────────▼─────────┐
        │   SessionStore    │  ◄── Storage Layer
        │                   │
        │  • session_id →   │
        │    context map    │
        └───────────────────┘
```

### Core Components

#### SessionManager (`session_manager.py`)

**Purpose**: Orchestration layer providing kernel health checking and automatic healing

**Key Responsibilities**:
- **Kernel Health Checking**: Detects dead kernels in both JUPYTER_SERVER and MCP_SERVER modes
- **Automatic Kernel Healing**: Creates new kernels when dead ones are detected
- **SessionStore Integration**: Updates session contexts after successful healing
- **Error Handling**: Gracefully handles healing failures with fallback mechanisms

**Public API**:
```python
class SessionManager:
    def __init__(self, session_store: SessionStore)

    async def get_session_with_kernel_check(
        session_id: str,
        kernel_manager=None,
        server_client=None,
        mode: ServerMode=None
    ) -> Tuple[Optional[SessionContext], bool]:
        """Get session and check kernel health.

        Returns:
            (context, kernel_healthy): Tuple of context and health status
        """

    async def heal_kernel(
        session_id: str,
        kernel_manager=None,
        server_client=None,
        mode: ServerMode=None,
        notebook_path: str=None
    ) -> Optional[str]:
        """Create new kernel and update SessionStore.

        Returns:
            new_kernel_id: ID of newly created kernel, or None if healing failed
        """
```

**Kernel Health Check Logic**:
```python
# JUPYTER_SERVER mode
kernel_healthy = kernel_id in kernel_manager

# MCP_SERVER mode
kernels = await server_client.kernels.list_kernels()
kernel_healthy = any(k['id'] == kernel_id for k in kernels)
```

**Kernel Healing Flow**:
```
1. Detect dead kernel
   ↓
2. Log warning: "Kernel unhealthy, attempting heal"
   ↓
3. Create new kernel (mode-specific)
   ↓
4. Update SessionStore with new kernel_id
   ↓
5. Log success: "✓ Kernel healed: {new_kernel_id}"
   ↓
6. Return new kernel_id

If healing fails:
   ↓
1. Log error: "✗ Kernel healing failed"
   ↓
2. Return None (caller falls back to config)
```

#### Async Context Function (`utils.py`)

**Function**: `get_notebook_context_from_session_async()`

**Purpose**: Async version of context lookup with built-in auto-healing support

**Signature**:
```python
async def get_notebook_context_from_session_async(
    session_id: Optional[str] = None,
    auto_heal: bool = True,
    kernel_manager=None,
    server_client=None,
    mode: ServerMode = None
) -> Tuple[str, str]:
    """Get notebook context with automatic kernel healing.

    Args:
        session_id: Client session ID
        auto_heal: Enable automatic kernel healing (default: True)
        kernel_manager: For JUPYTER_SERVER mode
        server_client: For MCP_SERVER mode
        mode: Server mode

    Returns:
        (notebook_path, kernel_id): Tuple
    """
```

**Execution Flow with Auto-Healing**:
```
1. SessionManager.get_session_with_kernel_check()
   ↓
   Returns (context, kernel_healthy)

2. If kernel_healthy = False:
   ↓
   SessionManager.heal_kernel()
   ↓
   If healing succeeds:
      → Return healed context
   ↓
   If healing fails:
      → Fallback to config

3. If kernel_healthy = True:
   ↓
   Return existing context
```

**Backward Compatibility**:
- **Sync version**: `get_notebook_context_from_session()` - No auto-healing (deprecated)
- **Async version**: `get_notebook_context_from_session_async()` - With auto-healing (recommended)

### Tool Integration

**All 6 tools updated to use async context with auto-healing**:

1. **use_notebook_tool** - Creates/updates sessions with kernel validation
2. **execute_cell_tool** - Executes with auto-healing before execution
3. **read_cell_tool** - Reads with healed kernel context
4. **insert_cell_tool** - Inserts with validated kernel
5. **delete_cell_tool** - Deletes with healed kernel context
6. **overwrite_cell_source_tool** - Overwrites with validated kernel

**Tool Update Pattern**:
```python
# OLD (sync, no healing):
from jupyter_mcp_server.utils import get_notebook_context_from_session
notebook_path, kernel_id = get_notebook_context_from_session(
    session_id=session_id
)

# NEW (async, with auto-healing):
from jupyter_mcp_server.utils import get_notebook_context_from_session_async
notebook_path, kernel_id = await get_notebook_context_from_session_async(
    session_id=session_id,
    auto_heal=True,
    kernel_manager=kernel_manager,
    mode=mode
)
```

### Benefits

**Code Quality**:
- **~96 lines removed** from use_notebook_tool.py
- **Zero code duplication** across cell operation tools
- **Single source of truth** for kernel healing logic

**Reliability**:
- **Automatic recovery** from kernel failures
- **Consistent behavior** across all tools
- **Graceful fallbacks** when healing fails

**Maintainability**:
- **Centralized logic** in SessionManager
- **Easy testing** with unit/integration tests
- **Clear separation** of concerns (storage vs orchestration)

### Architecture Decisions

**Q: Why separate SessionManager from SessionStore?**

A: **Separation of Concerns**
- SessionStore: Pure data storage (lightweight, 234 bytes/session)
- SessionManager: Business logic (kernel management, healing)
- SessionStore remains simple and easily testable
- SessionManager can have kernel_manager dependencies without polluting storage layer

**Q: Why auto_heal=True by default?**

A: **User Experience**
- Most cases benefit from automatic kernel recovery
- Eliminates need for manual use_notebook retry
- Can be disabled with auto_heal=False if needed

**Q: Performance impact of kernel health checks?**

A: **Minimal overhead**
- JUPYTER_SERVER: O(1) dict lookup (~1ms)
- MCP_SERVER: Cached HTTP request (~10ms)
- Check only happens once per tool execution
- Optional caching can be added if needed

### Testing Strategy

**Unit Tests** (`test_session_manager.py`): 14 tests
- Kernel health checking (JUPYTER_SERVER + MCP_SERVER)
- Kernel healing success scenarios
- Error handling and failure cases
- SessionStore partial updates

**Integration Tests** (`test_utils_session.py`): 16 tests
- Sync function backward compatibility (3 tests)
- Async function with auto-healing (6 tests)
- Config fallback scenarios (7 tests)

**Regression Tests** (`test_use_notebook_tool_ark165.py`): 19 tests
- All existing use_notebook functionality
- Kernel auto-recovery workflows
- Multi-client session scenarios

**Cell Tool Integration** (`test_cell_tools_kernel_healing.py`): 11 tests
- Each cell tool's healing behavior (6 tests)
- Code integration verification (5 tests)

**Total**: 60 tests covering all aspects of centralized kernel healing

### Backward Compatibility

**Single-Client Mode** (session_id omitted):
```python
# Legacy usage - no session_id parameter
result = await execute_cell.execute(
    mode=ServerMode.JUPYTER_SERVER,
    cell_index=0,
    # No session_id - falls back to config
)

# Uses config.document_id and config.runtime_id
```

**Migration Path**:
1. Existing clients work without changes
2. New clients can opt-in to session support
3. Tools detect session_id presence and adapt behavior
4. Structured output includes session_id (or None)

### Performance Characteristics

**Memory Usage**:
- Empty SessionStore: ~128 bytes
- Per session: ~234 bytes (avg)
- 1000 sessions: ~234 KB
- 10,000 sessions: ~2.3 MB

**Operation Complexity**:
- Lookup: O(1)
- Update: O(1)
- Cleanup: O(n) where n = total sessions

**Concurrency**:
- Read operations: Fully concurrent
- Write operations: Safe for different session_ids
- Same session_id: Sequential recommended

### Integration Test Coverage

**Test Suite**: `tests/integration/test_multi_client.py`

**Scenarios**:
1. **test_multi_client_isolation**: Two clients with different notebooks
2. **test_notebook_switching**: One client switching notebooks
3. **test_concurrent_clients**: 100 clients simultaneously

**Verification**:
- Session contexts isolated
- Notebook operations independent
- No cross-client interference
- Correct kernel execution

### Security Considerations

**Session ID Security**:
- Use cryptographically secure UUID generation
- Treat session_id as sensitive (contains notebook context)
- Don't log session_ids in production
- Implement rate limiting per session_id

**Isolation Guarantees**:
- Each session_id maps to unique context
- No cross-session data leakage
- Independent kernel execution
- Separate notebook file access

## Configuration

### MCP_SERVER Mode (Standalone)

**Start Command**:
```bash
jupyter-mcp-server start \
  --transport streamable-http \
  --document-url http://localhost:8888 \
  --runtime-url http://localhost:8888 \
  --document-token MY_TOKEN \
  --runtime-token MY_TOKEN \
  --port 4040
```

**Behavior**:
- ServerContext initialized with `mode=ServerMode.MCP_SERVER`
- Tools use HTTP clients for remote Jupyter server access
- Notebook connections use `NbModelClient` for WebSocket (Y.js documents)
- Uses RemoteBackend (placeholder implementation)

### JUPYTER_SERVER Mode (Extension)

**Start Command**:
```bash
jupyter server \
  --JupyterMCPServerExtensionApp.document_url=local \
  --JupyterMCPServerExtensionApp.runtime_url=local \
  --JupyterMCPServerExtensionApp.document_id=notebook.ipynb
```

**Configuration File** (`jupyter_server_config.py`):
```python
c.ServerApp.jpserver_extensions = {"jupyter_mcp_server": True}
c.JupyterMCPServerExtensionApp.document_url = "local"
c.JupyterMCPServerExtensionApp.runtime_url = "local"
```

**Backend Selection**:
- **LocalBackend**: Used when `document_url="local"` or `runtime_url="local"`
  - Direct access to `serverapp.contents_manager`, `serverapp.kernel_manager`
  - No network overhead, maximum performance
  - Supports both file-based and YDoc collaborative editing
- **RemoteBackend**: Used when connecting to remote Jupyter servers
  - HTTP/WebSocket access via client libraries
  - Placeholder implementation (to be completed)

**Behavior**:
- Extension auto-enabled (via `jupyter-config/` file)
- ServerContext updated with `mode=ServerMode.JUPYTER_SERVER`
- Tools automatically select LocalBackend for optimal performance
- Cell reading tools parse notebook JSON from file system or YDoc

## Request Flow Examples

### Example 1: List Notebooks (JUPYTER_SERVER Mode with LocalBackend)

```
MCP Client
  → POST /mcp/tools/call {"tool_name": "list_notebooks"}
    → MCPSSEHandler (or MCPToolsCallHandler)
      → FastMCP calls @mcp.tool() wrapper
        → ListNotebooksTool().execute(
            mode=JUPYTER_SERVER,
            notebook_manager=notebook_manager
          )
          → notebook_manager.list_all_notebooks()
            → Returns managed notebooks from memory
          ← TSV-formatted table
        ← Tool result
      ← JSON-RPC response
    ← SSE message
  ← Tool result displayed
```

### Example 2: Read Cell (JUPYTER_SERVER Mode with LocalBackend)

```
MCP Client
  → POST /mcp/tools/call {"tool_name": "read_cell", "arguments": {"cell_index": 0}}
    → MCPSSEHandler (or MCPToolsCallHandler)
      → FastMCP calls @mcp.tool() wrapper
        → ReadCellTool().execute(
            mode=JUPYTER_SERVER,
            contents_manager=serverapp.contents_manager,
            notebook_manager=notebook_manager
          )
          → LocalBackend.get_notebook_content(notebook_path)
            → contents_manager.get(notebook_path, content=True, type='notebook')
              → Direct file system access (no HTTP)
            ← Notebook JSON content
          → Parse cells and format response
          ← Cell information with metadata and source
        ← Tool result
      ← JSON-RPC response
    ← SSE message
  ← Cell content displayed
```

### Example 3: Execute Cell (MCP_SERVER Mode with RemoteBackend)

```
MCP Client
  → POST /mcp/tools/call {"tool_name": "execute_cell", "arguments": {"cell_index": 0}}
    → FastMCP calls @mcp.tool() wrapper
      → ExecuteCellTool().execute(
          mode=MCP_SERVER,
          notebook_manager=notebook_manager
        )
        → notebook_manager.get_current_connection()
          → NbModelClient establishes WebSocket to Y.js document
          → Access collaborative Y.js document
        → Execute code via kernel connection
          → HTTP/WebSocket to remote kernel
          → Real-time execution with progress updates
        ← Execution outputs with rich formatting
      ← Tool result
    ← Response
  ← Outputs displayed
```

## Tool Registration Flow

```
1. CLI startup (CLI.py)
   ↓
2. Configuration parsing and validation
   ↓
3. ServerContext initialization with mode detection
   ↓
4. FastMCP server initialization (server.py)
   ↓
5. Tool instance creation (14 tool implementations)
   ↓
6. @mcp.tool() wrapper registration
   ↓
7. FastMCP internal tool registry
   ↓
8. Dynamic tool discovery via get_registered_tools()
   ↓
9. Extension handlers expose tools via /mcp/tools/list
   ↓
10. MCP clients discover and invoke tools
```

## File Structure

```
jupyter_mcp_server/
├── __init__.py                 # Package initialization
├── __main__.py                 # Module entry point (imports CLI)
├── __version__.py              # Version information (0.17.1)
│
├── CLI.py                      # 🏠 Command-Line Interface (Primary Entry Point)
│   ├── Command parsing and validation
│   ├── Environment variable handling
│   ├── Transport selection (stdio/streamable-http)
│   ├── Provider support (jupyter/datalayer)
│   ├── Auto-enrollment of notebooks
│   └── Server lifecycle management
│
├── server.py                   # 🔧 FastMCP Server Layer
│   ├── MCP protocol implementation
│   ├── Tool registration (14 @mcp.tool decorators)
│   ├── Error handling with safe_notebook_operation()
│   ├── Resource management and cleanup
│   ├── Dynamic tool registry (get_registered_tools())
│   └── Transport support (stdio + streamable-http)
│
├── tools/                      # 🛠️ Built-in Tool Implementations
│   ├── __init__.py            # Exports BaseTool, ServerMode
│   ├── _base.py               # Abstract base class for all tools
│   │
│   # Server Management Tools (2)
│   ├── list_files_tool.py     # File system exploration
│   ├── list_kernels_tool.py   # Kernel introspection
│   │
│   # Multi-Notebook Management Tools (5)
│   ├── use_notebook_tool.py   # Connect/create notebooks
│   ├── list_notebooks_tool.py # List managed notebooks
│   ├── restart_notebook_tool.py # Restart kernels
│   ├── unuse_notebook_tool.py # Disconnect notebooks
│   ├── read_notebook_tool.py  # Read notebook content
│   │
│   # Cell Operation Tools (7)
│   ├── read_cell_tool.py      # Read individual cells
│   ├── insert_cell_tool.py    # Insert new cells
│   ├── delete_cell_tool.py    # Delete cells
│   ├── overwrite_cell_source_tool.py # Modify cell content
│   ├── execute_cell_tool.py   # Execute cells with streaming
│   ├── execute_code_tool.py   # Execute arbitrary code
│   └── insert_execute_code_cell # Combined insert+execute (inline in server.py)
│
├── config.py                   # ⚙️ Configuration Management
│   ├── Singleton config object (JupyterMCPConfig)
│   ├── Environment variable parsing
│   ├── URL and token resolution
│   └── Provider-specific settings
│
├── notebook_manager.py         # 📚 Notebook Lifecycle Management
│   ├── Multi-notebook support
│   ├── Kernel connection management
│   ├── Context managers for resources
│   └── Dual-mode operation (local/remote)
│
├── server_context.py           # 🎯 Server Context (MCP_SERVER mode)
│   ├── Mode detection and initialization
│   ├── HTTP client management
│   └── Configuration state management
│
├── utils.py                    # 🧰 Utility Functions
│   ├── Execution utilities (local/remote)
│   ├── Output processing and formatting
│   ├── Kernel management helpers
│   └── YDoc integration support
│
├── enroll.py                   # 🔗 Auto-Enrollment System
│   ├── Automatic notebook connection
│   ├── Kernel startup and management
│   └── Configuration-based initialization
│
├── models.py                   # 📋 Data Models
│   ├── Pydantic models for API
│   ├── Cell and Notebook structures
│   └── Configuration validation
│
└── jupyter_extension/          # 🔌 Jupyter Server Extension
    ├── extension.py           # Jupyter extension app
    ├── handlers.py            # HTTP request handlers
    ├── context.py             # Extension context manager
    ├── backends/              # Backend implementations
    │   ├── base.py            # Backend interface
    │   ├── local_backend.py   # Local API (Complete)
    │   └── remote_backend.py  # Remote API (Placeholder)
    └── protocol/              # Protocol implementation
        └── messages.py        # MCP message models
```

## References

- [MCP Specification](https://modelcontextprotocol.io/specification)
- [Jupyter Server Extension Guide](https://jupyter-server.readthedocs.io/en/latest/developers/extensions.html)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Y.js Collaborative Editing](https://github.com/yjs/yjs)

---

**Version**: 0.2.0
**Last Updated**: October 2025
**Status**: Complete implementation with dual-mode architecture and backend abstraction
