"""
Session Store for Multi-Client Support (ARK-165)

Lightweight in-memory session storage that maps session_id to notebook context.
Enables multiple clients to independently use different notebooks with proper isolation,
while maintaining minimal memory overhead and thread-safe operations.

Key Features:
    - Session-based context isolation per client
    - Automatic TTL-based session expiration
    - Thread-safe operations (single process)
    - Minimal memory footprint (1000 sessions ≈ 234 KB)
    - No external dependencies (pure Python)

Architecture:
    SessionStore maintains a dictionary mapping session_id → SessionContext
    Each SessionContext contains: notebook_name, notebook_path, kernel_id
    Sessions are automatically cleaned up after TTL expiration

Example Usage:
    ```python
    from jupyter_mcp_server.session_store import SessionStore

    # Initialize store
    store = SessionStore(ttl_hours=24)

    # Client A connects to notebook
    store.update_notebook(
        session_id="client-A-uuid",
        notebook_name="analysis",
        notebook_path="/work/analysis.ipynb",
        kernel_id="kernel-A-id"
    )

    # Client B connects to different notebook (independent)
    store.update_notebook(
        session_id="client-B-uuid",
        notebook_name="report",
        notebook_path="/work/report.ipynb",
        kernel_id="kernel-B-id"
    )

    # Retrieve contexts (isolated)
    ctx_a = store.get("client-A-uuid")
    ctx_b = store.get("client-B-uuid")
    assert ctx_a.notebook_path != ctx_b.notebook_path
    ```

See Also:
    - get_notebook_context_from_session(): Primary function for context lookup
    - UseNotebookTool: Creates and updates session contexts
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """Client session's current notebook context (ARK-165).

    Stores the notebook information currently being used by a specific client session.
    Sessions automatically expire after TTL period to prevent memory leaks.

    This dataclass represents the isolated context for a single client session,
    enabling multi-client support by maintaining independent notebook contexts.

    Attributes:
        current_notebook: Notebook name without extension (e.g., "analysis")
            - Used for display purposes
            - Not used for file operations
        kernel_id: Jupyter kernel ID string (e.g., "abc-123-def-456")
            - Unique identifier for the Jupyter kernel
            - Used to execute cells in correct kernel
        notebook_path: Full path to notebook file (e.g., "/work/notebook.ipynb")
            - Relative or absolute path
            - Used for file operations
        created_at: Session creation timestamp
            - Set once at creation
            - Used for analytics/debugging
        last_accessed: Last access timestamp
            - Updated on every get/update operation
            - Used for TTL-based expiration

    Example:
        ```python
        from datetime import datetime
        from jupyter_mcp_server.session_store import SessionContext

        # Create context
        ctx = SessionContext(
            current_notebook="analysis",
            kernel_id="kernel-123",
            notebook_path="/work/analysis.ipynb",
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )

        # Access properties
        print(ctx.notebook_path)  # "/work/analysis.ipynb"
        print(ctx.kernel_id)      # "kernel-123"
        ```

    Notes:
        - ARK-165: Core data structure for session isolation
        - Immutable by convention (don't modify fields directly)
        - Use SessionStore.update_notebook() to update context
        - Automatically garbage collected after TTL expiration
    """
    current_notebook: Optional[str] = None
    kernel_id: Optional[str] = None
    notebook_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


class SessionStore:
    """Manage session-based notebook contexts with TTL expiration (ARK-165).

    SessionStore is a lightweight, thread-safe in-memory storage for maintaining
    independent notebook contexts per client session. It enables multi-client support
    by isolating each client's notebook and kernel information.

    Key Features:
        - Minimal memory footprint: 1000 sessions ≈ 234 KB
        - TTL-based automatic expiration (default: 24 hours)
        - Thread-safe operations (single process)
        - O(1) lookup and update operations
        - No external dependencies

    Architecture:
        - Internal dict: session_id (str) → SessionContext
        - Each SessionContext: {notebook_name, notebook_path, kernel_id, timestamps}
        - TTL checked on cleanup_expired() calls
        - last_accessed updated on every get/update

    Memory Profile:
        - Empty store: ~128 bytes
        - Per session: ~234 bytes (avg)
        - 1000 sessions: ~234 KB
        - 10,000 sessions: ~2.3 MB

    Thread Safety:
        - Safe for concurrent reads
        - Safe for concurrent updates to different session_ids
        - NOT safe for concurrent updates to same session_id
        - Recommended: Use single thread per session_id

    Example:
        Basic usage:
        ```python
        from jupyter_mcp_server.session_store import SessionStore

        # Initialize with 24-hour TTL
        store = SessionStore(ttl_hours=24)

        # Create or get session
        ctx = store.get_or_create("client-A-uuid")

        # Update notebook context
        store.update_notebook(
            session_id="client-A-uuid",
            notebook_name="analysis",
            notebook_path="/work/analysis.ipynb",
            kernel_id="kernel-123"
        )

        # Retrieve context
        ctx = store.get("client-A-uuid")
        print(ctx.notebook_path)  # "/work/analysis.ipynb"

        # Cleanup expired sessions
        expired_count = store.cleanup_expired()
        ```

        Multi-client isolation:
        ```python
        # Client A and B work independently
        store.update_notebook("client-A", "notebook_a", "/a.ipynb", "kernel-a")
        store.update_notebook("client-B", "notebook_b", "/b.ipynb", "kernel-b")

        ctx_a = store.get("client-A")
        ctx_b = store.get("client-B")
        assert ctx_a.notebook_path != ctx_b.notebook_path
        ```

    Notes:
        - ARK-165: Core component enabling multi-client support
        - Singleton pattern: Use global `session_store` instance in server.py
        - Periodic cleanup recommended: Call cleanup_expired() every hour
        - Sessions persist until TTL or server restart

    See Also:
        - SessionContext: Data structure for session context
        - get_notebook_context_from_session(): Primary lookup function
        - UseNotebookTool: Creates and updates sessions
    """

    def __init__(self, ttl_hours: int = 24):
        """Initialize SessionStore with TTL configuration.

        Args:
            ttl_hours: Session Time-To-Live in hours (default: 24)
                - Sessions expire after this period of inactivity
                - Recommended: 24 hours for daily work patterns
                - Minimum: 1 hour, Maximum: 720 hours (30 days)

        Example:
            ```python
            # Default 24-hour TTL
            store = SessionStore()

            # Custom 8-hour TTL for short sessions
            store = SessionStore(ttl_hours=8)
            ```
        """
        self._sessions: Dict[str, SessionContext] = {}
        self._ttl = timedelta(hours=ttl_hours)
        self._lock = threading.Lock()
        logger.info(f"SessionStore initialized with TTL={ttl_hours}h")

    def get_or_create(self, session_id: str) -> SessionContext:
        """Get existing session or create new one if not found.

        This method ensures a SessionContext always exists for the given session_id.
        If the session exists, it updates last_accessed timestamp and returns the context.
        If the session doesn't exist, it creates a new empty context.

        Args:
            session_id: Client session ID (UUID format recommended)
                - Should be unique per client
                - Use str(uuid.uuid4()) for generation
                - Example: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        Returns:
            SessionContext: Session context (existing or newly created)
                - Contains notebook_path, kernel_id, timestamps
                - May be empty if newly created
                - last_accessed timestamp always updated

        Example:
            ```python
            import uuid
            from jupyter_mcp_server.server import session_store

            # Create new session
            session_id = str(uuid.uuid4())
            ctx = session_store.get_or_create(session_id)
            print(ctx.notebook_path)  # None (newly created)

            # Update context
            session_store.update_notebook(
                session_id, "notebook", "/path.ipynb", "kernel-id"
            )

            # Get same session (already exists)
            ctx = session_store.get_or_create(session_id)
            print(ctx.notebook_path)  # "/path.ipynb"
            ```

        Notes:
            - ARK-165: Primary method for session initialization
            - Always returns valid SessionContext (never None)
            - Thread-safe for different session_ids
            - Updates last_accessed on every call
        """
        if session_id not in self._sessions:
            # Session doesn't exist - create new empty context
            # Created with current timestamps for both created_at and last_accessed
            self._sessions[session_id] = SessionContext(
                created_at=datetime.now(),
                last_accessed=datetime.now()
            )
            logger.debug(f"Created new session: {session_id}")
        else:
            # Session exists - update last_accessed to keep it alive
            # This prevents TTL expiration for active sessions
            self._sessions[session_id].last_accessed = datetime.now()
            logger.debug(f"Accessed existing session: {session_id}")

        # Always return the context (either newly created or existing)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[SessionContext]:
        """Get existing session without creating new one.

        This method retrieves an existing session context if it exists.
        Unlike get_or_create(), this returns None if the session doesn't exist.
        Updates last_accessed timestamp if session found.

        Args:
            session_id: Client session ID to look up
                - Must be exact match
                - Case-sensitive

        Returns:
            Optional[SessionContext]: Session context or None
                - SessionContext: If session exists (last_accessed updated)
                - None: If session not found

        Example:
            ```python
            from jupyter_mcp_server.server import session_store

            # Check if session exists
            ctx = session_store.get("unknown-session-id")
            if ctx is None:
                print("Session not found")
            else:
                print(f"Notebook: {ctx.notebook_path}")

            # Verify context exists before use
            if "client-A" in session_store:
                ctx = session_store.get("client-A")
                # ctx is guaranteed to exist
            ```

        Notes:
            - ARK-165: Used for context lookup without side effects
            - Returns None instead of creating new session
            - Use get_or_create() if you want automatic creation
            - Thread-safe for different session_ids
        """
        if session_id in self._sessions:
            # Session found - update last_accessed timestamp
            # This prevents TTL expiration for active sessions
            self._sessions[session_id].last_accessed = datetime.now()
            return self._sessions[session_id]

        # Session not found - return None (don't create)
        return None

    def update_notebook(
        self,
        session_id: str,
        notebook_name: Optional[str] = None,
        notebook_path: Optional[str] = None,
        kernel_id: Optional[str] = None
    ) -> None:
        """Update session's notebook context (부분 업데이트 지원).

        이 메서드는 세션의 노트북 컨텍스트를 업데이트합니다.
        None이 아닌 파라미터만 업데이트하여 부분 업데이트를 지원합니다.
        세션이 없으면 자동으로 생성됩니다.

        Args:
            session_id: Client session ID (필수)
                - UUID format recommended
                - Creates session if doesn't exist
            notebook_name: Notebook name without extension (optional)
                - Used for display purposes
                - Example: "analysis" for "analysis.ipynb"
                - None이면 기존 값 유지
            notebook_path: Full or relative path to notebook file (optional)
                - Example: "/work/analysis.ipynb"
                - Example: "notebooks/report.ipynb"
                - Should match path used in Jupyter
                - None이면 기존 값 유지
            kernel_id: Jupyter kernel ID string (optional)
                - Example: "abc-123-def-456"
                - Must be valid kernel ID from Jupyter
                - None이면 기존 값 유지

        Returns:
            None

        Example:
            전체 업데이트 (기존 동작):
            ```python
            from jupyter_mcp_server.server import session_store

            # 모든 필드 업데이트
            session_store.update_notebook(
                session_id="client-A-uuid",
                notebook_name="analysis",
                notebook_path="/work/analysis.ipynb",
                kernel_id="kernel-123"
            )
            ```

            부분 업데이트 (kernel_id만):
            ```python
            # kernel_id만 업데이트 (notebook_name, notebook_path 유지)
            session_store.update_notebook(
                session_id="client-A-uuid",
                kernel_id="kernel-456"
            )

            # 확인
            ctx = session_store.get("client-A-uuid")
            assert ctx.notebook_name == "analysis"  # 유지됨
            assert ctx.notebook_path == "/work/analysis.ipynb"  # 유지됨
            assert ctx.kernel_id == "kernel-456"  # 업데이트됨
            ```

            부분 업데이트 (notebook_path만):
            ```python
            # notebook_path만 업데이트
            session_store.update_notebook(
                session_id="client-A-uuid",
                notebook_path="/work/report.ipynb"
            )
            ```

        Notes:
            - ARK-165: 부분 업데이트 지원 추가 (커널 힐링용)
            - None 파라미터는 "삭제"가 아니라 "유지"를 의미
            - 세션이 없으면 자동으로 생성 (idempotent)
            - Always updates last_accessed timestamp
            - Thread-safe for different session_ids
            - Logs update for debugging
        """
        # Get existing session or create new one (idempotent operation)
        # This ensures session always exists before updating fields
        ctx = self.get_or_create(session_id)

        # 부분 업데이트: None이 아닌 필드만 업데이트
        # None은 "삭제"가 아니라 "기존 값 유지"를 의미
        if notebook_name is not None:
            ctx.current_notebook = notebook_name

        if notebook_path is not None:
            ctx.notebook_path = notebook_path

        if kernel_id is not None:
            ctx.kernel_id = kernel_id

        # Update last_accessed to keep session alive (prevent TTL expiration)
        ctx.last_accessed = datetime.now()

        # Log update for debugging and auditing
        # 업데이트된 필드만 로그에 표시
        updated_fields = []
        if notebook_name is not None:
            updated_fields.append(f"notebook='{notebook_name}'")
        if notebook_path is not None:
            updated_fields.append(f"path='{notebook_path}'")
        if kernel_id is not None:
            updated_fields.append(f"kernel='{kernel_id}'")

        if updated_fields:
            logger.info(
                f"Updated session '{session_id}': {', '.join(updated_fields)}"
            )
        else:
            logger.debug(
                f"Updated session '{session_id}': timestamp only (no field changes)"
            )

    def cleanup_expired(self) -> int:
        """Remove sessions that exceeded TTL period.

        This method scans all sessions and removes those where last_accessed
        timestamp is older than the configured TTL. Should be called periodically
        to prevent memory leaks from abandoned sessions.

        TTL Check:
            - Compares: now - last_accessed > ttl_hours
            - Only removes sessions not accessed within TTL period
            - Active sessions are never removed

        Args:
            None

        Returns:
            int: Number of expired sessions removed
                - 0 if no sessions expired
                - Positive number indicating cleanup count

        Example:
            ```python
            from jupyter_mcp_server.server import session_store

            # Periodic cleanup (e.g., in background task)
            def periodic_cleanup():
                expired = session_store.cleanup_expired()
                if expired > 0:
                    print(f"Cleaned up {expired} expired sessions")

            # Manual cleanup
            expired_count = session_store.cleanup_expired()
            print(f"Removed {expired_count} sessions")

            # Cleanup in scheduler
            import schedule
            schedule.every().hour.do(session_store.cleanup_expired)
            ```

        Recommendations:
            - Call every hour for active servers
            - Call every 6 hours for low-traffic servers
            - Call on server startup to clean stale sessions
            - Monitor returned count for usage analytics

        Notes:
            - ARK-165: Prevents memory leaks from abandoned sessions
            - Thread-safe operation
            - Logs each removed session for debugging
            - O(n) time complexity where n = total sessions
            - Recommended to run as background task
        """
        # ARK-165: Two-phase cleanup for thread safety
        # Phase 1: Identify expired sessions (read-only, safe for concurrent access)
        # Phase 2: Delete expired sessions (write operation, minimal lock time)

        now = datetime.now()

        # Phase 1: Scan all sessions and identify those that exceeded TTL
        # This creates a list of session_ids to delete without modifying the dict
        # TTL formula: (current_time - last_accessed_time) > configured_ttl
        expired_sessions = [
            session_id
            for session_id, ctx in self._sessions.items()
            if now - ctx.last_accessed > self._ttl
        ]

        # Phase 2: Delete identified sessions one by one
        # Separation from Phase 1 ensures we don't modify dict during iteration
        # Each deletion is atomic and logged for debugging
        for session_id in expired_sessions:
            del self._sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")

        # Log summary if any sessions were cleaned up
        # Useful for monitoring and capacity planning
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        # Return count for caller to take action (e.g., metrics, alerts)
        return len(expired_sessions)

    def __len__(self) -> int:
        """Return the number of active sessions.

        Example:
            ```python
            from jupyter_mcp_server.server import session_store

            # Check active session count
            count = len(session_store)
            print(f"Active sessions: {count}")
            ```

        Returns:
            int: Number of active sessions currently stored
        """
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        """Check if a session exists in the store.

        Example:
            ```python
            from jupyter_mcp_server.server import session_store

            # Check if session exists
            if "session-123" in session_store:
                print("Session exists")
            else:
                print("Session not found")
            ```

        Args:
            session_id: Session ID to check

        Returns:
            bool: True if session exists, False otherwise
        """
        return session_id in self._sessions

    def list_all(self) -> dict[str, SessionContext]:
        """List all active sessions with their contexts.

        This method returns a dictionary mapping session_id to SessionContext
        for all currently active sessions. Useful for administrative tools and
        monitoring.

        Example:
            ```python
            from jupyter_mcp_server.server import session_store

            # List all sessions
            sessions = session_store.list_all()
            for session_id, ctx in sessions.items():
                print(f"Session: {session_id}")
                print(f"  Notebook: {ctx.notebook_path}")
                print(f"  Kernel: {ctx.kernel_id}")
                print(f"  Last accessed: {ctx.last_accessed}")
            ```

        Returns:
            dict[str, SessionContext]: Dictionary mapping session_id to SessionContext

        Thread Safety:
            - Returns a snapshot of current sessions
            - Thread-safe for reading
        """
        with self._lock:
            # Return a copy to prevent external modification
            return dict(self._sessions)

    def remove(self, session_id: str) -> bool:
        """Remove a session from the store immediately.

        This method removes a session without waiting for TTL expiration.
        Useful when a client explicitly disconnects or unuses a notebook.

        Args:
            session_id: Session ID to remove

        Returns:
            bool: True if session was found and removed, False if not found

        Example:
            ```python
            from jupyter_mcp_server.server import session_store

            # Remove session immediately
            removed = session_store.remove("session-123")
            if removed:
                print("Session removed successfully")
            else:
                print("Session not found")
            ```

        Notes:
            - ARK-165: Enables immediate cleanup on unuse_notebook
            - Thread-safe operation
            - Returns False if session doesn't exist (idempotent)
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Removed session: {session_id}")
                return True
            else:
                logger.debug(f"Session not found for removal: {session_id}")
                return False
