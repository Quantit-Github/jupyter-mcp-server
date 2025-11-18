"""
Unit tests for SessionStore (ARK-165)

SessionStore와 SessionContext의 모든 기능을 테스트합니다.
"""

import pytest
import time
from datetime import datetime, timedelta
from jupyter_mcp_server.session_store import SessionStore, SessionContext


class TestSessionContext:
    """SessionContext dataclass 테스트"""

    def test_session_context_creation(self):
        """SessionContext 생성 테스트"""
        ctx = SessionContext()
        assert ctx.current_notebook is None
        assert ctx.kernel_id is None
        assert ctx.notebook_path is None
        assert isinstance(ctx.created_at, datetime)
        assert isinstance(ctx.last_accessed, datetime)

    def test_session_context_with_values(self):
        """SessionContext 값 설정 테스트"""
        ctx = SessionContext(
            current_notebook="test_notebook",
            kernel_id="kernel-123",
            notebook_path="/path/to/notebook.ipynb"
        )
        assert ctx.current_notebook == "test_notebook"
        assert ctx.kernel_id == "kernel-123"
        assert ctx.notebook_path == "/path/to/notebook.ipynb"


class TestSessionStore:
    """SessionStore 클래스 테스트"""

    def test_session_store_initialization(self):
        """SessionStore 초기화 테스트"""
        store = SessionStore()
        assert len(store) == 0
        assert store._ttl.total_seconds() == 24 * 3600

    def test_session_store_custom_ttl(self):
        """커스텀 TTL 설정 테스트"""
        store = SessionStore(ttl_hours=12)
        assert store._ttl.total_seconds() == 12 * 3600

    def test_get_or_create_new_session(self):
        """새 세션 생성 테스트"""
        store = SessionStore()
        ctx = store.get_or_create("session-1")

        assert ctx is not None
        assert isinstance(ctx, SessionContext)
        assert ctx.created_at is not None
        assert ctx.last_accessed is not None
        assert len(store) == 1
        assert "session-1" in store

    def test_get_or_create_existing_session(self):
        """기존 세션 조회 테스트"""
        store = SessionStore()
        ctx1 = store.get_or_create("session-1")
        time.sleep(0.01)  # 약간의 시간 차이
        ctx2 = store.get_or_create("session-1")

        # 동일한 세션 객체
        assert ctx1 is ctx2
        assert len(store) == 1

        # last_accessed가 업데이트됨
        assert ctx2.last_accessed > ctx1.created_at

    def test_get_existing_session(self):
        """세션 조회 (get) 테스트"""
        store = SessionStore()
        store.get_or_create("session-1")

        ctx = store.get("session-1")
        assert ctx is not None
        assert isinstance(ctx, SessionContext)

    def test_get_non_existent_session(self):
        """존재하지 않는 세션 조회 테스트"""
        store = SessionStore()
        ctx = store.get("non-existent")
        assert ctx is None

    def test_update_notebook(self):
        """노트북 컨텍스트 업데이트 테스트"""
        store = SessionStore()
        store.update_notebook(
            session_id="session-1",
            notebook_name="nb1",
            notebook_path="path/nb1.ipynb",
            kernel_id="kernel-1"
        )

        ctx = store.get("session-1")
        assert ctx is not None
        assert ctx.current_notebook == "nb1"
        assert ctx.notebook_path == "path/nb1.ipynb"
        assert ctx.kernel_id == "kernel-1"

    def test_update_notebook_creates_session(self):
        """update_notebook이 세션 자동 생성 테스트"""
        store = SessionStore()

        # 세션이 없는 상태에서 update_notebook 호출
        store.update_notebook(
            session_id="session-new",
            notebook_name="nb1",
            notebook_path="path/nb1.ipynb",
            kernel_id="kernel-1"
        )

        # 세션이 생성되어야 함
        assert "session-new" in store
        ctx = store.get("session-new")
        assert ctx.current_notebook == "nb1"

    def test_update_notebook_overwrites(self):
        """노트북 컨텍스트 덮어쓰기 테스트"""
        store = SessionStore()

        # 첫 번째 노트북
        store.update_notebook("session-1", "nb1", "path/nb1.ipynb", "kernel-1")

        # 두 번째 노트북으로 전환
        store.update_notebook("session-1", "nb2", "path/nb2.ipynb", "kernel-2")

        ctx = store.get("session-1")
        assert ctx.current_notebook == "nb2"
        assert ctx.notebook_path == "path/nb2.ipynb"
        assert ctx.kernel_id == "kernel-2"

    def test_cleanup_expired_sessions(self):
        """TTL 기반 세션 만료 테스트"""
        # 매우 짧은 TTL (0.00002시간 ≈ 0.072초)
        store = SessionStore(ttl_hours=0.00002)

        # 세션 생성
        store.get_or_create("session-1")
        store.get_or_create("session-2")
        assert len(store) == 2

        # TTL 초과 대기
        time.sleep(0.1)

        # Cleanup
        removed_count = store.cleanup_expired()
        assert removed_count == 2
        assert len(store) == 0
        assert "session-1" not in store
        assert "session-2" not in store

    def test_cleanup_expired_keeps_active_sessions(self):
        """활성 세션은 유지되는지 테스트"""
        # 매우 짧은 TTL (0.00003시간 ≈ 0.108초)
        store = SessionStore(ttl_hours=0.00003)

        # 첫 번째 세션 생성
        store.get_or_create("session-1")

        # 약간 대기
        time.sleep(0.08)

        # 두 번째 세션 생성 (최근)
        store.get_or_create("session-2")

        # 더 대기 (session-1만 만료)
        time.sleep(0.05)

        # Cleanup
        removed_count = store.cleanup_expired()

        # session-1은 만료, session-2는 유지
        assert removed_count == 1
        assert "session-1" not in store
        assert "session-2" in store

    def test_last_accessed_update_on_get(self):
        """get 호출 시 last_accessed 업데이트 테스트"""
        store = SessionStore()
        ctx = store.get_or_create("session-1")
        original_time = ctx.last_accessed

        time.sleep(0.01)

        # get 호출
        ctx2 = store.get("session-1")
        assert ctx2.last_accessed > original_time

    def test_last_accessed_update_on_get_or_create(self):
        """get_or_create 호출 시 last_accessed 업데이트 테스트"""
        store = SessionStore()
        ctx = store.get_or_create("session-1")
        original_time = ctx.last_accessed

        time.sleep(0.01)

        # get_or_create 재호출
        ctx2 = store.get_or_create("session-1")
        assert ctx2.last_accessed > original_time

    def test_last_accessed_update_on_update_notebook(self):
        """update_notebook 호출 시 last_accessed 업데이트 테스트"""
        store = SessionStore()
        ctx = store.get_or_create("session-1")
        original_time = ctx.last_accessed

        time.sleep(0.01)

        # update_notebook 호출
        store.update_notebook("session-1", "nb1", "path/nb1.ipynb", "kernel-1")

        ctx2 = store.get("session-1")
        assert ctx2.last_accessed > original_time

    def test_multiple_sessions_isolation(self):
        """여러 세션의 독립성 테스트"""
        store = SessionStore()

        # 세션 A
        store.update_notebook("session-a", "nb-a", "path/a.ipynb", "kernel-a")

        # 세션 B
        store.update_notebook("session-b", "nb-b", "path/b.ipynb", "kernel-b")

        # 세션 C
        store.update_notebook("session-c", "nb-c", "path/c.ipynb", "kernel-c")

        # 각 세션이 독립적으로 유지
        ctx_a = store.get("session-a")
        ctx_b = store.get("session-b")
        ctx_c = store.get("session-c")

        assert ctx_a.current_notebook == "nb-a"
        assert ctx_b.current_notebook == "nb-b"
        assert ctx_c.current_notebook == "nb-c"

        assert ctx_a.kernel_id == "kernel-a"
        assert ctx_b.kernel_id == "kernel-b"
        assert ctx_c.kernel_id == "kernel-c"

    def test_len(self):
        """__len__ 매직 메서드 테스트"""
        store = SessionStore()
        assert len(store) == 0

        store.get_or_create("session-1")
        assert len(store) == 1

        store.get_or_create("session-2")
        assert len(store) == 2

        store.get_or_create("session-3")
        assert len(store) == 3

    def test_contains(self):
        """__contains__ 매직 메서드 테스트"""
        store = SessionStore()
        store.get_or_create("session-1")

        assert "session-1" in store
        assert "session-2" not in store

    def test_cleanup_expired_returns_count(self):
        """cleanup_expired가 올바른 개수를 반환하는지 테스트"""
        # 매우 짧은 TTL (0.00002시간 ≈ 0.072초)
        store = SessionStore(ttl_hours=0.00002)

        # 세션 5개 생성
        for i in range(5):
            store.get_or_create(f"session-{i}")

        assert len(store) == 5

        # TTL 초과 대기
        time.sleep(0.1)

        # Cleanup
        removed_count = store.cleanup_expired()
        assert removed_count == 5
        assert len(store) == 0

    def test_cleanup_expired_no_sessions(self):
        """세션이 없을 때 cleanup_expired 테스트"""
        store = SessionStore()
        removed_count = store.cleanup_expired()
        assert removed_count == 0
        assert len(store) == 0
