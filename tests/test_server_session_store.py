"""
Unit tests for server.py SessionStore integration (ARK-165)

server.py에 session_store 전역 변수가 정상적으로 통합되었는지 테스트합니다.
"""

import pytest
from jupyter_mcp_server.session_store import SessionStore


class TestServerSessionStoreIntegration:
    """server.py SessionStore 통합 테스트"""

    def test_session_store_import(self):
        """SessionStore import 테스트"""
        from jupyter_mcp_server.server import session_store

        assert session_store is not None
        assert isinstance(session_store, SessionStore)

    def test_session_store_global_access(self):
        """전역 인스턴스 접근 테스트 - 동일한 인스턴스 반환"""
        from jupyter_mcp_server.server import session_store as store1
        from jupyter_mcp_server.server import session_store as store2

        # 동일한 인스턴스여야 함 (싱글톤 패턴)
        assert store1 is store2

    def test_default_ttl(self):
        """기본 TTL 확인 (24시간)"""
        from jupyter_mcp_server.server import session_store

        # TTL이 24시간으로 설정되어 있어야 함
        assert session_store._ttl.total_seconds() == 24 * 3600

    def test_session_store_functionality(self):
        """SessionStore 기본 기능 테스트"""
        from jupyter_mcp_server.server import session_store

        # 세션 생성
        ctx = session_store.get_or_create("test-session")
        assert ctx is not None

        # 노트북 컨텍스트 업데이트
        session_store.update_notebook(
            session_id="test-session",
            notebook_name="test",
            notebook_path="/path/test.ipynb",
            kernel_id="kernel-123"
        )

        # 조회
        ctx = session_store.get("test-session")
        assert ctx.notebook_path == "/path/test.ipynb"
        assert ctx.kernel_id == "kernel-123"

        # 정리 (다른 테스트에 영향 없도록)
        session_store._sessions.clear()

    def test_session_store_independent_from_notebook_manager(self):
        """SessionStore가 NotebookManager와 독립적으로 작동하는지 테스트"""
        from jupyter_mcp_server.server import session_store, notebook_manager

        # session_store와 notebook_manager는 별개의 객체
        assert session_store is not notebook_manager

        # session_store에 세션 추가해도 notebook_manager에 영향 없음
        initial_notebook_count = len(notebook_manager._notebooks)

        session_store.update_notebook(
            session_id="independent-test",
            notebook_name="test",
            notebook_path="/path/test.ipynb",
            kernel_id="kernel-456"
        )

        # notebook_manager의 notebook 개수 변화 없음
        assert len(notebook_manager._notebooks) == initial_notebook_count

        # 정리
        session_store._sessions.clear()
