"""
통합 테스트: use_notebook_tool 커널 자동 복구 기능

커널 자동 복구 기능을 검증합니다:
1. JUPYTER_SERVER 모드: kernel_id가 kernel_manager에 없을 때 자동 복구
2. MCP_SERVER 모드: kernel_id가 Jupyter server에 없을 때 자동 복구
3. 경고 로그 출력 확인
4. SessionStore 동기화 확인
"""

import pytest
import json
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from jupyter_mcp_server.tools.use_notebook_tool import UseNotebookTool
from jupyter_mcp_server.tools._base import ServerMode
from jupyter_mcp_server.session_store import SessionStore


class TestJupyterServerModeKernelRecovery:
    """JUPYTER_SERVER 모드 커널 자동 복구 테스트"""

    @pytest.mark.asyncio
    async def test_jupyter_server_mode_kernel_recovery(self, caplog):
        """커널이 존재하지 않을 때 새 커널이 자동으로 생성됨"""
        # Arrange
        tool = UseNotebookTool()
        session_store = SessionStore()

        # 존재하지 않는 kernel_id를 SessionStore에 설정
        session_store.update_notebook(
            "test-session",
            "test",
            "test.ipynb",
            "non-existent-kernel-id"
        )

        # Mock kernel_manager: 기존 커널 없음, 새 커널 생성 가능
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=False)  # kernel_id not in kernel_manager

        # _start_kernel_local이 새 커널을 반환하도록 모킹
        new_kernel = {"id": "new-kernel-id-123"}
        tool._start_kernel_local = AsyncMock(return_value=new_kernel)

        # _check_path_local이 성공하도록 모킹
        tool._check_path_local = AsyncMock(return_value=(True, None))

        # Mock contents_manager
        mock_contents_manager = AsyncMock()
        mock_contents_manager.get = AsyncMock(return_value={
            "type": "notebook",
            "path": "test.ipynb",
            "content": {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 4
            }
        })

        # Mock session_manager
        mock_session_manager = AsyncMock()
        mock_session_manager.create_session = AsyncMock(return_value={"id": "jupyter-session-123"})

        # Mock notebook_manager
        mock_notebook_manager = Mock()
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Act
        with caplog.at_level(logging.WARNING):
            result = await tool.execute(
                mode=ServerMode.JUPYTER_SERVER,
                session_id="test-session",
                notebook_name="test",
                notebook_path="test.ipynb",
                kernel_manager=mock_kernel_manager,
                contents_manager=mock_contents_manager,
                session_manager=mock_session_manager,
                notebook_manager=mock_notebook_manager,
                session_store=session_store,
                use_mode="connect"
            )

        # Assert
        # 1. 새 커널이 생성되었는지 확인
        tool._start_kernel_local.assert_called_once_with(mock_kernel_manager)

        # 2. 경고 로그가 출력되었는지 확인
        assert any("not found" in record.message.lower() for record in caplog.records)
        assert any("auto-recovery" in record.message.lower() for record in caplog.records)

        # 3. 결과에 복구 메시지가 포함되어 있는지 확인
        result_json = json.loads(result)
        message = result_json.get("result", {}).get("message", "")
        assert "Previous kernel was not found" in message
        assert "Created new kernel" in message

        # 4. SessionStore가 새 kernel_id로 업데이트되었는지 확인
        ctx = session_store.get("test-session")
        assert ctx.kernel_id == "new-kernel-id-123"

    @pytest.mark.asyncio
    async def test_jupyter_server_mode_kernel_creation_failure(self, caplog):
        """커널 생성 실패 시 명확한 에러 메시지 반환"""
        # Arrange
        tool = UseNotebookTool()
        session_store = SessionStore()

        session_store.update_notebook(
            "test-session",
            "test",
            "test.ipynb",
            "non-existent-kernel-id"
        )

        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=False)

        # _start_kernel_local이 실패하도록 모킹
        tool._start_kernel_local = AsyncMock(side_effect=Exception("Kernel creation failed"))

        # _check_path_local이 성공하도록 모킹
        tool._check_path_local = AsyncMock(return_value=(True, None))

        mock_contents_manager = AsyncMock()
        mock_contents_manager.get = AsyncMock(return_value={
            "type": "notebook",
            "path": "test.ipynb"
        })

        mock_notebook_manager = Mock()

        # Act
        with caplog.at_level(logging.ERROR):
            result = await tool.execute(
                mode=ServerMode.JUPYTER_SERVER,
                session_id="test-session",
                notebook_name="test",
                notebook_path="test.ipynb",
                kernel_manager=mock_kernel_manager,
                contents_manager=mock_contents_manager,
                notebook_manager=mock_notebook_manager,
                session_store=session_store,
                use_mode="connect"
            )

        # Assert
        # 1. 에러 로그가 출력되었는지 확인
        assert any("Failed to create new kernel" in record.message for record in caplog.records)

        # 2. 명확한 에러 메시지가 반환되었는지 확인
        assert "not found and failed to create new kernel" in result
        assert "Kernel creation failed" in result


class TestMcpServerModeKernelRecovery:
    """MCP_SERVER 모드 커널 자동 복구 테스트"""

    @pytest.mark.asyncio
    async def test_mcp_server_mode_kernel_recovery(self, caplog):
        """커널이 존재하지 않을 때 kernel_id=None으로 새 커널 생성"""
        # Arrange
        tool = UseNotebookTool()
        session_store = SessionStore()

        # 존재하지 않는 kernel_id를 SessionStore에 설정
        session_store.update_notebook(
            "test-session",
            "test",
            "test.ipynb",
            "non-existent-kernel-id"
        )

        # Mock server_client: 커널 리스트에 해당 kernel_id 없음
        mock_server_client = Mock()
        mock_server_client.kernels.list_kernels = Mock(return_value=[])

        # _check_path_http이 성공하도록 모킹
        tool._check_path_http = AsyncMock(return_value=(True, None))

        # Mock KernelClient
        mock_kernel_instance = Mock()
        mock_kernel_instance.id = "new-kernel-id-456"
        mock_kernel_instance.start = Mock()

        # Mock notebook_manager
        mock_notebook_manager = Mock()
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Act
        with caplog.at_level(logging.WARNING):
            with patch('jupyter_mcp_server.tools.use_notebook_tool.KernelClient', return_value=mock_kernel_instance):
                result = await tool.execute(
                    mode=ServerMode.MCP_SERVER,
                    session_id="test-session",
                    notebook_name="test",
                    notebook_path="test.ipynb",
                    server_client=mock_server_client,
                    runtime_url="http://localhost:8888",
                    runtime_token="test-token",
                    notebook_manager=mock_notebook_manager,
                    session_store=session_store,
                    use_mode="connect"
                )

        # Assert
        # 1. 경고 로그가 출력되었는지 확인
        assert any("not found" in record.message.lower() for record in caplog.records)
        assert any("auto-recovery" in record.message.lower() for record in caplog.records)

        # 2. 결과에 복구 메시지가 포함되어 있는지 확인
        result_json = json.loads(result)
        message = result_json.get("result", {}).get("message", "")
        assert "Previous kernel was not found" in message
        assert "Creating new kernel" in message

        # 3. SessionStore가 새 kernel_id로 업데이트되었는지 확인
        ctx = session_store.get("test-session")
        assert ctx.kernel_id == "new-kernel-id-456"

        # 4. KernelClient가 kernel_id=None으로 호출되었는지 확인
        mock_kernel_instance.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_server_mode_kernel_connection_failure(self, caplog):
        """KernelClient 연결 실패 시 명확한 에러 메시지 반환"""
        # Arrange
        tool = UseNotebookTool()
        session_store = SessionStore()

        session_store.update_notebook(
            "test-session",
            "test",
            "test.ipynb",
            "non-existent-kernel-id"
        )

        mock_server_client = Mock()
        mock_server_client.kernels.list_kernels = Mock(return_value=[])

        # _check_path_http이 성공하도록 모킹
        tool._check_path_http = AsyncMock(return_value=(True, None))

        mock_notebook_manager = Mock()

        # Act
        with caplog.at_level(logging.ERROR):
            with patch('jupyter_mcp_server.tools.use_notebook_tool.KernelClient', side_effect=Exception("Connection failed")):
                result = await tool.execute(
                    mode=ServerMode.MCP_SERVER,
                    session_id="test-session",
                    notebook_name="test",
                    notebook_path="test.ipynb",
                    server_client=mock_server_client,
                    runtime_url="http://localhost:8888",
                    runtime_token="test-token",
                    notebook_manager=mock_notebook_manager,
                    session_store=session_store,
                    use_mode="connect"
                )

        # Assert
        # 1. 에러 로그가 출력되었는지 확인
        assert any("Failed to create/connect kernel" in record.message for record in caplog.records)

        # 2. 명확한 에러 메시지가 반환되었는지 확인
        assert "Failed to connect to kernel" in result
        assert "Connection failed" in result


class TestSessionStoreSync:
    """SessionStore 동기화 확인 테스트"""

    @pytest.mark.asyncio
    async def test_session_store_updates_with_new_kernel_id(self):
        """새 커널 생성 후 SessionStore가 자동으로 업데이트됨"""
        # Arrange
        tool = UseNotebookTool()
        session_store = SessionStore()

        # 기존 커널 ID 설정
        old_kernel_id = "old-kernel-id"
        session_store.update_notebook("test-session", "test", "test.ipynb", old_kernel_id)

        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=False)

        new_kernel_id = "new-kernel-id-789"
        new_kernel = {"id": new_kernel_id}
        tool._start_kernel_local = AsyncMock(return_value=new_kernel)

        # _check_path_local이 성공하도록 모킹
        tool._check_path_local = AsyncMock(return_value=(True, None))

        mock_contents_manager = AsyncMock()
        mock_contents_manager.get = AsyncMock(return_value={
            "type": "notebook",
            "path": "test.ipynb",
            "content": {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
        })

        mock_session_manager = AsyncMock()
        mock_session_manager.create_session = AsyncMock(return_value={"id": "jupyter-session-123"})

        mock_notebook_manager = Mock()
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Act
        await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            session_id="test-session",
            notebook_name="test",
            notebook_path="test.ipynb",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            session_manager=mock_session_manager,
            notebook_manager=mock_notebook_manager,
            session_store=session_store,
            use_mode="connect"
        )

        # Assert
        ctx = session_store.get("test-session")
        assert ctx is not None
        assert ctx.kernel_id == new_kernel_id
        assert ctx.kernel_id != old_kernel_id
