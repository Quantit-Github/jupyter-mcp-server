# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/ko/).

## [Unreleased] - 2025-01-18

### Added
- **[ARK-165] SessionManager 레이어 도입 및 커널 힐링 중앙화**: SessionManager 오케스트레이션 레이어를 추가하여 모든 cell operation tool에서 자동 커널 힐링을 지원합니다.
  - **SessionManager 클래스** (`session_manager.py`): SessionStore 위에 커널 헬스 체크 및 자동 힐링 로직을 제공하는 오케스트레이션 레이어
  - **Async 컨텍스트 함수**: `get_notebook_context_from_session_async()` 함수 추가 (auto_heal=True 기본값)
  - **6개 도구 업데이트**: use_notebook, execute_cell, read_cell, insert_cell, delete_cell, overwrite_cell_source
  - **코드 개선**: use_notebook_tool에서 ~96줄 중복 코드 제거
  - **관심사 분리**: SessionStore(저장소) vs SessionManager(오케스트레이션) 명확한 레이어 분리
  - **자동 복구**: 모든 cell operation이 죽은 커널을 자동으로 감지하고 복구
  - 파일:
    - 신규: `jupyter_mcp_server/session_manager.py` (SessionManager 클래스)
    - 수정: `jupyter_mcp_server/utils.py` (async 컨텍스트 함수 추가)
    - 수정: `jupyter_mcp_server/tools/use_notebook_tool.py` (~96줄 제거)
    - 수정: `jupyter_mcp_server/tools/execute_cell_tool.py` (async 컨텍스트 사용)
    - 수정: `jupyter_mcp_server/tools/read_cell_tool.py` (async 컨텍스트 사용)
    - 수정: `jupyter_mcp_server/tools/insert_cell_tool.py` (async 컨텍스트 사용)
    - 수정: `jupyter_mcp_server/tools/delete_cell_tool.py` (async 컨텍스트 사용)
    - 수정: `jupyter_mcp_server/tools/overwrite_cell_source_tool.py` (async 컨텍스트 사용)

- **[ARK-165] SessionStore 부분 업데이트 지원**: `update_notebook()` 메서드가 이제 선택적 파라미터를 지원하여 kernel_id만 업데이트할 수 있습니다.
  - 커널 힐링 시 notebook_name/path를 건드리지 않고 kernel_id만 업데이트
  - 기존 코드 호환성 유지 (모든 파라미터 전달 시 동일하게 동작)
  - 파일: `jupyter_mcp_server/session_store.py` (Line 79-99, update_notebook 메서드)

- **[ARK-165] 커널 힐링 테스트 추가** (60개 테스트):
  - **SessionManager 단위 테스트** 14개 (`test_session_manager.py`):
    - 커널 헬스 체크 (JUPYTER_SERVER + MCP_SERVER)
    - 커널 힐링 성공 시나리오
    - 에러 핸들링 및 실패 케이스
    - SessionStore 부분 업데이트
  - **Utils 통합 테스트** 16개 (`test_utils_session.py`):
    - Sync 함수 backward compatibility (3개)
    - Async 함수 자동 힐링 (6개)
    - Config fallback 시나리오 (7개)
  - **use_notebook 회귀 테스트** 19개 (`test_use_notebook_tool_ark165.py`):
    - 기존 기능 회귀 없음 검증
    - 커널 자동 복구 워크플로우
    - 멀티클라이언트 세션 시나리오
  - **Cell Tool 통합 테스트** 11개 (`test_cell_tools_kernel_healing.py`):
    - 5개 cell tool별 힐링 동작 검증 (6개)
    - 코드 통합 검증 (5개)

### Fixed
- **[ARK-165] NotebookManager와 SessionStore 상태 충돌 수정**: `use_notebook` 도구에서 NotebookManager 기반 검증 로직을 제거하여 멀티클라이언트 세션 격리 문제를 해결했습니다.
  - 서로 다른 세션이 같은 `notebook_name`을 독립적으로 사용할 수 있게 되었습니다.
  - SessionStore 기반 검증만 사용하여 상태 관리를 단순화했습니다.
  - 파일: `jupyter_mcp_server/tools/use_notebook_tool.py` (Line 373-389, 19줄 제거)

### Changed
- **[Breaking Change] session_id 필수 파라미터 강제**: `use_notebook` 도구에서 `session_id`가 명시적으로 필수가 되었습니다.
  - `session_id=None`일 때 명확한 에러 메시지를 반환합니다.
  - 멀티클라이언트 지원(ARK-165)을 위한 변경사항입니다.
  - 이전에는 backward compatibility를 위해 optional이었으나, 이제 필수입니다.

- **kernel_id 파라미터 추가 및 동작 개선**: `use_notebook` 함수 시그니처에 `kernel_id` 파라미터를 명시적으로 추가했습니다.
  - 기존 세션에서 kernel 재사용 시 덮어쓰기 버그를 수정했습니다.
  - 세션 전환 시 kernel을 올바르게 재사용합니다.

### Added
- **커널 자동 복구 기능**: `use_notebook` 도구가 커널이 존재하지 않을 때 자동으로 새 커널을 생성합니다.
  - **JUPYTER_SERVER 모드**: `kernel_manager`에 커널이 없을 때 자동으로 새 커널 생성
  - **MCP_SERVER 모드**: Jupyter 서버에 커널이 없을 때 자동으로 새 커널 생성
  - 커널 복구 시 경고 로그 및 사용자 알림 메시지 출력
  - SessionStore가 새 kernel_id로 자동 업데이트됨
  - 파일: `jupyter_mcp_server/tools/use_notebook_tool.py` (Line 322-354, 418-475, 세션 검증 및 복구 로직)

- **커널 자동 복구 통합 테스트 5개 추가**:
  - `test_jupyter_server_mode_kernel_recovery`: JUPYTER_SERVER 모드 커널 자동 복구 검증
  - `test_jupyter_server_mode_kernel_creation_failure`: 커널 생성 실패 시 명확한 에러 메시지 검증
  - `test_mcp_server_mode_kernel_recovery`: MCP_SERVER 모드 커널 자동 복구 검증
  - `test_mcp_server_mode_kernel_connection_failure`: 커널 연결 실패 시 명확한 에러 메시지 검증
  - `test_session_store_updates_with_new_kernel_id`: SessionStore 동기화 자동 업데이트 검증
  - 파일: `tests/test_use_notebook_kernel_recovery.py`

- **멀티클라이언트 시나리오 테스트 5개 추가**:
  - `test_different_session_same_name_different_path`: 서로 다른 세션이 같은 notebook_name을 독립적으로 사용 가능한지 검증
  - `test_same_session_notebook_switch`: 같은 세션이 노트북 전환 시 kernel 재사용을 검증
  - `test_conflict_detection_same_path`: 서로 다른 세션이 같은 파일 경로 사용 시 충돌 감지 검증
  - `test_idempotent_same_session_same_notebook`: 같은 세션/노트북 재호출 시 idempotent 동작 검증
  - `test_session_id_required`: session_id=None일 때 에러 반환 검증

### Technical Details
- **의존성 변경**: 없음
- **Breaking Changes**: `session_id`가 이제 필수 파라미터입니다. 기존 코드에서 `session_id=None`을 사용하던 경우 업데이트가 필요합니다.
- **관련 이슈**: ARK-165 (멀티클라이언트 세션 지원)

### Testing
- ✅ 커널 자동 복구 테스트 5개 모두 통과 (새로 추가)
- ✅ 기존 use_notebook 테스트 19개 모두 통과 (회귀 없음)
- ✅ SessionStore 테스트 26개 통과
- ✅ 전체 테스트: 24/24 use_notebook 관련 테스트 통과 (100%)

### 마이그레이션 가이드

**Breaking Change**: `session_id`가 필수가 되었습니다.

**변경 전:**
```python
result = use_notebook(
    notebook_name="my_notebook",
    notebook_path="work.ipynb"
    # session_id는 optional이었음
)
```

**변경 후:**
```python
result = use_notebook(
    notebook_name="my_notebook",
    notebook_path="work.ipynb",
    session_id="your-session-id"  # 필수!
)
```

**에러 발생 시:**
```json
{
  "result": {
    "status": "error",
    "message": "session_id is required for multi-client support (ARK-165). Please provide a valid session_id.",
    "action": "rejected"
  }
}
```
