# 아키텍처 결정사항

## 2025-01-18 - SessionStore 중심 설계로 전환 (ARK-165)

### 배경

기존 시스템은 두 개의 상태 관리 시스템을 동시에 사용하고 있었습니다:
- **NotebookManager**: `notebook_name` → kernel/path 매핑 (전역, 세션 미인식)
- **SessionStore**: `session_id` → SessionContext 매핑 (세션별 격리)

이로 인해 서로 다른 세션이 같은 `notebook_name`을 사용하려고 할 때 충돌이 발생했습니다.

### 문제 분석

**근본 원인**: NotebookManager와 SessionStore가 서로 다른 키를 사용하여 상태를 관리함
- NotebookManager: `notebook_name`을 키로 사용 → 세션 간 충돌
- SessionStore: `session_id`를 키로 사용 → 세션별 격리 지원

**실패 시나리오**:
```python
# Session A
use_notebook(session_id="A", notebook_name="nb1", path="a.ipynb")  # 성공

# Session B  
use_notebook(session_id="B", notebook_name="nb1", path="b.ipynb")  # 실패!
# NotebookManager에 "nb1"이 이미 존재하여 거부됨
```

### 결정 사항

**NotebookManager 검증 로직 완전 제거**:
- `use_notebook_tool.py`의 Line 373-389 (19줄) 삭제
- NotebookManager는 backward compatibility를 위해 `add_notebook()` 호출만 유지
- SessionStore 기반 검증만 사용하여 멀티클라이언트 격리 보장

**session_id 필수화**:
- `session_id` 파라미터를 명시적으로 필수로 강제
- `session_id=None`일 때 명확한 에러 메시지 반환
- Breaking change이지만 ARK-165 멀티클라이언트 지원을 위해 필요

### 영향받는 컴포넌트

1. **use_notebook_tool.py**:
   - notebook_manager 검증 로직 제거
   - session_id 검증 추가 (Line 297-310)
   - kernel_id 파라미터 명시적 추가
   - kernel_id 덮어쓰기 버그 수정

2. **test_use_notebook_tool_ark165.py**:
   - 5개 기존 테스트에 session_id 추가
   - 5개 새로운 멀티클라이언트 시나리오 테스트 추가
   - 파일명 충돌 해결

3. **NotebookManager** (전역 객체):
   - 검증 로직에서 제거됨
   - `add_notebook()` 호출만 유지 (backward compatibility)
   - 향후 deprecation 예정

4. **SessionStore** (전역 객체):
   - 유일한 검증 및 상태 관리 시스템
   - 멀티클라이언트 세션 격리 담당

### 기술적 근거

**SessionStore 중심 설계의 이점**:
1. **세션 격리**: 각 클라이언트가 독립적인 상태 유지
2. **단순성**: 단일 상태 관리 시스템으로 복잡도 감소
3. **확장성**: 새로운 도구들도 SessionStore만 사용하면 됨
4. **일관성**: 모든 도구가 동일한 상태 관리 패턴 사용

**NotebookManager 제거의 이점**:
1. **충돌 제거**: notebook_name 기반 충돌 완전히 해결
2. **코드 간소화**: 중복 검증 로직 제거
3. **유지보수성**: 단일 진실의 원천(SessionStore)
4. **테스트 용이성**: 멀티클라이언트 시나리오 테스트 가능

### 주의사항

1. **Breaking Change**: 기존 코드에서 `session_id` 제공 필수
2. **NotebookManager 의존성**: 일부 레거시 도구가 여전히 NotebookManager 사용 중
3. **마이그레이션 필요**: 다른 도구들(execute_cell, read_cell 등)도 점진적으로 SessionStore로 전환 필요

### 다음 단계

1. **P1**: 다른 도구들도 SessionStore로 마이그레이션
   - execute_cell_tool.py
   - read_cell_tool.py
   - delete_cell_tool.py
   - insert_cell_tool.py
   - overwrite_cell_source_tool.py

2. **P2**: NotebookManager deprecation 계획
   - 모든 도구가 SessionStore로 전환된 후
   - NotebookManager를 deprecated로 마킹
   - 최종적으로 제거

3. **P3**: 문서 업데이트
   - ARCHITECTURE.md에 SessionStore 기반 설계 반영
   - NotebookManager deprecation 문서화

### 참고 자료

- 분석 리포트: `USE_NOTEBOOK_NOTEBOOKMANAGER_CONFLICT_REPORT.md`
- 실행 계획: `USE_NOTEBOOK_NOTEBOOKMANAGER_REMOVAL_PLAN.md`
- SessionStore 구현: `jupyter_mcp_server/session_store.py`
- 테스트: `tests/test_use_notebook_tool_ark165.py`

---

## 2025-01-18 - SessionManager 레이어 도입 및 커널 힐링 중앙화 (ARK-165)

### 배경

기존 시스템에서 커널 힐링 로직은 `use_notebook_tool.py`에만 존재했습니다:
- **문제점**: 약 100줄의 커널 검증 및 복구 로직이 중복되어 있음
- **불완전한 보호**: 5개 cell operation tool(execute_cell, read_cell, insert_cell, delete_cell, overwrite_cell_source)은 커널 힐링 없음
- **실패 시나리오**: 커널이 죽으면 cell operation이 실패하고 사용자가 수동으로 use_notebook을 다시 호출해야 함

### 문제 분석

**근본 원인**: 커널 힐링 로직이 중앙화되지 않음
- use_notebook: 커널 힐링 로직 존재 (Line 329-361, 417-475)
- 5개 cell operation tool: 커널 힐링 없음 → 커널 죽으면 실패

**실패 시나리오**:
```python
# 1. use_notebook으로 notebook 연결 (커널 생성)
use_notebook(session_id="A", notebook_path="work.ipynb")  # ✅ 성공

# 2. 커널이 외부 요인으로 종료됨
kernel_manager.shutdown_kernel(kernel_id)

# 3. execute_cell 시도
execute_cell(session_id="A", cell_index=0)  # ❌ 실패! "Kernel not found"

# 4. 사용자가 다시 use_notebook 호출해야 함
use_notebook(session_id="A", notebook_path="work.ipynb")  # 수동 복구 필요
```

### 결정 사항

**SessionManager 오케스트레이션 레이어 도입**:
- **관심사 분리**: SessionStore(저장소) vs SessionManager(오케스트레이션)
- **중앙화된 커널 힐링**: SessionManager.heal_kernel() 메서드에 모든 힐링 로직 집중
- **자동 힐링 활성화**: 모든 cell operation tool이 자동으로 커널 힐링 혜택을 받음

**Async 컨텍스트 함수 추가**:
- 기존 `get_notebook_context_from_session()` (sync, 힐링 없음) 유지
- 새로운 `get_notebook_context_from_session_async()` (async, auto_heal=True) 추가
- Backward compatibility 보장 (sync 함수 deprecated로 마킹)

**SessionStore 부분 업데이트 지원**:
- `update_notebook()` 메서드가 선택적 파라미터 지원
- 커널 힐링 시 kernel_id만 업데이트 (notebook_name/path 유지)
- 기존 코드 호환성 유지

### 아키텍처 레이어링

```
Tool Layer (6개 도구)
    ↓ get_notebook_context_from_session_async(auto_heal=True)
SessionManager (오케스트레이션)
    ↓ get/update operations
SessionStore (저장소)
```

**레이어별 책임**:
1. **Tool Layer**: 비즈니스 로직 (cell 실행, 읽기, 쓰기)
2. **SessionManager**: 커널 헬스 체크 및 자동 힐링
3. **SessionStore**: 순수 데이터 저장 (session_id → context 매핑)

### 영향받는 컴포넌트

**신규 파일**:
1. **session_manager.py** (SessionManager 클래스):
   - `get_session_with_kernel_check()`: 커널 헬스 체크
   - `heal_kernel()`: 커널 자동 복구
   - JUPYTER_SERVER/MCP_SERVER 모드 모두 지원

**수정 파일**:
2. **utils.py**:
   - `get_notebook_context_from_session_async()` 추가
   - 기존 sync 함수는 deprecated로 유지

3. **use_notebook_tool.py** (~96줄 제거):
   - 커널 검증 로직 제거 (Line 329-361)
   - 커널 힐링 로직 제거 (Line 417-475)
   - SessionManager 위임으로 대체

4. **5개 cell operation tool**:
   - execute_cell_tool.py
   - read_cell_tool.py
   - insert_cell_tool.py
   - delete_cell_tool.py
   - overwrite_cell_source_tool.py
   - 모두 async 컨텍스트 함수로 업데이트

5. **session_store.py**:
   - `update_notebook()` 메서드 시그니처 변경
   - 선택적 파라미터 지원 (부분 업데이트)

### 기술적 근거

**SessionManager 도입의 이점**:
1. **코드 중복 제거**: use_notebook_tool에서 ~96줄 제거
2. **자동 복구**: 모든 cell operation이 자동으로 커널 힐링 혜택
3. **관심사 분리**: 저장소(SessionStore) vs 오케스트레이션(SessionManager)
4. **테스트 용이성**: 단위/통합 테스트 작성 쉬움
5. **확장성**: 새로운 tool도 자동으로 힐링 혜택

**Async 함수 도입의 이점**:
1. **명시적 의도**: auto_heal 파라미터로 힐링 동작 명확히
2. **Backward Compatibility**: sync 함수 유지하여 기존 코드 보호
3. **권장 패턴**: async 함수를 새로운 표준으로 설정

**설계 결정 사항**:

**Q: 왜 SessionStore에 직접 커널 체크를 넣지 않고 SessionManager를 만드는가?**

A: **관심사 분리 (Separation of Concerns)**
- SessionStore: 순수한 데이터 저장소 (lightweight, 234 bytes/session)
- SessionManager: 비즈니스 로직 (커널 관리, 힐링)
- SessionStore는 단순하고 테스트하기 쉬움
- SessionManager는 kernel_manager 의존성을 가져도 OK

**Q: auto_heal 기본값을 True로 하는 이유?**

A: **사용자 경험 향상**
- 대부분의 경우 커널 자동 복구가 바람직함
- 사용자가 수동으로 use_notebook 다시 호출할 필요 없음
- 필요시 auto_heal=False로 기존 동작 유지 가능

**Q: 성능 문제는 없나?**

A: **커널 체크는 매우 가벼운 연산**
- JUPYTER_SERVER: O(1) dict lookup (~1ms)
- MCP_SERVER: 캐싱된 HTTP 요청 (~10ms)
- 실제 오버헤드 < 10ms
- 필요시 SessionManager에 TTL 캐싱 추가 가능

### 테스트 전략

**60개 테스트 작성 및 통과**:

1. **SessionManager 단위 테스트** (14개):
   - 커널 헬스 체크 (JUPYTER_SERVER + MCP_SERVER)
   - 커널 힐링 성공 시나리오
   - 에러 핸들링 및 실패 케이스
   - SessionStore 부분 업데이트

2. **Utils 통합 테스트** (16개):
   - Sync 함수 backward compatibility (3개)
   - Async 함수 자동 힐링 (6개)
   - Config fallback 시나리오 (7개)

3. **use_notebook 회귀 테스트** (19개):
   - 기존 기능 회귀 없음 검증
   - 커널 자동 복구 워크플로우
   - 멀티클라이언트 세션 시나리오

4. **Cell Tool 통합 테스트** (11개):
   - 5개 cell tool별 힐링 동작 검증 (6개)
   - 코드 통합 검증 (5개)

**Test Results**: 60/60 통과 (100%)

### 주의사항

1. **Breaking Change 없음**: 모든 기존 API 호환성 유지
2. **Deprecation**: sync 함수는 deprecated, async 함수 사용 권장
3. **성능**: 커널 체크 오버헤드 < 10ms (무시 가능)
4. **확장성**: 새로운 tool 추가 시 자동으로 힐링 혜택

### 마이그레이션 가이드

**Cell Operation Tool 업데이트 패턴**:

**변경 전:**
```python
from jupyter_mcp_server.utils import get_notebook_context_from_session
notebook_path, kernel_id = get_notebook_context_from_session(
    session_id=session_id
)
```

**변경 후:**
```python
from jupyter_mcp_server.utils import get_notebook_context_from_session_async
notebook_path, kernel_id = await get_notebook_context_from_session_async(
    session_id=session_id,
    auto_heal=True,
    kernel_manager=kernel_manager,
    mode=mode
)
```

### 향후 계획

1. **완료**: 모든 cell operation tool에 커널 힐링 적용 ✅
2. **완료**: ARCHITECTURE.md 업데이트 ✅
3. **완료**: CHANGELOG.md 업데이트 ✅
4. **완료**: Serena memory 업데이트 ✅

### 참고 자료

- 실행 계획: `SESSIONMANAGER_KERNEL_HEALING_REFACTOR_PLAN.md`
- SessionManager 구현: `jupyter_mcp_server/session_manager.py`
- Async 컨텍스트 함수: `jupyter_mcp_server/utils.py`
- 테스트:
  - `tests/test_session_manager.py` (14 tests)
  - `tests/test_utils_session.py` (16 tests)
  - `tests/test_use_notebook_tool_ark165.py` (19 tests)
  - `tests/test_cell_tools_kernel_healing.py` (11 tests)
