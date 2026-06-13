# AI_USE_LOG.md
## HydroSense-Kenya — Level 6: AI-Assisted Scientific Programming

**AI Tool Used:** Claude (Anthropic) — claude.ai  
**Purpose:** Generating, reviewing, and iterating on test cases for numerical methods  
**Student Role:** Reviewed all AI output, identified gaps, requested corrections, and validated final tests against the actual source code in `src/numerical_methods.py`

---

## Task 1 — Generating Initial Test Cases

### Prompt
> "I need you to help me come up with tests for the root finding methods and linear methods. Not both methods are in different .py files so separate them. Don't test the edge cases just yet. Just come up with tests."

### AI Output
Claude generated two separate test files:
- `test_root_finding.py` — covering `bisection_method`, `newton_raphson_method`, and `secant_method`
- `test_linear_solver.py` — covering `gaussian_elimination`

Each method was tested with multiple known mathematical functions (quadratic, cubic, trigonometric, exponential) with analytically verifiable roots. Tests checked:
- Root accuracy against known values (e.g. root of x²−4 is exactly 2.0)
- Residual check: f(root) ≈ 0
- Return type: confirmed each method returns a `(root, iterations)` tuple
- Iteration count: confirmed `iterations >= 1`

### Modifications Made
- Import paths were initially wrong (`from numerical import ...`) — corrected in the next prompt session
- File naming did not match the project structure (`test_linear_solver.py` vs the required `test_linear_systems.py`) — corrected after reviewing the folder structure screenshot

---

## Task 2 — Fixing Import Paths to Match Project Structure

### Prompt
> "This is the project structure. The methods are in the numerical_methods.py. Ensure to import them correctly."  
> *(Screenshot of HydroSense-Kenya/ folder tree provided)*

### AI Output
Claude updated both test files with the correct relative import block:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from numerical_methods import bisection_method, newton_raphson_method, secant_method
```

Also renamed `test_linear_solver.py` → `test_linear_systems.py` to match the actual `tests/` folder in the project.

### Validation
Import was verified to resolve correctly by tracing the path: `tests/` → `../src/` → `numerical_methods.py`. Tests ran without `ModuleNotFoundError`.

---

## Task 3 — Clarifying AI Output (Helper Functions vs Redefinitions)

### Prompt
> "Tests pass however you have defined all functions again in the test files. You are supposed to import the functions and then test them."

### AI Clarification
Claude explained that the functions defined in the test file (`f_quadratic`, `f_cubic`, `f_trig`, `f_exp` and their derivatives) are **not redefinitions** of anything in `numerical_methods.py`. They are mathematical input functions passed as arguments into the solvers, for example:

```python
bisection_method(f_quadratic, a=0, b=3)
```

This is standard pytest practice — the solver functions are imported from `numerical_methods.py`; the test equations must be defined locally as inputs.

### Student Decision
After reviewing the explanation, the student confirmed the structure was correct and no changes were needed.

---

## Task 4 — Identifying Missing Test Cases

### Prompt
> "OK are there any test cases I need to consider that aren't defined in this code?"

### AI Output
Claude reviewed the existing tests and identified the following gaps:

**Root Finding:**
- No test for a root at the exact midpoint of the bracket (bisection)
- No test for a root close to one endpoint
- No test that `ValueError` is raised when `f(a)` and `f(b)` have the same sign (bisection)
- No test that `ZeroDivisionError` is raised when the derivative is zero at `x0` (Newton-Raphson)
- No test with a negative `x0` (Newton-Raphson)
- No test that `ZeroDivisionError` is raised when `f(x0) == f(x1)` (secant)
- No cross-method consistency tests (do all three agree on the same root?)

**Linear Systems:**
- No test for a singular matrix raising `ValueError`
- No test for a 1×1 system
- No test with negative values in `b`
- No test with floating-point (non-integer) coefficients in `A`
- No test verifying row-order independence (partial pivoting should handle any ordering)

### Modifications Made
None at this stage — gaps were identified and confirmed before implementation.

---

## Task 5 — Adding All Missing Test Cases

### Prompt
> "Let's modify both."

### AI Output
Both test files were fully rewritten to incorporate all identified gaps. New tests added:

**`test_root_finding.py`:**

| Test | Method | What It Covers |
|---|---|---|
| `test_root_near_endpoint` | Bisection | Root at x=2 with bracket [1.9, 3] |
| `test_root_at_midpoint` | Bisection | f(x)=x on [−1,1], root is exact midpoint |
| `test_raises_when_same_sign_at_endpoints` | Bisection | `ValueError` when f(a) and f(b) have same sign |
| `test_quadratic_root_negative_x0` | Newton-Raphson | Negative starting point converges correctly |
| `test_raises_when_derivative_is_zero` | Newton-Raphson | `ZeroDivisionError` when f'(x0)=0 |
| `test_raises_when_secant_line_is_flat` | Secant | `ZeroDivisionError` when f(x0)==f(x1) |
| `test_agrees_with_newton_raphson` | Bisection | Cross-method consistency check |
| `test_agrees_with_secant` | Bisection | Cross-method consistency check |
| `test_agrees_with_secant` | Newton-Raphson | Cross-method consistency check |
| `test_agrees_with_bisection` | Secant | Cross-method consistency check |

**`test_linear_systems.py`:**

| Test | What It Covers |
|---|---|
| `test_1x1_system` | Smallest valid input: [[5]], [10] → x=2 |
| `test_negative_rhs` | All-negative values in b |
| `test_float_coefficients` | Non-integer coefficients in A |
| `test_row_order_independence` | Swapped rows yield the same solution (validates pivoting) |
| `test_raises_for_singular_matrix` | `ValueError` for a singular matrix (row 2 = 2 × row 1) |

### Validation Method
All tests were designed around functions with known analytical solutions so expected values could be independently verified by hand or with a calculator. Residual checks (`f(root) ≈ 0`, `Ax ≈ b`) were used as a secondary validation layer independent of hardcoded expected values.

---

## Summary

| Task | AI Role | Student Role |
|---|---|---|
| Initial test generation | Generated full test structure for 4 methods across 2 files | Reviewed output, identified import and naming errors |
| Import path fix | Corrected imports and file name | Verified path resolution against actual folder structure |
| Helper function clarification | Explained why local functions are necessary in test files | Confirmed understanding, no changes needed |
| Gap identification | Audited existing tests and listed missing cases | Confirmed gaps before proceeding |
| Adding missing tests | Rewrote both files with all new cases | Reviewed each new test for correctness and relevance |

**Total test count after all iterations:**
- `test_root_finding.py`: 22 tests across 3 classes
- `test_linear_systems.py`: 13 tests in 1 class