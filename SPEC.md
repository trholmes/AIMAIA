# MAIA LCIO Event Viewer Specification

## 1. Document Purpose
This document defines the functional and operational specification for the Streamlit MAIA LCIO event viewer in this repository, derived from `README.md` and aligned with current code behavior.

## 2. Scope
The project provides a browser interface for interactive inspection of MAIA LCIO (`.slcio`) events:
- Browser viewer: `maia_event_viewer_streamlit.py`

The interface relies on LCIO Python bindings (`pylcio` or `pyLCIO`) to read event data.

## 3. Goals
- Support practical event-by-event visual inspection of MAIA detector data.
- Work reliably on remote/containerized environments where desktop GUI support may be unavailable.
- Provide collection-level filtering, thresholding, and summary information.
- Offer a lightweight workflow for quick debugging of file contents.

## 4. Non-Goals
- Physics reconstruction, fitting, or calibration.
- Batch production workflows or distributed processing.
- Automated report generation beyond on-screen summaries.
- Desktop GUI support and Tkinter-specific workflows.

## 5. System Context and Assumptions
- Input files are LCIO `.slcio` files readable by `pylcio`/`pyLCIO`.
- Typical deployment is remote Apptainer execution with local browser access via SSH tunnel.
- Streamlit + Plotly are required for the browser viewer.

## 6. User Personas
- Detector/analysis developer validating event content and collection behavior.
- Remote user running inside an HPC or Apptainer environment without native desktop forwarding.

## 7. Functional Requirements

### 7.1 Common Reader Behavior
1. The system shall attempt LCIO import in this order: `pyLCIO`, then `pylcio`.
2. The system shall provide clear runtime errors when LCIO bindings are unavailable.
3. The system shall expose event collections for selected events.

### 7.2 Input Selection
1. The browser viewer shall accept either:
   - an explicit file path, or
   - a glob pattern.
2. If a glob expands to multiple files, the browser viewer shall load exactly one file selected by index.
3. If no file matches, the browser viewer shall show an error and stop execution for that run.

### 7.3 Event Navigation
1. The browser viewer shall determine event count before event selection.
2. The browser viewer shall provide event selection controls (fixed event 0 for single-event files; slider otherwise).

### 7.4 Collection Visibility and Filtering
1. The browser viewer shall hide relation-like collections from default visible options.
2. The browser viewer shall offer point collection multiselect with sensible defaults favoring hit-like collections.
3. The browser viewer shall offer line collection multiselect when line display is enabled.
4. The browser viewer shall include `SiTracks`, `SiTracksRefitted`, and `SelectedTracks` in line defaults when present.
5. The browser viewer shall support Pandora PFO PDGID filtering when PFO line collections are selected.

### 7.5 Rendering Controls
1. The browser viewer shall provide controls for:
   - minimum energy threshold,
   - max points per collection,
   - base point size,
   - max lines per collection,
   - track segment length,
   - detector wireframe visibility.
2. The browser viewer shall support resetting camera orientation and quick zoom presets (tracker/calorimeter).
3. Initial camera orientation shall place the z-axis horizontally on screen.

### 7.6 Detector Context
1. The browser viewer shall support optional MAIA detector wireframe overlays.
2. Detector dimensions shall be represented in mm using embedded component definitions.

### 7.7 Summaries and Diagnostics
1. The browser viewer shall display per-collection summary metrics:
   - point count,
   - track count,
   - energy sum.
2. The browser viewer shall provide a collapsed debug panel listing collection names/types.

### 7.8 Error Handling
1. Missing input file(s), unreadable files, and out-of-range event access shall produce explicit user-facing errors.

## 8. Operational Requirements

### 8.1 Recommended Runtime Path
- Run Streamlit app on remote host bound to `127.0.0.1:8501`.
- Use local SSH port forwarding `localhost:8501 -> remote 127.0.0.1:8501`.
- Open browser at `http://localhost:8501`.

### 8.2 Python Path Compatibility
- For Apptainer/user-local package setups, `PYUSER_SITE` should be prepended via `PYTHONPATH` to avoid protobuf import conflicts.

## 9. CLI Interfaces
- Browser viewer:
  - `python3 -m streamlit run maia_event_viewer_streamlit.py --server.address 127.0.0.1 --server.port 8501`

## 10. Acceptance Criteria
1. Given a valid LCIO file path, the browser viewer loads, displays event controls, and renders at least one selected collection.
2. Given a matching glob with multiple files, the selected index determines the loaded file and no other file is read for rendering.
3. Toggling `Show detector wireframe` visibly adds/removes detector boundary traces.
4. Selecting `PandoraPFOs` as a line collection enables PDGID filtering and affects rendered line content.
5. `Reset 3D view` restores the default camera orientation.
6. The app displays collection summary rows (`points`, `tracks`, `energy_sum_GeV`) for rendered selections.
7. The `Collection debug` panel reports all-event collection visibility/type diagnostics.

## 11. Known Constraints and Risks
- LCIO Python binding API differences are handled heuristically and may require updates for new distributions.
- Large events can stress rendering performance; limits (`max points`, `max lines`) are required safety controls.
- Full-file iteration for event counting can be expensive on very large files.

## 12. Future Enhancements (Out of Current Scope)
- Lazy event indexing for faster initial load on large files.
- Persisted user presets for collection and view settings.
- Export of selected event snapshots and summary tables.
- Optional CLI-only batch summary mode for many files.
