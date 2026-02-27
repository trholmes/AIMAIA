# MAIA LCIO 3D Event Viewer

This repo now includes a standalone GUI event viewer for MAIA `.slcio` files:

- Script: `maia_event_viewer.py`
- Browser script: `maia_event_viewer_streamlit.py`
- Input: LCIO `.slcio`
- Backend: `pylcio` (as provided in MuonCollider container environments)
- GUI options:
  - `tkinter` + `matplotlib` 3D (desktop)
  - `streamlit` + `plotly` 3D (browser/remote-friendly)

## Run

```bash
cd /Users/tovaholmes/cernbox/Work/MuonCollider/Studies/10TeV/AIMAIA
python3 maia_event_viewer.py /Users/tovaholmes/Downloads/neutronGun_E_0_50_reco_0.slcio
```

Or launch without a file and use **Browse**:

```bash
python3 maia_event_viewer.py
```

If your container Python does not include `tkinter`, use headless summary mode:

```bash
python3 maia_event_viewer.py /path/to/file.slcio --summary-event 0
```

## Browser Mode (Recommended for Remote/Apptainer)

Run on remote host (inside your container/env with `pylcio`, `streamlit`, and `plotly`):

```bash
cd /path/to/AIMAIA
streamlit run maia_event_viewer_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

From your laptop, open an SSH tunnel:

```bash
ssh -L 8501:localhost:8501 <user>@<remote-host>
```

Then open:

- http://localhost:8501

## Features

- Load `.slcio` files from a file picker or CLI argument
- Event slider + event index input
- Collection multi-select (select all / clear)
- 3D scatter of hit-like collections (`getPosition`, `getVertex`, `getReferencePoint`)
- Optional track/MC segment drawing (`getVertex` -> `getEndpoint`)
- Display controls: minimum energy threshold, max points per collection, point size
- Status line with collection-level counts and energy sums

## Notes

- The viewer requires `pylcio`.
- If `pylcio` is not importable, the GUI shows a clear load error.
- The GUI requires `tkinter` support in Python.
- Some collections may not expose coordinates in the LCIO object API; those are skipped.
