# MAIA LCIO 3D Event Viewer

This repo now includes a standalone GUI event viewer for MAIA `.slcio` files:

- Script: `maia_event_viewer.py`
- Input: LCIO `.slcio`
- Backend: `pylcio` (as provided in MuonCollider container environments)
- GUI: `tkinter` + `matplotlib` 3D

## Run

```bash
cd /Users/tovaholmes/cernbox/Work/MuonCollider/Studies/10TeV/AIMAIA
python3 maia_event_viewer.py /Users/tovaholmes/Downloads/neutronGun_E_0_50_reco_0.slcio
```

Or launch without a file and use **Browse**:

```bash
python3 maia_event_viewer.py
```

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
- Some collections may not expose coordinates in the LCIO object API; those are skipped.
