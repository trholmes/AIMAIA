# MAIA LCIO Event Viewer

This repo contains two viewers for LCIO (`.slcio`) MAIA events:

- `maia_event_viewer_streamlit.py` (recommended): browser-based, works well on remote/Apptainer
- `maia_event_viewer.py`: local Tk desktop GUI (requires `tkinter`)

Both rely on `pylcio` for reading LCIO files.

## Run Again (Quick Routine)

### Remote server (inside Apptainer)

```bash
cd /scratch/trholmes/mucol/mucolstudies/AIMAIA
apptainer shell <your-image>.sif
export PYUSER_SITE="$(python3 -m site --user-site)"
export PYTHONPATH="$PYUSER_SITE:$PYTHONPATH"
python3 -m streamlit run maia_event_viewer_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

Leave that terminal running.

### Local laptop (for viewing)

```bash
ssh -N -L 8501:127.0.0.1:8501 trholmes@ap23.uc.osg-htc.org
```

Open:

- http://localhost:8501

## Recommended Path: Remote + Apptainer + Browser

### 1. SSH to remote host

```bash
ssh trholmes@ap23.uc.osg-htc.org
```

### 2. Clone repo (first time only)

```bash
git clone <your-repo-url>
cd AIMAIA
```

### 3. Start container

If you already have the image locally, use it. Otherwise pull first.

```bash
apptainer shell <your-mucoll-image>.sif
```

### 4. Verify `pylcio`

```bash
python3 -c "import pylcio; print('pylcio OK')"
```

### 5. Install browser dependencies (user-local)

```bash
python3 -m pip install --user streamlit plotly
```

### 6. Ensure user site-packages are first on import path

This avoids protobuf conflicts between user packages and container base packages.

```bash
export PYUSER_SITE="$(python3 -m site --user-site)"
export PYTHONPATH="$PYUSER_SITE:$PYTHONPATH"
python3 -c "import google.protobuf; print(google.protobuf.__file__, google.protobuf.__version__)"
```

### 7. Run the app on the remote host

```bash
cd /path/to/AIMAIA
python3 -m streamlit run maia_event_viewer_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

Leave this running.

### 8. From your laptop, open SSH tunnel

In a new local terminal:

```bash
ssh -N -L 8501:127.0.0.1:8501 trholmes@ap23.uc.osg-htc.org
```

### 9. Open browser on laptop

- http://localhost:8501

### 10. Use the app

In the sidebar:
- Set full `.slcio` path (example: `/data/.../neutronGun_E_0_50_reco_0.slcio`)
- Choose event index
- Select collections
- Tune energy threshold / point limits

## Quick Tunnel Test

Before running Streamlit, you can test forwarding:

On remote host:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

On laptop:

```bash
ssh -N -L 8765:127.0.0.1:8765 trholmes@ap23.uc.osg-htc.org
```

Then open http://localhost:8765

## Troubleshooting

### `ModuleNotFoundError: No module named '_tkinter'`

Use browser viewer (`maia_event_viewer_streamlit.py`), not Tk viewer.

### `ImportError ... google.protobuf.internal ... builder`

You have mixed protobuf versions on `PYTHONPATH`. Run:

```bash
export PYUSER_SITE="$(python3 -m site --user-site)"
export PYTHONPATH="$PYUSER_SITE:$PYTHONPATH"
```

Then restart Streamlit.

### Browser says cannot connect to `localhost:8501`

1. Confirm app is up on remote:
```bash
curl -I http://127.0.0.1:8501
```
2. Confirm tunnel command is running on laptop:
```bash
ssh -N -L 8501:127.0.0.1:8501 trholmes@ap23.uc.osg-htc.org
```

### `streamlit` command not found

Use module form:

```bash
python3 -m streamlit run maia_event_viewer_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

## Optional: Tk Desktop Viewer

If you are on a local machine with `tkinter` + `pylcio` available:

```bash
python3 maia_event_viewer.py /path/to/file.slcio
```

Headless summary mode:

```bash
python3 maia_event_viewer.py /path/to/file.slcio --summary-event 0
```
