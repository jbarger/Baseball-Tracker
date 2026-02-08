# Baseball Tracker — Bite-Sized Task Breakdown

Each task is small, independently verifiable, and builds on the previous one.
**Verification = a concrete test you can run to prove it works.**

---

## Phase 1: Stabilize What Exists (Get Green Across the Board)

### Task 1.1 — Commit & clean up in-progress Python fixes
- **What**: Commit the circular import fixes, Dockerfile cleanup, and stripped docstrings already in your working tree
- **Verify**: `git status` shows clean working directory

### Task 1.2 — Python CV service starts and passes health check
- **What**: `docker compose up python-cv` → service starts, no errors
- **Verify**: `curl http://localhost:8000/health` returns `{"status": "healthy"}`

### Task 1.3 — .NET API service starts and passes health check
- **What**: `docker compose up api` → service starts, connects to SQLite
- **Verify**: `curl http://localhost:5000/health` returns OK

### Task 1.4 — End-to-end stub flow works
- **What**: POST a fake video path to `/api/swings/process` → .NET calls Python → gets mock result → stores in SQLite → returns to caller
- **Verify**: `curl -X POST http://localhost:5000/api/swings/process -H "Content-Type: application/json" -d '{"videoPath":"test.mp4"}'` returns swing data with mock metrics

### Task 1.5 — Add basic automated tests
- **What**: Add pytest tests for Python endpoints (health, ball track, bat track with mock) + xUnit tests for .NET API (health, swing CRUD)
- **Verify**: `pytest` passes, `dotnet test` passes

---

## Phase 2: Real Video Ingestion (Accept & Store Real Videos)

### Task 2.1 — Video upload endpoint
- **What**: Add `POST /api/videos/upload` that accepts multipart file upload, saves to `/data/videos/`, returns a video ID
- **Verify**: Upload a .mp4 file via curl/Postman → file appears in `/data/videos/` → get back a video ID

### Task 2.2 — Video metadata storage
- **What**: Add `Video` model to Core (id, filename, path, uploadedAt, durationMs, cameraAngle) + EF migration + repository
- **Verify**: Upload a video → query `GET /api/videos/{id}` → get metadata back

### Task 2.3 — Multi-camera video linking
- **What**: Allow linking 2+ videos to the same swing (ball cam + swing cam). Add `SwingVideo` join table
- **Verify**: Upload 2 videos, link both to same swing → query swing → see both video references

---

## Phase 3: Real Ball Tracking (Replace Stubs with OpenCV + YOLO)

### Task 3.1 — OpenCV video frame extraction
- **What**: In Python, read a real .mp4 and extract frames as numpy arrays. Add `common/video_utils.py`
- **Verify**: Python script reads a test video → prints frame count + resolution → no errors

### Task 3.2 — YOLO ball detection on single frame
- **What**: Load YOLOv8 model, run inference on a single frame, detect baseball location (bounding box)
- **Verify**: Script outputs bounding box coordinates for a frame containing a baseball

### Task 3.3 — Frame-to-frame ball tracking
- **What**: Run YOLO across all frames, connect detections into a trajectory (list of x,y positions per frame)
- **Verify**: Script outputs a list of (frame#, x, y) positions showing ball movement

### Task 3.4 — Exit velocity calculation
- **What**: Given trajectory + camera calibration constants (hardcoded for now), calculate speed in mph
- **Verify**: Known test video → calculated velocity is within ±10% of expected value

### Task 3.5 — Launch angle calculation
- **What**: From trajectory, calculate vertical launch angle off the bat
- **Verify**: Known test video → calculated angle is within ±5° of expected

### Task 3.6 — Wire real ball tracker into API
- **What**: Replace stub `BallTracker.process_video()` with real implementation
- **Verify**: `POST /track/ball` with a real video → returns real (not random) metrics

---

## Phase 4: Real Bat Tracking

### Task 4.1 — Bat detection on single frame
- **What**: Detect bat in frame (may need custom YOLO training or color/edge detection)
- **Verify**: Script outputs bounding box for bat in a test frame

### Task 4.2 — Bat swing trajectory
- **What**: Track bat barrel position across frames through the swing zone
- **Verify**: Script outputs (frame#, x, y) for bat barrel through swing

### Task 4.3 — Bat speed calculation
- **What**: Calculate bat speed from trajectory + calibration
- **Verify**: Known test video → bat speed within reasonable range (50-90 mph)

### Task 4.4 — Wire real bat tracker into API
- **What**: Replace stub `BatTracker.process_video()` with real implementation
- **Verify**: `POST /track/bat` with real video → returns real metrics

---

## Phase 5: Blazor Web UI — Core Pages

### Task 5.1 — Blazor project scaffolding
- **What**: Create `BaseballTracker.Web` Blazor Server project, wire up DI, add to docker-compose
- **Verify**: Navigate to `http://localhost:5001` → see a homepage with "Baseball Tracker" title

### Task 5.2 — Session list page
- **What**: Page showing all sessions (date, player, # swings). Calls API.
- **Verify**: Create sessions via API → they appear in the web UI list

### Task 5.3 — New session creation page
- **What**: Form to create a new session (player name, date, location)
- **Verify**: Fill form → submit → session appears in list

### Task 5.4 — Video upload page
- **What**: Upload videos for a session, assign camera angle (ball cam / swing cam)
- **Verify**: Upload .mp4 → file stored → video linked to session

### Task 5.5 — Process swing button
- **What**: Button that triggers CV processing for uploaded videos. Shows processing status via SignalR
- **Verify**: Click "Process" → status changes Captured → Processing → Completed → metrics appear

### Task 5.6 — Swing results display
- **What**: Show exit velocity, launch angle, bat speed, spray angle for each swing in session
- **Verify**: After processing → metrics display correctly on the page

---

## Phase 6: Charts & Trends

### Task 6.1 — Session summary chart
- **What**: Chart.js bar chart showing all swings in a session (exit velo, launch angle)
- **Verify**: Session with 5+ swings → chart renders with correct values

### Task 6.2 — Player trend chart
- **What**: Line chart showing metrics over time across sessions
- **Verify**: Player with multiple sessions → chart shows trend lines

### Task 6.3 — Spray chart visualization
- **What**: Overhead baseball diamond view showing where balls are hit
- **Verify**: Swings with spray angles → dots appear on diamond at correct locations

---

## Phase 7: Polish & Advanced

### Task 7.1 — Video playback with overlay
### Task 7.2 — Pose estimation (MediaPipe)
### Task 7.3 — Export to CSV/PDF
### Task 7.4 — Multi-user authentication
### Task 7.5 — Cloud deployment (Azure/AWS)

---

## Quick Reference: Task Size Guide

| Phase | # Tasks | Estimated Effort Each | Total |
|-------|---------|----------------------|-------|
| 1. Stabilize | 5 | 30 min - 1 hr | ~3 hrs |
| 2. Video Ingestion | 3 | 1 - 2 hrs | ~4 hrs |
| 3. Ball Tracking | 6 | 2 - 4 hrs | ~18 hrs |
| 4. Bat Tracking | 4 | 2 - 4 hrs | ~12 hrs |
| 5. Web UI | 6 | 1 - 3 hrs | ~12 hrs |
| 6. Charts | 3 | 1 - 2 hrs | ~4 hrs |
| 7. Advanced | 5 | 4 - 8 hrs | ~30 hrs |

**Total estimated: ~80 hours to a fully functional system through Phase 6**
