# 📦 Automatic1111 Collections Extension

## 🎯 Project Overview

This extension adds a **local-first Civitai collections browser** directly inside AUTOMATIC1111 Stable Diffusion WebUI.

The goal is to create a **fully offline-capable reference system** that allows users to:

- Browse synced Civitai collections
- View full generation metadata
- Reconstruct images with correct parameters
- Track required resources (LoRAs, checkpoints, etc.)
- Build a personal, local reference library

---

## 🧠 Core Philosophy

- **Local-first** → No dependency on Civitai after sync  
- **Clean UI** → Pinterest-style browsing, minimal clutter  
- **Accurate reconstruction** → Preserve generation data exactly  
- **A1111-native** → Built using official extension hooks and Gradio  

---

## 🏗️ Current Feature Status

### ✅ Backend (Stable)

#### Image Data Extraction
- Prompt / Negative Prompt
- Steps / CFG / Sampler / Seed
- Resolution (Width / Height)
- Generator (A1111 / ComfyUI)

#### Resource Tracking
- Checkpoints
- LoRAs (with weights)
- Embeddings
- Upscalers
- Model + Version IDs

#### Database
- `items` table ✅
- `resources` table ✅
- `generation_params` table ✅
- Clean normalized schema

---

### ✅ Sync System

- Pulls collections from Civitai API
- Downloads:
  - Preview images
  - Full-resolution media
  - Video / GIF support
- Stores everything locally

---

### ✅ Existing UI Components

- Collection tab registered in A1111
- Sidebar with synced collections
- Feed rendering system
- NSFW filter system:
  - Toggle button
  - Settings-based filtering (R / X / XXX)
- Custom toolbar button system (SVG + CSS + Gradio state)

---

## 🚧 Current Focus (IN PROGRESS)

### 🟢 UI View System (PRIMARY PHASE)

We are building a **3-mode browsing system**:

---

### 1️⃣ Regular View (Default)

- Sidebar (collections)
- Masonry grid (Pinterest-style)
- Adjustable preview size
- Scrollable feed

---

### 2️⃣ Scrolling View (Immersive)

- Full-width vertical feed
- No sidebar
- Minimal UI
- Large image display

---

### 3️⃣ Detailed View (Image Inspector)

- Large image on left
- Full metadata panel on right:
  - Creator / links
  - Prompt / negative prompt
  - Generation settings
  - Resources used

**Future:**
- 🟢 / 🔴 indicators for local resource availability

---

## 🎮 Control System (Top Bar)

### View Switching Buttons
- Grid → Regular View
- Scroll → Scrolling View
- Focus → Detailed View

### Preview Scaling
- Slider-based control
- Affects:
  - Grid density
  - Image size

---

## 🧱 UI Architecture

```text
Collection Tab
├── Top Control Bar
│   ├── View Buttons
│   ├── Preview Slider
│
├── Main Container
│   ├── Sidebar
│   ├── Content Area
│       ├── Grid View
│       ├── Scrolling View
│       ├── Detailed View

🧠 UX Rules (LOCKED)
Minimalist design (no clutter)
Consistent controls across views
Persistent state:
Scroll position
Selected image
No aggressive animations


🧪 Development Timeline
✅ Phase 0 — Foundation (COMPLETE)
Extension registration
Settings integration
Data pipeline
Database schema
Media download system
✅ Phase 1 — Baseline UI (COMPLETE)
Collection sidebar
Feed rendering
NSFW filter system
Button architecture established
🔄 Phase 2 — View System Core (CURRENT)
View switching logic
Layout containers
Toolbar system
State handling between views
⏳ Phase 3 — View Implementation
Regular view polish
Scrolling view build
Detailed view layout + binding
⏳ Phase 4 — Navigation
Left / right image navigation
State persistence
⏳ Phase 5 — Resource Matching
🟢 Available locally
🔴 Missing resources
Applies to:
LoRAs
Checkpoints
Embeddings
Upscalers
⏳ Phase 6 — Reconstruction Tools
Copy generation data
Send to txt2img
Auto-apply LoRAs


🧩 Technical Structure
collection/
├── assets/
│   └── icons/
├── collection_lib/
│   ├── civitai_api.py
│   ├── database.py
│
├── data/
│   ├── collections.db
│   └── images/
│
├── scripts/
│   └── collection_tab.py

⚙️ Development Rules
✔ Use Gradio for behavior
button.click(fn=..., inputs=[...], outputs=[...])
✔ Use A1111 callbacks
script_callbacks.on_ui_tabs(...)
script_callbacks.on_ui_settings(...)
✔ Use CSS for visuals only
✔ Use elem_id for styling hooks

🚫 What to Avoid
DOM click hacks (querySelector().click())
Inline SVG inside gr.Button(value=...)
Mixing multiple experimental changes at once
Incorrect CSS targeting
Broken f-string CSS braces

🟢 Current Status Summary
Backend: Complete and stable
Sync system: Working
UI: Transitioning to multi-view system
Buttons: Standardized and ready for expansion


🔜 Next Immediate Task

Build view switching system + layout scaffolding

This includes:

Toolbar buttons (grid / scroll / detailed)
View state management
Container switching logic
📌 Final Note

This project is designed to:

Stay aligned with A1111 architecture
Remain stable and maintainable
Provide a clean, professional browsing experience
Enable full offline reconstruction workflows
