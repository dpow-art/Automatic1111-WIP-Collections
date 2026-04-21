A1111 Collections

A1111 Collections is an extension for AUTOMATIC1111 Stable Diffusion WebUI that creates a local, offline-capable library of Civitai images and their full generation data.

It allows users to browse, organize, and reuse generation setups without needing to constantly return to external websites.

---

✨ Core Idea

«Build a local-first reference library for Stable Diffusion workflows.»

Instead of:

- jumping between browser tabs
- manually copying prompts
- losing LoRA setups

You get:

- a clean visual browser
- complete generation data
- one-click reuse inside A1111

---

🚀 Features

🔄 Civitai Sync

- Sync images using API key
- Supports:
  - Auto
  - SFW (".com")
  - Full (".red")
- Stores data locally

---

🗂️ Local Library

- SQLite-backed database
- Offline browsing
- Cached image previews

---

🧩 Collections System

- Synced collections (Civitai)
- Local collections (user-created)
- Add images to custom collections

---

🧠 LoRA Awareness

- Detects LoRAs used in images
- Status indicators:
  - 🟢 Installed
  - 🟡 Possible match
  - 🔴 Missing

---

🎯 Send to txt2img

- Transfers:
  - prompt
  - negative prompt
  - LoRAs with weights
  - core generation settings

---

🔞 Content Filtering

- Toggle to hide:
  - R / X / XXX content
- Works offline

---

📊 Sorting

- Custom order
- Rating (G → XXX)
- Newest / Oldest

---

🖼️ Visual Browser (Planned)

- Pinterest-style masonry layout
- Adjustable thumbnail size
- Smooth scrolling

---

🎬 Focus Mode (Planned)

- Click image → expand to center
- Right panel slides in
- Clean viewing mode
- Arrow navigation

---

📦 Installation

Method 1: Git Clone (Recommended)

cd stable-diffusion-webui/extensions
git clone https://github.com/dpow-art/A1111-Collections.git

Restart A1111.

---

Method 2: Download ZIP

1. Download this repo as ZIP
2. Extract into:

stable-diffusion-webui/extensions/A1111-Collections

3. Restart A1111

---

⚙️ Setup

Go to:

Settings → Collection

Set:

- Civitai API Key
- Image Cache Directory
- Source Mode

Click:

- Apply Settings
- Reload UI

---

🧭 Usage

1. Open Collection tab
2. Click Sync collections
3. Browse images
4. Click an image to view details
5. Use:
   - Send to txt2img
   - Add to collections
   - Copy generation data

---

🧱 Project Status

✅ Completed

- Extension scaffold
- A1111 settings integration
- API connection (images endpoint)
- SQLite database system
- Basic sync flow
- LoRA parsing foundation

---

🚧 In Progress

- Storing synced images in DB
- Image preview caching
- Rendering real images in UI

---

⏳ Planned

- Masonry image browser
- Focus mode UI
- Drag-and-drop ordering
- Full collection sync
- Advanced filtering
- Performance optimization

---

🗺️ Roadmap

Phase 1 — Core Sync (Current)

- [x] API connection
- [x] Settings integration
- [ ] Store images in DB
- [ ] Cache preview images
- [ ] Display images in UI

---

Phase 2 — Library System

- [ ] Local collections
- [ ] Add/remove items
- [ ] Custom ordering

---

Phase 3 — Workflow Tools

- [ ] LoRA matching
- [ ] Full txt2img transfer
- [ ] Missing resource warnings

---

Phase 4 — UI Upgrade

- [ ] Masonry layout
- [ ] Thumbnail resizing
- [ ] Smooth transitions

---

Phase 5 — Focus Mode

- [ ] Expand image view
- [ ] Slide-in detail panel
- [ ] Arrow navigation
- [ ] GIF support

---

Phase 6 — Polish

- [ ] Cache management tools
- [ ] Performance improvements
- [ ] UI refinement

---

🧠 Design Philosophy

- Minimal UI
- Visual-first workflow
- Hide complexity, don’t remove it
- Offline-first design
- No friction between inspiration and creation

---

📁 Repo Structure

extensions/A1111-Collections/
├── scripts/
│   └── collection_tab.py
├── lib/
│   ├── civitai_api.py
│   ├── database.py
│   ├── lora_matcher.py
│   └── parser.py
├── data/
│   ├── cache.db
│   └── images/
├── javascript/
├── style.css

---

🤝 Contributing

This project is in active development.

---

📄 License

MIT License
