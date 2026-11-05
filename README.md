# KoboldCode
A tool inspired on ClaudeCode, fully local, free and private.

KoboldCode
An open-source, local-first alternative to Claude and Claude Code — powered entirely by KoboldCpp and your own hardware.

KoboldCode packs a fully featured, Claude-inspired chat interface and an agentic coding environment (Claude Code style) into a single, dependency-free Python file. It runs locally, connects seamlessly to local GGUF models via KoboldCpp, and keeps your code and data entirely on your machine.

Features
💬 Part 1: The Chat Interface
Claude-Like Layout: Features a sleek sidebar with chat history, pinned chats, search (Ctrl+K), projects, and a customization panel.

Rich Markdown Rendering: Real-time streaming with tables, syntax-highlighted code blocks (with instant Copy and Save buttons), and collapsible <think> reasoning blocks.

Time-of-Day Adaptation: Dynamic greetings, glow colors, suggestion chips, and automated light/dark themes that shift based on the hour (e.g., late-night sessions automatically switch to comfortable night modes).

Effort Modes: Switch between Low, Medium, High, Extra, and Ultracode to dynamically adjust response length, temperature, and reasoning guidelines sent to the model.

Projects: Group related chats with dedicated custom instructions and persistent reference files.

Full Customization: Configure your persona, custom instructions, response style, fonts, accent colors, slash commands, and KoboldCpp connection parameters with a built-in connection tester.

💻 Part 2: The Code Interface (Claude Code Style)
Agent Tools: The model can autonomously list, read, write, edit, search, and delete files, execute shell commands, maintain a live TODO task list, and complete coding objectives using a robust XML text format optimized for local models.

Flexible Permission Modes: Choose between Ask before edits, Edit automatically, Plan mode (read-only with an "Approve plan & build" button), and Bypass.

Interactive Workspace Panel: Track the live agent plan, explore file trees, preview or download any file, undo the last workspace change, open directories, and export the entire project as a .zip archive.

Project Memory & Commands: Use /init to generate a project-level KOBOLD.md memory file loaded every session. Built-in slash commands include /review, /test, /commit, /plan, /mode, /effort, /compact, and /help.

Hardware Recommendations & Best Models
Designed and tested around local setups (such as an RTX 3060 12GB paired with 32GB system RAM):

Smartest Coding Option (MoE Offloading): Qwen3.6-35B-A3B (or similar MoE models). Because only a fraction of the parameters are active per token, mixture-of-experts models can split layers between VRAM and system RAM via KoboldCpp.

Blazing Fast Local Chat: Gemma 4 12B (fits entirely in VRAM for high tokens-per-second performance).

Proven Local Coder: Qwen2.5-Coder 14B (reliable tool-calling and code generation out of the box).

Quick Start
1. Launch KoboldCpp
Start your KoboldCpp server with your model of choice, appropriate VRAM/CPU offloading flags, and context size:

Bash
koboldcpp --model your_model.gguf --port 5001 --contextsize 32768 (OR ANY CONTEXT YOU WANT) --flashattention
2. Run KoboldCode
Execute the single-file Python application (requires only the Python standard library):

Bash
python kobold_code.py



(Author note): Pls use a thinking model :)
