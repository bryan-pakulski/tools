# Gemini CLI

A powerful, interactive command-line interface for Google's Gemini models. This tool offers
persistent chat sessions, file attachments, syntax highlighting, and "thinking" mode support,
all from your terminal.

## Features

- **Persistent Sessions**: Chat history is saved automatically (`~/.gemini_chats/`). You can  
  switch between conversations, list them, or delete them.
- **Chat Summarisation**: Long chain conversations are summarised intermittently to reduce token usage whilst still maintaining long context.
- **Smart Formatting**:
  - Markdown rendering (via `glow` if installed).
  - Raw code block output for easy copying.
- **File Attachments**: Upload local files (images, PDFs, text) to the context using `/file`.
- **Autocomplete**: Tab-completion for commands, file paths, and session names.
- **Thinking Mode**: Toggle Gemini's "thinking" process for complex reasoning tasks.
- **System Instructions**: Set custom system prompts on the fly.
- **Multi-line Input**: End a line with `\` to continue typing on the next line.

## Prerequisites

1. **Python 3.9+**
2. **Google GenAI SDK**
3. **Glow** (Optional, for pretty Markdown rendering):
    - macOS: `brew install glow`
    - Linux: `sudo apt install glow` (or via package manager)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/gemini-cli.git                                        
cd gemini-cli                                                                                   
```

1. Install dependencies:

```bash
pip install -r requirements.txt                                                                 
```

1. Set your API Key:
    You need a Google Gemini API key. Export it as an environment variable:

```bash
export GOOGLE_API_KEY="your_api_key_here"                                                       
```

1. Make the script executable (optional):

```bash
chmod +x gemini_cli.py                                                                          
```

## Usage

Run the script:

```bash
python gemini_cli.py                                                                            
```

### Command Arguments

| Argument | Description |
| :--- | :--- |
| `--model` | Set default model (default: `gemini-2.0-flash-thinking-exp-1219`) |
| `--thinking` | Start with thinking mode enabled |
| `--system` | Set initial system instruction |

### In-Chat Commands

| Command | Alias | Action |
| :--- | :--- | :--- |
| `/help` | `/h` | Show help menu |
| `/new [name]` | | Start a new chat session |
| `/list` | `/ls` | List available saved sessions |
| `/load [name]` | `/open` | Switch to a specific session |
| `/delete [name]`| `/rm` | Delete a session |
| `/file [path]` | `/f` | Attach a file to the next message |
| `/clearfiles` | `/cf` | Clear currently staged files |
| `/view` | `/v` | View full history of current chat |
| `/clear` | `/c` | Clear history of current chat |
| `/system [txt]` | `/sys` | Update system prompt |
| `/tokens` | | View token usage (estimates) |
| `/summary` | | View current summarisation of conversation |
| `/thinking` | | Toggle thinking mode on/off |
| `/quit` | `/q` | Exit the application |

## Tips

- **Multi-line Input**: If you want to paste a large block of code or write a long prompt, end
  your line with a backslash `\` and press Enter. You can continue typing on the next line.
- **File Uploads**: You can attach multiple files before sending a message. Use `/f
  path/to/file1`, then `/f path/to/file2`, then type your prompt.

## License

MIT
