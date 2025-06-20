
# Taylor Swift Lyrics Finder

An AI-powered web app that finds the most relevant Taylor Swift lyrics based on your life scenarios.

## How to Use

1. **Add your lyrics files**: Place `.txt` files containing Taylor Swift song lyrics in the `lyrics/` folder. Each file should be named after the song (e.g., `Love Story.txt`, `Shake It Off.txt`).

2. **Run the app**: Click the Run button or use `python main.py`

3. **Find lyrics**: 
   - Open the app in your browser
   - Describe your life scenario in the text area
   - Click "Find Matching Lyrics" to get AI-powered matches
   - Use "Reload Songs" if you add new lyrics files

## Features

- AI-powered semantic matching using sentence transformers
- Beautiful, responsive web interface
- Relevance scoring for each match
- Support for multiple song files
- Real-time lyrics reloading

## Adding Lyrics

Create `.txt` files in the `lyrics/` folder with song lyrics. The app will automatically load them and create embeddings for AI matching.

Example file structure:
```
lyrics/
├── Love Story.txt
├── Shake It Off.txt
├── Anti-Hero.txt
└── Blank Space.txt
```
