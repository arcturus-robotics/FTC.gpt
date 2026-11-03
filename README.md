<<<<<<< HEAD
<h2>FTC.gpt</h2>

This is a project to help out teams who may benefit from a database of important documents, all fed to an AI model to answer questions and provide insight.
<h4>Dependencies:</h4>

* Ollama
  * gemma4:e4b
  * nomic-embed-text

<h4>Install Instructions:</h4>

1. Download the package for macOS or Windows, if you're using Linux you probably know how to run from source so do that.

2. Install Ollama via one of the following:

   - Windows: open powershell and paste the following:
     * `irm https://ollama.com/install.ps1 | iex`

   * macOS: open Terminal.app and paste the following:
     * `curl -fsSL https://ollama.com/install.sh | sh`

   * Linux: Open your terminal emulator of choice and paste the following:

     * `curl -fsSL https://ollama.com/install.sh | sh` (same command as macOS)

     * `cd` into the project directory

     * Run `pip install -r requirements.txt`

3. Regardless of OS, run these commands:

   1. `ollama pull gemma4:e4b` (installs the model responsible for the interface)
   2. `ollama pull nomic-embed-text` (installs the embedding model, necessary to access the existing database of knowledge)

<h4>Usage Instructions:</h4>

Windows:

1. Open powershell and  run `ollama serve`(this makes ollama reachable via localhost)
2. Run the executable file downloaded in step 1 of Install Instructions

macOS:

1. Open `Terminal.app` and run `ollama serve` (makes ollama reachable via localhost)
2. Run the executable file downloaded in step 1 of Install Instructions

Linux:

1. Run `ollama serve` (makes ollama reachable via localhost)
2. In a seperate terminal window, `cd` into the project directory
3. Run `python3 ai.py` (tested on python 3.9 and 3.12) 

Regarless of OS:

* Open the link shown in the terminal, or open a web browser and paste in `http://localhost:8000`

The interface should now be ready to use! Enjoy!

Feel free to refine the model to accommodate your team's needs. We'd love to hear your feedback!

>>>>>>> 737c0d0 (more data, updated readme, added setup script)
