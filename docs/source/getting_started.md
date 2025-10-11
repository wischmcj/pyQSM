'# Getting started

## Installation

1. **Create a Virtual Environment**: elow wBe use the native 'venv' module to create a virtual environment. This is not strictly necessary, but it is a good practice to keep your project dependencies separate from your system dependencies in a virtual environment.
   ```bash
   python -m venv venv
   ```

2. **Activate Your Environment**: Activate your virtual environment to install dependencies and run the project. The commands to activate the virtual environment depend on your operating system and shell. Below are the commands for activating the virtual environment in different operating systems and shells.

```bash
  # if using bash (Mac, Unix)
  source venv/bin/activate
  # if using PowerShell (Windows)
  source venv\Scripts\activate.ps1
```

3. **Install pyQSM**: pyQSM is published with PyPA (the python packacing authority), so you can install the latest stable release of pyQSM using pip. This installs our latest stable release as well as several libraries required for the use of the package's features. pyQSM currently supports Python versions 3.9. 3.10 and 3.11.

```bash
   pip install pyQSM
```
