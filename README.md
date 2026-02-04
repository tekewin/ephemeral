# Ephemeral
### Zero trace inference

![Alt text](/ephemeral-banner.png)

**Ephemeral** is a private, secure, and open-source web application for chatting with advanced AI models. It is designed to be lightweight and privacy-focused, ensuring your interactions remain within your control.

The application integrates seamlessly with **Together AI** to provide access to top-tier open-source models, featuring automatic multi-modal support.

## Features

* **Private & Secure:** Built with privacy in mind.
* **Open Source Models:** Chat with the latest open-source LLMs via the Together AI API.
* **Multi-Modal Support:** Automatically detects image uploads and switches to the **Kimi-K2** vision model to analyze and discuss your images.
* **Transparency:** View the model's "thinking" process in a dedicated text area.
* **Customizable:** Adjustable token limits and settings.

---

## Quick Start

Follow these steps to get Ephemeral running on your local machine.

### 1. Set up Together AI
1.  Go to [together.ai](https://together.ai) and sign up for an account.
2.  Navigate to your **Settings** > **API Keys**.
3.  Click **Create API Key** to generate a new key. Copy this string; you will need it for the next step.

### 2. Configure Your Environment
You need to set your API key as an environment variable so Ephemeral can authenticate with the service.

**Mac/Linux:**
Open your terminal and run:
```bash
export TOGETHER_API_KEY="your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set TOGETHER_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:TOGETHER_API_KEY="your-api-key-here"
```

### 3. Install Dependencies
Ensure you have Python installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Launch the Application
Run the application from your command line:

```bash
streamlit run app.py
```
*Note: If your main file is named differently (e.g., `main.py` or `ephemeral.py`), adjust the command accordingly.*

The app will automatically open in your default web browser (usually at `http://localhost:8501`).

---

## Usage Guide

### Chat & Image Handling
* **Standard Chat:** Type your message in the chat input to interact with the default text model.
* **Image Uploads:** Drag and drop an image into the sidebar or upload area.
    * **Automatic Model Switching:** When an image is detected, Ephemeral automatically switches the active model to **Kimi-K2**. This specialized vision model is capable of analyzing the visual content and answering questions about your image.

### Thinking Process
* **Thinking Text Area:** For models that support Chain of Thought or reasoning steps, a specialized "Thinking" expander or text area will appear. Click this to view the raw reasoning process the model used to arrive at its final answer.

### Configuration
* **Default Output Token Limit:** The application is configured with a default output token limit (typically **512 tokens**) to ensure concise and fast responses.
* You can adjust this limit in the application sidebar if you require longer, more detailed responses (e.g., for creative writing or complex code generation).
