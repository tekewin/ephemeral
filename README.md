# Ephemeral

![banner](/ephemeral-banner.png)

**Ephemeral** is a private, secure, and open-source web application for chatting with advanced AI models. It is designed to be lightweight and privacy-focused, ensuring your interactions remain within your control.

The application integrates seamlessly with **Together AI** to provide access to top-tier open-source models, featuring advanced reasoning capabilities, and Zero Data Retention.

## Features

* **Private:** Built with privacy in mind. Uses Zero Data Retention at together.ai and logs no information anywhere. When the session is over, the chat is gone forever.
* **Secure:** All API calls go through api.together.ai which is HTTPS-only. Your prompts and responses are encrypted in transit.
* **Open Source Models:** Chat with the latest open-source LLMs via the Together AI API.
* **Advanced Reasoning:** Ephemeral uses the **GLM-5** model, which is optimized for complex reasoning, coding, and agentic tasks.
* **Transparency:** View the model's "thinking" process in a dedicated text area.

---

## Quick Start

Follow these steps to get Ephemeral running on your local machine.

### 1. Set up Together AI
1.  Go to [together.ai](https://together.ai) and sign up for an account.
2.  Navigate to **Settings**, scroll to **Privacy & Security**. Set "Store prompts and model responses..." to NO. Set "Allow my data to be used for training models..." to NO. These settings enable the Zero Data Retention policy.
   
![privacy-settings](/privacy-settings.png)

3.  Navigate to your **Settings** > **API Keys**.
4.  Click **Create API Key** to generate a new key. Copy this string; you will need it for the next step. Note the "user key" is not the same as an API key.

### 2. Configure Your Environment
Create a local `.env` file so Ephemeral can authenticate with Together AI:

```bash
cp .env_example .env
```

Then edit `.env` and set:

```env
TOGETHER_API_KEY=your-api-key-here
```

`.env` is ignored by git and must be set up manually by each user.

### 3. Install Dependencies
Ensure you have Python installed (I recommend a [python virtual environment](https://docs.python.org/3/library/venv.html)), then install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Launch the Application
Run the application from your command line:

```bash
streamlit run app.py
```
The app will automatically open in your default web browser (usually at `http://localhost:8501`).

---

## Usage Guide

### Chat & Image Handling
* **Standard Chat:** Type your message in the chat input to interact with the default text model **GLM-5**.
* **Image Uploads:** Image uploads are currently disabled as the active model (**GLM-5**) is not a vision model.

### Thinking Process
* **Thinking Text Area:** For models that support Chain of Thought or reasoning steps, a specialized "Thinking" expander or text area will appear. Click this to view the raw reasoning process the model used to arrive at its final answer.

### Configuration
* **Default Output Token Limit:** The application is configured with a default output token limit of 128,000. It can be modified in the app.py script.

## Specifications for GLM-5 FP4

**Architecture:** Mixture-of-Experts (MoE) with DeepSeek Sparse Attention (DSA)

**Total Parameters:** 744 Billion (435.2 FP4 version)

**Active Parameters:** 40 Billion

**Training Data:** Optimized for complex systems engineering and long-horizon agent workflows.

**Max Context Window (Input + Output):** 128,000+ tokens.
