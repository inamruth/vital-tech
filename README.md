# 🎨 Audio2Art

Audio2Art is a powerful AI-based application that converts voice inputs into stunning AI-generated images. It utilizes OpenAI's Whisper for audio transcription and Stable Diffusion XL for image generation.

## 🚀 Features

- 🎤 **Voice-to-Text**: Transcribe recorded or uploaded audio using Whisper AI.
- 🖼 **AI Image Generation**: Convert transcribed text into stunning visuals using Stable Diffusion XL.
- 🎨 **Customizable Styles**: Select from various image styles, lighting, rendering engines, and camera angles.
- ⚡ **GPU/CPU Compatibility**: Automatically adjusts to available hardware for optimal performance.
- 🖥 **Gradio-Powered UI**: Interactive and user-friendly interface for smooth experience.

## 📦 Installation

Ensure you have Python installed, then run:

```bash
pip install gradio diffusers transformers accelerate torch safetensors openai-whisper
```

Alternatively, you can run this on **Google Colab** by setting up the environment with the following command:

```python
!pip install gradio diffusers transformers accelerate torch safetensors openai-whisper
```

## 🔧 Usage

### Running Locally
Run the application with:

```bash
python your_script.py
```

### Running on Google Colab
To run on Google Colab:
1. Upload your script to Colab.
2. Run the installation command.
3. Execute the script in a code cell:

```python
!python your_script.py
```

It will launch a Gradio interface where you can:

1. Record or upload an audio file.
2. Select image styles and settings.
3. Generate AI-powered artwork based on the transcribed text.

## 🛠 Dependencies

- [Gradio](https://www.gradio.app/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Stable Diffusion XL](https://huggingface.co/segmind/SSD-1B)
- [Torch](https://pytorch.org/)

## 🤖 How It Works

1. **Audio Processing**: Whisper AI transcribes your voice input.
2. **Image Prompt Creation**: The transcribed text is used to generate an AI prompt.
3. **Image Generation**: Stable Diffusion XL produces a high-quality image.
4. **Results Display**: The image is shown in an interactive Gradio UI.


## 🤝 Contributing

Feel free to submit issues or pull requests to enhance the project.

## 📜 License

This project is licensed under the MIT License.

---

Enjoy creating art from your voice! 🎙️🎨
