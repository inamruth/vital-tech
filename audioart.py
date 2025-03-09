!pip install gradio diffusers transformers accelerate torch safetensors openai-whisper -q
import gradio as gr
import torch
import gc
import whisper
from diffusers import StableDiffusionXLPipeline

# ---- Optimize Memory ----
torch.cuda.empty_cache()
gc.collect()

# ---- Force CPU Mode ----
device = "cpu"  # Force CPU to fix the dtype issue
print(f"Using device: {device}")

# ---- Load Whisper for Speech-to-Text ----
try:
    whisper_model = whisper.load_model("base")
    whisper_model = whisper_model.to(device)
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Error loading Whisper model: {e}")

# ---- Load Stable Diffusion XL Model ----
try:
    # Use float32 for CPU compatibility instead of float16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float32,  # Changed from float16 to float32 for CPU
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to(device)
    print("Stable Diffusion model loaded successfully")
except Exception as e:
    print(f"Error loading Stable Diffusion model: {e}")

# ---- Dropdown Options ----
image_style_options = ["Sci-fi", "Photorealistic", "Low Poly", "Cinematic", "Cartoon", "Graffiti", "Sketching"]
image_quality_options = ["High Resolution", "Clear", "Detailed", "Beautiful", "Realistic"]
render_options = ["Pixar", "Octane", "Unreal Engine", "Unity"]
angle_options = ["Wide-Angle", "Front View", "Top View", "Side View"]
lighting_options = ["Soft", "Ambient", "Neon", "Natural", "Dramatic"]
background_options = ["Outdoor", "Indoor", "Space", "Nature", "Abstract"]
device_options = ["Camera", "Professional Camera", "Smartphone", "Film Camera"]
emotion_options = ["Happy", "Sad", "Mysterious", "Neutral", "Dreamy"]

# ---- Function to Process Audio and Generate Image ----
def process_audio(audio_file, image_style, image_quality, render, angle, lighting, background, device_type, emotion):
    if audio_file is None:
        return "Please record or upload audio first.", None

    try:
        # ---- Convert Speech to Text ----
        print(f"Processing audio file: {audio_file}")
        result = whisper_model.transcribe(audio_file)
        text_prompt = result["text"]
        print(f"Transcribed text: {text_prompt}")

        # ---- Generate Image ----
        full_prompt = f"{text_prompt} in {image_style} style, {image_quality}, {render} render, {angle} angle, {lighting} lighting, {emotion} mood, {background} background, shot on {device_type}"
        neg_prompt = "ugly, blurry, poor quality, deformed, bad lighting, noisy"

        print(f"Generating image with prompt: {full_prompt}")
        # Don't use autocast with CPU
        output = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            height=512,
            width=512,
            num_inference_steps=20
        )

        image = output.images[0]
        print("Image generated successfully")
        return text_prompt, image

    except Exception as e:
        print(f"Error in process_audio: {e}")
        return f"Error processing: {str(e)}", None

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Audio2Art")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="🎤 Record Your Voice")

    with gr.Row():
        style = gr.Dropdown(choices=image_style_options, label="🎨 Image Style", value="Photorealistic")
        quality = gr.Dropdown(choices=image_quality_options, label="🖼 Quality", value="High Resolution")

    with gr.Row():
        render = gr.Dropdown(choices=render_options, label="🖥 Render Engine", value="Pixar")
        angle = gr.Dropdown(choices=angle_options, label="📷 Camera Angle", value="Front View")

    with gr.Row():
        lighting = gr.Dropdown(choices=lighting_options, label="💡 Lighting", value="Soft")
        background = gr.Dropdown(choices=background_options, label="🌆 Background", value="Outdoor")

    with gr.Row():
        device_type = gr.Dropdown(choices=device_options, label="📸 Device", value="Professional Camera")
        emotion = gr.Dropdown(choices=emotion_options, label="😃 Emotion", value="Happy")

    generate_button = gr.Button("🚀 Generate Image")

    output_text = gr.Textbox(label="📝 Transcribed Text")
    output_image = gr.Image(label="🖼 Generated Image")

    generate_button.click(
        process_audio,
        inputs=[audio_input, style, quality, render, angle, lighting, background, device_type, emotion],
        outputs=[output_text, output_image]
    )

# Launch with debug information
demo.launch(share=True, debug=True)
