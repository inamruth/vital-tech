!pip install gradio diffusers hugchat transformers accelerate safetensors --upgrade -q
!pip install git+https://github.com/huggingface/diffusers -q
!pip install ipywidgets -q
!pip install invisible_watermark -q

from transformers import pipeline
from ipywidgets import interactive, widgets
from IPython.display import HTML, Javascript, Image, display
from google.colab.output import eval_js
import base64
from diffusers import StableDiffusionXLPipeline
import torch
import gc

# Add memory optimization settings
torch.cuda.empty_cache()
gc.collect()

# Set PyTorch memory allocation settings
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Use a smaller model with lower memory requirements
pipe = StableDiffusionXLPipeline.from_pretrained(
    "segmind/SSD-1B", 
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    # Enable sequential CPU offloading to save memory
    low_cpu_mem_usage=True
)

# Enable memory-efficient attention mechanism
pipe.enable_attention_slicing(1)

# Move to CUDA with more memory-efficient settings
pipe = pipe.to("cuda")

# For whisper, select a smaller model if possible to save memory
whisper = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-base",  # Using smaller whisper model
    chunk_length_s=30, 
    device="cuda:0"
)

js = Javascript(
    """
    async function recordAudio() {
      const div = document.createElement('div');
      const audio = document.createElement('audio');
      const strtButton = document.createElement('button');
      const stopButton = document.createElement('button');

      strtButton.textContent = 'Start Recording';
      stopButton.textContent = 'Stop Recording';

      document.body.appendChild(div);
      div.appendChild(strtButton);
      div.appendChild(audio);

      const stream = await navigator.mediaDevices.getUserMedia({audio:true});
      let recorder = new MediaRecorder(stream);

      audio.style.display = 'block';
      audio.srcObject = stream;
      audio.controls = true;
      audio.muted = true;

      await new Promise((resolve) => strtButton.onclick = resolve);
        strtButton.replaceWith(stopButton);
        recorder.start();

      await new Promise((resolve) => stopButton.onclick = resolve);
        recorder.stop();
        let recData = await new Promise((resolve) => recorder.ondataavailable = resolve);
        let arrBuff = await recData.data.arrayBuffer();
        stream.getAudioTracks()[0].stop();
        div.remove()

        let binaryString = '';
        let bytes = new Uint8Array(arrBuff);
        bytes.forEach((byte) => { binaryString += String.fromCharCode(byte)});

      const url = URL.createObjectURL(recData.data);
      const player = document.createElement('audio');
      player.controls = true;
      player.src = url;
      document.body.appendChild(player);

    return btoa(binaryString)

          };
          """
)

display(js)

output = eval_js('recordAudio({})')
with open('audio.wav', 'wb') as file:
    binary = base64.b64decode(output)
    file.write(binary)
print('Recording saved to:', file.name)
speech_to_text = whisper("audio.wav")

img = speech_to_text['text']
print(img)

generated_image = None
image_display = display("", display_id=True)

def generate_image(button):
    global generated_image
    
    # Clear memory before generating
    torch.cuda.empty_cache()
    gc.collect()

    image_style = style_dropdown.value
    image_quality = quality_dropdown.value
    render = render_dropdown.value
    angle = angle_dropdown.value
    lighting = lighting_dropdown.value
    background = background_dropdown.value
    device = device_dropdown.value
    emotion = emotion_dropdown.value

    # Use a simpler prompt structure to reduce complexity
    prompt = f"{img} in {image_style} style, {image_quality}, {render} render, {lighting} lighting, {emotion} mood, {background} background"

    neg_prompt = "ugly, blurry, poor quality, deformed, bad lighting, noisy"
    
    # Use smaller image sizes to reduce memory usage
    with torch.autocast("cuda"):
        output = pipe(
            prompt=prompt, 
            negative_prompt=neg_prompt,
            height=512,  # Reduced from default 1024
            width=512,   # Reduced from default 1024
            num_inference_steps=20  # Reduced steps
        )
    
    generated_image = output.images[0]
    generated_image.save("generated_image.png")
    final_image_path = "generated_image.png"
    image_display.update(Image(filename=final_image_path))
    
    # Clean up after generation
    torch.cuda.empty_cache()
    gc.collect()

image_style_options = ["Sci-fi", "photorealistic", "low poly", "cinematic", "cartoon", "graffiti", "sketching"]
image_quality_options = ["High resolution", "clear", "detailed", "beautiful", "realistic"]
render_options = ["Pixar", "Octane", "Unreal engine", "Unity"]
angle_options = ["Wide-angle", "front view", "Top view", "Side view"]
lighting_options = ["Soft", "ambient", "neon", "Natural", "Dramatic"]
background_options = ["outdoor", "indoor", "space", "nature", "abstract"]
device_options = ["camera", "professional camera", "smartphone", "film camera"]
emotion_options = ["Happy", "Sad", "Mysterious", "Neutral", "dreamy"]

style_dropdown = widgets.Dropdown(options=image_style_options, description="Style:")
quality_dropdown = widgets.Dropdown(options=image_quality_options, description="Quality:")
render_dropdown = widgets.Dropdown(options=render_options, description="Render by:")
angle_dropdown = widgets.Dropdown(options=angle_options, description="Angle:")
lighting_dropdown = widgets.Dropdown(options=lighting_options, description="Lighting:")
background_dropdown = widgets.Dropdown(options=background_options, description="Background:")
device_dropdown = widgets.Dropdown(options=device_options, description="Device:")
emotion_dropdown = widgets.Dropdown(options=emotion_options, description="Emotion:")

generate_button = widgets.Button(description="Generate Image")
generate_button.on_click(generate_image)

interactive_widget = widgets.VBox([
    style_dropdown, quality_dropdown, angle_dropdown, render_dropdown,
    lighting_dropdown, background_dropdown, device_dropdown, emotion_dropdown,
    generate_button
])

display(interactive_widget)