import types
import random
import spaces
import logging
import os
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torchaudio
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import AutoModel
import gradio as gr
import tempfile
from huggingface_hub import hf_hub_download

from src.pipeline_wan_nag import NAGWanPipeline
from src.transformer_wan_nag import NagWanTransformer3DModel

# MMAudio imports
try:
    import mmaudio
except ImportError:
    os.system("pip install -e .")
    import mmaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate as mmaudio_generate, 
                                load_video, make_video, setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils

# NAG Video Settings
MOD_VALUE = 32
DEFAULT_DURATION_SECONDS = 4
DEFAULT_STEPS = 4
DEFAULT_SEED = 2025
DEFAULT_H_SLIDER_VALUE = 480
DEFAULT_W_SLIDER_VALUE = 832
NEW_FORMULA_MAX_AREA = 480.0 * 832.0

SLIDER_MIN_H, SLIDER_MAX_H = 128, 896
SLIDER_MIN_W, SLIDER_MAX_W = 128, 896
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 129

DEFAULT_NAG_NEGATIVE_PROMPT = "Static, motionless, still, ugly, bad quality, worst quality, poorly drawn, low resolution, blurry, lack of details"
DEFAULT_AUDIO_NEGATIVE_PROMPT = "music"

# NAG Model Settings
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
SUB_MODEL_ID = "vrgamedevgirl84/Wan14BT2VFusioniX"
SUB_MODEL_FILENAME = "Wan14BT2VFusioniX_fp16_.safetensors"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

# MMAudio Settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
log = logging.getLogger()
device = 'cuda'
dtype = torch.bfloat16
audio_model_config: ModelConfig = all_model_cfg['large_44k_v2']
audio_model_config.download_if_needed()
setup_eval_logging()

# Initialize NAG Video Model
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
wan_path = hf_hub_download(repo_id=SUB_MODEL_ID, filename=SUB_MODEL_FILENAME)
transformer = NagWanTransformer3DModel.from_single_file(wan_path, torch_dtype=torch.bfloat16)
pipe = NAGWanPipeline.from_pretrained(
    MODEL_ID, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
pipe.to("cuda")

pipe.transformer.__class__.attn_processors = NagWanTransformer3DModel.attn_processors
pipe.transformer.__class__.set_attn_processor = NagWanTransformer3DModel.set_attn_processor
pipe.transformer.__class__.forward = NagWanTransformer3DModel.forward

# Initialize MMAudio Model
def get_mmaudio_model() -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    seq_cfg = audio_model_config.seq_cfg
    
    net: MMAudio = get_my_mmaudio(audio_model_config.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(audio_model_config.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded MMAudio weights from {audio_model_config.model_path}')
    
    feature_utils = FeaturesUtils(tod_vae_ckpt=audio_model_config.vae_path,
                                  synchformer_ckpt=audio_model_config.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=audio_model_config.mode,
                                  bigvgan_vocoder_ckpt=audio_model_config.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()
    
    return net, feature_utils, seq_cfg

audio_net, audio_feature_utils, audio_seq_cfg = get_mmaudio_model()

# Audio generation function
@torch.inference_mode()
def add_audio_to_video(video_path, prompt, audio_negative_prompt, audio_steps, audio_cfg_strength, duration):
    """Generate and add audio to video using MMAudio"""
    rng = torch.Generator(device=device)
    rng.seed()  # Random seed for audio
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=audio_steps)
    
    video_info = load_video(video_path, duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    audio_seq_cfg.duration = duration
    audio_net.update_seq_lengths(audio_seq_cfg.latent_seq_len, audio_seq_cfg.clip_seq_len, audio_seq_cfg.sync_seq_len)
    
    audios = mmaudio_generate(clip_frames,
                              sync_frames, [prompt],
                              negative_text=[audio_negative_prompt],
                              feature_utils=audio_feature_utils,
                              net=audio_net,
                              fm=fm,
                              rng=rng,
                              cfg_strength=audio_cfg_strength)
    audio = audios.float().cpu()[0]
    
    # Create video with audio
    video_with_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    make_video(video_info, video_with_audio_path, audio, sampling_rate=audio_seq_cfg.sampling_rate)
    
    return video_with_audio_path

# Combined generation function
def get_duration(prompt, nag_negative_prompt, nag_scale, height, width, duration_seconds, 
                 steps, seed, randomize_seed, enable_audio, audio_negative_prompt, 
                 audio_steps, audio_cfg_strength):
    # Calculate total duration including audio processing if enabled
    video_duration = int(duration_seconds) * int(steps) * 2.25 + 5
    audio_duration = 30 if enable_audio else 0  # Additional time for audio processing
    return video_duration + audio_duration

@spaces.GPU(duration=get_duration)
def generate_video_with_audio(
        prompt,
        nag_negative_prompt, nag_scale,
        height=DEFAULT_H_SLIDER_VALUE, width=DEFAULT_W_SLIDER_VALUE, duration_seconds=DEFAULT_DURATION_SECONDS,
        steps=DEFAULT_STEPS,
        seed=DEFAULT_SEED, randomize_seed=False,
        enable_audio=True, audio_negative_prompt=DEFAULT_AUDIO_NEGATIVE_PROMPT,
        audio_steps=25, audio_cfg_strength=4.5,
):
    # Generate video first
    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    
    num_frames = np.clip(int(round(int(duration_seconds) * FIXED_FPS) + 1), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    
    with torch.inference_mode():
        nag_output_frames_list = pipe(
            prompt=prompt,
            nag_negative_prompt=nag_negative_prompt,
            nag_scale=nag_scale,
            nag_tau=3.5,
            nag_alpha=0.5,
            height=target_h, width=target_w, num_frames=num_frames,
            guidance_scale=0.,
            num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(current_seed)
        ).frames[0]
    
    # Save initial video without audio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        temp_video_path = tmpfile.name
    export_to_video(nag_output_frames_list, temp_video_path, fps=FIXED_FPS)
    
    # Add audio if enabled
    if enable_audio:
        try:
            final_video_path = add_audio_to_video(
                temp_video_path, 
                prompt,  # Use the same prompt for audio generation
                audio_negative_prompt,
                audio_steps,
                audio_cfg_strength,
                duration_seconds
            )
            # Clean up temp video
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except Exception as e:
            log.error(f"Audio generation failed: {e}")
            final_video_path = temp_video_path
    else:
        final_video_path = temp_video_path
    
    return final_video_path, current_seed

# Example generation function
def generate_with_example(prompt, nag_negative_prompt, nag_scale):
    video_path, seed = generate_video_with_audio(
        prompt=prompt,
        nag_negative_prompt=nag_negative_prompt, nag_scale=nag_scale,
        height=DEFAULT_H_SLIDER_VALUE, width=DEFAULT_W_SLIDER_VALUE, 
        duration_seconds=DEFAULT_DURATION_SECONDS,
        steps=DEFAULT_STEPS,
        seed=DEFAULT_SEED, randomize_seed=False,
        enable_audio=True, audio_negative_prompt=DEFAULT_AUDIO_NEGATIVE_PROMPT,
        audio_steps=25, audio_cfg_strength=4.5,
    )
    return video_path, \
        DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE, \
        DEFAULT_DURATION_SECONDS, DEFAULT_STEPS, seed, \
        True, DEFAULT_AUDIO_NEGATIVE_PROMPT, 25, 4.5

# Examples with audio descriptions
examples = [
    ["Midnight highway outside a neon-lit city. A black 1973 Porsche 911 Carrera RS speeds at 120 km/h. Inside, a stylish singer-guitarist sings while driving, vintage sunburst guitar on the passenger seat. Sodium streetlights streak over the hood; RGB panels shift magenta to blue on the driver. Camera: drone dive, Russian-arm low wheel shot, interior gimbal, FPV barrel roll, overhead spiral. Neo-noir palette, rain-slick asphalt reflections, roaring flat-six engine blended with live guitar.", DEFAULT_NAG_NEGATIVE_PROMPT, 11],
    ["Arena rock concert packed with 20 000 fans. A flamboyant lead guitarist in leather jacket and mirrored aviators shreds a cherry-red Flying V on a thrust stage. Pyro flames shoot up on every downbeat, CO₂ jets burst behind. Moving-head spotlights swirl teal and amber, follow-spots rim-light the guitarist’s hair. Steadicam 360-orbit, crane shot rising over crowd, ultra-slow-motion pick attack at 1 000 fps. Film-grain teal-orange grade, thunderous crowd roar mixes with screaming guitar solo.", DEFAULT_NAG_NEGATIVE_PROMPT, 11],
    ["Golden-hour countryside road winding through rolling wheat fields. A man and woman ride a vintage café-racer motorcycle, hair and scarf fluttering in the warm breeze. Drone chase shot reveals endless patchwork farmland; low slider along rear wheel captures dust trail. Sun-flare back-lights the riders, lens blooms on highlights. Soft acoustic rock underscore; engine rumble mixed at –8 dB. Warm pastel color grade, gentle film-grain for nostalgic vibe.", DEFAULT_NAG_NEGATIVE_PROMPT, 11],
]

# CSS styling
css = """
.container {
    max-width: 1400px;
    margin: auto;
    padding: 20px;
}
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 30px;
}
.prompt-container {
    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 1.2em;
    font-weight: bold;
    padding: 15px 30px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
    margin-top: 20px;
}
.generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}
.video-output {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    background: #1a1a1a;
    padding: 10px;
}
.settings-panel {
    background: #f9fafb;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}
.slider-container {
    background: white;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.info-box {
    background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}
.audio-settings {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-radius: 10px;
    padding: 15px;
    margin-top: 10px;
    border-left: 4px solid #f59e0b;
}
"""

# Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML("""
            <h1 class="main-title">🎬 VEO3 Free</h1>
            <p class="subtitle">Wan2.1-T2V-14B + Fast 4-step with NAG + Automatic Audio Generation</p>
        """)
        

        gr.HTML("""
        <div class="badge-container">

            <a href="https://huggingface.co/spaces/ginigen/VEO3-Free" target="_blank">
                <img src="https://img.shields.io/static/v1?label=Text%20to%20Video%2BAudio&message=VEO3%20free&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
            </a>
            <a href="https://huggingface.co/spaces/ginigen/VEO3-Free-mirror" target="_blank">
                <img src="https://img.shields.io/static/v1?label=Text%20to%20Video%2BAudio&message=VEO3%20free%28mirror%29&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
            </a>

            <a href="https://discord.gg/openfreeai" target="_blank">
                <img src="https://img.shields.io/static/v1?label=Discord&message=Openfree%20AI&color=%230000ff&labelColor=%23800080&logo=discord&logoColor=%23ffa500&style=for-the-badge" alt="badge">
            </a>
        </div>
        """)

        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes="prompt-container"):
                    prompt = gr.Textbox(
                        label="✨ Video Prompt (also used for audio generation)",
                        placeholder="Describe your video scene in detail...",
                        lines=3,
                        elem_classes="prompt-input"
                    )
                    
                    with gr.Accordion("🎨 Advanced Video Settings", open=False):
                        nag_negative_prompt = gr.Textbox(
                            label="Video Negative Prompt",
                            value=DEFAULT_NAG_NEGATIVE_PROMPT,
                            lines=2,
                        )
                        nag_scale = gr.Slider(
                            label="NAG Scale",
                            minimum=1.0,
                            maximum=20.0,
                            step=0.25,
                            value=11.0,
                            info="Higher values = stronger guidance"
                        )
                
                with gr.Group(elem_classes="settings-panel"):
                    gr.Markdown("### ⚙️ Video Settings")
                    
                    with gr.Row():
                        duration_seconds_input = gr.Slider(
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=DEFAULT_DURATION_SECONDS,
                            label="📱 Duration (seconds)",
                            elem_classes="slider-container"
                        )
                        steps_slider = gr.Slider(
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=DEFAULT_STEPS,
                            label="🔄 Inference Steps",
                            elem_classes="slider-container"
                        )
                    
                    with gr.Row():
                        height_input = gr.Slider(
                            minimum=SLIDER_MIN_H,
                            maximum=SLIDER_MAX_H,
                            step=MOD_VALUE,
                            value=DEFAULT_H_SLIDER_VALUE,
                            label=f"📐 Height (×{MOD_VALUE})",
                            elem_classes="slider-container"
                        )
                        width_input = gr.Slider(
                            minimum=SLIDER_MIN_W,
                            maximum=SLIDER_MAX_W,
                            step=MOD_VALUE,
                            value=DEFAULT_W_SLIDER_VALUE,
                            label=f"📐 Width (×{MOD_VALUE})",
                            elem_classes="slider-container"
                        )
                    
                    with gr.Row():
                        seed_input = gr.Slider(
                            label="🌱 Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=DEFAULT_SEED,
                            interactive=True
                        )
                        randomize_seed_checkbox = gr.Checkbox(
                            label="🎲 Random Seed",
                            value=True,
                            interactive=True
                        )
                
                with gr.Group(elem_classes="audio-settings"):
                    gr.Markdown("### 🎵 Audio Generation Settings")
                    
                    enable_audio = gr.Checkbox(
                        label="🔊 Enable Automatic Audio Generation",
                        value=True,
                        interactive=True
                    )
                    
                    with gr.Column(visible=True) as audio_settings_group:
                        audio_negative_prompt = gr.Textbox(
                            label="Audio Negative Prompt",
                            value=DEFAULT_AUDIO_NEGATIVE_PROMPT,
                            placeholder="Elements to avoid in audio (e.g., music, speech)",
                        )
                        
                        with gr.Row():
                            audio_steps = gr.Slider(
                                minimum=10,
                                maximum=50,
                                step=5,
                                value=25,
                                label="🎚️ Audio Steps",
                                info="More steps = better quality"
                            )
                            audio_cfg_strength = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                step=0.5,
                                value=4.5,
                                label="🎛️ Audio Guidance",
                                info="Strength of prompt guidance"
                            )
                    
                    # Toggle audio settings visibility
                    enable_audio.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[enable_audio],
                        outputs=[audio_settings_group]
                    )
                
                generate_button = gr.Button(
                    "🎬 Generate Video with Audio",
                    variant="primary",
                    elem_classes="generate-btn"
                )
            
            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="Generated Video with Audio",
                    autoplay=True,
                    interactive=False,
                    elem_classes="video-output"
                )
                
                gr.HTML("""
                    <div style="text-align: center; margin-top: 20px; color: #6b7280;">
                        <p>💡 Tip: The same prompt is used for both video and audio generation!</p>
                        <p>🎧 Audio is automatically matched to the visual content</p>
                    </div>
                """)
        
        gr.Markdown("### 🎯 Example Prompts")
        gr.Examples(
            examples=examples,
            fn=generate_with_example,
            inputs=[prompt, nag_negative_prompt, nag_scale],
            outputs=[
                video_output,
                height_input, width_input, duration_seconds_input,
                steps_slider, seed_input,
                enable_audio, audio_negative_prompt, audio_steps, audio_cfg_strength
            ],
            cache_examples="lazy"
        )
    
    # Connect UI elements
    ui_inputs = [
        prompt,
        nag_negative_prompt, nag_scale,
        height_input, width_input, duration_seconds_input,
        steps_slider,
        seed_input, randomize_seed_checkbox,
        enable_audio, audio_negative_prompt, audio_steps, audio_cfg_strength,
    ]
    
    generate_button.click(
        fn=generate_video_with_audio,
        inputs=ui_inputs,
        outputs=[video_output, seed_input],
    )

if __name__ == "__main__":
    demo.queue().launch()