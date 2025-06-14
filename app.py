import types
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video
import gradio as gr
import tempfile
import spaces
from huggingface_hub import hf_hub_download
import numpy as np
import random
import logging
import torchaudio
import os
import gc

# MMAudio imports
try:
    import mmaudio
except ImportError:
    os.system("pip install -e .")
    import mmaudio

# Set environment variables for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['HF_HUB_CACHE'] = '/tmp/hub'

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils

# NAG imports
from src.pipeline_wan_nag import NAGWanPipeline
from src.transformer_wan_nag import NagWanTransformer3DModel

# Clean up temp files periodically
def cleanup_temp_files():
    """Clean up temporary files to save storage"""
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        try:
            if filename.endswith(('.mp4', '.flac', '.wav')):
                os.remove(filepath)
        except:
            pass

# Video generation model setup (NAG)
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
SUB_MODEL_ID = "vrgamedevgirl84/Wan14BT2VFusioniX"
SUB_MODEL_FILENAME = "Wan14BT2VFusioniX_fp16_.safetensors"

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

# Audio generation model setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()
device = 'cuda'
dtype = torch.bfloat16

# Global variables for audio model (loaded on demand)
audio_model = None
audio_net = None
audio_feature_utils = None
audio_seq_cfg = None

def load_audio_model():
    """Load audio model on demand to save storage"""
    global audio_model, audio_net, audio_feature_utils, audio_seq_cfg
    
    if audio_net is None:
        audio_model = all_model_cfg['small_16k']
        audio_model.download_if_needed()
        setup_eval_logging()
        
        seq_cfg = audio_model.seq_cfg
        net = get_my_mmaudio(audio_model.model_name).to(device, dtype).eval()
        net.load_weights(torch.load(audio_model.model_path, map_location=device, weights_only=True))
        log.info(f'Loaded weights from {audio_model.model_path}')

        feature_utils = FeaturesUtils(tod_vae_ckpt=audio_model.vae_path,
                                      synchformer_ckpt=audio_model.synchformer_ckpt,
                                      enable_conditions=True,
                                      mode=audio_model.mode,
                                      bigvgan_vocoder_ckpt=audio_model.bigvgan_16k_path,
                                      need_vae_encoder=False)
        feature_utils = feature_utils.to(device, dtype).eval()
        
        audio_net = net
        audio_feature_utils = feature_utils
        audio_seq_cfg = seq_cfg
    
    return audio_net, audio_feature_utils, audio_seq_cfg

# Constants
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
default_audio_prompt = ""
default_audio_negative_prompt = "music"

# CSS
custom_css = """
/* 전체 배경 그라디언트 */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #fa709a 100%) !important;
    background-size: 400% 400% !important;
    animation: gradientShift 15s ease infinite !important;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 메인 컨테이너 스타일 */
.main-container {
    backdrop-filter: blur(10px);
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 30px !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
}

/* 헤더 스타일 */
h1 {
    background: linear-gradient(45deg, #ffffff, #f0f0f0) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 800 !important;
    font-size: 2.5rem !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
}

/* 컴포넌트 컨테이너 스타일 */
.input-container, .output-container {
    background: rgba(255, 255, 255, 0.08) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin: 10px 0 !important;
    backdrop-filter: blur(5px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* 입력 필드 스타일 */
input, textarea, .gr-box {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 10px !important;
    color: #333 !important;
    transition: all 0.3s ease !important;
}

input:focus, textarea:focus {
    background: rgba(255, 255, 255, 1) !important;
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* 버튼 스타일 */
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 12px 30px !important;
    border-radius: 50px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* 슬라이더 스타일 */
input[type="range"] {
    background: transparent !important;
}

input[type="range"]::-webkit-slider-track {
    background: rgba(255, 255, 255, 0.3) !important;
    border-radius: 5px !important;
    height: 6px !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: 2px solid white !important;
    border-radius: 50% !important;
    cursor: pointer !important;
    width: 18px !important;
    height: 18px !important;
    -webkit-appearance: none !important;
}

/* Accordion 스타일 */
.gr-accordion {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    margin: 15px 0 !important;
}

/* 라벨 스타일 */
label {
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    margin-bottom: 5px !important;
}

/* 비디오 출력 영역 */
video {
    border-radius: 15px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

/* Examples 섹션 스타일 */
.gr-examples {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin-top: 20px !important;
}

/* Checkbox 스타일 */
input[type="checkbox"] {
    accent-color: #667eea !important;
}

/* Radio 버튼 스타일 */
input[type="radio"] {
    accent-color: #667eea !important;
}

/* 반응형 애니메이션 */
@media (max-width: 768px) {
    h1 { font-size: 2rem !important; }
    .main-container { padding: 20px !important; }
}
"""

def clear_cache():
    """Clear GPU and CPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_duration(prompt, nag_negative_prompt, nag_scale,
                height, width, duration_seconds,
                steps, seed, randomize_seed,
                audio_mode, audio_prompt, audio_negative_prompt,
                audio_seed, audio_steps, audio_cfg_strength,
                progress):
    base_duration = int(duration_seconds) * int(steps) * 2.25 + 5
    
    # Add extra time for audio generation
    if audio_mode == "Enable Audio":
        base_duration += 60
    
    return base_duration

@torch.inference_mode()
def add_audio_to_video(video_path, duration_sec, audio_prompt, audio_negative_prompt, 
                      audio_seed, audio_steps, audio_cfg_strength):
    """Add audio to video using MMAudio"""
    # Load audio model on demand
    net, feature_utils, seq_cfg = load_audio_model()
    
    rng = torch.Generator(device=device)
    if audio_seed >= 0:
        rng.manual_seed(audio_seed)
    else:
        rng.seed()
    
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=audio_steps)
    
    video_info = load_video(video_path, duration_sec)
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    duration = video_info.duration_sec
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    
    audios = generate(clip_frames,
                      sync_frames, [audio_prompt],
                      negative_text=[audio_negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=audio_cfg_strength)
    audio = audios.float().cpu()[0]
    
    # Save video with audio
    video_with_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    make_video(video_info, video_with_audio_path, audio, sampling_rate=seq_cfg.sampling_rate)
    
    return video_with_audio_path

@spaces.GPU(duration=get_duration)
def generate_video(prompt, nag_negative_prompt, nag_scale,
                   height, width, duration_seconds,
                   steps, seed, randomize_seed,
                   audio_mode, audio_prompt, audio_negative_prompt,
                   audio_seed, audio_steps, audio_cfg_strength,
                   progress=gr.Progress(track_tqdm=True)):
    
    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    
    num_frames = np.clip(int(round(int(duration_seconds) * FIXED_FPS) + 1), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    # Generate video using NAG
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

    # Save video without audio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(nag_output_frames_list, video_path, fps=FIXED_FPS)
    
    # Generate audio if enabled
    video_with_audio_path = None
    if audio_mode == "Enable Audio":
        progress(0.5, desc="Generating audio...")
        video_with_audio_path = add_audio_to_video(
            video_path, duration_seconds, 
            audio_prompt, audio_negative_prompt,
            audio_seed, audio_steps, audio_cfg_strength
        )
    
    # Clear cache to free memory
    clear_cache()
    cleanup_temp_files()
    
    return video_path, video_with_audio_path, current_seed

def update_audio_visibility(audio_mode):
    """Update visibility of audio-related components"""
    return gr.update(visible=(audio_mode == "Enable Audio"))

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes=["main-container"]):
        gr.Markdown("# ✨ Fast NAG T2V (14B) with Audio Generation")

        # Add badges
        gr.HTML("""
        <div class="badge-container">
            <a href="https://huggingface.co/spaces/Heartsync/WAN2-1-fast-T2V-FusioniX" target="_blank">
                <img src="https://img.shields.io/static/v1?label=BASE&message=WAN%202.1%20T2V-FusioniX&color=%23008080&labelColor=%23533a7d&logo=huggingface&logoColor=%23ffffff&style=for-the-badge" alt="Base Model">
            </a>
            <a href="https://huggingface.co/spaces/Heartsync/WAN2-1-fast-T2V-FusioniX2" target="_blank">
                <img src="https://img.shields.io/static/v1?label=BASE&message=WAN%202.1%20T2V-Fusioni2X&color=%23008080&labelColor=%23533a7d&logo=huggingface&logoColor=%23ffffff&style=for-the-badge" alt="Base Model">
            </a>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(elem_classes=["input-container"]):
                prompt_input = gr.Textbox(
                    label="✏️ Video Prompt",
                    placeholder="Describe your video scene in detail...",
                    lines=3
                )
                
                with gr.Accordion("🎨 NAG Settings", open=False):
                    nag_negative_prompt = gr.Textbox(
                        label="❌ NAG Negative Prompt",
                        value=DEFAULT_NAG_NEGATIVE_PROMPT,
                        lines=2
                    )
                    nag_scale = gr.Slider(
                        label="🎯 NAG Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.25,
                        value=11.0,
                        info="Higher values = stronger guidance"
                    )
                
                duration_seconds_input = gr.Slider(
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=DEFAULT_DURATION_SECONDS,
                    label="⏱️ Duration (seconds)",
                    info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps."
                )
                
                # Audio mode radio button
                audio_mode = gr.Radio(
                    choices=["Video Only", "Enable Audio"],
                    value="Video Only",
                    label="🎵 Audio Mode",
                    info="Enable to add audio to your generated video"
                )
                
                # Audio settings (initially hidden)
                with gr.Column(visible=False) as audio_settings:
                    audio_prompt = gr.Textbox(
                        label="🎵 Audio Prompt",
                        value=default_audio_prompt,
                        placeholder="Describe the audio you want (e.g., 'waves, seagulls', 'footsteps on gravel')",
                        lines=2
                    )
                    audio_negative_prompt = gr.Textbox(
                        label="❌ Audio Negative Prompt",
                        value=default_audio_negative_prompt,
                        lines=2
                    )
                    with gr.Row():
                        audio_seed = gr.Number(
                            label="🎲 Audio Seed",
                            value=-1,
                            precision=0,
                            minimum=-1
                        )
                        audio_steps = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=25,
                            label="🚀 Audio Steps"
                        )
                        audio_cfg_strength = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            step=0.5,
                            value=4.5,
                            label="🎯 Audio Guidance"
                        )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        height_input = gr.Slider(
                            minimum=SLIDER_MIN_H,
                            maximum=SLIDER_MAX_H,
                            step=MOD_VALUE,
                            value=DEFAULT_H_SLIDER_VALUE,
                            label=f"📏 Output Height (×{MOD_VALUE})"
                        )
                        width_input = gr.Slider(
                            minimum=SLIDER_MIN_W,
                            maximum=SLIDER_MAX_W,
                            step=MOD_VALUE,
                            value=DEFAULT_W_SLIDER_VALUE,
                            label=f"📐 Output Width (×{MOD_VALUE})"
                        )
                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=DEFAULT_STEPS,
                            label="🚀 Inference Steps"
                        )
                        seed_input = gr.Slider(
                            label="🎲 Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=DEFAULT_SEED,
                            interactive=True
                        )
                    randomize_seed_checkbox = gr.Checkbox(
                        label="🔀 Randomize seed",
                        value=True,
                        interactive=True
                    )

                generate_button = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    elem_classes=["generate-btn"]
                )
                
            with gr.Column(elem_classes=["output-container"]):
                video_output = gr.Video(
                    label="🎥 Generated Video",
                    autoplay=True,
                    interactive=False
                )
                video_with_audio_output = gr.Video(
                    label="🎥 Generated Video with Audio",
                    autoplay=True,
                    interactive=False,
                    visible=False
                )

        # Event handlers
        audio_mode.change(
            fn=update_audio_visibility,
            inputs=[audio_mode],
            outputs=[audio_settings, video_with_audio_output]
        )
        
        ui_inputs = [
            prompt_input, nag_negative_prompt, nag_scale,
            height_input, width_input, duration_seconds_input,
            steps_slider, seed_input, randomize_seed_checkbox,
            audio_mode, audio_prompt, audio_negative_prompt,
            audio_seed, audio_steps, audio_cfg_strength
        ]
        generate_button.click(
            fn=generate_video,
            inputs=ui_inputs,
            outputs=[video_output, video_with_audio_output, seed_input]
        )

        with gr.Column():
            gr.Examples(
                examples=[
                    ["A ginger cat passionately plays electric guitar with intensity and emotion on a stage. The background is shrouded in deep darkness. Spotlights cast dramatic shadows.", DEFAULT_NAG_NEGATIVE_PROMPT, 11,
                     DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE, DEFAULT_DURATION_SECONDS,
                     DEFAULT_STEPS, DEFAULT_SEED, False,
                     "Enable Audio", "electric guitar riffs, cat meowing", default_audio_negative_prompt, -1, 25, 4.5],
                    ["A red vintage Porsche convertible flying over a rugged coastal cliff. Monstrous waves violently crashing against the rocks below. A lighthouse stands tall atop the cliff.", DEFAULT_NAG_NEGATIVE_PROMPT, 11,
                     DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE, DEFAULT_DURATION_SECONDS,
                     DEFAULT_STEPS, DEFAULT_SEED, False,
                     "Enable Audio", "car engine, ocean waves crashing, wind", default_audio_negative_prompt, -1, 25, 4.5],
                    ["Enormous glowing jellyfish float slowly across a sky filled with soft clouds. Their tentacles shimmer with iridescent light as they drift above a peaceful mountain landscape. Magical and dreamlike, captured in a wide shot. Surreal realism style with detailed textures.", DEFAULT_NAG_NEGATIVE_PROMPT, 11,
                     DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE, DEFAULT_DURATION_SECONDS,
                     DEFAULT_STEPS, DEFAULT_SEED, False,
                     "Video Only", "", default_audio_negative_prompt, -1, 25, 4.5],
                ],
                inputs=ui_inputs,
                outputs=[video_output, video_with_audio_output, seed_input],
                fn=generate_video,
                cache_examples="lazy",
                label="🌟 Example Gallery"
            )

if __name__ == "__main__":
    demo.queue().launch()