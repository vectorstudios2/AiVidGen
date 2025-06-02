import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel
import gradio as gr
import tempfile
import spaces
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import random
import logging
import torchaudio
import os

# MMAudio imports
try:
    import mmaudio
except ImportError:
    os.system("pip install -e .")
    import mmaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils

# Video generation model setup
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe.to("cuda")

causvid_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME)
pipe.load_lora_weights(causvid_path, adapter_name="causvid_lora")
pipe.set_adapters(["causvid_lora"], adapter_weights=[0.95])
pipe.fuse_lora()

# Audio generation model setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()
device = 'cuda'
dtype = torch.bfloat16

audio_model: ModelConfig = all_model_cfg['large_44k_v2']
audio_model.download_if_needed()
setup_eval_logging()

def get_audio_model() -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    seq_cfg = audio_model.seq_cfg
    net: MMAudio = get_my_mmaudio(audio_model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(audio_model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {audio_model.model_path}')

    feature_utils = FeaturesUtils(tod_vae_ckpt=audio_model.vae_path,
                                  synchformer_ckpt=audio_model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=audio_model.mode,
                                  bigvgan_vocoder_ckpt=audio_model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()
    return net, feature_utils, seq_cfg

audio_net, audio_feature_utils, audio_seq_cfg = get_audio_model()

# Constants
MOD_VALUE = 32
DEFAULT_H_SLIDER_VALUE = 512
DEFAULT_W_SLIDER_VALUE = 896
NEW_FORMULA_MAX_AREA = 480.0 * 832.0 

SLIDER_MIN_H, SLIDER_MAX_H = 128, 896
SLIDER_MIN_W, SLIDER_MAX_W = 128, 896
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 24
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81 

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, watermark, text, signature"
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

/* 이미지 업로드 영역 */
.image-upload {
    border: 2px dashed rgba(255, 255, 255, 0.3) !important;
    border-radius: 15px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    transition: all 0.3s ease !important;
}

.image-upload:hover {
    border-color: rgba(255, 255, 255, 0.5) !important;
    background: rgba(255, 255, 255, 0.1) !important;
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

def _calculate_new_dimensions_wan(pil_image, mod_val, calculation_max_area,
                                 min_slider_h, max_slider_h,
                                 min_slider_w, max_slider_w,
                                 default_h, default_w):
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return default_h, default_w

    aspect_ratio = orig_h / orig_w
    
    calc_h = round(np.sqrt(calculation_max_area * aspect_ratio))
    calc_w = round(np.sqrt(calculation_max_area / aspect_ratio))

    calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
    calc_w = max(mod_val, (calc_w // mod_val) * mod_val)
    
    new_h = int(np.clip(calc_h, min_slider_h, (max_slider_h // mod_val) * mod_val))
    new_w = int(np.clip(calc_w, min_slider_w, (max_slider_w // mod_val) * mod_val))
    
    return new_h, new_w

def handle_image_upload_for_dims_wan(uploaded_pil_image, current_h_val, current_w_val):
    if uploaded_pil_image is None:
        return gr.update(value=DEFAULT_H_SLIDER_VALUE), gr.update(value=DEFAULT_W_SLIDER_VALUE)
    try:
        new_h, new_w = _calculate_new_dimensions_wan(
            uploaded_pil_image, MOD_VALUE, NEW_FORMULA_MAX_AREA,
            SLIDER_MIN_H, SLIDER_MAX_H, SLIDER_MIN_W, SLIDER_MAX_W,
            DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE
        )
        return gr.update(value=new_h), gr.update(value=new_w)
    except Exception as e:
        gr.Warning("Error attempting to calculate new dimensions")
        return gr.update(value=DEFAULT_H_SLIDER_VALUE), gr.update(value=DEFAULT_W_SLIDER_VALUE)

def get_duration(input_image, prompt, height, width, 
                   negative_prompt, duration_seconds,
                   guidance_scale, steps,
                   seed, randomize_seed,
                   audio_mode, audio_prompt, audio_negative_prompt,
                   audio_seed, audio_steps, audio_cfg_strength,
                   progress):
    base_duration = 60
    if steps > 4 and duration_seconds > 2:
        base_duration = 90
    elif steps > 4 or duration_seconds > 2:
        base_duration = 75
    
    # Add extra time for audio generation
    if audio_mode == "Enable Audio":
        base_duration += 60
    
    return base_duration

@torch.inference_mode()
def add_audio_to_video(video_path, duration_sec, audio_prompt, audio_negative_prompt, 
                      audio_seed, audio_steps, audio_cfg_strength):
    """Add audio to video using MMAudio"""
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
    audio_seq_cfg.duration = duration
    audio_net.update_seq_lengths(audio_seq_cfg.latent_seq_len, audio_seq_cfg.clip_seq_len, audio_seq_cfg.sync_seq_len)
    
    audios = generate(clip_frames,
                      sync_frames, [audio_prompt],
                      negative_text=[audio_negative_prompt],
                      feature_utils=audio_feature_utils,
                      net=audio_net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=audio_cfg_strength)
    audio = audios.float().cpu()[0]
    
    # Save video with audio
    video_with_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    make_video(video_info, video_with_audio_path, audio, sampling_rate=audio_seq_cfg.sampling_rate)
    
    return video_with_audio_path

@spaces.GPU(duration=get_duration)
def generate_video(input_image, prompt, height, width, 
                   negative_prompt, duration_seconds,
                   guidance_scale, steps,
                   seed, randomize_seed,
                   audio_mode, audio_prompt, audio_negative_prompt,
                   audio_seed, audio_steps, audio_cfg_strength,
                   progress=gr.Progress(track_tqdm=True)):
    
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    
    num_frames = np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    resized_image = input_image.resize((target_w, target_h))

    # Generate video
    with torch.inference_mode():
        output_frames_list = pipe(
            image=resized_image, prompt=prompt, negative_prompt=negative_prompt,
            height=target_h, width=target_w, num_frames=num_frames,
            guidance_scale=float(guidance_scale), num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(current_seed)
        ).frames[0]

    # Save video without audio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)
    
    # Generate audio if enabled
    video_with_audio_path = None
    if audio_mode == "Enable Audio":
        progress(0.5, desc="Generating audio...")
        video_with_audio_path = add_audio_to_video(
            video_path, duration_seconds, 
            audio_prompt, audio_negative_prompt,
            audio_seed, audio_steps, audio_cfg_strength
        )
    
    return video_path, video_with_audio_path, current_seed

def update_audio_visibility(audio_mode):
    """Update visibility of audio-related components"""
    return gr.update(visible=(audio_mode == "Enable Audio"))

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes=["main-container"]):
        gr.Markdown("# ✨ Fast 4 steps Wan 2.1 I2V (14B) with CausVid LoRA + Audio")
        
        with gr.Row():
            with gr.Column(elem_classes=["input-container"]):
                input_image_component = gr.Image(
                    type="pil", 
                    label="🖼️ Input Image (auto-resized to target H/W)",
                    elem_classes=["image-upload"]
                )
                prompt_input = gr.Textbox(
                    label="✏️ Prompt", 
                    value=default_prompt_i2v,
                    lines=2
                )
                duration_seconds_input = gr.Slider(
                    minimum=round(MIN_FRAMES_MODEL/FIXED_FPS,1), 
                    maximum=round(MAX_FRAMES_MODEL/FIXED_FPS,1), 
                    step=0.1, 
                    value=2, 
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
                    negative_prompt_input = gr.Textbox(
                        label="❌ Negative Prompt", 
                        value=default_negative_prompt, 
                        lines=3
                    )
                    seed_input = gr.Slider(
                        label="🎲 Seed", 
                        minimum=0, 
                        maximum=MAX_SEED, 
                        step=1, 
                        value=42, 
                        interactive=True
                    )
                    randomize_seed_checkbox = gr.Checkbox(
                        label="🔀 Randomize seed", 
                        value=True, 
                        interactive=True
                    )
                    with gr.Row():
                        height_input = gr.Slider(
                            minimum=SLIDER_MIN_H, 
                            maximum=SLIDER_MAX_H, 
                            step=MOD_VALUE, 
                            value=DEFAULT_H_SLIDER_VALUE, 
                            label=f"📏 Output Height (multiple of {MOD_VALUE})"
                        )
                        width_input = gr.Slider(
                            minimum=SLIDER_MIN_W, 
                            maximum=SLIDER_MAX_W, 
                            step=MOD_VALUE, 
                            value=DEFAULT_W_SLIDER_VALUE, 
                            label=f"📐 Output Width (multiple of {MOD_VALUE})"
                        )
                    steps_slider = gr.Slider(
                        minimum=1, 
                        maximum=30, 
                        step=1, 
                        value=4, 
                        label="🚀 Inference Steps"
                    ) 
                    guidance_scale_input = gr.Slider(
                        minimum=0.0, 
                        maximum=20.0, 
                        step=0.5, 
                        value=1.0, 
                        label="🎯 Guidance Scale", 
                        visible=False
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
        
        input_image_component.upload(
            fn=handle_image_upload_for_dims_wan,
            inputs=[input_image_component, height_input, width_input],
            outputs=[height_input, width_input]
        )
        
        input_image_component.clear( 
            fn=handle_image_upload_for_dims_wan,
            inputs=[input_image_component, height_input, width_input],
            outputs=[height_input, width_input]
        )
        
        ui_inputs = [
            input_image_component, prompt_input, height_input, width_input,
            negative_prompt_input, duration_seconds_input,
            guidance_scale_input, steps_slider, seed_input, randomize_seed_checkbox,
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
                    ["peng.png", "a penguin playfully dancing in the snow, Antarctica", 896, 512, 
                     default_negative_prompt, 2, 1.0, 4, 42, False, 
                     "Video Only", "", default_audio_negative_prompt, -1, 25, 4.5],
                    ["forg.jpg", "the frog jumps around", 448, 832,
                     default_negative_prompt, 2, 1.0, 4, 42, False,
                     "Enable Audio", "frog croaking, water splashing", default_audio_negative_prompt, -1, 25, 4.5],
                ],
                inputs=ui_inputs, 
                outputs=[video_output, video_with_audio_output, seed_input], 
                fn=generate_video, 
                cache_examples="lazy",
                label="🌟 Example Gallery"
            )

if __name__ == "__main__":
    demo.queue().launch()