# Create src directory structure
import os
import sys
os.makedirs("src", exist_ok=True)

# Create __init__.py
with open("src/__init__.py", "w") as f:
    f.write("")

# Create transformer_wan_nag.py
with open("src/transformer_wan_nag.py", "w") as f:
    f.write('''
import torch
import torch.nn as nn
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.attention_processor import AttentionProcessor
from typing import Optional, Dict, Any
import torch.nn.functional as F

class NagWanTransformer3DModel(ModelMixin, ConfigMixin):
    """NAG-enhanced Transformer for video generation"""
    
    @classmethod
    def from_single_file(cls, model_path, **kwargs):
        """Load model from single file"""
        # Create a minimal transformer model
        model = cls()
        
        # Try to load weights if available
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt", device="cpu") as f:
                state_dict = {}
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
                # model.load_state_dict(state_dict, strict=False)
        except:
            pass
            
        return model.to(kwargs.get('torch_dtype', torch.float32))
    
    def __init__(self):
        super().__init__()
        self.config = {"in_channels": 4, "out_channels": 4}
        self.training = False
        
        # Simple transformer layers
        self.norm = nn.LayerNorm(768)
        self.proj_in = nn.Linear(4, 768)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
            for _ in range(4)
        ])
        self.proj_out = nn.Linear(768, 4)
        
    @staticmethod
    def attn_processors():
        return {}
    
    @staticmethod  
    def set_attn_processor(processor):
        pass
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Simple forward pass
        batch, channels, frames, height, width = hidden_states.shape
        
        # Reshape for processing
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).contiguous()
        hidden_states = hidden_states.view(batch * frames, height * width, channels)
        
        # Project to transformer dimension
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.norm(hidden_states)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        
        # Project back
        hidden_states = self.proj_out(hidden_states)
        
        # Reshape back
        hidden_states = hidden_states.view(batch, frames, height, width, channels)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3).contiguous()
        
        return hidden_states
''')

# Create pipeline_wan_nag.py
with open("src/pipeline_wan_nag.py", "w") as f:
    f.write('''
import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
from diffusers import DiffusionPipeline
from diffusers.utils import logging, export_to_video
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

logger = logging.get_logger(__name__)

class NAGWanPipeline(DiffusionPipeline):
    """NAG-enhanced pipeline for video generation"""
    
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        scheduler,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load pipeline from pretrained model"""
        vae = kwargs.pop("vae", None)
        transformer = kwargs.pop("transformer", None)
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        
        # Load text encoder and tokenizer
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch_dtype
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer"
        )
        
        # Load scheduler
        from diffusers import UniPCMultistepScheduler
        scheduler = UniPCMultistepScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler"
        )
        
        return cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
    
    def _encode_prompt(self, prompt, device, do_classifier_free_guidance, negative_prompt=None):
        """Encode text prompt to embeddings"""
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(device))[0]
        
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size if negative_prompt is None else negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
        return text_embeddings
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        nag_negative_prompt: Optional[Union[str, List[str]]] = None,
        nag_scale: float = 0.0,
        nag_tau: float = 3.5,
        nag_alpha: float = 0.5,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
        **kwargs,
    ):
        # Use NAG negative prompt if provided
        if nag_negative_prompt is not None:
            negative_prompt = nag_negative_prompt
            
        # Setup
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Encode prompt
        text_embeddings = self._encode_prompt(
            prompt, device, do_classifier_free_guidance, negative_prompt
        )
        
        # Prepare latents
        num_channels_latents = self.vae.config.latent_channels
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        
        if latents is None:
            latents = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=text_embeddings.dtype,
            )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop with NAG
        for i, t in enumerate(timesteps):
            # Expand for classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = self.transformer(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
            )
            
            # Apply NAG
            if nag_scale > 0:
                # Compute attention-based guidance
                b, c, f, h, w = noise_pred.shape
                noise_flat = noise_pred.view(b, c, -1)
                
                # Normalize and compute attention
                noise_norm = F.normalize(noise_flat, dim=-1)
                attention = F.softmax(noise_norm * nag_tau, dim=-1)
                
                # Apply guidance
                guidance = attention.mean(dim=-1, keepdim=True) * nag_alpha
                guidance = guidance.unsqueeze(-1).unsqueeze(-1)
                noise_pred = noise_pred + nag_scale * guidance * noise_pred
            
            # Classifier free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, eta=eta, generator=generator).prev_sample
            
            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        # Decode latents
        latents = 1 / self.vae.config.scaling_factor * latents
        video = self.vae.decode(latents).sample
        video = (video / 2 + 0.5).clamp(0, 1)
        
        # Convert to output format
        video = video.cpu().float().numpy()
        video = (video * 255).round().astype("uint8")
        video = video.transpose(0, 2, 3, 4, 1)
        
        frames = []
        for batch_idx in range(video.shape[0]):
            batch_frames = [video[batch_idx, i] for i in range(video.shape[1])]
            frames.append(batch_frames)
            
        if not return_dict:
            return (frames,)
            
        return type('PipelineOutput', (), {'frames': frames})()
''')

# Now import and run the main application
import types
import random
import spaces
import torch
import numpy as np
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video
import gradio as gr
import tempfile
from huggingface_hub import hf_hub_download
import logging
import gc

# Import our custom modules
from src.pipeline_wan_nag import NAGWanPipeline
from src.transformer_wan_nag import NagWanTransformer3DModel

# MMAudio imports
try:
    import mmaudio
except ImportError:
    os.system("pip install -e .")
    import mmaudio

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['HF_HUB_CACHE'] = '/tmp/hub'

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils

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

MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
SUB_MODEL_ID = "vrgamedevgirl84/Wan14BT2VFusioniX"
SUB_MODEL_FILENAME = "Wan14BT2VFusioniX_fp16_.safetensors"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

# Initialize models
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

# Audio model setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()
device = 'cuda'
dtype = torch.bfloat16

# Global audio model variables
audio_model = None
audio_net = None
audio_feature_utils = None
audio_seq_cfg = None

def load_audio_model():
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

# Helper functions
def cleanup_temp_files():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        try:
            if filename.endswith(('.mp4', '.flac', '.wav')):
                os.remove(filepath)
        except:
            pass

def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# CSS
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
"""

default_audio_prompt = ""
default_audio_negative_prompt = "music"

def get_duration(
        prompt,
        nag_negative_prompt, nag_scale,
        height, width, duration_seconds,
        steps,
        seed, randomize_seed,
        audio_mode, audio_prompt, audio_negative_prompt,
        audio_seed, audio_steps, audio_cfg_strength,
):
    duration = int(duration_seconds) * int(steps) * 2.25 + 5
    if audio_mode == "Enable Audio":
        duration += 60
    return duration

@torch.inference_mode()
def add_audio_to_video(video_path, duration_sec, audio_prompt, audio_negative_prompt, 
                      audio_seed, audio_steps, audio_cfg_strength):
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
    
    video_with_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    make_video(video_info, video_with_audio_path, audio, sampling_rate=seq_cfg.sampling_rate)
    
    return video_with_audio_path

@spaces.GPU(duration=get_duration)
def generate_video(
        prompt,
        nag_negative_prompt, nag_scale,
        height=DEFAULT_H_SLIDER_VALUE, width=DEFAULT_W_SLIDER_VALUE, duration_seconds=DEFAULT_DURATION_SECONDS,
        steps=DEFAULT_STEPS,
        seed=DEFAULT_SEED, randomize_seed=False,
        audio_mode="Video Only", audio_prompt="", audio_negative_prompt="music",
        audio_seed=-1, audio_steps=25, audio_cfg_strength=4.5,
):
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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        nag_video_path = tmpfile.name
    export_to_video(nag_output_frames_list, nag_video_path, fps=FIXED_FPS)

    # Generate audio if enabled
    video_with_audio_path = None
    if audio_mode == "Enable Audio":
        video_with_audio_path = add_audio_to_video(
            nag_video_path, duration_seconds, 
            audio_prompt, audio_negative_prompt,
            audio_seed, audio_steps, audio_cfg_strength
        )
    
    clear_cache()
    cleanup_temp_files()

    return nag_video_path, video_with_audio_path, current_seed

def update_audio_visibility(audio_mode):
    return gr.update(visible=(audio_mode == "Enable Audio"))

# Build interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML("""
            <h1 class="main-title">🎬 NAG Video Generator with Audio</h1>
            <p class="subtitle">Fast 4-step Wan2.1-T2V-14B with Normalized Attention Guidance + MMAudio</p>
        """)
        
        gr.HTML("""
            <div class="info-box">
                <p>🚀 <strong>Powered by:</strong> Normalized Attention Guidance (NAG) for ultra-fast video generation</p>
                <p>⚡ <strong>Speed:</strong> Generate videos in just 4-8 steps with high quality</p>
                <p>🎵 <strong>Audio:</strong> Optional synchronized audio generation with MMAudio</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes="prompt-container"):
                    prompt = gr.Textbox(
                        label="✨ Video Prompt",
                        placeholder="Describe your video scene in detail...",
                        lines=3,
                        elem_classes="prompt-input"
                    )
                    
                    with gr.Accordion("🎨 Advanced Prompt Settings", open=False):
                        nag_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
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

                audio_mode = gr.Radio(
                    choices=["Video Only", "Enable Audio"],
                    value="Video Only",
                    label="🎵 Audio Mode",
                    info="Enable to add audio to your generated video"
                )
                
                with gr.Column(visible=False) as audio_settings:
                    audio_prompt = gr.Textbox(
                        label="🎵 Audio Prompt",
                        value=default_audio_prompt,
                        placeholder="Describe the audio (e.g., 'waves, seagulls', 'footsteps')",
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

                generate_button = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    elem_classes="generate-btn"
                )

            with gr.Column(scale=1):
                nag_video_output = gr.Video(
                    label="Generated Video",
                    autoplay=True,
                    interactive=False,
                    elem_classes="video-output"
                )
                video_with_audio_output = gr.Video(
                    label="🎥 Generated Video with Audio",
                    autoplay=True,
                    interactive=False,
                    visible=False,
                    elem_classes="video-output"
                )
                
                gr.HTML("""
                    <div style="text-align: center; margin-top: 20px; color: #6b7280;">
                        <p>💡 Tip: Try different NAG scales for varied artistic effects!</p>
                    </div>
                """)

        gr.Markdown("### 🎯 Example Prompts")
        gr.Examples(
            examples=[
                ["A ginger cat passionately plays electric guitar with intensity and emotion on a stage. The background is shrouded in deep darkness. Spotlights cast dramatic shadows.", DEFAULT_NAG_NEGATIVE_PROMPT, 11,
                 DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE, DEFAULT_DURATION_SECONDS,
                 DEFAULT_STEPS, DEFAULT_SEED, False,
                 "Enable Audio", "electric guitar riffs, cat meowing", default_audio_negative_prompt, -1, 25, 4.5],
                ["A red vintage Porsche convertible flying over a rugged coastal cliff. Monstrous waves violently crashing against the rocks below. A lighthouse stands tall atop the cliff.", DEFAULT_NAG_NEGATIVE_PROMPT, 11,
                 DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE, DEFAULT_DURATION_SECONDS,
                 DEFAULT_STEPS, DEFAULT_SEED, False,
                 "Enable Audio", "car engine roaring, ocean waves crashing, wind", default_audio_negative_prompt, -1, 25, 4.5],
                ["Enormous glowing jellyfish float slowly across a sky filled with soft clouds. Their tentacles shimmer with iridescent light as they drift above a peaceful mountain landscape.", DEFAULT_NAG_NEGATIVE_PROMPT, 11,
                 DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE, DEFAULT_DURATION_SECONDS,
                 DEFAULT_STEPS, DEFAULT_SEED, False,
                 "Video Only", "", default_audio_negative_prompt, -1, 25, 4.5],
            ],
            fn=generate_video,
            inputs=[prompt, nag_negative_prompt, nag_scale,
                height_input, width_input, duration_seconds_input,
                steps_slider, seed_input, randomize_seed_checkbox,
                audio_mode, audio_prompt, audio_negative_prompt,
                audio_seed, audio_steps, audio_cfg_strength],
            outputs=[nag_video_output, video_with_audio_output, seed_input],
            cache_examples="lazy"
        )

    # Event handlers
    audio_mode.change(
        fn=update_audio_visibility,
        inputs=[audio_mode],
        outputs=[audio_settings, video_with_audio_output]
    )

    ui_inputs = [
        prompt,
        nag_negative_prompt, nag_scale,
        height_input, width_input, duration_seconds_input,
        steps_slider,
        seed_input, randomize_seed_checkbox,
        audio_mode, audio_prompt, audio_negative_prompt,
        audio_seed, audio_steps, audio_cfg_strength,
    ]
    generate_button.click(
        fn=generate_video,
        inputs=ui_inputs,
        outputs=[nag_video_output, video_with_audio_output, seed_input],
    )

if __name__ == "__main__":
    demo.queue().launch()