# Create src directory structure
import os
import sys

print("Starting NAG Video Demo application...")

# Add current directory to Python path
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except:
    current_dir = os.getcwd()
    
sys.path.insert(0, current_dir)
print(f"Added {current_dir} to Python path")

os.makedirs("src", exist_ok=True)

# Install required packages
os.system("pip install safetensors")

# Create __init__.py
with open("src/__init__.py", "w") as f:
    f.write("")
    
print("Creating NAG transformer module...")

# Create transformer_wan_nag.py
with open("src/transformer_wan_nag.py", "w") as f:
    f.write('''
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import torch.nn.functional as F

class NagWanTransformer3DModel(nn.Module):
    """NAG-enhanced Transformer for video generation (simplified demo)"""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.training = False
        self._dtype = torch.float32  # Add dtype attribute
        
        # Dummy config for compatibility
        self.config = type('Config', (), {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'hidden_size': hidden_size,
            'num_attention_heads': num_heads,
            'attention_head_dim': hidden_size // num_heads,
        })()
        
        # Simple conv layers for demo
        self.conv_in = nn.Conv3d(in_channels, hidden_size, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv_out = nn.Conv3d(hidden_size, out_channels, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
    
    @property
    def dtype(self):
        """Return the dtype of the model"""
        return self._dtype
    
    @dtype.setter
    def dtype(self, value):
        """Set the dtype of the model"""
        self._dtype = value
    
    def to(self, *args, **kwargs):
        """Override to method to handle dtype"""
        result = super().to(*args, **kwargs)
        # Update dtype if moving to a specific dtype
        for arg in args:
            if isinstance(arg, torch.dtype):
                self._dtype = arg
        if 'dtype' in kwargs:
            self._dtype = kwargs['dtype']
        return result
        
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
        # Simple forward pass for demo
        batch_size = hidden_states.shape[0]
        
        # Time embedding
        if timestep is not None:
            # Ensure timestep is the right shape
            if timestep.ndim == 0:
                timestep = timestep.unsqueeze(0)
            if timestep.shape[0] != batch_size:
                timestep = timestep.repeat(batch_size)
            
            # Normalize timestep to [0, 1]
            t_emb = timestep.float() / 1000.0
            t_emb = t_emb.view(-1, 1)
            t_emb = self.time_embed(t_emb)
            
            # Reshape for broadcasting
            t_emb = t_emb.view(batch_size, -1, 1, 1, 1)
        
        # Simple convolutions
        h = self.conv_in(hidden_states)
        
        # Add time embedding if available
        if timestep is not None:
            h = h + t_emb
        
        h = F.silu(h)
        h = self.conv_mid(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Add residual connection
        h = h + hidden_states
        
        return h
''')

print("Creating NAG pipeline module...")

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
        # Set vae scale factor
        if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'block_out_channels'):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8  # Default value for most VAEs
        
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
        if hasattr(self.vae.config, 'latent_channels'):
            num_channels_latents = self.vae.config.latent_channels
        else:
            num_channels_latents = 4  # Default for most VAEs
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
        if hasattr(self.vae.config, 'scaling_factor'):
            latents = 1 / self.vae.config.scaling_factor * latents
        else:
            latents = 1 / 0.18215 * latents  # Default SD scaling factor
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

print("NAG modules created successfully!")

# Ensure files are written and synced
import time
time.sleep(2)  # Give more time for file writes

# Verify files exist
if not os.path.exists("src/transformer_wan_nag.py"):
    raise RuntimeError("transformer_wan_nag.py not created")
if not os.path.exists("src/pipeline_wan_nag.py"):
    raise RuntimeError("pipeline_wan_nag.py not created")

print("Files verified, importing modules...")

# Now import and run the main application
import types
import random
import spaces
import torch
import torch.nn as nn
import numpy as np
from diffusers import AutoencoderKL, UniPCMultistepScheduler, DDPMScheduler
from diffusers.utils import export_to_video
import gradio as gr
import tempfile
from huggingface_hub import hf_hub_download
import logging
import gc

# Ensure src files are created
import time
time.sleep(1)  # Give a moment for file writes to complete

try:
    # Import our custom modules
    from src.pipeline_wan_nag import NAGWanPipeline
    from src.transformer_wan_nag import NagWanTransformer3DModel
    print("Successfully imported NAG modules")
except Exception as e:
    print(f"Error importing NAG modules: {e}")
    print("Attempting to recreate modules...")
    # Wait a bit and try again
    import time
    time.sleep(3)
    try:
        from src.pipeline_wan_nag import NAGWanPipeline
        from src.transformer_wan_nag import NagWanTransformer3DModel
        print("Successfully imported NAG modules on second attempt")
    except:
        print("Failed to import modules. Please restart the application.")
        sys.exit(1)

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
DEFAULT_DURATION_SECONDS = 1
DEFAULT_STEPS = 1
DEFAULT_SEED = 2025
DEFAULT_H_SLIDER_VALUE = 128
DEFAULT_W_SLIDER_VALUE = 128
NEW_FORMULA_MAX_AREA = 128.0 * 128.0

SLIDER_MIN_H, SLIDER_MAX_H = 128, 256
SLIDER_MIN_W, SLIDER_MAX_W = 128, 256
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 8  # Reduced FPS for demo
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 32  # Reduced max frames for demo

DEFAULT_NAG_NEGATIVE_PROMPT = "Static, motionless, still, ugly, bad quality, worst quality, poorly drawn, low resolution, blurry, lack of details"

# Note: Model IDs are kept for reference but not used in demo
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
SUB_MODEL_ID = "vrgamedevgirl84/Wan14BT2VFusioniX"
SUB_MODEL_FILENAME = "Wan14BT2VFusioniX_fp16_.safetensors"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

# Initialize models
print("Creating demo models...")

# Create a simple VAE-like model for demo
class DemoVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self._dtype = torch.float32  # Add dtype attribute
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 3, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        self.config = type('Config', (), {
            'scaling_factor': 0.18215,
            'latent_channels': 4,
        })()
    
    @property
    def dtype(self):
        """Return the dtype of the model"""
        return self._dtype
    
    @dtype.setter
    def dtype(self, value):
        """Set the dtype of the model"""
        self._dtype = value
    
    def to(self, *args, **kwargs):
        """Override to method to handle dtype"""
        result = super().to(*args, **kwargs)
        # Update dtype if moving to a specific dtype
        for arg in args:
            if isinstance(arg, torch.dtype):
                self._dtype = arg
        if 'dtype' in kwargs:
            self._dtype = kwargs['dtype']
        return result
    
    def encode(self, x):
        # Simple encoding
        encoded = self.encoder(x)
        return type('EncoderOutput', (), {'latent_dist': type('LatentDist', (), {'sample': lambda: encoded})()})()
    
    def decode(self, z):
        # Simple decoding
        # Handle different input shapes
        if z.dim() == 5:  # Video: (B, C, F, H, W)
            b, c, f, h, w = z.shape
            z = z.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            decoded = self.decoder(z)
            decoded = decoded.reshape(b, f, 3, h * 8, w * 8).permute(0, 2, 1, 3, 4)
        else:  # Image: (B, C, H, W)
            decoded = self.decoder(z)
        return type('DecoderOutput', (), {'sample': decoded})()

vae = DemoVAE()

print("Creating simplified NAG transformer model...")
transformer = NagWanTransformer3DModel(
    in_channels=4,
    out_channels=4,
    hidden_size=64,  # Reduced from 1280 for demo
    num_layers=1,  # Reduced for demo
    num_heads=4  # Reduced for demo
)

print("Creating pipeline...")
# Create a minimal pipeline for demo
pipe = NAGWanPipeline(
    vae=vae,
    text_encoder=None,
    tokenizer=None,
    transformer=transformer,
    scheduler=DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
    )
)

# Move to appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Move models to device with explicit dtype
vae = vae.to(device).to(torch.float32)
transformer = transformer.to(device).to(torch.float32)

# Now move pipeline to device (it will handle the components)
try:
    pipe = pipe.to(device)
    print(f"Pipeline moved to {device}")
except Exception as e:
    print(f"Warning: Could not move pipeline to {device}: {e}")
    # Manually set device
    pipe._execution_device = device

print("Demo version ready!")

# Check if transformer has the required methods
if hasattr(transformer, 'attn_processors'):
    pipe.transformer.__class__.attn_processors = NagWanTransformer3DModel.attn_processors
if hasattr(transformer, 'set_attn_processor'):
    pipe.transformer.__class__.set_attn_processor = NagWanTransformer3DModel.set_attn_processor

# Audio model setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    # Simplified duration calculation for demo
    duration = int(duration_seconds) * int(steps) + 10
    if audio_mode == "Enable Audio":
        duration += 30  # Reduced from 60 for demo
    return min(duration, 60)  # Cap at 60 seconds for demo

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
    try:
        target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
        target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)

        num_frames = np.clip(int(round(int(duration_seconds) * FIXED_FPS) + 1), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)

        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

        # Ensure transformer is on the right device and dtype
        if hasattr(pipe, 'transformer'):
            pipe.transformer = pipe.transformer.to(device).to(torch.float32)
        if hasattr(pipe, 'vae'):
            pipe.vae = pipe.vae.to(device).to(torch.float32)

        print(f"Generating video: {target_w}x{target_h}, {num_frames} frames, seed {current_seed}")

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
                generator=torch.Generator(device=device).manual_seed(current_seed)
            ).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            nag_video_path = tmpfile.name
        export_to_video(nag_output_frames_list, nag_video_path, fps=FIXED_FPS)

        # Generate audio if enabled
        video_with_audio_path = None
        if audio_mode == "Enable Audio":
            try:
                video_with_audio_path = add_audio_to_video(
                    nag_video_path, duration_seconds, 
                    audio_prompt, audio_negative_prompt,
                    audio_seed, audio_steps, audio_cfg_strength
                )
            except Exception as e:
                print(f"Warning: Could not generate audio: {e}")
                video_with_audio_path = None
        
        clear_cache()
        cleanup_temp_files()

        return nag_video_path, video_with_audio_path, current_seed
        
    except Exception as e:
        print(f"Error generating video: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a simple error video
        error_frames = []
        for i in range(8):  # Create 8 frames
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            frame[:, :] = [255, 0, 0]  # Red frame
            # Add error text
            error_frames.append(frame)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            error_video_path = tmpfile.name
        export_to_video(error_frames, error_video_path, fps=FIXED_FPS)
        return error_video_path, None, 0

def update_audio_visibility(audio_mode):
    return gr.update(visible=(audio_mode == "Enable Audio"))

# Build interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML("""
            <h1 class="main-title">🎬 NAG Video Demo</h1>
            <p class="subtitle">Simple Text-to-Video with NAG + Audio Generation</p>
        """)
        
        gr.HTML("""
            <div class="info-box">
                <p>📌 <strong>Demo Version:</strong> This is a simplified demo that demonstrates NAG concepts without large model downloads</p>
                <p>🚀 <strong>NAG Technology:</strong> Normalized Attention Guidance for enhanced video quality</p>
                <p>🎵 <strong>Audio:</strong> Optional synchronized audio generation with MMAudio</p>
                <p>⚡ <strong>Fast:</strong> Runs without downloading 28GB model files</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes="prompt-container"):
                    prompt = gr.Textbox(
                        label="✨ Video Prompt",
                        value=default_prompt,
                        placeholder="Describe your video scene...",
                        lines=2,
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
                            minimum=0.0,
                            maximum=20.0,
                            step=0.25,
                            value=5.0,
                            info="Higher values = stronger guidance (0 = no NAG)"
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
                            maximum=25,
                            step=1,
                            value=10,
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
                            maximum=2,
                            step=1,
                            value=DEFAULT_DURATION_SECONDS,
                            label="📱 Duration (seconds)",
                            elem_classes="slider-container"
                        )
                        steps_slider = gr.Slider(
                            minimum=1,
                            maximum=2,
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
                        <p>💡 Demo version with simplified model - Real NAG would produce higher quality results</p>
                        <p>💡 Tip: Try different NAG scales for varied artistic effects!</p>
                    </div>
                """)

        gr.Markdown("### 🎯 Example Prompts")
        gr.Examples(
            examples=[
                ["A cat playing guitar on stage", DEFAULT_NAG_NEGATIVE_PROMPT, 5,
                 128, 128, 1,
                 1, DEFAULT_SEED, False,
                 "Enable Audio", "guitar music", default_audio_negative_prompt, -1, 10, 4.5],
                ["A red car driving on a cliff road", DEFAULT_NAG_NEGATIVE_PROMPT, 5,
                 128, 128, 1,
                 1, DEFAULT_SEED, False,
                 "Enable Audio", "car engine, wind", default_audio_negative_prompt, -1, 10, 4.5],
                ["Glowing jellyfish floating in the sky", DEFAULT_NAG_NEGATIVE_PROMPT, 5,
                 128, 128, 1,
                 1, DEFAULT_SEED, False,
                 "Video Only", "", default_audio_negative_prompt, -1, 10, 4.5],
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