import sys
import numpy as np
import logging
from typing import List, Tuple, Optional
import logging
import platform
from whisperlivekit.timed_objects import ASRToken, Transcript, ChangeSpeaker
from whisperlivekit.warmup import load_file
from .whisper import load_model, tokenizer
from .whisper.audio import TOKENS_PER_SECOND
import os
import gc
from pathlib import Path
logger = logging.getLogger(__name__)

import torch
from whisperlivekit.simul_whisper.config import AlignAttConfig
from whisperlivekit.simul_whisper.simul_whisper import PaddedAlignAttWhisper
from whisperlivekit.simul_whisper.whisper import tokenizer

try:
    from .mlx_encoder import mlx_model_mapping, load_mlx_encoder
    HAS_MLX_WHISPER = True
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print(f"""{"="*50}\nMLX Whisper not found but you are on Apple Silicon. Consider installing mlx-whisper for better performance: pip install mlx-whisper\n{"="*50}""")
    HAS_MLX_WHISPER = False
if HAS_MLX_WHISPER:
    HAS_FASTER_WHISPER = False
else:
    try:
        from faster_whisper import WhisperModel
        HAS_FASTER_WHISPER = True
    except ImportError:
        HAS_FASTER_WHISPER = False

def model_path_and_type(model_path):
    path = Path(model_path)
    
    compatible_whisper_mlx = False
    compatible_faster_whisper = False
    pt_path = path if path.is_file() and path.suffix.lower() == '.pt' else None
    
    if path.is_dir():
        for file in path.iterdir():
            if file.is_file():
                if file.name in ['weights.npz', "weights.safetensors"]:
                    compatible_whisper_mlx = True
                elif file.suffix.lower() == '.bin':
                    compatible_faster_whisper = True
                elif file.suffix.lower() == '.pt':
                    pt_path = file
    return pt_path, compatible_whisper_mlx, compatible_faster_whisper


class SimulStreamingOnlineProcessor:
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        logfile=sys.stderr,
    ):        
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer = []
        self.committed: List[ASRToken] = []
        self.last_result_tokens: List[ASRToken] = []
        self.load_new_backend()
        
        #can be moved
        if asr.tokenizer:
            self.model.tokenizer = asr.tokenizer

    def load_new_backend(self):
        model, fw_encoder = self.asr.get_new_model_instance()
        self.model = PaddedAlignAttWhisper(
            cfg=self.asr.cfg,
            loaded_model=model,
            mlx_encoder=self.asr.mlx_encoder,  # MLX encoder can be shared
            fw_encoder=fw_encoder,  # Use per-connection Faster-Whisper encoder
        )
        logger.info(f"Loaded new PaddedAlignAttWhisper backend with model on device {self.model.device}")

    def insert_silence(self, silence_duration, offset):
        """
        If silences are > 5s, we do a complete context clear. Otherwise, we just insert a small silence and shift the last_attend_frame
        """
        if silence_duration < 5:
            gap_silence = torch.zeros(int(16000*silence_duration))
            self.model.insert_audio(gap_silence)
            # self.global_time_offset += silence_duration
        else:
            self.process_iter(is_last=True) #we want to totally process what remains in the buffer.
            self.model.refresh_segment(complete=True)
            self.model.global_time_offset = silence_duration + offset


        
    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time):
        """Append an audio chunk to be processed by SimulStreaming."""
            
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        self.end = audio_stream_end_time #Only to be aligned with what happens in whisperstreaming backend.
        self.model.insert_audio(audio_tensor)

    def new_speaker(self, change_speaker: ChangeSpeaker):
            self.process_iter(is_last=True)
            self.model.refresh_segment(complete=True)
            self.model.speaker = change_speaker.speaker
            self.global_time_offset = change_speaker.start
            
    def get_buffer(self):
        concat_buffer = Transcript.from_tokens(tokens= self.buffer, sep='')
        return concat_buffer

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        """
        Process accumulated audio chunks using SimulStreaming.
        
        Returns a tuple: (list of committed ASRToken objects, float representing the audio processed up to time).
        """
        try:
            timestamped_words = self.model.infer(is_last=is_last)
            if self.model.cfg.language == "auto" and timestamped_words and timestamped_words[0].detected_language == None:
                self.buffer.extend(timestamped_words)
                return [], self.end
            
            self.committed.extend(timestamped_words)
            self.buffer = []
            return timestamped_words, self.end

            
        except RuntimeError as e:
            error_msg = str(e)
            if "size of tensor" in error_msg and "must match" in error_msg:
                # KV cache corruption - COMPLETELY RELOAD the backend
                logger.error(f"SimulStreaming KV cache corruption detected: {e}")
                logger.warning("CRITICAL: Reloading entire backend to recover from cache corruption...")
                try:
                    # Remove old hooks before destroying model
                    if hasattr(self.model, 'remove_hooks'):
                        self.model.remove_hooks()
                    
                    # Clear all state
                    self.buffer = []
                    self.committed = []
                    self.last_result_tokens = []
                    
                    # Reload a completely fresh backend with new model instance
                    self.load_new_backend()
                    
                    logger.info("✅ Backend successfully reloaded with fresh model instance")
                    return [], self.end
                except Exception as recovery_error:
                    logger.error(f"❌ Backend reload failed: {recovery_error}")
                    logger.exception("Recovery exception details:")
                    return [], self.end
            else:
                logger.exception(f"SimulStreaming processing error: {e}")
                return [], self.end
        except Exception as e:
            logger.exception(f"SimulStreaming processing error: {e}")
            return [], self.end

    def warmup(self, audio, init_prompt=""):
        """Warmup the SimulStreaming model."""
        try:
            self.model.insert_audio(audio)
            self.model.infer(True)
            self.model.refresh_segment(complete=True)
            logger.info("SimulStreaming model warmed up successfully")
        except Exception as e:
            logger.exception(f"SimulStreaming warmup failed: {e}")

    def __del__(self):
        # free the model and add a new model to stack.
        # del self.model
        gc.collect()
        torch.cuda.empty_cache()
        # self.asr.new_model_to_stack()
        self.model.remove_hooks()

class SimulStreamingASR():
    """SimulStreaming backend with AlignAtt policy."""
    sep = ""

    def __init__(self, logfile=sys.stderr, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}
        
        # GPU device configuration
        self.gpu_id = kwargs.get('gpu_id', None)
        self.device = kwargs.get('device', None)
        
        if self.device is None:
            if self.gpu_id is not None and torch.cuda.is_available():
                self.device = torch.device(f'cuda:{self.gpu_id}')
                logger.info(f"SimulStreamingASR using GPU {self.gpu_id}")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"SimulStreamingASR using default CUDA device")
            else:
                self.device = torch.device('cpu')
                logger.warning(f"SimulStreamingASR using CPU")
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.decoder_type is None:
            self.decoder_type = 'greedy' if self.beams == 1 else 'beam'

        self.fast_encoder = False
        
        pt_path, compatible_whisper_mlx, compatible_faster_whisper = None, True, True
        if self.model_path:
            pt_path, compatible_whisper_mlx, compatible_faster_whisper = model_path_and_type(self.model_path)
            
        elif self.model_size is not None:
            model_mapping = {
                'tiny': './tiny.pt',
                'base': './base.pt',
                'small': './small.pt',
                'medium': './medium.pt',
                'medium.en': './medium.en.pt',
                'large-v1': './large-v1.pt',
                'base.en': './base.en.pt',
                'small.en': './small.en.pt',
                'tiny.en': './tiny.en.pt',
                'large-v2': './large-v2.pt',
                'large-v3': './large-v3.pt',
                'large': './large-v3.pt'
            }
            pt_path = Path(model_mapping.get(self.model_size, f'./{self.model_size}.pt'))
        
        self.model_name = pt_path.name.replace(".pt", "")
        
        self.cfg = AlignAttConfig(
                tokenizer_is_multilingual= not self.model_name.endswith(".en"),
                segment_length=self.min_chunk_size,
                frame_threshold=self.frame_threshold,
                language=self.lan,
                audio_max_len=self.audio_max_len,
                audio_min_len=self.audio_min_len,
                cif_ckpt_path=self.cif_ckpt_path,
                decoder_type="beam",
                beam_size=self.beams,
                task=self.task,
                never_fire=self.never_fire,
                init_prompt=self.init_prompt,
                max_context_tokens=self.max_context_tokens,
                static_init_prompt=self.static_init_prompt,
        )  
        
        # Set up tokenizer for translation if needed
        if self.task == "translate":
            self.tokenizer = self.set_translate_task()
        else:
            self.tokenizer = None
        
        
            
    
        self.mlx_encoder, self.fw_encoder = None, None
        if not self.disable_fast_encoder:
            if HAS_MLX_WHISPER:
                print('Simulstreaming will use MLX whisper for a faster encoder.')
                if self.model_path and compatible_whisper_mlx:
                    mlx_model = self.model_path
                else:
                    mlx_model = mlx_model_mapping[self.model_name]
                self.mlx_encoder = load_mlx_encoder(path_or_hf_repo=mlx_model)
                self.fast_encoder = True
            elif HAS_FASTER_WHISPER and compatible_faster_whisper:
                print('Simulstreaming will use Faster Whisper for the encoder.')
                if self.model_path and compatible_faster_whisper:
                    fw_model = self.model_path
                else:
                    fw_model = self.model_name
                
                # CRITICAL FIX: Use assigned GPU device for Faster-Whisper encoder
                if self.gpu_id is not None:
                    # CTranslate2 expects GPU index as single integer
                    fw_device = 'cuda'
                    fw_device_index = self.gpu_id
                    logger.info(f"🎯 Faster-Whisper encoder targeting GPU {self.gpu_id}")
                else:
                    fw_device = 'auto'
                    fw_device_index = 0
                    logger.info("Faster-Whisper encoder using auto device")
                
                self.fw_encoder = WhisperModel(
                    fw_model,
                    device=fw_device,
                    device_index=fw_device_index,
                    compute_type='auto',
                )
                logger.info(f"✅ Faster-Whisper encoder loaded on GPU {fw_device_index}")
                self.fast_encoder = True

        self.models = [self.load_model() for i in range(self.preload_model_count)]


    def load_model(self):
        """Load a model and return (whisper_model, fw_encoder) tuple."""
        whisper_model = load_model(
            name=self.model_path if self.model_path else self.model_name,
            download_root=self.model_path,
            decoder_only=self.fast_encoder,
            custom_alignment_heads=self.custom_alignment_heads
            )
        
        # CRITICAL FIX: Create per-connection Faster-Whisper encoder instance
        # The encoder MUST be instantiated per connection to use the correct GPU
        fw_encoder_for_this_model = None
        if self.fast_encoder and HAS_FASTER_WHISPER and not self.mlx_encoder:
            if self.model_path:
                fw_model = self.model_path
            else:
                fw_model = self.model_name
            
            if self.gpu_id is not None:
                fw_device = 'cuda'
                fw_device_index = self.gpu_id
                logger.info(f"🔧 Creating Faster-Whisper encoder for GPU {self.gpu_id}")
            else:
                fw_device = 'auto'
                fw_device_index = 0
            
            fw_encoder_for_this_model = WhisperModel(
                fw_model,
                device=fw_device,
                device_index=fw_device_index,
                compute_type='auto',
            )
            logger.info(f"✅ Per-model Faster-Whisper encoder created on GPU {fw_device_index}")
        
        # Move decoder model to assigned GPU device
        if self.device is not None:
            whisper_model = whisper_model.to(self.device)
            logger.info(f"✅ Loaded Whisper decoder model on {self.device} (GPU {self.gpu_id if self.gpu_id is not None else 'default'})")
            
            # Log GPU memory after model load
            if torch.cuda.is_available() and self.gpu_id is not None:
                allocated = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
                logger.info(f"📊 GPU {self.gpu_id} memory after decoder load: {allocated:.2f} GB")
        
        warmup_audio = load_file(self.warmup_file)
        if warmup_audio is not None:
            warmup_audio = torch.from_numpy(warmup_audio).float()
            if self.device is not None:
                warmup_audio = warmup_audio.to(self.device)
            if self.fast_encoder:                
                temp_model = PaddedAlignAttWhisper(
                    cfg=self.cfg,
                    loaded_model=whisper_model,
                    mlx_encoder=self.mlx_encoder,
                    fw_encoder=fw_encoder_for_this_model,  # Use per-model encoder
                )
                temp_model.warmup(warmup_audio)
                temp_model.remove_hooks()
            else:
                # For standard encoder, use the original transcribe warmup
                warmup_audio = load_file(self.warmup_file)
                whisper_model.transcribe(warmup_audio, language=self.lan if self.lan != 'auto' else None)
        
        # Return both model and its dedicated encoder
        return (whisper_model, fw_encoder_for_this_model)
    
    def get_new_model_instance(self):
        """
        SimulStreaming cannot share the same backend because it uses global forward hooks on the attention layers.
        Therefore, each user requires a separate model instance, which can be memory-intensive. To maintain speed, we preload the models into memory.
        
        Returns:
            Tuple of (whisper_model, fw_encoder)
        """
        if len(self.models) == 0:
            self.models.append(self.load_model())
        model_tuple = self.models.pop()
        return model_tuple

    def new_model_to_stack(self):
        self.models.append(self.load_model())
        

    def set_translate_task(self):
        """Set up translation task."""
        if self.cfg.language == 'auto':
            raise Exception('Translation cannot be done with language = auto')
        return tokenizer.get_tokenizer(
            multilingual=True,
            language=self.cfg.language,
            num_languages=99,
            task="translate"
        )

    def transcribe(self, audio):
        """
        Warmup is done directly in load_model
        """
        pass
