from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from whisperlivekit import TranscriptionEngine, AudioProcessor, get_inline_ui_html, parse_args
from whisperlivekit.gpu_manager import gpu_manager
from whisperlivekit.performance_monitor import performance_monitor
import asyncio
import logging
import uuid
import time
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()

# DEBUG: Print argument values
logger.debug(f"Backend: {args.backend}")
logger.debug(f"CUDA available: {torch.cuda.is_available()}")
logger.debug(f"Preload model count: {getattr(args, 'preload_model_count', 'NOT_SET')}")

# Track active connections for cleanup
active_connections = {}

@asynccontextmanager
async def lifespan(app: FastAPI):    
    # Log GPU availability at startup
    logger.info("=" * 80)
    logger.info("WhisperLiveKit Server Starting")
    logger.info("=" * 80)
    gpu_manager.log_all_gpu_stats()
    
    # CRITICAL FIX: Model preloading causes server startup deadlock
    # Models will be loaded on-demand instead (first connection takes ~60s, subsequent are instant)
    logger.debug(f"Checking preloading conditions: backend={args.backend}, cuda_available={torch.cuda.is_available()}, preload_model_count={getattr(args, 'preload_model_count', 'NOT_SET')}")
    if args.backend == "simulstreaming" and torch.cuda.is_available() and args.preload_model_count > 0:  # ENABLED when preload_model_count > 0
        num_gpus = torch.cuda.device_count()
        models_per_gpu = args.preload_model_count if args.preload_model_count > 0 else 1
        
        logger.info(f"🚀 Scheduling PRELOADING of {models_per_gpu} models × {num_gpus} GPUs in background")
        
        try:
            from whisperlivekit.simul_whisper.backend import global_model_pool, create_model_loader_for_gpu
            
            # Build args dict for model loading
            model_args = {
                "disable_fast_encoder": False,
                "custom_alignment_heads": None,
                "frame_threshold": 25,
                "beams": 1,
                "decoder_type": None,
                "audio_max_len": 20.0,
                "audio_min_len": 0.0,
                "cif_ckpt_path": None,
                "never_fire": False,
                "init_prompt": None,
                "static_init_prompt": None,
                "max_context_tokens": None,
                "model_path": getattr(args, 'model_path', None),  # FIXED: Use getattr to avoid AttributeError
                "model_size": args.model_size,
                "min_chunk_size": args.min_chunk_size,
                "lan": args.lan,
                "task": args.task,
                "warmup_file": args.warmup_file,
                "preload_model_count": 0,  # Don't preload in ASR __init__
            }
            
            # Create model loaders for each GPU
            logger.info("🔧 Creating model loaders for each GPU...")
            loaders = {}
            for gpu_id in range(num_gpus):
                loaders[gpu_id] = create_model_loader_for_gpu(model_args, gpu_id)
                logger.info(f"✅ Model loader created for GPU {gpu_id}")
            
            # Preload models using the loaders in background task
            def preload_for_gpu(gpu_id):
                return loaders[gpu_id](gpu_id)
            
            # Schedule preloading in background to avoid blocking server startup
            import threading
            
            def background_preload():
                try:
                    logger.info("🔄 Starting background model preloading...")
                    global_model_pool.preload_models(num_gpus, models_per_gpu, preload_for_gpu)
                    logger.info(f"🎉 Background model preloading complete! All GPUs ready.")
                    # Log final GPU stats after preloading
                    gpu_manager.log_all_gpu_stats()
                except Exception as e:
                    logger.error(f"❌ Background model preloading failed: {e}")
                    logger.exception("Preload exception:")
                    logger.warning("⚠️ Models will load on-demand (slow!) for first connections")
            
            # Start preloading in background thread
            preload_thread = threading.Thread(target=background_preload, daemon=True)
            preload_thread.start()
            logger.info("🚀 Background model preloading started")
            
        except Exception as e:
            logger.error(f"❌ Model preloading setup failed: {e}")
            logger.exception("Preload setup exception:")
            logger.warning("⚠️ Server will continue but models will load on-demand (slow!)")
    else:
        if args.backend == "simulstreaming":
            logger.warning("⚠️ Model preloading DISABLED - models will load on-demand for each connection")
            logger.warning("⚠️ First connection will take ~60 seconds to load model, subsequent connections are instant")
    
    yield
    # Cleanup on shutdown
    logger.info("Server shutting down, cleaning up GPU allocations...")
    gpu_manager.clear_all()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


@app.get("/gpu-status")
async def gpu_status():
    """Get current GPU allocation and memory status."""
    stats = gpu_manager.get_all_gpu_stats()
    
    return {
        "total_gpus": gpu_manager.num_gpus,
        "total_connections": gpu_manager.get_total_active_connections(),
        "max_connections_per_gpu": gpu_manager.max_connections_per_gpu,
        "gpus": [
            {
                "gpu_id": gpu_id,
                "active_connections": stat.active_connections,
                "connection_ids": stat.connection_ids,
                "total_memory_gb": round(stat.total_memory_gb, 2),
                "allocated_memory_gb": round(stat.allocated_memory_gb, 2),
                "free_memory_gb": round(stat.free_memory_gb, 2),
                "utilization_percent": round(stat.utilization_percent, 1),
            }
            for gpu_id, stat in stats.items()
        ]
    }


@app.get("/performance")
async def performance_stats():
    """Get performance metrics and timing statistics."""
    return JSONResponse(performance_monitor.get_performance_summary())


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    gpu_stats = gpu_manager.get_all_gpu_stats()
    perf_summary = performance_monitor.get_performance_summary()
    
    # Calculate overall system health
    total_capacity = gpu_manager.num_gpus * gpu_manager.max_connections_per_gpu
    current_load = gpu_manager.get_total_active_connections()
    capacity_percent = (current_load / total_capacity * 100) if total_capacity > 0 else 0
    
    # Determine health status
    if capacity_percent < 70:
        health_status = "healthy"
    elif capacity_percent < 90:
        health_status = "degraded"
    else:
        health_status = "critical"
    
    return {
        "status": health_status,
        "timestamp": time.time(),
        "capacity": {
            "total_slots": total_capacity,
            "used_slots": current_load,
            "available_slots": total_capacity - current_load,
            "utilization_percent": round(capacity_percent, 1),
        },
        "gpus": {
            "total": gpu_manager.num_gpus,
            "with_connections": sum(1 for stat in gpu_stats.values() if stat.active_connections > 0),
        },
        "performance": perf_summary,
    }


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    gpu_stats = gpu_manager.get_all_gpu_stats()
    perf_summary = performance_monitor.get_performance_summary()
    
    metrics = []
    
    # GPU metrics
    metrics.append("# HELP whisperlivekit_gpu_total Total number of GPUs")
    metrics.append("# TYPE whisperlivekit_gpu_total gauge")
    metrics.append(f"whisperlivekit_gpu_total {gpu_manager.num_gpus}")
    
    metrics.append("# HELP whisperlivekit_connections_active Active WebSocket connections")
    metrics.append("# TYPE whisperlivekit_connections_active gauge")
    for gpu_id, stat in gpu_stats.items():
        metrics.append(f'whisperlivekit_connections_active{{gpu="{gpu_id}"}} {stat.active_connections}')
    
    metrics.append("# HELP whisperlivekit_gpu_memory_allocated_gb GPU memory allocated in GB")
    metrics.append("# TYPE whisperlivekit_gpu_memory_allocated_gb gauge")
    for gpu_id, stat in gpu_stats.items():
        metrics.append(f'whisperlivekit_gpu_memory_allocated_gb{{gpu="{gpu_id}"}} {stat.allocated_memory_gb:.2f}')
    
    metrics.append("# HELP whisperlivekit_gpu_memory_free_gb GPU memory free in GB")
    metrics.append("# TYPE whisperlivekit_gpu_memory_free_gb gauge")
    for gpu_id, stat in gpu_stats.items():
        metrics.append(f'whisperlivekit_gpu_memory_free_gb{{gpu="{gpu_id}"}} {stat.free_memory_gb:.2f}')
    
    # Performance metrics
    metrics.append("# HELP whisperlivekit_connections_total Total completed connections")
    metrics.append("# TYPE whisperlivekit_connections_total counter")
    metrics.append(f"whisperlivekit_connections_total {perf_summary.get('total_completed_connections', 0)}")
    
    if 'averages' in perf_summary:
        avg = perf_summary['averages']
        metrics.append("# HELP whisperlivekit_setup_time_ms_avg Average connection setup time in milliseconds")
        metrics.append("# TYPE whisperlivekit_setup_time_ms_avg gauge")
        metrics.append(f"whisperlivekit_setup_time_ms_avg {avg.get('setup_time_ms', 0)}")
    
    return "\n".join(metrics)


async def handle_websocket_results(websocket, results_generator, connection_id):
    """Consumes results from the audio processor and sends them via WebSocket."""
    result_count = 0
    try:
        logger.info(f"Connection {connection_id}: Starting to listen for transcription results...")
        async for response in results_generator:
            # Track first transcription
            performance_monitor.record_first_transcription(connection_id)
            performance_monitor.record_transcription(connection_id)
            result_count += 1
            
            logger.info(f"Connection {connection_id}: Sending transcription result #{result_count}: {response.to_dict()}")
            try:
                await websocket.send_json(response.to_dict())
                logger.info(f"Connection {connection_id}: Successfully sent result #{result_count}")
            except Exception as send_error:
                logger.error(f"Connection {connection_id}: Failed to send result #{result_count}: {send_error}")
                raise
        # when the results_generator finishes it means all audio has been processed
        logger.info(f"Connection {connection_id}: Results generator finished after {result_count} results. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info(f"Connection {connection_id}: WebSocket disconnected while handling results (sent {result_count} results).")
    except Exception as e:
        logger.exception(f"Connection {connection_id}: Error in WebSocket results handler after {result_count} results: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    # Generate unique connection ID
    connection_id = str(uuid.uuid4())
    gpu_id = None
    audio_processor = None
    websocket_task = None
    
    import time
    start_time = time.time()
    
    try:
        # STEP 1: Accept WebSocket IMMEDIATELY (fast response to client)
        await websocket.accept()
        accept_time = time.time()
        logger.info(f"Connection {connection_id}: WebSocket accepted ({(accept_time - start_time)*1000:.0f}ms)")
        
        # CRITICAL FIX: Send config IMMEDIATELY so client shows "Connected"
        # This makes the UI responsive while models are loading
        try:
            await websocket.send_json({
                "type": "config", 
                "useAudioWorklet": bool(args.pcm_input),
                "connection_id": connection_id,
                "status": "loading_models"
            })
        except Exception as e:
            logger.warning(f"Connection {connection_id}: Failed to send initial config: {e}")
        
        # Send status update
        await websocket.send_json({
            "type": "status",
            "message": "Allocating GPU..."
        })
        
        # STEP 2: Allocate GPU
        gpu_id = gpu_manager.allocate_gpu(connection_id)
        alloc_time = time.time()
        
        # Start performance tracking
        perf_metrics = performance_monitor.start_connection(connection_id, gpu_id if gpu_id is not None else -1)
        perf_metrics.accept_time = accept_time
        performance_monitor.record_gpu_allocation(connection_id)
        
        if gpu_id is None:
            logger.error(f"Connection {connection_id}: No GPU available")
            await websocket.send_json({
                "type": "error",
                "error": "No GPU resources available. All GPUs are at capacity."
            })
            await websocket.close(code=1008, reason="No GPU resources available")
            return
        
        logger.info(f"Connection {connection_id}: GPU {gpu_id} allocated ({(alloc_time - accept_time)*1000:.0f}ms)")
        
        # Notify client about GPU assignment
        await websocket.send_json({
            "type": "status",
            "message": f"Loading models on GPU {gpu_id}..."
        })
        
        # STEP 3: Load models (this is the slow part)
        transcription_engine = TranscriptionEngine(
            gpu_id=gpu_id,
            **vars(args),
        )
        engine_time = time.time()
        performance_monitor.record_engine_loaded(connection_id)
        logger.info(f"Connection {connection_id}: TranscriptionEngine loaded ({(engine_time - alloc_time)*1000:.0f}ms)")
        
        # CRITICAL FIX: Defer AudioProcessor creation to avoid blocking event loop
        logger.info(f"Connection {connection_id}: Deferring AudioProcessor creation...")
        audio_processor = None  # Will be created on first audio chunk
        audio_processor_engine = transcription_engine  # Store for later creation
        
        total_time = engine_time - start_time
        logger.info(f"Connection {connection_id}: Initial setup time: {total_time*1000:.0f}ms (GPU {gpu_id})")
        
        # Track active connection
        active_connections[connection_id] = {
            'gpu_id': gpu_id,
            'audio_processor': audio_processor,
            'audio_processor_engine': audio_processor_engine,  # Store for deferred creation
            'websocket': websocket
        }
        
        # Log current GPU stats
        gpu_manager.log_all_gpu_stats()

        # Send final ready status with GPU info
        try:
            await websocket.send_json({
                "type": "status",
                "message": "Ready to transcribe",
                "gpu_id": gpu_id,
                "connection_id": connection_id
            })
        except Exception as e:
            logger.warning(f"Connection {connection_id}: Failed to send ready status: {e}")
        
        # CRITICAL FIX: Defer results_generator creation until first audio chunk
        results_generator = None
        websocket_task = None
        
        logger.info(f"Connection {connection_id}: Ready to receive audio")

        first_audio_chunk = True
        while True:
            message = await websocket.receive_bytes()
            
            # Track first audio chunk
            performance_monitor.record_first_audio(connection_id)
            performance_monitor.record_audio_chunk(connection_id)
            
            logger.debug(f"Connection {connection_id}: Received audio chunk ({len(message)} bytes)")
            
            # CRITICAL FIX: Create AudioProcessor and start tasks on first audio chunk
            if first_audio_chunk:
                logger.info(f"Connection {connection_id}: First audio chunk received, creating AudioProcessor...")
                audio_processor = AudioProcessor(
                    transcription_engine=audio_processor_engine,
                )
                # Update the active connection reference
                active_connections[connection_id]['audio_processor'] = audio_processor
                
                results_generator = await audio_processor.create_tasks()
                websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator, connection_id))
                first_audio_chunk = False
                logger.info(f"Connection {connection_id}: AudioProcessor and tasks created successfully")
            
            await audio_processor.process_audio(message)
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"Connection {connection_id}: Client closed the connection.")
        else:
            logger.error(f"Connection {connection_id}: Unexpected KeyError: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info(f"Connection {connection_id}: WebSocket disconnected by client.")
    except Exception as e:
        logger.error(f"Connection {connection_id}: Unexpected error: {e}", exc_info=True)
    finally:
        logger.info(f"Connection {connection_id}: Cleaning up...")
        if websocket_task and not websocket_task.done():
            websocket_task.cancel()
        if websocket_task:
            try:
                await websocket_task
            except asyncio.CancelledError:
                logger.info(f"Connection {connection_id}: WebSocket results handler task cancelled.")
            except Exception as e:
                logger.warning(f"Connection {connection_id}: Exception awaiting websocket_task: {e}")
        
        if audio_processor:
            await audio_processor.cleanup()
        
        # End performance tracking
        performance_monitor.end_connection(connection_id)
        
        # Release GPU allocation
        if gpu_id is not None:
            gpu_manager.release_gpu(connection_id)
        
        # Remove from active connections
        if connection_id in active_connections:
            del active_connections[connection_id]
        
        # Log updated GPU stats
        gpu_manager.log_all_gpu_stats()
        
        logger.info(f"Connection {connection_id}: Cleanup complete.")

def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if args.forwarded_allow_ips:
        uvicorn_kwargs = { **uvicorn_kwargs, "forwarded_allow_ips" : args.forwarded_allow_ips }

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
