from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from whisperlivekit import TranscriptionEngine, AudioProcessor, get_inline_ui_html, parse_args
from whisperlivekit.gpu_manager import gpu_manager
import asyncio
import logging
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()

# Track active connections for cleanup
active_connections = {}

@asynccontextmanager
async def lifespan(app: FastAPI):    
    # Log GPU availability at startup
    logger.info("=" * 80)
    logger.info("WhisperLiveKit Server Starting")
    logger.info("=" * 80)
    gpu_manager.log_all_gpu_stats()
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


async def handle_websocket_results(websocket, results_generator, connection_id):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info(f"Connection {connection_id}: Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info(f"Connection {connection_id}: WebSocket disconnected while handling results.")
    except Exception as e:
        logger.exception(f"Connection {connection_id}: Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    # Generate unique connection ID
    connection_id = str(uuid.uuid4())
    gpu_id = None
    audio_processor = None
    websocket_task = None
    
    try:
        # Allocate GPU for this connection
        gpu_id = gpu_manager.allocate_gpu(connection_id)
        
        if gpu_id is None:
            logger.error(f"Connection {connection_id}: No GPU available")
            await websocket.close(code=1008, reason="No GPU resources available")
            return
        
        # Create GPU-specific transcription engine for this connection
        transcription_engine = TranscriptionEngine(
            gpu_id=gpu_id,
            **vars(args),
        )
        
        audio_processor = AudioProcessor(
            transcription_engine=transcription_engine,
        )
        
        await websocket.accept()
        logger.info(f"Connection {connection_id}: WebSocket opened on GPU {gpu_id}")
        
        # Track active connection
        active_connections[connection_id] = {
            'gpu_id': gpu_id,
            'audio_processor': audio_processor,
            'websocket': websocket
        }
        
        # Log current GPU stats
        gpu_manager.log_all_gpu_stats()

        try:
            await websocket.send_json({
                "type": "config", 
                "useAudioWorklet": bool(args.pcm_input),
                "gpu_id": gpu_id,
                "connection_id": connection_id
            })
        except Exception as e:
            logger.warning(f"Connection {connection_id}: Failed to send config to client: {e}")
                
        results_generator = await audio_processor.create_tasks()
        websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator, connection_id))

        while True:
            message = await websocket.receive_bytes()
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
