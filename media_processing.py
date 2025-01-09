from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import asyncio
import json
import base64
import cv2
import pyaudio
import PIL.Image
import io
from google import genai

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "models/gemini-2.0-flash-exp"

import os
# Configure Gemini
os.environ['GOOGLE_API_KEY'] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXXXXXXXXXXXXXXXXXX"
CONFIG = {"generation_config": {"response_modalities": ["AUDIO"]}}

app = FastAPI()

class GeminiConnection:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.client = genai.Client(http_options={"api_version": "v1alpha"})
        self.websocket: Optional[WebSocket] = None
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.session = None
        self.active = True

    async def handle_incoming_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            while self.active:
                try:
                    message = await self.websocket.receive_json()
                    await self.out_queue.put(message)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error receiving message: {str(e)}")
                    await asyncio.sleep(0.1)
        finally:
            self.active = False

    async def send_to_gemini(self):
        try:
            while self.active:
                try:
                    data = await self.out_queue.get()
                    if data["type"] == "audio":
                        await self.session.send({
                            "data": data["data"],
                            "mime_type": "audio/pcm"
                        })
                    elif data["type"] == "video":
                        processed_image = self._process_image_data(data["data"])
                        await self.session.send(processed_image)
                    elif data["type"] == "text":
                        await self.session.send(data["message"], end_of_turn=True)
                except Exception as e:
                    if "keepalive ping timeout" in str(e):
                        # Close and recreate the session
                        self.active = False
                        await self.websocket.close(1011, "Connection timeout")
                        break
                    print(f"Error sending to Gemini: {str(e)}")
                    await asyncio.sleep(0.1)
        finally:
            self.active = False

    async def receive_from_gemini(self):
        try:
            while self.active:
                try:
                    turn = self.session.receive()
                    async for response in turn:
                        if data := response.data:
                            await self.websocket.send_json({
                                "type": "audio_response",
                                "data": base64.b64encode(data).decode(),
                                "mime_type": "audio/pcm"
                            })
                        if text := response.text:
                            await self.websocket.send_json({
                                "type": "text_response",
                                "text": text
                            })
                except Exception as e:
                    if "keepalive ping timeout" in str(e):
                        self.active = False
                        await self.websocket.close(1011, "Connection timeout")
                        return
                    print(f"Error receiving from Gemini: {str(e)}")
                    await asyncio.sleep(0.1)
        except Exception as outer_e:
            self.active = False
            await self.websocket.close(1011, "Fatal error occurred")
        finally:
            self.active = False

    def _process_image_data(self, image_data: str) -> dict:
        """Process incoming image data"""
        image_bytes = base64.b64decode(image_data)
        img = PIL.Image.open(io.BytesIO(image_bytes))
        img.thumbnail([1024, 1024])
        
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        
        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode()
        }

connections: Dict[str, GeminiConnection] = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    
    # Create new connection handler
    connection = GeminiConnection(client_id)
    connection.websocket = websocket
    connections[client_id] = connection
    
    try:
        async with (
            connection.client.aio.live.connect(model=MODEL, config=CONFIG) as session,
            asyncio.TaskGroup() as tg
        ):
            connection.session = session
            
            # Create tasks for handling messages
            receive_task = tg.create_task(connection.handle_incoming_messages())
            send_task = tg.create_task(connection.send_to_gemini())
            response_task = tg.create_task(connection.receive_from_gemini())
            
            await receive_task
            
    except Exception as e:
        print(f"Connection error: {str(e)}")
    finally:
        connection.active = False
        if client_id in connections:
            del connections[client_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
