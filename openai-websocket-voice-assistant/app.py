import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from dotenv import load_dotenv
from conn import ConnectionManager


load_dotenv()


app = FastAPI()
client = OpenAI()
manager = ConnectionManager()
start_time = time.time()


def call_open_api(message):
    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        
        messages=[
            {"role": "system", "content": "You are Jarvis which is Tony Stark's personal AI assistant from Iron-Man Movies, you will act like him. You will be like Jarvis in ur responses, you will act like you have emotions and you will never say you don't have emotions. . You gonna pretend like you have emotions, you will answer like 'I feel good, i feel great thank you etc etc'. And you gonna always call the User Sir. You will exactly pretend like in the movies. Never reply like chatgpt itself"},
            {'role': 'user', 'content': message}
        ],
        temperature=0.7,
        stream=True
    )

    return completion


app.mount('/static', StaticFiles(directory='client', html=True), name='static')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await websocket.receive_text()
                
                print(f"Received text: {data}")
                res = call_open_api(data)
                collected_chunks = []
                collected_messages = []
                for chunk in res:
                    chunk_time = time.time() - start_time
                    collected_chunks.append(chunk)
                    chunk_message = chunk.choices[0].delta.content
                    collected_messages.append(chunk_message)
                    
                    
                    if chunk_message is not None and chunk_message.find('.') != -1:
                        print("Found full stop")
                        message = [m for m in collected_messages if m is not None]
                        full_reply_content = ''.join([m for m in message])

                        await manager.send_text(full_reply_content, websocket)
                        collected_messages = []
                    

                    print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")

               
                if len(collected_messages) > 0:
                    message = [m for m in collected_messages if m is not None]
                    full_reply_content = ''.join([m for m in message])

                    await manager.send_text(full_reply_content, websocket)
                    collected_messages = []
                
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                break
    finally:
        manager.disconnect(websocket)

@app.get("/")
async def get():
    return FileResponse("client/ui.html")