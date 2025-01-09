import os
import streamlit as st
import requests
from streamlit_chat import message
import uuid
import time
import langraph
import tools
import shutil
import CONSTANT
import sqlalchemy
import streamlit as st
import streamlit.components.v1 as components
import os
import base64
from PIL import Image
import requests
from io import BytesIO
import json


class KnowledgeBase:
    def __init__(self):
        self.bm25_retriever = None
        self.bm25_corpus = None
        self.bm25_stemmer = None
        self.qdrant = None
        self.session_id = None
OCR_API_URL = "http://localhost:8001/extract_text/"
CREATE_KB_URL = "http://localhost:8001/create_kb/"
LLM_API_URL = "http://localhost:8001/search_and_respond/"

st.markdown('''
    <style>
        [data-testid="stBottomBlockContainer"] {
            width: 100%;
            padding: 1rem 2rem 1rem 2rem;
            min-width: auto;
            max-width: initial;
        }
        [data-testid="column"]{
            position:sticky;
            align-content:center;
            padding:1rem;
        }
        [data-testid="stAppViewBlockContainer"]{
            width: 100%;
            padding: 2rem;
            min-width: auto;
            max-width: initial;
        }
        [data-testid="stVerticalBlock"]{
            gap:0.6rem;
        }
        .uploadedFiles {
            display: none;
        }   
        .stButton>button {
            margin: 0;
            padding: 0.15rem 0.4rem; /* Adjust padding for buttons */
        }
        .conversation-row {
            margin-bottom: 3px; /* Reduce spacing between rows */
        }
    </style>
''', unsafe_allow_html=True)

def database_configuration_section():
    st.subheader("Database Configuration")
    
    # Database Type Selection
    CONSTANT.db_type = st.selectbox("Database Type", [
        "PostgreSQL", 
        "MySQL", 
        "SQLite"
    ])
    
    # Connection Details
    col1, col2 = st.columns(2)
    
    with col1:
        CONSTANT.host = st.text_input("Host", placeholder="localhost or IP address",value="localhost")
        CONSTANT.port = st.text_input("Port", placeholder="Default port",value="5432")
        CONSTANT.Schema = st.text_input("Schema", placeholder="Schema",value="Public")
        
    with col2:
        CONSTANT.user = st.text_input("Username", placeholder="Database username",value="postgres")
        CONSTANT.password = st.text_input("Password", type="password", placeholder="Database password",value="Moodle")
        CONSTANT.database = st.text_input("Database Name", placeholder="Your database name",value="Vanna_db")
    
    # Test Connection Button
    if st.button("Test Database Connection"):
        try:
            # Placeholder for actual connection testing logic
            # You would replace this with actual database connection code
            test_connection(
                CONSTANT.db_type, 
                CONSTANT.host, 
                CONSTANT.port, 
                CONSTANT.user, 
                CONSTANT.password, 
                CONSTANT.database,
                CONSTANT.Schema
            )
        except Exception as e:
            st.error(f"Connection Failed: {e}")

    st.subheader("Web Search")
    CONSTANT.webpages_to_include = int(st.text_input("Web search result", placeholder="No of web result to include",value=2))

    st.subheader("AMO Details")
    CONSTANT.AMO_EMAIL = st.text_input("AMO EMAIL", placeholder="xxxxxxXXX@domain.com")
    CONSTANT.AMO_PASS = st.text_input("AMO password", type="password",placeholder="xxxxXXXXX")



def test_connection(db_type, host, port, username, password, database_name,schema):
    """
    Placeholder function to test database connection
    Replace with actual implementation based on your database type
    """
    # Construct connection string based on database type
    if db_type == "PostgreSQL":
        # PostgreSQL connection string with schema
        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database_name}?options=-csearch_path={schema}"
    elif db_type == "MySQL":
        # MySQL connection string with schema (set the database as the schema)
        connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}.{schema}"
    # Add more database type mappings as needed
    
    try:
        engine = sqlalchemy.create_engine(connection_string)
        
        with engine.connect() as connection:
            st.success("Database Connection Successful!")
    except Exception as e:
        st.error(f"Connection Error: {e}")



        
col1, col2 = st.columns([9, 1])
with col1:
    # create_media_streaming_component()
    pass
with col2:
    with st.popover("⚙️"):
        st.header("Document and Settings")
        CONSTANT.model_name = st.selectbox("Select LLM", ["azureai","gemini-2.0-flash-thinking-exp-1219","gemini-2.0-flash-exp","gemini-exp-1206", "claude3-sonnet", "qwen"])
        ocr_method = st.selectbox("Select OCR Method", ["tesseract", "openai", "florence", "google", "claude"])
        search_method = st.selectbox("Select Search Method", ["Embedding + Qdrant", "BM25S", "RRF"])
        database_configuration_section()
        


# Initialize session variables
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}  
if 'current_convo_id' not in st.session_state:
    st.session_state.current_convo_id = None  
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}  

message_and_id = {}


def get_conversation_title(messages):
    """Get the title from the first message or return a default title"""
    if messages and len(messages) > 0:
        first_msg = messages[0]["content"]
        return first_msg[:7] + "...." 
    return "New Conv...."

def initialize_session_state():
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
    if 'current_convo_id' not in st.session_state:
        st.session_state.current_convo_id = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'use_web_search' not in st.session_state:
        st.session_state.use_web_search = False
    if 'use_docx' not in st.session_state:
        st.session_state.use_docx = False
    if 'use_database' not in st.session_state:
        st.session_state.use_database = False



def create_media_streaming_component():
    html_content = """
    <div class="streaming-container">
    <div id="connectionStatus" class="status disconnected">
       Disconnected
    </div>
    
    <div class="button-group">
        <button id="connectBtn">Connect</button>
        <button id="startMediaBtn">Start</button>
        <button id="stopMediaBtn">Stop</button>
    </div>
    
    <video id="videoPreview" style="width: 100%; display: none" autoplay muted></video>
    
    <style>
        .streaming-container {
            padding: 16px;
            border-radius: 8px;
            background: #1e1e1e;
            margin: 0 0 16px 0;
            font-size: 14px;
            width: 275px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .status {
            margin: 4px 0 12px 0;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            text-align: center;
            transition: all 0.3s ease;
        }

        .connected { 
            background: rgba(46, 125, 50, 0.2);
            color: #4caf50;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }

        .disconnected { 
            background: rgba(198, 40, 40, 0.2);
            color: #D7CCC8 ;
            border: 1px solid rgba(244, 67, 54, 0.3);
        }

        .button-group { 
            display: flex;
            gap: 8px;
            margin: 4px 0;
        }

        button {
            flex: 1;
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s ease;
            text-transform: none;
        }

        button:hover:not(:disabled) {
            opacity: 0.9;
        }

        button:active:not(:disabled) {
            transform: translateY(1px);
        }

        button:disabled {
            background: #2d2d2d;
            cursor: not-allowed;
            color: #666;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        #connectBtn {
            background: #8BC34A  ;
            color: white;
        }

        #startMediaBtn {
            background: #FFB74D  ;
            color: white;
        }

        #stopMediaBtn {
            background: #E57373;
            color: white;
        }

        video {
            margin-top: 12px;
            border-radius: 6px;
            background: #000;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Media dialog styles */
        .media-dialog {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .media-options {
            background: #1e1e1e;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            width: 280px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .media-options h3 {
            margin: 0 0 16px 0;
            font-size: 16px;
            font-weight: 500;
            color: #fff;
            text-align: center;
        }

        .media-options button {
            width: 100%;
            margin-bottom: 8px;
            background: #2d2d2d;
            color: white;
        }

        .media-options button:last-child {
            background: #424242;
        }
    </style>

        <script>
        class MediaStreamingClient {
            constructor() {
                this.ws = null;
                this.mediaStream = null;
                this.videoTrack = null;
                this.audioTrack = null;
                this.isStreaming = false;
                this.clientId = 'streamlit-' + Math.random().toString(36).substr(2, 9);
                
                // Audio processing setup
                this.audioContext = null;
                this.audioProcessor = null;
                this.FORMAT = 16;
                this.CHANNELS = 1;
                this.SAMPLE_RATE = 16000;
                this.RECEIVE_SAMPLE_RATE = 24000;
                this.CHUNK_SIZE = 4000;
                this.audioQueue = [];
                this.isPlaying = false;
                
                this.setupElements();
                this.setupEventListeners();
            }

            setupElements() {
                this.videoPreview = document.getElementById('videoPreview');
                this.connectionStatus = document.getElementById('connectionStatus');
                this.connectBtn = document.getElementById('connectBtn');
                this.startMediaBtn = document.getElementById('startMediaBtn');
                this.stopMediaBtn = document.getElementById('stopMediaBtn');
            }

            setupEventListeners() {
                this.connectBtn.onclick = () => this.connect();
                this.startMediaBtn.onclick = () => this.showMediaOptions();
                this.stopMediaBtn.onclick = () => this.stopMedia();
            }

            showMediaOptions() {
                const dialog = document.createElement('div');
                dialog.className = 'media-dialog';
                dialog.innerHTML = `
                    <div class="media-options">
                        <h3>Choose Media Source</h3>
                        <button onclick="mediaClient.startMedia('audio')">Audio Only</button>
                        <button onclick="mediaClient.startMedia('camera')">Camera + Audio</button>
                        <button onclick="mediaClient.startMedia('screen')">Screen Share + Audio</button>
                        <button onclick="this.parentElement.parentElement.remove()">Cancel</button>
                    </div>
                `;
                document.body.appendChild(dialog);
            }

            async connect() {
                try {
                    this.ws = new WebSocket(`ws://localhost:8000/ws/${this.clientId}`);
                    this.stopMediaBtn.disabled = false;
                    this.ws.onopen = () => {
                        this.connectionStatus.textContent = 'Connected';
                        this.connectionStatus.className = 'status connected';
                        this.startMediaBtn.disabled = false;
                    };
                    
                    this.ws.onclose = () => {
                        this.connectionStatus.textContent = 'Disconnected';
                        this.connectionStatus.className = 'status disconnected';
                        this.startMediaBtn.disabled = true;
                        this.stopMediaBtn.disabled = true;
                        this.stopMedia();
                    };
                    
                    this.ws.onmessage = (event) => {
                        const response = JSON.parse(event.data);
                        if (response.type === 'audio_response') {
                            this.playAudioResponse(response.data);
                        }
                    };
                    
                } catch (error) {
                    console.error('Connection error:', error);
                }
            }

            async startMedia(type) {
                try {
                    const dialog = document.querySelector('.media-dialog');
                    if (dialog) dialog.remove();

                    const audioConstraints = {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    };

                    const audioStream = await navigator.mediaDevices.getUserMedia({
                        audio: audioConstraints,
                        video: false
                    });

                    this.audioTrack = audioStream.getAudioTracks()[0];
                    
                    if (type === 'camera') {
                        const videoStream = await navigator.mediaDevices.getUserMedia({
                            video: true
                        });
                        this.videoTrack = videoStream.getVideoTracks()[0];
                    } else if (type === 'screen') {
                        const screenStream = await navigator.mediaDevices.getDisplayMedia({
                            video: {
                                cursor: 'always'
                            },
                            audio: false
                        });
                        this.videoTrack = screenStream.getVideoTracks()[0];
                        
                        this.videoTrack.onended = () => {
                            if (this.streamType === 'screen') {
                                this.stopVideo();
                            }
                        };
                    }

                    this.mediaStream = new MediaStream();
                    this.mediaStream.addTrack(this.audioTrack);
                    if (this.videoTrack) {
                        this.mediaStream.addTrack(this.videoTrack);
                    }

                    this.streamType = type;
                    
                    if (this.videoTrack) {
                        this.videoPreview.srcObject = this.mediaStream;
                        this.videoPreview.style.display = 'block';
                    } else {
                        this.videoPreview.style.display = 'none';
                    }
                    
                    this.startMediaBtn.disabled = true;
                    this.stopMediaBtn.disabled = false;
                    
                    this.isStreaming = true;
                    if (this.videoTrack) {
                        this.startVideoCapture();
                    }
                    this.startAudioCapture();
                    
                } catch (error) {
                    console.error('Media error:', error);
                }
            }

            startVideoCapture() {
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = 640;
                canvas.height = 480;

                const captureFrame = () => {
                    if (!this.isStreaming || !this.videoTrack) return;
                    
                    context.drawImage(this.videoPreview, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob((blob) => {
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            const base64Data = reader.result.split(',')[1];
                            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                                this.ws.send(JSON.stringify({
                                    type: 'video',
                                    data: base64Data,
                                    source: this.streamType
                                }));
                            }
                        };
                        reader.readAsDataURL(blob);
                    }, 'image/jpeg', 0.8);
                    
                    setTimeout(captureFrame, 1000);
                };

                captureFrame();
            }

            startAudioCapture() {
                this.audioContext = new AudioContext({
                    sampleRate: 16000
                });
                
                const source = this.audioContext.createMediaStreamSource(this.mediaStream);
                this.audioProcessor = this.audioContext.createScriptProcessor(1024, 1, 1);

                source.connect(this.audioProcessor);
                this.audioProcessor.connect(this.audioContext.destination);

                this.audioProcessor.onaudioprocess = (e) => {
                    if (!this.isStreaming) return;
                    
                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcmData = new Int16Array(inputData.length);
                    
                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({
                            type: 'audio',
                            data: btoa(String.fromCharCode.apply(null, new Uint8Array(pcmData.buffer)))
                        }));
                    }
                };
            }

            stopVideo() {
                if (this.videoTrack) {
                    this.videoTrack.stop();
                    this.videoTrack = null;
                }
                this.videoPreview.srcObject = null;
                this.videoPreview.style.display = 'none';
                this.streamType = 'audio';
            }

            stopMedia() {
                this.isStreaming = false;
                if (this.mediaStream) {
                    this.mediaStream.getTracks().forEach(track => track.stop());
                }
                if (this.audioContext) {
                    this.audioContext.close();
                    this.audioContext = null;
                }
                this.audioProcessor.disconnect();
                this.audioProcessor = null;
                this.ws.close(1000, "Normal closure");
                this.ws.removeEventListener('message', this.handleMessage);  // Remove any listeners
                this.ws = null;  // Clear the reference
                this.videoPreview.srcObject = null;
                this.videoPreview.style.display = 'none';
                this.startMediaBtn.disabled = false;
                this.stopMediaBtn.disabled = true;
                this.connectionStatus.textContent = 'Disconnected';
                this.connectionStatus.className = 'status disconnected';
                this.startMediaBtn.disabled = true;
            }

            async playAudioResponse(base64Audio) {
                try {
                    await this.initAudioContext();
                    
                    const rawData = Uint8Array.from(atob(base64Audio), c => c.charCodeAt(0));
                    
                    for (let i = 0; i < rawData.length; i += this.CHUNK_SIZE * 2) {
                        const chunk = rawData.slice(i, i + this.CHUNK_SIZE * 2);
                        this.audioQueue.push(chunk);
                    }
                    
                    if (!this.isPlaying) {
                        this.isPlaying = true;
                        await this.processAudioQueue();
                    }
                } catch (error) {
                    console.error('Error processing audio:', error);
                }
            }

            async processAudioQueue() {
                while (this.audioQueue.length > 0 && this.isPlaying) {
                    const rawData = this.audioQueue.shift();
                    const audioBuffer = await this.createAudioBuffer(rawData);
                    await this.playBuffer(audioBuffer);
                }
                this.isPlaying = false;
            }

            async createAudioBuffer(rawData) {
                const float32Array = new Float32Array(rawData.length/2);
                for (let i = 0; i < rawData.length; i += 2) {
                    const sample = (rawData[i] & 0xff) | ((rawData[i + 1] & 0xff) << 8);
                    const signedSample = sample >= 0x8000 ? sample - 0x10000 : sample;
                    float32Array[i/2] = signedSample / 0x8000;
                }

                const buffer = this.audioContext.createBuffer(
                    this.CHANNELS,
                    float32Array.length,
                    this.RECEIVE_SAMPLE_RATE
                );
                buffer.getChannelData(0).set(float32Array);
                return buffer;
            }

            playBuffer(buffer) {
                return new Promise(resolve => {
                    const source = this.audioContext.createBufferSource();
                    source.buffer = buffer;
                    
                    const gainNode = this.audioContext.createGain();
                    gainNode.gain.value = 1.0;
                    
                    source.connect(gainNode);
                    gainNode.connect(this.audioContext.destination);
                    
                    source.onended = resolve;
                    source.start(0);
                });
            }
            
            async initAudioContext() {
                if (!this.audioContext) {
                    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: this.RECEIVE_SAMPLE_RATE,
                        latencyHint: 'interactive'
                    });
                }

                if (this.audioContext.state === 'suspended') {
                    await this.audioContext.resume();
                }

                this.scriptProcessor = this.audioContext.createScriptProcessor(
                    1024,
                    3,
                    this.CHANNELS
                );

                this.scriptProcessor.connect(this.audioContext.destination);
            }
        }

        // Initialize the client when the component loads
        const mediaClient = new MediaStreamingClient();
        </script>
    </div>
    """
    # Render the component
    components.html(html_content, height=250)




def render_sidebar():
    with st.sidebar:
        create_media_streaming_component()
        st.markdown("---")  
        st.session_state.use_web_search = st.checkbox("Web Search", value=False, key="web_search_checkbox")
        st.session_state.use_docx = st.checkbox("Files", value=False, key="docx_checkbox")
        st.session_state.use_database = st.checkbox("Database", value=False, key="database_checkbox")
        CONSTANT.WEB_SEARCH = st.session_state.use_web_search
        CONSTANT.DOC_SEARCH = st.session_state.use_docx 
        CONSTANT.DB_SEARCH = st.session_state.use_database 
        st.header("Manage Conversations")
        
        if st.button("➕ New Conversation", key="new_conv_btn"):
            new_convo_id = str(uuid.uuid4())
            st.session_state.conversations[new_convo_id] = []
            st.session_state.current_convo_id = new_convo_id
            # time.sleep(.5)
        
        st.markdown("---")  
        
        conversations_container = st.container()
        with conversations_container:
            st.markdown("""
            <style>
            .conversation-list {
                max-height: 300px;  /* Adjust height as needed */
                overflow-y: auto;
                padding-right: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="conversation-list">', unsafe_allow_html=True)
            
            for convo_id, messages in st.session_state.conversations.items():
                convo_title = get_conversation_title(messages)
                
                is_current = convo_id == st.session_state.current_convo_id
                button_style = (
                    "background-color: #2D2D2D;" if is_current 
                    else "background-color: transparent;"
                )
                col1, col2 = st.columns([2.5, 1])
                with col1:    
                    if st.button(
                        convo_title,
                        key=f"conv_{convo_id}",
                        use_container_width=True
                        # help=f"Click to switch to this conversation"
                    ):
                        st.session_state.current_convo_id = convo_id
                with col2:
                        if st.button(f"x", key=f"delete_{convo_id}"):
                            if os.path.exists(f"uploads/{convo_id}"):
                                shutil.rmtree(f"uploads/{convo_id}")
                            del st.session_state.conversations[convo_id]
                            st.rerun()
            
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")  # Divide
        
        
        if st.session_state.current_convo_id:
            if 'conversation_uploads' not in st.session_state:
                st.session_state.conversation_uploads = {}

            if st.session_state.current_convo_id not in st.session_state.conversation_uploads:
                st.session_state.conversation_uploads[st.session_state.current_convo_id] = {}

            # Get the current conversation's uploads
            current_conversation_files = st.session_state.conversation_uploads[st.session_state.current_convo_id]

            # File uploader
            uploaded_files = st.file_uploader(
                "Upload files (PDF/DOCX)",
                type=["pdf", "docx","png","jpeg"],
                accept_multiple_files=True,
                key=f"file_uploader_{st.session_state.current_convo_id}"  # Unique key per conversation,
            )

            if current_conversation_files:
                st.write("Uploaded Files:")
                for file_name, file_data in current_conversation_files.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(file_name)
                    with col2:
                        if st.button(f"x", key=f"delete_{file_name}"):
                            del current_conversation_files[file_name]
                            st.rerun()

            if uploaded_files:
                for file in uploaded_files:
                    if file.name not in current_conversation_files:
                        current_conversation_files[file.name] = {
                            "content": file,
                            "ocr_text": "",
                            "selected": False
                        }
                        # Optional: Show a success message for each new file
                        # st.success(f"Added {file.name} to current conversation")


                with st.expander("Uploaded Files"):
                    # Ensure current conversation has an entry in conversation_uploads
                    if st.session_state.current_convo_id not in st.session_state.conversation_uploads:
                        st.session_state.conversation_uploads[st.session_state.current_convo_id] = {}
                    
                    # Use the current conversation's uploaded files
                    current_conversation_files = st.session_state.conversation_uploads[st.session_state.current_convo_id]
                    CONSTANT.search_method = search_method
                    all_text = ""
                    for file_name, file_data in current_conversation_files.items():
                        files = {"file": (file_data["content"].name, file_data["content"].getvalue())}
                        ocr_data = {
                            "ocr_method": ocr_method,
                            "session_id": st.session_state.current_convo_id,
                        }
                        
                        ocr_response = requests.post(OCR_API_URL, files=files, data=ocr_data)
                        
                        if ocr_response.status_code == 200:
                            # Update the file data with OCR text in the conversation-specific dictionary
                            current_conversation_files[file_name]["ocr_text"] = ocr_response.json()
                            all_text += ocr_response.json()
                            # print(ocr_response.json())
                        else:
                            st.error(f"OCR error for {file_name}: {ocr_response.text}")
                    
                    # Create KB only if there's text to process
                    if all_text:
                        data = {
                            "session_id": st.session_state.current_convo_id,
                            "text": all_text,
                            "search_method": search_method
                        }
                        ocr_response = requests.post(CREATE_KB_URL, data=data)
                        
                        if ocr_response.status_code == 200:
                            print(f"Successful KB creation for {st.session_state.current_convo_id}")
                        else:
                            st.error(f"KB creation failed for {st.session_state.current_convo_id}")
                        



def initialize_session_state():
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
    if 'current_convo_id' not in st.session_state:
        st.session_state.current_convo_id = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'thinking' not in st.session_state:
        st.session_state.thinking = False

def handle_chat_input(prompt, current_conversation, message_container, combined_ocr_text=None):
    """Handle chat input and update messages in real-time"""
    if prompt:
        # Add the user's message to the conversation
        current_conversation.append({"role": "user", "content": prompt})
        
        # Update the message container with the new user message
        with message_container:
            unique_key = f"{st.session_state.current_convo_id}_{len(current_conversation)-1}_{uuid.uuid4()}"
            message(prompt, is_user=True, key=unique_key, avatar_style='no-avatar')
        
        # Show the "Bot is typing..." indicator
        typing_container = message_container.container()
        with typing_container:
            message("Bot is typing...", is_user=False, key="typing_indicator", avatar_style='no-avatar')

        try:
            CONSTANT.session_id = st.session_state.current_convo_id
            if combined_ocr_text:
                print("calling with ocr")
                try:    
                    response = langraph.graph.invoke(
                        {
                            "messages": prompt,
                            "session_id":st.session_state.current_convo_id,
                            "search_method":search_method,
                            "web_search":CONSTANT.WEB_SEARCH,
                            "docx_search":CONSTANT.DOC_SEARCH,
                            "db_search":CONSTANT.DB_SEARCH
                        },
                        config={"thread_id": st.session_state.current_convo_id}
                    )["messages"][-1].content
                except Exception as e:
                    response = e.message
            else:
                # Get the assistant's response
                try:
                    response = langraph.graph.invoke(
                        {
                            "messages": prompt,
                            "session_id":st.session_state.current_convo_id,
                            "web_search":CONSTANT.WEB_SEARCH,
                            "docx_search":CONSTANT.DOC_SEARCH,
                            "db_search":CONSTANT.DB_SEARCH
                        },
                        config={"thread_id": st.session_state.current_convo_id}
                    )["messages"][-1].content
                except Exception as e:
                    response = e.message
            # Add the assistant's response to the conversation
            current_conversation.append({"role": "assistant", "content": response})
            
            typing_container.empty()
            
            with message_container:
                unique_key = f"{st.session_state.current_convo_id}_{len(current_conversation)-1}_{uuid.uuid4()}"
                if response.endswith('.png') and len(response)==54:
                    create_clickable_image(response)
                    
                else:
                    if is_valid_json(response):
                        with st.chat_message("assistant"):
                            st.json(json.loads(response))
                    else:
                        message(response, is_user=False, key=unique_key, avatar_style='no-avatar')
                    
            st.rerun()
        except Exception as e:
            typing_container.error(f"Error processing request: {str(e)}")
        


def create_clickable_image(image_url, width=1200, max_height=800, caption=None):
    """
    Create a clickable image that opens in a popover with Streamlit.
    
    Parameters:
    - image_url (str): URL or local path of the image
    - width (int): Display width of the image
    - max_height (int, optional): Maximum height of the image
    - caption (str, optional): Caption for the image
    
    Returns:
    - None (displays image in Streamlit)
    """
    try:
        # Fetch image from URL or load local image
        if image_url.startswith(('http://', 'https://')):
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_url)
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image if max_height is specified
        if max_height:
            # Calculate aspect ratio
            aspect_ratio = image.width / image.height
            new_height = min(max_height, image.height)
            new_width = int(new_height * aspect_ratio)
            
            # Resize the image
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create columns for layout
        col1, col2 = st.columns([1,22])
        
        with col1:
            # Display clickable image in popover
            with st.popover(f"🔍"):
                st.image(image, width=width)
            
            
            # Optional: Thumbnail preview
            st.image(image, width=width//2, caption=caption)
    
    except Exception as e:
        st.error(f"Error loading image: {e}")
    

def _create_clickable_image(image_path, width=850):
    
    # Read the image file
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    popup_html = f"""
    <div>
    

    <div id="imageModal" class="modal" style="
         display:none;
         position:fixed;
         z-index:1;
         left:0;
         top:0;
         width:100%;
         height:100%;
         overflow:auto;
         background-color:rgba(0,0,0,0.9);">
        <div class="modal-content" style="
             display:block;
             width:100%;
             max-width:1200px;
             position:relative;
             max-height:550vh;
             overflow:auto;">
            <span onclick="closeModal()" style="
                 position:absolute;
                 top:15px;
                 right:35px;
                 color:#00B5E2;
                 font-size:40px;
                 font-weight:bold;
                 cursor:pointer;">×</span>
            <img src="data:image/png;base64,{encoded_image}"
                 style="width:100%; max-height:900px;" />
        </div>
    </div>
        <script>
        
        function openModal() {{
            document.getElementById('imageModal').style.display = "block";
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').style.display = "none";
        }}
        </script>
    </div>
    """
    
    # st.markdown(popup_html, unsafe_allow_html=True)
    st.components.v1.html(popup_html, height=550)
    

def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False
    
def main():
    initialize_session_state()
    render_sidebar()
    if st.session_state.current_convo_id is None:
        def get_image_base64(image_path):
                with open(image_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()

        # Get base64 encoded image
        img_base64 = get_image_base64("chip.png")

        # Your HTML with the base64 image
        icon = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .logo-container {{
                    padding: 20px;
                    display: flex;
                    justify-content: flex-start;
                    align-items: center;
                }}
                
                .logo {{
                    width: 100px;
                    height: auto;
                    animation: fadeIn 1s ease-in;
                }}
                
                @keyframes fadeIn {{
                    from {{ opacity: 0; }}
                    to {{ opacity: 1; }}
                }}
            </style>
        </head>
        <body>
            <div class="logo-container">
                <img src="data:image/png;base64,{img_base64}" alt="Logo" class="logo">
            </div>
        </body>
        </html>
        """

        st.components.v1.html(icon)
        welcome_text = "WELCOME"
        welcome_placeholder = st.empty()
        displayed_welcome = ""

        # Stream the WELCOME text
        for char in welcome_text:
            displayed_welcome += char
            welcome_placeholder.markdown(f"""
            <div style='
                padding: 0.5rem;
                margin-bottom: 1rem;'>
                <p style='
                    font-size: 3rem;
                    font-weight: 800;
                    margin: 0;
                    background: linear-gradient(90deg, #00a2ff, #ff00ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    letter-spacing: 0.2em;
                    text-shadow: 0 0 5px rgba(0, 162, 255, 0.2),
                                0 0 10px rgba(255, 0, 255, 0.2);
                    animation: glow 2s ease-in-out infinite alternate;'>
                    {displayed_welcome}▌
                </p>
            </div>
            <style>
                @keyframes glow {{
                    from {{
                        text-shadow: 0 0 5px rgba(0, 162, 255, 0.5),
                                    0 0 10px rgba(255, 0, 255, 0.3);
                    }}
                    to {{
                        text-shadow: 0 0 8px rgba(0, 162, 255, 0.8),
                                    0 0 12px rgba(255, 0, 255, 0.5);
                    }}
                }}
            </style>
            """, unsafe_allow_html=True)
            time.sleep(0.2)  # Slower speed for dramatic effect
    
        
        message_ = """Welcome to your all-in-one Agentic AI assistant! 
                Effortlessly perform web searches, query databases, search documents, 
                and engage in real-time conversations including screen share, video 
                and audio conversation. Start exploring the possibilities today!"""
    
        placeholder = st.empty()
        displayed_message = ""
        
        for char in message_:
            displayed_message += char
            placeholder.markdown(f"""
            <div style='padding: 1rem; 
                        border-radius: 0.5rem; 
                        border: 1px solid #f0f7ff;'>
                <p style='font-size: 1.1rem; margin: 0;'>
                    {displayed_message}▌
                </p>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.05)  # Adjust typing speed here
        
    else:
        current_conversation = st.session_state.conversations[st.session_state.current_convo_id]

        message_container = st.container()

        with message_container:
            for i, msg in enumerate(current_conversation):
                unique_key = f"{st.session_state.current_convo_id}_{i}_{uuid.uuid4()}"  
                if msg["role"] == "user":
                    message(msg["content"], is_user=True, key=unique_key, avatar_style='no-avatar')
                else:
                    if isinstance(msg["content"], str) and is_valid_json(msg["content"]):
                        with st.chat_message("assistant"):
                            st.json(json.loads(msg["content"]))
                    else:
                        # Existing image or text message logic
                        if msg["content"].endswith('.png') and len(msg["content"]) == 54:
                            create_clickable_image(msg["content"])
                            
                        else:
                            message(msg["content"], is_user=False, key=unique_key, avatar_style='no-avatar')

        combined_ocr_text = None
        if st.session_state.conversation_uploads[st.session_state.current_convo_id]:
            combined_ocr_text = "with ocr"
        
        
        prompt = st.chat_input("Ask a question about the document")
        if prompt:
            handle_chat_input(prompt,current_conversation, message_container, combined_ocr_text)
            prompt = None

if __name__ == "__main__":
    class KnowledgeBase:
        def __init__(self):
            self.bm25_retriever = None
            self.bm25_corpus = None
            self.bm25_stemmer = None
            self.qdrant = None
            self.session_id = None
    main()
