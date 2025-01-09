import configparser
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
import boto3
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models.bedrock import BedrockChat
import os

# Load configuration
config = configparser.ConfigParser()
config.read('config.properties')

def get_llm(model_name: str):
    if model_name == "azureai":
        api_key = config['azure']['api_key']
        gpt_endpoint = config['azure']['endpoint']
        gpt_deployment = config['azure']['deployment']
        gpt_version = config['azure']['version']

        return AzureChatOpenAI(
            temperature=0.3,
            azure_endpoint=gpt_endpoint,
            api_key=api_key,
            deployment_name=gpt_deployment,
            api_version=gpt_version
        )
    
    elif model_name == "claude3-sonnet":
        AWS_ACCESS_KEY_ID = config['aws']['access_key_id']
        AWS_SECRET_ACCESS_KEY = config['aws']['secret_access_key']
        REGION_NAME = config['aws']['region']

        boto3.setup_default_session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=REGION_NAME
        )

        bedrock = boto3.client(service_name='bedrock-runtime')
        return BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock, model_kwargs={"max_tokens": 1500})
    
    elif model_name == "qwen":
        return Ollama(model="qwen2.5:1.5b")
    
    elif model_name == "nvidia":
        return Ollama(model="nemotron-mini:4b")
    
    elif model_name in ["gemini-2.0-flash-thinking-exp-1219","gemini-2.0-flash-exp","gemini-exp-1206"]:
        os.environ["GOOGLE_API_KEY"] = config['google']['api_key']
        return ChatGoogleGenerativeAI(model=model_name)
