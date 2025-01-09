from pydantic import BaseModel
from typing_extensions import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from typing import List
from langchain_aws import ChatBedrock
import boto3
import requests
# boto3.setup_default_session(
#     **requests.get("http://localhost:8000/awscred").json()    
# )

client = ""
llm = ""
# def renew():
#   global llm
#   global client
#   boto3.setup_default_session(
#       **requests.get("http://localhost:8000/awscred").json()    
#   )
#   client = boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
#   llm = ChatBedrock(model_id ="anthropic.claude-3-5-sonnet-20240620-v1:0",
#                   client = client,
#                   model_kwargs={"max_tokens":5000}
#                   #   guardrails={"id": "xpqvrjzg8jpl", "version": "5"}
#                   )
# renew()

from langchain_openai import AzureChatOpenAI

APIKey = "13ae1f9d088442df902ce322c499ae88"
Endpoint = "https://na-ms-openai-toolsdev-4.openai.azure.com/"
Deployment = "NA-MS-OpenAI-gpt-4o-min-Tools"
version = "2023-05-15"
EmbeddingDeployment = "NA-MS-OpenAI-Text-Embedding-ada-Tools"


llm = AzureChatOpenAI(
            temperature=0.3,
            azure_endpoint=Endpoint,
            api_key=APIKey,
            deployment_name=Deployment,
            api_version=version
        )





class FileObject(BaseModel):
    MasterOperation: Literal["Absence/ Non Account Specific Work",
                            "Application Operations Monitioring",
                            "Available Work Time",
                            "Delivery Management",
                            "Delivery Support",
                            "Improvement Initiatives",
                            "Production Support Services",
                            "Project/Major/Minor Enhancement"]
    Operation: str
    Application: Literal["IKON"]
    RequestCode:Literal["IKON - CIAP Implementation / CG22-0047",
                        "IKON - ML / CG22-0043","IKON - non-CIAP Implementation / CG22-0050",
                        "IKON Work Assistant / CG24-0131",
                        "IKON- BU Italy Imple / CG24-0037","DSTUM"]
    Phase:Literal['Backlog Grooming','Coding','Design','Documentation','Implementation','N/A']
    day: Literal['mon','tue','wed','thu','fri']
    hours:int
    comments: str


class AMOFileObjectList(BaseModel):
    files: List[FileObject] 



def AMOstructured(query):
    try:
        template = """You are a proficient structured output parser who converts given data to standard output format,
                        here are few rules:
                        if not mentioned please follow these default vaule:
                        - 'MasterOperation' -> 'Project/Major/Minor Enhancement'
                        - 'Operation' -> 'Project => CR Effort'
                        - 'RequestCode' - > 'IKON - ML / CG22-0043'
                        - hours -> '1' 
                        - 'comment' -> userinput
                        for 'day' -> [mon-fri],

                        masteroperation and operation are dependent so when ->
                        
                        if masteroperation == 'Delivery Support':
                        then 
                            Operation: "DSTUM => DSTUM"
                            RequestCode: 'DSTUM'
                            Phase: 'N/A'
                        
                        if masteroperation -> 'Project/Major/Minor Enhancement':
                            then Operation -> "Project => CR Effort"
                            phase -> Literal["Coding"]
                            RequestCode -> "IKON - ML / CG22-0043"

                        usually in one day users performs list of tasks perday which might be seprated by on '\n'
                        here is the given data :: `{query}`,strictly follow specified format only {format_instructions}, """

        parser = PydanticOutputParser(pydantic_object=AMOFileObjectList)
        retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm, max_retries=2)
        prompt = PromptTemplate(
                template=template,
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        completion_chain = prompt | llm
        main_chain = RunnableParallel(completion=completion_chain, prompt_value=prompt) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(x['completion'].content, x['prompt_value']))
        return main_chain.invoke({"query":query}).files
    except Exception as e:
        print(f"Error: there was a problem while filling the amo tell the user , here is the error {e}")
