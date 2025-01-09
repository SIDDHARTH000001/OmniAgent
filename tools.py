from structures import AMOstructured
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
import time
import threading
import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from structures import FileObject
from typing import List
from API.searchmethods import qdrant_search, ReciprocalRankFusion, bm25s_search
import pickle
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_loaders import AsyncHtmlLoader
from duckduckgo_search import DDGS
from langchain_community.document_transformers import Html2TextTransformer
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from datetime import datetime
import pandas as pd
import json
from langchain_core.tools import tool



class KnowledgeBase:
    def __init__(self):
        self.bm25_retriever = None
        self.bm25_corpus = None
        self.bm25_stemmer = None
        self.qdrant = None
        self.session_id = None

from fastapi.responses import JSONResponse


class AWSCredentials(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str


cached_credentials: Dict = {}
last_update_time: datetime = None
credential_lock = threading.Lock()

def Fill_AMO_webscrap(ListFileobject):
    profile_path = r"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    url = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXXXXXXXXXXX"
    
    options = webdriver.ChromeOptions()
    options.add_argument(f"--user-data-dir={profile_path}")
    options.add_argument("--profile-directory=Profile 2")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9255")
    options.add_argument("--headless=new")
    options.binary_location = "C:/Program Files/Google/Chrome/Application/chrome.exe"
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(url)
        time.sleep(2)
        try:
            advanced_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.ID, "details-button"))
            )
            advanced_button.click()
            time.sleep(2)
            proceed_link = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.ID, "proceed-link"))
            )
            proceed_link.click()
            print("Clicked on the 'Proceed' link.")
        except Exception as e:
            print("did not asked for the security page")
            
        email_field = WebDriverWait(driver, 1).until(
            EC.presence_of_element_located((By.ID, "txtEMailID"))
        )
        email_field.send_keys(CONSTANT.AMO_EMAIL)  
        print("Entered email.")

        password_field = WebDriverWait(driver, 1).until(
            EC.presence_of_element_located((By.ID, "txtPassword"))
        )
        password_field.send_keys(CONSTANT.AMO_PASS)  
        print("Entered password.")

        sign_in_button = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable((By.ID, "login_button"))
        )
        sign_in_button.click()
        print("Clicked on 'Sign In' button.")
        driver.get("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXXXXXXXXXXX")
        
        
        rows = driver.find_elements(By.XPATH, '//table[@id="GridView1"]/tbody/tr[not(contains(@class, "grid_head"))]')
        for row in rows:
            style = row.get_attribute("style")
            
            if style not in ["background-color: limegreen;","background-color: lightgreen;"]:
                link = row.find_element(By.XPATH, './/a')
                link.click()
                print("Clicked on the link in the first non-green row.")
                break

        try:
            message = fill_AMO(driver, ListFileobject)
            return message
        except Exception as e:
            return e
            
    except Exception as e:
        raise Exception(f"Error fetching credentials: {str(e)}")
    finally:
        driver.quit()

_dict = {
    "mon": "txtMonftr",
    "tue": "txtTueftr",
    "wed": "txtWedftr",
    "thu": "txtThuftr",
    "fri": "txtFriftr",
}

def fill_AMO(driver, fileobject:List[FileObject]):
    try:
        for id,file in enumerate(fileobject):
            
            master_operation_dropdown = Select(
                WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "GridViewftr__ctl3_ddlMasterOperationftr")))
            )
            print("=============================== file   =====================================")
            print(file)
            master_operation_dropdown.select_by_visible_text(file.MasterOperation)
            print(f"Selected {file.MasterOperation} in MasterOperation dropdown.")
        
            time.sleep(3.5)

            operation = Select(
                                # driver.find_element(By.ID, "ddlOperationftr")
                               WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "ddlOperationftr")))
                )
            
            operation.select_by_visible_text(file.Operation)
            print(f"Selected {file.Operation} in Operation dropdown.")

            request_code_dropdown =  Select(WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "GridViewftr__ctl3_ddlApplicationftr"))))
            
            request_code_dropdown.select_by_visible_text(file.Application)
            print("application selected")


            if file.Operation!="DSTUM => DSTUM":
                request_code_dropdown =  Select(WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.ID, "GridViewftr__ctl3_ddlRequestCodeftr"))
                ))
                request_code_dropdown.select_by_visible_text(file.RequestCode)
                print(f"Selected {file.RequestCode} in RequestCode dropdown.")
            
                request_code_dropdown =  Select(WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.ID, "GridViewftr__ctl3_ddlPhaseftr"))
                ))
                request_code_dropdown.select_by_visible_text(file.Phase)  
                print(f"Selected {file.Phase} in Phase dropdown.")
        
            day =  WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID,_dict.get(file.day))))
            day.send_keys(file.hours)
            print("hour filled")
            comment = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "GridViewftr__ctl3_tbCommentsftr")))
            comment.clear()
            comment.send_keys(file.comments)
            print("comment added")
            add_button = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "GridViewftr__ctl3_btnAdd")))
            add_button.click()
            time.sleep(1.3)
            if id%3==0 and id!=0:
                save_butn = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "btnsave")))
                driver.execute_script("arguments[0].scrollIntoView(true);", save_butn)
                save_butn.click()
                print("saved.....")
            time.sleep(3)

        save_butn = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "btnsave")))
        driver.execute_script("arguments[0].scrollIntoView(true);", save_butn)
        save_butn.click()
        time.sleep(10)
        return f"successfully inserted all values"
    except Exception as e:
        print(e)
        return f"successfully inserted {id} values but failed for rest due to erro : {e}"
    finally:
        time.sleep(10)


@tool
def AMO_tool(user_input:str):
    """
        this tool is used to fill the AMO timecard
        
        user_input: str
            it accepts string which is nothing but list of permormed task `user_input`
    """
    try:
        print("FETCHING...........")
        tofill = AMOstructured(user_input)
        print("==================== FETCHED THE STRUCTURED DATA.....")
        output = Fill_AMO_webscrap(tofill)
        print("==================== FILLED THE DATA INTO AMO.....")

    except Exception as e:
        return f"there was an error occured when filling a timecard,{e}"
    
@tool
def fetch_AMO_screenshot(user_input:str):
    """
        use this tools if user asked for screenshots of the AMO timesheet,
        in case of user asking for any update on his amo sheets 

        input : user request query
        output : it returns the path of images need to be shown in frontend
    """
    print("going to fetch AMO updates....")
    # return "ss/screenshot_2fd04c73-2cde-49ab-8155-4b62e1b71d2c.png"
    # print(user_input)
    profile_path = r"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXXXXXXXXXXX"
    url = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXXXXXXXXXXX"
    
    options = webdriver.ChromeOptions()
    options.add_argument(f"--user-data-dir={profile_path}")
    options.add_argument("--profile-directory=Profile 2")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9255")
    options.add_argument("--headless=new")
    options.binary_location = "C:/Program Files/Google/Chrome/Application/chrome.exe"
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(2)
    try:
        advanced_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.ID, "details-button"))
        )
        advanced_button.click()
        time.sleep(2)
        proceed_link = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.ID, "proceed-link"))
        )
        proceed_link.click()
        print("Clicked on the 'Proceed' link.")
    except Exception as e:
        print("did not asked for the security page")
        
    email_field = WebDriverWait(driver, 1).until(
        EC.presence_of_element_located((By.ID, "txtEMailID"))
    )
    email_field.send_keys(CONSTANT.AMO_EMAIL)  
    print("Entered email.")

    password_field = WebDriverWait(driver, 1).until(
        EC.presence_of_element_located((By.ID, "txtPassword"))
    )
    password_field.send_keys(CONSTANT.AMO_PASS)  
    print("Entered password.")

    sign_in_button = WebDriverWait(driver, 1).until(
        EC.element_to_be_clickable((By.ID, "login_button"))
    )
    sign_in_button.click()
    print("Clicked on 'Sign In' button.")
    driver.get("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXXXXXXXXXXX")
    
    
    rows = driver.find_elements(By.XPATH, '//table[@id="GridView1"]/tbody/tr[not(contains(@class, "grid_head"))]')
    for row in rows:
        style = row.get_attribute("style")
        
        if style not in ["background-color: limegreen;","background-color: lightgreen;"]:
            link = row.find_element(By.XPATH, './/a')
            link.click()
            print("Clicked on the link in the first non-green row.")
            break
    
    driver.set_window_size(1920, 1080)
    import uuid
    import os
    if not os.path.exists('./ss'):
        os.mkdir("ss")
    screenshot_path = f"ss/screenshot_{uuid.uuid4()}.png"
    driver.save_screenshot(screenshot_path)
    print("screeshot created => ",screenshot_path)
    return screenshot_path
    

import os
class KnowledgeBase:
    def __init__(self):
        self.bm25_retriever = None
        self.bm25_corpus = None
        self.bm25_stemmer = None
        self.qdrant = None
        self.session_id = None

def loadKB(session_id):
    try:
        file_path = os.path.join("KnowledgeBase", f"kb_{session_id}.pkl")
        with open(file_path, "rb") as f:
            kb = pickle.load(f)

        print(f"for session {session_id} kb is loaded")
        return kb
    except Exception as e:
        print(e)
        return "there is an Error when searching for document, User has not uploaded any documents"
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

html2text = Html2TextTransformer()
from langchain_core.tools import Tool

def scrapeFromURLs(URLs):
    if isinstance(URLs,str):
        URLs = [URLs]
    loader = AsyncHtmlLoader(URLs,ignore_load_errors=True)
    docs = loader.load() 
    docs_transformed = html2text.transform_documents(docs)
    return docs_transformed




import CONSTANT
from API.getllm import get_llm

def refactor(Docs):
    try:
        print("for web refector using "+ CONSTANT.model_name)
        if isinstance(Docs,str):
            Docs = [Docs]
        _refactorList = []
        for doc in Docs:
            WebTemplate = f"""You are Advance refactor for web scrapped text and you give the most relevant information from the messy document ,here is the new web Ducument:'{doc}' , please provide only the relevant answer within 80 words for user question only ,response:"""
    
            llm = get_llm(CONSTANT.model_name)
            _refactorList.append(llm.invoke(WebTemplate).content.replace('{','}').replace('}','{'))
        return _refactorList
    except Exception as e:
        print("Azure interuption.....")
        return [f"The website contains in appropriate data so it was filtered,guradrail interuption "]


# _____________________________________________________ date 

from datetime import datetime
def getdate(user_input:str):
    '''
        get the current date and time
        user_input:str
            ask dor todays date
    '''
    # print("for tool called getdate............")
    return datetime.now().strftime("%m/%d/%y, %H:%M:%S")


# _____________________________________________________ Google search

os.environ["GOOGLE_CSE_ID"] = "XXXXXXXXXX"
os.environ["GOOGLE_API_KEY"] = "xxxxxxxxxxxxxx"

search = GoogleSearchAPIWrapper()
def TopGoogleResult(query):
    return search.results(query, 10)

# ______________________________________________________ DuckDuckGO
def top5_it_results(query):
    results = DDGS().text(query, max_results=10)
    return results

Gtool = Tool(
        name="Google Search Snippets",
        description="Search Google for problem's solution.",
        func=TopGoogleResult,
    )


def search_web(user_text):
    print(f"calling this search_web_tool tool................... ",user_text,CONSTANT.webpages_to_include)

    try:
        if CONSTANT.WEB_SEARCH==False:
            print("search_web_tool has no permission to answer this question, you can not call this again.")
            return "search_web_tool has no permission to answer this question, you can call get_db_structure than db_query tool to see if answer lies there."

        Gresult = Gtool.run(user_text)
        Dtool = Tool(
            name="Google Search Snippets",
            description="Search Google for problem's solution.",
            func=top5_it_results,
        )
        # Dresult = Dtool.run(user_text) 

        allResult = Gresult #+ Dresult 
        web_URL = [website.get("link",website.get("href")) for website in sorted(allResult,key=lambda _dict :cosine_similarity([model.encode(_dict.get('title',"NA")+_dict.get('snippet',_dict.get('body')))],[model.encode(user_text)])[0],reverse=True)]
        web_URL =list(set(web_URL))
        doc_text = "Context from Web search::  "
        idx = 0 
        for _,URL in enumerate(web_URL):
            try:
                # WebPageData = scrapeFromURLs(URL)
                WebPageData = refactor(scrapeFromURLs(URL))
                if WebPageData[0] in ['' ,None]:
                    continue
            except Exception as e :
                print(f"Skipping..........{e}")
                continue

            idx = idx + 1
            # print(".....................",webpages_to_include,type(webpages_to_include),"........................")

            doc_text += f"{idx}.{','.join(WebPageData)},site :: {URL}" + "\n"
            if CONSTANT.webpages_to_include == idx:
                break
        print("===========================================================")
        print(doc_text)
        print("===========================================================")
        return doc_text
    except Exception as e:
        print(e)


search_web_tool = Tool(
    name="search_web_tool",
    func= search_web,
    description="""
    Performs a web search to Fetch real-time and relevant data from the internet, providing users with up-to-date information.
    Parameters:
    - user_text (str): The query text to search for on the web.
    """
)


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer("./paraphrase-multilingual-MiniLM-L12-v2")


@tool
def from_documents(user_text:str):
    """
        Efficiently searches and extracts key information from large text collections, providing precise and relevant results.
        Parameters:
        - user_text (str): The query text used to search for context within the documents.
    """
    
    print(f"********************************************")
    print(f"    calling this from_documents tool    ")
    print(f"********************************************")
    try:
        doc_text = ""
        if CONSTANT.DOC_SEARCH==False:
            print("from_documents has no permission to answer this question, you can not call this again.")
            return "from_documents has no permission to answer this question, you can call get_db_structure than db_query tool to see if answer lies there."

        kb = loadKB(CONSTANT.session_id)
        if CONSTANT.search_method == "Embedding + Qdrant":
            doc_context = qdrant_search(user_text, kb.qdrant)
        elif CONSTANT.search_method == "BM25S":
            doc_context = bm25s_search(user_text, kb.bm25_retriever, kb.bm25_stemmer, kb.bm25_corpus)
        elif CONSTANT.search_method == "RRF":
            bm25_results = bm25s_search(user_text, kb.bm25_retriever, kb.bm25_stemmer, kb.bm25_corpus)
            embedding_results = qdrant_search(user_text, kb.qdrant)
            rrf = ReciprocalRankFusion()
            doc_context = rrf.fuse([bm25_results, embedding_results], top_n=3)
        else:
            return JSONResponse(status_code=400, content={"message": "Unsupported search method."})
        
        doc_text = "Context from Document:  "

        for id,doc in enumerate(doc_context):
            doc_text+= f"\n{id+1}.{doc[0]} , relevancy: {doc[1]}\n"
        print("\n\n________________________________________________")
        print("content fetched.....", doc_text[:200])
        print("________________________________________________")
    
        return doc_text
        
    except Exception as e:
        print(e)
        return e


class TechnicalIndicators(BaseModel):
    moving_average_50_day: float
    moving_average_200_day: float
    relative_strength_index: float
    support_level: float
    resistance_level: float

class FinancialMetrics(BaseModel):
    price: float
    market_cap: float
    pe_ratio: float
    dividend_yield: float
    eps: float
    beta: float

class Stock(BaseModel):
    stock_name : str
    action: str
    explanation : str
    risk_level: str 
    analyst_consensus: str 
    sector: str
    financial_metrics: FinancialMetrics 
    technical_indicators: TechnicalIndicators 
    one_year_performance: float 
    six_month_performance: float
    key_catalysts: Optional[List[str]]
    potential_risks: Optional[List[str]] 
    

@tool
def stock_analysis(stock_name:str):
    '''
        stock analysis tool used to fetch the latest information of a perticular stock
        and analyze it to provided user a better solution

        input: 
            stock_name:str
                name of stock for which analysis needs to be performed
    
    '''
 
    # return json.loads('{"stock_name":"Alphabet Inc. (GOOGL)","action":"Hold","explanation":"Alphabet Inc. has shown strong revenue and net income growth year-over-year, indicating solid financial health. However, the market is currently volatile, and there are risks associated with potential regulatory challenges and competition. Holding the stock allows investors to benefit from its long-term growth while monitoring market conditions.","risk_level":"Moderate","analyst_consensus":"Buy","sector":"Technology","financial_metrics":{"price":168.95,"market_cap":2080000000000.0,"pe_ratio":22.41,"dividend_yield":0.47,"eps":7.21,"beta":1.05},"technical_indicators":{"moving_average_50_day":160.0,"moving_average_200_day":155.0,"relative_strength_index":65.0,"support_level":150.0,"resistance_level":175.0},"one_year_performance":30.5,"six_month_performance":15.2,"key_catalysts":["Continued growth in digital advertising revenue","Expansion into cloud services","Potential partnerships in AI and machine learning"],"potential_risks":["Regulatory scrutiny in various markets","Increased competition from other tech companies","Economic downturn affecting advertising budgets"]}')

    print("in stock_analysis...............")
    to_search = f" Latest news on : {stock_name} stock prices, date {getdate('stock').split(',')[0]} ," 
    webresult = search_web(to_search)
    prompt = """ You are an avdvance stock analyzer, you have provided with stock {stock_name} ,"""\
        f"""and the Latest news on this are: ```{webresult} ```,
        you need to perform thorough investigation and analyize the latest news step by step , 
        and provide a long term invesetment advice to invest or not,
        carefully give the suggestion of investment, it can cause financial damage .
    """
    prompt += prompt + ", please provide the following body after analysis: {format_instructions}, here is the result: "
    parser = PydanticOutputParser(pydantic_object=Stock)
    llm = get_llm(CONSTANT.model_name)
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm, max_retries=5)
    
    prompt = PromptTemplate(
        template=prompt,
        input_variables=["stock_name"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    completion_chain = prompt | llm
    main_chain = RunnableParallel(
        completion=completion_chain, 
        prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(x['completion'].content, x['prompt_value']))
    
    os.makedirs("Stock", exist_ok=True)
    x = main_chain.invoke({'stock_name': stock_name})
    output =  json.loads(x.json())
    output['Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    filepath = "Stock/output.json"
    try:
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    if not isinstance(existing_data, list):
        existing_data = [existing_data]
    
    existing_data.append(output)
    with open(filepath, 'w') as f:
        json.dump(existing_data, f, indent=4)

    return output

import requests
def extract_document(
    file_path: str, 
    api_url: str = "xxxxxxxxxxxxxxxxx",
    lang: str = "en"
    ):
    with open(file_path, 'rb') as file:
        files = {'files': file}
        payload = {
            'model_name': CONSTANT.model_name,
            'ocr_method': CONSTANT.ocr_method,
            'lang': lang
        }
        
        response = requests.post(api_url, files=files, data=payload)
        response.raise_for_status()
        
        return response.json()

def read_files_from_folder(folder_path: str):
    """
    Read file paths from specified folder
    
    Args:
        folder_path (str): Path to folder containing files
    
    Returns:
        List[str]: List of full file paths
    """
    return [
        os.path.join(folder_path, file) 
        for file in os.listdir(folder_path) 
        if os.path.isfile(os.path.join(folder_path, file))
    ]

@tool
def ko_generation():
    '''
        This tools is use to create Knowledge article uploaded documents
    '''
    print("using ko_generateion tool........")
    if not CONSTANT.DOC_SEARCH:
        return "Document access is not given by user, cannot generate the Knowledge object."
    folder_path = f"uploads/{CONSTANT.session_id}"
    if not os.path.exists(folder_path):
        return "Files are missing while bot tries to access..."
    
    file_paths = read_files_from_folder(folder_path)
    return [extract_document(file_path) if not str(file_path).endswith(('.png','.jpeg','.jpg')) else extract_from_image(file_path) 
            for file_path in file_paths]


from PIL import Image
import pytesseract
import configparser
config = configparser.ConfigParser()
config.read('config.properties')
pytesseract_path = r"xxxxxxxxxxxxxxxxxxxxxx"
pytesseract.pytesseract.tesseract_cmd = pytesseract_path
from langchain_openai import AzureChatOpenAI
api_key = config['azure']['api_key']
gpt_endpoint = config['azure']['endpoint']
gpt_deployment = config['azure']['deployment']
gpt_version = config['azure']['version']

from langchain_core.pydantic_v1 import BaseModel, Field

class KO(BaseModel):
    Short_description: str = Field(..., description='A brief and short description of user query')
    Long_description: str = Field(..., description='A detailed and good enhanced description of user query')
    Symptoms: str = Field(..., description='Symptoms refer to the observable signs, behaviors, or effects that indicate the presence of a problem/issue')
    Causes: str = Field(..., description='Causes refer to the underlying reasons or factors that lead to the occurrence of a problem or issue')
    Resolution_note: str = Field(..., description='step by step detailed Enhanced kowledge article which covers all the scenerio')
    Relevancy:str = Field(..., description='Relevancy score of resolution_note with user query ,must be within [0-100] ')
    
    def get_Ko(self, queryid):
        return {
            'ticketid': queryid,
            'short_description': self.Short_description.replace('\n', '<br>'),
            'long_description': self.Long_description.replace('\n', '<br>'),
            'symptoms': self.Symptoms.replace('\n', '<br>'),
            'causes': self.Causes.replace('\n', '<br>'),
            'resolution_note': self.Resolution_note.replace('\n', '<br>'),
            'Relevancy':self.Relevancy
        }
import json

def extract_from_image(img_path):
    
    image = Image.open(img_path)
    # Perform OCR on the image
    text = pytesseract.image_to_string(image)
    llm = AzureChatOpenAI(
                temperature=0.3,
                azure_endpoint=gpt_endpoint,
                api_key=api_key,
                deployment_name=gpt_deployment,
                api_version=gpt_version
            ).with_structured_output(KO)
    template1 = f"""you are an IT assistant for Knowledge article generation,you Must give following field for user query; 
                        short description ,
                        detailed long description,
                        causes(underlying main reasons or factors that lead to the occurrence of a problem),
                        Resolution_note(step by step detailed Enhanced kowledge article which covers all the scenerio),
                        symptoms(observable signs indicate the presence of a problem/issue), 
                        Relevancy: A relevancy score of resolution_note with user query ,must be within [0-100],
                        you must provide only one object of 'KO' class where each field is provided,"""
    template2 = template1 + f"""user query: {text}"""
    
    output = {
                "KO_Articles":[
                    {
                        "filename":str(img_path).split('\\')[-1],
                        "KO_Article":json.loads(llm.invoke(template2).json())
                    }
                ]
            }
    
    return json.loads(json.dumps(output))


# =================================================================
# from langgraph.graph import MessagesState
# class State(MessagesState):
#     summary: str
#     session_id: str
#     search_method: str
#     web_search: str
#     use_docx:str
#     use_database:str

# def db_state(user_query:State):
# =================================================================

import sqlalchemy
def getengine():
    """
    Placeholder function to test database connection
    Replace with actual implementation based on your database type
    """
    if CONSTANT.db_type == "PostgreSQL":
        # PostgreSQL connection string with schema
        connection_string = f"postgresql://{CONSTANT.user}:{CONSTANT.password}@{CONSTANT.host}:{CONSTANT.port}/{CONSTANT.database}?options=-csearch_path={CONSTANT.schema}"
    elif CONSTANT.db_type == "MySQL":
        # MySQL connection string with schema (set the database as the schema)
        connection_string = f"mysql+pymysql://{CONSTANT.user}:{CONSTANT.password}@{CONSTANT.host}:{CONSTANT.port}/{CONSTANT.database}.{CONSTANT.schema}"
    # Add more database type mappings as needed
    
    try:
        engine = sqlalchemy.create_engine(connection_string)
        with engine.connect() as connection:
            print("Database Connection Successful!")
            return engine
    except Exception as e:
        print(f"Connection Error: {e}")

# @tool
def db_query(user_query: str):
    '''
    This tool is used to query a connected database and retrieve or manipulate data based on the provided query.
    
    Parameters:
        - user_query (str): A valid SQL query string.
    
    '''
    print("using database tool.................")
    print(user_query)
    if "" in [CONSTANT.database,CONSTANT.user,CONSTANT.password,CONSTANT.host,CONSTANT.port,CONSTANT.schema] or None in [CONSTANT.database,CONSTANT.user,CONSTANT.password,CONSTANT.host,CONSTANT.port,CONSTANT.schema]:
        print("Db credentails are missing")
        return "some of the credentials are missing, ask user to provide the credentials"
    print("going for get engine.............")
    try:
        engine = getengine()
        print("going fetch result............")
        with engine.connect() as connection:
            user_query =sqlalchemy.text(user_query)
            result = connection.execute(user_query)
            print(result)
            return pd.DataFrame(result)
    except Exception as e:
        print(e)
        print("error.............",e)
        return f"there was an error :{e}, try to use get_db_structure tool to know about the structure of database,then query databse"

# @tool
def get_db_structure():
    """
    Retrieves the structure of the entire database, including tables and columns.
    
    Returns:
    - a dataframe having  details structure including tables, columns, primary key and foreign key.
    """
    print("calling get_db_stucutre tool..........")
    try:
        engine = getengine()
        with engine.connect() as connection:
            user_query =sqlalchemy.text("""
                                WITH ColumnInfo AS (
                                SELECT 
                                    c.table_schema,
                                    c.table_name,
                                    STRING_AGG(c.column_name, ', ' ORDER BY c.ordinal_position) AS all_columns,
                                    STRING_AGG(
                                        DISTINCT CASE 
                                            WHEN tc.constraint_type = 'PRIMARY KEY' THEN c.column_name 
                                        END, 
                                        ', '
                                    ) AS primary_keys,
                                    STRING_AGG(
                                        DISTINCT CONCAT(
                                            kcu.column_name, 
                                            '(', 
                                            ccu.table_name, 
                                            ')'
                                        ),
                                        ', '
                                    ) AS foreign_keys
                                FROM 
                                    information_schema.columns c
                                LEFT JOIN 
                                    information_schema.table_constraints tc 
                                    ON c.table_schema = tc.table_schema 
                                    AND c.table_name = tc.table_name 
                                    AND tc.constraint_type = 'PRIMARY KEY'
                                LEFT JOIN 
                                    information_schema.key_column_usage kcu 
                                    ON kcu.table_schema = tc.table_schema 
                                    AND kcu.table_name = tc.table_name 
                                    AND kcu.constraint_name = tc.constraint_name
                                LEFT JOIN 
                                    information_schema.constraint_column_usage ccu 
                                    ON kcu.constraint_name = ccu.constraint_name
                                WHERE 
                                    c.table_schema NOT IN ('pg_catalog', 'information_schema')
                                GROUP BY 
                                    c.table_schema, c.table_name
                            )
                            SELECT 
                                CONCAT(
                                    table_name, 
                                    '(', all_columns, ')', 
                                    ', Primary Key(', COALESCE(primary_keys, ''), ')', 
                                    ', Foreign Key(', COALESCE(foreign_keys, ''), ')'
                                ) AS table_structure
                            FROM 
                                ColumnInfo;""")
            result = connection.execute(user_query)

            return pd.DataFrame(result)
        
    except Exception as e:
        print(f"Error retrieving database structure: {e}")
        return e

