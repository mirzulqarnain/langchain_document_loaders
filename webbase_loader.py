from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import SimpleJsonOutputParser,StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "Answer the Following Questions  \n {question} from the following text - \n {text} ",
    input_variables=["question",'text']
)
parser = StrOutputParser()
url = 'https://www.flipkart.com/6bo/b5g/~cs-bso4fpdrdh/pr?sid=6bo%2Cb5g&collection-tab-name=ASUS+Vivobook+15&sort=price_asc&BU=CoreElectronics'
loader = WebBaseLoader(url)
docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({"question":'what is the product we are talking about?',"text":docs[0].page_content}))
