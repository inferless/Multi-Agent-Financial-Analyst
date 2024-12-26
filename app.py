from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent, FunctionCallingAgentWorker
from llama_index.llms.huggingface import HuggingFaceLLM
from tools import yf_fundamental_analysis
from utils import flatten_dict, create_stock_analysis_prompt
from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer


class InferlessPythonModel:
    def initialize(self):
        yf_fundamental_analysis_tool = FunctionTool.from_defaults(fn=yf_fundamental_analysis)
        tools = [yf_fundamental_analysis_tool]+YahooFinanceToolSpec().to_tool_list()
        llm = HuggingFaceLLM(
                                model_name="unsloth/Meta-Llama-3.1-8B-Instruct", 
                                tokenizer_name="unsloth/Meta-Llama-3.1-8B-Instruct", 
                                device_map="auto", 
                                generate_kwargs={"temperature": 0.001, "do_sample":True}, 
                                max_new_tokens=5000
                            )
        self.finance_agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

    def infer(self, inputs):
        prompt = create_stock_analysis_prompt("AAPL", "Give me a complete Analysis of Apple stock")
        response = self.finance_agent.query(prompt)

        return {'generated_summary': response.response}

    def finalize(self):
        self.llm = None