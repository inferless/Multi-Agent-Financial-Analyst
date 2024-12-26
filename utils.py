def flatten_dict(mapping: dict, prefix="")-> str:
    '''
    Flattens dictionary to string
    Example: 
    Input -> {'name': 'Apple', 'Address': 'CA, US', 'closePrice': 229.9}
    Output ->
    name - Apple
    Address - CA, US
    closePrice - 229.9
    '''
    total_txt = ""
    for key, val in mapping.items():
        if type(val) in [str, float, int]:
            total_txt += f"{prefix}{key} - {val}\n"
        elif isinstance(val, dict):
            total_txt += f"{key}\n"
            total_txt += flatten_dict(val, prefix="\t")
        elif isinstance(val, list):
            if isinstance(val[0], dict):
                total_txt += f"{key}\n"
                for v in val:
                    total_txt += flatten_dict(v, prefix="\t")
            else:
                total_txt += f"{key} - {val}\n"
    return total_txt

# prompt = "Give me a complete Analysis of Apple stock"
# agent_prompt = f"""You are a stock analysis expert and you need to response to you user query in a format.
#                    Conduct an in-depth analysis of the financial performance and market position of "stock_symbol",
#                    Conduct a comprehensive fundamental analysis of "stock_symbol", focusing on financial statements, valuation metrics, and key value drivers to assess intrinsic value.,
#                    Provide comprehensive stock analyses and strategic investment recommendations to impress a high-profile client.,
#                    Here is you a query from an user: {prompt}
# """
# response = finance_agent.query(agent_prompt)

# print(str(response))



def create_stock_analysis_prompt(stock_symbol, user_query):
    return f"""You are a stock analysis expert. Analyze {stock_symbol} and respond in this exact format:

                SUMMARY
                [Provide 2-3 sentence overview of current stock position]

                FUNDAMENTAL ANALYSIS
                Financial Health:
                - Revenue: [Latest revenue figures and growth]
                - Profit Margins: [Current margins and trends]
                - Debt Ratio: [Current ratio and assessment]
                - Cash Flow: [Operating cash flow status]

                Valuation Metrics:
                - P/E Ratio: [Current P/E and industry comparison]
                - Market Cap: [Current value]
                - Book Value: [Per share]
                - PEG Ratio: [Current value and interpretation]

                TECHNICAL ANALYSIS
                - Current Price: [Latest price]
                - 52-Week Range: [High/Low]
                - Moving Averages: [50-day and 200-day]
                - Trading Volume: [Average daily volume]

                RISK ASSESSMENT
                [List 3 key risks, one sentence each]

                RECOMMENDATION
                Rating: [Buy/Sell/Hold]
                Target Price: [12-month target]
                Confidence Level: [High/Medium/Low]
                Rationale: [2-3 sentences explaining the recommendation]

                USER QUERY: {user_query}
            """