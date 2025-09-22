# langchain_pipeline.py
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI   # pip install langchain-openai

from schema import IRSchema
import json


# 1) output schema definition
output_parser = PydanticOutputParser(pydantic_object=IRSchema)

SYSTEM_PROMPT = """
You are a schema-aware query planner.
Convert the following NL query into a JSON IR that matches the given KeyframeQL schema:
- Use only allowed ops and argument types
- Respond ONLY with valid JSON
- Do NOT add code fences (```), language tags, or commentary
"""

IR_SCHEMA_EXAMPLE = {
      "select": "event",
      "dataset": "TrafficVideo",
      "objects": [
        {"name": "car1", "cls": "car", "idx": 0},
        {"name": "car2", "cls": "car", "idx": 1}
      ],
      "keyframes": [
        {
          "name": "k1",
          "predicates": [
            {"op":"heading_diff_to", "args":["car1","car2",90.0], "kind":"single"},
            {"op":"velocity_above", "args":["car1",2.0], "kind":"single"},
            {"op":"velocity_above", "args":["car2",2.0], "kind":"single"}
          ]
        }
      ],
      "temporal_always": [
        {"target":"k1","dur_sec":3.0,"tol":0.01,"anchor":None}
      ],
      "temporal_interframe": [
        {"anchor":"k1","target":"k2","time_shift":3.0,
         "comparators":[
           {"op":"heading_diff","args":["car1",5.0],"kind":"inter"},
           {"op":"heading_diff","args":["car2",5.0],"kind":"inter"}
         ]}
      ],
      "returns":["clip","timestamps"],
      "query_name":"GENERATED_QUERY"
    }


# 2) Prompt template
prompt = PromptTemplate(
    template=SYSTEM_PROMPT + """

Schema IR Example:
{example}

NL query: {nl_query}

{format_instructions}
""",
    input_variables=["nl_query"],
    partial_variables={
        "format_instructions": output_parser.get_format_instructions(),
        "example": json.dumps(IR_SCHEMA_EXAMPLE, indent=2) 
    },
)

# 3) LangChain LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# 0) chain configuration
ir_chain = prompt | llm | output_parser
